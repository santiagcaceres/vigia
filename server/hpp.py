"""Hik-Partner Pro (HPP) OpenAPI: alarmas de los DVR de los clientes.

Según la guía oficial "Hik-Partner Pro OpenAPI Developer Guide V2.15.500":

- La API key (AppKey/SecretKey) se saca en el portal de HPP: Site & Device → My Service →
  API Integration. Token: POST https://api.hik-partner.com/api/hpcgw/v1/token/get, dura 7 días
  y devuelve el `areaDomain` que se usa para todo lo demás.
- Las alarmas (VMD = movimiento, fielddetection = intrusión, linedetection = cruce de línea...)
  se reciben por **webhook** (HPP hace POST a nuestra URL HTTPS) después de suscribirse a los
  equipos con mq/subscribe. Cada envío viene firmado con HMAC-SHA256.
- **El video no viene por esta API**: ver, reproducir o descargar requiere HPNetSDK (clave
  VSaaS que se pide por mail a Hikvision). Por eso la alarma solo dice "qué DVR, qué canal,
  a qué hora" y la PC central baja ese tramo de video por otro camino (hoy: VPN al DVR).

El formato exacto del cuerpo del webhook no está documentado en detalle: se interpreta de forma
tolerante (lista de mensajes con deviceSerial y alarmData en XML o JSON) y se guarda el original.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import httpx

log = logging.getLogger("vigia.hpp")

TOKEN_URL = "https://api.hik-partner.com/api/hpcgw/v1/token/get"
# Tipos de alarma de video que tienen sentido para buscar hurtos (los demás se ignoran).
VIDEO_EVENT_TYPES = ("VMD", "fielddetection", "linedetection", "regionEntrance", "regionExiting")
DEFAULT_EVENT_TYPES = ("VMD", "fielddetection", "linedetection")
MAX_SKEW_SECONDS = 300


class HppError(RuntimeError):
    pass


# ---------------------------------------------------------------- firma del webhook
def sign(secret: str, timestamp: str, batch_id: str) -> str:
    mac = hmac.new(secret.encode(), f"{timestamp}.{batch_id}".encode(), hashlib.sha256).hexdigest()
    return f"sha256={mac}"


def verify(secret: str, timestamp: str, batch_id: str, signature: str, now_ms: int | None = None) -> bool:
    if not (secret and timestamp and batch_id and signature):
        return False
    try:
        ts = int(timestamp)
    except ValueError:
        return False
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    if ts < 10**11:  # vino en segundos
        ts *= 1000
    if abs(now_ms - ts) > MAX_SKEW_SECONDS * 1000:
        return False
    return hmac.compare_digest(sign(secret, timestamp, batch_id), signature.strip())


# ---------------------------------------------------------------- interpretación de alarmas
@dataclass
class HppAlarm:
    device_serial: str
    channel: int
    event_type: str
    trigger_time: str  # tal como vino (hora del equipo)
    target: str | None
    raw: dict

    def occurred_utc(self, tz: str) -> datetime:
        """La hora del DVR viene sin zona (hora local del equipo) o con desfase ISO."""
        return parse_device_time(self.trigger_time, tz)


def parse_device_time(value: str, tz: str) -> datetime:
    value = (value or "").strip()
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(timezone.utc).replace(tzinfo=None)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(tz))
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def _strip_ns(root: ET.Element) -> ET.Element:
    for el in root.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return root


def _find_ci(d, *names: str):
    """Busca la primera clave (sin importar mayúsculas) en un dict anidado."""
    wanted = {n.lower() for n in names}
    stack = [d]
    while stack:
        cur = stack.pop(0)
        if isinstance(cur, dict):
            for k, v in cur.items():
                if k.lower() in wanted and v not in (None, "", [], {}):
                    return v
            stack.extend(v for v in cur.values() if isinstance(v, (dict, list)))
        elif isinstance(cur, list):
            stack.extend(cur)
    return None


def _xml_to_dict(root: ET.Element) -> dict:
    out: dict = {}
    for child in root:
        val = _xml_to_dict(child) if len(child) else (child.text or "").strip()
        if child.tag in out:
            if not isinstance(out[child.tag], list):
                out[child.tag] = [out[child.tag]]
            out[child.tag].append(val)
        else:
            out[child.tag] = val
    return out


def _alarm_data(value) -> dict:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    text = value.strip()
    if text.startswith("<"):
        try:
            return _xml_to_dict(_strip_ns(ET.fromstring(text)))
        except ET.ParseError:
            return {}
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def extract_messages(body) -> tuple[str | None, list[dict]]:
    """Devuelve (batchId, mensajes) de un cuerpo de webhook o de mq/messages."""
    if not isinstance(body, dict):
        return None, []
    data = body.get("data") if isinstance(body.get("data"), (dict, list)) else body
    batch = (data.get("batchId") if isinstance(data, dict) else None) or body.get("batchId")
    if isinstance(data, list):
        items = data
    else:
        items = data.get("list") or data.get("messages") or data.get("msgs")
        if items is None and ("deviceSerial" in data or "alarmData" in data):
            items = [data]
    return batch, [m for m in (items or []) if isinstance(m, dict)]


def parse_message(msg: dict) -> HppAlarm | None:
    data = _alarm_data(msg.get("alarmData"))
    serial = str(_find_ci(data, "deviceSerial") or msg.get("deviceSerial") or "").strip()
    etype = str(_find_ci(data, "eventType") or msg.get("eventType") or "").strip()
    if not serial or not etype:
        return None
    channel = _find_ci(data, "channelID", "channelNo", "dynChannelID", "channel") or msg.get("channelNo") or 1
    try:
        channel = int(str(channel).strip())
    except ValueError:
        channel = 1
    when = str(_find_ci(data, "triggerTime", "dateTime", "alarmTime") or "")
    target = _find_ci(data, "detectionTarget", "targetType")
    raw = {"deviceSerial": msg.get("deviceSerial"), "formatType": msg.get("formatType"), "alarmData": data}
    return HppAlarm(serial, channel, etype, when, str(target) if isinstance(target, str) else None, raw)


# ---------------------------------------------------------------- cliente de la API
class HppClient:
    def __init__(self, app_key: str, secret_key: str, http: httpx.Client | None = None):
        self.app_key, self.secret_key = app_key, secret_key
        self.http = http or httpx.Client(timeout=20.0)
        self._token: str | None = None
        self._expires = 0.0
        self.area_domain: str | None = None

    def _check(self, r: httpx.Response) -> dict:
        try:
            body = r.json()
        except ValueError as exc:
            raise HppError(f"Respuesta inválida de Hik-Partner Pro (HTTP {r.status_code})") from exc
        if str(body.get("errorCode", "0")) != "0":
            raise HppError(f"Hik-Partner Pro respondió {body.get('errorCode')}: {body.get('message', '')}")
        return body

    def token(self) -> str:
        if self._token and time.time() < self._expires - 3600:
            return self._token
        r = self.http.post(TOKEN_URL, json={"appKey": self.app_key, "secretKey": self.secret_key})
        body = self._check(r)
        data = body.get("data") or body
        # la guía muestra claves con espacio al final ("accessToken "): se normalizan
        data = {str(k).strip(): v for k, v in data.items()}
        self._token = data.get("accessToken")
        if not self._token:
            raise HppError("Hik-Partner Pro no devolvió token: revisá AppKey y SecretKey")
        self._expires = float(data.get("expireTime") or 0) / 1000 or time.time() + 6 * 86400
        self.area_domain = (data.get("areaDomain") or "https://isaapi.hik-partner.com").rstrip("/")
        return self._token

    def call(self, path: str, payload: dict | None = None) -> dict:
        tok = self.token()
        r = self.http.post(f"{self.area_domain}/api/hpcgw/{path.lstrip('/')}", json=payload or {},
                           headers={"Authorization": f"Bearer {tok}", "Token": tok})
        return self._check(r)

    def devices(self) -> list[dict]:
        out, page = [], 1
        while page <= 50:
            body = self.call("v1/device/list", {"page": page, "pageSize": 50})
            data = body.get("data")
            rows = data if isinstance(data, list) else (data or {}).get("rows") or (data or {}).get("list") or []
            out += rows
            if len(rows) < 50:
                break
            page += 1
        return out

    def channels(self, serial: str) -> list[dict]:
        data = self.call("v1/device/camera/list", {"deviceSerial": serial}).get("data")
        return data if isinstance(data, list) else []

    def setup_webhook(self, callback_url: str, sign_secret: str | None, serials: list[str] | None) -> None:
        cfg = {"callbackUrl": callback_url, "retryTimes": 3, "retryDelay": 2000}
        if sign_secret:
            cfg["signSecret"] = sign_secret
        self.call("webhook/v1/config/save", cfg)
        sub = {"subType": 1, "subMode": "list", "deviceSerialList": serials} if serials else {"subType": 1, "subMode": "all"}
        self.call("v1/mq/subscribe", sub)

    def webhook_config(self) -> dict:
        return self.call("webhook/v1/config/query").get("data") or {}
