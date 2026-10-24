"""Alarmas que avisa el propio grabador (sin nube, sin API key).

Los dos fabricantes ofrecen una conexión HTTP larga por la que el equipo va mandando sus
eventos (movimiento, intrusión, cruce de línea) a medida que ocurren:

- Dahua:     GET /cgi-bin/eventManager.cgi?action=attach&codes=[VideoMotion,CrossLineDetection,...]
             Responde líneas de texto `Code=VideoMotion;action=Start;index=0;data={...}`.
- Hikvision: GET /ISAPI/Event/notification/alertStream
             Responde bloques XML <EventNotificationAlert> (uno por evento, más latidos
             "videoloss inactive" cada segundo).

Ambos con autenticación Digest, igual que el resto de la API. Acá se interpretan esos
formatos y se normalizan al mismo nombre de tipo que usa Hik-Partner Pro (VMD,
fielddetection, linedetection, regionEntrance, regionExiting), así una alarma vale lo mismo
venga de donde venga.

Verificado contra la documentación de ambas marcas; los detalles (índice de canal en Dahua,
path exacto en Hikvision) conviene confirmarlos con un equipo real la primera vez.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterator

import httpx

from .cameras import BaseCamera

log = logging.getLogger("vigia.eventos")

# Tipos de alarma de video que sirven para buscar hurtos (mismos nombres que server/hpp.py).
VIDEO_EVENT_TYPES = ("VMD", "fielddetection", "linedetection", "regionEntrance", "regionExiting")

DAHUA_CODES = {
    "VideoMotion": "VMD", "SmartMotionHuman": "VMD", "SmartMotionVehicle": "VMD",
    "CrossLineDetection": "linedetection", "CrossRegionDetection": "fielddetection",
    "LeftDetection": "fielddetection", "TakenAwayDetection": "fielddetection",
}
HIK_TYPES = {
    "VMD": "VMD", "fielddetection": "fielddetection", "linedetection": "linedetection",
    "regionEntrance": "regionEntrance", "regionExiting": "regionExiting",
    "intrusion": "fielddetection", "attendedBaggage": "fielddetection", "unattendedBaggage": "fielddetection",
}
# Tiempo máximo sin datos antes de reconectar. Hikvision manda latidos cada ~1 s; Dahua
# manda un latido (`Heartbeat`) si se pide con heartbeat=N en la URL.
READ_TIMEOUT = 90.0


@dataclass
class DeviceEvent:
    event_type: str  # normalizado (VMD, fielddetection, ...)
    channel: int | None  # None si el equipo no lo dice (cámara suelta)
    target: str | None = None  # human / vehicle si el equipo lo distingue
    raw: dict = field(default_factory=dict)


# ---------------------------------------------------------------- Dahua
def dahua_attach_url(cam: BaseCamera, codes: list[str] | None = None, heartbeat: int = 30) -> str:
    codes = codes or sorted(DAHUA_CODES)
    return (f"http://{cam.host}:{cam.http_port}/cgi-bin/eventManager.cgi?action=attach"
            f"&codes=[{','.join(codes)}]&heartbeat={heartbeat}")


def parse_dahua_line(line: str) -> DeviceEvent | None:
    """`Code=VideoMotion;action=Start;index=0;data={...}` -> evento, o None si no interesa."""
    line = line.strip()
    if not line.startswith("Code="):
        return None
    fields: dict[str, str] = {}
    for part in line.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            fields[k.strip()] = v.strip()
            if k.strip() == "data":  # el JSON puede traer ';' adentro: es lo último de la línea
                fields["data"] = line.split("data=", 1)[1]
                break
    code = fields.get("Code", "")
    if fields.get("action", "Start") != "Start":  # Stop / Pulse: no es un disparo nuevo
        return None
    etype = DAHUA_CODES.get(code)
    if not etype:
        return None
    channel = None
    try:
        channel = int(fields["index"]) + 1  # Dahua numera los canales desde 0 en los eventos
    except (KeyError, ValueError):
        pass
    target = None
    if code == "SmartMotionHuman":
        target = "human"
    elif code == "SmartMotionVehicle":
        target = "vehicle"
    else:
        m = re.search(r'"ObjectType"\s*:\s*"(\w+)"', fields.get("data", ""))
        if m:
            target = m.group(1).lower()
    return DeviceEvent(etype, channel, target, {"brand": "dahua", "code": code, "line": line[:500]})


def dahua_events(cam: BaseCamera, http: httpx.Client | None = None) -> Iterator[DeviceEvent]:
    """Se queda conectado al grabador y va entregando los eventos. Corta al perder la conexión."""
    client = http or httpx.Client(auth=httpx.DigestAuth(cam.username, cam.password),
                                  timeout=httpx.Timeout(10.0, read=READ_TIMEOUT))
    with client.stream("GET", dahua_attach_url(cam)) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            ev = parse_dahua_line(line)
            if ev is not None:
                yield ev


# ---------------------------------------------------------------- Hikvision
def hik_alert_url(cam: BaseCamera) -> str:
    return f"http://{cam.host}:{cam.http_port}/ISAPI/Event/notification/alertStream"


_ALERT_RE = re.compile(rb"<EventNotificationAlert.*?</EventNotificationAlert>", re.S)


def parse_hik_alert(xml: bytes | str) -> DeviceEvent | None:
    """Un bloque <EventNotificationAlert> -> evento, o None (latido, inactivo, tipo sin interés)."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return None
    for el in root.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    etype = HIK_TYPES.get((root.findtext("eventType") or "").strip())
    if not etype or (root.findtext("eventState") or "active").strip() != "active":
        return None
    channel = None
    for tag in ("channelID", "dynChannelID"):
        try:
            channel = int((root.findtext(tag) or "").strip())
            break
        except ValueError:
            continue
    target = (root.findtext(".//detectionTarget") or "").strip().lower() or None
    return DeviceEvent(etype, channel, target, {"brand": "hikvision", "eventType": root.findtext("eventType"),
                                                "dateTime": root.findtext("dateTime")})


def split_hik_stream(chunks: Iterator[bytes]) -> Iterator[bytes]:
    """Junta los pedazos del multipart y entrega cada bloque XML completo."""
    buf = b""
    for chunk in chunks:
        buf += chunk
        while True:
            m = _ALERT_RE.search(buf)
            if not m:
                break
            yield m.group(0)
            buf = buf[m.end():]
        if len(buf) > 1_000_000:  # algo raro: no acumular sin límite
            buf = buf[-100_000:]


def hik_events(cam: BaseCamera, http: httpx.Client | None = None) -> Iterator[DeviceEvent]:
    client = http or httpx.Client(auth=httpx.DigestAuth(cam.username, cam.password),
                                  timeout=httpx.Timeout(10.0, read=READ_TIMEOUT))
    with client.stream("GET", hik_alert_url(cam)) as r:
        r.raise_for_status()
        for block in split_hik_stream(r.iter_bytes()):
            ev = parse_hik_alert(block)
            if ev is not None:
                yield ev


# ---------------------------------------------------------------- común
def device_events(cam: BaseCamera, http: httpx.Client | None = None) -> Iterator[DeviceEvent]:
    if hasattr(cam, "resolve"):  # marca detectada sola (conexión por IP y puertos)
        cam = cam.resolve()
    if cam.brand == "dahua":
        return dahua_events(cam, http)
    return hik_events(cam, http)


def matches_channel(ev: DeviceEvent, channel: int) -> bool:
    """Un grabador manda los eventos de todos sus canales: nos quedamos con el nuestro.
    Si el equipo no dice canal (cámara suelta), se acepta."""
    return ev.channel is None or ev.channel == channel


def now_iso(tz) -> str:
    return datetime.now(tz).replace(microsecond=0).isoformat()
