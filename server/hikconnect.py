"""Hik-Connect for Teams (ex HikCentral Connect) OpenAPI: video de grabadores Hikvision SIN abrir puertos.

El grabador ya está conectado a la nube de Hik-Connect (como cuando el dueño lo mira en el celular).
Con la API del modo equipo se pide un link de video temporal (HLS) que la PC central lee como
cualquier otro stream. **Hallado el 23 oct 2026 y todavía sin probar con un equipo real.**

- Credenciales: en el portal de Hik-Connect, Team Management → Team Configuration → API Integration
  (AppKey y SecretKey). Se guardan por cliente; el SecretKey cifrado. Nunca salen del servidor.
- Token: POST https://<región>.hikcentralconnect.com/api/hccgw/platform/v1/token/get
  {appKey, secretKey} → {accessToken, expireTime, areaDomain}. Las demás llamadas van al areaDomain
  con el encabezado `Token`.
- Cámaras: POST /api/hccgw/resource/v1/areas/cameras/get (id = resourceId, device.devInfo.serialNo).
- Video: POST /api/hccgw/video/v1/live/address/get {resourceId, deviceSerial, type, protocol,
  quality, expireTime[, startTime, stopTime]} → {url}. type 1 = vivo, 2 = nube, 3 = grabación del
  disco. protocol 2 = HLS (solo H.264 y con el cifrado de video de Hik-Connect apagado).

Fuentes: integración de Home Assistant "hikconnect-nvr" (el vivo por HLS anda así) y una referencia
de terceros de la API V2.11.800 (los tipos de grabación). Si algo no coincide, los errores se
registran completos en el log del servidor (Vercel) para corregir.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from datetime import datetime

import httpx

log = logging.getLogger("vigia.hikconnect")

# Servidores por región. Se prueban en orden hasta que uno acepta las credenciales (o se usa el
# que se cargó en el cliente, ej. el dominio que muestra el navegador al entrar al portal).
# `protocol` de /video/v1/live/address/get (los de la plataforma de EZVIZ que Hik-Connect envuelve)
PROTOCOLS = {"hls": 2, "rtmp": 3, "flv": 4}
REGION_URLS = ("https://isa.hikcentralconnect.com", "https://ius.hikcentralconnect.com",
               "https://ieu.hikcentralconnect.com", "https://isgp.hikcentralconnect.com")
# `type` de /video/v1/live/address/get: 1 vivo, 2 grabación del disco del grabador, 3 grabación en la nube (guía
# V2.15.0 de la API y la plataforma de EZVIZ que envuelve). Antes estaban 2 y 3 al revés: pidiendo «3» Hik-Connect
# contestaba «Record setting does not exist{OPEN000502}» (no hay plan de nube), 28 oct, Minas.
TYPES = {"live": "1", "playback": "2", "cloud": "3"}


class HikConnectError(RuntimeError):
    pass


# (appKey + huella del SecretKey) -> (token, areaDomain, vence, servidor). Con la huella: un cliente cargado con el AppKey
# de otro (que no es secreto) y una clave cualquiera no usa el token del otro
_tokens: dict[str, tuple[str, str, float, str]] = {}
_lock = threading.Lock()


def _ok(r: httpx.Response, what: str) -> dict:
    try:
        data = r.json()
    except ValueError as exc:
        raise HikConnectError(f"{what}: respuesta no válida (HTTP {r.status_code})") from exc
    if str(data.get("errorCode")) != "0":
        log.warning("Hik-Connect %s falló: %s", what, str(data)[:500])
        msg, code = str(data.get("message") or ""), str(data.get("errorCode") or "")
        raise HikConnectError(f"{what}: {msg}" + (f" {{{code}}}" if code and code not in msg else ""))
    return data.get("data") or {}


def short_serial(serial: str | None) -> str:
    """Serie corta del equipo (la que usa la API, ej. «E02380030»). Hikvision también muestra la larga:
    modelo + fecha + «CCWR» + serie + «WCVU» (ej. «DVR/DVS-34-A1620191217CCWRE02380030WCVU»)."""
    s = (serial or "").strip().upper()
    if "CCWR" in s:
        s = s.rsplit("CCWR", 1)[1]
        s = s[:-4] if len(s) > 4 and s[-4:].isalpha() else s
    elif len(s) > 16 and s[-4:].isalpha():
        s = s[-13:-4]
    return s


ALLOWED_DOMAIN = ".hikcentralconnect.com"  # a donde se manda el AppKey y el SecretKey del cliente: solo a Hik-Connect


def allowed_url(url: str | None) -> bool:
    """La dirección es de Hik-Connect for Teams (https y un nombre de *.hikcentralconnect.com, sin puerto ni usuario)."""
    try:
        u = httpx.URL(url or "")
    except Exception:  # noqa: BLE001 (una dirección mal escrita)
        return False
    host = (u.host or "").lower()
    return (u.scheme == "https" and host.endswith(ALLOWED_DOMAIN) and u.port in (None, 443) and not u.userinfo
            and host.replace(".", "").replace("-", "").isalnum())


def normalize_url(url: str | None) -> str | None:
    """La dirección de la API que se pegó (o None si se borró), con solo el dominio. Si no es de Hik-Connect, ValueError:
    el servidor le manda ahí el SecretKey del cliente, así que no puede ser cualquiera (se lo podrían llevar)."""
    url = (url or "").strip().rstrip("/")
    if not url:
        return None
    if not url.startswith("http"):
        url = "https://" + url
    # si pegaron la dirección del portal con una ruta, quedarse con el dominio
    parts = url.split("/")
    url = "/".join(parts[:3])
    if not allowed_url(url):
        raise ValueError("La dirección de la API tiene que ser de Hik-Connect (https://….hikcentralconnect.com)")
    return url


def channel_of(cam: dict) -> int | None:
    """El número de canal de una cámara de la lista. La API lo manda en `device.channelInfo.no` (visto el 6 nov con el
    grabador de Sanfer: {"device": {"devInfo": {...}, "channelInfo": {"id": "…", "no": "1"}}}); otras versiones, en
    `channelNo` y parecidos; si no, el nombre por defecto lo dice («camera 3@E02380030»). Sin esto, las cámaras con
    nombre propio («Camera 01») quedaban sin canal y la app las guardaba a todas como canal 1."""
    info = (cam.get("device") or {}).get("channelInfo") or cam.get("channelInfo") or {}
    info = info if isinstance(info, dict) else {}
    for v in (info.get("no"), info.get("channelNo"), cam.get("channelNo"), cam.get("channel"), cam.get("cameraNo"),
              cam.get("chanNo")):
        if isinstance(v, (int, str)) and not isinstance(v, bool) and str(v).strip().isdigit() and int(v) > 0:
            return int(v)
    m = re.match(r"\s*camera\s*0*(\d+)\s*@", str(cam.get("name") or ""), re.IGNORECASE)
    return int(m.group(1)) if m and int(m.group(1)) > 0 else None


class HikConnect:
    def __init__(self, app_key: str, secret: str, api_url: str | None = None, timeout: float = 20.0):
        if not app_key or not secret:
            raise HikConnectError("Falta la clave de la API de Hik-Connect (AppKey y SecretKey) en el cliente")
        self.app_key, self.secret = app_key.strip(), secret.strip()
        self._key = self.app_key + ":" + hashlib.sha256(self.secret.encode()).hexdigest()
        try:
            self.api_url = normalize_url(api_url)
        except ValueError:  # una dirección guardada antes de controlarla: se prueban las regiones de siempre
            log.warning("Hik-Connect: dirección de la API que no es de Hik-Connect, se ignora")
            self.api_url = None
        self.http = httpx.Client(timeout=timeout)
        self.server: str | None = None  # el que aceptó las credenciales

    def close(self) -> None:
        self.http.close()

    # ---------- token ----------
    def _token(self) -> tuple[str, str]:
        with _lock:
            t = _tokens.get(self._key)
            if t and t[2] > time.time() + 300:
                self.server = t[3]
                return t[0], t[1]
        errors = []
        for base in ([self.api_url] if self.api_url else list(REGION_URLS)):
            try:
                r = self.http.post(f"{base}/api/hccgw/platform/v1/token/get",
                                   json={"appKey": self.app_key, "secretKey": self.secret})
                data = _ok(r, f"token ({base})")
            except (httpx.HTTPError, HikConnectError) as exc:
                errors.append(f"{base.split('//')[1].split('.')[0]}: {exc}")
                continue
            token, area = data.get("accessToken"), (data.get("areaDomain") or base).rstrip("/")
            if not allowed_url(area):  # el área donde siguen los pedidos (con el token): también de Hik-Connect
                area = base
            if not token:
                errors.append(f"{base}: sin accessToken")
                continue
            exp = data.get("expireTime")
            exp = exp / 1000 if isinstance(exp, (int, float)) and exp > 10**11 else (exp or time.time() + 3600)
            with _lock:
                _tokens[self._key] = (token, area, float(exp), base)
            self.server = base
            log.info("Hik-Connect: token OK en %s (área %s)", base, area)
            return token, area
        raise HikConnectError("Hik-Connect no aceptó la clave de la API. " + " · ".join(errors))

    def _post(self, path: str, body: dict, what: str) -> dict:
        token, area = self._token()
        try:
            r = self.http.post(f"{area}{path}", json=body, headers={"Token": token})
        except httpx.HTTPError as exc:
            raise HikConnectError(f"{what}: sin conexión con Hik-Connect ({exc})") from exc
        if r.status_code == 401:
            with _lock:
                _tokens.pop(self._key, None)
        return _ok(r, what)

    # ---------- recursos ----------
    def cameras(self) -> list[dict]:
        """Cámaras visibles para la API: id (resourceId), nombre, serie del grabador y canal (si viene)."""
        out: list[dict] = []
        for page in range(1, 40):
            data = self._post("/api/hccgw/resource/v1/areas/cameras/get", {
                "pageIndex": str(page), "pageSize": 200,
                "filter": {"areaID": "-1", "includeSubArea": "1", "deviceID": "", "deviceSerialNo": ""}},
                "lista de cámaras")
            items = data.get("camera") or []
            if page == 1 and items:
                log.info("Hik-Connect: ejemplo de cámara: %s", str(items[0])[:800])
            for c in items:
                if not isinstance(c, dict) or not c.get("id"):
                    continue
                dev = (c.get("device") or {}).get("devInfo") or {}
                out.append({"id": str(c["id"]), "name": c.get("name") or str(c["id"]),
                            "serial": short_serial(dev.get("serialNo") or dev.get("deviceSerial")),
                            "device_name": dev.get("name") or "",
                            "channel": channel_of(c),
                            "online": c.get("onlineStatus", c.get("online"))})
            if len(items) < 200:
                break
        return out

    def stream_url(self, resource_id: str, serial: str, kind: str = "live", quality: str = "sub",
                   start: datetime | None = None, end: datetime | None = None, expire: int = 3600,
                   code: str | None = None, protocol: str = "hls") -> str:
        """Link de video temporal. `kind`: live (vivo), playback (disco del grabador) o cloud (nube).
        `start`/`end` en hora del grabador (local). `protocol`: hls (el de siempre), flv o rtmp: el 5 nov el HLS
        del servidor de video de São Paulo cortaba todas las conexiones y se prueban los otros formatos de la
        plataforma de EZVIZ (1 EZOPEN, 2 HLS, 3 RTMP, 4 FLV)."""
        body = {"resourceId": resource_id, "deviceSerial": serial, "type": TYPES[kind],
                "protocol": PROTOCOLS.get(protocol, 2),
                "quality": "2" if quality == "sub" else "1", "expireTime": int(expire)}
        if code:
            body["code"] = code  # código de verificación del grabador (si tiene el cifrado de video prendido)
        if kind != "live":
            if not (start and end):
                raise HikConnectError("Para la grabación hace falta inicio y fin")
            body["startTime"] = start.strftime("%Y-%m-%d %H:%M:%S")
            body["stopTime"] = end.strftime("%Y-%m-%d %H:%M:%S")
        try:
            data = self._post("/api/hccgw/video/v1/live/address/get", body, "link de video")
        except HikConnectError as exc:
            if "EVZ60019" in str(exc) or "ncryption" in str(exc):
                raise HikConnectError(
                    "El grabador tiene el cifrado de video prendido y así Hik-Connect no entrega el video. "
                    "Apagalo: app Hik-Connect → grabador → Configuración → Cifrado de video, o en el menú del "
                    "grabador: Red → Avanzado → Acceso a plataforma → Hik-Connect → cifrado. (" + str(exc) + ")") from exc
            raise
        url = data.get("url")
        if not url:
            log.warning("Hik-Connect: sin url en la respuesta: %s", str(data)[:500])
            raise HikConnectError("Hik-Connect no devolvió el link de video")
        return url
