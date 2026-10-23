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

import logging
import threading
import time
from datetime import datetime

import httpx

log = logging.getLogger("vigia.hikconnect")

# Servidores por región. Se prueban en orden hasta que uno acepta las credenciales (o se usa el
# que se cargó en el cliente, ej. el dominio que muestra el navegador al entrar al portal).
REGION_URLS = ("https://isa.hikcentralconnect.com", "https://ius.hikcentralconnect.com",
               "https://ieu.hikcentralconnect.com", "https://isgp.hikcentralconnect.com")
TYPES = {"live": "1", "cloud": "2", "playback": "3"}


class HikConnectError(RuntimeError):
    pass


_tokens: dict[str, tuple[str, str, float, str]] = {}  # appKey -> (token, areaDomain, vence, servidor)
_lock = threading.Lock()


def _ok(r: httpx.Response, what: str) -> dict:
    try:
        data = r.json()
    except ValueError as exc:
        raise HikConnectError(f"{what}: respuesta no válida (HTTP {r.status_code})") from exc
    if str(data.get("errorCode")) != "0":
        log.warning("Hik-Connect %s falló: %s", what, str(data)[:500])
        raise HikConnectError(f"{what}: {data.get('message') or data.get('errorCode')}")
    return data.get("data") or {}


def normalize_url(url: str | None) -> str | None:
    url = (url or "").strip().rstrip("/")
    if not url:
        return None
    if not url.startswith("http"):
        url = "https://" + url
    # si pegaron la dirección del portal con una ruta, quedarse con el dominio
    parts = url.split("/")
    return "/".join(parts[:3])


class HikConnect:
    def __init__(self, app_key: str, secret: str, api_url: str | None = None, timeout: float = 20.0):
        if not app_key or not secret:
            raise HikConnectError("Falta la clave de la API de Hik-Connect (AppKey y SecretKey) en el cliente")
        self.app_key, self.secret = app_key.strip(), secret.strip()
        self.api_url = normalize_url(api_url)
        self.http = httpx.Client(timeout=timeout)
        self.server: str | None = None  # el que aceptó las credenciales

    def close(self) -> None:
        self.http.close()

    # ---------- token ----------
    def _token(self) -> tuple[str, str]:
        with _lock:
            t = _tokens.get(self.app_key)
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
            if not token:
                errors.append(f"{base}: sin accessToken")
                continue
            exp = data.get("expireTime")
            exp = exp / 1000 if isinstance(exp, (int, float)) and exp > 10**11 else (exp or time.time() + 3600)
            with _lock:
                _tokens[self.app_key] = (token, area, float(exp), base)
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
                _tokens.pop(self.app_key, None)
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
                channel = c.get("channelNo") or c.get("channel") or c.get("cameraNo") or c.get("chanNo")
                out.append({"id": str(c["id"]), "name": c.get("name") or str(c["id"]),
                            "serial": dev.get("serialNo") or "", "device_name": dev.get("name") or "",
                            "channel": int(channel) if str(channel or "").isdigit() else None,
                            "online": c.get("onlineStatus", c.get("online"))})
            if len(items) < 200:
                break
        return out

    def stream_url(self, resource_id: str, serial: str, kind: str = "live", quality: str = "sub",
                   start: datetime | None = None, end: datetime | None = None, expire: int = 3600) -> str:
        """Link HLS temporal. `kind`: live (vivo), playback (disco del grabador) o cloud (nube).
        `start`/`end` en hora del grabador (local)."""
        body = {"resourceId": resource_id, "deviceSerial": serial, "type": TYPES[kind], "protocol": 2,
                "quality": "2" if quality == "sub" else "1", "expireTime": int(expire)}
        if kind != "live":
            if not (start and end):
                raise HikConnectError("Para la grabación hace falta inicio y fin")
            body["startTime"] = start.strftime("%Y-%m-%d %H:%M:%S")
            body["stopTime"] = end.strftime("%Y-%m-%d %H:%M:%S")
        data = self._post("/api/hccgw/video/v1/live/address/get", body, "link de video")
        url = data.get("url")
        if not url:
            log.warning("Hik-Connect: sin url en la respuesta: %s", str(data)[:500])
            raise HikConnectError("Hik-Connect no devolvió el link de video")
        return url
