"""Cámaras Hikvision por la nube de Hik-Connect (sin abrir puertos en el local).

La PC central no conoce la IP del grabador: le pide al servidor de Vigía un link de video temporal
(HLS) que el servidor saca de la API de Hik-Connect for Teams con la clave del cliente
(`server/hikconnect.py`). Ese link se abre con OpenCV como cualquier RTSP. Cada reconexión pide un
link nuevo, así que no importa que venzan.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlsplit

import httpx

from .cameras import BaseCamera

log = logging.getLogger("vigia.hikconnect")

# (camera_id, kind, quality, inicio, fin[, protocol]) -> url. Lo instala el agente con su conexión al servidor.
UrlProvider = Callable[..., str]
# Formatos en que se le pide el vivo a Hik-Connect, en orden. El 5 nov, desde las 6, el servidor de video de São Paulo
# cortaba todos los HLS (las dos cuentas, los dos grabadores) mientras la app de Hikvision andaba, y por FLV el video
# llegaba: si un formato no abre, la conexión siguiente prueba el próximo, y se queda con el que anda.
# RTMP no se prueba: con las opciones de FFmpeg de todas las cámaras (`timeout` en OPENCV_FFMPEG_CAPTURE_OPTIONS, que
# las cámaras por puertos necesitan) FFmpeg abre RTMP en modo escucha y nunca conecta (lo vio la revisión del 5 nov).
PROTOCOLS = ("hls", "flv")
# El formato que abrió, por cámara y el último de cualquier cámara de este programa (el corte fue del servidor de
# video, para todas a la vez): una cámara que se vuelve a armar (cambio de configuración, «En vivo», «Nombres e
# imágenes») arranca por el que ya anduvo en vez de perder un intento con el que está caído.
_good: dict[int, str] = {}
_last_good: str | None = None
_lock = threading.Lock()
_provider: UrlProvider | None = None


def set_url_provider(fn: UrlProvider | None) -> None:
    global _provider
    _provider = fn


def forget_formats() -> None:
    """Olvidar qué formato anduvo (pruebas)."""
    global _last_good
    with _lock:
        _good.clear()
        _last_good = None


@dataclass
class HikConnectCamera(BaseCamera):
    camera_id: int = 0
    serial: str | None = None
    brand: str = "hikconnect"
    _proto: int = -1  # índice en PROTOCOLS del formato que se pide ahora (-1 = todavía no se eligió)
    _fails: int = 0  # aperturas seguidas que fallaron con el formato de ahora
    _last_kind: str = "live"  # qué fue el último link (la grabación siempre va en HLS: no mueve el formato del vivo)

    def _url(self, kind: str, quality: str, start: datetime | None = None, end: datetime | None = None,
             protocol: str = "hls") -> str:
        if _provider is None:
            raise RuntimeError("Las cámaras por Hik-Connect solo andan en la PC central conectada al servidor")
        self._last_kind = kind
        if protocol == "hls":
            return _provider(self.camera_id, kind, quality, start, end)
        return _provider(self.camera_id, kind, quality, start, end, protocol)

    @property
    def protocol(self) -> str:
        """El formato en que se pide el vivo ahora."""
        if self._proto < 0:
            with _lock:
                first = _good.get(self.camera_id) or _last_good or PROTOCOLS[0]
            self._proto = PROTOCOLS.index(first)
        return PROTOCOLS[self._proto % len(PROTOCOLS)]

    def _known_good(self) -> str | None:
        with _lock:
            return _good.get(self.camera_id)

    def _next(self) -> None:
        self._proto = PROTOCOLS.index(self.protocol) + 1
        self._fails = 0

    def live_url(self, stream: str | None = None) -> str:
        """Link nuevo en el formato de ahora. Si Hik-Connect no da ese formato (FLV o RTMP), prueba el próximo. Un error
        con HLS (cifrado prendido, grabador sin conexión…) o con el formato que ya anduvo es de la conexión, no del
        formato: se muestra como siempre y no se cambia de formato."""
        quality = stream or self.detect_stream
        begin = PROTOCOLS.index(self.protocol)
        for _ in PROTOCOLS:
            proto = self.protocol
            try:
                return self._url("live", quality, protocol=proto)
            except RuntimeError as exc:
                if proto == "hls" or proto == self._known_good():
                    self._proto, self._fails = begin, 0
                    raise
                log.info("[cámara %s] Hik-Connect no dio el video en %s (%s): pruebo el siguiente", self.camera_id,
                         proto.upper(), exc)
                self._next()
        self._proto = begin
        raise RuntimeError("Hik-Connect no dio el link de video en ningún formato")

    def opened(self) -> None:
        """El video abrió: este formato anda (para esta cámara y para las que se armen después)."""
        global _last_good
        if self._last_kind != "live":
            return
        proto = self.protocol
        with _lock:
            changed = _good.get(self.camera_id) != proto
            _good[self.camera_id] = proto
            _last_good = proto
        self._fails = 0
        if changed and proto != PROTOCOLS[0]:
            log.info("[cámara %s] el video de Hik-Connect anda en %s", self.camera_id, proto.upper())

    def playback_url(self, start: datetime, end: datetime) -> str:
        return self._url("playback", "main", self._utc_or_local(start), self._utc_or_local(end))

    def ffmpeg_input_args(self) -> list[str]:
        return []

    def redacted(self, url: str) -> str:
        return url.split("?")[0]  # el link lleva un token en la consulta

    def open_failure(self, url: str) -> str | None:
        """Por qué no abrió y, si es el vivo, qué formato se prueba en la conexión siguiente (al principio: el estado
        que ve la oficina se corta a 300 letras). El formato que ya anduvo se reintenta una vez antes de probar otro, y
        si el otro tampoco abre se vuelve a él: un corte del grabador o de internet no es culpa del formato, y así
        cuando el grabador vuelve la cámara está casi siempre en el formato que anda."""
        why = diagnose(url)
        if self._last_kind != "live":
            return why
        tried, good = self.protocol, self._known_good()
        self._fails += 1
        if tried == good and self._fails < 2:
            return f"[{tried.upper()}, reintento] {why}"
        if good and tried != good:
            self._proto, self._fails = PROTOCOLS.index(good), 0
        else:
            self._next()
        return f"[{tried.upper()} no abrió; pruebo {self.protocol.upper()}] {why}"


def diagnose(url: str, timeout: float = 10.0) -> str:
    """Pide el link directo para saber por qué OpenCV no lo pudo abrir (de un video en vivo, solo el principio)."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        return f"no se pudo abrir el video por {parts.scheme.upper()} desde {parts.hostname}"
    cut = None
    try:
        with httpx.stream("GET", url, timeout=timeout, follow_redirects=True) as resp:
            head, deadline = b"", time.monotonic() + 2 * timeout  # (un goteo de bytes no lo traba para siempre)
            try:
                for chunk in resp.iter_bytes():
                    head += chunk
                    if len(head) >= 2048 or time.monotonic() > deadline:
                        break
            except (httpx.TimeoutException, httpx.ReadError, httpx.RemoteProtocolError) as exc:
                cut = type(exc).__name__  # contestó y después dejó de mandar o cortó: no es «no se pudo conectar»
            status = resp.status_code
    except httpx.RemoteProtocolError:
        # Conectó y el servidor de video de Hikvision cortó sin mandar nada (5 nov: todas las cámaras de Hik-Connect,
        # de dos cuentas y dos grabadores, desde la máquina y desde una PC a la vez): no es el antivirus de la PC.
        return (f"el servidor de video de Hik-Connect ({parts.hostname}) cortó la conexión sin mandar el video. Suele "
                "ser del lado de Hikvision: si les pasa a todas las cámaras de Hik-Connect a la vez, es su servidor; si es "
                "solo un grabador, puede que esté sin internet o con todas sus transmisiones ocupadas. Probá verla en la "
                "app Hik-Connect")
    except httpx.HTTPError as exc:
        return (f"no se pudo conectar con {parts.hostname}:{parts.port or 443} ({type(exc).__name__}); "
                "revisá que el antivirus o el firewall de esta PC o de la red no lo bloqueen")
    if cut is not None and status == 200:
        what = "empezó a mandar el video (FLV)" if head[:3] == b"FLV" else f"mandó {len(head)} bytes"
        return (f"Hik-Connect respondió y {what}, pero después dejó de mandarlo ({cut}): suele ser el grabador sin "
                "conexión o sin transmisiones libres, no esta PC")
    r = httpx.Response(status, content=head[:2048])
    body = " ".join(r.text[:300].split())
    if r.status_code != 200:
        return f"Hik-Connect respondió {r.status_code}: {body[:200] or 'sin detalle'}"
    if head[:3] == b"FLV":
        return "el video llega (FLV) pero no se pudo leer: revisá que la segunda señal de esa cámara esté en H.264"
    if "#EXTM3U" not in body:
        return f"Hik-Connect no mandó una lista de video: {body[:200] or 'respuesta vacía'}"
    return ("la lista de video llega bien pero el video no se pudo leer: revisá que la segunda señal de esa "
            "cámara esté en H.264 (no H.265)")


def probe(cam: HikConnectCamera) -> dict:
    """No hay página del grabador a la que entrar: se informa la serie. La prueba real es el video."""
    return {"ok": True, "model": "por Hik-Connect (sin puertos)", "serialNumber": cam.serial, "channels": []}
