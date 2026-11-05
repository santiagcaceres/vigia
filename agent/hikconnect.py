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


class FormatNotGiven(RuntimeError):
    """El servidor no confirmó el formato pedido (un servidor de antes del plan B, que ignora el formato y manda HLS).
    Siempre hace pasar al formato siguiente, aunque el pedido haya andado antes."""


LINK_FAILS_BEFORE_SWITCH = 3  # pedidos de link fallidos seguidos con el formato bueno antes de probar el otro


@dataclass
class HikConnectCamera(BaseCamera):
    camera_id: int = 0
    serial: str | None = None
    brand: str = "hikconnect"
    _proto: int = -1  # índice en PROTOCOLS del formato que se pide ahora (-1 = todavía no se eligió)
    _fails: int = 0  # aperturas seguidas que fallaron con el formato de ahora
    _link_fails: int = 0  # pedidos de link seguidos que fallaron con el formato de ahora
    _last_live: str = ""  # el último link del vivo (open_failure de otra cosa, como la grabación, no mueve el formato)

    def _url(self, kind: str, quality: str, start: datetime | None = None, end: datetime | None = None,
             protocol: str = "hls") -> str:
        if _provider is None:
            raise RuntimeError("Las cámaras por Hik-Connect solo andan en la PC central conectada al servidor")
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
        self._fails = self._link_fails = 0

    def live_url(self, stream: str | None = None) -> str:
        """Link nuevo en el formato de ahora. Si Hik-Connect (o el servidor) no da ese formato, prueba el siguiente.
        Un error con HLS (cifrado prendido, grabador sin conexión…) o con el formato que ya anduvo suele ser de la
        conexión, no del formato: se muestra y no se cambia de formato, salvo que se repita LINK_FAILS_BEFORE_SWITCH
        veces seguidas (un error que no se va solo, por ejemplo Hik-Connect que dejó de dar ese formato)."""
        quality = stream or self.detect_stream
        begin = PROTOCOLS.index(self.protocol)
        first_error: RuntimeError | None = None
        for _ in PROTOCOLS:
            proto = self.protocol
            try:
                url = self._url("live", quality, protocol=proto)
            except FormatNotGiven as exc:
                log.info("[cámara %s] %s: pruebo el siguiente formato", self.camera_id, exc)
                first_error = first_error or exc
                with _lock:
                    if _good.get(self.camera_id) == proto:
                        del _good[self.camera_id]
                self._next()
                continue
            except RuntimeError as exc:
                first_error = first_error or exc
                if proto == "hls" or proto == self._known_good():
                    self._link_fails += 1
                    if self._link_fails < LINK_FAILS_BEFORE_SWITCH:
                        self._proto = begin
                        raise
                    log.info("[cámara %s] %d errores seguidos pidiendo el video en %s (%s): pruebo el otro formato",
                             self.camera_id, self._link_fails, proto.upper(), exc)
                else:
                    log.info("[cámara %s] Hik-Connect no dio el video en %s (%s): pruebo el siguiente",
                             self.camera_id, proto.upper(), exc)
                self._next()
                continue
            self._link_fails = 0
            self._last_live = url
            return url
        self._proto = begin
        raise first_error or RuntimeError("Hik-Connect no dio el link de video en ningún formato")

    def opened(self) -> None:
        """El video del vivo abrió: este formato anda (para esta cámara y para las que se armen después)."""
        global _last_good
        proto = self.protocol
        with _lock:
            changed = _good.get(self.camera_id) != proto
            _good[self.camera_id] = proto
            _last_good = proto
        self._fails = self._link_fails = 0
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
        cuando el grabador vuelve la cámara está casi siempre en el formato que anda. Otro link (la grabación, que
        puede bajarse en otro hilo con la misma cámara) no mueve el formato del vivo."""
        why = diagnose(url)
        if url != self._last_live:
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
    """Pide el link directo para saber por qué OpenCV no lo pudo abrir (de un video en vivo, solo el principio). Corre
    en otro hilo con un tope total: un servidor que gotea bytes (o cabeceras) no traba el hilo de la cámara."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        return f"no se pudo abrir el video por {parts.scheme.upper()} desde {parts.hostname}"
    out: list[str] = []
    t = threading.Thread(target=lambda: out.append(_diagnose(url, timeout)), daemon=True, name="diagnostico-hc")
    t.start()
    t.join(2 * timeout + 2)
    if out:
        return out[0]
    return (f"el servidor de video de Hik-Connect ({parts.hostname}) tardó más de {2 * timeout:.0f} s en responder: "
            "suele ser del lado de Hikvision o el grabador sin conexión")


def _diagnose(url: str, timeout: float) -> str:
    parts = urlsplit(url)
    cut, ended = None, False
    try:
        deadline = time.monotonic() + 2 * timeout  # (un goteo de bytes no lo traba)
        with httpx.stream("GET", url, timeout=timeout, follow_redirects=True) as resp:
            head = b""
            try:
                for chunk in resp.iter_bytes():
                    head += chunk
                    if len(head) >= 2048 or time.monotonic() > deadline:
                        break
                else:
                    ended = True  # el servidor terminó la respuesta solo
            except (httpx.TimeoutException, httpx.ReadError, httpx.RemoteProtocolError) as exc:
                cut = exc  # contestó y después dejó de mandar o cortó: no es «no se pudo conectar»
            status = resp.status_code
    except httpx.RemoteProtocolError:
        # Conectó y el servidor de video de Hikvision cortó sin mandar nada (5 nov: todas las cámaras de Hik-Connect,
        # de dos cuentas y dos grabadores, desde la máquina y desde una PC a la vez): no es el antivirus de la PC.
        return (f"el servidor de video de Hik-Connect ({parts.hostname}) cortó la conexión sin mandar el video: suele "
                "ser del lado de Hikvision (si es un solo grabador, que esté sin internet). Probá verla en la app "
                "Hik-Connect")
    except (httpx.ReadTimeout, httpx.ReadError, httpx.WriteError, httpx.WriteTimeout) as exc:
        host = exc.request.url.host if getattr(exc, "_request", None) is not None else parts.hostname
        return (f"el servidor de video de Hik-Connect ({host}) aceptó la conexión pero no respondió: suele ser del "
                "lado de Hikvision o el grabador sin conexión. Probá verla en la app Hik-Connect")
    except httpx.HTTPError as exc:
        req = getattr(exc, "_request", None)
        host, port = (req.url.host, req.url.port) if req is not None else (parts.hostname, parts.port)
        return (f"no se pudo conectar con {host}:{port or 443}; revisá que el antivirus o el firewall de esta PC o de la "
                "red no lo bloqueen")
    flv = head[:3] == b"FLV"
    if status == 200 and (cut is not None or (ended and flv and len(head) < 2048)):
        if not head:
            return (f"Hik-Connect respondió pero no mandó el video en {timeout:.0f} s: suele ser el grabador sin "
                    "conexión o sin transmisiones libres, no esta PC")
        what = "empezó a mandar el video (FLV)" if flv else "empezó a mandar el video"
        return (f"Hik-Connect respondió y {what}, pero después lo cortó: suele ser el grabador sin conexión o sin "
                "transmisiones libres, no esta PC")
    r = httpx.Response(status, content=head[:2048])
    body = " ".join(r.text[:300].split())
    if r.status_code != 200:
        return f"Hik-Connect respondió {r.status_code}: {body[:200] or 'sin detalle'}"
    if flv:
        return "el video llega (FLV) pero no se pudo leer: revisá que la segunda señal de esa cámara esté en H.264"
    if "#EXTM3U" not in body:
        return f"Hik-Connect no mandó una lista de video: {body[:200] or 'respuesta vacía'}"
    return ("la lista de video llega bien pero el video no se pudo leer: revisá que la segunda señal de esa "
            "cámara esté en H.264 (no H.265)")


def probe(cam: HikConnectCamera) -> dict:
    """No hay página del grabador a la que entrar: se informa la serie. La prueba real es el video."""
    return {"ok": True, "model": "por Hik-Connect (sin puertos)", "serialNumber": cam.serial, "channels": []}
