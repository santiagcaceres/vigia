"""Cámaras Hikvision por la nube de Hik-Connect (sin abrir puertos en el local).

La PC central no conoce la IP del grabador: le pide al servidor de Vigía un link de video temporal
(HLS) que el servidor saca de la API de Hik-Connect for Teams con la clave del cliente
(`server/hikconnect.py`). Ese link se abre con OpenCV como cualquier RTSP. Cada reconexión pide un
link nuevo, así que no importa que venzan.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from .cameras import BaseCamera

# (camera_id, kind, quality, inicio, fin) -> url. Lo instala el agente con su conexión al servidor.
UrlProvider = Callable[[int, str, str, datetime | None, datetime | None], str]
_provider: UrlProvider | None = None


def set_url_provider(fn: UrlProvider | None) -> None:
    global _provider
    _provider = fn


@dataclass
class HikConnectCamera(BaseCamera):
    camera_id: int = 0
    serial: str | None = None
    brand: str = "hikconnect"

    def _url(self, kind: str, quality: str, start: datetime | None = None, end: datetime | None = None) -> str:
        if _provider is None:
            raise RuntimeError("Las cámaras por Hik-Connect solo andan en la PC central conectada al servidor")
        return _provider(self.camera_id, kind, quality, start, end)

    def live_url(self, stream: str | None = None) -> str:
        return self._url("live", stream or self.detect_stream)

    def playback_url(self, start: datetime, end: datetime) -> str:
        return self._url("playback", "main", self._utc_or_local(start), self._utc_or_local(end))

    def ffmpeg_input_args(self) -> list[str]:
        return []

    def redacted(self, url: str) -> str:
        return url.split("?")[0]  # el link lleva un token en la consulta


def probe(cam: HikConnectCamera) -> dict:
    """No hay página del grabador a la que entrar: se informa la serie. La prueba real es el video."""
    return {"ok": True, "model": "por Hik-Connect (sin puertos)", "serialNumber": cam.serial, "channels": []}
