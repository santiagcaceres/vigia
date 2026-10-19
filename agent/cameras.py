"""Capa común a todas las marcas de cámaras y grabadores (NVR/DVR).

El agente siempre hace lo mismo: abre un RTSP en vivo para analizar y, cuando hay un
evento, baja el tramo grabado del equipo. Lo único que cambia entre marcas son las
direcciones RTSP y la API HTTP de consulta. Marcas soportadas:

- Hikvision (`agent/hikvision.py`)  — iVMS-4200 / Hik-Connect son solo visores.
- Dahua     (`agent/dahua.py`)      — SmartPSS Lite / DMSS son solo visores.

Las apps de nube de ambas marcas NO entregan el video a software de terceros: hay que
conectarse al equipo, en la red del local o por VPN.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo

from core.clips import redact, transcode_h264, video_duration

log = logging.getLogger("vigia.camaras")

BRANDS = ("hikvision", "dahua")


@dataclass
class BaseCamera:
    """Datos de conexión comunes. Cada marca define cómo se arman sus URLs."""
    host: str
    username: str = "admin"
    password: str = ""
    channel: int = 1
    rtsp_port: int = 554
    http_port: int = 80
    detect_stream: str = "sub"  # "sub" (recomendado) o "main"
    nvr_clips: bool = True
    playback_time: str = "local"  # cómo interpreta el equipo starttime/endtime: "local" o "utc"
    timezone: str = "America/Argentina/Buenos_Aires"
    source_url: str | None = None  # URL RTSP manual (otra marca o caso especial)
    brand: str = "hikvision"

    @classmethod
    def from_config(cls, d: dict) -> "BaseCamera":
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in fields and v is not None})

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    def _auth(self) -> str:
        if not self.username:
            return ""
        return f"{quote(self.username, safe='')}:{quote(self.password, safe='')}@"

    def _base(self) -> str:
        return f"rtsp://{self._auth()}{self.host}:{self.rtsp_port}"

    def _utc_or_local(self, dt: datetime) -> datetime:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=self.tz)
        return dt.astimezone(timezone.utc if self.playback_time == "utc" else self.tz)

    def live_url(self, stream: str | None = None) -> str:
        raise NotImplementedError

    def playback_url(self, start: datetime, end: datetime) -> str:
        raise NotImplementedError

    def redacted(self, url: str) -> str:
        return redact(url)


def build_camera(cfg: dict) -> BaseCamera:
    """Crea la cámara que corresponda según `brand` (por defecto, Hikvision)."""
    brand = (cfg.get("brand") or "hikvision").lower()
    if brand == "dahua":
        from .dahua import DahuaCamera
        return DahuaCamera.from_config(cfg)
    from .hikvision import HikCamera
    return HikCamera.from_config(cfg)


def download_nvr_clip(cam: BaseCamera, start: datetime, end: datetime, out: str | Path,
                      timeout: int = 120) -> bool:
    """Descarga del grabador el tramo [start, end] y lo deja en H.264. False si no sirve.

    Sirve para cualquier marca: solo usa `playback_url` de la cámara.
    """
    duration = (end - start).total_seconds()
    url = cam.playback_url(start, end)
    ok = transcode_h264(url, out, timeout=timeout, duration=duration,
                        extra_input=["-rtsp_transport", "tcp", "-timeout", "10000000"])
    if not ok:
        log.info("No se pudo bajar la grabación del equipo (%s)", cam.redacted(url))
        return False
    got = video_duration(out) or 0.0
    if got < duration * 0.5:
        log.info("Grabación incompleta (%.1fs de %.1fs); se usa el buffer local", got, duration)
        Path(out).unlink(missing_ok=True)
        return False
    return True


def test_stream(cam: BaseCamera, stream: str = "sub") -> dict:
    """Abre el RTSP y lee un cuadro. Sirve para verificar usuario, clave, puerto y canal."""
    import os

    import cv2

    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    cap = cv2.VideoCapture(cam.live_url(stream), cv2.CAP_FFMPEG)
    try:
        ok, frame = cap.read()
        if not ok:
            return {"ok": False, "error": "No se pudo leer video (revisar usuario, clave, puerto RTSP y canal)"}
        return {"ok": True, "width": frame.shape[1], "height": frame.shape[0],
                "fps": cap.get(cv2.CAP_PROP_FPS)}
    finally:
        cap.release()


def probe(cam: BaseCamera) -> dict:
    """Consulta datos del equipo y sus canales, según la marca."""
    if cam.brand == "dahua":
        from .dahua import probe as _probe
    else:
        from .hikvision import probe as _probe
    return _probe(cam)
