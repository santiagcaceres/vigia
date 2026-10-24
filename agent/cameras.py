"""Capa común a todas las marcas de cámaras y grabadores (NVR/DVR).

El agente siempre hace lo mismo: abre un RTSP en vivo para analizar y, cuando hay un
evento, baja el tramo grabado del equipo. Lo único que cambia entre marcas son las
direcciones RTSP y la API HTTP de consulta. Marcas soportadas:

- Hikvision (`agent/hikvision.py`)  — iVMS-4200 / Hik-Connect son solo visores.
- Dahua     (`agent/dahua.py`)      — SmartPSS Lite / DMSS son solo visores.
- Hikvision por la nube (`agent/hikconnect.py`): sin IP ni puertos; el link de video lo da la API de
  Hik-Connect for Teams a través del servidor de Vigía.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo

from core.clips import redact, transcode_h264, video_duration

log = logging.getLogger("vigia.camaras")

BRANDS = ("hikvision", "dahua", "hikconnect", "ip")  # "ip" = por IP y puertos, la marca se detecta sola


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

    def ffmpeg_input_args(self) -> list[str]:
        return ["-rtsp_transport", "tcp", "-timeout", "10000000"]


@dataclass
class AutoCamera(BaseCamera):
    """Grabador por IP y puertos de marca desconocida: la primera vez que se usa, prueba si responde como
    Hikvision (ISAPI) o como Dahua (CGI) y desde ahí se comporta como esa marca."""
    brand: str = "ip"
    cfg: dict = field(default_factory=dict, repr=False)
    _impl: BaseCamera | None = field(default=None, repr=False)

    def resolve(self) -> BaseCamera:
        if self._impl is None:
            self._impl = build_camera({**self.cfg, "brand": detect_brand(self.cfg)})
        return self._impl

    def live_url(self, stream: str | None = None) -> str:
        return self.resolve().live_url(stream)

    def playback_url(self, start: datetime, end: datetime) -> str:
        return self.resolve().playback_url(start, end)

    def ffmpeg_input_args(self) -> list[str]:
        return self.resolve().ffmpeg_input_args()


_detected: dict[tuple, str] = {}
_detect_lock = threading.Lock()


def detect_brand(cfg: dict) -> str:
    """Hikvision o Dahua, según cómo responde el equipo (primero su página/API, después el video)."""
    key = (cfg.get("host"), cfg.get("http_port"), cfg.get("rtsp_port"), cfg.get("username"))
    with _detect_lock:
        if key in _detected:
            return _detected[key]
    brand = None
    for b in ("hikvision", "dahua"):
        if not build_and_probe({**cfg, "brand": b}).get("error"):
            brand = b
            break
    if brand is None:  # la página del equipo no contesta: probar el video de cada marca
        for b in ("hikvision", "dahua"):
            try:
                if test_stream(build_camera({**cfg, "brand": b})).get("ok"):
                    brand = b
                    break
            except Exception:  # noqa: BLE001
                pass
    if brand is None:
        log.info("No se pudo detectar la marca de %s: se prueba como Hikvision", cfg.get("host"))
        return "hikvision"  # no se guarda: se vuelve a intentar en la próxima conexión
    log.info("Marca detectada para %s: %s", cfg.get("host"), brand.capitalize())
    with _detect_lock:
        _detected[key] = brand
    return brand


def build_and_probe(cfg: dict) -> dict:
    try:
        return probe(build_camera(cfg))
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def build_camera(cfg: dict) -> BaseCamera:
    """Crea la cámara que corresponda según `brand` (por defecto, Hikvision)."""
    brand = (cfg.get("brand") or "hikvision").lower()
    if brand == "ip":
        cam = AutoCamera.from_config({k: v for k, v in cfg.items() if k not in ("brand", "cfg", "_impl")})
        cam.cfg = dict(cfg)
        return cam
    if brand == "dahua":
        from .dahua import DahuaCamera
        return DahuaCamera.from_config(cfg)
    if brand == "hikconnect":
        from .hikconnect import HikConnectCamera
        return HikConnectCamera.from_config(cfg)
    from .hikvision import HikCamera
    return HikCamera.from_config(cfg)


def download_nvr_clip(cam: BaseCamera, start: datetime, end: datetime, out: str | Path,
                      timeout: int = 120) -> bool:
    """Descarga del grabador el tramo [start, end] y lo deja en H.264. False si no sirve.

    Sirve para cualquier marca: solo usa `playback_url` de la cámara.
    """
    duration = (end - start).total_seconds()
    url = cam.playback_url(start, end)
    ok = transcode_h264(url, out, timeout=timeout, duration=duration, extra_input=cam.ffmpeg_input_args())
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
    if isinstance(cam, AutoCamera):
        impl = cam.resolve()
        return {**probe(impl), "brand_detected": impl.brand}
    if cam.brand == "dahua":
        from .dahua import probe as _probe
    elif cam.brand == "hikconnect":
        from .hikconnect import probe as _probe
    else:
        from .hikvision import probe as _probe
    return _probe(cam)
