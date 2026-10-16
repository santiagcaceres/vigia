"""Conexión con cámaras y grabadores (NVR/DVR) Hikvision.

iVMS-4200 es solo un programa para mirar: no expone video a terceros. Este sistema se
conecta directo al NVR o a cada cámara, igual que iVMS, usando:

- RTSP en vivo:      rtsp://usuario:clave@IP:554/Streaming/Channels/<canal>01   (principal)
                     rtsp://usuario:clave@IP:554/Streaming/Channels/<canal>02   (secundario)
- RTSP de grabación: rtsp://usuario:clave@IP:554/Streaming/tracks/<canal>01?starttime=...&endtime=...
- ISAPI (HTTP):      /ISAPI/System/deviceInfo, /ISAPI/ContentMgmt/InputProxy/channels

Para analizar se usa el stream secundario (menos carga); para el clip del evento se
descarga la grabación del NVR en calidad completa, y si eso falla se usa el buffer local.
"""

from __future__ import annotations

import argparse
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo

import httpx

from core.clips import redact, transcode_h264, video_duration

log = logging.getLogger("vigia.hikvision")


@dataclass
class HikCamera:
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

    @classmethod
    def from_config(cls, d: dict) -> "HikCamera":
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in fields and v is not None})

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    def _auth(self) -> str:
        if not self.username:
            return ""
        return f"{quote(self.username, safe='')}:{quote(self.password, safe='')}@"

    def live_url(self, stream: str | None = None) -> str:
        if self.source_url:
            return self.source_url
        suffix = "01" if (stream or self.detect_stream) == "main" else "02"
        return f"rtsp://{self._auth()}{self.host}:{self.rtsp_port}/Streaming/Channels/{self.channel}{suffix}"

    def playback_url(self, start: datetime, end: datetime) -> str:
        return (f"rtsp://{self._auth()}{self.host}:{self.rtsp_port}/Streaming/tracks/{self.channel}01"
                f"?starttime={self._fmt(start)}&endtime={self._fmt(end)}")

    def _fmt(self, dt: datetime) -> str:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=self.tz)
        dt = dt.astimezone(timezone.utc if self.playback_time == "utc" else self.tz)
        return dt.strftime("%Y%m%dT%H%M%SZ")

    def redacted(self, url: str) -> str:
        return redact(url)


def download_nvr_clip(cam: HikCamera, start: datetime, end: datetime, out: str | Path,
                      timeout: int = 120) -> bool:
    """Descarga del NVR el tramo [start, end] y lo deja en H.264. Devuelve False si no sirve."""
    duration = (end - start).total_seconds()
    url = cam.playback_url(start, end)
    ok = transcode_h264(url, out, timeout=timeout, duration=duration,
                        extra_input=["-rtsp_transport", "tcp", "-timeout", "10000000"])
    if not ok:
        log.info("No se pudo bajar la grabación del NVR (%s)", cam.redacted(url))
        return False
    got = video_duration(out) or 0.0
    if got < duration * 0.5:
        log.info("Grabación del NVR incompleta (%.1fs de %.1fs); se usa el buffer local", got, duration)
        Path(out).unlink(missing_ok=True)
        return False
    return True


def isapi_get(cam: HikCamera, path: str, timeout: float = 8.0) -> ET.Element:
    url = f"http://{cam.host}:{cam.http_port}{path}"
    r = httpx.get(url, auth=httpx.DigestAuth(cam.username, cam.password), timeout=timeout)
    r.raise_for_status()
    root = ET.fromstring(r.content)
    for el in root.iter():  # quitar namespaces para buscar más fácil
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return root


def probe(cam: HikCamera) -> dict:
    """Datos del equipo y lista de canales (en NVR). Útil para dar de alta las cámaras."""
    info: dict = {"host": cam.host}
    try:
        dev = isapi_get(cam, "/ISAPI/System/deviceInfo")
        for k in ("deviceName", "model", "serialNumber", "firmwareVersion", "deviceType"):
            info[k] = dev.findtext(k)
    except Exception as exc:
        info["error"] = f"ISAPI no responde: {exc}"
        return info
    channels = []
    try:
        root = isapi_get(cam, "/ISAPI/ContentMgmt/InputProxy/channels")
        for ch in root.iter("InputProxyChannel"):
            channels.append({"channel": int(ch.findtext("id") or 0), "name": ch.findtext("name"),
                             "ip": ch.findtext("sourceInputPortDescriptor/ipAddress")})
    except Exception:
        try:  # cámara suelta: un solo canal
            root = isapi_get(cam, "/ISAPI/Streaming/channels")
            ids = sorted({int(c.findtext("id") or 0) // 100 for c in root.iter("StreamingChannel")})
            channels = [{"channel": i, "name": f"Canal {i}", "ip": cam.host} for i in ids if i]
        except Exception as exc:
            info["channels_error"] = str(exc)
    info["channels"] = channels
    return info


def test_stream(cam: HikCamera, stream: str = "sub") -> dict:
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


def main() -> None:
    p = argparse.ArgumentParser(description="Probar conexión con un NVR o cámara Hikvision")
    p.add_argument("--host", required=True)
    p.add_argument("--user", default="admin")
    p.add_argument("--password", required=True)
    p.add_argument("--http-port", type=int, default=80)
    p.add_argument("--rtsp-port", type=int, default=554)
    p.add_argument("--channel", type=int, default=1)
    a = p.parse_args()
    cam = HikCamera(a.host, a.user, a.password, a.channel, a.rtsp_port, a.http_port)
    info = probe(cam)
    print("Equipo:", {k: v for k, v in info.items() if k != "channels"})
    for ch in info.get("channels", []):
        print(f"  canal {ch['channel']:>2}: {ch['name']} ({ch['ip']})")
    for stream in ("sub", "main"):
        print(f"Stream {stream} canal {a.channel}:", test_stream(cam, stream))


if __name__ == "__main__":
    main()
