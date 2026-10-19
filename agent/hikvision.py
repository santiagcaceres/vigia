"""Conexión con cámaras y grabadores (NVR/DVR) Hikvision.

iVMS-4200 y Hik-Connect son solo programas para mirar: no exponen video a terceros. Este
sistema se conecta directo al NVR o a cada cámara, igual que iVMS, usando:

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
from datetime import datetime

import httpx

from .cameras import BaseCamera, download_nvr_clip, test_stream

log = logging.getLogger("vigia.hikvision")

__all__ = ["HikCamera", "download_nvr_clip", "isapi_get", "probe", "test_stream"]

@dataclass
class HikCamera(BaseCamera):
    brand: str = "hikvision"

    def live_url(self, stream: str | None = None) -> str:
        if self.source_url:
            return self.source_url
        suffix = "01" if (stream or self.detect_stream) == "main" else "02"
        return f"{self._base()}/Streaming/Channels/{self.channel}{suffix}"

    def playback_url(self, start: datetime, end: datetime) -> str:
        return (f"{self._base()}/Streaming/tracks/{self.channel}01"
                f"?starttime={self._fmt(start)}&endtime={self._fmt(end)}")

    def _fmt(self, dt: datetime) -> str:
        return self._utc_or_local(dt).strftime("%Y%m%dT%H%M%SZ")

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
