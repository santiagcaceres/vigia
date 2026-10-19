"""Conexión con cámaras y grabadores (NVR/DVR/XVR) Dahua.

SmartPSS Lite y DMSS son solo programas para mirar: no exponen el video a terceros.
Este sistema se conecta directo al grabador, igual que SmartPSS, usando:

- RTSP en vivo:      rtsp://usuario:clave@IP:554/cam/realmonitor?channel=<canal>&subtype=0  (principal)
                     rtsp://usuario:clave@IP:554/cam/realmonitor?channel=<canal>&subtype=1  (secundario)
- RTSP de grabación: rtsp://usuario:clave@IP:554/cam/playback?channel=<canal>&starttime=...&endtime=...
                     con fecha en formato AAAA_MM_DD_HH_MM_SS
- API CGI (HTTP):    /cgi-bin/magicBox.cgi?action=getSystemInfo, /cgi-bin/configManager.cgi?...
                     con autenticación Digest

Ojo con la hora de la grabación: algunos equipos interpretan starttime/endtime en hora
local y otros en UTC. Eso se elige por cámara con `playback_time`; si el clip llega
corrido, probar la otra opción.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime

import httpx

from .cameras import BaseCamera, download_nvr_clip, test_stream

log = logging.getLogger("vigia.dahua")

__all__ = ["DahuaCamera", "download_nvr_clip", "cgi_get", "probe", "test_stream"]


@dataclass
class DahuaCamera(BaseCamera):
    brand: str = "dahua"

    def live_url(self, stream: str | None = None) -> str:
        if self.source_url:
            return self.source_url
        subtype = 0 if (stream or self.detect_stream) == "main" else 1
        return f"{self._base()}/cam/realmonitor?channel={self.channel}&subtype={subtype}"

    def playback_url(self, start: datetime, end: datetime) -> str:
        return (f"{self._base()}/cam/playback?channel={self.channel}"
                f"&starttime={self._fmt(start)}&endtime={self._fmt(end)}")

    def _fmt(self, dt: datetime) -> str:
        return self._utc_or_local(dt).strftime("%Y_%m_%d_%H_%M_%S")


def cgi_get(cam: DahuaCamera, path: str, timeout: float = 8.0) -> dict[str, str]:
    """Llama a la API CGI y devuelve las líneas `clave=valor` como diccionario."""
    url = f"http://{cam.host}:{cam.http_port}{path}"
    r = httpx.get(url, auth=httpx.DigestAuth(cam.username, cam.password), timeout=timeout)
    r.raise_for_status()
    out: dict[str, str] = {}
    for line in r.text.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def probe(cam: DahuaCamera) -> dict:
    """Datos del equipo y lista de canales. Útil para dar de alta las cámaras."""
    info: dict = {"host": cam.host, "brand": "dahua"}
    try:
        sys_info = cgi_get(cam, "/cgi-bin/magicBox.cgi?action=getSystemInfo")
        info["model"] = sys_info.get("deviceType") or sys_info.get("model")
        info["serialNumber"] = sys_info.get("serialNumber")
    except Exception as exc:
        info["error"] = f"La API CGI no responde: {exc}"
        return info
    for path, key in (("/cgi-bin/magicBox.cgi?action=getDeviceType", "deviceType"),
                      ("/cgi-bin/magicBox.cgi?action=getSoftwareVersion", "version")):
        try:
            info.setdefault(key, next(iter(cgi_get(cam, path).values()), None))
        except Exception:
            pass
    channels = []
    try:  # en un grabador, los nombres de canal dan la cantidad y el nombre de cada cámara
        titles = cgi_get(cam, "/cgi-bin/configManager.cgi?action=getConfig&name=ChannelTitle")
        for k, v in titles.items():
            # table.ChannelTitle[0].Name=Camara 01  ->  canal 1
            if ".Name" in k and "[" in k:
                idx = int(k.split("[", 1)[1].split("]", 1)[0])
                channels.append({"channel": idx + 1, "name": v or f"Canal {idx + 1}", "ip": cam.host})
        channels.sort(key=lambda c: c["channel"])
    except Exception as exc:
        info["channels_error"] = str(exc)
    info["channels"] = channels
    return info


def main() -> None:
    p = argparse.ArgumentParser(description="Probar conexión con un grabador o cámara Dahua")
    p.add_argument("--host", required=True)
    p.add_argument("--user", default="admin")
    p.add_argument("--password", required=True)
    p.add_argument("--http-port", type=int, default=80)
    p.add_argument("--rtsp-port", type=int, default=554)
    p.add_argument("--channel", type=int, default=1)
    a = p.parse_args()
    cam = DahuaCamera(a.host, a.user, a.password, a.channel, a.rtsp_port, a.http_port)
    info = probe(cam)
    print("Equipo:", {k: v for k, v in info.items() if k != "channels"})
    for ch in info.get("channels", []):
        print(f"  canal {ch['channel']:>2}: {ch['name']} ({ch['ip']})")
    for stream in ("sub", "main"):
        print(f"Stream {stream} canal {a.channel}:", test_stream(cam, stream))


if __name__ == "__main__":
    main()
