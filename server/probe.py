"""Prueba de conexión a un grabador desde el servidor (sin OpenCV: solo HTTP).

Sirve cuando el DVR del cliente se expone a internet por el router (IP pública o DDNS +
puertos redirigidos): el servidor en Vercel entra por el puerto HTTP, pide los datos del
equipo y la lista de canales, y así se confirma que la IP, el puerto, el usuario y la clave
están bien antes de que el agente intente leer el video. El RTSP lo prueba el agente.

Mismas rutas que usa el agente (agent/hikvision.py y agent/dahua.py), pero sin importar esos
módulos: traen OpenCV y no entra en el build de Vercel.
"""

from __future__ import annotations

import ipaddress
import re
import xml.etree.ElementTree as ET
from urllib.parse import quote

import httpx

TIMEOUT = 8.0


def rtsp_url(brand: str, host: str, rtsp_port: int, channel: int, username: str, password: str | None,
             stream: str = "sub") -> str:
    auth = f"{quote(username, safe='')}:{quote(password or '', safe='')}@" if username else ""
    base = f"rtsp://{auth}{host}:{rtsp_port}"
    if (brand or "hikvision") == "dahua":
        return f"{base}/cam/realmonitor?channel={channel}&subtype={0 if stream == 'main' else 1}"
    return f"{base}/Streaming/Channels/{channel}{'01' if stream == 'main' else '02'}"


def is_public_host(host: str) -> bool:
    """True si la IP (o el nombre) apunta a internet y no a una red local / VPN."""
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return not host.endswith((".local", ".ts.net", ".lan"))  # un DDNS cuenta como público
    cgnat = ipaddress.ip_network("100.64.0.0/10")  # también es el rango de Tailscale
    return not (ip.is_private or ip.is_loopback or ip.is_link_local or ip in cgnat)


def _strip(root: ET.Element) -> ET.Element:
    for el in root.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return root


def _hik(client: httpx.Client, base: str) -> dict:
    r = client.get(base + "/ISAPI/System/deviceInfo")
    r.raise_for_status()
    dev = _strip(ET.fromstring(r.content))
    info = {k: dev.findtext(k) for k in ("deviceName", "model", "serialNumber", "firmwareVersion", "deviceType")}
    channels = []
    try:
        root = _strip(ET.fromstring(client.get(base + "/ISAPI/ContentMgmt/InputProxy/channels").content))
        for ch in root.iter("InputProxyChannel"):
            channels.append({"channel": int(ch.findtext("id") or 0), "name": ch.findtext("name")})
    except Exception:
        pass
    if not channels:
        try:
            root = _strip(ET.fromstring(client.get(base + "/ISAPI/System/Video/inputs/channels").content))
            for ch in root.iter("VideoInputChannel"):
                channels.append({"channel": int(ch.findtext("id") or 0), "name": ch.findtext("name")})
        except Exception:
            pass
    if not channels:
        try:
            root = _strip(ET.fromstring(client.get(base + "/ISAPI/Streaming/channels").content))
            ids = sorted({int(c.findtext("id") or 0) // 100 for c in root.iter("StreamingChannel")})
            channels = [{"channel": i, "name": f"Canal {i}"} for i in ids if i]
        except Exception:
            pass
    return {**info, "channels": channels}


def _dahua_kv(client: httpx.Client, url: str) -> dict[str, str]:
    r = client.get(url)
    r.raise_for_status()
    out = {}
    for line in r.text.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _dahua(client: httpx.Client, base: str) -> dict:
    sysinfo = _dahua_kv(client, base + "/cgi-bin/magicBox.cgi?action=getSystemInfo")
    info = {"model": sysinfo.get("deviceType") or sysinfo.get("model"), "serialNumber": sysinfo.get("serialNumber")}
    try:
        info["deviceType"] = next(iter(_dahua_kv(client, base + "/cgi-bin/magicBox.cgi?action=getDeviceType").values()), None)
    except Exception:
        pass
    channels = []
    try:
        titles = _dahua_kv(client, base + "/cgi-bin/configManager.cgi?action=getConfig&name=ChannelTitle")
        for k, v in titles.items():
            m = re.search(r"\[(\d+)\]\.Name$", k)
            if m:
                channels.append({"channel": int(m.group(1)) + 1, "name": v or f"Canal {int(m.group(1)) + 1}"})
        channels.sort(key=lambda c: c["channel"])
    except Exception:
        pass
    return {**info, "channels": channels}


def probe(brand: str, host: str, http_port: int, username: str, password: str | None,
          http: httpx.Client | None = None) -> dict:
    """Devuelve {"ok": bool, "error": str | None, "model", "serialNumber", "channels": [...]}."""
    base = f"http://{host}:{http_port}"
    client = http or httpx.Client(auth=httpx.DigestAuth(username, password or ""), timeout=TIMEOUT)
    try:
        data = _dahua(client, base) if (brand or "hikvision") == "dahua" else _hik(client, base)
        return {"ok": True, "error": None, **data}
    except httpx.HTTPStatusError as exc:
        code = exc.response.status_code
        if code == 401:
            msg = "El grabador rechazó el usuario o la clave (401)"
        elif code in (403, 404):
            msg = f"El grabador respondió {code}: ¿la marca es la correcta? ¿la API HTTP (ISAPI / CGI) está habilitada?"
        else:
            msg = f"El grabador respondió {code}"
        return {"ok": False, "error": msg}
    except httpx.ConnectError:
        return {"ok": False, "error": "No se pudo conectar: revisá la IP o el DDNS, el puerto HTTP y que el router "
                                      "redirija ese puerto al grabador"}
    except httpx.TimeoutException:
        return {"ok": False, "error": "No respondió a tiempo: casi siempre es el puerto sin redirigir en el router, "
                                      "o un firewall"}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
