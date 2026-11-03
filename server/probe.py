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
import socket
import xml.etree.ElementTree as ET
from urllib.parse import quote

import httpx

TIMEOUT = 8.0
MAX_BODY = 512 * 1024  # un grabador contesta unos KB: más que esto no es un grabador (y no se junta en memoria)
# una IP o un nombre (DDNS), sin «http://», puerto, ruta, usuario ni espacios: lo demás va en sus campos
HOST_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9.-]{0,253})$")


def valid_host(host: str | None) -> bool:
    """El host de una cámara o de un grabador es solo una IP o un nombre (vacío también vale: Hik-Connect o link)."""
    host = (host or "").strip()
    if not host:
        return True
    if HOST_RE.match(host):
        return True
    try:
        ipaddress.IPv6Address(host.strip("[]"))
        return True
    except ValueError:
        return False


def reachable_public(host: str) -> str | None:
    """Por qué el servidor no va a probar ese host (None si se puede): tiene que ser un nombre o IP que apunte a internet.
    El servidor no prueba direcciones de redes internas (las de un local no las alcanza igual, y así nadie puede usarlo
    para tocar servicios internos de la nube)."""
    if not valid_host(host) or not host:
        return "Poné solo la IP o el DDNS del grabador (sin http://, puerto ni ruta)"
    try:
        infos = socket.getaddrinfo(host.strip("[]"), None, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, UnicodeError):
        return "No existe ese nombre (DDNS) o no se pudo resolver"
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if not ip.is_global:
            return ("Esa dirección es de una red interna: desde el servidor no se llega. Probala desde la PC del local "
                    "(«Probar conexión» de la app)")
    return None


def public_https_url(url: str | None) -> str | None:
    """Por qué no se le manda nada a esa dirección (None si sirve): tiene que ser https, sin usuario ni clave en la
    dirección, y apuntar a internet (no a una red interna ni a los datos de la máquina de la nube)."""
    try:
        u = httpx.URL((url or "").strip())
    except Exception:  # noqa: BLE001
        return "La dirección no es válida"
    if u.scheme != "https" or not u.host:
        return "La dirección tiene que empezar con https://"
    if u.userinfo:
        return "La dirección no puede llevar usuario ni clave"
    return reachable_public(u.host)


def _get(client: httpx.Client, url: str) -> httpx.Response:
    """GET con tope de tamaño (`MAX_BODY`)."""
    with client.stream("GET", url) as r:
        body = bytearray()
        for chunk in r.iter_bytes():
            body += chunk
            if len(body) > MAX_BODY:
                raise ValueError("respuesta demasiado grande para ser un grabador")
        return httpx.Response(r.status_code, headers=r.headers, content=bytes(body), request=r.request)


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
    r = _get(client, base + "/ISAPI/System/deviceInfo")
    r.raise_for_status()
    dev = _strip(ET.fromstring(r.content))
    info = {k: dev.findtext(k) for k in ("deviceName", "model", "serialNumber", "firmwareVersion", "deviceType")}
    channels = []
    try:
        root = _strip(ET.fromstring(_get(client, base + "/ISAPI/ContentMgmt/InputProxy/channels").content))
        for ch in root.iter("InputProxyChannel"):
            channels.append({"channel": int(ch.findtext("id") or 0), "name": ch.findtext("name")})
    except Exception:
        pass
    if not channels:
        try:
            root = _strip(ET.fromstring(_get(client, base + "/ISAPI/System/Video/inputs/channels").content))
            for ch in root.iter("VideoInputChannel"):
                channels.append({"channel": int(ch.findtext("id") or 0), "name": ch.findtext("name")})
        except Exception:
            pass
    if not channels:
        try:
            root = _strip(ET.fromstring(_get(client, base + "/ISAPI/Streaming/channels").content))
            ids = sorted({int(c.findtext("id") or 0) // 100 for c in root.iter("StreamingChannel")})
            channels = [{"channel": i, "name": f"Canal {i}"} for i in ids if i]
        except Exception:
            pass
    return {**info, "channels": channels}


def _dahua_kv(client: httpx.Client, url: str) -> dict[str, str]:
    r = _get(client, url)
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
    if http is None and (why := reachable_public(host)):
        return {"ok": False, "error": why}
    if not valid_host(host):
        return {"ok": False, "error": "Poné solo la IP o el DDNS del grabador (sin http://, puerto ni ruta)"}
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
    except Exception as exc:  # noqa: BLE001 (sin el texto crudo: puede traer lo que contestó otro servicio)
        return {"ok": False, "error": f"No contestó como un grabador ({type(exc).__name__})"}
