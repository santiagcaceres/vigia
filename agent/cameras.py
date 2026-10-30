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

    def open_failure(self, url: str) -> str | None:
        """Por qué no abrió el video, si se puede averiguar (OpenCV no lo dice). None = no se sabe."""
        return rtsp_diagnose(url) if url.lower().startswith("rtsp://") else None


@dataclass
class AutoCamera(BaseCamera):
    """Grabador por IP y puertos de marca desconocida: la primera vez que se usa, prueba si responde como
    Hikvision (ISAPI) o como Dahua (CGI) y desde ahí se comporta como esa marca."""
    brand: str = "ip"
    cfg: dict = field(default_factory=dict, repr=False)
    _impl: BaseCamera | None = field(default=None, repr=False)

    def resolve(self) -> BaseCamera:
        if self._impl is None:
            brand = None
            if self.cfg.get("source_url"):  # con el link pegado la marca se lee del link: no hace falta probar
                try:
                    brand = parse_rtsp(self.cfg["source_url"]).get("brand")
                except ValueError:
                    brand = None
                brand = brand or "hikvision"
            self._impl = build_camera({**self.cfg, "brand": brand or detect_brand(self.cfg)})
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
    if brand is None:  # la página del equipo no contesta: probar el video de cada marca, en las dos calidades (si el
        # grabador no tiene activada la secundaria, con la principal igual se sabe la marca)
        first = cfg.get("detect_stream") or "sub"
        for stream in (first, "main" if first == "sub" else "sub"):
            for b in ("hikvision", "dahua"):
                try:
                    if test_stream(build_camera({**cfg, "brand": b}), stream).get("ok"):
                        brand = b
                        break
                except Exception:  # noqa: BLE001
                    pass
            if brand:
                break
    if brand is None:
        log.info("%s: desde esta PC no contestan ni la página del grabador ni el video, así que no se sabe si es "
                 "Hikvision o Dahua: se prueba como Hikvision (el motivo, en el aviso que sigue)", cfg.get("host"))
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


def test_stream(cam: BaseCamera, stream: str = "sub", explain: bool = False) -> dict:
    """Abre el RTSP y lee un cuadro. Sirve para verificar usuario, clave, puerto y canal. `explain`: si no anda,
    averiguar por qué (`rtsp_diagnose`: tarda unos segundos más)."""
    import os

    import cv2

    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    url = cam.live_url(stream)
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    try:
        ok, frame = cap.read()
        if not ok:
            why = cam.open_failure(url) if explain else None
            return {"ok": False, "error": why or "No se pudo leer video (revisar usuario, clave, puerto RTSP y canal)"}
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


def parse_rtsp(url: str) -> dict:
    """Los datos de un link RTSP como los que pasan los técnicos, para no copiarlos a mano:
    rtsp://usuario:clave@host:puerto/Streaming/Channels/401 (Hikvision: canal 4, calidad principal; 402 secundaria) o
    rtsp://usuario:clave@host:puerto/cam/realmonitor?channel=4&subtype=0 (Dahua). ValueError si no se entiende."""
    import re
    from urllib.parse import parse_qs, unquote, urlsplit

    u = urlsplit((url or "").strip())
    if u.scheme.lower() != "rtsp" or not u.hostname:
        raise ValueError("No parece un link RTSP (tiene que empezar con rtsp://)")
    try:
        port = u.port or 554
    except ValueError as exc:
        raise ValueError("El puerto del link no es un número") from exc
    out = {"host": u.hostname, "rtsp_port": port, "username": unquote(u.username or ""),
           "password": unquote(u.password or ""), "channel": None, "brand": None, "stream": None}
    path = u.path.rstrip("/")
    m = re.search(r"/Streaming/(?:Channels|tracks)/(\d+)$", path, re.I)
    if m and len(m.group(1)) >= 3:  # 401 = canal 4 · 01 principal / 02 secundaria
        out.update(channel=int(m.group(1)[:-2]), brand="hikvision",
                   stream="main" if m.group(1).endswith("01") else "sub")
    elif m:
        out.update(channel=int(m.group(1)), brand="hikvision")
    elif re.search(r"/h26[45]/ch(\d+)/", path, re.I):  # Hikvision viejo: /h264/ch1/main/av_stream
        mm = re.search(r"/h26[45]/ch(\d+)/(main|sub)", path, re.I)
        out.update(channel=int(mm.group(1)) if mm else None, brand="hikvision",
                   stream=(mm.group(2).lower() if mm else None))
    elif "realmonitor" in path.lower():
        q = parse_qs(u.query)
        ch = (q.get("channel") or [None])[0]
        sub = (q.get("subtype") or [None])[0]
        out.update(channel=int(ch) if ch and ch.isdigit() else None, brand="dahua",
                   stream=None if sub is None else ("main" if sub == "0" else "sub"))
    return out


def probe_connection(cfg: dict) -> dict:
    """Probar un grabador por IP y puertos DESDE ESTA PC, antes de cargar sus cámaras: la página (modelo, canales; si
    el puerto HTTP está abierto) y el video del canal (la calidad secundaria y, si no anda, la principal). `cfg`: host,
    rtsp_port, http_port (None = no está abierto), username, password, channel y, si se pegó, `source_url` (el link:
    se prueba tal cual, como VLC)."""
    base = {"host": cfg["host"], "rtsp_port": int(cfg.get("rtsp_port") or 554),
            "http_port": int(cfg.get("http_port") or 80), "username": cfg.get("username") or "",
            "password": cfg.get("password") or "", "channel": int(cfg.get("channel") or 1)}
    http: dict = {"error": "puerto HTTP no indicado"}
    brand = None
    if cfg.get("http_port"):
        for b in ("hikvision", "dahua"):
            res = build_and_probe({**base, "brand": b})
            if not res.get("error"):
                http, brand = res, b
                break
            http = res
    video: dict = {"ok": False, "error": "no se probó"}
    if cfg.get("source_url"):  # el link que pasaron, tal cual
        link = parse_rtsp(cfg["source_url"])
        try:
            video = test_stream(build_camera({**base, "brand": link.get("brand") or "hikvision",
                                              "source_url": cfg["source_url"]}), explain=True)
        except Exception as exc:  # noqa: BLE001
            video = {"ok": False, "error": str(exc)}
        if video.get("ok"):
            video["quality"] = link.get("stream")
            return {"http": http, "video": video, "brand": brand or link.get("brand")}
    brands = [brand] if brand else ["hikvision", "dahua"]
    for i, b in enumerate(brands):
        cam = build_camera({**base, "brand": b})
        for stream in ("sub", "main"):  # la secundaria (menos internet) y, si no anda, la principal
            try:  # el porqué se averigua una vez (con la última marca): tarda unos segundos
                video = test_stream(cam, stream, explain=i == len(brands) - 1 and stream == "main")
            except Exception as exc:  # noqa: BLE001
                video = {"ok": False, "error": str(exc)}
            if video.get("ok"):
                video["quality"] = stream
                return {"http": http, "video": video, "brand": brand or b}
    return {"http": http, "video": video, "brand": brand}


class _NoReply(Exception):
    """El grabador aceptó la conexión pero no contestó el pedido (no es lo mismo que un puerto que no contesta)."""


def _rtsp_request(host: str, port: int, method: str, url: str, headers: dict, timeout: float) -> tuple[int, dict] | None:
    """Un pedido RTSP en una conexión propia: (código, cabeceras en minúscula; las repetidas, juntas). None = la otra
    punta no contestó como RTSP. Si conecta pero no contesta a tiempo: `_NoReply`."""
    import socket

    lines = [f"{method} {url} RTSP/1.0", "CSeq: 1", "Accept: application/sdp", "User-Agent: Vigia"]
    lines += [f"{k}: {v}" for k, v in headers.items()]
    with socket.create_connection((host, port), timeout) as sock:
        sock.settimeout(timeout)
        data = b""
        try:
            sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode())
            while b"\r\n\r\n" not in data and len(data) < 16384:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                data += chunk
        except (TimeoutError, socket.timeout) as exc:
            raise _NoReply from exc
    head = data.split(b"\r\n\r\n", 1)[0].decode("latin-1", "replace").split("\r\n")
    parts = head[0].split(" ", 2) if head else []
    if len(parts) < 2 or not parts[0].upper().startswith("RTSP/") or not parts[1].isdigit():
        return None
    out: dict[str, str] = {}
    for line in head[1:]:
        k, _, v = line.partition(":")
        key = k.strip().lower()
        out[key] = (out[key] + "\n" if key in out else "") + v.strip()
    return int(parts[1]), out


def _authorization(challenges: str, user: str, password: str, method: str, uri: str) -> str | None:
    """La cabecera Authorization para el desafío del grabador (Digest, si lo ofrece; si no, Basic)."""
    import base64
    import hashlib
    import re
    import secrets

    digest = next((c for c in challenges.split("\n") if c.lower().startswith("digest")), None)
    if digest:
        p = dict(re.findall(r'(\w+)="?([^",]*)"?', digest[6:]))
        md5 = lambda t: hashlib.md5(t.encode()).hexdigest()  # noqa: E731,S324 - lo pide el protocolo
        ha1, ha2 = md5(f"{user}:{p.get('realm', '')}:{password}"), md5(f"{method}:{uri}")
        nonce = p.get("nonce", "")
        if "auth" in (p.get("qop") or "").split(","):
            cnonce, nc = secrets.token_hex(8), "00000001"
            resp = md5(f"{ha1}:{nonce}:{nc}:{cnonce}:auth:{ha2}")
            extra = f', qop=auth, nc={nc}, cnonce="{cnonce}"'
        else:
            resp, extra = md5(f"{ha1}:{nonce}:{ha2}"), ""
        return (f'Digest username="{user}", realm="{p.get("realm", "")}", nonce="{nonce}", uri="{uri}", '
                f'response="{resp}"' + (f', opaque="{p["opaque"]}"' if p.get("opaque") else "") + extra)
    if challenges.lower().startswith("basic") or "\nbasic" in challenges.lower():
        return "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()
    return None


def _other_quality(url: str) -> str | None:
    """El mismo canal en la otra calidad: Hikvision …/Channels/402 ↔ 401; Dahua subtype=1 ↔ 0."""
    import re

    m = re.search(r"(/Streaming/Channels/\d+)(0[12])(\b|$)", url, re.I)
    if m:
        return url[:m.start(2)] + ("01" if m.group(2) == "02" else "02") + url[m.end(2):]
    if "subtype=1" in url:
        return url.replace("subtype=1", "subtype=0")
    if "subtype=0" in url:
        return url.replace("subtype=0", "subtype=1")
    return None


def rtsp_diagnose(url: str, timeout: float = 6.0) -> str:
    """Por qué no abre un video RTSP, en castellano (OpenCV solo dice que no pudo): el nombre no existe, el puerto no
    contesta, no es un servidor de video, usuario o clave mal, ese canal o calidad no existe, sin permiso, sin
    transmisiones libres, o el grabador entrega el video y lo que falla es leerlo en esta PC."""
    import socket
    from urllib.parse import unquote, urlsplit, urlunsplit

    u = urlsplit(url)
    host, port = u.hostname or "", u.port or 554
    user, password = unquote(u.username or ""), unquote(u.password or "")
    try:
        socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError):
        return (f"no se encuentra «{host}»: el DDNS no existe, venció o está mal escrito (o esta PC no tiene "
                "internet)")

    def ask(target: str) -> tuple[int, dict] | None:
        clean = urlunsplit((u.scheme, f"{host}:{port}", urlsplit(target).path, urlsplit(target).query, ""))
        res = _rtsp_request(host, port, "DESCRIBE", clean, {}, timeout)
        if res and res[0] == 401 and user:
            auth = _authorization(res[1].get("www-authenticate", ""), user, password, "DESCRIBE", clean)
            if auth:
                res = _rtsp_request(host, port, "DESCRIBE", clean, {"Authorization": auth}, timeout)
        return res

    try:
        res = ask(url)
    except _NoReply:
        return (f"{host}:{port} acepta la conexión pero no contesta el pedido de video: el grabador puede haber "
                "bloqueado a esta PC por intentos fallidos (esperá unos 30 minutos) o la conexión es muy lenta")
    except (TimeoutError, socket.timeout):
        return (f"{host} no contesta en el puerto {port} desde esta PC: no está abierto (o no está redirigido al "
                "grabador) en el router del local, el DDNS apunta a otra IP, o el proveedor de internet del local no "
                "deja entrar. Si VLC sí lo abre desde esta misma PC, es el firewall o el antivirus de Windows: "
                "permitile a «Analítica Crítica» conectarse")
    except ConnectionRefusedError:
        return (f"{host} rechaza la conexión en el puerto {port}: ese puerto no está redirigido al grabador en el "
                "router del local")
    except OSError as exc:
        return (f"no se puede llegar a {host}:{port} ({exc.strerror or exc}): revisá el DDNS, el router del local y "
                "el internet de esta PC")
    if res is None:
        return (f"{host}:{port} contesta pero no como video (RTSP): ¿ese es el puerto de video? (el 554 de adentro; "
                "quizá se redirigió el de la página)")
    code = res[0]
    if code == 200:
        return ("el grabador entrega el video (usuario, clave y canal están bien) pero esta PC no lo pudo leer: "
                "¿video en H.265 o un códec raro? Probá poner el subflujo del grabador en H.264")
    if code == 401:
        return ("el grabador rechazó el usuario o la clave (o el usuario quedó bloqueado por intentos fallidos: "
                "esperá unos 30 minutos)")
    if code == 403:
        return "el usuario no tiene permiso para ver ese canal: dale «vista en vivo» en el grabador"
    if code in (404, 400, 454, 457):
        other = _other_quality(url)
        try:
            alt = ask(other) if other else None
        except OSError:
            alt = None
        if alt and alt[0] == 200:
            return ("ese canal existe pero la calidad que se pide no está activada en el grabador (el subflujo): "
                    "activalo, o elegí la calidad principal")
        return "el grabador contesta pero no tiene ese canal: revisá el número de canal"
    if code in (453, 503):
        return "el grabador no da más transmisiones a la vez: cerrá otras (otra PC, el celular, la máquina de análisis)"
    return f"el grabador respondió {code} al pedir el video"
