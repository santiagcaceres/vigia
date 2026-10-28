"""Máquina de análisis (servidor en la nube, sin pantalla): quién soy y cómo me doy de alta.

Cada máquina arranca con la misma configuración (servidor + clave de máquinas, ver
deploy/maquina-analisis/). La primera vez se da de alta sola (`POST /api/agent/enroll`) y guarda su
token en la carpeta de datos; si después el servidor la rechaza (la borraron), se vuelve a dar de alta.

Máquinas Spot (AWS y Google las venden mucho más baratas, pero las pueden reclamar): `Interruptions` mira cada
pocos segundos si la nube avisó que se la lleva (AWS avisa 2 minutos antes; Google, 30 s). Entonces la máquina
le avisa al servidor (`POST /api/agent/leaving`), que pasa sus cámaras a las otras en el momento, y suelta las
transmisiones (Hik-Connect da pocas por grabador: la máquina que las recibe tiene que poder abrirlas).
"""

from __future__ import annotations

import logging
import os
import socket
import time
from pathlib import Path

import httpx

log = logging.getLogger("vigia.maquina")
METADATA = "http://169.254.169.254"  # AWS (IMDSv2) y Google Cloud responden en la misma dirección


def instance_id() -> str | None:
    """Id de la instancia en la nube (para que el encendedor sepa cuál apagar). En AWS, el id de la instancia; en
    Google Cloud, su nombre; en otras nubes, None (se usa el nombre de la máquina)."""
    if os.getenv("VIGIA_INSTANCE_ID"):
        return os.environ["VIGIA_INSTANCE_ID"]
    try:  # AWS (IMDSv2). Google Cloud contesta 405 con una página HTML: no es un token
        t = httpx.put(f"{METADATA}/latest/api/token", timeout=1.0,
                      headers={"X-aws-ec2-metadata-token-ttl-seconds": "300"})
        if t.status_code == 200 and t.text.isascii():
            r = httpx.get(f"{METADATA}/latest/meta-data/instance-id", timeout=1.0,
                          headers={"X-aws-ec2-metadata-token": t.text})
            if r.status_code == 200 and r.text.startswith("i-"):
                return r.text.strip()
    except (httpx.HTTPError, UnicodeError):
        pass
    try:  # Google Cloud: el nombre de la instancia
        r = httpx.get(f"{METADATA}/computeMetadata/v1/instance/name", timeout=1.0,
                      headers={"Metadata-Flavor": "Google"})
        if r.status_code == 200 and r.text.strip() and r.text.isascii():
            return r.text.strip()
    except (httpx.HTTPError, UnicodeError):
        pass
    return None


class Interruptions:
    """¿La nube avisó que va a sacar esta máquina? `check()` devuelve el motivo, o None.
    - AWS: `spot/instance-action` (Spot reclamada) y `autoscaling/target-lifecycle-state` = Terminated (el grupo
      de Auto Scaling la va a terminar: la vació el encendedor o la reemplaza por otra).
    - Google Cloud: `instance/preempted` = TRUE (Spot reclamada).
    En otra nube o en una PC, `cloud` queda en None y no se mira nada."""

    def __init__(self, timeout: float = 1.0):
        self.timeout = timeout
        self._token: str | None = None
        self._token_at = 0.0
        self.cloud = self._detect()

    def _aws_token(self, force: bool = False) -> str | None:
        if self._token and not force and time.monotonic() - self._token_at < 240:
            return self._token
        t = httpx.put(f"{METADATA}/latest/api/token", timeout=self.timeout,
                      headers={"X-aws-ec2-metadata-token-ttl-seconds": "300"})
        if t.status_code == 200 and t.text.isascii() and t.text.strip():
            self._token, self._token_at = t.text.strip(), time.monotonic()
            return self._token
        return None

    def _detect(self) -> str | None:
        try:
            if self._aws_token(force=True):
                return "aws"
        except (httpx.HTTPError, UnicodeError):
            pass
        try:
            r = httpx.get(f"{METADATA}/computeMetadata/v1/instance/id", timeout=self.timeout,
                          headers={"Metadata-Flavor": "Google"})
            if r.status_code == 200 and r.headers.get("metadata-flavor", "").lower() == "google":
                return "gcp"
        except (httpx.HTTPError, UnicodeError):
            pass
        return None

    def check(self) -> str | None:
        try:
            if self.cloud == "aws":
                token = self._aws_token()
                if not token:
                    return None
                h = {"X-aws-ec2-metadata-token": token}
                r = httpx.get(f"{METADATA}/latest/meta-data/spot/instance-action", headers=h, timeout=self.timeout)
                if r.status_code == 200 and r.text.strip():
                    try:
                        d = r.json()
                    except ValueError:
                        d = {}
                    what = {"terminate": "terminar", "stop": "detener", "hibernate": "hibernar"}.get(
                        str(d.get("action")), "sacar")
                    return f"AWS va a {what} esta máquina Spot" + (f" ({d['time']})" if d.get("time") else "")
                r = httpx.get(f"{METADATA}/latest/meta-data/autoscaling/target-lifecycle-state", headers=h,
                              timeout=self.timeout)
                if r.status_code == 200 and r.text.strip() == "Terminated":
                    return "El grupo de Auto Scaling de AWS va a terminar esta máquina"
            elif self.cloud == "gcp":
                r = httpx.get(f"{METADATA}/computeMetadata/v1/instance/preempted", timeout=self.timeout,
                              headers={"Metadata-Flavor": "Google"})
                if r.status_code == 200 and r.text.strip().upper() == "TRUE":
                    return "Google Cloud va a sacar esta máquina Spot"
        except (httpx.HTTPError, UnicodeError):
            pass
        return None


def machine_name(instance: str | None, proceso: str | None) -> str:
    base = os.getenv("VIGIA_PC_NAME") or f"maquina-{instance or socket.gethostname()}"
    return f"{base}-{proceso}" if proceso else base


def enroll(server: str, key: str, name: str, instance: str | None, capacity: float | None,
           verify: bool | str = True) -> str:
    """Se da de alta y devuelve el token. ValueError con el motivo si el servidor la rechaza."""
    try:
        r = httpx.post(server.rstrip("/") + "/api/agent/enroll", timeout=30.0, verify=verify,
                       json={"key": key, "name": name, "instance_id": instance, "capacity": capacity})
    except httpx.HTTPError as exc:
        raise ValueError(f"No se pudo conectar con el servidor ({type(exc).__name__})") from exc
    if r.status_code != 200:
        try:
            msg = r.json().get("detail") or r.text
        except ValueError:
            msg = r.text
        raise ValueError(str(msg)[:200])
    return r.json()["token"]


def token_file(data_dir: Path, instance: str | None = None) -> Path:
    """Un archivo por instancia: si se clona la máquina (imagen para hacer más copias), la copia no hereda
    el token de la original y se da de alta como máquina nueva."""
    return Path(data_dir) / (f"token-maquina-{instance}" if instance else "token-maquina")


def saved_token(data_dir: Path, instance: str | None = None) -> str | None:
    try:
        return token_file(data_dir, instance).read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def save_token(data_dir: Path, token: str, instance: str | None = None) -> None:
    p = token_file(data_dir, instance)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(token, encoding="utf-8")
    try:
        p.chmod(0o600)
    except OSError:
        pass
