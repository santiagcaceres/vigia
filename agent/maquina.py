"""Máquina de análisis (servidor en la nube, sin pantalla): quién soy y cómo me doy de alta.

Cada máquina arranca con la misma configuración (servidor + clave de máquinas, ver
deploy/maquina-analisis/). La primera vez se da de alta sola (`POST /api/agent/enroll`) y guarda su
token en la carpeta de datos; si después el servidor la rechaza (la borraron), se vuelve a dar de alta.
"""

from __future__ import annotations

import logging
import os
import socket
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
