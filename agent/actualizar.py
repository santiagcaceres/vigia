"""Actualización de la app de escritorio con un botón.

1. El servidor dice cuál es la última versión publicada (`GET /api/app/update`) y, cuando se pide,
   da un link de descarga que vence en minutos (si el repositorio fuera privado, el token de GitHub
   queda en el servidor).
2. Se baja el .zip, se saca `VigiaAgente.exe` como `VigiaAgente.nuevo.exe` y se verifica.
3. Windows no deja borrar un .exe que está corriendo pero sí renombrarlo: el actual pasa a
   `VigiaAgente.viejo.exe`, el nuevo toma su nombre y se abre. El viejo se borra en el próximo inicio.
`agente.env` (sesión, token de la PC) no se toca: el programa nuevo arranca ya con la sesión.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
import zipfile
from collections.abc import Callable
from pathlib import Path

import httpx

from .version import VERSION

log = logging.getLogger("vigia.actualizar")
EXE_NAME = "VigiaAgente.exe"


def parse_version(v: str | None) -> tuple[int, ...] | None:
    """«1.8.0» → (1, 8, 0). Lo que no tenga esa forma («dev», «dev-12») es una versión de desarrollo."""
    m = re.fullmatch(r"v?(\d+(?:\.\d+){0,2})", (v or "").strip())
    return tuple(int(n) for n in m.group(1).split(".")) if m else None


def is_newer(remote: str | None, local: str = VERSION) -> bool:
    """¿La versión publicada es más nueva que la que corre? Una versión de desarrollo siempre ofrece actualizar."""
    r, l_ = parse_version(remote), parse_version(local)
    if r is None:
        return False
    return l_ is None or r > l_


def can_self_update() -> bool:
    """Solo el .exe se reemplaza solo; corriendo con Python se actualiza con git pull."""
    return bool(getattr(sys, "frozen", False)) and sys.platform == "win32"


def current_exe() -> Path:
    return Path(sys.executable).resolve()


def cleanup_old() -> None:
    """Borra el .exe viejo que dejó una actualización anterior (si ya no está corriendo)."""
    if not getattr(sys, "frozen", False):
        return
    old = current_exe().with_name("VigiaAgente.viejo.exe")
    for _ in range(30):  # el proceso anterior puede tardar unos segundos en cerrarse
        if not old.exists():
            return
        try:
            old.unlink()
            log.info("Actualización terminada: versión %s", VERSION)
            return
        except OSError:
            time.sleep(2)


def download(url: str, dest: Path, progress: Callable[[int, int], None] | None = None,
             verify: bool | str = True) -> Path:
    """Baja `url` a `dest` (un .zip) e informa (bajado, total)."""
    tmp = dest.with_suffix(".parcial")
    with httpx.stream("GET", url, timeout=httpx.Timeout(30, read=120), follow_redirects=True, verify=verify) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_bytes(1 << 16):
                f.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
    os.replace(tmp, dest)
    return dest


def extract_exe(zip_path: Path, dest: Path) -> Path:
    """Saca VigiaAgente.exe del .zip (esté en la raíz o en una carpeta) y lo guarda como `dest`."""
    with zipfile.ZipFile(zip_path) as z:
        name = next((n for n in z.namelist() if n.replace("\\", "/").split("/")[-1].lower() == EXE_NAME.lower()), None)
        if not name:
            raise ValueError("El .zip descargado no trae VigiaAgente.exe")
        with z.open(name) as src, open(dest, "wb") as out:
            while chunk := src.read(1 << 20):
                out.write(chunk)
    if dest.stat().st_size < 1_000_000:
        raise ValueError("El VigiaAgente.exe descargado está incompleto")
    return dest


def swap_exe(new_exe: Path, exe: Path) -> None:
    """Pone el .exe nuevo en el lugar del actual (que está corriendo: se renombra, no se borra)."""
    old = exe.with_name("VigiaAgente.viejo.exe")
    try:
        if old.exists():
            old.unlink()
    except OSError as exc:
        raise RuntimeError("Quedó abierta una versión anterior de Vigía; cerrala y probá de nuevo") from exc
    os.replace(exe, old)
    try:
        os.replace(new_exe, exe)
    except OSError:
        os.replace(old, exe)  # volver atrás: que el programa siga existiendo
        raise


def install(url: str, progress: Callable[[int, int], None] | None = None, verify: bool | str = True) -> Path:
    """Baja e instala la versión nueva junto al .exe actual. Después hay que reabrir el programa."""
    exe = current_exe()
    folder = exe.parent
    zip_path = folder / "VigiaAgente-actualizacion.zip"
    new_exe = folder / "VigiaAgente.nuevo.exe"
    try:
        download(url, zip_path, progress, verify)
        extract_exe(zip_path, new_exe)
        swap_exe(new_exe, exe)
    finally:
        for p in (zip_path, new_exe, zip_path.with_suffix(".parcial")):
            try:
                p.unlink()
            except OSError:
                pass
    log.info("Versión nueva instalada en %s", exe)
    return exe
