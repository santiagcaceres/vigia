"""Nombre y logo de la marca en la app de escritorio: «Analítica Crítica» (antes se llamaba Vigía).

Los archivos de `recursos/` los genera `python -m tools.logo` a partir de la imagen original. Tk no escala bien
las imágenes, así que hay un PNG por tamaño. En el .exe (PyInstaller) la carpeta viaja adentro.
Por dentro el código sigue usando «vigia» (paquetes, variables VIGIA_*, registros): solo cambia lo que se ve.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

log = logging.getLogger("vigia.marca")

NOMBRE = "Analítica Crítica"
LEMA = "Analítica de video para comercios"
APP_ID = "AnaliticaCritica.App"  # Windows agrupa la barra de tareas por esto (y usa nuestro ícono, no el de Python)


def recursos() -> Path:
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass and (Path(meipass) / "agent" / "recursos").is_dir():
        return Path(meipass) / "agent" / "recursos"
    return Path(__file__).resolve().parent / "recursos"


def set_app_id() -> None:
    """Antes de abrir la ventana: que Windows muestre nuestro ícono en la barra de tareas."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
    except Exception as exc:  # noqa: BLE001 - es cosmético
        log.debug("No se pudo fijar el AppUserModelID: %s", exc)


def apply_icon(root) -> None:
    """Ícono de la ventana y de todas las que se abran después (diálogos): el .ico en Windows, un PNG en el resto."""
    ico, png = recursos() / "icono.ico", recursos() / "icono-64.png"
    try:
        if sys.platform == "win32" and ico.exists():
            root.iconbitmap(default=str(ico))
        elif png.exists():
            import tkinter as tk

            img = tk.PhotoImage(master=root, file=str(png))
            root.iconphoto(True, img)
            root._icono_marca = img  # que no lo borre el recolector de basura
    except Exception as exc:  # noqa: BLE001 - es cosmético
        log.debug("No se pudo poner el ícono: %s", exc)


def logo(root, size: int):
    """El logo (`logo-{size}.png`) como imagen de Tk, o None si no está. Guardar la referencia mientras se use."""
    path = recursos() / f"logo-{size}.png"
    if not path.exists():
        return None
    try:
        import tkinter as tk

        return tk.PhotoImage(master=root, file=str(path))
    except Exception as exc:  # noqa: BLE001
        log.debug("No se pudo cargar el logo: %s", exc)
        return None
