"""Pantalla de carga del .exe (pedido del dueño, 3 nov: «cuando abro el programa demora un poco… un pop up de carga con
el logo hasta que abra, para no alterar a los usuarios»).

La dibuja PyInstaller (`Splash` en agente_pc.spec) apenas se hace doble clic, **antes** de que arranque Python: así se ve
también mientras el .exe se descomprime, que es lo que más tarda (unos 1900 archivos; con el antivirus, más). Mientras
tanto la barra avanza sola (hasta el 70 %). Después, desde acá, se cambia el texto y la barra en cada etapa
(`aviso`) y se cierra cuando la ventana ya se ve (`cerrar`). Fuera del .exe (con Python, en las máquinas de análisis,
en las pruebas) no hay pantalla de carga y todo esto no hace nada.

El mensaje viaja como «=NN|texto» (NN = avance de la barra, 0-100): el «=» separa nuestros mensajes de los nombres de
archivo que manda PyInstaller mientras descomprime (ver el .spec, `SPLASH_TCL`). Este módulo no importa nada pesado:
se usa al principio de todo, antes de cargar OpenCV y el detector.
"""

from __future__ import annotations

import logging
import time

log = logging.getLogger("vigia.arranque")

# la imagen (agent/recursos/splash.png, la genera `python -m tools.logo`) y lo que PyInstaller dibuja encima: el riel
# de la barra de avance (x0, y0, x1, y1) y el texto (abajo a la izquierda). El .spec y tools/logo.py las leen de acá.
SPLASH_SIZE = (560, 320)
SPLASH_BAR = (48, 252, 512, 258)
SPLASH_TEXT = (48, 290)
EXTRACT_SHARE = 70  # hasta qué parte de la barra llega mientras se descomprime el .exe (el resto, las etapas de acá)

_t0 = time.monotonic()  # cuando arrancó Python (lo de antes, descomprimir, no se puede medir desde acá)

try:  # solo existe dentro del .exe armado con pantalla de carga
    import pyi_splash  # type: ignore[import-not-found]
except ImportError:  # con Python, o un .exe sin pantalla de carga
    pyi_splash = None


def abierta() -> bool:
    try:
        return pyi_splash is not None and bool(pyi_splash.is_alive())
    except Exception:  # noqa: BLE001 - la pantalla de carga nunca puede trabar el programa
        return False


def aviso(texto: str, avance: int | None = None) -> None:
    """Cambia el texto (y, con `avance`, la barra) de la pantalla de carga. Una línea, sin saltos."""
    if not abierta():
        return
    texto = " ".join(str(texto).split())
    msg = f"={max(0, min(100, int(avance)))}|{texto}" if avance is not None else f"={texto}"
    try:
        pyi_splash.update_text(msg)
    except Exception:  # noqa: BLE001
        pass


def cerrar() -> None:
    """Cierra la pantalla de carga (si sigue abierta). Llamarla antes de cualquier mensaje o pantalla que espere al
    usuario: la de carga queda siempre encima y lo taparía."""
    if not abierta():
        return
    try:
        pyi_splash.close()
        log.info("Pantalla de carga cerrada: la ventana tardó %.1f s desde que arrancó Python", time.monotonic() - _t0)
    except Exception:  # noqa: BLE001
        pass
