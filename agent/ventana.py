"""Ventana de la PC central: mosaico con todas las cámaras y los recuadros de la IA en tiempo real.

Es la vista "de escritorio" para la oficina: muestra cada cámara en vivo (todos los cuadros que
llegan del grabador, no 1 por segundo como la web) con las personas recuadradas, su número,
los segundos en escena y la zona. Las cámaras en modo alarmas aparecen como "Escuchando
alarmas" y solo muestran video mientras alguien las mira desde el panel.

Se activa con `--ventana` (en el .exe viene activada; `--sin-ventana` la apaga, por ejemplo en
un servidor sin pantalla). Cerrar la ventana no detiene el agente.
"""

from __future__ import annotations

import logging
import math

import cv2
import numpy as np

log = logging.getLogger("vigia.ventana")

TITULO = "Vigia - PC central"
TILE_W, TILE_H = 640, 360


def tile(frame: np.ndarray | None, name: str, state: str) -> np.ndarray:
    """Una celda del mosaico: el cuadro (o un fondo oscuro con el estado) y el nombre arriba."""
    if frame is None:
        img = np.full((TILE_H, TILE_W, 3), (30, 30, 30), np.uint8)
        cv2.putText(img, state, (16, TILE_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (170, 170, 170), 1, cv2.LINE_AA)
    else:
        h, w = frame.shape[:2]
        scale = min(TILE_W / w, TILE_H / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        img = np.zeros((TILE_H, TILE_W, 3), np.uint8)
        y, x = (TILE_H - nh) // 2, (TILE_W - nw) // 2
        img[y:y + nh, x:x + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    cv2.rectangle(img, (0, 0), (TILE_W, 22), (0, 0, 0), -1)
    cv2.putText(img, name, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def mosaic(tiles: list[tuple[str, np.ndarray | None, str]]) -> np.ndarray:
    """Arma la grilla: 1 cámara = 1 celda grande; 2 = lado a lado; 3-4 = 2x2; 5-9 = 3x3..."""
    if not tiles:
        return tile(None, "Vigia", "Sin camaras asignadas a esta PC central")
    n = len(tiles)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    grid = np.zeros((rows * TILE_H, cols * TILE_W, 3), np.uint8)
    for i, (name, frame, state) in enumerate(tiles):
        r, c = divmod(i, cols)
        grid[r * TILE_H:(r + 1) * TILE_H, c * TILE_W:(c + 1) * TILE_W] = tile(frame, name, state)
    return grid


class Ventana:
    """Muestra el mosaico; `show` devuelve False cuando el usuario cerró la ventana."""

    def __init__(self):
        self.open = False
        self.failed = False

    def show(self, tiles: list[tuple[str, np.ndarray | None, str]], wait_ms: int = 40) -> bool:
        if self.failed:
            return False
        try:
            if not self.open:
                cv2.namedWindow(TITULO, cv2.WINDOW_NORMAL)
                self.open = True
            cv2.imshow(TITULO, mosaic(tiles))
            key = cv2.waitKey(wait_ms) & 0xFF
            if key == ord("q") or cv2.getWindowProperty(TITULO, cv2.WND_PROP_VISIBLE) < 1:
                self.close()
                return False
            return True
        except cv2.error as exc:  # OpenCV sin soporte de ventanas (headless) o sin pantalla
            self.failed = True
            log.warning("No se puede abrir la ventana (%s). El agente sigue sin ella.", str(exc).splitlines()[0])
            return False

    def close(self) -> None:
        if self.open:
            try:
                cv2.destroyWindow(TITULO)
            except cv2.error:
                pass
        self.open = False
