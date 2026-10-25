"""Filtro de movimiento: no pasar por el detector los cuadros donde no cambió nada.

Antes de gastar la placa de video en buscar personas, se compara el cuadro (achicado a 96×54 y en
gris, casi gratis) con el de hace un segundo. Si la escena está quieta **y no hay nadie siguiéndose**,
el cuadro se saltea. Apenas algo se mueve, se vuelve a analizar a los fps de siempre, así que no se
pierde a nadie que entra. Igual se analiza un cuadro cada `every` segundos por las dudas (alguien que el
detector no vio al entrar y quedó quieto).

Lo que NO ahorra: el video igual llega y se decodifica (internet y procesador). Para eso está el
horario del local (core/schedule.py).
"""

from __future__ import annotations

import cv2
import numpy as np

SIZE = (96, 54)


class MotionGate:
    def __init__(self, pixel_delta: int = 18, min_fraction: float = 0.0015, every: float = 2.0,
                 ref_seconds: float = 1.0):
        self.pixel_delta = pixel_delta  # cuánto tiene que cambiar un punto (0-255) para contar
        self.min_fraction = min_fraction  # qué parte de la imagen tiene que cambiar (0,15 % ≈ 8 puntos)
        self.every = every
        self.ref_seconds = ref_seconds
        self._ref: np.ndarray | None = None
        self._ref_ts = -1e9
        self._last_full = -1e9
        self.checked = 0
        self.skipped = 0

    def _small(self, frame: np.ndarray) -> np.ndarray:
        small = cv2.resize(frame, SIZE, interpolation=cv2.INTER_AREA)
        if small.ndim == 3:
            small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(small, (3, 3), 0)  # saca el ruido de la imagen nocturna

    def moved(self, frame: np.ndarray, ts: float) -> bool:
        small = self._small(frame)
        if self._ref is None:
            self._ref, self._ref_ts = small, ts
            return True
        diff = cv2.absdiff(small, self._ref)
        frac = float(np.count_nonzero(diff > self.pixel_delta)) / diff.size
        if ts - self._ref_ts >= self.ref_seconds or frac >= self.min_fraction:
            self._ref, self._ref_ts = small, ts  # se compara contra lo de hace ~1 s (movimientos lentos)
        return frac >= self.min_fraction

    def should_analyze(self, frame: np.ndarray, ts: float, busy: bool) -> bool:
        """`busy`: hay personas siguiéndose o un clip abierto; ahí se analiza siempre."""
        self.checked += 1
        moved = self.moved(frame, ts)
        if busy or moved or ts - self._last_full >= self.every:
            self._last_full = ts
            return True
        self.skipped += 1
        return False

    def take_saved_fraction(self) -> float | None:
        """Parte de los cuadros salteados desde la última consulta (para mostrar el ahorro)."""
        if not self.checked:
            return None
        frac = self.skipped / self.checked
        self.checked = self.skipped = 0
        return frac
