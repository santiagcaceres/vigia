"""Píxeles no cuadrados: los grabadores graban «1080p lite» (Hikvision; «1080N» en Dahua) como 960x1080 con píxeles
el doble de anchos. OpenCV no lo corrige: las personas quedan flacas, el detector falla más y los gestos (que comparan
distancias horizontales con verticales) no andan. Lo usan las máquinas (video en vivo y tramos de alarmas) y
«Configurar…», para que lo que se prueba sea lo mismo que se analiza.
"""

from __future__ import annotations

import cv2
import numpy as np

# Muchos videos exportados y transmisiones no lo declaran (el de Góndolas ventana de Minas, 27 oct): se reconocen por
# el tamaño.
LITE_SIZES = {(960, 1080), (640, 720), (1280, 1440)}


def pixel_aspect(cap: cv2.VideoCapture) -> float | None:
    """Forma de los píxeles del video (ancho/alto) si no son cuadrados: lo que declara el archivo o, si no lo declara,
    2,0 en los tamaños «lite»."""
    try:
        num, den = cap.get(cv2.CAP_PROP_SAR_NUM), cap.get(cv2.CAP_PROP_SAR_DEN)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    except Exception:  # noqa: BLE001 - backends sin ese dato
        return None
    if num > 0 and den > 0 and abs(num / den - 1) > 0.05:
        return num / den
    if size in LITE_SIZES:
        return 2.0
    return None


def frame_aspect(frame: np.ndarray) -> float | None:
    """Lo mismo mirando solo el tamaño del cuadro (cuando no se tiene el archivo o la transmisión abiertos)."""
    h, w = frame.shape[:2]
    return 2.0 if (w, h) in LITE_SIZES else None


def square_pixels(frame: np.ndarray, sar: float, shrink: bool = False) -> np.ndarray:
    """El cuadro con píxeles cuadrados. Por defecto se estira (960x1080 con píxeles 2:1 → 1920x1080, como se ve en el
    grabador); con `shrink` se achica el otro lado (→ 960x540: igual de bien para analizar y cuesta menos)."""
    h, w = frame.shape[:2]
    if shrink:
        if sar > 1:
            return cv2.resize(frame, (w, max(1, int(round(h / sar)))), interpolation=cv2.INTER_AREA)
        return cv2.resize(frame, (max(1, int(round(w * sar))), h), interpolation=cv2.INTER_AREA)
    if sar > 1:
        return cv2.resize(frame, (int(round(w * sar)), h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(frame, (w, int(round(h / sar))), interpolation=cv2.INTER_LINEAR)
