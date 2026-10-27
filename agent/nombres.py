"""«Nombres e imágenes» de la app: una imagen de cada cámara para reconocerla y ponerle nombre.

`grab_frame` abre el video de una cámara unos segundos (igual que «En vivo»: RTSP o Hik-Connect), se queda con un
cuadro y lo cierra. Se usa de a una cámara por vez: Hik-Connect da pocas transmisiones a la vez por grabador.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from .cameras import build_camera

GRAB_FRAMES = 12  # los primeros cuadros de un video a veces vienen grises (falta el cuadro completo)


def grab_frame(conn: dict, timeout: float = 25.0) -> np.ndarray:
    """Un cuadro del video en vivo de la cámara. RuntimeError con el motivo si no se pudo."""
    cam = build_camera(conn)
    url = cam.live_url()
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    try:
        if not cap.isOpened():
            raise RuntimeError(cam.open_failure(url) or "no se pudo abrir el video")
        frame, n, t0 = None, 0, time.monotonic()
        while n < GRAB_FRAMES and time.monotonic() - t0 < timeout:
            ok, f = cap.read()
            if ok and f is not None:
                frame, n = f, n + 1
            elif frame is not None:
                break  # el video se cortó: sirve el último cuadro bueno
        if frame is None:
            raise RuntimeError("el video abrió pero no llegó ninguna imagen")
        return frame
    finally:
        cap.release()
