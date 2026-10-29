"""Postura de cada persona: 17 puntos del cuerpo (hombros, codos, muñecas, caderas…) para ver los gestos de
ocultamiento (core/gestos.py).

- Se usa YOLO11n-pose (Ultralytics, AGPL-3.0, la misma familia que el detector de personas) exportado a ONNX con
  entrada de 256x256, **sobre el recorte de cada persona** que ya encontró el detector («de arriba hacia abajo»).
  Sobre la imagen entera el modelo de pose encuentra muchas menos personas que el detector (medido el 29 oct en 128
  imágenes de COCO: AP50 0,53 contra 0,76; se pierde a la gente chica), y en el recorte cada persona llega al modelo
  agrandada. Cuesta ~15 ms por persona con 1 hilo de procesador.
- Solo lo importa el agente (usa OpenCV y onnxruntime): el servidor no.
- Orden COCO de los puntos: 0 nariz, 1-2 ojos, 3-4 orejas, 5-6 hombros, 7-8 codos, 9-10 muñecas, 11-12 caderas,
  13-14 rodillas, 15-16 tobillos (izquierdo, derecho: los de la persona, no los de la imagen).
- La «confianza» de cada punto de YOLO-pose NO es visibilidad: se entrena como «punto anotado», y COCO anota también
  los puntos tapados que se pueden adivinar. Una muñeca metida en el bolsillo puede salir con confianza alta, ubicada
  en el bolsillo. Por eso los gestos miran sobre todo dónde está la muñeca, y la caída de confianza es una pista más.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import cv2
import numpy as np

from .detectors import OnnxSession, SessionPool
from .geometry import BBox, iou

log = logging.getLogger("vigia.pose")

KEYPOINTS = ("nariz", "ojo_izq", "ojo_der", "oreja_izq", "oreja_der", "hombro_izq", "hombro_der", "codo_izq",
             "codo_der", "muñeca_izq", "muñeca_der", "cadera_izq", "cadera_der", "rodilla_izq", "rodilla_der",
             "tobillo_izq", "tobillo_der")
# huesos que se dibujan (de la cabeza para abajo no hace falta la cara)
SKELETON = ((5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14),
            (14, 16))
MIN_PERSON_PX = 90  # personas más chicas que esto (alto en el cuadro) no se miran: muñecas de pocos píxeles
MAX_PER_FRAME = 8  # como mucho tantas personas por cuadro (las más grandes primero)


class PoseEstimator:
    """Puntos del cuerpo de una persona a partir de su recuadro. `estimate` devuelve un arreglo (17, 3) con x, y en
    píxeles del cuadro y la confianza de cada punto, o None si el modelo no encuentra a la persona en el recorte."""

    name = "YOLO11n-pose"

    def __init__(self, model_path: str | Path | None = None, runner=None, pad: float = 0.12, min_conf: float = 0.25,
                 threads: int = 1):
        self.runner = runner or OnnxSession(model_path, threads=threads, max_batch=1)
        self.device, self.on_gpu, self.size = self.runner.device, self.runner.on_gpu, self.runner.size
        self.pad = pad
        self.min_conf = min_conf

    def estimate(self, frame: np.ndarray, bbox: BBox) -> np.ndarray | None:
        fh, fw = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        if bw < 4 or bh < 8:
            return None
        # recorte con margen (los brazos estirados a veces quedan fuera del recuadro del detector)
        px, py = self.pad * bh, self.pad * bh * 0.6
        cx1, cy1 = int(max(0, x1 - px)), int(max(0, y1 - py))
        cx2, cy2 = int(min(fw, x2 + px)), int(min(fh, y2 + py))
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None
        tw, th = self.size
        ch, cw = crop.shape[:2]
        r = min(tw / cw, th / ch)
        nw, nh = max(1, int(round(cw * r))), max(1, int(round(ch * r)))
        canvas = np.full((th, tw, 3), 114, np.uint8)
        ox, oy = (tw - nw) // 2, (th - nh) // 2
        canvas[oy:oy + nh, ox:ox + nw] = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)
        out = self.runner.run(canvas)  # (56, N): cx, cy, w, h, puntaje, 17 x (x, y, conf)
        if out.shape[0] > out.shape[1]:
            out = out.T
        scores = out[4]
        cand = np.flatnonzero(scores >= self.min_conf)
        if not len(cand):
            return None
        # la persona del recuadro (en el recorte puede haber otras): la que más se superpone, pesada por el puntaje
        want = ((x1 - cx1) * r + ox, (y1 - cy1) * r + oy, (x2 - cx1) * r + ox, (y2 - cy1) * r + oy)
        best, best_val = -1, 0.0
        for i in cand[np.argsort(-scores[cand])[:20]]:
            b = (out[0, i] - out[2, i] / 2, out[1, i] - out[3, i] / 2, out[0, i] + out[2, i] / 2,
                 out[1, i] + out[3, i] / 2)
            val = iou(b, want) * float(scores[i])
            if val > best_val:
                best, best_val = int(i), val
        if best < 0 or best_val < 0.1:
            return None
        kp = out[5:, best].reshape(17, 3).astype(np.float32)
        kp[:, 0] = (kp[:, 0] - ox) / r + cx1
        kp[:, 1] = (kp[:, 1] - oy) / r + cy1
        return kp


def pick(tracks: list, limit: int = MAX_PER_FRAME, min_px: float = MIN_PERSON_PX) -> list:
    """Las personas a las que se les mira la postura en este cuadro: visibles, de alto suficiente, las más grandes
    primero y como mucho `limit`."""
    cand = [t for t in tracks if getattr(t, "visible", True) and getattr(t, "label", "persona") == "persona"
            and t.bbox[3] - t.bbox[1] >= min_px]
    cand.sort(key=lambda t: -(t.bbox[3] - t.bbox[1]))
    return cand[:limit]


def look(estimator: PoseEstimator | None, frame: np.ndarray, tracks: list, min_px: float = MIN_PERSON_PX) -> int:
    """Postura de las personas elegidas (`pick`) en `t.kpts`; a las demás, None. Devuelve a cuántas se les miró."""
    chosen = {t.id for t in pick(tracks, min_px=min_px)} if estimator is not None else set()
    n = 0
    for t in tracks:
        t.kpts = estimator.estimate(frame, t.bbox) if t.id in chosen else None
        n += t.kpts is not None
    return n


def detect_with_bags(detector, frame: np.ndarray, bags: bool) -> tuple[list, list]:
    """Personas (y lo demás que el detector busque) y, en la misma pasada del modelo, bolsos, carteras y mochilas
    para el gesto «mano en la bolsa». Un detector sin `detect_also` (movimiento) no da bolsos."""
    if bags and hasattr(detector, "detect_also"):
        return detector.detect_also(frame, BAG_CLASSES, 0.3)
    return detector.detect(frame), []


BAG_CLASSES = {24: "mochila", 26: "cartera", 28: "valija"}  # COCO

_lock = threading.Lock()
_estimators: dict[str, PoseEstimator] = {}


def default_model() -> Path | None:
    """models/postura.onnx: dentro del .exe (PyInstaller) o en el repo."""
    import sys

    for base in (getattr(sys, "_MEIPASS", None), Path(__file__).resolve().parent.parent):
        if base:
            p = Path(base) / "models" / "postura.onnx"
            if p.exists():
                return p
    return None


def build_pose(model_path: str | Path | None = None) -> PoseEstimator | None:
    """Un estimador compartido por todas las cámaras del proceso (como el detector). Sin el archivo del modelo
    devuelve None (y los gestos quedan apagados en esa máquina)."""
    path = Path(model_path) if model_path else default_model()
    if path is None or not path.exists():
        log.warning("No está el modelo de postura (models/postura.onnx): los gestos quedan apagados")
        return None
    key = str(path)
    with _lock:
        if key not in _estimators:
            hilos = int(os.getenv("VIGIA_HILOS") or 0)
            runner = SessionPool(path, hilos) if hilos > 1 else OnnxSession(path, threads=2, max_batch=1)
            est = PoseEstimator(runner=runner)
            _estimators[key] = est
            log.info("Postura de las personas (%s) en %s", est.name, est.device)
        return _estimators[key]


def draw(img: np.ndarray, kp: np.ndarray | None, color=(0, 255, 255), scale: float = 1.0, min_conf: float = 0.3,
         hot_wrists: tuple[int, ...] = ()) -> None:
    """Dibuja el esqueleto (de los hombros para abajo) sobre `img`. `scale`: si la imagen está achicada respecto del
    cuadro original. Las muñecas en `hot_wrists` (9 izquierda, 10 derecha) van en rojo."""
    if kp is None:
        return
    pts = [(int(x * scale), int(y * scale), c) for x, y, c in kp]
    thick = max(1, int(round(img.shape[1] / 640)))
    for a, b in SKELETON:
        if pts[a][2] >= min_conf and pts[b][2] >= min_conf:
            cv2.line(img, pts[a][:2], pts[b][:2], color, thick, cv2.LINE_AA)
    for i in (5, 6, 7, 8, 9, 10, 11, 12):
        x, y, c = pts[i]
        if c < min_conf:
            continue
        hot = i in hot_wrists
        cv2.circle(img, (x, y), (4 if i in (9, 10) else 3) * thick, (0, 0, 255) if hot else color, -1, cv2.LINE_AA)
