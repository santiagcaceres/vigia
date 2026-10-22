"""Detectores de personas.

- OnnxYoloDetector: YOLO exportado a ONNX, corre en CPU con onnxruntime (sin GPU ni PyTorch).
  Es el detector para producción.
- MotionDetector: sustracción de fondo. Solo para pruebas y demos: no distingue personas.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .geometry import BBox

log = logging.getLogger("vigia.detectors")


@dataclass
class Detection:
    bbox: BBox  # píxeles: x1, y1, x2, y2
    label: str = "persona"
    conf: float = 1.0


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]: ...


class MotionDetector:
    """Movimiento por sustracción de fondo (MOG2). Aprende el fondo en los primeros frames.

    Necesita arrancar con la escena vacía: si hay alguien cuando aprende el fondo, queda una
    "sombra" que tarda minutos en desaparecer. Por eso es solo para pruebas; en producción
    se usa YOLO, que detecta personas en cada cuadro sin modelo de fondo.
    """

    def __init__(self, min_area_ratio: float = 0.002, warmup_frames: int = 10,
                 learning_rate: float = 0.001, var_threshold: float = 25.0, max_width: int = 640):
        self.min_area_ratio = min_area_ratio
        self.warmup_frames = warmup_frames
        self.learning_rate = learning_rate
        self.max_width = max_width
        self._bg = cv2.createBackgroundSubtractorMOG2(history=60, varThreshold=var_threshold, detectShadows=False)
        self._seen = 0
        self._open = np.ones((3, 3), np.uint8)
        self._close = np.ones((15, 15), np.uint8)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        scale = 1.0
        small = frame
        if w > self.max_width:
            scale = self.max_width / float(w)
            small = cv2.resize(frame, (self.max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (5, 5), 0)
        if self._seen < self.warmup_frames:
            self._bg.apply(small, learningRate=-1)
            self._seen += 1
            return []
        mask = self._bg.apply(small, learningRate=self.learning_rate)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._close)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sh, sw = small.shape[:2]
        min_area = self.min_area_ratio * sw * sh
        inv = 1.0 / scale
        out: list[Detection] = []
        for c in contours:
            if cv2.contourArea(c) < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            out.append(Detection((x * inv, y * inv, (x + bw) * inv, (y + bh) * inv), "persona", 0.5))
        return out


# Clases COCO del modelo: 0 persona; 2 auto, 3 moto, 5 ómnibus, 7 camión → todas "vehiculo".
PERSON_CLASSES = {0: "persona"}
VEHICLE_CLASSES = {2: "vehiculo", 3: "vehiculo", 5: "vehiculo", 7: "vehiculo"}


class OnnxYoloDetector:
    """Personas (y opcionalmente vehículos) con YOLO (v8/11) exportado a ONNX.
    Salida esperada: (1, 4+num_clases, N). `classes`: índice COCO → etiqueta."""

    def __init__(self, model_path: str | Path, conf: float = 0.35, iou: float = 0.5,
                 person_class: int = 0, threads: int = 2, classes: dict[int, str] | None = None):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model_path), opts, providers=_providers(ort))
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape  # [1, 3, H, W]
        self.size = (int(shape[3]) if isinstance(shape[3], int) else 640,
                     int(shape[2]) if isinstance(shape[2], int) else 640)
        self.conf = conf
        self.iou = iou
        self.person_class = person_class
        self.classes = dict(classes) if classes else {person_class: "persona"}
        self._idx = np.array(sorted(self.classes))
        self._labels = [self.classes[i] for i in self._idx]

    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        tw, th = self.size
        r = min(tw / w, th / h)
        nw, nh = int(round(w * r)), int(round(h * r))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((th, tw, 3), 114, np.uint8)
        px, py = (tw - nw) // 2, (th - nh) // 2
        canvas[py:py + nh, px:px + nw] = resized
        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0

        out = self.session.run(None, {self.input_name: blob})[0][0]  # (4+C, N)
        if out.shape[0] < out.shape[1]:
            out = out.T  # (N, 4+C)
        cls_scores = out[:, 4 + self._idx]  # (N, clases elegidas)
        best = cls_scores.argmax(axis=1)
        scores = cls_scores[np.arange(len(best)), best]
        keep = scores >= self.conf
        if not np.any(keep):
            return []
        boxes, scores, best = out[keep, :4], scores[keep], best[keep]
        cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = (cx - bw / 2 - px) / r
        y1 = (cy - bh / 2 - py) / r
        x2 = (cx + bw / 2 - px) / r
        y2 = (cy + bh / 2 - py) / r
        dets: list[Detection] = []
        for k, label in enumerate(self._labels):  # NMS por etiqueta
            sel = np.flatnonzero(best == k)
            if not len(sel):
                continue
            rects = [[float(x1[i]), float(y1[i]), float(x2[i] - x1[i]), float(y2[i] - y1[i])] for i in sel]
            idx = cv2.dnn.NMSBoxes(rects, scores[sel].astype(float).tolist(), self.conf, self.iou)
            for j in np.array(idx).flatten():
                i = sel[j]
                dets.append(Detection((max(0.0, float(x1[i])), max(0.0, float(y1[i])),
                                       min(float(w), float(x2[i])), min(float(h), float(y2[i]))), label, float(scores[i])))
        return dets


def _providers(ort) -> list[str]:
    available = ort.get_available_providers()
    preferred = ["CUDAExecutionProvider", "OpenVINOExecutionProvider", "CPUExecutionProvider"]
    return [p for p in preferred if p in available] or available


def build_detector(name: str, model_path: str | Path | None = None, vehicles: bool = False) -> Detector:
    """`vehicles=True`: además de personas detecta autos, motos, ómnibus y camiones (estaciones de servicio)."""
    if name == "motion":
        return MotionDetector()
    if name == "yolo":
        if not model_path or not Path(model_path).exists():
            raise RuntimeError(f"No se encontró el modelo ONNX de personas: {model_path}")
        return OnnxYoloDetector(model_path, classes={**PERSON_CLASSES, **(VEHICLE_CLASSES if vehicles else {})})
    raise ValueError(f"Detector desconocido: {name!r} (usar 'yolo' o 'motion')")
