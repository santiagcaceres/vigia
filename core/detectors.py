"""Detectores de personas.

- OnnxYoloDetector: YOLO exportado a ONNX con onnxruntime (sin PyTorch). Usa la placa de video si hay
  (CUDA/TensorRT en servidores NVIDIA, DirectML en Windows, con un solo modelo compartido por todas las
  cámaras) y si no, el procesador. En NVIDIA analiza los cuadros de varias cámaras juntos (lotes).
  Es el detector para producción.
- MotionDetector: sustracción de fondo. Solo para pruebas y demos: no distingue personas.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from concurrent.futures import Future
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


class OnnxSession:
    """Una sesión ONNX compartida por todas las cámaras que usan la placa de video.

    - `max_batch == 1` (app de Windows con DirectML, procesador): de a un cuadro, con un candado.
    - `max_batch > 1` (servidor con GPU NVIDIA): un hilo junta los cuadros que llegan de todas las cámaras
      (hasta `max_batch`, esperando a lo sumo `max_wait` segundos) y los analiza en una sola llamada. Usa el
      modelo de core/lotes.py: recibe la imagen tal cual sale de OpenCV y devuelve solo las clases que se usan.
    """

    def __init__(self, model_path: str | Path, threads: int = 2, max_batch: int | None = None,
                 max_wait: float = 0.008):
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls") and "CUDAExecutionProvider" in ort.get_available_providers():
            try:  # CUDA y cuDNN que vienen por pip: sin instalar el kit de NVIDIA en la máquina
                ort.preload_dlls()
            except Exception:  # noqa: BLE001
                pass
        providers = _providers(ort)
        nvidia = any(p in providers for p in ("TensorrtExecutionProvider", "CUDAExecutionProvider"))
        if max_batch is None:
            max_batch = int(os.getenv("VIGIA_LOTE") or (16 if nvidia else 1))
        self.max_batch = max(1, max_batch)
        self.max_wait = max_wait
        self.uint8 = False
        self.col_of = {c: 4 + c for c in range(80)}
        model: str | bytes = str(model_path)
        if self.max_batch > 1:
            try:
                from .lotes import make_batch_model

                bm = make_batch_model(model_path)
                model, self.uint8, self.col_of = bm.data, True, bm.col_of
            except Exception as exc:  # noqa: BLE001 - sin el paquete onnx o modelo raro: de a un cuadro
                log.warning("No se pudo preparar el modelo para lotes (%s): se analiza de a un cuadro", exc)
                self.max_batch = 1
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        if "DmlExecutionProvider" in providers:  # DirectML (GPU en Windows) pide esto
            opts.enable_mem_pattern = False
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        if "TensorrtExecutionProvider" in providers:
            providers = [_tensorrt(self.max_batch, "imagenes" if self.uint8 else None)
                         if p == "TensorrtExecutionProvider" else p for p in providers]
        try:
            self.session = ort.InferenceSession(model, opts, providers=providers)
        except Exception:  # noqa: BLE001 - placa sin soporte: seguir con el procesador
            self.session = ort.InferenceSession(model, opts, providers=["CPUExecutionProvider"])
        used = self.session.get_providers()
        self.device = ("GPU (" + used[0].replace("ExecutionProvider", "") + ")"
                       if used and used[0] != "CPUExecutionProvider" else "procesador")
        self.on_gpu = self.device != "procesador"
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape  # [1, 3, H, W] o, en el de lotes, [lote, H, W, 3]
        hw = (shape[1], shape[2]) if self.uint8 else (shape[2], shape[3])
        self.size = (int(hw[1]) if isinstance(hw[1], int) else 640, int(hw[0]) if isinstance(hw[0], int) else 640)
        self._lock = threading.Lock()
        self.batches = self.frames = 0  # para medir el tamaño promedio del lote
        if self.max_batch > 1:
            self._queue: queue.Queue = queue.Queue()
            threading.Thread(target=self._loop, name="detector-lotes", daemon=True).start()

    def run(self, canvas: np.ndarray) -> np.ndarray:
        """`canvas`: imagen BGR ya llevada al tamaño del modelo. Devuelve (4+clases, N) de ese cuadro."""
        if self.max_batch == 1:
            blob = canvas[None] if self.uint8 else (
                cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
            with self._lock:
                self.batches += 1
                self.frames += 1
                return self.session.run(None, {self.input_name: blob})[0][0]
        fut: Future = Future()
        self._queue.put((canvas, fut))
        return fut.result()

    def _loop(self) -> None:
        while True:
            items = [self._queue.get()]
            deadline = time.monotonic() + self.max_wait
            while len(items) < self.max_batch:
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                try:
                    items.append(self._queue.get(timeout=left))
                except queue.Empty:
                    break
            try:
                out = self.session.run(None, {self.input_name: np.stack([c for c, _ in items])})[0]
            except Exception as exc:  # noqa: BLE001 - que cada cámara reciba el error
                for _, fut in items:
                    fut.set_exception(exc)
                continue
            self.batches += 1
            self.frames += len(items)
            for i, (_, fut) in enumerate(items):
                fut.set_result(out[i])

    def take_stats(self) -> tuple[int, float]:
        """(cuadros analizados, tamaño promedio del lote) desde la última consulta."""
        frames, batches = self.frames, self.batches
        self.frames = self.batches = 0
        return frames, (frames / batches if batches else 0.0)


class OnnxYoloDetector:
    """Personas (y opcionalmente vehículos) con YOLO (v8/11) exportado a ONNX.
    Salida esperada: (1, 4+num_clases, N). `classes`: índice COCO → etiqueta. Varias instancias (por ejemplo,
    con y sin vehículos) pueden compartir la misma `OnnxSession`."""

    def __init__(self, model_path: str | Path | None = None, conf: float = 0.35, iou: float = 0.5,
                 person_class: int = 0, threads: int = 2, classes: dict[int, str] | None = None,
                 session: OnnxSession | None = None, max_batch: int | None = 1):
        self.runner = session or OnnxSession(model_path, threads, max_batch=max_batch)
        self.device, self.on_gpu, self.size = self.runner.device, self.runner.on_gpu, self.runner.size
        self.session = self.runner.session
        self.conf = conf
        self.iou = iou
        self.person_class = person_class
        self.classes = dict(classes) if classes else {person_class: "persona"}
        self._idx = np.array(sorted(self.classes))
        self._cols = np.array([self.runner.col_of[c] for c in self._idx])
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

        out = self.runner.run(canvas)  # (4+C, N)
        if out.shape[0] < out.shape[1]:
            out = out.T  # (N, 4+C)
        cls_scores = out[:, self._cols]  # (N, clases elegidas)
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
    preferred = ["CUDAExecutionProvider", "DmlExecutionProvider", "OpenVINOExecutionProvider", "CPUExecutionProvider"]
    if os.getenv("VIGIA_TENSORRT", "0") not in ("0", "", "no"):  # TensorRT: más rápido, si está instalado
        preferred.insert(0, "TensorrtExecutionProvider")
    if os.getenv("VIGIA_DISPOSITIVO") == "procesador":
        preferred = ["CPUExecutionProvider"]
    return [p for p in preferred if p in available] or available


def _tensorrt(max_batch: int, input_name: str | None):
    """TensorRT en media precisión, guardando el motor compilado (la primera vez tarda unos minutos)."""
    cache = os.getenv("VIGIA_TRT_CACHE") or str(Path.home() / ".cache" / "vigia-tensorrt")
    opts = {"trt_fp16_enable": True, "trt_engine_cache_enable": True, "trt_engine_cache_path": cache}
    if input_name and max_batch > 1:
        opts.update(trt_profile_min_shapes=f"{input_name}:1x640x640x3",
                    trt_profile_opt_shapes=f"{input_name}:{max_batch // 2 or 1}x640x640x3",
                    trt_profile_max_shapes=f"{input_name}:{max_batch}x640x640x3")
    return ("TensorrtExecutionProvider", opts)


_shared: dict[tuple, "OnnxYoloDetector"] = {}
_sessions: dict[str, OnnxSession] = {}
_shared_lock = threading.Lock()
_logged: set[tuple] = set()


def session_stats() -> tuple[int, float]:
    """Cuadros analizados en la placa de video y tamaño promedio del lote, desde la última consulta."""
    frames = batches = 0
    for sess in list(_sessions.values()):
        f, avg = sess.take_stats()
        frames += f
        batches += f / avg if avg else 0
    return frames, (frames / batches if batches else 0.0)


def build_detector(name: str, model_path: str | Path | None = None, vehicles: bool = False) -> Detector:
    """`vehicles=True`: además de personas detecta autos, motos, ómnibus y camiones (estaciones de servicio)."""
    if name == "motion":
        return MotionDetector()
    if name == "yolo":
        if not model_path or not Path(model_path).exists():
            raise RuntimeError(f"No se encontró el modelo ONNX de personas: {model_path}")
        key = (str(model_path), vehicles)
        classes = {**PERSON_CLASSES, **(VEHICLE_CLASSES if vehicles else {})}
        with _shared_lock:
            if key in _shared:  # en la GPU, un solo modelo atiende todas las cámaras
                return _shared[key]
            sess = _sessions.get(str(model_path))
            if sess is None:
                sess = OnnxSession(model_path, max_batch=None)
            det = OnnxYoloDetector(classes=classes, session=sess)
            if det.on_gpu or sess.max_batch > 1:  # con y sin vehículos usan la misma sesión (y los mismos lotes)
                _sessions[str(model_path)] = sess
                _shared[key] = det
            if key not in _logged:
                _logged.add(key)
                log.info("Detector de personas%s en %s%s", " y vehículos" if vehicles else "", det.device,
                         f" (lotes de hasta {sess.max_batch} cuadros)" if sess.max_batch > 1 else "")
            return det
    raise ValueError(f"Detector desconocido: {name!r} (usar 'yolo' o 'motion')")
