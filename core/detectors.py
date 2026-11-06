"""Detectores de personas.

- OnnxYoloDetector: YOLO exportado a ONNX con onnxruntime (sin PyTorch). Usa la placa de video si hay
  (CUDA/TensorRT en servidores NVIDIA, DirectML en Windows, con un solo modelo compartido por todas las
  cámaras) y si no, el procesador. En NVIDIA analiza los cuadros de varias cámaras juntos (lotes).
  Es el detector para producción.
- MotionDetector: sustracción de fondo. Solo para pruebas y demos: no distingue personas.
- TunedDetector / `tuned`: el detector compartido con la «Sensibilidad» y «Personas lejos» de una cámara
  (core/deteccion.py). El umbral y el mosaico van en cada llamada: el detector compartido no cambia.
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

from .deteccion import DEFAULT_SENSITIVITY, FAR_PASSES, DetectionConfig, conf_for  # noqa: F401 (se usan desde acá)
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

    def run_many(self, canvases: list[np.ndarray]) -> list[np.ndarray]:
        """Varios lienzos de una misma cámara (con «Personas lejos»: la imagen entera y sus 4 partes). En lotes van
        todos juntos a la cola, así caen en el mismo lote de la placa de video en vez de esperar uno por uno; de a un
        cuadro, uno tras otro (cada uno toma el candado: las otras cámaras no quedan esperando las 5 pasadas)."""
        if self.max_batch == 1:
            return [self.run(c) for c in canvases]
        futs: list[Future] = []
        for c in canvases:
            fut = Future()
            self._queue.put((c, fut))
            futs.append(fut)
        return [f.result() for f in futs]

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


class SessionPool:
    """Máquina sin placa de video: una sesión de 1 hilo por núcleo, compartidas por todas las cámaras (cada cuadro
    usa la primera libre). Rinde bastante más que una sola sesión que reparte cada cuadro entre todos los núcleos:
    medido con 4 núcleos, 42 cuadros por segundo contra 29."""

    def __init__(self, model_path: str | Path, size: int):
        first = OnnxSession(model_path, threads=1, max_batch=1)
        self.pool = [first] + ([] if first.on_gpu else
                               [OnnxSession(model_path, threads=1, max_batch=1) for _ in range(size - 1)])
        self.session, self.device, self.on_gpu, self.size = first.session, first.device, first.on_gpu, first.size
        self.uint8, self.col_of, self.max_batch = first.uint8, first.col_of, 1
        self._free: queue.Queue = queue.Queue()
        for s in self.pool:
            self._free.put(s)

    def run(self, canvas: np.ndarray) -> np.ndarray:
        s = self._free.get()
        try:
            return s.run(canvas)
        finally:
            self._free.put(s)

    def run_many(self, canvases: list[np.ndarray]) -> list[np.ndarray]:
        """Uno tras otro, cada uno en la primera sesión libre (las otras cámaras ocupan los demás núcleos)."""
        return [self.run(c) for c in canvases]

    def take_stats(self) -> tuple[int, float]:
        frames = sum(s.take_stats()[0] for s in self.pool)
        return frames, (1.0 if frames else 0.0)


# «Personas lejos»: la imagen en 2x2 partes que se pisan un poco, cada una agrandada a la entrada del modelo.
FAR_OVERLAP = 0.25  # cuánto se pisan dos partes vecinas (fracción de la parte): una persona chica entra entera en una
FAR_EDGE = 0.02  # una persona a menos de esto (fracción de la parte) de un borde interno de su parte está cortada
FAR_MIN_SIDE = 64  # un cuadro más chico que esto no se parte (no queda nada que agrandar)


def far_tiles(w: int, h: int, overlap: float = FAR_OVERLAP) -> list[tuple[int, int, int, int]]:
    """Las 4 partes (x1, y1, x2, y2) de un cuadro de `w` x `h`: 2 columnas y 2 filas, cada una del 57 % del lado (con
    `overlap` 0,25), que se pisan en el 25 % de la parte (el 14 % del cuadro) y cubren todo el cuadro."""
    if w < FAR_MIN_SIDE or h < FAR_MIN_SIDE:
        return []
    tw = min(w, max(1, int(round(w / (2 - overlap)))))
    th = min(h, max(1, int(round(h / (2 - overlap)))))
    return [(x, y, x + tw, y + th) for y in (0, h - th) for x in (0, w - tw)]


def _covers(a: BBox, b: BBox, iou_thr: float, cover: float) -> bool:
    """Si `a` y `b` son la misma persona: se pisan (IoU) o una está casi entera dentro de la otra (una persona cortada
    en una parte queda dentro de la misma persona vista entera)."""
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return False
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    small = min(area_a, area_b)
    return (union > 0 and inter / union >= iou_thr) or (small > 0 and inter / small >= cover)


def merge_far(full: list[Detection], parts: list[Detection], iou_thr: float = 0.5,
              cover: float = 0.7) -> list[Detection]:
    """Junta lo de la imagen entera (`full`, que gana siempre) con las personas de las partes (`parts`, en coordenadas
    del cuadro): de las partes se suma solo lo que no es alguien ya visto (de la imagen entera o de otra parte, de más
    puntaje). Solo se comparan detecciones de la misma etiqueta (una persona dentro de un auto no se pierde)."""
    out = list(full)
    for d in sorted(parts, key=lambda d: -d.conf):
        if any(o.label == d.label and _covers(d.bbox, o.bbox, iou_thr, cover) for o in out):
            continue
        out.append(d)
    return out


class OnnxYoloDetector:
    """Personas (y opcionalmente vehículos) con YOLO (v8/11) exportado a ONNX.
    Salida esperada: (1, 4+num_clases, N). `classes`: índice COCO → etiqueta. Varias instancias (por ejemplo,
    con y sin vehículos) pueden compartir la misma `OnnxSession`.

    El mismo detector lo comparten todas las cámaras: la sensibilidad de cada una (core/deteccion.py) llega en cada
    llamada (`conf` / `person_conf`, solo para personas; los vehículos y lo demás siguen con `self.conf`) y lo mismo
    «Personas lejos» (`far`). `self.conf` no se toca nunca."""

    name = "YOLO11n"
    tunable = True  # acepta la sensibilidad y «Personas lejos» por llamada (core/detectors.tuned)

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

    def detect(self, frame: np.ndarray, conf: float | None = None, far: bool = False) -> list[Detection]:
        """`conf`: confianza mínima de las personas para esta llamada (None = `self.conf`). `far`: además de la imagen
        entera, la mira en 4 partes agrandadas y suma las personas que solo se ven ahí (unas 5 veces el trabajo)."""
        main, _ = self._main(frame, conf, far)
        return main

    def detect_also(self, frame: np.ndarray, classes: dict[int, str], conf: float, person_conf: float | None = None,
                    far: bool = False) -> tuple[list[Detection], list[Detection]]:
        """Una sola pasada del modelo: las detecciones de siempre (`self.classes`, idénticas a `detect`) y, aparte,
        las de `classes` (por ejemplo mochilas o botellas, para mostrar qué más ve YOLO sin tocar el seguimiento ni
        las reglas). `conf` es para esas otras clases; `person_conf` y `far`, como en `detect`. Con `far` lo demás sale
        solo de la imagen entera (las partes suman personas)."""
        main, res = self._main(frame, person_conf, far)
        idx = [c for c in sorted(classes) if c in self.runner.col_of and c not in self.classes]
        extra = (self._decode(res, np.array([self.runner.col_of[c] for c in idx]), [classes[c] for c in idx], conf)
                 if idx else [])
        return main, extra

    def detect_groups(self, frame: np.ndarray, groups: list[tuple[dict[int, str], float]],
                      person_conf: float | None = None, far: bool = False) -> tuple[
            list[Detection], list[list[Detection]]]:
        """Como `detect_also` pero con varios grupos de clases, cada uno decodificado aparte (una caja es la clase de
        más puntaje DENTRO de su grupo). Así «Configurar…» saca los bolsos igual que las máquinas (entre mochila,
        cartera y valija) y además muestra todo lo demás que ve YOLO, con una sola pasada del modelo."""
        main, res = self._main(frame, person_conf, far)
        out = []
        for classes, conf in groups:
            idx = [c for c in sorted(classes) if c in self.runner.col_of and c not in self.classes]
            out.append(self._decode(res, np.array([self.runner.col_of[c] for c in idx]), [classes[c] for c in idx],
                                    conf) if idx else [])
        return main, out

    def _main(self, frame: np.ndarray, person_conf: float | None, far: bool) -> tuple[list[Detection], tuple]:
        """Las detecciones de `self.classes` (con el umbral de personas pedido y, con `far`, las personas de las
        partes) y la salida del modelo para la imagen entera (para decodificar otras clases)."""
        pc = self.conf if person_conf is None else person_conf
        if not far:
            res = self._infer(frame)
            return self._decode(res, self._cols, self._labels, self.conf, pc), res
        res, parts = self._far(frame, pc)
        return merge_far(self._decode(res, self._cols, self._labels, self.conf, pc), parts), res

    def _far(self, frame: np.ndarray, person_conf: float) -> tuple[tuple, list[Detection]]:
        """La imagen entera y sus 4 partes en una tanda (en NVIDIA, el mismo lote). Devuelve la salida de la imagen
        entera y las personas de las partes, en coordenadas del cuadro, sin las que quedaron cortadas por el borde
        interno de su parte (esas las ve enteras la parte vecina, que se pisa con esta, o la imagen entera)."""
        h, w = frame.shape[:2]
        tiles = far_tiles(w, h)
        res = self._infer_many([frame] + [frame[y1:y2, x1:x2] for x1, y1, x2, y2 in tiles])
        parts: list[Detection] = []
        for (x1, y1, x2, y2), r in zip(tiles, res[1:]):
            tw, th = x2 - x1, y2 - y1
            ex, ey = FAR_EDGE * tw, FAR_EDGE * th
            for d in self._decode(r, self._cols, self._labels, self.conf, person_conf):
                if d.label != "persona":
                    continue  # de las partes solo personas: vehículos y lo demás salen de la imagen entera
                bx1, by1, bx2, by2 = d.bbox
                if ((x1 > 0 and bx1 <= ex) or (x2 < w and bx2 >= tw - ex)
                        or (y1 > 0 and by1 <= ey) or (y2 < h and by2 >= th - ey)):
                    continue
                parts.append(Detection((bx1 + x1, by1 + y1, bx2 + x1, by2 + y1), d.label, d.conf))
        return res[0], parts

    def _letterbox(self, frame: np.ndarray) -> tuple[np.ndarray, tuple]:
        """La imagen llevada al tamaño del modelo sin deformarla (bordes grises) y cómo volver a sus coordenadas."""
        h, w = frame.shape[:2]
        tw, th = self.size
        r = min(tw / w, th / h)
        nw, nh = max(1, int(round(w * r))), max(1, int(round(h * r)))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((th, tw, 3), 114, np.uint8)
        px, py = (tw - nw) // 2, (th - nh) // 2
        canvas[py:py + nh, px:px + nw] = resized
        return canvas, (w, h, r, px, py)

    @staticmethod
    def _shape(out: np.ndarray) -> np.ndarray:
        return out.T if out.shape[0] < out.shape[1] else out  # (4+C, N) → (N, 4+C)

    def _infer(self, frame: np.ndarray) -> tuple:
        canvas, geo = self._letterbox(frame)
        return (self._shape(self.runner.run(canvas)), *geo)

    def _infer_many(self, frames: list[np.ndarray]) -> list[tuple]:
        if len(frames) == 1:
            return [self._infer(frames[0])]
        boxes = [self._letterbox(f) for f in frames]
        run_many = getattr(self.runner, "run_many", None)  # (un runner de prueba puede no tenerlo)
        outs = run_many([c for c, _ in boxes]) if run_many else [self.runner.run(c) for c, _ in boxes]
        return [(self._shape(out), *geo) for out, (_, geo) in zip(outs, boxes)]

    def _decode(self, res: tuple, cols: np.ndarray, labels: list[str], conf: float,
                person_conf: float | None = None) -> list[Detection]:
        """Cajas de las clases `cols` (cada caja es la clase de más puntaje entre ellas). Las personas se aceptan desde
        `person_conf` (None = `conf`) y lo demás desde `conf`: la NMS es por etiqueta, así que cada una usa su
        umbral."""
        out, w, h, r, px, py = res
        pc = conf if person_conf is None else person_conf
        cls_scores = out[:, cols]  # (N, clases elegidas)
        best = cls_scores.argmax(axis=1)
        scores = cls_scores[np.arange(len(best)), best]
        keep = scores >= (min(conf, pc) if "persona" in labels else conf)
        if not np.any(keep):
            return []
        boxes, scores, best = out[keep, :4], scores[keep], best[keep]
        cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = (cx - bw / 2 - px) / r
        y1 = (cy - bh / 2 - py) / r
        x2 = (cx + bw / 2 - px) / r
        y2 = (cy + bh / 2 - py) / r
        dets: list[Detection] = []
        for k, label in enumerate(labels):  # NMS por etiqueta, cada una con su umbral
            thr = pc if label == "persona" else conf
            sel = np.flatnonzero((best == k) & (scores >= thr))
            if not len(sel):
                continue
            rects = [[float(x1[i]), float(y1[i]), float(x2[i] - x1[i]), float(y2[i] - y1[i])] for i in sel]
            idx = cv2.dnn.NMSBoxes(rects, scores[sel].astype(float).tolist(), thr, self.iou)
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
    """Cuadros analizados en la placa de video y tamaño promedio del lote, desde la última consulta. Con «Personas
    lejos» cada parte cuenta como un cuadro (son pasadas del modelo: un cuadro de esa cámara son 5)."""
    frames = batches = 0
    for sess in list(_sessions.values()):
        f, avg = sess.take_stats()
        frames += f
        batches += f / avg if avg else 0
    return frames, (frames / batches if batches else 0.0)


def build_detector(name: str, model_path: str | Path | None = None, vehicles: bool = False,
                   shared: bool = False) -> Detector:
    """`vehicles=True`: además de personas detecta autos, motos, ómnibus y camiones (estaciones de servicio).
    `shared=True` (la app de escritorio): un solo modelo para todas las cámaras también en el procesador. Cargarlo
    frena todo el programa unos segundos (onnxruntime no suelta Python mientras arma la sesión): sin esto, en una PC sin
    placa de video cada cámara que se abría con «Ver con la IA» lo volvía a cargar y la ventana quedaba «No responde»."""
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
                # VIGIA_HILOS (máquina sin placa de video): una sesión por núcleo, compartidas por todas las cámaras
                hilos = int(os.getenv("VIGIA_HILOS") or 0)
                # la app: una sola sesión con más hilos, en vez de una de 2 hilos por cámara
                threads = hilos or (min(4, max(2, os.cpu_count() or 2)) if shared else 2)
                sess = (SessionPool(model_path, hilos) if hilos > 1 else
                        OnnxSession(model_path, threads=threads, max_batch=None))
            det = OnnxYoloDetector(classes=classes, session=sess)
            if det.on_gpu or sess.max_batch > 1 or os.getenv("VIGIA_HILOS") or shared:  # una sola sesión para todas
                _sessions[str(model_path)] = sess
                _shared[key] = det
            if key not in _logged:
                _logged.add(key)
                log.info("Detector de personas%s en %s%s", " y vehículos" if vehicles else "", det.device,
                         f" (lotes de hasta {sess.max_batch} cuadros)" if sess.max_batch > 1 else
                         f" ({len(sess.pool)} cuadros a la vez)" if isinstance(sess, SessionPool) else "")
            return det
    raise ValueError(f"Detector desconocido: {name!r} (usar 'yolo' o 'motion')")


class TunedDetector:
    """El detector compartido con la sensibilidad y «Personas lejos» de UNA cámara (core/deteccion.py).

    No cambia nada del detector compartido (otras cámaras lo usan con otro umbral, a veces al mismo tiempo desde otros
    hilos): pasa el umbral de personas y el mosaico en cada llamada. Lo demás (`name`, `device`, `on_gpu`, `size`,
    `runner`…) es el del detector de abajo. `detect_also` y `detect_groups` están solo si el de abajo los tiene (el
    resto del programa pregunta con `hasattr` qué sabe hacer un detector). Se arma con `tuned`, no a mano."""

    tunable = True

    def __init__(self, base, cfg: DetectionConfig):
        self.base = base
        self.cfg = cfg
        self.conf = cfg.conf  # confianza mínima de las personas en esta cámara
        if hasattr(base, "detect_also"):
            self.detect_also = self._detect_also
        if hasattr(base, "detect_groups"):
            self.detect_groups = self._detect_groups

    @property
    def name(self) -> str:
        return self.base.name

    @property
    def device(self) -> str:
        return self.base.device

    @property
    def on_gpu(self) -> bool:
        return self.base.on_gpu

    @property
    def size(self):
        return self.base.size

    def _person_conf(self, conf: float | None) -> float:
        return self.conf if conf is None else conf

    def _far(self, far: bool | None) -> bool:
        return self.cfg.far if far is None else far

    def detect(self, frame: np.ndarray, conf: float | None = None, far: bool | None = None) -> list[Detection]:
        return self.base.detect(frame, conf=self._person_conf(conf), far=self._far(far))

    def _detect_also(self, frame: np.ndarray, classes: dict[int, str], conf: float, person_conf: float | None = None,
                     far: bool | None = None) -> tuple[list[Detection], list[Detection]]:
        return self.base.detect_also(frame, classes, conf, person_conf=self._person_conf(person_conf),
                                     far=self._far(far))

    def _detect_groups(self, frame: np.ndarray, groups: list[tuple[dict[int, str], float]],
                       person_conf: float | None = None, far: bool | None = None) -> tuple[
            list[Detection], list[list[Detection]]]:
        return self.base.detect_groups(frame, groups, person_conf=self._person_conf(person_conf), far=self._far(far))

    def __getattr__(self, name: str):
        # solo llega acá lo que no está en el envoltorio: lo del detector de abajo
        if name.startswith("__") or name in ("base", "cfg", "conf", "detect_also", "detect_groups"):
            raise AttributeError(name)
        return getattr(self.base, name)

    def __repr__(self) -> str:
        return f"TunedDetector({self.base!r}, {self.cfg.describe()})"


def tuned(detector, cfg: DetectionConfig | dict | None):
    """El detector de una cámara con su detección (`cfg`: DetectionConfig, el `analysis["detection"]` tal cual llega
    del servidor, o None = lo de siempre).

    - Con lo de siempre devuelve el detector de abajo tal cual (sin envoltorio: nada cambia para esa cámara).
    - Un detector que no acepta el ajuste (movimiento, los de las pruebas) también vuelve tal cual.
    - Si ya es un `TunedDetector`, se arma sobre el de abajo (no se envuelve dos veces): así cualquiera vuelve a
      aplicar la detección en caliente con `tuned(lo_que_tenga, cfg_nueva)`; con la misma configuración devuelve el
      mismo objeto."""
    base = detector.base if isinstance(detector, TunedDetector) else detector
    if isinstance(cfg, DetectionConfig):
        pass
    elif isinstance(cfg, dict) or cfg is None:
        cfg = DetectionConfig.from_dict(cfg)
    else:
        cfg = DetectionConfig()
    if cfg.is_default or not getattr(base, "tunable", False):
        return base
    if isinstance(detector, TunedDetector) and detector.cfg == cfg:
        return detector
    return TunedDetector(base, cfg)
