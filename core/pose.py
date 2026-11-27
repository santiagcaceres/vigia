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
import time
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
        self.model_path = model_path or getattr(getattr(self.runner, "session", None), "_model_path", None)
        self._lock = threading.Lock()

    def _run(self, canvas: np.ndarray) -> np.ndarray:
        """Corre el modelo; si falla en la placa de video (le pasó a otro modelo con DirectML en la Dell), sigue en
        el procesador en vez de cortar el análisis de la cámara."""
        runner = self.runner
        try:
            return runner.run(canvas)
        except Exception as exc:  # noqa: BLE001
            if not self.model_path:
                raise
            with self._lock:
                if self.runner is runner:  # nadie lo cambió todavía
                    if not self.on_gpu:
                        raise
                    log.warning("La postura falló en %s (%s): sigue en el procesador", self.device, exc)
                    self.runner = _CpuRunner(self.model_path)
                    self.device, self.on_gpu = "procesador", False
            return self.runner.run(canvas)  # (otra cámara ya la pasó al procesador: reintentar ahí)

    def _run_many(self, canvases: list[np.ndarray]) -> list[np.ndarray]:
        """Varios recortes en una tanda: en la placa NVIDIA van juntos a la cola de lotes (una sola llamada al modelo
        para todas las personas del cuadro, y las de otras cámaras que lleguen en ese instante); si no, de a uno."""
        runner = self.runner
        many = getattr(runner, "run_many", None)
        if many is None or len(canvases) == 1:
            return [self._run(c) for c in canvases]
        try:
            return many(canvases)
        except Exception:  # noqa: BLE001 - falló el lote: de a uno, que tiene su propio plan B (pasar al procesador)
            return [self._run(c) for c in canvases]

    def _prepare(self, frame: np.ndarray, bbox: BBox):
        """El recorte de la persona llevado al tamaño del modelo y cómo volver a las coordenadas del cuadro (None si
        el recuadro no sirve)."""
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
        return canvas, (bbox, cx1, cy1, r, ox, oy)

    def _keypoints(self, out: np.ndarray, geo: tuple) -> np.ndarray | None:
        """De la salida del modelo (56, N: cx, cy, w, h, puntaje, 17 x (x, y, conf)), los puntos de la persona del
        recuadro, en píxeles del cuadro."""
        (x1, y1, x2, y2), cx1, cy1, r, ox, oy = geo
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

    def estimate(self, frame: np.ndarray, bbox: BBox) -> np.ndarray | None:
        ready = self._prepare(frame, bbox)
        if ready is None:
            return None
        return self._keypoints(self._run(ready[0]), ready[1])

    def estimate_many(self, frame: np.ndarray, bboxes: list[BBox]) -> list[np.ndarray | None]:
        """La postura de varias personas del mismo cuadro: lo mismo que `estimate` de a una, pero con una sola tanda
        del modelo (25 nov: en la máquina de AWS cada llamada al modelo cuesta procesador aunque el trabajo lo haga la
        placa; con 3 personas a la vista eran 3 llamadas por cuadro)."""
        ready = [self._prepare(frame, b) for b in bboxes]
        idx = [i for i, r in enumerate(ready) if r is not None]
        out: list[np.ndarray | None] = [None] * len(bboxes)
        if idx:
            for i, res in zip(idx, self._run_many([ready[i][0] for i in idx])):
                out[i] = self._keypoints(res, ready[i][1])
        return out


class _CpuRunner:
    """La sesión del modelo de postura en el procesador (cuando la placa de video falla)."""

    def __init__(self, model_path: str | Path, threads: int = 2):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self._lock = threading.Lock()

    def run(self, canvas: np.ndarray) -> np.ndarray:
        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        with self._lock:
            return self.session.run(None, {self.input_name: blob})[0][0]


def pick(tracks: list, limit: int = MAX_PER_FRAME, min_px: float = MIN_PERSON_PX) -> list:
    """Las personas a las que se les mira la postura en este cuadro: visibles, de alto suficiente, las más grandes
    primero y como mucho `limit`."""
    cand = [t for t in tracks if getattr(t, "visible", True) and getattr(t, "label", "persona") == "persona"
            and t.bbox[3] - t.bbox[1] >= min_px]
    cand.sort(key=lambda t: -(t.bbox[3] - t.bbox[1]))
    return cand[:limit]


def look(estimator: PoseEstimator | None, frame: np.ndarray, tracks: list, min_px: float = MIN_PERSON_PX) -> int:
    """Postura de las personas elegidas (`pick`) en `t.kpts`; a las demás, None. Devuelve a cuántas se les miró."""
    chosen = pick(tracks, min_px=min_px) if estimator is not None else []
    for t in tracks:
        t.kpts = None
    many = getattr(estimator, "estimate_many", None)  # (un estimador de prueba puede tener solo `estimate`)
    found = (many(frame, [t.bbox for t in chosen]) if many is not None and len(chosen) > 1
             else [estimator.estimate(frame, t.bbox) for t in chosen])
    for t, kp in zip(chosen, found):
        t.kpts = kp
    return sum(kp is not None for kp in found)


def detect_with_bags(detector, frame: np.ndarray, bags: bool) -> tuple[list, list]:
    """Personas (y lo demás que el detector busque) y, en la misma pasada del modelo, bolsos, carteras y mochilas
    para el gesto «mano en la bolsa». Un detector sin `detect_also` (movimiento) no da bolsos. El detector de una
    cámara con su sensibilidad (core/detectors.tuned) pone él el umbral de las personas y «Personas lejos»: los
    bolsos salen siempre de la imagen entera, desde 0,3."""
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


CHECK_TOLERANCE = 1.5  # el lote tiene que dar lo mismo que de a una persona (en píxeles del recorte de 256 y puntaje)
POSE_WAIT_S = 0.002  # cuánto espera el lote por más personas: las de un cuadro llegan juntas, no hace falta esperar


RETRY_AFTER_S = 7 * 86400.0  # una marca de «se cerró preparando el lote» vale una semana; después se prueba de nuevo


def _auto_batch() -> int:
    """De fábrica: en la placa NVIDIA, lotes de hasta `MAX_PER_FRAME` personas (justo los tamaños que se comprueban al
    arrancar); en lo demás (procesador, DirectML de la app), de a una como siempre. `VIGIA_LOTE=1` (apagar los lotes
    del detector) apaga también estos."""
    if (os.getenv("VIGIA_LOTE") or "").strip() == "1":
        return 1
    try:
        import onnxruntime as ort

        from .detectors import _providers

        nvidia = any(p in _providers(ort) for p in ("TensorrtExecutionProvider", "CUDAExecutionProvider"))
    except Exception:  # noqa: BLE001 - sin onnxruntime no hay nada que decidir
        return 1
    return MAX_PER_FRAME if nvidia else 1


def _crashed_before(mark: Path | None) -> bool:
    """¿La vez anterior el proceso se cerró mientras preparaba la postura por lotes? (queda la marca; vale una semana)"""
    try:
        return bool(mark) and mark.exists() and time.time() - mark.stat().st_mtime < RETRY_AFTER_S
    except OSError:
        return False


def _session(path: Path) -> OnnxSession:
    """La sesión del modelo de postura. En la placa NVIDIA, por lotes (`VIGIA_LOTE_POSTURA`: otro tamaño; 1 = de a una
    persona como antes): todas las personas de un cuadro en una sola llamada. Antes de usarla se comprueba contra el
    modelo original en el procesador, con recortes de prueba y todos los tamaños de lote que se van a usar: si el lote
    no da lo mismo (o falla), se sigue de a una persona como siempre. De paso esos tamaños quedan «en caliente» (la
    placa elige cómo calcular cada uno la primera vez que lo ve, y eso tarda).

    `VIGIA_MARCA_POSTURA` (lo pone la máquina de análisis): un archivo que existe mientras se arma y se comprueba el
    lote. Si el proceso se cierra en ese rato (una falla de la placa o del driver no avisa: lo tira abajo), la marca
    queda y la vez siguiente arranca de a una persona en vez de volver a caerse."""
    env = (os.getenv("VIGIA_LOTE_POSTURA") or "").strip()
    try:
        batch = max(1, int(env)) if env else _auto_batch()
    except ValueError:
        batch = _auto_batch()
    if batch <= 1:
        return OnnxSession(path, threads=2, max_batch=1)
    mark = Path(os.environ["VIGIA_MARCA_POSTURA"]) if os.getenv("VIGIA_MARCA_POSTURA") else None
    if _crashed_before(mark):
        log.warning("La vez anterior el proceso se cerró preparando la postura por lotes: se calcula de a una persona "
                    "(para volver a probar, borrar %s)", mark)
        return OnnxSession(path, threads=2, max_batch=1)
    try:
        if mark:
            mark.parent.mkdir(parents=True, exist_ok=True)
            mark.write_text("Preparando la postura por lotes. Si este archivo queda, el proceso se cerró en ese rato.\n",
                            encoding="utf-8")
    except OSError:
        mark = None
    try:
        # (TensorRT para la postura solo con VIGIA_TENSORRT=todo; se prueba después de comprobar el lote con CUDA)
        from .detectors import tensorrt_mode
        sess = OnnxSession(path, threads=2, max_batch=batch, max_wait=POSE_WAIT_S, all_outputs=True)
        if sess.max_batch <= 1:
            return sess
        if not env and not sess.on_gpu:  # la placa no quiso el modelo por lotes: con el de siempre, que quizá sí
            log.warning("La postura por lotes no quedó en la placa de video: se calcula de a una persona")
            return OnnxSession(path, threads=2, max_batch=1)
        try:
            ref = _CpuRunner(path, threads=1)
            rng = np.random.default_rng(7)
            canvases = [cv2.GaussianBlur(rng.integers(0, 255, (sess.size[1], sess.size[0], 3), dtype=np.uint8),
                                         (0, 0), 6) for _ in range(sess.max_batch)]
            want = [ref.run(c) for c in canvases[:2]]
            worst = 0.0
            for n in range(1, len(canvases) + 1):
                got = sess.run_many(canvases[:n])
                for w, g in zip(want, got):
                    if g.shape != w.shape:
                        raise ValueError(f"el lote devuelve otra forma ({g.shape} en vez de {w.shape})")
                    worst = max(worst, float(np.abs(g - w).max()))
            if worst > CHECK_TOLERANCE:
                raise ValueError(f"el lote da distinto que de a una (diferencia {worst:.2f})")
            if tensorrt_mode() == "todo":
                return _with_tensorrt(path, sess, canvases) or sess
            return sess
        except Exception as exc:  # noqa: BLE001 - ante la duda, como siempre
            log.warning("La postura por lotes no pasó la comprobación (%s): se calcula de a una persona", exc)
            return OnnxSession(path, threads=2, max_batch=1)
    finally:
        if mark:  # (si el proceso se cerró antes de llegar acá, la marca queda)
            try:
                mark.unlink()
            except OSError:
                pass


# TensorRT en media precisión no da exactamente lo mismo que CUDA (27 nov, medido en la máquina de AWS con 17 personas
# de dos fotos: los puntos se mueven 0,004 largos de torso en promedio, el peor 0,17; los gestos usan umbrales de 0,35
# a 0,5), así que no se le pide la comprobación exacta del lote (1,5 píxeles en todo): se compara con el lote de CUDA
# ya comprobado y tiene que dar casi lo mismo en lo típico (la mitad de los números a menos de esto) y en el puntaje
TRT_MEDIAN_TOLERANCE = 1.0  # píxeles del recorte de 256 (mediana de todas las salidas)
TRT_SCORE_TOLERANCE = 0.05  # el puntaje de cada persona encontrada (de 0 a 1)


def _with_tensorrt(path: Path, cuda: OnnxSession, canvases: list[np.ndarray]) -> OnnxSession | None:
    """La postura por lotes con TensorRT (`VIGIA_TENSORRT=todo`), si queda en TensorRT y da casi lo mismo que el lote de
    CUDA ya comprobado. None = sigue con CUDA (y lo dice en el registro). Medido: 2,3 a 3,6 veces menos placa y
    procesador por persona."""
    try:
        trt = OnnxSession(path, threads=2, max_batch=cuda.max_batch, max_wait=POSE_WAIT_S, all_outputs=True, trt=True)
    except Exception as exc:  # noqa: BLE001
        log.warning("La postura con TensorRT no se pudo armar (%s): sigue con CUDA", str(exc)[:200])
        return None
    if "Tensorrt" not in trt.device:
        trt.close()
        return None
    try:
        a = np.stack(cuda.run_many(canvases))
        b = np.stack(trt.run_many(canvases))
        if a.shape != b.shape or not np.isfinite(b).all():
            raise ValueError(f"devuelve otra cosa ({b.shape} en vez de {a.shape})")
        diff = np.abs(a - b)
        median = float(np.median(diff))
        score = float(diff[:, 4].max()) if a.ndim == 3 and a.shape[1] > 4 else 0.0
        if median > TRT_MEDIAN_TOLERANCE or score > TRT_SCORE_TOLERANCE:
            raise ValueError(f"da distinto que CUDA (mediana {median:.2f}, puntaje {score:.3f})")
    except Exception as exc:  # noqa: BLE001
        log.warning("La postura con TensorRT no pasó la comprobación (%s): sigue con CUDA", exc)
        trt.close()
        return None
    cuda.close()
    log.info("Postura con TensorRT (diferencia con CUDA: mediana %.3f, puntaje %.4f)", median, score)
    return trt


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
            est = PoseEstimator(path, runner=SessionPool(path, hilos) if hilos > 1 else _session(path))
            _estimators[key] = est
            lote = getattr(est.runner, "max_batch", 1)
            log.info("Postura de las personas (%s) en %s%s", est.name, est.device,
                     f" (lotes de hasta {lote} personas)" if lote > 1 else "")
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
