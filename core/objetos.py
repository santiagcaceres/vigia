"""Objetos propios que se le enseñan a la IA (pestaña «Entrenar objetos» de la app, 18 nov; el dueño: «así podemos
hacer los pallets, medir cuántos pallets hay y cuántos se están haciendo»).

Un **objeto** («Pallets») tiene sus **clases** (los estados que la IA tiene que distinguir: «Pallet vacío», «En armado»,
«Terminado con film») y el plan de métricas donde se va a medir. Se juntan fotos (subidas, sacadas de un video o de una
cámara), una persona marca un recuadro en cada objeto y una máquina con Ultralytics entrena un YOLO propio (YOLO11n
afinado) que se exporta a ONNX.

Lo de acá lo usan el servidor (sin OpenCV ni onnxruntime: solo las funciones de datos), las máquinas y la app (el
detector y las fotos de un video, que importan OpenCV adentro de cada función).

Formatos:
- clases: `[{"key": "c1", "name": "Pallet vacío", "prompt": "empty pallet"}]`. La `key` no cambia nunca (renombrar
  no rompe lo marcado); `prompt` (en inglés, opcional) es lo que se le escribe a YOLOE para que proponga recuadros
  antes de tener un modelo propio.
- recuadro: `{"k": "c1", "x1": .., "y1": .., "x2": .., "y2": ..}` en fracciones de la foto (0 a 1); los propuestos por
  la IA llevan además `"p"` (confianza).
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata

MAX_CLASSES = 8
MAX_NAME = 40
MAX_PROMPT = 60
MAX_BOXES = 300  # por foto
MIN_BOX = 0.004  # lado mínimo de un recuadro (5 píxeles en una foto de 1280)
IMG_MAX_SIDE = 1280  # las fotos se guardan con el lado mayor hasta esto (se entrena a 640)
JPEG_QUALITY = 90
# para entrenar: lo mínimo (si no, no se deja) y lo que recomendamos (con eso empieza a andar bien)
MIN_PHOTOS = 30
MIN_BOXES = 20  # por clase: con menos casi no la aprende
GOOD_PHOTOS = 300
GOOD_BOXES = 100
TEST_SHARE = 5  # 1 de cada 5 fotos queda para probar (la IA no la ve al entrenar)
MIN_TEST_PHOTOS = 5
IMGSZ = 640
STATUSES = ("subiendo", "sin_marcar", "marcada", "descartada")
STATUS_NAMES = {"subiendo": "Subiendo", "sin_marcar": "Sin marcar", "marcada": "Marcada", "descartada": "Descartada"}
JOB_KINDS = ("video", "premarcar", "entrenar")
JOB_NAMES = {"video": "Sacar fotos de un video", "premarcar": "Pre-marcar con la IA", "entrenar": "Entrenar"}
JOB_STATES = ("subiendo", "en_cola", "procesando", "listo", "error", "cancelado")
JOB_STATE_NAMES = {"subiendo": "Subiendo el video", "en_cola": "En cola (esperando una máquina)",
                   "procesando": "Trabajando", "listo": "Listo", "error": "Error", "cancelado": "Cancelado"}
VIDEO_EVERY_S = 2.0  # de un video: una foto cada 2 s como mucho
VIDEO_MAX_IMAGES = 300
VIDEO_MIN_CHANGE = 2.5  # diferencia media (0-255) en 64x36 gris para que una foto cuente como distinta (el ruido
# de compresión de una escena quieta da menos de 1,5)
DEFAULT_CONF = 0.35  # confianza para mostrar lo que encuentra un modelo propio
PROPOSE_CONF = 0.15  # para pre-marcar se muestra más (una persona decide): mejor proponer de más que de menos
_PROMPT_OK = re.compile(r"^[A-Za-z0-9 ,.'-]*$")


class ObjectDataError(ValueError):
    """Un dato que no sirve (lo dice en castellano, para mostrarlo tal cual)."""


# ------------------------------------------------------------------ clases
def _plain(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower().strip()


def clean_classes(raw, old: list[dict] | None = None) -> list[dict]:
    """Las clases de un objeto. Cada una trae `name` (y opcional `prompt` y la `key` que ya tenía): las nuevas reciben
    la siguiente key libre («c1», «c2»…, nunca se reusa una). Lanza ObjectDataError con el motivo."""
    if not isinstance(raw, list) or not raw:
        raise ObjectDataError("Poné al menos una clase (por ejemplo «Pallet»)")
    if len(raw) > MAX_CLASSES:
        raise ObjectDataError(f"Hasta {MAX_CLASSES} clases por objeto: si son más, conviene separarlo en dos objetos")
    old = old or []
    old_keys = {c["key"] for c in old}
    used = [int(k[1:]) for k in old_keys if re.fullmatch(r"c\d+", k)]
    nxt = max(used, default=0) + 1
    out, seen, keys = [], set(), set()
    for c in raw:
        if not isinstance(c, dict):
            raise ObjectDataError("Cada clase va con su nombre")
        name = c.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ObjectDataError("Hay una clase sin nombre")
        name = " ".join(name.split())
        if len(name) > MAX_NAME:
            raise ObjectDataError(f"El nombre «{name[:20]}…» es muy largo (hasta {MAX_NAME} letras)")
        if _plain(name) in seen:
            raise ObjectDataError(f"La clase «{name}» está repetida")
        seen.add(_plain(name))
        prompt = c.get("prompt") or ""
        if not isinstance(prompt, str):
            raise ObjectDataError("La palabra para la IA va como texto")
        prompt = " ".join(prompt.split())
        if len(prompt) > MAX_PROMPT or not _PROMPT_OK.match(prompt):
            raise ObjectDataError(f"La palabra para la IA de «{name}» va en inglés, sin tildes y hasta {MAX_PROMPT} "
                                  f"letras (por ejemplo «pallet» o «wooden pallet»)")
        key = c.get("key")
        if key is not None:
            if key not in old_keys or key in keys:
                raise ObjectDataError("Una clase trae una clave que no es de este objeto")
        else:
            key, nxt = f"c{nxt}", nxt + 1
        keys.add(key)
        out.append({"key": key, "name": name, "prompt": prompt})
    return out


def class_names(classes: list[dict]) -> dict[str, str]:
    return {c["key"]: c["name"] for c in classes or []}


# ------------------------------------------------------------------ recuadros
def _num(v) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return None
    return float(v)


def clean_boxes(raw, keys, with_conf: bool = False) -> list[dict]:
    """Los recuadros de una foto, validados y redondeados (fracciones de 0 a 1, con x1 < x2 e y1 < y2). `keys`: las
    clases que existen. Con `with_conf` (lo que propone la IA) cada uno trae `p` y la clase puede faltar (None: la
    IA vio «un objeto» y la persona elige cuál es)."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ObjectDataError("Los recuadros van en una lista")
    if len(raw) > MAX_BOXES:
        raise ObjectDataError(f"Hasta {MAX_BOXES} recuadros por foto")
    keys = set(keys)
    out = []
    for b in raw:
        if not isinstance(b, dict):
            raise ObjectDataError("Un recuadro no tiene sus datos")
        k = b.get("k")
        if not (k in keys or (with_conf and k is None)):
            raise ObjectDataError("Un recuadro es de una clase que este objeto no tiene")
        vals = [_num(b.get(n)) for n in ("x1", "y1", "x2", "y2")]
        if any(v is None for v in vals):
            raise ObjectDataError("Un recuadro no tiene bien sus esquinas")
        x1, y1, x2, y2 = (min(1.0, max(0.0, v)) for v in vals)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        if x2 - x1 < MIN_BOX or y2 - y1 < MIN_BOX:
            continue  # un clic sin arrastrar: no es un recuadro
        box = {"k": k, "x1": round(x1, 4), "y1": round(y1, 4), "x2": round(x2, 4), "y2": round(y2, 4)}
        if with_conf:
            p = _num(b.get("p"))
            box["p"] = round(min(1.0, max(0.0, p if p is not None else 0.0)), 3)
        out.append(box)
    return out


def doubt_of(proposed: list[dict]) -> float:
    """Qué tan dudosa es una foto para la IA (0 a 1): 1 si algún recuadro propuesto está al 50 % de confianza. Las
    dudosas van primero al marcar (aprende más con lo que no sabe)."""
    best = 0.0
    for b in proposed or []:
        p = b.get("p")
        if isinstance(p, (int, float)):
            best = max(best, 1.0 - abs(float(p) - 0.5) * 2)
    return round(best, 3)


def yolo_lines(boxes: list[dict], keys: list[str]) -> list[str]:
    """Las líneas del archivo de etiquetas de YOLO: «clase cx cy ancho alto» (fracciones)."""
    idx = {k: i for i, k in enumerate(keys)}
    lines = []
    for b in boxes or []:
        if b.get("k") not in idx:
            continue
        cx, cy = (b["x1"] + b["x2"]) / 2, (b["y1"] + b["y2"]) / 2
        w, h = b["x2"] - b["x1"], b["y2"] - b["y1"]
        lines.append(f"{idx[b['k']]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


# ------------------------------------------------------------------ fotos
def image_uid(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def split_of(uid: str) -> str:
    """«prueba» (1 de cada 5, siempre la misma para la misma foto) o «entrenar». Con las fotos de prueba se mide qué
    tan bien anduvo: la IA no las ve al entrenar."""
    return "prueba" if int(hashlib.sha1(uid.encode()).hexdigest()[:8], 16) % TEST_SHARE == 0 else "entrenar"


# ------------------------------------------------------------------ ¿se puede entrenar?
def train_check(marked: int, boxes: dict[str, int], classes: list[dict], test_marked: int | None = None) -> dict:
    """Si alcanza para entrenar. `marked` = fotos marcadas (también las «no hay ninguno»), `boxes` = recuadros por clase.
    Devuelve {"ok", "problems" (lo que falta: no deja entrenar), "warnings" (deja, pero avisa), "tips"}."""
    names = class_names(classes)
    problems, warnings, tips = [], [], []
    if marked < MIN_PHOTOS:
        problems.append(f"Faltan fotos marcadas: hay {marked} y se necesitan al menos {MIN_PHOTOS}.")
    if not any(boxes.get(k, 0) >= MIN_BOXES for k in names):
        problems.append(f"Ninguna clase tiene todavía {MIN_BOXES} recuadros marcados (es lo mínimo para que la IA "
                        f"aprenda algo).")
    for k, n in names.items():
        c = boxes.get(k, 0)
        if c < MIN_BOXES and any(boxes.get(x, 0) >= MIN_BOXES for x in names):
            warnings.append(f"«{n}» tiene {c} recuadro{'s' if c != 1 else ''}: con menos de {MIN_BOXES} casi no la "
                            f"va a reconocer.")
    if test_marked is not None and marked >= MIN_PHOTOS and test_marked < MIN_TEST_PHOTOS:
        warnings.append(f"Solo {test_marked} de las fotos marcadas quedaron para probar: el resultado va a ser poco "
                        f"confiable. Marcá más fotos.")
    if marked < GOOD_PHOTOS or any(boxes.get(k, 0) < GOOD_BOXES for k in names):
        tips.append(f"Para que ande bien: unas {GOOD_PHOTOS} fotos marcadas y {GOOD_BOXES} recuadros de cada clase, "
                    f"de distintos momentos (día, noche, lleno, vacío, con gente adelante).")
    return {"ok": not problems, "problems": problems, "warnings": warnings, "tips": tips}


def auto_epochs(n_images: int) -> int:
    """Cuántas vueltas sobre las fotos: más con pocas fotos (cada vuelta es corta). Corta sola si deja de mejorar."""
    if n_images < 150:
        return 120
    if n_images < 500:
        return 100
    if n_images < 1500:
        return 80
    return 60


# ------------------------------------------------------------------ resultado en palabras
def grade(map50: float | None) -> tuple[str, str]:
    """(nota, qué hacer) según el mAP50 en las fotos de prueba."""
    if map50 is None:
        return "Sin medir", "No había fotos de prueba marcadas para medirlo."
    if map50 >= 0.85:
        return "Muy bien", "Ya se puede usar. Igual conviene probarlo con fotos nuevas."
    if map50 >= 0.70:
        return "Bien", "Se puede usar; con más fotos marcadas (sobre todo de lo que confunde) mejora."
    if map50 >= 0.50:
        return "Regular", "Todavía se equivoca seguido: marcá más fotos, sobre todo distintas entre sí, y volvé a entrenar."
    return "Todavía no sirve", "Hacen falta muchas más fotos marcadas (o las clases se parecen demasiado entre sí)."


def _of10(x: float | None) -> str:
    return "—" if x is None else str(int(round(max(0.0, min(1.0, x)) * 10)))


def metrics_text(metrics: dict, classes: list[dict]) -> list[str]:
    """Renglones para la app: la nota y, por clase, cuántos encuentra y cuántos acierta de cada 10."""
    if not metrics:
        return []
    note, todo = grade(metrics.get("map50"))
    lines = [f"Resultado: {note}. {todo}"]
    n_test = metrics.get("test_images")
    if n_test:
        lines.append(f"Se probó con {n_test} foto{'s' if n_test != 1 else ''} que la IA no vio al entrenar.")
    per = metrics.get("per_class") or {}
    for c in classes:
        m = per.get(c["key"])
        if not m:
            continue
        lines.append(f"«{c['name']}»: encuentra {_of10(m.get('recall'))} de cada 10; de cada 10 que marca, "
                     f"{_of10(m.get('precision'))} están bien.")
    return lines


# ------------------------------------------------------------------ detector propio (ONNX) y fotos de un video
class ObjectDetector:
    """Un modelo propio exportado a ONNX (salida (1, 4+clases, N), como el de personas). `keys` en el orden en que se
    entrenó. `detect` devuelve recuadros en fracciones de la imagen con su confianza."""

    def __init__(self, onnx_path, keys: list[str], conf: float = DEFAULT_CONF, iou: float = 0.5, threads: int = 2):
        from .detectors import OnnxSession, OnnxYoloDetector

        self.keys = list(keys)
        sess = OnnxSession(onnx_path, threads=threads, max_batch=1)  # de a una foto (nunca el modelo de lotes)
        self.det = OnnxYoloDetector(classes={i: k for i, k in enumerate(self.keys)}, session=sess, conf=conf, iou=iou)
        self.device = self.det.device

    def detect(self, frame, conf: float | None = None) -> list[dict]:
        h, w = frame.shape[:2]
        old = self.det.conf
        if conf is not None:
            self.det.conf = conf
        try:
            dets = self.det.detect(frame)
        finally:
            self.det.conf = old
        boxes = [{"k": d.label, "x1": round(d.bbox[0] / w, 4), "y1": round(d.bbox[1] / h, 4),
                  "x2": round(d.bbox[2] / w, 4), "y2": round(d.bbox[3] / h, 4), "p": round(d.conf, 3)}
                 for d in dets]
        return one_class_each(boxes)


def iou(a: dict, b: dict) -> float:
    ix = max(0.0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))
    iy = max(0.0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))
    inter = ix * iy
    union = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"]) + (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]) - inter
    return inter / union if union > 0 else 0.0


def one_class_each(boxes: list[dict], thr: float = 0.7) -> list[dict]:
    """Un objeto, una clase: si el modelo ve el mismo pallet como «vacío» y como «con film», queda la más segura (si no,
    se contaría dos veces)."""
    kept: list[dict] = []
    for b in sorted(boxes, key=lambda x: -x.get("p", 0)):
        if all(iou(b, k) < thr for k in kept):
            kept.append(b)
    return kept


def fit_photo(frame):
    """La foto con el lado mayor hasta IMG_MAX_SIDE (lo que se guarda)."""
    import cv2

    h, w = frame.shape[:2]
    s = IMG_MAX_SIDE / max(h, w)
    if s < 1:
        frame = cv2.resize(frame, (max(1, int(round(w * s))), max(1, int(round(h * s)))), interpolation=cv2.INTER_AREA)
    return frame


def encode_photo(frame) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", fit_photo(frame), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ObjectDataError("No se pudo guardar la foto")
    return buf.tobytes()


def read_photo(path):
    """Una foto del disco (también con tildes en la ruta, en Windows). None si no es una imagen."""
    import cv2
    import numpy as np

    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if not data.size:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def video_frames(path, every_s: float = VIDEO_EVERY_S, max_images: int = VIDEO_MAX_IMAGES,
                 min_change: float = VIDEO_MIN_CHANGE, progress=None, stop=None):
    """Fotos de un video: como mucho una cada `every_s` segundos, salteando las que casi no cambian (una cámara quieta
    daría cientos de fotos iguales, que no enseñan nada) y hasta `max_images`. Genera (segundo, cuadro BGR con píxeles
    cuadrados). `progress(fracción)` se llama de vez en cuando; `stop()` True corta."""
    import cv2
    import numpy as np

    from .imagen import pixel_aspect, square_pixels

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ObjectDataError("No se pudo abrir el video (probá exportarlo como MP4)")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        fps = fps if 1 <= fps <= 120 else 25.0
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        sar = pixel_aspect(cap)
        n, kept, nxt, last = 0, 0, 0.0, None
        while kept < max_images:
            if stop is not None and stop():
                return
            if not cap.grab():
                break
            ts = n / fps
            n += 1
            if progress is not None and total > 0 and n % 250 == 0:
                progress(min(1.0, n / total))
            if ts + 1e-6 < nxt:
                continue
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                continue
            if sar:
                frame = square_pixels(frame, sar)
            small = cv2.cvtColor(cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            small = small.astype(np.float32)
            nxt = ts + every_s
            if last is not None and float(np.mean(np.abs(small - last))) < min_change:
                continue
            last = small
            kept += 1
            yield round(ts, 2), frame
    finally:
        cap.release()
