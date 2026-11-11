"""«Entrenar IA hurtos» (pestaña de la app, agent/entrenar_ia.py): analizar un clip EN ESTA PC y guardar lo que salió.

Pedido del dueño (9 nov): «subiremos clips y pondremos si hubo hurto y dónde; aplicarle la IA de YOLO y la postura a
estos videos y ahí entrenaremos a la IA». El video NO sale de la PC: a la nube va solo el análisis (recuadros,
esqueletos, bolsos y los rasgos por ventana que arma core/hurto_ia.py).

- `analyze`: lee el video a 5 cuadros por segundo (por el reloj del video, como «🎞 Clips»), con píxeles cuadrados
  («1080p lite»), y en cada cuadro: personas con el YOLO de la app y la sensibilidad de la cámara
  (core/detectors.tuned), bolsos en la misma pasada (core/pose.detect_with_bags), seguimiento (core/tracker) y la
  postura de TODAS las personas del alto mínimo de los gestos (core/pose.look; aunque la cámara no tenga gestos).
  Devuelve los registros de cuadro de core/hurto_ia.frame_record (los mismos que arman las máquinas en vivo):
  `{"t", "w", "h", "people": [{"id", "box", "kp"}], "bags"}`, en píxeles del cuadro analizado.
- `clip_uuid`: el mismo video da el mismo identificador en cualquier PC (sha1 del tamaño, el primer y el último MB):
  subirlo dos veces actualiza el mismo clip.
- `LocalClips`: en `<datos>/entrenamiento/` quedan la ruta del video de cada clip (`videos.json`) y sus cuadros
  (`<uuid>.json.gz`, lo mismo que se sube a la nube): así se dibuja el esqueleto sin volver a analizar.

Nada de esto importa el servidor (usa OpenCV).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import cv2

from core.deteccion import DetectionConfig
from core.detectors import tuned
from core.gestos import GestureConfig
from core.imagen import pixel_aspect, square_pixels
from core.pose import detect_with_bags, look
from core.tracker import CentroidTracker

log = logging.getLogger("vigia.ia_clips")

FPS = 5.0  # cuadros por segundo que se analizan (como las máquinas con hurto)
MAX_CLIP_S = 1800.0  # tope por clip (30 min; core.hurto_ia.MAX_CLIP_S)
HASH_CHUNK = 1024 * 1024
DATA_VERSION = 1  # formato del .json.gz (los cuadros)


class ClipError(Exception):
    """El clip no se pudo analizar (no abre, no tiene cuadros…): el mensaje es para el usuario."""


def max_clip_seconds() -> float:
    try:
        from core.hurto_ia import MAX_CLIP_S as m
        return float(m)
    except Exception:  # noqa: BLE001 - (core/hurto_ia.py todavía no está o es de otra versión)
        return MAX_CLIP_S


def clip_uuid(path: str | Path) -> str:
    """Identificador del video: sha1 del tamaño + el primer y el último MB (40 letras). Leer el archivo entero de un
    clip de 30 min tardaría; así es instantáneo y dos copias del mismo video dan lo mismo."""
    p = Path(path)
    size = p.stat().st_size
    h = hashlib.sha1(str(size).encode())
    with open(p, "rb") as fh:
        h.update(fh.read(HASH_CHUNK))
        if size > HASH_CHUNK:
            fh.seek(max(HASH_CHUNK, size - HASH_CHUNK))
            h.update(fh.read(HASH_CHUNK))
    return h.hexdigest()


def analyze(path: str | Path, detector, pose, fps: float = FPS, detection: DetectionConfig | dict | None = None,
            stop: threading.Event | None = None, progress: Callable[[float, float], None] | None = None,
            max_seconds: float | None = None, on_frame: Callable[[dict], None] | None = None) -> dict:
    """Analiza el video `path` en esta PC. `detector`: el YOLO compartido de la app (con `detection`, la sensibilidad
    y «Personas lejos» de la cámara, sin tocar el compartido); `pose`: el estimador de postura (None = sin
    esqueletos). `stop`: frena (ClipError «se frenó»). `progress(parte_hecha, segundos)`. `on_frame(registro)`: cada
    cuadro apenas se analiza (el reproductor de «Alertas» dibuja lo que ya está, sin esperar al final).

    Devuelve `{"frames": [registros], "duration_s", "fps", "width", "height", "people", "video_fps", "truncated",
    "posed"}`: `people` = cuántas personas distintas se siguieron; `posed` = en cuántos registros de persona se vio la
    postura. ClipError si no se puede abrir o no tiene cuadros."""
    stop = stop or threading.Event()
    limit = max_seconds if max_seconds is not None else max_clip_seconds()
    det = tuned(detector, detection if isinstance(detection, DetectionConfig) else DetectionConfig.from_dict(detection))
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise ClipError("no se pudo abrir (si es .dav o .264, exportalo del grabador como MP4 o AVI)")
    from core.hurto_ia import frame_record  # (el registro de cuadro es el mismo que usan las máquinas en vivo)

    from .worker import MediaClock
    gest = GestureConfig()
    frames: list[dict] = []
    ids: set[int] = set()
    posed = 0
    truncated = False
    size = (0, 0)
    try:
        n, vfps = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0, cap.get(cv2.CAP_PROP_FPS) or 0
        dur = n / vfps if n > 0 and 1 <= vfps <= 120 else None
        sar = pixel_aspect(cap)
        clock, last, ts = MediaClock(cap), -1e9, 0.0
        tracker = CentroidTracker()
        step = 1.0 / max(1.0, float(fps))
        while True:
            if stop.is_set():
                raise ClipError("se frenó")
            if not cap.grab():
                break
            ts = clock.tick()
            if ts - last < step - 1e-3:
                continue
            if ts > limit:
                truncated = True
                break
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                continue
            if sar:
                frame = square_pixels(frame, sar)
            last = ts
            h, w = frame.shape[:2]
            size = (w, h)
            dets, bags = detect_with_bags(det, frame, True)
            tracks = tracker.update(dets, ts, frame)
            if pose is not None:  # todas las personas del alto mínimo de los gestos (el mismo que en las máquinas)
                look(pose, frame, tracks, gest.min_height(h))
            else:
                for t in tracks:
                    t.kpts = None
            rec = frame_record(ts, (w, h), tracks, bags)
            frames.append(rec)
            if on_frame is not None:
                on_frame(rec)
            ids.update(p["id"] for p in rec["people"])
            posed += sum(1 for p in rec["people"] if p["kp"] is not None)
            if progress is not None:
                progress(min(ts / dur, 1.0) if dur else 0.0, ts)
    finally:
        cap.release()
    if not frames:
        raise ClipError("no tiene cuadros que se puedan leer")
    duration = max(ts, frames[-1]["t"]) if not truncated else frames[-1]["t"]
    if dur and not truncated:
        duration = max(duration, min(dur, limit))
    return {"frames": frames, "duration_s": round(duration, 2), "fps": float(fps), "width": size[0],
            "height": size[1], "people": len(ids), "video_fps": round(float(vfps), 2) if vfps else None,
            "truncated": truncated, "posed": posed}


def pack(meta: dict, frames: list[dict]) -> bytes:
    """El .json.gz de un clip: los cuadros y unos datos (lo que se sube a la nube y queda en esta PC)."""
    body = {"version": DATA_VERSION, **meta, "frames": frames}
    return gzip.compress(json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8"), 6)


def unpack(data: bytes) -> dict:
    """Lee un .json.gz de cuadros (también una lista sola de cuadros, o ya descomprimido: un almacenamiento que lo
    sirve con «Content-Encoding: gzip» llega descomprimido). ValueError si no se entiende."""
    try:
        raw = gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data
        body = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError, EOFError) as exc:
        raise ValueError(f"los datos del clip no se pueden leer ({exc})") from exc
    if isinstance(body, list):
        body = {"frames": body}
    if not isinstance(body, dict) or not isinstance(body.get("frames"), list):
        raise ValueError("los datos del clip no tienen cuadros")
    return body


class LocalClips:
    """Lo que queda en esta PC de cada clip de entrenamiento (`<datos>/entrenamiento/`): la ruta del video
    (`videos.json`, por uuid) y sus cuadros (`<uuid>.json.gz`)."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.file = self.folder / "videos.json"
        self._lock = threading.Lock()
        self._data: dict[str, dict] = {}
        try:
            if self.file.exists():
                self._data = json.loads(self.file.read_text(encoding="utf-8")).get("videos", {})
        except (OSError, ValueError) as exc:
            log.warning("No se pudo leer la lista de videos de entrenamiento (%s): empieza vacía", exc)

    def path_of(self, uuid: str | None) -> str | None:
        """La ruta del video si está en esta PC (y el archivo sigue ahí)."""
        with self._lock:
            d = self._data.get(uuid or "")
        p = (d or {}).get("path")
        return p if p and Path(p).is_file() else None

    def known(self, uuid: str | None) -> bool:
        with self._lock:
            return (uuid or "") in self._data

    def link(self, uuid: str, path: str | Path) -> None:
        with self._lock:
            self._data[uuid] = {"path": os.path.abspath(str(path)), "added": datetime.now().isoformat(timespec="seconds")}
        self._save()

    def forget(self, uuid: str) -> None:
        with self._lock:
            self._data.pop(uuid, None)
        self._save()
        try:
            self.data_file(uuid).unlink(missing_ok=True)
        except OSError:
            pass

    def data_file(self, uuid: str) -> Path:
        return self.folder / f"{uuid}.json.gz"

    def save_data(self, uuid: str, data: bytes) -> None:
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            tmp = self.data_file(uuid).with_suffix(".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, self.data_file(uuid))
        except OSError as exc:
            log.warning("No se pudieron guardar los cuadros del clip en esta PC: %s", exc)

    def load_data(self, uuid: str) -> dict | None:
        f = self.data_file(uuid)
        if not f.is_file():
            return None
        try:
            return unpack(f.read_bytes())
        except (OSError, ValueError) as exc:
            log.warning("Los cuadros guardados del clip %s no se pueden leer (%s)", uuid[:8], exc)
            return None

    def _save(self) -> None:
        with self._lock:
            text = json.dumps({"videos": self._data}, ensure_ascii=False, indent=1)
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            tmp = self.file.with_suffix(".tmp")
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, self.file)
        except OSError as exc:
            log.warning("No se pudo guardar la lista de videos de entrenamiento: %s", exc)
