"""Análisis de un clip corto completo (el tramo de video alrededor de una alarma del DVR).

A diferencia del análisis en vivo, acá el clip ya está cortado: se recorre entero y se
resume en **un solo resultado**: si hubo personas, la regla más grave que se disparó (o
"alarma_dvr" si ninguna) y las medidas de la persona más interesante, para el modelo
entrenado con las etiquetas de los empleados.

Si en todo el clip no aparece ninguna persona, la alarma se descarta sin mostrarla a nadie:
es lo que más trabajo ahorra (el detector de movimiento del DVR salta con luces, sombras,
insectos, mascotas...).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import cv2
import numpy as np

from .analyzer import annotate
from .pose import detect_with_bags, look
from .detectors import Detector
from .rules import ALARMA_DVR, RULE_PRIORITY, SEVERITY, RuleEngine, RuleEvent, parse_config
from .scorer import Scorer
from .tracker import CentroidTracker, Track

# Cuántos cuadros con persona hacen falta para no descartar la alarma (evita un falso positivo suelto).
MIN_PERSON_FRAMES = 2


@dataclass
class ClipResult:
    frames: int = 0
    person_frames: int = 0
    max_people: int = 0
    duration_s: float = 0.0
    type: str | None = None
    severity: str | None = None
    description: str = ""
    zone: str | None = None
    track_id: int = 0
    ts: float = 0.0
    wall_time: datetime | None = None
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    frame_size: tuple[int, int] = (0, 0)
    features: dict[str, float] = field(default_factory=dict)
    rules: list[str] = field(default_factory=list)
    score: float | None = None
    alert: bool = True
    snapshot_jpg: bytes = field(default=b"", repr=False)

    @property
    def has_person(self) -> bool:
        return self.person_frames >= MIN_PERSON_FRAMES

    def stats(self) -> dict:
        return {"frames": self.frames, "person_frames": self.person_frames, "max_people": self.max_people,
                "duration_s": round(self.duration_s, 1), "rules": self.rules}

    def meta(self) -> dict:
        return {
            "type": self.type, "severity": self.severity, "zone": self.zone, "description": self.description,
            "track_id": self.track_id, "video_ts": round(self.ts, 3),
            "occurred_at": self.wall_time.isoformat() if self.wall_time else None,
            "bbox": [round(v, 1) for v in self.bbox], "frame_size": list(self.frame_size),
            "features": self.features, "score": self.score, "alert": self.alert,
        }


@dataclass
class _Moment:
    interest: float
    features: dict[str, float]
    ts: float
    bbox: tuple[float, float, float, float]
    frame: np.ndarray
    tracks: list[Track]
    event: RuleEvent | None = None


def _interest(f: dict[str, float]) -> float:
    """Qué persona del clip conviene mostrar y medir: la que más tiempo pasó en góndola o
    zona restringida; a igualdad, la que más tiempo estuvo en escena."""
    return f["shelf_total_s"] + 2.0 * f["restricted_total_s"] + 0.01 * f["track_age_s"]


def analyze_clip(video: str, config: dict, detector: Detector, scorer: Scorer | None = None,
                 process_fps: float = 5.0, start_wall: datetime | None = None, pose=None) -> ClipResult:
    """`pose`: core/pose.PoseEstimator para los gestos de ocultamiento (si la cámara los tiene activados)."""
    zones, rule_cfg = parse_config(config)
    engine = RuleEngine(zones, rule_cfg)
    gestures = pose is not None and engine.gestures is not None
    tracker = CentroidTracker()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el clip {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    fps = fps if 0 < fps < 120 else 25.0
    step = max(1, int(round(fps / max(process_fps, 0.5))))
    res = ClipResult()
    best: dict[int, _Moment] = {}  # por persona, su momento más interesante
    fired: _Moment | None = None  # la regla más grave que se disparó
    idx = 0
    try:
        while cap.grab():
            if idx % step:
                idx += 1
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            ts = idx / fps
            idx += 1
            h, w = frame.shape[:2]
            res.frame_size = (w, h)
            wall = start_wall + timedelta(seconds=ts) if start_wall else None
            dets, objects = detect_with_bags(detector, frame, gestures)
            res.frames += 1
            if dets:
                res.person_frames += 1
                res.max_people = max(res.max_people, len(dets))
            tracks = tracker.update(dets, ts, frame)
            if gestures:
                look(pose, frame, tracks, rule_cfg.gestures.min_person_px)
            events = engine.update(tracks, ts, (w, h), wall, objects)
            feats = engine.current_features(tracks, ts, (w, h), wall)
            snap = [dataclasses.replace(t) for t in tracks]
            for ev in events:
                if ev.type not in res.rules:
                    res.rules.append(ev.type)
                if fired is None or RULE_PRIORITY.index(ev.type) < RULE_PRIORITY.index(fired.event.type):
                    fired = _Moment(0.0, ev.features, ts, ev.bbox, frame.copy(), snap, ev)
            for t in tracks:
                f = feats.get(t.id)
                if f is None:
                    continue
                score = _interest(f)
                if t.id not in best or score >= best[t.id].interest:
                    best[t.id] = _Moment(score, f, ts, t.bbox, frame, snap)
    finally:
        cap.release()
    res.duration_s = idx / fps
    if not res.has_person:
        return res

    if fired is not None:
        m = fired
        res.type, res.zone, res.description = m.event.type, m.event.zone, m.event.description
        res.track_id = m.event.track_id
    elif best:
        tid, m = max(best.items(), key=lambda kv: kv[1].interest)
        f = m.features
        res.type, res.track_id = ALARMA_DVR, tid
        res.description = (f"Alarma con persona: {f['track_age_s']:.0f}s en escena, "
                           f"{f['shelf_total_s']:.0f}s en góndola, {f['checkout_total_s']:.0f}s en caja")
    else:  # hubo detecciones sueltas pero nunca una persona confirmada por el seguimiento
        res.type = ALARMA_DVR
        res.description = "Alarma con persona (aparece muy poco tiempo en el clip)"
        m = None
    res.severity = SEVERITY[res.type]
    if m is not None:
        res.features, res.ts, res.bbox = m.features, m.ts, tuple(m.bbox)
        res.wall_time = start_wall + timedelta(seconds=m.ts) if start_wall else None
        img = annotate(m.frame, zones, m.tracks, m.event)
    else:
        res.wall_time = start_wall
        img = _first_frame(video)
    if img is not None:
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        res.snapshot_jpg = jpg.tobytes() if ok else b""
    if res.features and scorer is not None:
        res.score = round(scorer.score(res.type, res.features), 4)
        res.alert = scorer.is_alert(res.score)
    return res


def _first_frame(video: str) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video))
    try:
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()
