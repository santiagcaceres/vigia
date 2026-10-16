"""Análisis de una cámara: detección -> seguimiento -> reglas -> hallazgos con foto y clip.

Lo usan el agente (video en vivo) y el servidor (videos importados para entrenar).
"""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np

from .detectors import Detector
from .rules import RuleEngine, RuleEvent, Zone, parse_config
from .scorer import Scorer
from .tracker import CentroidTracker, Track

ZONE_COLORS = {"shelf": (0, 200, 255), "checkout": (0, 200, 0), "exit": (255, 150, 0), "restricted": (0, 0, 255)}
ASCII_LABELS = {"salida_sin_pago": "SALIDA SIN PAGO", "merodeo": "MERODEO",
                "zona_restringida": "ZONA RESTRINGIDA", "fuera_de_horario": "FUERA DE HORARIO"}


@dataclass
class Finding:
    uuid: str
    type: str
    severity: str
    zone: str | None
    description: str
    track_id: int
    ts: float  # segundos desde el inicio del stream o del video
    wall_time: datetime | None
    bbox: tuple[float, float, float, float]
    frame_size: tuple[int, int]
    features: dict[str, float]
    score: float | None
    alert: bool
    snapshot_jpg: bytes

    def meta(self) -> dict:
        return {
            "uuid": self.uuid, "type": self.type, "severity": self.severity, "zone": self.zone,
            "description": self.description, "track_id": self.track_id, "video_ts": round(self.ts, 3),
            "occurred_at": self.wall_time.isoformat() if self.wall_time else None,
            "bbox": [round(v, 1) for v in self.bbox], "frame_size": list(self.frame_size),
            "features": self.features, "score": self.score, "alert": self.alert,
        }


@dataclass
class _PendingClip:
    uuid: str
    frames: list[tuple[float, np.ndarray]]
    end_ts: float


@dataclass
class ClipReady:
    uuid: str
    frames: list[tuple[float, np.ndarray]] = field(repr=False)


def annotate(frame: np.ndarray, zones: list[Zone], tracks: list[Track], event: RuleEvent | None = None) -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]
    overlay = img.copy()
    polys = []
    for z in zones:
        pts = np.array([[int(x * w), int(y * h)] for x, y in z.polygon], np.int32)
        polys.append((z, pts))
        cv2.fillPoly(overlay, [pts], ZONE_COLORS.get(z.type, (200, 200, 200)))
    img = cv2.addWeighted(overlay, 0.18, img, 0.82, 0)
    thick = max(1, w // 640)
    for z, pts in polys:
        color = ZONE_COLORS.get(z.type, (200, 200, 200))
        cv2.polylines(img, [pts], True, color, 2 * thick)
        cv2.putText(img, z.name, (int(pts[:, 0].min()) + 4, int(pts[:, 1].min()) + 16 * thick),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * thick, color, thick, cv2.LINE_AA)
    for t in tracks:
        if not t.visible:
            continue
        hot = event is not None and t.id == event.track_id
        color = (0, 0, 255) if hot else (0, 255, 0)
        x1, y1, x2, y2 = (int(v) for v in t.bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, (3 if hot else 2) * thick)
        cv2.putText(img, f"#{t.id}", (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * thick, color,
                    thick, cv2.LINE_AA)
    if event is not None:
        bar = 26 * thick
        cv2.rectangle(img, (0, 0), (w, bar), (0, 0, 160), -1)
        cv2.putText(img, ASCII_LABELS.get(event.type, event.type), (8, int(bar * 0.7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * thick, (255, 255, 255), 2, cv2.LINE_AA)
    return img


class Analyzer:
    def __init__(self, config: dict, detector: Detector, scorer: Scorer | None = None,
                 clip_pre: float = 6.0, clip_post: float = 6.0, clip_width: int = 960):
        self.zones, rule_cfg = parse_config(config)
        self.detector = detector
        self.tracker = CentroidTracker()
        self.engine = RuleEngine(self.zones, rule_cfg)
        self.scorer = scorer
        self.clip_pre = clip_pre
        self.clip_post = clip_post
        self.clip_width = clip_width
        self.frames = 0
        self._buffer: deque[tuple[float, np.ndarray]] = deque()
        self._pending: list[_PendingClip] = []
        self._ready: list[ClipReady] = []
        self.last_tracks: list[Track] = []

    def process(self, frame: np.ndarray, ts: float, wall: datetime | None = None) -> list[Finding]:
        h, w = frame.shape[:2]
        tracks = self.tracker.update(self.detector.detect(frame), ts)
        self.last_tracks = tracks
        small = self._small(frame)
        self._buffer.append((ts, small))
        while self._buffer and ts - self._buffer[0][0] > self.clip_pre:
            self._buffer.popleft()
        still = []
        for p in self._pending:
            if ts <= p.end_ts:
                p.frames.append((ts, small))
                still.append(p)
            else:
                self._ready.append(ClipReady(p.uuid, p.frames))
        self._pending = still

        findings = []
        for ev in self.engine.update(tracks, ts, (w, h), wall):
            score = self.scorer.score(ev.type, ev.features) if self.scorer else None
            alert = self.scorer.is_alert(score) if self.scorer and score is not None else True
            ok, jpg = cv2.imencode(".jpg", annotate(frame, self.zones, tracks, ev), [cv2.IMWRITE_JPEG_QUALITY, 85])
            f = Finding(uuid.uuid4().hex, ev.type, ev.severity, ev.zone, ev.description, ev.track_id, ts, wall,
                        ev.bbox, (w, h), ev.features, None if score is None else round(score, 4), alert,
                        jpg.tobytes() if ok else b"")
            self._pending.append(_PendingClip(f.uuid, list(self._buffer), ts + self.clip_post))
            findings.append(f)
        self.frames += 1
        return findings

    def take_ready_clips(self) -> list[ClipReady]:
        ready, self._ready = self._ready, []
        return ready

    def flush(self) -> list[ClipReady]:
        """Al terminar el video: entrega también los clips que no llegaron a completarse."""
        self._ready += [ClipReady(p.uuid, p.frames) for p in self._pending]
        self._pending = []
        return self.take_ready_clips()

    def _small(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if w <= self.clip_width:
            return frame.copy()
        return cv2.resize(frame, (self.clip_width, int(h * self.clip_width / w)), interpolation=cv2.INTER_AREA)
