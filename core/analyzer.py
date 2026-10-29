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
from .gestos import MANO_BOLSA, MANO_BOLSILLO, MANO_ROPA
from .pose import detect_with_bags, look
from .pose import draw as draw_pose
from .rules import RuleEngine, RuleEvent, Zone, parse_config
from .scorer import Scorer
from .tracker import CentroidTracker, Track

ZONE_COLORS = {"shelf": (0, 200, 255), "checkout": (0, 200, 0), "exit": (255, 150, 0), "restricted": (0, 0, 255),
               "entrance": (255, 0, 200), "pump": (0, 120, 255), "shop": (200, 200, 0), "staff": (140, 140, 140),
               "aisle": (255, 255, 120)}
ASCII_LABELS = {"salida_sin_pago": "SALIDA SIN PAGO", "merodeo": "MERODEO",
                "zona_restringida": "ZONA RESTRINGIDA", "fuera_de_horario": "FUERA DE HORARIO",
                MANO_BOLSILLO: "POSIBLE OCULTAMIENTO: MANO AL BOLSILLO", MANO_ROPA: "POSIBLE OCULTAMIENTO: MANO EN LA ROPA",
                MANO_BOLSA: "POSIBLE OCULTAMIENTO: MANO EN LA BOLSA"}


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
    frames: list[tuple[float, bytes]]  # (ts, JPEG)
    start_ts: float
    end_ts: float
    track_id: int
    max_end_ts: float


@dataclass
class ClipReady:
    """Clip listo: cuadros JPEG (ts, bytes) y el tramo que cubre, en segundos del stream."""
    uuid: str
    frames: list[tuple[float, bytes]] = field(repr=False)
    start_ts: float = 0.0
    end_ts: float = 0.0

    def decoded(self):
        """Cuadros decodificados de a uno (para no tener 3 minutos de video en memoria)."""
        for ts, jpg in self.frames:
            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                yield ts, img


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
        color = (0, 0, 255) if hot else ((255, 200, 0) if t.label == "vehiculo" else (0, 255, 0))
        x1, y1, x2, y2 = (int(v) for v in t.bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, (3 if hot else 2) * thick)
        if getattr(t, "kpts", None) is not None:
            draw_pose(img, t.kpts, (0, 220, 255) if not hot else (0, 140, 255))
        cv2.putText(img, f"#{t.id}", (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * thick, color,
                    thick, cv2.LINE_AA)
    if event is not None:
        bar = 26 * thick
        cv2.rectangle(img, (0, 0), (w, bar), (0, 0, 160), -1)
        cv2.putText(img, ASCII_LABELS.get(event.type, event.type), (8, int(bar * 0.7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * thick, (255, 255, 255), 2, cv2.LINE_AA)
    return img


class Analyzer:
    """Detecta, sigue, aplica reglas y arma los clips.

    El clip de una alerta cubre **toda la estadía de la persona**: arranca unos segundos antes de
    que apareciera (`clip_pre` como mínimo) y sigue mientras se la vea, hasta `clip_post` después
    de que se fue, con tope `clip_max`. Para poder volver hasta 3 minutos atrás sin gastar memoria,
    el buffer guarda los cuadros ya comprimidos en JPEG (640 px, ~40 KB: 3 min a 5 fps ≈ 36 MB).
    """

    def __init__(self, config: dict, detector: Detector, scorer: Scorer | None = None,
                 clip_pre: float = 6.0, clip_post: float = 6.0, clip_width: int = 640, clip_max: float = 180.0,
                 lead_seconds: float = 3.0, clips: bool = True, pose=None):
        self.zones, rule_cfg = parse_config(config)
        self.clips = clips  # False = solo métricas: sin buffer de cuadros (no comprime JPEG ni gasta memoria)
        self.detector = detector
        self.pose = pose  # core/pose.PoseEstimator: postura para los gestos (None = sin gestos)
        self.tracker = CentroidTracker()
        self.engine = RuleEngine(self.zones, rule_cfg)
        self.scorer = scorer
        self.clip_pre = clip_pre
        self.clip_post = clip_post
        self.clip_width = clip_width
        self.clip_max = max(clip_max, clip_pre + clip_post)
        self.lead_seconds = lead_seconds
        self.frames = 0
        self._buffer: deque[tuple[float, bytes]] = deque()
        self._pending: list[_PendingClip] = []
        self._ready: list[ClipReady] = []
        self.last_tracks: list[Track] = []

    @property
    def gestures(self) -> bool:
        """Se buscan gestos: hay modelo de postura y la cámara los tiene activados."""
        return self.pose is not None and self.engine.gestures is not None and self.clips

    def process(self, frame: np.ndarray, ts: float, wall: datetime | None = None) -> list[Finding]:
        h, w = frame.shape[:2]
        dets, objects = detect_with_bags(self.detector, frame, self.gestures)
        tracks = self.tracker.update(dets, ts, frame)
        self.last_tracks = tracks
        if not self.clips:  # solo métricas: no hay alertas ni clips
            self.frames += 1
            return []
        if self.gestures:
            look(self.pose, frame, tracks, self.engine.cfg.gestures.min_height(h))
        else:  # sin gestos (o recién apagados) no queda un esqueleto viejo dibujado
            for t in tracks:
                t.kpts = None
        jpg_small = self._small_jpg(frame)
        self._remember(ts, jpg_small)
        visible_ids = {t.id for t in tracks if t.visible}
        still = []
        for p in self._pending:
            back = bool(p.frames) and ts < p.frames[-1][0]  # el reloj volvió atrás: el clip se cierra como está
            if not back and p.track_id in visible_ids and ts + self.clip_post <= p.max_end_ts:
                p.end_ts = ts + self.clip_post  # la persona sigue ahí: el clip sigue
            if not back and ts <= p.end_ts:
                p.frames.append((ts, jpg_small))
                still.append(p)
            else:
                self._ready.append(ClipReady(p.uuid, p.frames, p.start_ts, p.frames[-1][0] if p.frames else ts))
        self._pending = still

        findings = []
        for ev in self.engine.update(tracks, ts, (w, h), wall, objects):
            score = self.scorer.score(ev.type, ev.features) if self.scorer else None
            alert = self.scorer.is_alert(score) if self.scorer and score is not None else True
            ok, jpg = cv2.imencode(".jpg", annotate(frame, self.zones, tracks, ev), [cv2.IMWRITE_JPEG_QUALITY, 85])
            f = Finding(uuid.uuid4().hex, ev.type, ev.severity, ev.zone, ev.description, ev.track_id, ts, wall,
                        ev.bbox, (w, h), ev.features, None if score is None else round(score, 4), alert,
                        jpg.tobytes() if ok else b"")
            # desde que la persona apareció (con un margen), nunca menos que clip_pre ni más que el tope
            back = max(self.clip_pre, ts - ev.track_start_ts + self.lead_seconds)
            start = ts - min(back, self.clip_max - self.clip_post)
            self._pending.append(_PendingClip(f.uuid, [fr for fr in self._buffer if fr[0] >= start], start,
                                              ts + self.clip_post, ev.track_id, start + self.clip_max))
            findings.append(f)
        self.frames += 1
        return findings

    @property
    def busy(self) -> bool:
        """Se ve a alguien (aunque esté quieto) o hay un clip abierto: no saltear cuadros."""
        return bool(self.tracker.seeing or self._pending)

    def idle(self, frame: np.ndarray, ts: float) -> None:
        """Cuadro sin movimiento y sin nadie: no se detecta. Solo se guarda 1 por segundo en el buffer
        para que el clip tenga los segundos de antes de que apareciera la persona."""
        if not self.clips or (self._buffer and ts - self._buffer[-1][0] < 1.0):
            return
        self._remember(ts, self._small_jpg(frame))

    def _remember(self, ts: float, jpg: bytes) -> None:
        if self._buffer and ts < self._buffer[-1][0]:  # el reloj volvió atrás: lo guardado ya no se puede recortar
            self._buffer.clear()
        self._buffer.append((ts, jpg))
        while self._buffer and ts - self._buffer[0][0] > self.clip_max:
            self._buffer.popleft()

    def take_ready_clips(self) -> list[ClipReady]:
        ready, self._ready = self._ready, []
        return ready

    def flush(self) -> list[ClipReady]:
        """Al terminar el video: entrega también los clips que no llegaron a completarse."""
        self._ready += [ClipReady(p.uuid, p.frames, p.start_ts, p.frames[-1][0] if p.frames else p.start_ts)
                        for p in self._pending]
        self._pending = []
        return self.take_ready_clips()

    def _small_jpg(self, frame: np.ndarray) -> bytes:
        ok, jpg = cv2.imencode(".jpg", self._small(frame), [cv2.IMWRITE_JPEG_QUALITY, 70])
        return jpg.tobytes() if ok else b""

    def _small(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if w <= self.clip_width:
            return frame.copy()
        return cv2.resize(frame, (self.clip_width, int(h * self.clip_width / w)), interpolation=cv2.INTER_AREA)
