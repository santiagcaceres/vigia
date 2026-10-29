from __future__ import annotations

from dataclasses import dataclass, field

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .detectors import Detection
from .geometry import BBox, bbox_center, bbox_size, distance


@dataclass
class Track:
    id: int
    bbox: BBox
    label: str
    first_ts: float
    last_ts: float
    hits: int = 1
    missed: int = 0
    hist: np.ndarray | None = field(default=None, repr=False)  # colores de la ropa (para reconocerla)
    at_border: bool = False  # la última vez se la vio tocando el borde de la imagen (saliendo)
    kpts: np.ndarray | None = field(default=None, repr=False)  # postura (17x3) de este cuadro, si se miró

    @property
    def visible(self) -> bool:
        """True si la persona fue vista en el último frame procesado."""
        return self.missed == 0


def appearance(frame: np.ndarray, bbox: BBox) -> np.ndarray | None:
    """Histograma de color (tono y saturación) del cuerpo: alcanza para distinguir la ropa de dos personas."""
    import cv2

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(max(0, min(v, lim))) for v, lim in zip(bbox, (w, h, w, h)))
    # la parte central del recuadro (sin piso ni fondo de los costados)
    bw, bh = x2 - x1, y2 - y1
    if bw < 6 or bh < 12:
        return None
    crop = frame[y1 + bh // 8: y2 - bh // 8, x1 + bw // 6: x2 - bw // 6]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def similarity(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    import cv2

    return float(cv2.compareHist(a.astype(np.float32), b.astype(np.float32), cv2.HISTCMP_CORREL))


class CentroidTracker:
    """Seguimiento por distancia entre centros (asignación voraz), con dos ayudas para que la misma
    persona no cambie de número cuando el detector la pierde unos segundos (la tapa una góndola, se
    agacha, cambia la luz):

    1. **Memoria:** una persona perdida se recuerda `lost_seconds` (30 s). Si aparece alguien cerca de
       donde se la perdió, y no entrando por el borde de la imagen, es la misma: mantiene el número y el
       tiempo en escena. Cuanto más tiempo pasó, más lejos se acepta (puede haber caminado).
    2. **Ropa:** se compara el color de la ropa (histograma); dos personas distintas en el mismo lugar no
       se confunden si visten distinto.

    Las personas nuevas entran por el borde: alguien que se perdió tocando el borde se dio por ido.
    Para escenas con mucha gente o cruces frecuentes conviene ByteTrack/BoT-SORT con un modelo de re-ID.
    """

    RECENT_SECONDS = 1.5  # perdida hace menos que esto: seguimiento normal
    MIN_SIMILARITY = 0.35  # ropa parecida para reconocer a alguien perdido hace rato

    def __init__(self, gate: float = 1.0, max_missed: int = 20, min_hits: int = 3, lost_seconds: float = 30.0,
                 border: float = 0.04):
        self.gate = gate
        self.max_missed = max_missed
        self.min_hits = min_hits
        self.lost_seconds = lost_seconds
        self.border = border
        self._tracks: dict[int, Track] = {}
        self._next_id = 1

    @property
    def active(self) -> bool:
        """Hay alguien en seguimiento (visto o recordado)."""
        return bool(self._tracks)

    @property
    def seeing(self) -> bool:
        """Se vio a alguien en el último cuadro (aunque todavía no esté confirmado). A los que se recuerdan
        sin verlos no hace falta buscarlos en cada cuadro: si reaparecen, se mueven."""
        return any(t.missed == 0 for t in self._tracks.values())

    def _touches_border(self, bbox: BBox, size: tuple[int, int] | None) -> bool:
        if size is None:
            return False
        w, h = size
        mx, my = self.border * w, self.border * h
        x1, y1, x2, y2 = bbox
        return x1 <= mx or y1 <= my or x2 >= w - mx or y2 >= h - my

    def update(self, detections: list[Detection], ts: float, frame: np.ndarray | None = None) -> list[Track]:
        size = (frame.shape[1], frame.shape[0]) if frame is not None else None
        hists = [appearance(frame, d.bbox) if frame is not None else None for d in detections]
        at_border = [self._touches_border(d.bbox, size) for d in detections]
        used_t: set[int] = set()
        used_d: set[int] = set()

        def assign(pairs: list[tuple[float, int, int]]) -> None:
            for _cost, tid, di in sorted(pairs):
                if tid in used_t or di in used_d:
                    continue
                used_t.add(tid)
                used_d.add(di)
                t, d = self._tracks[tid], detections[di]
                t.bbox, t.label = d.bbox, d.label
                t.hits += 1
                t.missed = 0
                t.last_ts = ts
                t.at_border = at_border[di]
                if hists[di] is not None:  # la ropa se va actualizando de a poco
                    t.hist = hists[di] if t.hist is None else 0.8 * t.hist + 0.2 * hists[di]

        # 1) personas vistas hace poco: por cercanía, como siempre
        pairs = []
        for tid, t in self._tracks.items():
            if ts - t.last_ts > self.RECENT_SECONDS:
                continue
            tc = bbox_center(t.bbox)
            limit = self.gate * max(*bbox_size(t.bbox), 1.0)
            for di, d in enumerate(detections):
                if d.label != t.label:
                    continue  # una persona no se convierte en auto ni al revés
                dist = distance(tc, bbox_center(d.bbox))
                if dist <= limit:
                    pairs.append((dist / limit, tid, di))
        assign(pairs)

        # 2) personas perdidas hace rato: alguien que aparece cerca de donde se la perdió (y no entrando
        #    por el borde) es la misma, si la ropa no dice lo contrario
        pairs = []
        for tid, t in self._tracks.items():
            gap = ts - t.last_ts
            if tid in used_t or gap <= self.RECENT_SECONDS or t.hits < self.min_hits or t.at_border:
                continue
            tc = bbox_center(t.bbox)
            limit = self.gate * max(*bbox_size(t.bbox), 1.0) * min(1.0 + 0.3 * gap, 4.0)
            for di, d in enumerate(detections):
                if di in used_d or d.label != t.label or at_border[di]:
                    continue
                dist = distance(tc, bbox_center(d.bbox))
                if dist > limit:
                    continue
                sim = similarity(t.hist, hists[di])
                if sim is not None and sim < self.MIN_SIMILARITY:
                    continue
                pairs.append((dist / limit + (1.0 - (sim if sim is not None else 0.5)), tid, di))
        assign(pairs)

        for tid in list(self._tracks):
            if tid in used_t:
                continue
            t = self._tracks[tid]
            t.missed += 1
            t.kpts = None
            confirmed = t.hits >= self.min_hits
            gone = (not confirmed and t.missed > 2) \
                or (t.at_border and t.missed > self.max_missed) \
                or (ts - t.last_ts > self.lost_seconds and t.missed > self.max_missed)
            if gone:
                del self._tracks[tid]

        for di, d in enumerate(detections):
            if di in used_d:
                continue
            self._tracks[self._next_id] = Track(id=self._next_id, bbox=d.bbox, label=d.label, first_ts=ts, last_ts=ts,
                                                hist=hists[di], at_border=at_border[di])
            self._next_id += 1

        return [t for t in self._tracks.values() if t.hits >= self.min_hits]
