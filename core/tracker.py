from __future__ import annotations

from dataclasses import dataclass

from typing import TYPE_CHECKING

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

    @property
    def visible(self) -> bool:
        """True si la persona fue vista en el último frame procesado."""
        return self.missed == 0


class CentroidTracker:
    """Seguimiento simple por distancia entre centros (asignación voraz).

    Suficiente para escenas con pocas personas. Para producción con mucha gente o
    cruces frecuentes conviene reemplazarlo por ByteTrack/BoT-SORT.
    """

    def __init__(self, gate: float = 1.0, max_missed: int = 20, min_hits: int = 3):
        self.gate = gate
        self.max_missed = max_missed
        self.min_hits = min_hits
        self._tracks: dict[int, Track] = {}
        self._next_id = 1

    def update(self, detections: list[Detection], ts: float) -> list[Track]:
        pairs: list[tuple[float, int, int]] = []
        for tid, t in self._tracks.items():
            tc = bbox_center(t.bbox)
            tw, th = bbox_size(t.bbox)
            limit = self.gate * max(tw, th, 1.0)
            for di, d in enumerate(detections):
                dist = distance(tc, bbox_center(d.bbox))
                if dist <= limit:
                    pairs.append((dist, tid, di))
        pairs.sort()

        used_t: set[int] = set()
        used_d: set[int] = set()
        for _dist, tid, di in pairs:
            if tid in used_t or di in used_d:
                continue
            used_t.add(tid)
            used_d.add(di)
            t = self._tracks[tid]
            d = detections[di]
            t.bbox = d.bbox
            t.label = d.label
            t.hits += 1
            t.missed = 0
            t.last_ts = ts

        for tid in list(self._tracks):
            if tid in used_t:
                continue
            t = self._tracks[tid]
            t.missed += 1
            confirmed = t.hits >= self.min_hits
            if t.missed > self.max_missed or (not confirmed and t.missed > 2):
                del self._tracks[tid]

        for di, d in enumerate(detections):
            if di in used_d:
                continue
            self._tracks[self._next_id] = Track(
                id=self._next_id, bbox=d.bbox, label=d.label, first_ts=ts, last_ts=ts
            )
            self._next_id += 1

        return [t for t in self._tracks.values() if t.hits >= self.min_hits]
