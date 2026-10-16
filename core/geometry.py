from __future__ import annotations

import math
from typing import Sequence

BBox = tuple[float, float, float, float]  # x1, y1, x2, y2
Point = tuple[float, float]


def point_in_polygon(x: float, y: float, polygon: Sequence[Point]) -> bool:
    """Ray casting. Polígono como lista de (x, y); el último vértice conecta con el primero."""
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def bbox_foot(b: BBox) -> Point:
    """Punto de apoyo de la persona: centro del borde inferior."""
    return ((b[0] + b[2]) / 2.0, b[3])


def bbox_center(b: BBox) -> Point:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def bbox_size(b: BBox) -> tuple[float, float]:
    return (b[2] - b[0], b[3] - b[1])


def iou(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def distance(p: Point, q: Point) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])
