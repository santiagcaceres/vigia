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


# ---------------------------------------------------------------------------------------------- línea de entrada
def line_side(a: Point, b: Point, p: Point) -> float:
    """De qué lado de la línea a→b queda `p`: > 0 del lado hacia donde apunta su flecha (a la derecha de quien camina
    de `a` a `b` en la imagen, donde «y» crece hacia abajo: una línea marcada de izquierda a derecha apunta hacia
    abajo), < 0 del otro y 0 encima. Es un producto cruz: el signo no cambia con la escala de la imagen."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def line_distance(a: Point, b: Point, p: Point) -> float:
    """Distancia de `p` a la recta que pasa por `a` y `b`, con el signo de `line_side` (en las unidades de los
    puntos: pasarlos en píxeles para que la distancia sea de verdad perpendicular)."""
    length = math.hypot(b[0] - a[0], b[1] - a[1])
    return line_side(a, b, p) / length if length > 0 else 0.0


def crosses_segment(a: Point, b: Point, p: Point, q: Point, slack: float = 0.0) -> bool:
    """¿El camino recto de `p` a `q` (que quedaron de lados distintos de la recta) corta la línea entre sus puntas
    `a` y `b`? `slack`: cuánto vale la línea más allá de cada punta, como parte de su largo."""
    sp, sq = line_side(a, b, p), line_side(a, b, q)
    if sp == sq:
        return False
    k = sp / (sp - sq)
    x, y = p[0] + (q[0] - p[0]) * k, p[1] + (q[1] - p[1]) * k
    dx, dy = b[0] - a[0], b[1] - a[1]
    length2 = dx * dx + dy * dy
    if length2 <= 0:
        return False
    t = ((x - a[0]) * dx + (y - a[1]) * dy) / length2
    return -slack <= t <= 1.0 + slack


def line_arrow(a: Point, b: Point, w: float, h: float, length: float) -> tuple[Point, Point]:
    """La flecha de una línea de entrada, en píxeles de una imagen de `w` x `h`: (desde, hasta). Sale del medio de la
    línea, perpendicular a ella, hacia el lado de adentro (`line_side` > 0) y mide `length` píxeles."""
    ax, ay, bx, by = a[0] * w, a[1] * h, b[0] * w, b[1] * h
    dx, dy = bx - ax, by - ay
    norm = math.hypot(dx, dy) or 1.0
    mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
    return (mx, my), (mx - dy / norm * length, my + dx / norm * length)


def oriented_line(a: Point, b: Point, toward: Point = (0.0, 1.0), w: float = 1.0, h: float = 1.0) -> tuple[Point, Point]:
    """Las dos puntas de una línea de entrada en el orden que hace apuntar su flecha hacia `toward` (un vector de la
    imagen). Por defecto hacia abajo, que es hacia la cámara, y una línea vertical hacia la derecha."""
    nx, ny = -(b[1] - a[1]) * h, (b[0] - a[0]) * w  # hacia dónde apunta la flecha de a→b
    dot = nx * toward[0] + ny * toward[1]
    if dot < 0 or (dot == 0 and (ny < 0 or (ny == 0 and nx < 0))):
        return b, a
    return a, b


def segment_distance(p: Point, a: Point, b: Point) -> float:
    """Distancia de `p` al tramo entre `a` y `b`."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length2 = dx * dx + dy * dy
    t = 0.0 if length2 <= 0 else min(max(((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length2, 0.0), 1.0)
    return math.hypot(p[0] - (a[0] + dx * t), p[1] - (a[1] + dy * t))
