"""Mapa de pisadas de una cámara: por dónde camina y dónde se para la gente, y zonas sugeridas a partir de eso.

Las zonas se miran donde **pisa** cada persona (el medio del borde de abajo de su recuadro). Dibujarlas a ojo sobre
una foto falla seguido, sobre todo en cámaras de techo u ojo de pez: el piso no está donde parece. Con el mapa se
dibujan donde la gente de verdad se para (frente a una góndola, en la caja) y por donde camina (pasillos).

- `FootMap.build(frames)`: con los recuadros de cada cuadro ya analizado (lo que guarda «Ajustar…»), suma por celda
  los segundos de personas caminando y parados (parado = se mueve menos de `STILL_SPEED` del ancho por segundo), y
  dónde aparece y desaparece cada persona (para encontrar las puertas).
- `overlay`: pinta el mapa encima de la imagen (celeste = caminan, rojo = se paran).
- `suggest`: propone zonas: manchas donde la gente se para → «Góndola» (o caja: eso lo sabe quien conoce el local);
  lo que es solo paso → «Pasillo»; donde muchas personas aparecen o desaparecen pegadas al borde → «Entrada», con el
  sentido en que caminan al aparecer. Son una propuesta: se nombran, se corrigen y se prueban antes de guardar.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

GRID = (96, 54)  # celdas (ancho, alto): en 1920x1080, 20 px por celda
STILL_SPEED = 0.03  # moverse menos que esto (fracción del ancho de la imagen por segundo) = está parado
SPEED_WINDOW = 1.0  # la velocidad se mide contra dónde estaba hace 1 s (el recuadro tiembla de un cuadro a otro)
BORDER = 0.07  # aparecer o desaparecer a menos de esto del borde = entró o salió de la imagen
MIN_PERSON_SECONDS = 1.0  # recorridos más cortos: ruido del detector


@dataclass
class FootMap:
    walk: np.ndarray  # segundos-persona caminando por celda (alto, ancho)
    still: np.ndarray  # segundos-persona parados por celda
    seconds: float = 0.0  # cuánto video se miró
    persons: int = 0
    appear: list[tuple[float, float, float, float]] = field(default_factory=list)  # (x, y, dx, dy) al aparecer
    vanish: list[tuple[float, float, float, float]] = field(default_factory=list)  # (x, y, dx, dy) al irse
    passed: np.ndarray | None = None  # cuántas personas distintas pasaron caminando por cada celda

    @classmethod
    def build(cls, frames) -> "FootMap":
        """`frames`: [(ts, (ancho, alto), tracks)] en orden; los videos distintos van por separado (`merge`)."""
        gw, gh = GRID
        walk = np.zeros((gh, gw), np.float32)
        still = np.zeros((gh, gw), np.float32)
        last: dict[int, tuple[float, float, float]] = {}  # id -> (ts, x, y) del cuadro anterior
        back: dict[int, list[tuple[float, float, float]]] = {}  # id -> posiciones del último segundo
        paths: dict[int, list[tuple[float, float, float]]] = {}
        walked: dict[int, set[tuple[int, int]]] = {}
        t0 = t1 = None
        for ts, (w, h), tracks in frames:
            t0 = ts if t0 is None else t0
            t1 = ts
            for t in tracks:
                if not getattr(t, "visible", True) or getattr(t, "label", "persona") != "persona":
                    continue
                x1, y1, x2, y2 = t.bbox
                fx, fy = min(max((x1 + x2) / 2 / w, 0.0), 0.999), min(max(y2 / h, 0.0), 0.999)
                paths.setdefault(t.id, []).append((ts, fx, fy))
                prev = last.get(t.id)
                last[t.id] = (ts, fx, fy)
                hist = back.setdefault(t.id, [])
                hist.append((ts, fx, fy))
                while len(hist) > 2 and hist[1][0] <= ts - SPEED_WINDOW:
                    hist.pop(0)
                if prev is None:
                    continue
                dt = ts - prev[0]
                if dt <= 0 or dt > 2.0:
                    continue
                ref = hist[0]
                span = max(ts - ref[0], dt)
                speed = float(np.hypot(fx - ref[1], (fy - ref[2]) * h / w)) / span
                cell = (int(fy * gh), int(fx * gw))
                (still if speed < STILL_SPEED else walk)[cell] += dt
                if speed >= STILL_SPEED:  # las celdas entre los dos puntos: a pocos cuadros por segundo se saltean
                    steps = max(1, int(max(abs(fx - prev[1]) * gw, abs(fy - prev[2]) * gh)))
                    for k in range(steps + 1):
                        cx, cy = prev[1] + (fx - prev[1]) * k / steps, prev[2] + (fy - prev[2]) * k / steps
                        walked.setdefault(t.id, set()).add((int(cy * gh), int(cx * gw)))
        passed = np.zeros((gh, gw), np.float32)
        for tid, cells in walked.items():
            pts = paths.get(tid) or []
            if len(pts) >= 2 and pts[-1][0] - pts[0][0] >= MIN_PERSON_SECONDS:
                for c in cells:
                    passed[c] += 1
        fm = cls(walk, still, (t1 - t0) if t0 is not None else 0.0, passed=passed)
        for pts in paths.values():
            if len(pts) < 2 or pts[-1][0] - pts[0][0] < MIN_PERSON_SECONDS:
                continue
            fm.persons += 1
            fm.appear.append((pts[0][1], pts[0][2], *_heading(pts[: max(2, len(pts) // 4)])))
            fm.vanish.append((pts[-1][1], pts[-1][2], *_heading(pts[-max(2, len(pts) // 4):])))
        return fm

    def merge(self, other: "FootMap") -> "FootMap":
        return FootMap(self.walk + other.walk, self.still + other.still, self.seconds + other.seconds,
                       self.persons + other.persons, self.appear + other.appear, self.vanish + other.vanish,
                       _add(self.passed, other.passed))

    @classmethod
    def empty(cls) -> "FootMap":
        gw, gh = GRID
        return cls(np.zeros((gh, gw), np.float32), np.zeros((gh, gw), np.float32))


def _add(a: np.ndarray | None, b: np.ndarray | None) -> np.ndarray | None:
    if a is None or b is None:
        return a if b is None else b
    return a + b


def _heading(pts: list[tuple[float, float, float]]) -> tuple[float, float]:
    return pts[-1][1] - pts[0][1], pts[-1][2] - pts[0][2]


def overlay(image: np.ndarray, fm: FootMap, alpha: float = 0.6) -> np.ndarray:
    """La imagen con el mapa encima: celeste donde caminan, rojo donde se paran (más fuerte = más tiempo)."""
    h, w = image.shape[:2]
    out = image.copy()
    for grid, color in ((fm.walk, (255, 200, 0)), (fm.still, (0, 0, 255))):
        if grid.max() <= 0:
            continue
        g = cv2.GaussianBlur(grid, (0, 0), 1.0)
        g = np.sqrt(g / g.max())  # (que se vean también las zonas con poco tiempo)
        a = cv2.resize(g, (w, h), interpolation=cv2.INTER_LINEAR)[..., None] * alpha
        out = (out * (1 - a) + np.array(color, np.float32) * a).astype(np.uint8)
    return out


def _polygons(mask: np.ndarray, min_cells: int, grow: int = 1) -> list[np.ndarray]:
    """Polígonos (en celdas) de las manchas de la máscara, un poco agrandadas (una zona tiene que tener ancho: los
    pies de cada persona caen en lugares un poco distintos). Siguen la forma (un pasillo en L queda en L)."""
    m = mask.astype(np.uint8)
    if grow:
        m = cv2.dilate(m, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        cells = int(cv2.fillPoly(np.zeros_like(m), [c], 1).sum())
        if cells < min_cells:
            continue
        poly = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2)
        if len(poly) >= 3:
            out.append((cells, poly))
    return [p for _, p in sorted(out, key=lambda cp: -cp[0])]


def _norm(poly: np.ndarray) -> list[list[float]]:
    gw, gh = GRID
    return [[round(float(min(max(x / gw, 0.0), 1.0)), 3), round(float(min(max(y / gh, 0.0), 1.0)), 3)]
            for x, y in poly]


def _inside(zones: list[dict], x: float, y: float) -> bool:
    from .geometry import point_in_polygon
    return any(point_in_polygon(x, y, [tuple(p) for p in z.get("polygon") or []]) for z in zones)


def suggest(fm: FootMap, existing: list[dict] | None = None, max_zones: int = 8) -> list[dict]:
    """Zonas propuestas: [{name, type, polygon, direction?, why}]. Nada que ya caiga adentro de `existing`."""
    existing = existing or []
    out: list[dict] = []
    if fm.persons == 0:
        return out
    gw, gh = GRID
    # dónde se para la gente: manchas de permanencia (≥ 25 % del máximo, y al menos unos segundos)
    still = cv2.GaussianBlur(fm.still, (0, 0), 1.2)
    if still.max() > 0:
        mask = still >= max(0.25 * still.max(), 0.3)
        for poly in _polygons(mask, 6):
            secs = float(fm.still[cv2.fillPoly(np.zeros((gh, gw), np.uint8), [poly], 1) > 0].sum())
            cx, cy = (float(v) for v in poly.mean(axis=0) / (gw, gh))
            if secs < 5 or _inside(existing, cx, cy):
                continue
            n = sum(1 for z in out if z["type"] == "shelf") + 1
            out.append({"name": f"Parada {n}", "type": "shelf", "polygon": _norm(poly),
                        "why": f"la gente se paró acá {secs:.0f} s en total (¿góndola o caja?)"})
    # por dónde camina: celdas por las que pasó buena parte de la gente (y que no son paradas)
    walk = cv2.GaussianBlur(fm.passed if fm.passed is not None else fm.walk, (0, 0), 0.8)
    if walk.max() > 0:
        taken = np.zeros((gh, gw), np.uint8)
        for z in out:
            pts = (np.array(z["polygon"]) * (gw, gh)).astype(np.int32)
            cv2.fillPoly(taken, [pts], 1)
        mask = (walk >= max(0.3 * walk.max(), 1.5)) & (taken == 0)
        for poly in _polygons(mask, 10)[:3]:
            cx, cy = (float(v) for v in poly.mean(axis=0) / (gw, gh))
            if _inside(existing, cx, cy):
                continue
            n = sum(1 for z in out if z["type"] == "aisle") + 1
            out.append({"name": f"Pasillo {n}", "type": "aisle", "polygon": _norm(poly),
                        "why": "por acá camina la gente"})
    # puertas: muchas personas que aparecen (o se van) pegadas al mismo borde
    out += _doors(fm, existing)
    return out[:max_zones]


def _doors(fm: FootMap, existing: list[dict]) -> list[dict]:
    pts = [(x, y, dx, dy, +1) for x, y, dx, dy in fm.appear] + [(x, y, dx, dy, -1) for x, y, dx, dy in fm.vanish]
    sides = {"abajo": [], "arriba": [], "izquierda": [], "derecha": []}  # borde por el que entran/salen
    for x, y, dx, dy, kind in pts:
        side = min((("arriba", y), ("abajo", 1 - y), ("izquierda", x), ("derecha", 1 - x)), key=lambda s: s[1])
        if side[1] <= BORDER:
            sides[side[0]].append((x, y, dx, dy, kind))
    out = []
    for side, items in sides.items():
        if len(items) < max(3, 0.25 * fm.persons):
            continue
        along = [p[0] if side in ("abajo", "arriba") else p[1] for p in items]
        lo, hi = float(np.percentile(along, 10)), float(np.percentile(along, 90))
        lo, hi = max(0.0, lo - 0.05), min(1.0, hi + 0.05)
        depth = 0.15
        if side == "abajo":
            poly, direction = [[lo, 1 - depth], [hi, 1 - depth], [hi, 1.0], [lo, 1.0]], "arriba"
        elif side == "arriba":
            poly, direction = [[lo, 0.0], [hi, 0.0], [hi, depth], [lo, depth]], "abajo"
        elif side == "izquierda":
            poly, direction = [[0.0, lo], [depth, lo], [depth, hi], [0.0, hi]], "derecha"
        else:
            poly, direction = [[1 - depth, lo], [1.0, lo], [1.0, hi], [1 - depth, hi]], "izquierda"
        cx, cy = np.mean(poly, axis=0)
        if _inside(existing, float(cx), float(cy)):
            continue
        n_in = sum(1 for p in items if p[4] > 0)
        moves = [(p[2], p[3]) for p in items if p[4] > 0 and abs(p[2]) + abs(p[3]) > 0.02]
        if moves:  # hacia dónde caminan de verdad los que aparecen por ese borde
            mx, my = float(np.mean([m[0] for m in moves])), float(np.mean([m[1] for m in moves]))
            direction = ("derecha" if mx > 0 else "izquierda") if abs(mx) > abs(my) else ("abajo" if my > 0 else "arriba")
        out.append({"name": f"Puerta {side}", "type": "entrance", "direction": direction,
                    "polygon": [[round(a, 3), round(b, 3)] for a, b in poly],
                    "why": f"{n_in} personas aparecieron y {len(items) - n_in} se fueron por el borde de {side} "
                           f"(entran hacia {direction}: revisalo)"})
    return out
