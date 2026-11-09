"""IA de movimientos de hurto: aprende con los esqueletos de los clips marcados en la app (pestaña «Entrenar IA
hurtos») y da alertas propias (`hurto_ia`, «Posible hurto (IA de movimientos)») aunque no salte ninguna regla.

Cómo funciona:

1. **Registro de cuadro** (`frame_record`): cada cuadro analizado se resume en las personas a la vista (número de
   seguimiento, recuadro y los 17 puntos de la postura de core/pose.py, o None si no se le miró) y los bolsos que ve el
   detector, en píxeles del cuadro analizado (píxeles cuadrados)::

       {"t": seg, "w": ancho, "h": alto,
        "people": [{"id": 3, "box": [x1, y1, x2, y2], "kp": [[x, y, conf] × 17] | None}],
        "bags": [[x1, y1, x2, y2], …]}

   Es lo que arma la PC al analizar un clip (el video queda en la PC: a la nube va esto y las ventanas) y lo que usa el
   detector en vivo (las máquinas, «Ver con la IA», la ventana de la cámara y las alarmas).
2. **Ventanas** (`person_windows`): por persona, cada `STEP_S` segundos se mira lo que hizo en los últimos `WINDOW_S`
   y se calculan los rasgos de `FEATURE_NAMES` (`FEATURE_HELP` dice qué es cada uno): dónde andan las muñecas respecto
   del cuerpo, los codos, las velocidades, los alcances hacia el estante, los bolsos… medidos en **largos de torso**
   (de la mitad de los hombros a la mitad de las caderas, como core/gestos.py), así no importa el tamaño de la persona
   en la imagen. Solo se usan las ventanas con la postura vista en `MIN_POSED` de los cuadros (y al menos
   `MIN_POSED_FRAMES`); un punto que falta (muñeca tapada, de espaldas) no rompe nada: ese rasgo toma un valor neutro.
   El detector en vivo usa **el mismo código** que el análisis de los clips: lo que se aprendió es lo que se mide.
3. **Etiquetas** (`window_labels`): en un clip con hurto, la ventana de la persona marcada que se pisa 1 s o más con la
   marca es 1; las de otras personas o de otros momentos, 0; las del borde (2 s alrededor de la marca) no se usan. Un
   clip normal: todo 0. Sin marcar: no se usa.
4. **Entrenamiento** (`train`): regresión logística con regularización L2 y pesos balanceados, en numpy (como el
   filtro de alertas de server/training.py), validada **separando por clip**: lo que se mide es cómo anda con un clip
   que el modelo no vio. El umbral se elige para ver `target_recall` de los hurtos marcados.
5. **En vivo** (`OnlineTheftDetector`): cuando la probabilidad de una persona pasa el umbral en `consecutive` ventanas
   seguidas, hay un aviso (una vez por racha); después, `cooldown_s` de enfriamiento para esa persona.

Lo honesto: con pocos hurtos marcados no sirve; empieza a servir con unos 30-50 hurtos marcados más clips normales de
la misma clase de cámara. Aprende lo que se le muestra (si todos los hurtos marcados son «a la bolsa», no va a ver
«al bolsillo») y, como todo en Vigía, cada aviso es un indicio para que lo revise una persona, nunca una prueba.

Python puro + numpy (sin OpenCV): lo importa el servidor (Vercel) para entrenar y validar.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .geometry import bbox_foot
from .gestos import GOOD, HIDDEN, SEEN, SIDES, GestureEngine, _angle, _inside_any, _pt, body_of
from .rules import Zone, _belongs, _inside_ratio, parse_zones

FEATURE_VERSION = 1
WINDOW_S = 3.0  # cada ventana mira los últimos 3 s de la persona
STEP_S = 1.0  # una ventana por persona cada 1 s
MIN_POSED = 0.5  # parte de los cuadros de la ventana con la postura de la persona vista, para usarla
MIN_POSED_FRAMES = 3  # y al menos tantos cuadros con postura (con menos no hay velocidades ni recorridos)
MAX_CLIP_S = 1800  # un clip de entrenamiento mira como mucho 30 min
FORGET_S = 10.0  # una persona que no se ve hace tanto se olvida (memoria acotada)
CONSECUTIVE = 2  # ventanas seguidas sobre el umbral para avisar
COOLDOWN_S = 30.0  # después de un aviso, la misma persona no vuelve a avisar en tanto
MARK_OVERLAP_S = 1.0  # una ventana es «del hurto» si se pisa al menos esto con la marca
MARK_MARGIN_S = 2.0  # alrededor de la marca, las ventanas que no llegan a eso no se usan (ni sí ni no)
MIN_MARKS = 5  # para entrenar: al menos tantos hurtos marcados (con la persona a la vista y postura)
MIN_NORMAL_CLIPS = 3  # y al menos tantos clips normales…
MIN_NEGATIVE_MINUTES = 10.0  # …o tantos minutos de ventanas sin hurto
VERDICTS = ("hurto", "normal", "sin_marcar")
MARK_KINDS = ("bolsillo", "ropa", "bolsa", "otro")
KIND = "ia_hurto"
C = 0.5  # inversa de la regularización (como en scikit-learn y server/training.py)

_FAR = 3.0  # tope de las distancias (en largos de torso): «lejos» ya es lejos
_STILL = GestureEngine.STILL_SPEED  # «parada»: las caderas se mueven menos de tanto (altos de su cuerpo por segundo)
_RETRACT = GestureEngine.RETRACT_SECONDS  # la mano vuelve al cuerpo hasta tanto después del alcance (una «toma»)
_WAIST = 0.35  # muñeca a menos de tanto (T) de la cadera del mismo lado: «a la cintura / al bolsillo»
_NEAR_BODY = 0.8  # muñeca a menos de tanto (T) del centro del torso: «junto al cuerpo»
_PAIR_GAP = 1.0  # dos cuadros con postura más separados que esto no se usan para velocidades
_STREAK_GAP = 1.5 * STEP_S  # dos ventanas de la misma persona más separadas que esto no son «seguidas»

# (nombre, qué mide). El orden es fijo: de él salen los coeficientes de los modelos entrenados. Si cambia un rasgo o
# su cálculo, subir FEATURE_VERSION (los modelos viejos dejan de usarse y hay que volver a entrenar).
FEATURES = (
    ("ia_muneca_cadera_min", "distancia mínima de una muñeca a la cadera del mismo lado (largos de torso; tope 3)"),
    ("ia_muneca_cadera_media", "distancia media de la muñeca más cercana a su cadera (largos de torso; tope 3)"),
    ("ia_mano_cintura", "parte del tiempo con una muñeca junto a su cadera (bolsillo, cintura)"),
    ("ia_muneca_torso", "parte del tiempo con una muñeca dentro del torso (de los hombros a las caderas)"),
    ("ia_muneca_cruza", "parte del tiempo con una muñeca del otro lado del eje del cuerpo"),
    ("ia_dos_manos", "parte del tiempo con las dos muñecas en el torso o la cintura a la vez (acomodarse)"),
    ("ia_codo_min", "ángulo mínimo de un codo (0 = cerrado, 1 = recto)"),
    ("ia_codo_medio", "ángulo medio de los codos (0 = cerrado, 1 = recto)"),
    ("ia_muneca_vel_max", "velocidad máxima de una muñeca (largos de torso por segundo; tope 20)"),
    ("ia_muneca_vel_media", "velocidad media de las muñecas (largos de torso por segundo; tope 20)"),
    ("ia_alcance", "parte del tiempo con un brazo estirado lejos del cuerpo, del pecho para arriba (hacia el estante)"),
    ("ia_alcance_max", "lo más lejos que llegó una muñeca del hombro (largos de torso; tope 3)"),
    ("ia_toma", "1 si estiró el brazo y en menos de 3 s volvió la mano al cuerpo (una «toma»)"),
    ("ia_cadera_vel", "velocidad media de las caderas (altos de su cuerpo por segundo; tope 5)"),
    ("ia_quieto", "parte del tiempo parada (las caderas casi no se mueven)"),
    ("ia_bolso_mano", "parte del tiempo con una muñeca metida en un bolso, cartera o mochila suyo"),
    ("ia_bolso_visto", "parte del tiempo con un bolso, cartera o mochila suyo a la vista"),
    ("ia_muneca_oculta", "parte del tiempo con un codo a la vista y su muñeca no (tapada)"),
    ("ia_caida_conf", "mayor caída de la confianza de una muñeca de un cuadro al siguiente (0 a 1)"),
    ("ia_mano_otra_persona", "parte del tiempo con una muñeca sobre otra persona (pasarle algo, de la mano)"),
    ("ia_alto_rel", "alto del recuadro de la persona respecto del alto del cuadro (mediana)"),
    ("ia_con_postura", "parte de los cuadros de la ventana con la postura vista"),
    ("ia_perfil", "parte del tiempo de perfil (hombros casi superpuestos)"),
    ("ia_de_espaldas", "parte del tiempo de espaldas (sin la cara a la vista)"),
    ("ia_torso_incl", "inclinación media del torso (0 = derecho, 1 = acostado: agacharse al estante de abajo)"),
    ("ia_zonas", "1 si la cámara tiene zonas de góndola, caja o salida marcadas"),
    ("ia_gondola", "parte del tiempo con los pies frente a una góndola (0 sin zonas)"),
    ("ia_caja_salida", "parte del tiempo en la caja o la salida (0 sin zonas)"),
)
FEATURE_NAMES = tuple(n for n, _ in FEATURES)
FEATURE_HELP = dict(FEATURES)


class NotEnoughData(ValueError):
    """Faltan clips marcados para entrenar (el mensaje, en castellano, dice qué falta)."""


# ---------------------------------------------------------------------------------------------- registro de cuadro
def _r(v: float, nd: int = 1) -> float:
    return round(float(v), nd)


def frame_record(ts: float, frame_size: tuple[int, int], tracks: list, bags: list | tuple = ()) -> dict:
    """El registro de un cuadro a partir del seguimiento (core/tracker.Track con `kpts`, la postura de core/pose.look)
    y los bolsos del detector (Detection con `bbox`, o recuadros). Solo las personas que se ven en este cuadro (de las
    que el seguimiento recuerda, el recuadro es viejo) y no los vehículos."""
    w, h = frame_size
    people = []
    for t in tracks:
        if not getattr(t, "visible", True) or getattr(t, "label", "persona") != "persona":
            continue
        kp = getattr(t, "kpts", None)
        people.append({"id": int(t.id), "box": [_r(v) for v in t.bbox],
                       "kp": None if kp is None else [[_r(p[0]), _r(p[1]), _r(p[2], 2)] for p in kp]})
    boxes = []
    for b in bags or ():
        box = getattr(b, "bbox", b)
        boxes.append([_r(v) for v in box])
    return {"t": _r(ts, 3), "w": int(w), "h": int(h), "people": people, "bags": boxes}


def _finite(*vals) -> bool:
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in vals)


def _box(b) -> tuple[float, float, float, float] | None:
    try:
        x1, y1, x2, y2 = (float(v) for v in b)
    except (TypeError, ValueError):
        return None
    if not _finite(x1, y1, x2, y2) or x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _kp(kp) -> list[tuple[float, float, float]] | None:
    """La postura como 17 (x, y, conf), o None si falta o está mal armada."""
    if kp is None:
        return None
    try:
        pts = [(float(p[0]), float(p[1]), float(p[2])) for p in kp]
    except (TypeError, ValueError, IndexError):
        return None
    if len(pts) != 17 or not all(_finite(*p) for p in pts):
        return None
    return pts


def _people(people) -> list[tuple[int, tuple, list | None]]:
    """(id, recuadro, postura) de cada persona del registro (dicts) o del seguimiento (Track); lo raro se saltea."""
    out = []
    seen = set()
    for p in people or ():
        if isinstance(p, dict):
            tid, box, kp = p.get("id"), p.get("box"), p.get("kp")
        else:
            if not getattr(p, "visible", True) or getattr(p, "label", "persona") != "persona":
                continue
            tid, box, kp = getattr(p, "id", None), getattr(p, "bbox", None), getattr(p, "kpts", None)
        try:
            tid = int(tid)
        except (TypeError, ValueError):
            continue
        b = _box(box)
        if b is None or tid in seen:
            continue
        seen.add(tid)
        out.append((tid, b, _kp(kp)))
    return out


def _bag_boxes(bags) -> list[tuple]:
    out = []
    for b in bags or ():
        box = _box(getattr(b, "bbox", b))
        if box is not None:
            out.append(box)
    return out


def _as_zones(zones) -> list[Zone]:
    """Las zonas de la cámara: objetos Zone, las zonas como las guarda el servidor (dicts) o el análisis entero
    (`{"zones": […], "extra_zones": […]}`). Lo que no se entiende se saltea (no frena el análisis)."""
    if not zones:
        return []
    if isinstance(zones, dict):
        zones = list(zones.get("zones") or []) + list(zones.get("extra_zones") or zones.get("staff_zones") or [])
    out: list[Zone] = []
    for z in zones:
        if isinstance(z, Zone):
            out.append(z)
            continue
        try:
            out += parse_zones([z])
        except (ValueError, TypeError, AttributeError):
            continue
    return out


# ---------------------------------------------------------------------------------------------- rasgos por ventana
def _reaching(body, sh, el, wr) -> bool:
    """El brazo estirado lejos del cuerpo, del pecho para arriba (hacia el estante): lo mismo que mira el motor de
    gestos (core/gestos.GestureEngine._reaching)."""
    if wr[2] < GOOD or sh[2] < GOOD:
        return False
    if el[2] >= SEEN:
        arm = math.hypot(sh[0] - el[0], sh[1] - el[1]) + math.hypot(el[0] - wr[0], el[1] - wr[1])
        if arm <= 1e-6 or math.hypot(wr[0] - sh[0], wr[1] - sh[1]) / arm < 0.6:
            return False
    if body.dist(wr, sh) < 0.75:
        return False
    along, across = body.local(wr)
    out_front = abs(across) >= body.half_width + 0.6 and along <= 0.65
    up = along < -0.15
    low_far = along > 0.65 and abs(across) >= body.half_width + 1.0
    return out_front or up or low_far


@dataclass
class _Arm:
    """Un brazo en un cuadro."""
    conf: float  # confianza de la muñeca
    pos: tuple[float, float] | None = None  # la muñeca en píxeles (None si no se ve)
    hip: float | None = None  # distancia a la cadera del mismo lado (T)
    elbow: float | None = None  # ángulo del codo (grados)
    shoulder: float | None = None  # distancia al hombro (T)
    torso: bool = False
    crosses: bool = False
    waist: bool = False
    reach: bool = False
    near_body: bool = False
    in_bag: bool = False
    hidden: bool = False
    other: bool = False


@dataclass
class _Frame:
    """Lo que se midió de una persona en un cuadro con postura."""
    t: float
    T: float  # largo de torso (px)
    hips: tuple[float, float]
    box_h: float
    h_rel: float
    arms: dict
    profile: bool
    away: bool
    incl: float
    bag_seen: bool
    shelf: bool
    cash: bool


def _arm(body, kp, side: str, bags: list, others: list) -> _Arm:
    si, ei, wi, hi = SIDES[side]
    sh, el, wr, hip = _pt(kp, si), _pt(kp, ei), _pt(kp, wi), _pt(kp, hi)
    a = _Arm(conf=wr[2], hidden=wr[2] < HIDDEN and el[2] >= GOOD)
    if wr[2] < SEEN:
        return a
    a.pos = (wr[0], wr[1])
    along, across = body.local(wr)
    if sh[2] >= SEEN:
        a.shoulder = min(_FAR, body.dist(wr, sh))
        if el[2] >= SEEN:
            a.elbow = _angle(sh, el, wr)
        sh_across = body.local(sh)[1]
        a.crosses = sh_across * across < 0 and abs(across) > 0.1
    if hip[2] >= SEEN:
        a.hip = min(_FAR, body.dist(wr, hip))
        # la región «bolsillo» de los gestos: junto a la cadera, no por fuera de ella y con el codo doblado (el brazo
        # que cuelga también pasa cerca de la cadera, pero estirado)
        outside = abs(across) - abs(body.local(hip)[1])
        a.waist = (a.hip <= _WAIST and 0.9 <= along <= 1.5 and outside <= 0.15
                   and (a.elbow is None or a.elbow <= 160))
    band = 0.35 if body.profile else 0.9 * body.half_width
    a.torso = 0.05 <= along <= 0.95 and abs(across) <= band
    a.reach = _reaching(body, sh, el, wr)
    center = ((body.S[0] + body.H[0]) / 2, (body.S[1] + body.H[1]) / 2)
    a.near_body = body.dist(wr, center) <= _NEAR_BODY
    a.in_bag = bool(bags) and GestureEngine._in_bag(wr, body, bags)
    a.other = _inside_any(wr, others)
    return a


def _mean(vals, default: float) -> float:
    return sum(vals) / len(vals) if vals else default


def _frac(flags) -> float:
    flags = list(flags)
    return sum(1 for f in flags if f) / len(flags) if flags else 0.0




def _features(frames: list[_Frame], n_frames: int, has_zones: bool) -> list[float]:
    """Los rasgos de una ventana (en el orden de FEATURE_NAMES) a partir de los cuadros con postura de la persona."""
    arms = [a for f in frames for a in f.arms.values()]
    hip_min = [min(d) for f in frames if (d := [a.hip for a in f.arms.values() if a.hip is not None])]
    elbows = [a.elbow for a in arms if a.elbow is not None]
    shoulders = [a.shoulder for a in arms if a.shoulder is not None]
    wrist_v, hip_v, drop = [], [], 0.0
    for a, b in zip(frames, frames[1:]):
        dt = b.t - a.t
        if not 0 < dt <= _PAIR_GAP:
            continue
        hip_v.append(min(5.0, math.hypot(b.hips[0] - a.hips[0], b.hips[1] - a.hips[1])
                         / max(1.0, (a.box_h + b.box_h) / 2) / dt))
        for side in SIDES:
            pa, pb = a.arms[side], b.arms[side]
            if pa.pos is not None and pb.pos is not None:
                wrist_v.append(min(20.0, math.hypot(pb.pos[0] - pa.pos[0], pb.pos[1] - pa.pos[1])
                                   / max(1.0, b.T) / dt))
            drop = max(drop, pa.conf - pb.conf)
    took = 0.0
    for side in SIDES:
        reached = None
        for f in frames:
            a = f.arms[side]
            if a.reach:
                reached = f.t
            elif reached is not None and a.near_body and f.t - reached <= _RETRACT:
                took = 1.0
    h_rel = sorted(f.h_rel for f in frames)
    vals = {
        "ia_muneca_cadera_min": min(hip_min) if hip_min else _FAR,
        "ia_muneca_cadera_media": _mean(hip_min, _FAR),
        "ia_mano_cintura": _frac(any(a.waist for a in f.arms.values()) for f in frames),
        "ia_muneca_torso": _frac(any(a.torso for a in f.arms.values()) for f in frames),
        "ia_muneca_cruza": _frac(any(a.crosses for a in f.arms.values()) for f in frames),
        "ia_dos_manos": _frac(all(a.torso or a.waist for a in f.arms.values()) for f in frames),
        "ia_codo_min": min(elbows) / 180.0 if elbows else 1.0,
        "ia_codo_medio": _mean(elbows, 180.0) / 180.0,
        "ia_muneca_vel_max": max(wrist_v, default=0.0),
        "ia_muneca_vel_media": _mean(wrist_v, 0.0),
        "ia_alcance": _frac(any(a.reach for a in f.arms.values()) for f in frames),
        "ia_alcance_max": max(shoulders) if shoulders else 0.0,
        "ia_toma": took,
        "ia_cadera_vel": _mean(hip_v, 0.0),
        "ia_quieto": _frac(v <= _STILL for v in hip_v) if hip_v else 1.0,
        "ia_bolso_mano": _frac(any(a.in_bag for a in f.arms.values()) for f in frames),
        "ia_bolso_visto": _frac(f.bag_seen for f in frames),
        "ia_muneca_oculta": _frac(any(a.hidden for a in f.arms.values()) for f in frames),
        "ia_caida_conf": max(0.0, min(1.0, drop)),
        "ia_mano_otra_persona": _frac(any(a.other for a in f.arms.values()) for f in frames),
        "ia_alto_rel": h_rel[len(h_rel) // 2] if h_rel else 0.0,
        "ia_con_postura": min(1.0, len(frames) / max(1, n_frames)),
        "ia_perfil": _frac(f.profile for f in frames),
        "ia_de_espaldas": _frac(f.away for f in frames),
        "ia_torso_incl": _mean([f.incl for f in frames], 0.0),
        "ia_zonas": 1.0 if has_zones else 0.0,
        "ia_gondola": _frac(f.shelf for f in frames),
        "ia_caja_salida": _frac(f.cash for f in frames),
    }
    return [round(float(vals[n]), 4) for n in FEATURE_NAMES]


@dataclass
class _Person:
    torso: deque = field(default_factory=lambda: deque(maxlen=12))  # largos de torso de los últimos cuadros (px)
    frames: deque = field(default_factory=deque)  # _Frame de los últimos segundos (solo con postura)
    last_seen: float = 0.0


class _Windower:
    """Arma las ventanas de cada persona a medida que llegan los cuadros (lo mismo en un clip y en vivo)."""

    def __init__(self, zones=None):
        self.zones = _as_zones(zones)
        self.reset()

    def reset(self) -> None:
        self.people: dict[int, _Person] = {}
        self.times: deque = deque()  # cuándo fue cada cuadro de los últimos segundos (todos, con o sin personas)
        self.next_tick: float | None = None
        self.last_t: float | None = None
        self.ticked = False

    @property
    def zones(self) -> list[Zone]:
        return self._zones

    @zones.setter
    def zones(self, zones) -> None:
        self._zones = _as_zones(zones)
        self._has_zones = any(z.type in ("shelf", "checkout", "exit") for z in self._zones)

    def feed(self, t: float, people, bags, size) -> list[tuple[int, float, float, list[float]]]:
        """Un cuadro. Devuelve las ventanas (persona, t0, t1, rasgos) que terminan en este cuadro (cada STEP_S)."""
        self.ticked = False
        try:
            t = float(t)
            w, h = (float(v) for v in size)
        except (TypeError, ValueError):
            return []
        if not _finite(t, w, h) or w <= 0 or h <= 0:
            return []
        if self.last_t is not None and t < self.last_t - 1e-9:  # el reloj volvió atrás (otro video): de cero
            self.reset()
        if self.next_tick is None:
            self.next_tick = t + WINDOW_S
        self.last_t = t
        self.times.append(t)
        keep = t - WINDOW_S - STEP_S
        while self.times and self.times[0] <= keep:
            self.times.popleft()
        persons = _people(people)
        bag_boxes = _bag_boxes(bags)
        for tid, box, kp in persons:
            st = self.people.get(tid)
            if st is None:
                st = self.people[tid] = _Person()
            st.last_seen = t
            if kp is not None:
                others = [b for oid, b, _ in persons if oid != tid and _inside_ratio(b, box) < 0.8]
                fr = self._frame(st, t, box, kp, [b for b in bag_boxes if _belongs(b, box)], others, w, h)
                if fr is not None:
                    st.frames.append(fr)
        for tid in [k for k, st in self.people.items() if t - st.last_seen > FORGET_S]:
            del self.people[tid]
        out = []
        if t >= self.next_tick - 1e-6:
            self.ticked = True
            start = t - WINDOW_S + 1e-6
            n_frames = sum(1 for x in self.times if x > start)
            for tid, st in self.people.items():
                while st.frames and st.frames[0].t <= keep:
                    st.frames.popleft()
                frames = [f for f in st.frames if f.t > start]
                if len(frames) < MIN_POSED_FRAMES or len(frames) < MIN_POSED * n_frames - 1e-9:
                    continue
                out.append((tid, round(t - WINDOW_S, 3), round(t, 3), _features(frames, n_frames, self._has_zones)))
            while self.next_tick <= t + 1e-6:
                self.next_tick += STEP_S
        return out

    def _frame(self, st: _Person, t: float, box, kp, bags, others, w: float, h: float) -> _Frame | None:
        hint = sorted(st.torso)[len(st.torso) // 2] if len(st.torso) >= 3 else None
        body = body_of(kp, box, hint)
        if body is None:  # sin hombros y caderas a la vista (o puntos imposibles): este cuadro no cuenta con postura
            return None
        st.torso.append(math.hypot(body.H[0] - body.S[0], body.H[1] - body.S[1]))
        fx, fy = bbox_foot(box)
        here = {z.type for z in self._zones if z.contains(fx / w, fy / h)}
        box_h = box[3] - box[1]
        return _Frame(t=t, T=body.T, hips=body.H, box_h=box_h, h_rel=min(1.5, box_h / h),
                      arms={side: _arm(body, kp, side, bags, others) for side in SIDES},
                      profile=body.profile, away=body.facing_away, incl=min(1.0, abs(body.u[0])),
                      bag_seen=bool(bags), shelf="shelf" in here, cash=bool(here & {"checkout", "exit"}))


def person_windows(frames, zones=None) -> list[dict]:
    """Las ventanas de un clip: `{"track_id", "t0", "t1", "f": [rasgos]}` cada STEP_S, de WINDOW_S, para las personas
    con la postura vista en MIN_POSED de los cuadros de la ventana. `frames`: los registros de cuadro, en orden (los
    que vienen mal armados se saltean; si el reloj vuelve atrás se empieza de cero). `zones`: las de la cámara (objetos
    Zone, dicts como los guarda el servidor, o nada). Mira como mucho MAX_CLIP_S segundos."""
    win = _Windower(zones)
    out = []
    first = None
    for fr in frames or ():
        if not isinstance(fr, dict):
            continue
        t = fr.get("t")
        if not _finite(t):
            continue
        if first is None or (win.last_t is not None and t < win.last_t - 1e-9):
            first = t
        if t - first > MAX_CLIP_S:
            continue
        for tid, t0, t1, f in win.feed(t, fr.get("people"), fr.get("bags"), (fr.get("w"), fr.get("h"))):
            out.append({"track_id": tid, "t0": t0, "t1": t1, "f": f})
    return out


# ---------------------------------------------------------------------------------------------- etiquetas
def _marks(marks) -> list[dict]:
    """Las marcas que se entienden (inicio < fin, números de verdad); lo demás se saltea."""
    out = []
    for m in marks or ():
        if not isinstance(m, dict):
            continue
        s, e = m.get("start"), m.get("end")
        if not _finite(s, e) or e <= s:
            continue
        tid = m.get("track_id")
        if tid is not None:
            try:
                tid = int(tid)
            except (TypeError, ValueError):
                tid = None
        out.append({"start": float(s), "end": float(e), "track_id": tid})
    return out


def _labels(verdict, marks, windows) -> tuple[list, list]:
    """(etiqueta, índice de la marca) de cada ventana; ver `window_labels`."""
    n = len(windows or ())
    if verdict == "normal":
        return [0] * n, [None] * n
    ms = _marks(marks)
    if verdict != "hurto" or not ms:  # sin marcar (o un hurto sin marcas: no se sabe dónde fue)
        return [None] * n, [None] * n
    labels, which = [], []
    for w in windows:
        try:
            t0, t1 = float(w["t0"]), float(w["t1"])
            tid = w.get("track_id")
            tid = None if tid is None else int(tid)
        except (TypeError, ValueError, KeyError):
            labels.append(None)
            which.append(None)
            continue
        best, best_i, near = 0.0, None, False
        for i, m in enumerate(ms):
            if m["track_id"] is not None and m["track_id"] != tid:
                continue  # otra persona: lo que hizo esa no es este hurto
            over = min(t1, m["end"]) - max(t0, m["start"])
            if over >= MARK_OVERLAP_S - 1e-9 and over > best:
                best, best_i = over, i
            elif min(t1, m["end"] + MARK_MARGIN_S) - max(t0, m["start"] - MARK_MARGIN_S) > 0:
                near = True
        if best_i is not None:
            labels.append(1)
            which.append(best_i)
        else:
            labels.append(None if near else 0)
            which.append(None)
    return labels, which


def window_labels(verdict: str, marks: list[dict] | None, windows: list[dict]) -> list[int | None]:
    """La etiqueta de cada ventana de un clip para entrenar. `verdict`: «hurto», «normal» o «sin_marcar». `marks`:
    `[{"start", "end", "track_id": int | None, "kind"}]` (segundos del clip). 1 = la ventana se pisa MARK_OVERLAP_S o
    más con una marca y es de esa persona (o la marca no dice persona); None = la persona de la marca cerca de la marca
    (MARK_MARGIN_S alrededor) sin llegar a eso: no se usa; 0 = clip normal, u otra persona, u otro momento del clip.
    Sin marcar (o «hurto» sin marcas): todas None."""
    return _labels(verdict, marks, windows)[0]


def build_samples(clips) -> tuple[list, list, dict]:
    """Para `train`, a partir de los clips (dicts u objetos con `id`, `verdict`, `marks`, `windows`, `duration_s` y,
    si lo tienen, `feature_version`): `(samples, groups, durations)`, con cada muestra `(f, etiqueta, datos)` donde
    `datos` = persona, t0, t1 y qué marca es (para contar hurtos vistos y falsas alarmas por persona, como en vivo).
    Los clips de otra versión de los rasgos se saltean."""
    samples, groups, durations = [], [], {}
    for c in clips or ():
        get = c.get if isinstance(c, dict) else (lambda k, d=None, _c=c: getattr(_c, k, d))
        cid = get("id")
        fv = get("feature_version", FEATURE_VERSION)
        if fv is not None and fv != FEATURE_VERSION:
            continue
        windows = [w for w in (get("windows") or []) if isinstance(w, dict)]
        labels, which = _labels(get("verdict"), get("marks"), windows)
        for w, lab, mi in zip(windows, labels, which):
            if lab is None:
                continue
            samples.append((w.get("f"), lab, {"track_id": w.get("track_id"), "t0": w.get("t0"), "t1": w.get("t1"),
                                              "mark": None if mi is None else (cid, mi)}))
            groups.append(cid)
        d = get("duration_s")
        if _finite(d):
            durations[cid] = float(d)
    return samples, groups, durations


# ---------------------------------------------------------------------------------------------- entrenamiento
def _fit(X: np.ndarray, y: np.ndarray, C: float = C, iters: int = 60) -> tuple[np.ndarray, float]:
    """Newton-Raphson sobre: sum_i w_i * logloss_i + ||coef||² / (2C), con pesos balanceados (como el filtro)."""
    n, d = X.shape
    pos = y.sum()
    w = np.where(y == 1, n / (2 * max(pos, 1)), n / (2 * max(n - pos, 1)))
    Xb = np.hstack([X, np.ones((n, 1))])
    theta = np.zeros(d + 1)
    reg = np.full(d + 1, 1.0 / C)
    reg[-1] = 0.0  # el intercepto no se regulariza
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ theta, -50, 50)))
        grad = Xb.T @ (w * (p - y)) + reg * theta
        H = (Xb * (w * p * (1 - p))[:, None]).T @ Xb + np.diag(reg) + 1e-9 * np.eye(d + 1)
        step = np.linalg.solve(H, grad)
        theta -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return theta[:-1], float(theta[-1])


def _standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale < 1e-9] = 1.0
    return mean, scale


def _prob(X: np.ndarray, coef: np.ndarray, b: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(X @ coef + b, -50, 50)))


def _auc(p: np.ndarray, y: np.ndarray) -> float | None:
    """Área bajo la curva ROC (Mann-Whitney, con empates promediados)."""
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return None
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p))
    sp = p[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _alarms(seq: list[tuple[float, float]], thr: float, consecutive: int, cooldown: float) -> list[float]:
    """Cuándo avisaría el detector en vivo con estas ventanas de UNA persona (`(t1, puntaje)` en orden): `consecutive`
    ventanas seguidas sobre el umbral, un aviso por racha, y `cooldown` entre avisos (lo mismo que
    OnlineTheftDetector)."""
    out: list[float] = []
    count, fired, last_t1, last_fire = 0, False, None, None
    for t1, s in seq:
        if s >= thr:
            if last_t1 is None or count == 0 or t1 - last_t1 > _STREAK_GAP:
                count, fired = 0, False
            count += 1
            last_t1 = t1
            if count >= consecutive and not fired and (last_fire is None or t1 - last_fire >= cooldown):
                fired, last_fire = True, t1
                out.append(t1)
        else:
            count, fired, last_t1 = 0, False, None
    return out


def _mark_score(seq: list[tuple[float, float]], consecutive: int) -> float:
    """Con qué umbral se vería este hurto: el mejor puntaje sostenido `consecutive` ventanas seguidas de la persona
    marcada (si la marca no tiene tantas ventanas seguidas, el mejor puntaje suelto)."""
    best_run = None
    for i in range(len(seq) - consecutive + 1):
        part = seq[i:i + consecutive]
        if all(b[0] - a[0] <= _STREAK_GAP for a, b in zip(part, part[1:])):
            v = min(s for _, s in part)
            best_run = v if best_run is None else max(best_run, v)
    if best_run is not None:
        return best_run
    return max((s for _, s in seq), default=0.0)


def _folds(y: np.ndarray, groups: list, k_max: int = 5) -> list[np.ndarray]:
    """Grupos de validación por clip: cada clip entero va a un solo grupo, con los hurtos repartidos."""
    keys = sorted(set(groups), key=lambda g: str(g))
    g_idx = {g: [] for g in keys}
    for i, g in enumerate(groups):
        g_idx[g].append(i)
    pos = {g: int(y[g_idx[g]].sum()) for g in keys}
    k = min(k_max, len(keys))
    bins: list[list] = [[] for _ in range(k)]
    bpos, bn = [0] * k, [0] * k
    for g in sorted(keys, key=lambda g: (-pos[g], -len(g_idx[g]), str(g))):
        i = min(range(k), key=lambda j: (bpos[j], bn[j])) if pos[g] else min(range(k), key=lambda j: (bn[j], j))
        bins[i].append(g)
        bpos[i] += pos[g]
        bn[i] += len(g_idx[g])
    return [np.array(sorted(i for g in b for i in g_idx[g]), dtype=int) for b in bins if b]


def _union_seconds(spans: list[tuple[float, float]]) -> float:
    total, end = 0.0, None
    for a, b in sorted(spans):
        if end is None or a > end:
            total += b - a
            end = b
        elif b > end:
            total += b - end
            end = b
    return total


def train(samples, groups, target_recall: float = 0.8, durations: dict | None = None) -> tuple[dict, dict]:
    """Entrena la IA de movimientos de hurto. `samples`: `[(f, etiqueta)]` o `[(f, etiqueta, datos)]` (`datos` =
    `{"track_id", "t0", "t1", "mark"}`, como los arma `build_samples`: con eso los hurtos vistos y las falsas alarmas
    se cuentan por marca y por persona, como en vivo; sin eso, cada clip con hurto cuenta como una marca y las ventanas
    van en el orden en que llegan). `groups`: el clip de cada muestra (la validación separa por clip). `durations`:
    segundos de cada clip (para las horas de video normal; si falta, se cuenta el tiempo cubierto por las ventanas).

    Devuelve `(spec, metrics)`; NotEnoughData (con el mensaje en castellano) si faltan hurtos marcados o clips
    normales, o si los hurtos están todos en un mismo clip."""
    target = min(0.99, max(0.3, float(target_recall)))
    groups = list(groups)
    if len(groups) != len(samples):
        raise ValueError("samples y groups tienen que tener el mismo largo")
    n_f = len(FEATURE_NAMES)
    X, y, gs, metas = [], [], [], []
    skipped = 0
    for i, (item, g) in enumerate(zip(samples, groups)):
        try:
            f, lab = item[0], item[1]
            meta = item[2] if len(item) > 2 and isinstance(item[2], dict) else None
        except (TypeError, IndexError, KeyError):
            skipped += 1
            continue
        if isinstance(f, dict):  # (una ventana entera de person_windows)
            meta = meta or {"track_id": f.get("track_id"), "t0": f.get("t0"), "t1": f.get("t1"), "mark": None}
            f = f.get("f")
        if lab is None or isinstance(lab, str) or int(bool(lab)) != lab:
            skipped += 1
            continue
        try:
            vec = [float(v) for v in f]
        except (TypeError, ValueError):
            skipped += 1
            continue
        if len(vec) != n_f or not all(math.isfinite(v) for v in vec):
            skipped += 1
            continue
        X.append(vec)
        y.append(int(lab))
        gs.append(g)
        metas.append(meta)
    X = np.array(X, dtype=np.float64).reshape(len(y), n_f)
    y = np.array(y, dtype=np.int64)
    with_meta = bool(metas) and all(m is not None for m in metas)

    # ---- qué hay
    pos_groups = sorted({g for g, lab in zip(gs, y) if lab == 1}, key=str)
    # cada hurto marcado: (clip, cuál marca); sin datos de la marca, el clip entero cuenta como una
    mark_of = [None if lab != 1 else (g, m.get("mark") if with_meta and m.get("mark") is not None else "clip")
               for g, m, lab in zip(gs, metas, y)]
    marks = sorted({mk for mk in mark_of if mk is not None}, key=str)
    neg_groups = sorted({g for g in gs} - set(pos_groups), key=str)
    n_neg = int(len(y) - y.sum())
    neg_minutes = n_neg * STEP_S / 60.0
    if len(marks) < MIN_MARKS:
        raise NotEnoughData(
            f"Faltan hurtos marcados: hay {len(marks)} y se necesitan al menos {MIN_MARKS} (cada uno con la persona a "
            "la vista y su postura). Empieza a servir de verdad con 30 a 50.")
    if len(neg_groups) < MIN_NORMAL_CLIPS and neg_minutes < MIN_NEGATIVE_MINUTES:
        raise NotEnoughData(
            f"Faltan clips normales: hay {len(neg_groups)} y se necesitan al menos {MIN_NORMAL_CLIPS} (o "
            f"{MIN_NEGATIVE_MINUTES:.0f} minutos de gente a la vista sin hurto; hay {neg_minutes:.0f}).")
    if len(pos_groups) < 2:
        raise NotEnoughData("Los hurtos marcados están todos en un mismo clip: para poder probar el modelo con un clip "
                            "que no vio hacen falta hurtos marcados en al menos 2 clips distintos.")

    # ---- validación separando por clip
    folds = _folds(y, gs)
    oof = np.full(len(y), np.nan)
    for te in folds:
        tr = np.setdiff1d(np.arange(len(y)), te)
        if len(set(y[tr].tolist())) < 2:
            raise NotEnoughData("Con estos clips no se puede validar (en algún grupo de prueba quedan todos los hurtos "
                                "o todos los normales): sumá hurtos marcados en más clips.")
        mean, scale = _standardize(X[tr])
        coef, b = _fit((X[tr] - mean) / scale, y[tr])
        oof[te] = _prob((X[te] - mean) / scale, coef, b)

    # ---- por persona y por marca, como avisaría en vivo
    def seqs(sel) -> dict:
        out: dict = {}
        order: dict = {}
        for i in sel:
            if with_meta:
                key = (gs[i], metas[i].get("track_id"))
                t1 = metas[i].get("t1")
                t1 = float(t1) if _finite(t1) else None
            else:
                key, t1 = (gs[i], None), None
            if t1 is None:  # sin tiempos: en el orden en que llegaron, una ventana por paso
                order[key] = order.get(key, -1) + 1
                t1 = order[key] * STEP_S
            out.setdefault(key, []).append((t1, float(oof[i])))
        return {k: sorted(v) for k, v in out.items()}

    pos_idx = [i for i in range(len(y)) if y[i] == 1]
    by_mark: dict = {}
    for i in pos_idx:
        by_mark.setdefault(mark_of[i], []).append(i)
    mark_scores = {}
    for mk, idx in by_mark.items():
        per = seqs(idx)
        mark_scores[mk] = max(_mark_score(v, CONSECUTIVE) for v in per.values())
    ranked = sorted(mark_scores.values(), reverse=True)
    need = max(1, int(math.ceil(target * len(ranked) - 1e-9)))
    cut = ranked[need - 1]  # el umbral más alto que todavía ve `target` de los hurtos marcados
    neg_idx = [i for i in range(len(y)) if y[i] == 0]
    # …y se baja hasta la mitad del camino a la ventana normal más alta que queda por debajo: en la validación da lo
    # mismo (ninguna ventana normal cae en el medio) y deja margen para los hurtos que el modelo no vio
    below = [float(oof[i]) for i in neg_idx if oof[i] < cut]
    threshold = float(min(0.999, max(0.01, (cut + max(below, default=0.0)) / 2)))
    marks_seen = sum(1 for s in ranked if s >= threshold)

    false_alarms = sum(len(_alarms(v, threshold, CONSECUTIVE, COOLDOWN_S)) for v in seqs(neg_idx).values())
    neg_seconds = 0.0
    durations = durations or {}
    for g in sorted(set(gs), key=str):
        if g not in pos_groups and _finite(durations.get(g)):
            neg_seconds += float(durations[g])
            continue
        idx = [i for i in neg_idx if gs[i] == g]
        if with_meta and all(_finite(metas[i].get("t0"), metas[i].get("t1")) for i in idx):
            neg_seconds += _union_seconds([(float(metas[i]["t0"]), float(metas[i]["t1"])) for i in idx])
        else:
            neg_seconds += len(idx) * STEP_S
    neg_hours = neg_seconds / 3600.0
    auc = _auc(oof, y)

    # ---- el modelo final, con todo
    mean, scale = _standardize(X)
    coef, b = _fit((X - mean) / scale, y)
    spec = {"kind": KIND, "version": FEATURE_VERSION, "names": list(FEATURE_NAMES),
            "mean": [float(v) for v in mean], "scale": [float(v) for v in scale], "coef": [float(v) for v in coef],
            "intercept": float(b), "threshold": round(threshold, 6), "window_s": WINDOW_S, "step_s": STEP_S,
            "consecutive": CONSECUTIVE, "cooldown_s": COOLDOWN_S}
    metrics = {
        "marks": len(ranked), "marks_seen": int(marks_seen),
        "false_per_hour": round(false_alarms / neg_hours, 2) if neg_hours > 0 else 0.0,
        "clips": len(set(gs)), "windows": int(len(y)), "positives": int(y.sum()),
        "threshold": round(threshold, 4), "auc": None if auc is None else round(auc, 3),
        # (además de lo del contrato: para explicar el resultado)
        "negatives": n_neg, "false_alarms": int(false_alarms), "negative_hours": round(neg_hours, 3),
        "hurto_clips": len(pos_groups), "normal_clips": len(neg_groups), "folds": len(folds),
        "validation": "por_clip", "target_recall": target, "by_mark": with_meta, "skipped": skipped,
    }
    return spec, metrics


# ---------------------------------------------------------------------------------------------- en vivo
def _num(v, lo: float, hi: float, default: float) -> float:
    if not _finite(v):
        return default
    return float(v) if lo <= float(v) <= hi else default


class TheftModel:
    """Un modelo entrenado (`spec` de `train`). `ok`: es de esta versión de los rasgos y está bien armado (si no, no
    puntúa: `score` da 0 y no se avisa nunca). Nunca falla al armarlo, venga lo que venga."""

    def __init__(self, spec):
        self.spec = spec if isinstance(spec, dict) else {}
        self.ok = False
        self.id = self.spec.get("id")
        self.threshold = 1.0
        self.consecutive = CONSECUTIVE
        self.cooldown_s = COOLDOWN_S
        try:
            self._load(self.spec)
        except (TypeError, ValueError, KeyError, AttributeError, OverflowError):
            self.ok = False

    def _load(self, s: dict) -> None:
        if s.get("kind", KIND) != KIND or int(s.get("version")) != FEATURE_VERSION:
            return
        if list(s.get("names") or []) != list(FEATURE_NAMES):
            return
        n = len(FEATURE_NAMES)
        mean = np.array([float(v) for v in s["mean"]], dtype=np.float64)
        scale = np.array([float(v) for v in s["scale"]], dtype=np.float64)
        coef = np.array([float(v) for v in s["coef"]], dtype=np.float64)
        intercept, thr = float(s["intercept"]), float(s["threshold"])
        if not (mean.shape == scale.shape == coef.shape == (n,)):
            return
        if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(scale)) and np.all(np.isfinite(coef))
                and math.isfinite(intercept) and 0.0 <= thr <= 1.0):
            return
        if abs(_num(s.get("window_s", WINDOW_S), 0, 1e9, -1) - WINDOW_S) > 1e-6 or \
                abs(_num(s.get("step_s", STEP_S), 0, 1e9, -1) - STEP_S) > 1e-6:
            return  # ventanas de otro largo: los rasgos no serían los mismos
        scale[np.abs(scale) < 1e-12] = 1.0
        self.mean, self.scale, self.coef, self.intercept, self.threshold = mean, scale, coef, intercept, thr
        self.consecutive = int(_num(s.get("consecutive"), 1, 10, CONSECUTIVE))
        self.cooldown_s = _num(s.get("cooldown_s"), 0, 3600, COOLDOWN_S)
        self.ok = True

    @classmethod
    def maybe(cls, spec) -> "TheftModel | None":
        m = cls(spec)
        return m if m.ok else None

    def score(self, f) -> float:
        """Probabilidad (0 a 1) de que la ventana sea un movimiento de hurto; 0 si el modelo no sirve."""
        if not self.ok:
            return 0.0
        try:
            x = np.array([float(v) for v in f], dtype=np.float64)
        except (TypeError, ValueError):
            return 0.0
        if x.shape != self.mean.shape:
            return 0.0
        bad = ~np.isfinite(x)
        x[bad] = self.mean[bad]  # un rasgo roto pesa como el promedio (no empuja para ningún lado)
        z = float(((x - self.mean) / self.scale) @ self.coef + self.intercept)
        return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, z))))

    def is_alert(self, score: float) -> bool:
        return self.ok and score >= self.threshold


def spec_ok(spec) -> bool:
    """El modelo de la IA de hurtos sirve para esta versión (para decidir si mirar la postura)."""
    return TheftModel(spec).ok


@dataclass
class _Streak:
    count: int
    t0: float
    last_t1: float
    fired: bool = False


class OnlineTheftDetector:
    """El detector en vivo: se le pasa cada cuadro analizado (`update`) y avisa cuando una persona se mueve como en
    los hurtos marcados. Mismo cálculo de ventanas que los clips de entrenamiento. Memoria acotada: las personas que
    no se ven hace FORGET_S se olvidan. Si el modelo no sirve (`ok` False) no hace nada."""

    def __init__(self, spec, zones=None):
        self.model = TheftModel(spec)
        self.spec = self.model.spec
        self._win = _Windower(zones)
        self._streaks: dict[int, _Streak] = {}
        self._last_fire: dict[int, float] = {}
        self.scores: dict[int, float] = {}  # persona -> puntaje de su última ventana (para mostrarlo)

    @property
    def ok(self) -> bool:
        return self.model.ok

    @property
    def threshold(self) -> float:
        return self.model.threshold

    @property
    def zones(self) -> list[Zone]:
        return self._win.zones

    @zones.setter
    def zones(self, zones) -> None:
        self._win.zones = zones

    def reset(self) -> None:
        self._win.reset()
        self._streaks.clear()
        self._last_fire.clear()
        self.scores.clear()

    def update(self, ts: float, people, bags=(), frame_size=(0, 0)) -> list[dict]:
        """Un cuadro: `people` como en el registro de cuadro (dicts con id, box, kp) o el seguimiento (Track con
        `kpts`); `bags` recuadros (o Detection); `frame_size` (ancho, alto). Devuelve los avisos:
        `{"track_id", "score", "t0", "t1", "features": {nombre: valor}}` (t0: desde dónde viene la racha)."""
        if not self.model.ok:
            return []
        last = self._win.last_t
        if last is not None and _finite(ts) and ts < last - 1e-9:  # el reloj volvió atrás: de cero
            self.reset()
        wins = self._win.feed(ts, people, bags, frame_size)
        live = set(self._win.people)
        for d in (self._streaks, self.scores):
            for tid in [k for k in d if k not in live]:
                del d[tid]
        if _finite(ts):
            for tid in [k for k, t in self._last_fire.items() if ts - t > self.model.cooldown_s]:
                del self._last_fire[tid]
        if not self._win.ticked:
            return []
        hits = []
        emitted = set()
        m = self.model
        for tid, t0, t1, f in wins:
            emitted.add(tid)
            s = m.score(f)
            self.scores[tid] = s
            st = self._streaks.get(tid)
            if s < m.threshold:
                self._streaks.pop(tid, None)
                continue
            if st is None or t1 - st.last_t1 > _STREAK_GAP:
                st = self._streaks[tid] = _Streak(0, t0, t1)
            st.count += 1
            st.last_t1 = t1
            last_fire = self._last_fire.get(tid)
            if st.count >= m.consecutive and not st.fired and (last_fire is None or t1 - last_fire >= m.cooldown_s):
                st.fired = True
                self._last_fire[tid] = t1
                hits.append({"track_id": tid, "score": round(s, 4), "t0": st.t0, "t1": t1,
                             "features": dict(zip(FEATURE_NAMES, f))})
        for tid in [k for k in self._streaks if k not in emitted]:  # sin ventana en este paso: la racha se corta
            del self._streaks[tid]
        for tid in [k for k in self.scores if k not in emitted]:
            del self.scores[tid]
        return hits
