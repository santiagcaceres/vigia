"""Gestos de ocultamiento a partir de la postura de cada persona (core/pose.py): mano al bolsillo, mano dentro de la
ropa y mano en la bolsa (o cartera, mochila).

Cómo decide (por persona y por brazo, todo medido en «largos de torso» T = de la mitad de los hombros a la mitad de
las caderas, así no importa el tamaño de la persona en la imagen):

1. **Toma.** El brazo se estira lejos del cuerpo (codo casi recto, la muñeca a más de 0,8·T del hombro y fuera del
   torso) y después la mano vuelve al cuerpo. Es lo que más falsas alarmas saca: sin una toma en los 10 s anteriores,
   meter la mano en el bolsillo es normal (el celular, la billetera, el frío) y no se avisa.
2. **Ocultamiento.** Después de la toma, la muñeca entra a una región y se queda ahí un rato:
   - bolsillo: junto a la cadera del mismo lado (a menos de 0,35·T; sale a más de 0,5·T), con el codo doblado;
   - ropa: dentro del torso (de los hombros a las caderas) y además cruza el cuerpo, baja a la cintura en el centro o
     deja de verse con el codo a la vista;
   - bolsa: dentro de un bolso, cartera o mochila que el detector ve junto a la persona; o, sin bolso detectado, las
     manos bajas delante del cuerpo (de la cintura a la cadera) con el codo doblado, un rato más largo.
3. **Descartes.** Las dos manos a la vez al bolsillo o la ropa (acomodarse, manos en los bolsillos), la mano que ya
   estaba en el bolsillo antes de la toma o que se queda más de 15 s (postura), la caja (pagar), la zona de personal
   (empleados), personas chicas o sin hombros y caderas a la vista.

Es un indicio para que lo revise una persona, nunca una prueba. Los tiempos y distancias salen de la investigación
del 29 oct (Intel behavioral-analysis, Paza, PoseLift, sigmas de COCO) y se ajustan por cámara en «Configurar…»
con videos de esa cámara. Python puro: lo importa también el servidor (para validar la configuración).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

MANO_BOLSILLO = "mano_bolsillo"
MANO_ROPA = "mano_ropa"
MANO_BOLSA = "mano_bolsa"
GESTURE_TYPES = (MANO_BOLSILLO, MANO_ROPA, MANO_BOLSA)
GESTURE_NAMES = {MANO_BOLSILLO: "Mano al bolsillo", MANO_ROPA: "Mano dentro de la ropa",
                 MANO_BOLSA: "Mano en la bolsa"}
REGION_OF = {MANO_BOLSILLO: "bolsillo", MANO_ROPA: "ropa", MANO_BOLSA: "bolsa"}
GESTURE_OF = {v: k for k, v in REGION_OF.items()}

GOOD = 0.5  # confianza para usar un punto en la geometría
SEEN = 0.3  # confianza para contarlo como «visto»
HIDDEN = 0.35  # por debajo, la muñeca «dejó de verse» (pista de oclusión, secundaria)
SIDES = {"izq": (5, 7, 9, 11), "der": (6, 8, 10, 12)}  # hombro, codo, muñeca, cadera de cada lado
SIDE_NAMES = {"izq": "izquierda", "der": "derecha"}
_NUM_KEYS = ("pocket_seconds", "clothes_seconds", "bag_seconds", "take_window", "cooldown_seconds",
             "min_person_px")


@dataclass
class GestureConfig:
    enabled: tuple[str, ...] = ()  # vacío = gestos apagados
    pocket_seconds: float = 0.6  # mano en el bolsillo al menos esto
    clothes_seconds: float = 0.8  # mano dentro de la ropa
    bag_seconds: float = 1.0  # mano en la bolsa (sin bolso detectado se pide el doble)
    take_window: float = 10.0  # el ocultamiento cuenta si hubo una toma hasta tantos segundos antes
    require_take: bool = True  # sin toma vista antes no se avisa
    cooldown_seconds: float = 30.0  # el mismo gesto de la misma persona, una vez cada tanto
    min_person_px: float = 90.0  # personas más chicas (alto en píxeles del cuadro) no se miran

    @classmethod
    def from_dict(cls, d: dict | None) -> "GestureConfig":
        d = dict(d or {})
        cfg = cls()
        en = d.pop("enabled", None)
        if en is not None:
            bad = [g for g in en if g not in GESTURE_TYPES]
            if bad:
                raise ValueError(f"gestures.enabled: gestos desconocidos {bad}")
            cfg.enabled = tuple(g for g in GESTURE_TYPES if g in en)
        for k in _NUM_KEYS:
            if k in d:
                v = float(d.pop(k))
                if v < 0 or v > 600:
                    raise ValueError(f"gestures.{k} fuera de rango")
                setattr(cfg, k, v)
        if "require_take" in d:
            cfg.require_take = bool(d.pop("require_take"))
        if d:
            raise ValueError(f"gestures: claves desconocidas {sorted(d)}")
        return cfg

    def to_dict(self) -> dict:
        return {"enabled": list(self.enabled), **{k: getattr(self, k) for k in _NUM_KEYS},
                "require_take": self.require_take}


@dataclass
class GestureHit:
    type: str
    side: str
    dwell: float
    since_take: float | None
    description: str
    features: dict[str, float]


@dataclass
class _Hand:
    region: str | None = None  # bolsillo | ropa | bolsa | None
    since: float = 0.0
    last_in: float = -1e9  # última vez que se la vio en la región
    frames: int = 0
    reach_since: float | None = None
    reach_frames: int = 0
    reached_ts: float | None = None  # último alcance confirmado (brazo estirado 2 cuadros o más)
    free_ts: float = -1e9  # última vez que se la vio fuera de toda región (libre)


@dataclass
class _Person:
    torso: deque = field(default_factory=lambda: deque(maxlen=12))  # largos de torso de los últimos cuadros
    hands: dict = field(default_factory=lambda: {"izq": _Hand(), "der": _Hand()})
    take_ts: float | None = None
    take_side: str | None = None
    take_hist: deque = field(default_factory=lambda: deque(maxlen=8))  # cuándo fueron las últimas tomas
    takes: int = 0
    reaches: int = 0
    fired: dict = field(default_factory=dict)  # gesto → ts del último aviso
    last_fire: float = -1e9
    bad_frames: int = 0
    # lo que se mide para explicar por qué dio o no («Configurar…») y para el modelo
    max_dwell: dict = field(default_factory=lambda: {"bolsillo": 0.0, "ropa": 0.0, "bolsa": 0.0})
    suppressed: dict = field(default_factory=dict)
    conf_drop: float = 0.0
    frames_ok: int = 0
    frames_seen: int = 0
    first_ts: float = 0.0
    feet: deque = field(default_factory=lambda: deque(maxlen=40))  # (ts, x, y) de las caderas, en altos de persona

    def speed(self, ts: float, window: float = 1.0) -> float:
        """Cuánto se movió en el último segundo, en altos de su cuerpo por segundo (0,2 = casi quieta)."""
        pts = [f for f in self.feet if ts - f[0] <= window]
        if len(pts) < 2 or pts[-1][0] - pts[0][0] < 0.3:
            return 0.0
        return math.hypot(pts[-1][1] - pts[0][1], pts[-1][2] - pts[0][2]) / (pts[-1][0] - pts[0][0])


def _inside_any(p, boxes, grow: float = 0.1) -> bool:
    """El punto cae dentro de alguno de los recuadros (agrandados un poco: la mano que se le acerca a otra persona
    queda en el borde de su recuadro)."""
    for b in boxes or ():
        mx, my = grow * (b[2] - b[0]), grow * (b[3] - b[1])
        if b[0] - mx <= p[0] <= b[2] + mx and b[1] - my <= p[1] <= b[3] + my:
            return True
    return False


def _pt(kp, i) -> tuple[float, float, float]:
    x, y, c = kp[i][0], kp[i][1], kp[i][2]
    return float(x), float(y), float(c)


def _mid(kp, a: int, b: int) -> tuple[float, float] | None:
    """Punto medio de dos puntos (o el que se vea bien, si el otro no)."""
    pa, pb = _pt(kp, a), _pt(kp, b)
    if pa[2] >= GOOD and pb[2] >= GOOD:
        return (pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2
    if max(pa[2], pb[2]) >= GOOD:
        p = pa if pa[2] >= pb[2] else pb
        return p[0], p[1]
    return None


def _angle(a, b, c) -> float:
    """Ángulo en b (grados) entre a-b-c."""
    v1, v2 = (a[0] - b[0], a[1] - b[1]), (c[0] - b[0], c[1] - b[1])
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0
    cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cos))


@dataclass
class Body:
    """El cuerpo de una persona en un cuadro, en coordenadas del torso: `along` 0 en los hombros y 1 en las
    caderas; `across` la distancia al eje del cuerpo; las dos en largos de torso."""
    S: tuple[float, float]
    H: tuple[float, float]
    T: float
    u: tuple[float, float]
    v: tuple[float, float]
    half_width: float  # medio ancho del torso (en T)
    profile: bool  # de perfil (hombros casi superpuestos)
    facing_away: bool  # de espaldas (no se ve la cara)

    def local(self, p) -> tuple[float, float]:
        dx, dy = p[0] - self.S[0], p[1] - self.S[1]
        return (dx * self.u[0] + dy * self.u[1]) / self.T, (dx * self.v[0] + dy * self.v[1]) / self.T

    def dist(self, p, q) -> float:
        return math.hypot(p[0] - q[0], p[1] - q[1]) / self.T


def body_of(kp, bbox, torso_hint: float | None = None) -> Body | None:
    """Ejes del cuerpo, o None si no se ven bien los hombros y las caderas."""
    S, H = _mid(kp, 5, 6), _mid(kp, 11, 12)
    if S is None or H is None:
        return None
    T = math.hypot(H[0] - S[0], H[1] - S[1])
    bh = bbox[3] - bbox[1]
    if T < 0.12 * bh or T > 0.6 * bh:  # torso imposible (puntos mal puestos o persona agachada del todo)
        return None
    if torso_hint:
        T = torso_hint
    u = ((H[0] - S[0]) / max(1e-6, math.hypot(H[0] - S[0], H[1] - S[1])),
         (H[1] - S[1]) / max(1e-6, math.hypot(H[0] - S[0], H[1] - S[1])))
    v = (-u[1], u[0])
    b = Body(S, H, T, u, v, 0.0, False, False)
    acr = [abs(b.local(_pt(kp, i))[1]) for i in (5, 6, 11, 12) if _pt(kp, i)[2] >= GOOD]
    b.half_width = max(acr) if acr else 0.2
    ls, rs = _pt(kp, 5), _pt(kp, 6)
    b.profile = (ls[2] < GOOD or rs[2] < GOOD) or math.hypot(ls[0] - rs[0], ls[1] - rs[1]) < 0.4 * T
    b.facing_away = max(_pt(kp, i)[2] for i in (0, 1, 2)) < SEEN
    return b


class GestureEngine:
    """Lleva el estado de los gestos de cada persona. Lo llama el RuleEngine en cada cuadro (core/rules.py)."""

    POSTURE_SECONDS = 15.0  # una mano que se queda tanto en la región es postura (manos en los bolsillos)
    BILATERAL_SECONDS = 0.5  # las dos manos entran a la vez: acomodarse la ropa
    RETRACT_SECONDS = 3.0  # la mano vuelve al cuerpo hasta tanto después del alcance
    GRACE_SECONDS = 0.35  # un cuadro perdido no corta la permanencia
    OCCLUSION_SECONDS = 1.5  # muñeca tapada que sigue contando como «adentro»
    STILL_SPEED = 0.35  # «parada»: las caderas se mueven menos de tanto (altos de su cuerpo por segundo)
    SETTLE_SECONDS = 1.5  # recién aparecida, una persona no «toma» nada todavía

    def __init__(self, cfg: GestureConfig):
        self.cfg = cfg
        self.people: dict[int, _Person] = {}

    def forget(self, live: set[int]) -> None:
        for tid in [k for k in self.people if k not in live]:
            del self.people[tid]

    def step(self, tid: int, kp, bbox, ts: float, *, near_shelf: bool, at_checkout: bool, staff: bool,
             bags: list | None = None, others: list | None = None) -> list[GestureHit]:
        """Un cuadro de la persona `tid`. `near_shelf`: está (o estuvo hace poco) en una góndola, o la cámara no
        tiene góndolas marcadas. `bags`: recuadros de bolsos, carteras o mochilas que ve el detector. `others`:
        recuadros de las otras personas (estirar el brazo hacia otra persona, pasarle algo, no es tomar)."""
        cfg = self.cfg
        if tid not in self.people:
            self.people[tid] = _Person(first_ts=ts)
        p = self.people[tid]
        if kp is None or not cfg.enabled or bbox[3] - bbox[1] < cfg.min_person_px:
            self._lost(p, ts)
            return []
        p.frames_seen += 1
        hint = sorted(p.torso)[len(p.torso) // 2] if len(p.torso) >= 3 else None
        body = body_of(kp, bbox, hint)
        if body is None:
            self._lost(p, ts)
            return []
        raw = math.hypot(body.H[0] - body.S[0], body.H[1] - body.S[1])
        p.torso.append(raw)
        # ¿parada? por la mitad de las caderas (el recuadro se agranda al estirar un brazo; las caderas no)
        bh = max(1.0, bbox[3] - bbox[1])
        p.feet.append((ts, body.H[0] / bh, body.H[1] / bh))
        still = p.speed(ts) <= self.STILL_SPEED
        p.frames_ok += 1
        p.bad_frames = 0
        center = ((body.S[0] + body.H[0]) / 2, (body.S[1] + body.H[1]) / 2)
        hits: list[GestureHit] = []
        entered: dict[str, float] = {}
        for side, (si, ei, wi, hi) in SIDES.items():
            h = p.hands[side]
            sh, el, wr, hip = _pt(kp, si), _pt(kp, ei), _pt(kp, wi), _pt(kp, hi)
            # la mano junto a otra persona (le pasa algo, la toma de la mano, un chico): no es bolsillo ni bolsa
            with_other = wr[2] >= SEEN and _inside_any(wr, others)
            region = None if with_other else self._region(body, sh, el, wr, hip, h, ts, bbox, bags)
            # alcance → toma
            # tomar del estante se hace parado, y no apenas aparece (entrando por el borde el cuerpo se ve cortado)
            settled = ts - p.first_ts >= self.SETTLE_SECONDS
            if still and settled and self._reaching(body, sh, el, wr) and not with_other:
                if h.reach_since is None:
                    h.reach_since, h.reach_frames = ts, 0
                h.reach_frames += 1
                if h.reach_frames >= 2 and (h.reached_ts is None or ts - h.reached_ts > 0.5):
                    if h.reached_ts is None or ts - h.reached_ts > self.RETRACT_SECONDS:
                        p.reaches += 1
                    h.reached_ts = ts
            else:
                h.reach_since, h.reach_frames = None, 0
                if (h.reached_ts is not None and ts - h.reached_ts <= self.RETRACT_SECONDS and wr[2] >= SEEN
                        and body.dist(wr, center) <= 0.8 and near_shelf):
                    p.take_ts, p.take_side = ts, side
                    p.take_hist.append(ts)
                    p.takes += 1
                    h.reached_ts = None
            # permanencia en la región (con un respiro de un cuadro)
            if region is not None and region == h.region:
                h.last_in = ts
                h.frames += 1
            elif region is not None:
                h.region, h.since, h.last_in, h.frames = region, ts, ts, 1
                entered[side] = ts
            elif h.region is not None and ts - h.last_in > self.GRACE_SECONDS:
                h.region, h.frames = None, 0
            if region is None and wr[2] >= SEEN:
                h.free_ts = ts
            if h.region is not None:
                dwell = h.last_in - h.since
                p.max_dwell[h.region] = max(p.max_dwell[h.region], dwell)
                if wr[2] < HIDDEN and el[2] >= GOOD:
                    p.conf_drop = max(p.conf_drop, 1.0 - wr[2])
        # ¿alguno ya cumple?
        for side, h in p.hands.items():
            if h.region is None:
                continue
            gesture = GESTURE_OF[h.region]
            if gesture not in cfg.enabled:
                continue
            dwell = h.last_in - h.since
            need = {"bolsillo": cfg.pocket_seconds, "ropa": cfg.clothes_seconds, "bolsa": cfg.bag_seconds}[h.region]
            weak_bag = h.region == "bolsa" and not self._near_bag(kp, side, body, bags)
            if weak_bag:
                need *= 2
                if not still:  # sin bolso a la vista, «guardar» es parada: caminando con algo en brazos es normal
                    h.since, h.frames = ts, 1
                    continue
            if dwell + 1e-6 < need or h.frames < 3:
                continue
            since_take = self._since_take(p, h.since)
            why = self._suppress(p, h, side, gesture, dwell, since_take, at_checkout, staff, near_shelf)
            if why:
                p.suppressed[why] = p.suppressed.get(why, 0) + 1
                continue
            last = p.fired.get(gesture)
            if (last is not None and ts - last < cfg.cooldown_seconds) or ts - p.last_fire < 10.0:
                continue
            p.fired[gesture], p.last_fire = ts, ts
            hits.append(self._hit(tid, p, h, side, gesture, dwell, since_take, weak_bag, body, kp))
        return hits

    # ------------------------------------------------------------------ geometría
    def _reaching(self, body: Body, sh, el, wr) -> bool:
        """La mano lejos del cuerpo, a la altura del pecho para arriba (hacia el estante): no colgando ni sosteniendo
        algo delante. Pegado a la góndola el brazo no se estira del todo (Minas, Pasillo bebidas: codo a 85-115°), así
        que se mira dónde está la muñeca y no solo el codo."""
        if wr[2] < GOOD or sh[2] < GOOD:
            return False
        if el[2] >= SEEN:
            arm = math.hypot(sh[0] - el[0], sh[1] - el[1]) + math.hypot(el[0] - wr[0], el[1] - wr[1])
            if arm <= 1e-6 or math.hypot(wr[0] - sh[0], wr[1] - sh[1]) / arm < 0.6:  # brazo plegado: no
                return False
        if body.dist(wr, sh) < 0.75:
            return False
        along, across = body.local(wr)
        out_front = abs(across) >= body.half_width + 0.6 and along <= 0.65
        up = along < -0.15  # por encima de los hombros (estante alto)
        low_far = along > 0.65 and abs(across) >= body.half_width + 1.0  # estante bajo, agachado hacia el costado
        return out_front or up or low_far

    def _region(self, body: Body, sh, el, wr, hip, h: _Hand, ts: float, bbox, bags) -> str | None:
        """En qué región está la muñeca de este lado (o None)."""
        elbow = _angle(sh, el, wr) if (el[2] >= SEEN and wr[2] >= SEEN and sh[2] >= SEEN) else None
        if wr[2] < HIDDEN:
            # pista de oclusión: la muñeca dejó de verse con el codo a la vista, justo después de estar en la región
            if (h.region in ("bolsillo", "ropa") and el[2] >= GOOD and not body.facing_away
                    and ts - h.last_in <= self.OCCLUSION_SECONDS):
                return h.region
            return None
        along, across = body.local(wr)
        # bolsillo: junto a la cadera del mismo lado, con histéresis
        if hip[2] >= GOOD:
            d_hip = body.dist(wr, hip)
            limit = 0.5 if h.region == "bolsillo" else 0.35
            _, hip_across = body.local(hip)
            outside = abs(across) - abs(hip_across)
            if (d_hip <= limit and 0.9 <= along <= 1.5 and outside <= 0.15
                    and (elbow is None or elbow <= 160)):
                return "bolsillo"
        # bolsa: un bolso detectado junto a la mano
        if bags:
            for b in bags:
                m = 0.15 * body.T
                if b[0] - m <= wr[0] <= b[2] + m and b[1] - m <= wr[1] <= b[3] + m:
                    return "bolsa"
        # ropa: dentro del torso, más una señal de que no es «sostener el producto delante»
        band = 0.35 if body.profile else 0.9 * body.half_width
        if 0.05 <= along <= 0.95 and abs(across) <= band and (elbow is None or elbow <= 120):
            _, sh_across = body.local(sh)
            crosses = sh_across * across < 0 and abs(across) > 0.1  # pasó al otro lado del cuerpo
            waist = along >= 0.75 and abs(across) <= max(0.12, 0.35 * body.half_width)
            if crosses or waist:
                return "ropa"
        # bolsa sin bolso detectado: manos bajas delante del cuerpo, codo doblado
        if 0.55 <= along <= 1.35 and 0.3 <= abs(across) <= 1.0 and elbow is not None and elbow <= 150:
            return "bolsa"
        return None

    @staticmethod
    def _since_take(p: _Person, entered: float) -> float | None:
        """Segundos entre la última toma antes de que la mano entrara a la región y la entrada (una toma justo
        después de entrar también vale: 0)."""
        before = [x for x in p.take_hist if x <= entered + 0.5]
        if before:
            return max(0.0, entered - max(before))
        return 0.0 if p.take_hist else None

    def _near_bag(self, kp, side: str, body: Body, bags) -> bool:
        if not bags:
            return False
        wr = _pt(kp, SIDES[side][2])
        m = 0.15 * body.T
        return any(b[0] - m <= wr[0] <= b[2] + m and b[1] - m <= wr[1] <= b[3] + m for b in bags)

    # ------------------------------------------------------------------ descartes y aviso
    def _suppress(self, p: _Person, h: _Hand, side: str, gesture: str, dwell: float, since_take: float | None,
                  at_checkout: bool, staff: bool, near_shelf: bool) -> str | None:
        cfg = self.cfg
        if staff:
            return "zona de personal"
        if gesture == MANO_BOLSILLO and at_checkout:
            return "en la caja (pagar)"
        if dwell >= self.POSTURE_SECONDS:
            return "postura (mano quieta mucho rato)"
        if p.take_hist:
            took = [x for x in p.take_hist if x <= h.since + 0.5]
            if not took or h.free_ts < max(took) - 0.5:
                # la mano ya estaba en el bolsillo o la cintura (apoyada) cuando tomó, o no estuvo libre desde la toma:
                # no la llevó ahí después de tomar
                return "la mano ya estaba ahí antes de tomar"
        other = p.hands["der" if side == "izq" else "izq"]
        if (gesture != MANO_BOLSA and other.region == h.region
                and abs(other.since - h.since) <= self.BILATERAL_SECONDS):
            return "las dos manos a la vez (acomodarse)"
        if cfg.require_take:
            if since_take is None:
                return "sin toma vista antes"
            if since_take < -0.5 or since_take > cfg.take_window:
                return "la toma fue hace mucho"
        return None

    def _hit(self, tid, p: _Person, h: _Hand, side, gesture, dwell, since_take, weak_bag, body, kp) -> GestureHit:
        mano = "mano " + SIDE_NAMES[side]
        what = {MANO_BOLSILLO: f"{mano} al bolsillo", MANO_ROPA: f"{mano} dentro de la ropa",
                MANO_BOLSA: f"{mano} en la bolsa" + ("" if not weak_bag else " o a la cintura")}[gesture]
        after = (" (no se vio que tomara algo)" if since_take is None else
                 ", enseguida después de tomar algo" if since_take < 1.5 else
                 f", {since_take:.0f} s después de tomar algo")
        desc = f"Persona #{tid}: posible ocultamiento, {what} {dwell:.1f} s{after}"
        feats = {"gesture_dwell_s": dwell, "since_take_s": -1.0 if since_take is None else max(0.0, since_take),
                 "takes": float(p.takes), "reaches": float(p.reaches), "wrist_conf_drop": p.conf_drop,
                 "torso_px": body.T, "bag_seen": 0.0 if weak_bag or gesture != MANO_BOLSA else 1.0,
                 "pose_ok_ratio": p.frames_ok / max(1, p.frames_seen)}
        return GestureHit(gesture, side, dwell, since_take, desc, {k: round(float(v), 4) for k, v in feats.items()})

    def _lost(self, p: _Person, ts: float) -> None:
        p.bad_frames += 1
        if p.bad_frames >= 5:  # ~1 s sin ver bien a la persona: las permanencias se cortan
            for h in p.hands.values():
                h.region, h.frames, h.reach_since, h.reach_frames = None, 0, None, 0

    # ------------------------------------------------------------------ para mostrar
    def progress(self, tid: int, ts: float) -> dict | None:
        """Estado del gesto de una persona: qué mano está en qué región y hace cuánto (y lo que pide), y si tomó algo
        hace poco. Lo dibuja «Configurar…»."""
        p = self.people.get(tid)
        if p is None:
            return None
        cfg = self.cfg
        need = {"bolsillo": cfg.pocket_seconds, "ropa": cfg.clothes_seconds, "bolsa": cfg.bag_seconds}
        hands = {s: (h.region, h.last_in - h.since, need[h.region]) for s, h in p.hands.items() if h.region}
        reaching = [s for s, h in p.hands.items() if h.reach_frames >= 2]
        took = None if p.take_ts is None or ts - p.take_ts > cfg.take_window else ts - p.take_ts
        return {"hands": hands, "reaching": reaching, "took_ago": took}

    def summary(self, tid: int) -> dict | None:
        """Lo medido de una persona (para explicar por qué dio o no alerta)."""
        p = self.people.get(tid)
        if p is None:
            return None
        return {"takes": p.takes, "reaches": p.reaches, "max_dwell": dict(p.max_dwell),
                "suppressed": dict(p.suppressed), "pose_ok_ratio": p.frames_ok / max(1, p.frames_seen),
                "frames": p.frames_seen}
