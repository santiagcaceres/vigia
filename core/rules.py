"""Motor de reglas: convierte trayectorias de personas en eventos sospechosos.

El sistema no "sabe" que alguien robó: marca comportamientos que suelen acompañar un hurto
y cada evento lleva un conjunto de medidas (features) que después usa el modelo entrenado
con las etiquetas de la persona de entrenamiento para separar alertas reales de falsas.

Zonas (polígonos en coordenadas normalizadas 0..1):
  shelf       góndola / exhibidor con mercadería
  checkout    caja
  exit        salida del local
  restricted  zona prohibida (depósito, trastienda, detrás del mostrador)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, time

from .gestos import GESTURE_TYPES, MANO_BOLSA, MANO_BOLSILLO, MANO_ROPA, GestureConfig, GestureEngine
from .geometry import BBox, Point, bbox_foot, point_in_polygon
from .schedule import is_open, parse_hours
from .tracker import Track

# shelf/checkout/exit/restricted: reglas de hurto. entrance: conteo de ingresos (comercios); con `direction`,
# cuenta por separado los que entran y los que salen. pump/shop: estaciones de servicio (surtidor y tienda).
# staff: lugar de empleados (detrás del mostrador, depósito): quien lo pisa no cuenta como cliente en las métricas.
# aisle: pasillo (por donde se camina): solo métricas («a dónde van»); las reglas de hurto no lo miran.
ZONE_TYPES = ("shelf", "checkout", "exit", "restricted", "entrance", "pump", "shop", "staff", "aisle")
# tipos que un agente viejo no conoce: viajan aparte (`extra_zones`) para que no deje de usar las demás zonas
NEW_ZONE_TYPES = ("staff", "aisle")
# hacia dónde se mueve en la imagen una persona que ENTRA al local (zona entrance)
DIRECTIONS = {"abajo": (0.0, 1.0), "arriba": (0.0, -1.0), "derecha": (1.0, 0.0), "izquierda": (-1.0, 0.0)}
VEHICLE_LABEL = "vehiculo"
# una zona dibujada casi hasta el borde de la imagen va hasta el borde: a quien le quedan los pies cortados por el
# borde, el punto de los pies cae pegado al borde (Pasillo bebidas de Minas, 28 oct: la zona llegaba al 98 % del alto,
# la señora pisaba en el 99 % y nunca «estuvo» en la zona)
EDGE_SNAP = 0.03

SALIDA_SIN_PAGO = "salida_sin_pago"
MERODEO = "merodeo"
ZONA_RESTRINGIDA = "zona_restringida"
FUERA_DE_HORARIO = "fuera_de_horario"
RULE_TYPES = (SALIDA_SIN_PAGO, MERODEO, ZONA_RESTRINGIDA, FUERA_DE_HORARIO)
# Alarma del DVR (llegada por Hik-Partner Pro) con persona, sin ninguna regla disparada en el clip.
ALARMA_DVR = "alarma_dvr"
# Los gestos (core/gestos.py) son tipos de evento pero NO reglas de `rules`: un agente viejo rechaza nombres que no
# conoce en enabled_rules o cooldown_seconds. Se configuran aparte, en analysis["gestures"].
EVENT_TYPES = RULE_TYPES + (ALARMA_DVR,) + GESTURE_TYPES
# De más a menos grave: para elegir el tipo de un clip de alarma donde se disparan varias reglas.
RULE_PRIORITY = (SALIDA_SIN_PAGO, MANO_ROPA, MANO_BOLSILLO, MANO_BOLSA, ZONA_RESTRINGIDA, MERODEO, FUERA_DE_HORARIO)

SEVERITY = {SALIDA_SIN_PAGO: "alta", ZONA_RESTRINGIDA: "alta", MERODEO: "media", FUERA_DE_HORARIO: "media",
            ALARMA_DVR: "baja", MANO_BOLSILLO: "alta", MANO_ROPA: "alta", MANO_BOLSA: "media"}
BAG_LABELS = ("mochila", "cartera", "valija")  # objetos del detector (core/escena.py) que cuentan como bolsa
DEFAULT_COOLDOWNS = {SALIDA_SIN_PAGO: 8.0, MERODEO: 30.0, ZONA_RESTRINGIDA: 15.0, FUERA_DE_HORARIO: 300.0}
_NUMERIC_KEYS = ("shelf_dwell_seconds", "checkout_min_seconds", "loiter_seconds", "restricted_min_seconds")


@dataclass(frozen=True)
class Zone:
    name: str
    type: str
    polygon: tuple[Point, ...]
    direction: str | None = None  # solo entrance: hacia dónde se entra (DIRECTIONS)

    def contains(self, nx: float, ny: float) -> bool:
        return point_in_polygon(nx, ny, self.polygon)


@dataclass
class RuleConfig:
    shelf_dwell_seconds: float = 3.0
    checkout_min_seconds: float = 1.0
    loiter_seconds: float = 45.0
    restricted_min_seconds: float = 1.0
    cooldown_seconds: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_COOLDOWNS))
    opening_hours: tuple[time, time] | None = None
    # horario del local por día (core/schedule.py); si está, manda sobre opening_hours
    schedule: dict | None = None
    enabled_rules: tuple[str, ...] = RULE_TYPES
    # gestos de ocultamiento (viajan aparte de `rules`, en analysis["gestures"]); vacío = apagados
    gestures: GestureConfig = field(default_factory=GestureConfig)

    @classmethod
    def from_dict(cls, d: dict | None) -> "RuleConfig":
        d = dict(d or {})
        cfg = cls()
        for k in _NUMERIC_KEYS:
            if k in d:
                v = float(d.pop(k))
                if v < 0:
                    raise ValueError(f"{k} no puede ser negativo")
                setattr(cfg, k, v)
        cd = d.pop("cooldown_seconds", None)
        if cd:
            for k, v in cd.items():
                if k not in DEFAULT_COOLDOWNS:
                    raise ValueError(f"cooldown_seconds: regla desconocida {k!r}")
                cfg.cooldown_seconds[k] = float(v)
        oh = d.pop("opening_hours", None)
        if oh:
            cfg.opening_hours = (_hhmm(oh["start"]), _hhmm(oh["end"]))
        er = d.pop("enabled_rules", None)
        if er is not None:
            bad = [r for r in er if r not in RULE_TYPES]
            if bad:
                raise ValueError(f"enabled_rules: reglas desconocidas {bad}")
            cfg.enabled_rules = tuple(er)
        if d:
            raise ValueError(f"Reglas desconocidas: {sorted(d)}")
        return cfg


def _hhmm(s: str) -> time:
    try:
        hh, mm = s.split(":")
        return time(int(hh), int(mm))
    except Exception as exc:
        raise ValueError(f"Hora inválida {s!r}, usar 'HH:MM'") from exc


def snap_edge(v: float) -> float:
    """Una coordenada (0-1) a menos de `EDGE_SNAP` de un borde de la imagen, pegada al borde."""
    return 0.0 if v < EDGE_SNAP else 1.0 if v > 1.0 - EDGE_SNAP else v


def parse_zones(items: list[dict] | None) -> list[Zone]:
    zones: list[Zone] = []
    names: set[str] = set()
    for z in items or []:
        name, ztype = z.get("name"), z.get("type")
        if not name:
            raise ValueError("Cada zona necesita 'name'")
        if name in names:
            raise ValueError(f"Zona repetida: {name!r}")
        if ztype not in ZONE_TYPES:
            raise ValueError(f"Tipo de zona inválido {ztype!r}; usar uno de {ZONE_TYPES}")
        poly = tuple((float(x), float(y)) for x, y in z.get("polygon", []))
        if len(poly) < 3:
            raise ValueError(f"La zona {name!r} necesita al menos 3 puntos")
        if any(not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0) for x, y in poly):
            raise ValueError(f"La zona {name!r} debe usar coordenadas normalizadas entre 0 y 1")
        poly = tuple((snap_edge(x), snap_edge(y)) for x, y in poly)
        direction = z.get("direction") or None
        if direction is not None and (ztype != "entrance" or direction not in DIRECTIONS):
            raise ValueError(f"La zona {name!r}: el sentido de entrada va solo en zonas de entrada "
                             f"({', '.join(DIRECTIONS)})")
        names.add(name)
        zones.append(Zone(name, ztype, poly, direction))
    return zones


def parse_config(config: dict | None) -> tuple[list[Zone], RuleConfig]:
    config = config or {}
    cfg = RuleConfig.from_dict(config.get("rules"))
    # el horario del local viaja aparte de las reglas (un agente viejo lo ignora en vez de fallar)
    cfg.schedule = parse_hours(config.get("schedule")) or None
    # los gestos también (un agente viejo ignora la clave)
    cfg.gestures = GestureConfig.from_dict(config.get("gestures"))
    # las zonas de tipos nuevos viajan aparte (un agente viejo no conoce el tipo y rechazaría todas las zonas)
    extra = list(config.get("extra_zones") or config.get("staff_zones") or [])
    return parse_zones(list(config.get("zones") or []) + extra), cfg


@dataclass
class RuleEvent:
    type: str
    severity: str
    track_id: int
    zone: str | None
    description: str
    ts: float
    bbox: BBox
    features: dict[str, float] = field(default_factory=dict)
    track_start_ts: float = 0.0  # desde cuándo está la persona en escena (para que el clip arranque ahí)


@dataclass
class _TrackState:
    first_ts: float
    last_ts: float
    last_foot: Point | None = None
    last_shelf: tuple[str, float] | None = None  # última góndola pisada y cuándo (para los gestos)
    zone_since: dict[str, float] = field(default_factory=dict)
    totals: dict[str, float] = field(default_factory=lambda: {t: 0.0 for t in ZONE_TYPES})
    visits: dict[str, int] = field(default_factory=lambda: {t: 0 for t in ZONE_TYPES})
    max_shelf_dwell: float = 0.0
    path_len: float = 0.0
    pending_items: bool = False
    pending_zone: str | None = None
    pending_dwell: float = 0.0
    fired: set[tuple[str, str | None]] = field(default_factory=set)


class RuleEngine:
    NEAR_SHELF_SECONDS = 5.0  # la toma cuenta si la persona pisó una góndola hasta tanto antes

    def __init__(self, zones: list[Zone], config: RuleConfig | None = None):
        self.zones = zones
        self.cfg = config or RuleConfig()
        self._states: dict[int, _TrackState] = {}
        self._last_fired: dict[tuple[str, str | None], float] = {}
        self.gestures = GestureEngine(self.cfg.gestures) if self.cfg.gestures.enabled else None

    def update(self, tracks: list[Track], ts: float, frame_size: tuple[int, int],
               wall_time: datetime | None = None, objects: list | None = None) -> list[RuleEvent]:
        """`objects`: lo demás que ve el detector en el cuadro (Detection con label y bbox); de ahí salen los bolsos,
        carteras y mochilas para el gesto «mano en la bolsa»."""
        w, h = frame_size
        events: list[RuleEvent] = []
        live = {t.id for t in tracks}
        for tid in [k for k in self._states if k not in live]:
            del self._states[tid]
        if self.gestures is not None:
            self.gestures.forget(live)
        bags = [o.bbox for o in objects or () if getattr(o, "label", "") in BAG_LABELS]
        has_shelves = any(z.type == "shelf" for z in self.zones)
        visible = sum(1 for t in tracks if t.visible and t.label != VEHICLE_LABEL)

        for t in tracks:
            if t.label == VEHICLE_LABEL:
                continue  # las reglas de hurto son solo para personas
            st = self._states.setdefault(t.id, _TrackState(first_ts=t.first_ts, last_ts=ts))
            if not t.visible:
                continue
            dt = min(max(ts - st.last_ts, 0.0), 1.0)
            st.last_ts = ts

            fx, fy = bbox_foot(t.bbox)
            nx, ny = fx / w, fy / h
            if st.last_foot is not None:
                st.path_len += math.hypot(nx - st.last_foot[0], ny - st.last_foot[1])
            st.last_foot = (nx, ny)

            now = {z.name: z for z in self.zones if z.contains(nx, ny)}
            entered = [n for n in now if n not in st.zone_since]
            for n in entered:
                st.zone_since[n] = ts
                st.visits[now[n].type] += 1
            for n in [n for n in st.zone_since if n not in now]:
                del st.zone_since[n]
            for ztype in {z.type for z in now.values()}:
                st.totals[ztype] += dt

            def emit(rule: str, zone: str | None, desc: str) -> None:
                self._emit(events, rule, t, st, zone, ts, desc, visible, (w, h), wall_time)

            for name, zone in now.items():
                dwell = ts - st.zone_since[name]
                if zone.type == "shelf":
                    st.last_shelf = (name, ts)
                    st.max_shelf_dwell = max(st.max_shelf_dwell, dwell)
                    if dwell >= self.cfg.shelf_dwell_seconds and not st.pending_items:
                        st.pending_items, st.pending_zone, st.pending_dwell = True, name, dwell
                    elif st.pending_items and st.pending_zone == name:
                        st.pending_dwell = max(st.pending_dwell, dwell)
                    if dwell >= self.cfg.loiter_seconds:
                        emit(MERODEO, name, f"Persona #{t.id} lleva {dwell:.0f}s frente a «{name}»")
                elif zone.type == "checkout":
                    if dwell >= self.cfg.checkout_min_seconds:
                        st.pending_items, st.pending_zone = False, None
                elif zone.type == "restricted":
                    if dwell >= self.cfg.restricted_min_seconds:
                        emit(ZONA_RESTRINGIDA, name, f"Persona #{t.id} en zona restringida «{name}»")
                elif zone.type == "exit":
                    if name in entered and st.pending_items:
                        emit(SALIDA_SIN_PAGO, name,
                             f"Persona #{t.id} salió por «{name}» tras {st.pending_dwell:.0f}s en "
                             f"«{st.pending_zone}» sin pasar por caja")

            if self.gestures is not None:
                self._gestures(t, st, now, ts, has_shelves, bags, events, visible, (w, h), wall_time)

            if self.cfg.schedule and wall_time is not None:
                if not is_open(self.cfg.schedule, wall_time, 0):
                    emit(FUERA_DE_HORARIO, None, f"Persona #{t.id} detectada con el local cerrado")
            elif self.cfg.opening_hours and wall_time is not None:
                start, end = self.cfg.opening_hours
                if not _within(wall_time.time(), start, end):
                    emit(FUERA_DE_HORARIO, None,
                         f"Persona #{t.id} detectada fuera de horario ({start:%H:%M}-{end:%H:%M})")
        return events

    def _gestures(self, t: Track, st: _TrackState, now: dict, ts: float, has_shelves: bool, bags: list,
                  events: list, visible: int, size: tuple[int, int], wall) -> None:
        """Gestos de ocultamiento de una persona (si se le miró la postura en este cuadro)."""
        near = (not has_shelves or any(z.type == "shelf" for z in now.values())
                or (st.last_shelf is not None and ts - st.last_shelf[1] <= self.NEAR_SHELF_SECONDS))
        mine = [b for b in bags if _belongs(b, t.bbox)]
        hits = self.gestures.step(t.id, t.kpts, t.bbox, ts, near_shelf=near,
                                  at_checkout=any(z.type == "checkout" for z in now.values()),
                                  staff=any(z.type == "staff" for z in now.values()), bags=mine)
        person = self.gestures.people.get(t.id)
        if person is not None and person.take_ts == ts and st.last_shelf is not None and not st.pending_items:
            # una toma de verdad (brazo al estante y vuelta) frente a una góndola: «tomó algo»
            st.pending_items, st.pending_zone, st.pending_dwell = True, st.last_shelf[0], 0.0
        for hit in hits:
            zone = st.last_shelf[0] if st.last_shelf is not None else None
            feats = {**track_features(t, st, ts, visible, size, wall), **hit.features}
            events.append(RuleEvent(hit.type, SEVERITY[hit.type], t.id, zone, hit.description, ts, t.bbox, feats,
                                    st.first_ts))

    def _emit(self, events, rule, t, st, zone, ts, description, visible, size, wall) -> None:
        if rule not in self.cfg.enabled_rules:
            return
        key = (rule, zone)
        if key in st.fired:
            return
        last = self._last_fired.get(key)
        if last is not None and ts - last < self.cfg.cooldown_seconds.get(rule, 10.0):
            return
        st.fired.add(key)
        self._last_fired[key] = ts
        feats = track_features(t, st, ts, visible, size, wall)
        events.append(RuleEvent(rule, SEVERITY[rule], t.id, zone, description, ts, t.bbox, feats, st.first_ts))

    def progress(self, ts: float) -> dict[int, dict]:
        """Por persona (llamar después de `update`): en qué zonas está parada y hace cuántos segundos, y si ya
        «tomó algo» (estuvo en una góndola y todavía no pasó por caja). Lo muestra el ajuste de cámaras."""
        types = {z.name: z.type for z in self.zones}
        return {tid: {"zones": [(n, types.get(n, ""), ts - since) for n, since in st.zone_since.items()],
                      "took": st.pending_items,
                      "gesture": self.gestures.progress(tid, ts) if self.gestures is not None else None}
                for tid, st in self._states.items()}

    def current_features(self, tracks: list[Track], ts: float, frame_size: tuple[int, int],
                         wall_time: datetime | None = None) -> dict[int, dict[str, float]]:
        """Medidas actuales de cada persona visible (llamar después de update con el mismo cuadro)."""
        visible = sum(1 for t in tracks if t.visible and t.label != VEHICLE_LABEL)
        return {t.id: track_features(t, self._states[t.id], ts, visible, frame_size, wall_time)
                for t in tracks if t.visible and t.id in self._states}


def track_features(t: Track, st: _TrackState, ts: float, visible: int,
                   size: tuple[int, int], wall: datetime | None) -> dict[str, float]:
    age = max(ts - st.first_ts, 0.0)
    _, h = size
    feats = {
        "track_age_s": age,
        "shelf_total_s": st.totals["shelf"],
        "checkout_total_s": st.totals["checkout"],
        "restricted_total_s": st.totals["restricted"],
        "max_shelf_dwell_s": st.max_shelf_dwell,
        "shelf_visits": float(st.visits["shelf"]),
        "path_len": st.path_len,
        "mean_speed": st.path_len / max(age, 0.5),
        "people_in_scene": float(visible),
        "bbox_h": (t.bbox[3] - t.bbox[1]) / h if h else 0.0,
        "has_time": 0.0,
        "hour_sin": 0.0,
        "hour_cos": 0.0,
        # las medidas de los gestos (core/gestos.py) las completa la alerta de gesto
        "gesture_dwell_s": 0.0, "since_take_s": 0.0, "takes": 0.0, "reaches": 0.0, "wrist_conf_drop": 0.0,
        "bag_seen": 0.0, "pose_ok_ratio": 0.0,
    }
    if wall is not None:
        hour = wall.hour + wall.minute / 60.0
        feats.update(has_time=1.0, hour_sin=math.sin(2 * math.pi * hour / 24),
                     hour_cos=math.cos(2 * math.pi * hour / 24))
    return {k: round(float(v), 4) for k, v in feats.items()}


def _belongs(bag: BBox, person: BBox) -> bool:
    """El bolso es de esta persona: su centro cae dentro del recuadro de la persona agrandado 20 %."""
    cx, cy = (bag[0] + bag[2]) / 2, (bag[1] + bag[3]) / 2
    mx, my = 0.2 * (person[2] - person[0]), 0.2 * (person[3] - person[1])
    return person[0] - mx <= cx <= person[2] + mx and person[1] - my <= cy <= person[3] + my


def _within(now: time, start: time, end: time) -> bool:
    if start <= end:
        return start <= now <= end
    return now >= start or now <= end
