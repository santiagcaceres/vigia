"""Métricas de la escena: cuántas personas (y vehículos) pasan, cuánto se quedan y a dónde van.

Se alimenta con los mismos tracks del detector/tracker que usan las reglas de hurto, así que no
cuesta procesamiento extra. Cada persona o vehículo seguido es una **visita**: empieza cuando aparece
y termina cuando el tracker lo pierde. Al terminar se suma al **cubo horario** de la cámara
(hora de la pared del local). Cada cubo es un diccionario JSON que el servidor guarda tal cual y
suma entre cámaras y horas.

Qué se mide por cubo:
- `people`: visitas de personas; `people_dwell_s`: suma de segundos en escena (para el promedio);
  `people_max`: máximo de personas al mismo tiempo.
- `zones`: por nombre de zona, `{type, visits, dwell_s}` = cuántas personas entraron y cuánto tiempo
  sumaron ahí ("a dónde van").
- `entries` / `exits`: ingresos y egresos de clientes por la zona `entrance`. Con sentido (`direction`: hacia
  dónde se mueve en la imagen quien entra) cuenta cada cruce de la puerta por separado y `door` queda en 1; sin
  sentido (zonas viejas) `entries` cuenta a todos los que la pisaron, entren o salgan.
- `staff` / `staff_entries`: personas que se tomaron como empleados (pisaron una zona `staff` al menos 2 s, o
  aparecieron con el local cerrado según su horario) y sus ingresos: no cuentan en `people`, `entries` ni zonas.
- `vehicles`, `vehicles_dwell_s`, `vehicles_max`: lo mismo para vehículos (estaciones de servicio).
- `pump_vehicles`: vehículos que entraron a una zona `pump` (surtidor); `pump_wait_s`/`pump_wait_n`:
  segundos desde que el vehículo entró al surtidor hasta que llegó una persona a esa misma zona
  (tiempo de espera hasta el pistero); `pump_unattended`: vehículos que se fueron sin que llegara nadie.
- `shop_visits`: personas que entraron a una zona `shop` (tienda).
- `waits` (plan «comercio»): por zona marcada «medir espera» (`Zone.wait`: caja, mostrador, fila), cuánto esperaron
  los clientes. Cada cliente que estuvo ahí `WAIT_MIN_SECONDS` o más es una espera: `n`, `sum_s` (para el promedio),
  `max_s` y `hist` (cuántas esperas de menos de 1 min, 1-3, 3-5, 5-10 y 10 o más: `WAIT_EDGES`). Por cuadro, la fila:
  `obs_s` (segundos mirados), `queue_s` (personas-segundo esperando: promedio de la fila = queue_s / obs_s),
  `queue_max`, `busy_s` (segundos con alguien esperando); en la fila cuenta quien ya lleva `WAIT_MIN_SECONDS` en la
  zona (el que pasa de largo no). Si la cámara tiene zona de personal (`staffed` = 1): `unattended_s` (segundos con
  clientes esperando y nadie en **su puesto**: la zona de personal más cercana a la de espera, así un empleado en el
  depósito no cuenta como que atiende la caja) y `left_alone` (clientes que esperaron sin que apareciera nadie en ese
  puesto en todo ese rato).
- Plan «empleados»: `posts`, por zona de personal (el puesto: detrás del mostrador, la caja por dentro), `obs_s`,
  `covered_s` (segundos con alguien adentro), `people_s` (personas-segundo: promedio de gente en el puesto) y `max`;
  `staff_s` (personas-segundo de empleados a la vista en la cámara) y `staff_obs_s` (segundos mirados por la cámara:
  promedio de empleados a la vista = staff_s / staff_obs_s, sumando cámaras = promedio por cámara) y `staff_max`
  (máximo de empleados a la vez); `staff_dwell`: por zona, segundos que pasaron ahí los empleados (dónde trabajan), que
  se suman cuadro a cuadro (un turno de horas queda repartido en sus horas).
- Plan «gastronomía»: `tables`, por zona «Mesa» (la mesa con sus sillas y el lugar donde se para el mozo). Una
  **ocupación** empieza cuando un cliente lleva `TABLE_SEAT_SECONDS` en la mesa y termina cuando queda vacía
  `TABLE_GRACE_SECONDS`; cuenta si duró `TABLE_MIN_SECONDS` o más (limpiar la mesa no es ocuparla): `n`, `sum_s`,
  `max_s`, `party` (suma de comensales, el máximo sentados a la vez). Por cuadro: `obs_s` (segundos mirados) y `occ_s`
  (segundos ocupada). **Visitas a la mesa** (el mozo): alguien del personal (pisó una zona «Personal»), o cualquiera que
  llega caminando con la mesa ya ocupada, se queda entre `TABLE_VISIT_MIN` y `TABLE_VISIT_MAX` segundos y se va
  mientras los demás siguen sentados. La primera visita es la **atención** (`attend_n`, `attend_s`, `attend_max_s`:
  segundos desde que se sentaron); la segunda, la **entrega estimada** (`serve_n`, `serve_s`, `serve_max_s`);
  `visits` = todas; `no_visit` = ocupaciones que terminaron sin ninguna visita.
- `frames`: cuadros procesados (para saber si la cámara estuvo activa).

Lo que NO mide (y hay que decirlo): gestos (lavar vidrios, guardar algo) ni caras. A los empleados los reconoce
solo si pisan su zona (2 s o más) o llegan con el local cerrado: no sabe quién es quién ni sigue a una persona de una
cámara a otra. Quien pisa la zona de personal cuenta como «alguien en el puesto», sea o no empleado. La espera es el
tiempo en la zona marcada: si la zona incluye el lugar donde se atiende, incluye la atención. En una fila apretada,
quien queda tapado por otro no se ve. Una persona que el tracker pierde y vuelve a encontrar cuenta dos veces (salvo
en la puerta con sentido, donde cuenta el cruce). En las mesas no ve qué lleva el mozo en la mano: «entrega» es la
segunda vez que alguien se acerca a la mesa (la primera se toma como el pedido); un cliente que se suma tarde, deja
algo y se va cuenta como visita; y en un local donde se pide en el mostrador no hay visitas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .geometry import bbox_foot
from .rules import DIRECTIONS, VEHICLE_LABEL, Zone
from .schedule import is_open
from .tracker import Track

MIN_VISIT_SECONDS = 2.0  # menos que esto es ruido del detector
STAFF_MIN_SECONDS = 2.0  # en la zona de personal: menos que esto puede ser un cliente que se acercó al mostrador
CROSS_FRACTION = 0.4  # para contar un cruce, moverse al menos este tanto del ancho de la puerta en el sentido marcado
WAIT_MIN_SECONDS = 5.0  # en una zona de espera: menos que esto es pasar de largo, no esperar
WAIT_EDGES = (60, 180, 300, 600)  # cortes del histograma de esperas: < 1 min, 1-3, 3-5, 5-10 y 10 min o más
# entre dos cuadros cuenta como mucho esto (un corte del video no suma tiempo mirado). Más que el cuadro cada 2 s que
# analiza el filtro de movimiento (core/motion.py) con la escena quieta y vacía: si no, ese rato contaría la mitad
FRAME_GAP_MAX = 2.5
# mesas (plan «gastronomía»)
TABLE_SEAT_SECONDS = 10.0  # un cliente que lleva esto en la mesa se sentó (el que pasa entre las mesas no)
TABLE_GRACE_SECONDS = 60.0  # vacía este rato = se fueron (al que el detector pierde un momento no lo da por ido)
TABLE_MIN_SECONDS = 60.0  # una ocupación más corta no cuenta (limpiar o acomodar la mesa)
TABLE_VISIT_MIN = 2.0  # una visita a la mesa (el mozo) dura por lo menos esto…
TABLE_VISIT_MAX = 120.0  # …y, si no es del personal, como mucho esto (más es un comensal que llegó tarde)
TABLE_VISIT_AFTER = 5.0  # no cuenta como visita quien llega junto con los que se sientan
TABLE_VISIT_GAP = 45.0  # dos acercamientos más seguidos que esto son la misma visita
POST_GRACE_SECONDS = 1.5  # en el puesto: quien el detector dejó de ver hace menos que esto (tapado por el cliente) sigue ahí


def new_wait() -> dict:
    return {"n": 0, "sum_s": 0.0, "max_s": 0.0, "hist": [0] * (len(WAIT_EDGES) + 1), "obs_s": 0.0, "queue_s": 0.0,
            "queue_max": 0, "busy_s": 0.0, "unattended_s": 0.0, "left_alone": 0, "staffed": 0}


def new_post() -> dict:
    return {"obs_s": 0.0, "covered_s": 0.0, "people_s": 0.0, "max": 0}


def new_table() -> dict:
    return {"n": 0, "sum_s": 0.0, "max_s": 0.0, "party": 0, "obs_s": 0.0, "occ_s": 0.0, "visits": 0, "no_visit": 0,
            "attend_n": 0, "attend_s": 0.0, "attend_max_s": 0.0, "serve_n": 0, "serve_s": 0.0, "serve_max_s": 0.0}


def wait_bin(secs: float) -> int:
    return sum(1 for edge in WAIT_EDGES if secs >= edge)


def bucket_start(wall: datetime) -> datetime:
    return wall.replace(minute=0, second=0, microsecond=0)


def empty_bucket() -> dict:
    return {"people": 0, "people_dwell_s": 0.0, "people_max": 0, "entries": 0, "exits": 0, "door": 0,
            "staff": 0, "staff_entries": 0, "zones": {},
            "vehicles": 0, "vehicles_dwell_s": 0.0, "vehicles_max": 0,
            "pump_vehicles": 0, "pump_wait_s": 0.0, "pump_wait_n": 0, "pump_unattended": 0,
            "shop_visits": 0, "frames": 0,
            "waits": {}, "posts": {}, "staff_s": 0.0, "staff_obs_s": 0.0, "staff_max": 0, "staff_dwell": {},
            "tables": {}}


_MAX_KEYS = {"max_s", "queue_max", "staffed", "max", "attend_max_s", "serve_max_s"}  # dentro de waits/posts: se quedan con el mayor
MAX_NAMES = 64  # zonas por cubo: más que esto no es una cámara de verdad (se descarta lo que sobra)
MAX_VALUE = 1e9


def _num(v) -> float | int:
    """Un dato del cubo como número de verdad (0 si es texto, negativo, infinito o disparatado): un cubo roto que mande
    una PC no puede tirar abajo las métricas de todo el cliente."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return 0
    return v if math.isfinite(v) and 0 <= v <= MAX_VALUE else 0


def _merge_stats(a: dict, b: dict, blank: dict) -> dict:
    a, b = (a if isinstance(a, dict) else {}), (b if isinstance(b, dict) else {})
    out = dict(blank)
    for k, v0 in blank.items():
        va, vb = a.get(k, v0), b.get(k, v0)
        if isinstance(v0, list):
            n = len(v0)
            va = (list(va) if isinstance(va, list) else []) + [0] * n
            vb = (list(vb) if isinstance(vb, list) else []) + [0] * n
            out[k] = [_num(x) + _num(y) for x, y in zip(va[:n], vb[:n])]
        elif k in _MAX_KEYS:
            out[k] = max(_num(va), _num(vb))
        else:
            out[k] = _num(va) + _num(vb)
    return out


def merge_bucket(a: dict, b: dict) -> dict:
    """Suma dos cubos (de cámaras u horas distintas). Los máximos se quedan con el mayor."""
    out = dict(empty_bucket())
    a, b = (a if isinstance(a, dict) else {}), (b if isinstance(b, dict) else {})
    for k in out:
        if k in _DICT_KEYS:
            continue
        va, vb = _num(a.get(k, 0)), _num(b.get(k, 0))
        out[k] = max(va, vb) if k.endswith("_max") or k == "door" else va + vb
    for key, blank in (("waits", new_wait()), ("posts", new_post()), ("tables", new_table())):
        merged: dict[str, dict] = {}
        for src in (_names(a.get(key)), _names(b.get(key))):
            for name, st in src.items():
                merged[name] = _merge_stats(merged.get(name, blank), st or {}, blank)
        out[key] = merged
    dwell: dict[str, float] = {}
    for src in (_names(a.get("staff_dwell")), _names(b.get("staff_dwell"))):
        for name, secs in src.items():
            dwell[name] = dwell.get(name, 0.0) + _num(secs)
    out["staff_dwell"] = dwell
    zones: dict[str, dict] = {}
    for src in (_names(a.get("zones")), _names(b.get("zones"))):
        for name, z in src.items():
            z = z if isinstance(z, dict) else {}
            ztype = z.get("type") if isinstance(z.get("type"), str) else None
            cur = zones.setdefault(name, {"type": ztype and ztype[:20], "visits": 0, "dwell_s": 0.0})
            cur["visits"] += _num(z.get("visits", 0))
            cur["dwell_s"] += _num(z.get("dwell_s", 0.0))
    out["zones"] = zones
    return out


def _names(d) -> dict:
    """Un diccionario por nombre de zona, con nombres de texto cortos y como mucho MAX_NAMES."""
    if not isinstance(d, dict):
        return {}
    return {k[:120]: v for k, v in list(d.items())[:MAX_NAMES] if isinstance(k, str) and k}


# Qué parte del cubo es de cada plan del cliente (server: Client.plans / Camera.plans). Una cámara mide solo los planes
# que se le eligieron; lo demás que haya contado (una cámara de afuera también ve gente) no se suma al cliente.
PEOPLE_KEYS = ("people", "people_dwell_s", "people_max", "entries", "exits", "door", "staff", "staff_entries",
               "zones", "waits")  # plan «comercio» (la espera de los clientes también)
STATION_KEYS = ("vehicles", "vehicles_dwell_s", "vehicles_max", "pump_vehicles", "pump_wait_s", "pump_wait_n",
                "pump_unattended", "shop_visits")  # plan «estacion»
STAFF_KEYS = ("posts", "staff_s", "staff_obs_s", "staff_max", "staff_dwell", "staff",
              "staff_entries")  # plan «empleados» (cuántos empleados y sus ingresos, también de «comercio»)
GASTRO_KEYS = ("tables",)  # plan «gastronomia»
_DICT_KEYS = ("zones", "waits", "posts", "staff_dwell", "tables")


def restrict_bucket(data: dict, client_plans, camera_plans) -> dict:
    """Copia del cubo de una cámara con solo lo que esa cámara mide para su cliente: si el cliente tiene «comercio» y
    la cámara no lo mide, la gente queda en cero; si tiene «estacion» y la cámara no la mide, los vehículos; si tiene
    «empleados», lo del personal. Si el cliente no tiene ese plan no se toca (se ve lo de siempre). `frames` no se toca
    nunca."""
    out = dict(data or {})
    client_plans, camera_plans = set(client_plans or ()), set(camera_plans or ())
    blank = empty_bucket()
    owners: dict[str, set[str]] = {}  # dato -> planes del cliente que lo usan (`staff` es de «comercio» y «empleados»)
    for plan, keys in (("comercio", PEOPLE_KEYS), ("estacion", STATION_KEYS), ("empleados", STAFF_KEYS),
                       ("gastronomia", GASTRO_KEYS)):
        if plan in client_plans:
            for k in keys:
                owners.setdefault(k, set()).add(plan)
    for k, plans in owners.items():
        if not plans & camera_plans:  # ninguno de los planes que lo usan lo mide esta cámara
            out[k] = {} if k in _DICT_KEYS else blank[k]
    return out


def wait_stats(w: dict | None) -> dict:
    """Resumen de la espera en una zona (lo que muestran la app, la web, el PDF y el Excel): cuántos esperaron, el
    promedio y el máximo (segundos), cuántos esperaron 1, 3, 5 y 10 minutos o más (`over`, por segundos del corte),
    la fila promedio y máxima, y —si la cámara tiene zona de personal— los segundos con clientes esperando y nadie del
    personal y cuántos esperaron solos (None si no hay zona de personal)."""
    w = _merge_stats(new_wait(), w or {}, new_wait())
    n, hist = w["n"], w["hist"]
    over = {edge: sum(hist[i + 1:]) for i, edge in enumerate(WAIT_EDGES)}
    staffed = bool(w["staffed"])
    return {"n": n, "avg_s": (w["sum_s"] / n) if n else None, "max_s": w["max_s"] if n else None, "over": over,
            "queue_avg": (w["queue_s"] / w["obs_s"]) if w["obs_s"] else None, "queue_max": w["queue_max"],
            "busy_s": w["busy_s"], "obs_s": w["obs_s"], "staffed": staffed,
            "unattended_s": w["unattended_s"] if staffed else None, "left_alone": w["left_alone"] if staffed else None}


def post_stats(p: dict | None) -> dict:
    """Resumen de un puesto (zona de personal): qué parte del tiempo mirado hubo alguien (`covered_pct`, 0-100), los
    segundos sin nadie, el promedio y el máximo de gente en el puesto."""
    p = _merge_stats(new_post(), p or {}, new_post())
    obs = p["obs_s"]
    return {"obs_s": obs, "covered_s": p["covered_s"], "uncovered_s": max(0.0, obs - p["covered_s"]),
            "covered_pct": (100.0 * p["covered_s"] / obs) if obs else None,
            "avg_people": (p["people_s"] / obs) if obs else None, "max": p["max"]}


def table_stats(t: dict | None) -> dict:
    """Resumen de una mesa (o de varias sumadas): cuántas veces se ocupó, el promedio y el máximo en la mesa (segundos),
    qué parte del tiempo mirado estuvo ocupada (`occupied_pct`, 0-100), comensales por ocupación, cuánto tardó la
    primera visita (la atención) y la segunda (la entrega estimada), y cuántas ocupaciones no tuvieron ninguna."""
    t = _merge_stats(new_table(), t or {}, new_table())
    n, obs = t["n"], t["obs_s"]
    return {"n": n, "avg_s": (t["sum_s"] / n) if n else None, "max_s": t["max_s"] if n else None,
            "obs_s": obs, "occ_s": t["occ_s"], "occupied_pct": (100.0 * min(t["occ_s"], obs) / obs) if obs else None,
            "party_avg": (t["party"] / n) if n else None, "visits": t["visits"], "no_visit": t["no_visit"],
            "attend_n": t["attend_n"], "attend_avg_s": (t["attend_s"] / t["attend_n"]) if t["attend_n"] else None,
            "attend_max_s": t["attend_max_s"] if t["attend_n"] else None,
            "serve_n": t["serve_n"], "serve_avg_s": (t["serve_s"] / t["serve_n"]) if t["serve_n"] else None,
            "serve_max_s": t["serve_max_s"] if t["serve_n"] else None}


@dataclass
class _TableUse:
    """Una ocupación de una mesa, mientras dura."""
    start_ts: float
    last_busy_ts: float
    confirmed: bool = False
    party: int = 0
    visits: list[float] = field(default_factory=list)  # cuándo llegó cada visita (reloj del video)


@dataclass
class _Visit:
    label: str
    first_ts: float
    last_ts: float
    wall_first: datetime
    zone_dwell: dict[str, float] = field(default_factory=dict)  # nombre de zona -> segundos
    inside: set[str] = field(default_factory=set)
    last_seen_ts: float = 0.0
    last_pos: tuple[float, float] | None = None
    door_from: dict[str, tuple[float, float]] = field(default_factory=dict)  # puerta -> dónde estaba antes de pisarla
    crossings: list[int] = field(default_factory=list)  # +1 entró, -1 salió
    door_of: list[str] = field(default_factory=list)  # por qué puerta fue cada cruce (para contar cada puerta aparte)
    attended: set[str] = field(default_factory=set)  # zonas de espera donde, estando ahí, hubo alguien en el puesto
    table_from: dict[str, float] = field(default_factory=dict)  # mesa -> cuándo llegó caminando (posible visita del mozo)
    dwell_booked: bool = False  # empleado: lo que lleva en cada zona ya se sumó a `staff_dwell` (de ahí en más, por cuadro)


@dataclass
class _PumpWait:
    vehicle_id: int
    since_ts: float
    attended: bool = False


def crossing(zone: Zone, start: tuple[float, float], end: tuple[float, float]) -> int:
    """+1 si de `start` a `end` (pies, normalizados) se atravesó la puerta en el sentido de entrar, -1 en el
    contrario, 0 si no la atravesó (se asomó y volvió, o se quedó)."""
    dx, dy = DIRECTIONS[zone.direction or "abajo"]
    xs, ys = [p[0] for p in zone.polygon], [p[1] for p in zone.polygon]
    extent = (max(xs) - min(xs)) if dx else (max(ys) - min(ys))
    moved = (end[0] - start[0]) * dx + (end[1] - start[1]) * dy
    need = max(CROSS_FRACTION * extent, 0.02)
    return 1 if moved >= need else -1 if moved <= -need else 0


def _center(poly) -> tuple[float, float]:
    return sum(p[0] for p in poly) / len(poly), sum(p[1] for p in poly) / len(poly)


def nearest_post(zone: Zone, posts: list[Zone]) -> str | None:
    """La zona de personal que atiende una zona de espera: la más cercana (por el centro). Así, en una cámara con
    «Mostrador» y «Depósito», el empleado que está en el depósito no cuenta como que atiende la fila del mostrador."""
    if not posts:
        return None
    cx, cy = _center(zone.polygon)
    return min(posts, key=lambda z: (_center(z.polygon)[0] - cx) ** 2 + (_center(z.polygon)[1] - cy) ** 2).name


class MetricsCollector:
    def __init__(self, zones: list[Zone], schedule=None):
        self.zones = zones
        self.schedule = schedule  # horario del local: con el local cerrado, quien aparece es empleado
        self.by_type: dict[str, list[Zone]] = {}
        for z in zones:
            self.by_type.setdefault(z.type, []).append(z)
        self.doors = [z for z in self.by_type.get("entrance", []) if z.direction]
        self._staff_names = {z.name for z in self.by_type.get("staff", [])}
        self._wait_names = [z.name for z in zones if z.wait]
        self._post_of = {z.name: nearest_post(z, self.by_type.get("staff", [])) for z in zones if z.wait}
        self._table_names = [z.name for z in self.by_type.get("table", [])]
        self._tables: dict[str, _TableUse] = {}  # mesa -> la ocupación en curso
        self._last_frame_ts: float | None = None
        self._waiting: dict[str, int] = {}  # zona de espera -> clientes esperando en el último cuadro
        self._visits: dict[int, _Visit] = {}
        self._pump: dict[str, list[_PumpWait]] = {}  # zona surtidor -> esperas abiertas
        self._buckets: dict[datetime, dict] = {}
        self._dirty: set[datetime] = set()
        self._sent_before: datetime | None = None  # los cubos de antes de esta hora ya se mandaron y se olvidaron
        self._now: dict[str, int] = {}  # zona -> personas a la vista con los pies adentro en el último cuadro
        # lo ya terminado por zona que el cubo no separa (para mostrar en vivo): cruces de cada puerta y personal de
        # cada zona de personal
        self._door_done: dict[str, list[int]] = {}  # puerta -> [entraron, salieron]
        self._staff_done: dict[str, int] = {}  # zona de personal -> personas
        self.since: datetime | None = None  # hora del primer cuadro contado (para decir «desde las HH:MM»)

    # ---------------- por cuadro ----------------
    def update(self, tracks: list[Track], ts: float, frame_size: tuple[int, int], wall: datetime) -> None:
        w, h = frame_size
        if self.since is None:
            self.since = wall
        frame_dt = 0.0 if self._last_frame_ts is None else min(max(ts - self._last_frame_ts, 0.0), FRAME_GAP_MAX)
        self._last_frame_ts = ts
        b = self._bucket(wall)
        b["frames"] += 1
        if self.doors:
            b["door"] = 1
        live_ids = {t.id for t in tracks}
        visible_people = visible_vehicles = 0
        people_in_pump: dict[str, bool] = {}
        now_counts: dict[str, int] = {}
        in_post: dict[str, int] = {}  # zona de personal -> personas adentro ahora (sean o no empleados)
        waiting: dict[str, int] = {}  # zona de espera -> clientes esperando ahora
        waiting_visits: list[tuple[_Visit, set[str]]] = []
        staff_now = 0
        seated: dict[str, int] = {}  # mesa -> clientes sentados ahora (llevan TABLE_SEAT_SECONDS)
        at_table: dict[str, int] = {}  # mesa -> clientes en la mesa ahora (sentados o recién llegados)
        longest: dict[str, float] = {}  # mesa -> lo que lleva el que más lleva
        table_exits: list[tuple[str, float, float, bool]] = []  # (mesa, llegó, se quedó, ¿del personal?)
        for t in tracks:
            v = self._visits.get(t.id)
            if v is None:
                v = _Visit(t.label, t.first_ts, ts, wall - timedelta(seconds=max(0.0, ts - t.first_ts)), last_seen_ts=ts)
                self._visits[t.id] = v
            if not t.visible:
                # tapado (la cajera detrás del cliente que atiende): el puesto sigue ocupado un momento
                if t.label != VEHICLE_LABEL and ts - v.last_seen_ts <= POST_GRACE_SECONDS:
                    for name in self._staff_names & v.inside:
                        in_post[name] = in_post.get(name, 0) + 1
                    if self._staff_names & v.inside or self.is_staff(v):
                        staff_now += 1
                continue
            dt = min(max(ts - v.last_seen_ts, 0.0), 1.0)
            v.last_seen_ts = v.last_ts = ts
            if t.label == VEHICLE_LABEL:
                visible_vehicles += 1
            else:
                visible_people += 1
            fx, fy = bbox_foot(t.bbox)
            nx, ny = fx / w, fy / h
            now_in = {z.name for z in self.zones if z.contains(nx, ny)}
            for name in now_in:
                v.zone_dwell[name] = v.zone_dwell.get(name, 0.0) + dt
            if t.label != VEHICLE_LABEL:
                for z in self.doors:  # puerta: de dónde venía al pisarla y hacia dónde salió
                    if z.name in now_in and z.name not in v.inside:
                        v.door_from[z.name] = v.last_pos or (nx, ny)
                    elif z.name in v.inside and z.name not in now_in and z.name in v.door_from:
                        c = crossing(z, v.door_from.pop(z.name), (nx, ny))
                        if c:
                            v.crossings.append(c)
                            v.door_of.append(z.name)
                v.last_pos = (nx, ny)
            entered = now_in - v.inside
            if self._table_names and t.label != VEHICLE_LABEL:
                staff = self.is_staff(v)
                for name in self._table_names:
                    if name in entered and name not in v.table_from and ts - v.first_ts >= 1.0:
                        v.table_from[name] = ts  # llegó caminando (ya se lo venía viendo afuera de la mesa)
                    elif name in v.table_from and name not in now_in:
                        since = v.table_from.pop(name)
                        table_exits.append((name, since, ts - since, staff))
                    elif name in v.table_from and staff and ts - v.table_from[name] >= TABLE_VISIT_MIN:
                        since = v.table_from.pop(name)  # del personal: cuenta apenas se queda un momento
                        table_exits.append((name, since, ts - since, True))
                    if name in now_in and not staff:
                        at_table[name] = at_table.get(name, 0) + 1
                        secs = v.zone_dwell.get(name, 0.0)
                        longest[name] = max(longest.get(name, 0.0), secs)
                        if secs >= TABLE_SEAT_SECONDS:
                            seated[name] = seated.get(name, 0) + 1
            v.inside = now_in
            for name in entered:
                z = next(z for z in self.zones if z.name == name)
                if z.type == "pump":
                    if t.label == VEHICLE_LABEL:
                        self._pump.setdefault(name, []).append(_PumpWait(t.id, ts))
                        b["pump_vehicles"] += 1
            if t.label != VEHICLE_LABEL:
                for name in now_in:
                    now_counts[name] = now_counts.get(name, 0) + 1
                    if any(z.name == name and z.type == "pump" for z in self.zones):
                        people_in_pump[name] = True
                posts_now = self._staff_names & now_in
                for name in posts_now:
                    in_post[name] = in_post.get(name, 0) + 1
                if posts_now or self.is_staff(v):
                    staff_now += 1
                    if self.is_staff(v) and v.last_ts - v.first_ts >= MIN_VISIT_SECONDS:
                        # dónde pasan el tiempo los empleados, cuadro a cuadro: un turno de horas queda en sus horas
                        if not v.dwell_booked:
                            v.dwell_booked = True
                            for name, secs in v.zone_dwell.items():
                                b["staff_dwell"][name] = b["staff_dwell"].get(name, 0.0) + secs
                        else:
                            for name in now_in:
                                b["staff_dwell"][name] = b["staff_dwell"].get(name, 0.0) + dt
                elif self._wait_names:
                    # en la fila cuenta quien ya lleva WAIT_MIN_SECONDS ahí (como las esperas): el que cruza no
                    here = {n for n in self._wait_names if n in now_in and v.zone_dwell.get(n, 0.0) >= WAIT_MIN_SECONDS}
                    for name in here:
                        waiting[name] = waiting.get(name, 0) + 1
                    if here:
                        waiting_visits.append((v, here))
        # espera hasta el pistero: la primera persona que pisa el surtidor atiende a los vehículos que esperan
        for name, waits in self._pump.items():
            if people_in_pump.get(name):
                for pw in waits:
                    if not pw.attended:
                        pw.attended = True
                        b["pump_wait_s"] += ts - pw.since_ts
                        b["pump_wait_n"] += 1
        self._now = now_counts
        self._waiting = waiting
        # la fila de cada zona de espera y si había alguien en su puesto (la zona de personal más cercana)
        for name in self._wait_names:
            wz = b["waits"].setdefault(name, new_wait())
            n = waiting.get(name, 0)
            wz["obs_s"] += frame_dt
            wz["queue_s"] += frame_dt * n
            wz["queue_max"] = max(wz["queue_max"], n)
            if self._staff_names:
                wz["staffed"] = 1
            if n:
                wz["busy_s"] += frame_dt
                if self._staff_names and not in_post.get(self._post_of.get(name), 0):
                    wz["unattended_s"] += frame_dt
        for v, here in waiting_visits:
            v.attended |= {name for name in here if in_post.get(self._post_of.get(name), 0)}
        # cada puesto (zona de personal): cuánto tiempo hubo alguien y cuántos; y los empleados a la vista. Con horario
        # cargado, solo con el local abierto (que de noche no haya nadie en el mostrador no es un puesto descubierto)
        open_now = self.schedule is None or is_open(self.schedule, wall, 0)
        for name in (self._staff_names if open_now else ()):
            p = b["posts"].setdefault(name, new_post())
            c = in_post.get(name, 0)
            p["obs_s"] += frame_dt
            p["people_s"] += frame_dt * c
            p["max"] = max(p["max"], c)
            if c:
                p["covered_s"] += frame_dt
        self._update_tables(b, ts, frame_dt, open_now, at_table, seated, longest, table_exits)
        b["staff_s"] += frame_dt * staff_now
        b["staff_obs_s"] += frame_dt
        b["staff_max"] = max(b["staff_max"], staff_now)
        b["people_max"] = max(b["people_max"], visible_people)
        b["vehicles_max"] = max(b["vehicles_max"], visible_vehicles)
        # visitas terminadas: el tracker dejó de informar el track
        for tid in [k for k in self._visits if k not in live_ids]:
            self._finish(tid, self._visits.pop(tid))
        self._dirty.add(bucket_start(wall))

    # ---------------- mesas (plan «gastronomía») ----------------
    def _update_tables(self, b: dict, ts: float, frame_dt: float, open_now: bool, at_table: dict, seated: dict,
                       longest: dict, exits: list) -> None:
        for name in self._table_names:
            st = b["tables"].setdefault(name, new_table())
            use = self._tables.get(name)
            n = at_table.get(name, 0)
            if use is None and seated.get(name):
                use = self._tables[name] = _TableUse(ts - longest.get(name, 0.0), ts)
            if use is not None:
                if n:
                    use.last_busy_ts = ts
                    use.party = max(use.party, seated.get(name, 0))
                    if use.confirmed:
                        st["occ_s"] += frame_dt
                    elif seated.get(name) and ts - use.start_ts >= TABLE_MIN_SECONDS:  # (el que pasa no la confirma)
                        use.confirmed = True
                        st["occ_s"] += ts - use.start_ts  # lo que llevaba hasta confirmarse
                elif ts - use.last_busy_ts > TABLE_GRACE_SECONDS:
                    self._close_table(b, name, use)
                    use = None
            if open_now:
                st["obs_s"] += frame_dt
        for name, since, stay, staff in exits:
            use = self._tables.get(name)
            if use is None or since < use.start_ts + TABLE_VISIT_AFTER:
                continue
            if not staff and not (TABLE_VISIT_MIN <= stay <= TABLE_VISIT_MAX and at_table.get(name, 0)):
                continue  # no es del personal: solo si se fue con los demás todavía sentados
            if use.visits and since - max(use.visits) < TABLE_VISIT_GAP:
                continue
            use.visits.append(since)
            st = b["tables"].setdefault(name, new_table())
            st["visits"] += 1
            wait = since - use.start_ts
            if len(use.visits) == 1:  # la atención: la primera vez que alguien se acerca
                st["attend_n"] += 1
                st["attend_s"] += wait
                st["attend_max_s"] = max(st["attend_max_s"], wait)
            elif len(use.visits) == 2:  # la entrega (estimada): la segunda
                st["serve_n"] += 1
                st["serve_s"] += wait
                st["serve_max_s"] = max(st["serve_max_s"], wait)

    def _close_table(self, b: dict, name: str, use: _TableUse) -> None:
        self._tables.pop(name, None)
        dur = use.last_busy_ts - use.start_ts
        if not use.confirmed or dur < TABLE_MIN_SECONDS:
            return
        st = b["tables"].setdefault(name, new_table())
        st["n"] += 1
        st["sum_s"] += dur
        st["max_s"] = max(st["max_s"], dur)
        st["party"] += max(1, use.party)
        if not use.visits:
            st["no_visit"] += 1

    def _finish(self, tid: int, v: _Visit) -> None:
        dur = v.last_ts - v.first_ts
        for name, start in v.door_from.items():  # se perdió adentro de la puerta: hasta donde se lo vio
            z = next((z for z in self.doors if z.name == name), None)
            if z is not None and v.last_pos is not None and (c := crossing(z, start, v.last_pos)):
                v.crossings.append(c)
                v.door_of.append(name)
        v.door_from = {}
        if dur < MIN_VISIT_SECONDS and not v.crossings:  # (cruzar la puerta lleva poco: ese sí cuenta)
            return
        key = bucket_start(v.wall_first)
        if key not in self._buckets and self._buckets and key < min(self._buckets):
            # empezó en una hora que ya se mandó y se olvidó (un turno de horas) o de antes de que existiera este
            # contador (se cambiaron las zonas con la persona a la vista): recrearla vacía pisaría en el servidor todo
            # lo de esa hora; va a la hora más vieja que sigue en memoria
            key = min(self._buckets)
        b = self._buckets.setdefault(key, empty_bucket())
        zone_type = {z.name: z.type for z in self.zones}
        if v.label == VEHICLE_LABEL:
            b["vehicles"] += 1
            b["vehicles_dwell_s"] += dur
            for name, waits in self._pump.items():
                for pw in list(waits):
                    if pw.vehicle_id == tid:
                        if not pw.attended:
                            b["pump_unattended"] += 1
                        waits.remove(pw)
            return
        if self.is_staff(v):
            b["staff"] += 1
            b["staff_entries"] += sum(1 for c in v.crossings if c > 0)
            for name in self._staff_names & set(v.zone_dwell):
                self._staff_done[name] = self._staff_done.get(name, 0) + 1
            if not v.dwell_booked:  # (lo normal es que ya se haya sumado cuadro a cuadro)
                for name, secs in v.zone_dwell.items():  # dónde pasan el tiempo los empleados
                    b["staff_dwell"][name] = b["staff_dwell"].get(name, 0.0) + secs
            self._dirty.add(key)
            return
        b["people"] += 1
        b["people_dwell_s"] += dur
        if v.crossings:
            b["entries"] += sum(1 for c in v.crossings if c > 0)
            b["exits"] += sum(1 for c in v.crossings if c < 0)
            for name, c in zip(v.door_of, v.crossings):
                tally = self._door_done.setdefault(name, [0, 0])
                tally[0 if c > 0 else 1] += 1
        elif any(zone_type.get(n) == "entrance" and not self._is_door(n) for n in v.zone_dwell):
            b["entries"] += 1  # puerta sin sentido (zonas viejas): cuenta a todos los que la pisaron
        if any(zone_type.get(n) == "shop" for n in v.zone_dwell):
            b["shop_visits"] += 1
        for name, secs in v.zone_dwell.items():
            z = b["zones"].setdefault(name, {"type": zone_type.get(name), "visits": 0, "dwell_s": 0.0})
            z["visits"] += 1
            z["dwell_s"] += secs
        for name in self._wait_names:  # cuánto esperó en cada zona de espera
            secs = v.zone_dwell.get(name, 0.0)
            if secs < WAIT_MIN_SECONDS:
                continue
            wz = b["waits"].setdefault(name, new_wait())
            wz["n"] += 1
            wz["sum_s"] += secs
            wz["max_s"] = max(wz["max_s"], secs)
            wz["hist"][wait_bin(secs)] += 1
            if self._staff_names:
                wz["staffed"] = 1
                if name not in v.attended:
                    wz["left_alone"] += 1
        self._dirty.add(key)

    def live_counts(self) -> dict:
        """Lo contado hasta ahora, para mostrarlo en vivo sin esperar a que se cierren las visitas, con las mismas
        reglas que las métricas que guardan las máquinas (personas vistas 2 s o más, sin el personal): por zona,
        cuántas la pisaron (`visits`) y cuántas hay ahora (`now`); cada puerta con sentido, cuántos entraron y salieron
        por ella (`entries`, `exits`); cada zona de personal, cuántos empleados (`staff`). Y los totales de la cámara.
        `{"zones": {nombre: {"type", "visits", "now", ...}}, "entries", "exits", "door", "staff", "since"}`."""
        zones = {z.name: {"type": z.type, "visits": 0, "now": self._now.get(z.name, 0)} for z in self.zones}
        for z in self.doors:
            done = self._door_done.get(z.name, [0, 0])
            zones[z.name].update(entries=done[0], exits=done[1])
        for name in self._staff_names:
            zones[name]["staff"] = self._staff_done.get(name, 0)
        entrance_plain = {z.name for z in self.by_type.get("entrance", []) if not z.direction}
        out = {"zones": zones, "entries": 0, "exits": 0, "door": bool(self.doors), "staff": 0, "since": self.since}
        for b in self._buckets.values():  # visitas ya terminadas
            for name, z in (b.get("zones") or {}).items():
                if name in zones:
                    zones[name]["visits"] += z.get("visits", 0)
            for k in ("entries", "exits", "staff"):
                out[k] += b.get(k, 0)
        for v in self._visits.values():  # las que siguen abiertas (la persona sigue a la vista o recién se perdió)
            if v.label == VEHICLE_LABEL or (v.last_ts - v.first_ts < MIN_VISIT_SECONDS and not v.crossings):
                continue
            if self.is_staff(v):
                out["staff"] += 1
                for name in self._staff_names & set(v.zone_dwell):
                    zones[name]["staff"] += 1
                continue
            for name in v.zone_dwell:
                if name in zones:
                    zones[name]["visits"] += 1
            if v.crossings:
                out["entries"] += sum(1 for c in v.crossings if c > 0)
                out["exits"] += sum(1 for c in v.crossings if c < 0)
                for name, c in zip(v.door_of, v.crossings):
                    if name in zones and "entries" in zones[name]:
                        zones[name]["entries" if c > 0 else "exits"] += 1
            elif entrance_plain & set(v.zone_dwell):
                out["entries"] += 1  # puerta sin sentido: cuenta a todos los que la pisaron (como `_finish`)
        # zonas de espera: las esperas terminadas más las de quien sigue ahí (con lo que lleva), y la fila ahora
        for name in self._wait_names:
            agg = new_wait()
            for b in self._buckets.values():
                agg = _merge_stats(agg, (b.get("waits") or {}).get(name) or {}, new_wait())
            n, total, longest = agg["n"], agg["sum_s"], agg["max_s"]
            for v in self._visits.values():
                secs = v.zone_dwell.get(name, 0.0)
                if v.label != VEHICLE_LABEL and secs >= WAIT_MIN_SECONDS and not self.is_staff(v):
                    n, total, longest = n + 1, total + secs, max(longest, secs)
            zones[name]["wait"] = {"n": n, "avg_s": total / n if n else None, "max_s": longest if n else None,
                                   "now": self._waiting.get(name, 0)}
        # mesas: las ocupaciones terminadas más la que sigue (con lo que lleva), y si está ocupada ahora
        for name in self._table_names:
            agg = new_table()
            for b in self._buckets.values():
                agg = _merge_stats(agg, (b.get("tables") or {}).get(name) or {}, new_table())
            use = self._tables.get(name)
            busy = use is not None and use.confirmed
            if busy:
                agg["n"] += 1
                agg["sum_s"] += use.last_busy_ts - use.start_ts
            st = table_stats(agg)
            zones[name]["table"] = {"busy": busy, "for_s": (use.last_busy_ts - use.start_ts) if busy else None,
                                    "n": st["n"], "avg_s": st["avg_s"], "attend_avg_s": st["attend_avg_s"],
                                    "serve_avg_s": st["serve_avg_s"], "occupied_pct": st["occupied_pct"]}
        # puestos (zonas de personal): qué parte del tiempo hubo alguien
        for name in self._staff_names:
            agg = new_post()
            for b in self._buckets.values():
                agg = _merge_stats(agg, (b.get("posts") or {}).get(name) or {}, new_post())
            zones[name]["covered_pct"] = post_stats(agg)["covered_pct"]
        return out

    def is_staff(self, v: _Visit) -> bool:
        if sum(v.zone_dwell.get(n, 0.0) for n in self._staff_names) >= STAFF_MIN_SECONDS:
            return True
        return self.schedule is not None and not is_open(self.schedule, v.wall_first, 0)

    def _is_door(self, name: str) -> bool:
        return any(z.name == name for z in self.doors)

    def _bucket(self, wall: datetime) -> dict:
        return self._buckets.setdefault(bucket_start(wall), empty_bucket())

    # ---------------- envío ----------------
    def flush(self, keep_hours: int = 2) -> list[dict]:
        """Cubos modificados desde el último envío: [{bucket_start, data}]. Olvida los viejos."""
        out = [{"bucket_start": k.isoformat(), "data": self._buckets[k]} for k in sorted(self._dirty) if k in self._buckets]
        self._dirty.clear()
        if self._buckets:
            newest = max(self._buckets)
            for k in [k for k in self._buckets if newest - k > timedelta(hours=keep_hours)]:
                del self._buckets[k]
            self._sent_before = min(self._buckets)
        return out

    def with_zones(self, zones: list[Zone], schedule=None) -> "MetricsCollector":
        """Un contador nuevo con otras zonas (se guardaron zonas con la cámara andando) que sigue sumando en las horas
        que este ya tenía: si arrancara de cero, su primer envío pisaría en el servidor lo de la hora en curso. Las
        visitas abiertas se cierran acá (con las zonas viejas); devuelve el nuevo y deja en `pending` lo que hay que
        mandar."""
        self.pending = self.close()
        new = MetricsCollector(zones, schedule)
        new._buckets, new._sent_before, new.since = self._buckets, self._sent_before, self.since
        return new

    def close(self) -> list[dict]:
        """Al detener la cámara: cierra las visitas abiertas y devuelve todo lo pendiente."""
        for tid, v in list(self._visits.items()):
            self._finish(tid, v)
        self._visits.clear()
        if self._tables and self._buckets:  # las mesas que seguían ocupadas: hasta donde se las vio
            b = self._buckets[max(self._buckets)]
            for name, use in list(self._tables.items()):
                self._close_table(b, name, use)
            self._dirty.add(max(self._buckets))
        return self.flush()


def door_counts(frames, zones: list[Zone], schedule=None) -> dict:
    """Ingresos, egresos y personal de cuadros ya analizados [(tracks, ts, (ancho, alto), hora)]: para comparar con
    un conteo a mano en «Ajustar…» sin volver a pasar el detector."""
    mc = MetricsCollector(zones, schedule)
    for tracks, ts, size, wall in frames:
        mc.update(tracks, ts, size, wall)
    total = empty_bucket()
    for b in mc.close():
        total = merge_bucket(total, b["data"])
    return {k: total[k] for k in ("entries", "exits", "staff", "staff_entries", "door", "people")}
