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
- `frames`: cuadros procesados (para saber si la cámara estuvo activa).

Lo que NO mide (y hay que decirlo): gestos (lavar vidrios, guardar algo) ni caras. A los empleados los descuenta
solo si pisan su zona o llegan con el local cerrado. Una persona que el tracker pierde y vuelve a encontrar cuenta
dos veces (salvo en la puerta con sentido, donde cuenta el cruce).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .geometry import bbox_foot
from .rules import DIRECTIONS, VEHICLE_LABEL, Zone
from .schedule import is_open
from .tracker import Track

MIN_VISIT_SECONDS = 2.0  # menos que esto es ruido del detector
STAFF_MIN_SECONDS = 2.0  # en la zona de personal: menos que esto puede ser un cliente que se acercó al mostrador
CROSS_FRACTION = 0.4  # para contar un cruce, moverse al menos este tanto del ancho de la puerta en el sentido marcado


def bucket_start(wall: datetime) -> datetime:
    return wall.replace(minute=0, second=0, microsecond=0)


def empty_bucket() -> dict:
    return {"people": 0, "people_dwell_s": 0.0, "people_max": 0, "entries": 0, "exits": 0, "door": 0,
            "staff": 0, "staff_entries": 0, "zones": {},
            "vehicles": 0, "vehicles_dwell_s": 0.0, "vehicles_max": 0,
            "pump_vehicles": 0, "pump_wait_s": 0.0, "pump_wait_n": 0, "pump_unattended": 0,
            "shop_visits": 0, "frames": 0}


def merge_bucket(a: dict, b: dict) -> dict:
    """Suma dos cubos (de cámaras u horas distintas). Los máximos se quedan con el mayor."""
    out = dict(empty_bucket())
    for k in out:
        if k == "zones":
            continue
        va, vb = a.get(k, 0), b.get(k, 0)
        out[k] = max(va, vb) if k.endswith("_max") or k == "door" else va + vb
    zones: dict[str, dict] = {}
    for src in (a.get("zones") or {}, b.get("zones") or {}):
        for name, z in src.items():
            cur = zones.setdefault(name, {"type": z.get("type"), "visits": 0, "dwell_s": 0.0})
            cur["visits"] += z.get("visits", 0)
            cur["dwell_s"] += z.get("dwell_s", 0.0)
    out["zones"] = zones
    return out


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


class MetricsCollector:
    def __init__(self, zones: list[Zone], schedule=None):
        self.zones = zones
        self.schedule = schedule  # horario del local: con el local cerrado, quien aparece es empleado
        self.by_type: dict[str, list[Zone]] = {}
        for z in zones:
            self.by_type.setdefault(z.type, []).append(z)
        self.doors = [z for z in self.by_type.get("entrance", []) if z.direction]
        self._staff_names = {z.name for z in self.by_type.get("staff", [])}
        self._visits: dict[int, _Visit] = {}
        self._pump: dict[str, list[_PumpWait]] = {}  # zona surtidor -> esperas abiertas
        self._buckets: dict[datetime, dict] = {}
        self._dirty: set[datetime] = set()

    # ---------------- por cuadro ----------------
    def update(self, tracks: list[Track], ts: float, frame_size: tuple[int, int], wall: datetime) -> None:
        w, h = frame_size
        b = self._bucket(wall)
        b["frames"] += 1
        if self.doors:
            b["door"] = 1
        live_ids = {t.id for t in tracks}
        visible_people = visible_vehicles = 0
        people_in_pump: dict[str, bool] = {}
        for t in tracks:
            v = self._visits.get(t.id)
            if v is None:
                v = _Visit(t.label, t.first_ts, ts, wall - timedelta(seconds=max(0.0, ts - t.first_ts)), last_seen_ts=ts)
                self._visits[t.id] = v
            if not t.visible:
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
                v.last_pos = (nx, ny)
            entered = now_in - v.inside
            v.inside = now_in
            for name in entered:
                z = next(z for z in self.zones if z.name == name)
                if z.type == "pump":
                    if t.label == VEHICLE_LABEL:
                        self._pump.setdefault(name, []).append(_PumpWait(t.id, ts))
                        b["pump_vehicles"] += 1
            if t.label != VEHICLE_LABEL:
                for name in now_in:
                    if any(z.name == name and z.type == "pump" for z in self.zones):
                        people_in_pump[name] = True
        # espera hasta el pistero: la primera persona que pisa el surtidor atiende a los vehículos que esperan
        for name, waits in self._pump.items():
            if people_in_pump.get(name):
                for pw in waits:
                    if not pw.attended:
                        pw.attended = True
                        b["pump_wait_s"] += ts - pw.since_ts
                        b["pump_wait_n"] += 1
        b["people_max"] = max(b["people_max"], visible_people)
        b["vehicles_max"] = max(b["vehicles_max"], visible_vehicles)
        # visitas terminadas: el tracker dejó de informar el track
        for tid in [k for k in self._visits if k not in live_ids]:
            self._finish(tid, self._visits.pop(tid))
        self._dirty.add(bucket_start(wall))

    def _finish(self, tid: int, v: _Visit) -> None:
        dur = v.last_ts - v.first_ts
        for name, start in v.door_from.items():  # se perdió adentro de la puerta: hasta donde se lo vio
            z = next((z for z in self.doors if z.name == name), None)
            if z is not None and v.last_pos is not None and (c := crossing(z, start, v.last_pos)):
                v.crossings.append(c)
        v.door_from = {}
        if dur < MIN_VISIT_SECONDS and not v.crossings:  # (cruzar la puerta lleva poco: ese sí cuenta)
            return
        b = self._bucket(v.wall_first)
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
            self._dirty.add(bucket_start(v.wall_first))
            return
        b["people"] += 1
        b["people_dwell_s"] += dur
        if v.crossings:
            b["entries"] += sum(1 for c in v.crossings if c > 0)
            b["exits"] += sum(1 for c in v.crossings if c < 0)
        elif any(zone_type.get(n) == "entrance" and not self._is_door(n) for n in v.zone_dwell):
            b["entries"] += 1  # puerta sin sentido (zonas viejas): cuenta a todos los que la pisaron
        if any(zone_type.get(n) == "shop" for n in v.zone_dwell):
            b["shop_visits"] += 1
        for name, secs in v.zone_dwell.items():
            z = b["zones"].setdefault(name, {"type": zone_type.get(name), "visits": 0, "dwell_s": 0.0})
            z["visits"] += 1
            z["dwell_s"] += secs
        self._dirty.add(bucket_start(v.wall_first))

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
        return out

    def close(self) -> list[dict]:
        """Al detener la cámara: cierra las visitas abiertas y devuelve todo lo pendiente."""
        for tid, v in list(self._visits.items()):
            self._finish(tid, v)
        self._visits.clear()
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
