"""Máquinas de análisis (servidores en la nube): cuánto pesa cada cámara, cómo se reparten y cuántas
máquinas hacen falta prendidas. Solo funciones puras: la base de datos la maneja server/main.py.

- **Peso** de una cámara = cuánto de una máquina ocupa ahora (1,0 = cámara de hurto a 5 fps). Con el local
  cerrado y «No mirar», pesa 0: no hace falta ninguna máquina para ella.
- **Reparto**: cada cámara se queda en su máquina mientras esa máquina esté en línea y le entre; las demás
  van a la menos cargada. Si una máquina se cae (90 s sin noticias), sus cámaras pasan a las otras.
- **Plan**: cuántas máquinas tienen que estar prendidas (mirando también los próximos 20 min, para
  prenderlas antes de que abran los locales) y cuáles se pueden apagar (ya sin cámaras).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from core.schedule import DEFAULT_MARGIN_MIN, is_open

HEADROOM = 0.85  # no llenar las máquinas más allá del 85 %
LOOKAHEAD = timedelta(minutes=20)  # una máquina tarda 2-3 min en arrancar: prenderla antes de que abran
ONLINE = timedelta(seconds=90)  # sin noticias en este tiempo = máquina caída
DEFAULT_CAPACITY = 30.0  # cámaras de hurto a 5 fps por máquina (hasta medirlo)


def camera_weight(*, fps: float, mode: str, brand: str, hours: dict | None, tz: str, off_hours: str,
                  now: datetime) -> float:
    """Cuánto ocupa la cámara en este momento (`now` en UTC, con o sin zona)."""
    base = 0.1 if mode in ("alarmas", "alarmas_equipo") else 0.2 + 0.8 * fps / 5.0  # 0,2 = decodificar el video
    if hours:
        local = (now if now.tzinfo else now.replace(tzinfo=timezone.utc)).astimezone(ZoneInfo(tz or "UTC"))
        if not is_open(hours, local, DEFAULT_MARGIN_MIN):
            if off_hours == "vivo":
                return base
            if off_hours == "alarmas" and brand != "hikconnect":
                return 0.05
            return 0.0
    return base


def assign(cameras: list[tuple[int, float, int | None]], servers: dict[int, float]) -> dict[int, int]:
    """`cameras`: (id, peso, máquina actual). `servers`: máquina en línea → capacidad. Devuelve id → máquina.
    Las cámaras se quedan donde están si entran; si no hay lugar en ninguna, van igual a la menos cargada
    (mejor una máquina apretada que una cámara sin mirar: el plan pide otra máquina)."""
    if not servers:
        return {}
    load = dict.fromkeys(servers, 0.0)
    out: dict[int, int] = {}
    pending = []
    for cid, w, cur in sorted(cameras, key=lambda c: (-c[1], c[0])):
        if cur in servers and load[cur] + w <= servers[cur] + 1e-9:
            load[cur] += w
            out[cid] = cur
        else:
            pending.append((cid, w))
    for cid, w in pending:
        best = min(servers, key=lambda a: ((load[a] + w) / max(servers[a], 0.1), a))
        load[best] += w
        out[cid] = best
    return out


@dataclass
class Machine:
    key: str  # id de la instancia en la nube, o «agente-N» si no se sabe
    instance_id: str | None
    agent_ids: list[int]
    online: bool
    draining: bool
    capacity: float
    load: float  # peso de las cámaras que tiene ahora
    names: list[str] = field(default_factory=list)


@dataclass
class Plan:
    units: float  # peso total (el mayor entre ahora y dentro de 20 min)
    capacity: float  # capacidad típica de una máquina
    desired: int  # máquinas que tienen que estar prendidas
    online: int
    draining: dict[str, bool]  # máquina → si se la está vaciando para apagarla
    stop: list[str]  # instancias que ya se pueden apagar

    def as_dict(self) -> dict:
        return {"units": round(self.units, 2), "capacity": self.capacity, "desired": self.desired,
                "online": self.online, "stop": self.stop}


def plan(units_now: float, units_soon: float, machines: list[Machine], default_capacity: float = DEFAULT_CAPACITY,
         spare: int = 0) -> Plan:
    units = max(units_now, units_soon)
    caps = sorted(m.capacity for m in machines if m.capacity > 0)
    cap = caps[len(caps) // 2] if caps else default_capacity
    desired = 0 if units <= 1e-9 else math.ceil(units / (cap * HEADROOM)) + max(0, spare)
    online = [m for m in machines if m.online]
    # se quedan las que ya estaban trabajando (las más cargadas); el resto se vacía para apagarla
    ordered = sorted(online, key=lambda m: (m.draining, -m.load, m.key))
    extra = ordered[desired:]
    draining = {m.key: m in extra for m in online}
    stop = [m.instance_id for m in extra if m.instance_id and m.load <= 1e-9]
    return Plan(units, cap, desired, len(online), draining, stop)


def soon(now: datetime) -> datetime:
    return now + LOOKAHEAD
