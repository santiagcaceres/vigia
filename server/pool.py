"""Máquinas de análisis (servidores en la nube): cuánto pesa cada cámara, cómo se reparten y cuántas
máquinas hacen falta prendidas. Solo funciones puras: la base de datos la maneja server/main.py.

- **Peso** de una cámara = cuánto de una máquina ocupa ahora (1,0 = cámara de hurto a 5 fps; con gestos o con la IA
  de movimientos de hurto, un 30 % más por la postura; con «Personas lejos», 4 veces: la imagen entera más 4 partes).
  Con el local cerrado y «No mirar», pesa 0: no hace falta ninguna máquina para ella.
- **Reparto**: cada cámara se queda en su máquina mientras esa máquina esté en línea y le entre (los procesos de una
  máquina comparten su lugar); las demás van a la menos cargada. Si una máquina se cae (90 s sin noticias), sus
  cámaras pasan a las otras.
- **Recomendación** (`recommended`): cuántas cámaras aguanta bien cada máquina. Con placa de video, 12 por procesador
  (vCPU) y hasta 140 (la placa es la misma en todos los tamaños); sin placa, lo que midió; o el número que ponga un
  administrador. No es un tope: al marcar una cámara para analizar se la marca igual y, si la máquina queda al 85 % o
  más, se avisa (`level`: «alto» en naranja; «pasada» en rojo, con un cartel en la app).
- **Plan**: cuántas máquinas tienen que estar prendidas (mirando también los próximos 20 min, para
  prenderlas antes de que abran los locales) y cuáles se pueden apagar (ya sin cámaras).
- **Máquina fija** (`Machine.fixed`, la avisa la máquina con `VIGIA_FIJA=si`): siempre prendida (la de la reserva
  de AWS). Cuenta para el plan como cualquiera, pero no se vacía ni se apaga: lo que sobra es siempre de las otras.
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
# cámara con gestos de ocultamiento o con la IA de movimientos de hurto (core/hurto_ia.py): además del detector, la
# postura (core/pose.py, ~18 ms por persona) suma un 20-40 % con gente a la vista; se cuenta un 30 % más (sin gente, el
# filtro de movimiento ahorra igual). Con las dos, la postura se calcula una sola vez: el mismo 30 %
POSE_FACTOR = 1.3
# «Personas lejos» (core/deteccion.py): el detector mira la imagen entera + 4 partes agrandadas ≈ 5 veces el detector
# por cuadro; decodificar el video, seguir a las personas y las reglas no se multiplican: se cuenta 4 veces la cámara
FAR_FACTOR = 4


def uses_pose(analysis: dict) -> bool:
    """La cámara mira la postura de las personas: tiene gestos o la IA de movimientos de hurto (`analysis` es lo que
    arma `camera_analysis` en server/main.py)."""
    return bool(analysis.get("gestures") or analysis.get("ia_hurto"))


def camera_weight(*, fps: float, mode: str, brand: str, hours: dict | None, tz: str, off_hours: str,
                  now: datetime, pose: bool = False, far: bool = False) -> float:
    """Cuánto ocupa la cámara en este momento (`now` en UTC, con o sin zona). `pose`: mira la postura (gestos o IA de
    hurtos, `uses_pose`). `far`: «Personas lejos» (el detector pasa 5 veces por cuadro); solo cuenta mirando en vivo:
    en los modos de alarmas el peso es por escuchar al grabador, y cada clip de alarma se analiza aparte."""
    alarms = mode in ("alarmas", "alarmas_equipo")
    base = 0.1 if alarms else 0.2 + 0.8 * fps / 5.0  # 0,2 = decodificar el video
    if pose:
        base *= POSE_FACTOR
    if far and not alarms:
        base *= FAR_FACTOR
    if hours:
        local = (now if now.tzinfo else now.replace(tzinfo=timezone.utc)).astimezone(ZoneInfo(tz or "UTC"))
        if not is_open(hours, local, DEFAULT_MARGIN_MIN):
            if off_hours == "vivo":
                return base
            if off_hours == "alarmas" and brand != "hikconnect":
                return 0.05
            return 0.0
    return base


# Cuántas cámaras aguanta cada máquina (27 nov, el dueño: «no pongas límite por cámara; avisá y ponelo con colores, una
# recomendación de cuántas aguanta esa PC, y permití poner las que sea; si está en rojo mostrá un cartel»). Antes (24
# nov) era un tope que no dejaba marcar más. Con todo lo pasado a la placa (video dentro del proceso, TensorRT, clips
# con NVENC) una g4dn.xlarge de 4 vCPU gasta ~0,065 procesadores por cámara de hurto: al 85 % son ~50, unas 12 por
# vCPU; la 2xlarge el doble; la 4xlarge tiene la misma placa, que pasa a ser el límite (~140). Es una estimación: la
# confirma una prueba de llenado y, si hace falta, el administrador pone otro número en la app. El reparto nunca le saca
# la máquina a una cámara que ya se analiza (un reinicio o un corte no pueden dejar afuera a las que andan).
GPU_PER_VCPU = 12.0
GPU_MAX = 140.0
LEVEL_HIGH = HEADROOM  # desde el 85 % de lo recomendado, «alto» (naranja); pasado el 100 %, «pasada» (rojo)
ALIVE_S = 72 * 3600.0  # una cámara que dio imagen en los últimos 3 días ocupa su lugar (aguanta un fin de semana cerrado)
RECENT_S = 30 * 60.0  # dio imagen hace menos de media hora: se la está analizando
OVERFILL = 1.25  # un proceso puede pasarse un 25 % de su parte si la máquina entera tiene lugar (no mover cámaras)


def life_rank(image_age_s: float | None) -> int:
    """Qué tan viva está una cámara por su última imagen (dato que queda guardado: no se pierde si la máquina se
    reinicia): 0 = se la está analizando, 1 = anduvo en los últimos días (local cerrado, un corte), 2 = no anda hace
    días o nunca dio imagen. En un proceso muy cargado, las que no andan son las primeras en pasar a otro."""
    if image_age_s is None:
        return 2
    return 0 if image_age_s < RECENT_S else 1 if image_age_s < ALIVE_S else 2


def recommended(measured: float, cores: float | None, gpu: bool, override: float | None = None) -> float:
    """Cuántas cámaras (de hurto a 5 cuadros/s; con gestos cuentan 1,3, solo métricas 0,5) aguanta bien una máquina:
    lo que puso el administrador (`override`) o, si no, con placa de video 12 por procesador y hasta 140, y sin placa lo
    que midieron sus procesos al instalarse (`measured`, la suma)."""
    if override and override > 0:
        return float(override)
    if gpu and cores and cores > 0:
        return min(float(cores) * GPU_PER_VCPU, GPU_MAX)
    return float(measured)


def level(used: float, rec: float) -> str:
    """«ok» (verde), «alto» (desde el 85 %, naranja) o «pasada» (más de lo recomendado, rojo). Vacío sin
    recomendación (ninguna máquina en línea)."""
    if rec <= 0:
        return ""
    if used > rec + 0.05:
        return "pasada"
    return "alto" if used >= rec * LEVEL_HIGH - 1e-9 else "ok"


def machine_caps(servers: dict[int, float], groups: dict[int, str] | None = None,
                 cap: float | None = None) -> dict[str, float]:
    """Lo que aguanta cada máquina: la suma de sus procesos (comparten el procesador y la placa), con `cap` si es
    menor (las pruebas; el servidor ya les pasa a los procesos su parte de lo recomendado)."""
    out: dict[str, float] = {}
    for a, c in servers.items():
        key = (groups or {}).get(a) or f"agente-{a}"
        out[key] = out.get(key, 0.0) + c
    return {k: (min(v, cap) if cap else v) for k, v in out.items()}


def assign(cameras: list[tuple[int, float, int | None]], servers: dict[int, float],
           groups: dict[int, str] | None = None, rank: dict[int, int] | None = None,
           cap: float | None = None) -> dict[int, int]:
    """`cameras`: (id, peso, proceso actual). `servers`: proceso en línea → capacidad. `groups`: proceso → máquina
    (sin eso, cada proceso es una máquina). `rank`: id → `life_rank` (las que andan se acomodan primero). `cap`: lo que
    aguanta como mucho cada máquina (opcional). Devuelve id → proceso, para todas.
    Las cámaras se quedan donde están mientras su máquina tenga lugar (no se reconecta nada); las demás van a una
    máquina donde entren y, si no entran en ninguna, igual a la menos cargada (el reparto no deja a ninguna sin
    mirar: si quedan de más, la app lo avisa en rojo)."""
    if not servers:
        return {}
    rank = rank or {}
    key = {a: (groups or {}).get(a) or f"agente-{a}" for a in servers}
    room = machine_caps(servers, groups, cap)
    used = dict.fromkeys(room, 0.0)
    load = dict.fromkeys(servers, 0.0)
    out: dict[int, int] = {}

    def fits(a: int, w: float) -> bool:
        return used[key[a]] + w <= room[key[a]] + 1e-9

    def take(cid: int, a: int, w: float) -> None:
        used[key[a]] += w
        load[a] += w
        out[cid] = a

    def lightest(agents, w: float) -> int:
        return min(agents, key=lambda a: ((load[a] + w) / max(servers[a], 0.1), a))

    pending = []
    for cid, w, cur in sorted(cameras, key=lambda c: (rank.get(c[0], 1), c[0])):
        if cur not in servers or not fits(cur, w):
            pending.append((cid, w, cur))
        elif load[cur] + w <= servers[cur] * OVERFILL + 1e-9:
            take(cid, cur, w)
        else:  # su máquina tiene lugar pero su proceso quedó muy cargado: pasa a otro proceso de la misma máquina
            take(cid, lightest([a for a in servers if key[a] == key[cur]], w), w)
    for cid, w, cur in pending:
        free = [a for a in servers if fits(a, w)]
        if free:
            take(cid, lightest(free, w), w)
        elif cur in servers and load[cur] + w <= servers[cur] * OVERFILL + 1e-9:
            take(cid, cur, w)  # no entra en ninguna: se queda donde está (sin reconectar)
        else:
            take(cid, lightest(servers, w), w)
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
    leaving: bool = False  # la nube avisó que se la lleva (Spot): no cuenta como prendida
    # máquina fija: está siempre prendida (la de la reserva o «Savings Plan») y no la maneja el encendedor. Es la
    # última en vaciarse: si alcanza con una máquina, se queda ella y se apaga la otra (que se paga por hora)
    fixed: bool = False


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
    # se quedan las fijas (ya están pagas y nadie las apaga) y, después, las que ya estaban trabajando (las más
    # cargadas); el resto se vacía para apagarla. Una fija nunca se vacía
    ordered = sorted(online, key=lambda m: (not m.fixed, m.draining, -m.load, m.key))
    extra = [m for m in ordered[desired:] if not m.fixed]
    draining = {m.key: m in extra for m in online}
    stop = [m.instance_id for m in extra if m.instance_id and m.load <= 1e-9]
    return Plan(units, cap, desired, len(online), draining, stop)


def soon(now: datetime) -> datetime:
    return now + LOOKAHEAD
