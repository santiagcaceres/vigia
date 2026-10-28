"""Encendedor de máquinas de análisis: prende y apaga las máquinas en la nube según los horarios de los locales.

    python -m agent.escalador            # lee /etc/vigia/escalador.env o variables VIGIA_* / AWS_*

Cada minuto pregunta al servidor de Vigía (`GET /api/agent/scale`, con la clave de máquinas) cuántas
máquinas tienen que estar prendidas, mirando también los próximos 20 minutos (así una máquina está lista
antes de que abra el local). Si faltan, prende o crea; si sobran, el servidor primero les saca las cámaras y
recién entonces aparecen en `stop` y se apagan. Nunca pasa de `VIGIA_MAX_MAQUINAS` (protege la factura).

Corre en una máquina chiquita siempre prendida (en AWS una t4g.nano, ~US$ 3 por mes) o en cualquier PC que
quede prendida. Las claves de la nube viven solo acá (mejor: un rol de IAM), nunca en Vigía.

Nubes:
- `aws-asg` (la recomendada en AWS, la arma deploy/aws/vigia-analisis.yaml): un grupo de Auto Scaling con
  máquinas Spot de varios tipos. El encendedor solo sube la cantidad deseada del grupo y, para bajar, termina
  una por una las máquinas que el servidor ya vació (las demás están protegidas: AWS nunca apaga por su cuenta
  una máquina con cámaras). Si AWS reclama una Spot, el grupo pone otra solo. Permisos:
  autoscaling:DescribeAutoScalingGroups, autoscaling:SetDesiredCapacity,
  autoscaling:TerminateInstanceInAutoScalingGroup y ec2:DescribeInstances.
- `aws`: EC2 sin grupo. Las máquinas llevan la etiqueta `vigia-grupo=<grupo>`. Prende las detenidas y, con
  plantilla, crea nuevas (Spot si `VIGIA_SPOT=si`, probando los tipos de `VIGIA_TIPOS` y, si no hay Spot, a
  demanda con `VIGIA_RESPALDO=si`). Para bajar, detiene las de a demanda y termina las Spot (una Spot común no
  se puede detener). Permisos: ec2:DescribeInstances, ec2:StartInstances, ec2:StopInstances,
  ec2:TerminateInstances (y ec2:RunInstances + ec2:CreateTags si se usa plantilla).
- `ninguna`: no prende ni apaga nada, solo registra cuántas máquinas harían falta (una nube que no se maneja por
  programa: se prenden a mano).
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

log = logging.getLogger("vigia.escalador")
BROKEN_AFTER = timedelta(minutes=20)  # prendida hace más de esto y sin darse de alta en Vigía = rota
# AWS no tiene lugar para ese tipo de máquina (o no hay Spot a ese precio): se prueba el siguiente
CAPACITY_ERRORS = {"InsufficientInstanceCapacity", "InsufficientCapacity", "SpotMaxPriceTooLow",
                   "MaxSpotInstanceCountExceeded", "InsufficientHostCapacity", "Unsupported",
                   "InsufficientFreeAddressesInSubnet"}


@dataclass
class Instance:
    id: str
    state: str  # pending | running | stopping | stopped
    launched: datetime | None = None
    spot: bool = False


@dataclass
class Decision:
    start: list[str]
    launch: int  # máquinas nuevas a crear con la plantilla
    stop: list[str]
    missing: int  # faltan máquinas en el grupo y no hay plantilla


@dataclass
class AsgDecision:
    desired: int | None  # nueva cantidad deseada del grupo (None = no tocar)
    terminate: list[str] = field(default_factory=list)  # vaciadas: se terminan bajando la cantidad deseada


def _excess_to_stop(plan: dict, running: list[Instance], excess: int, now: datetime) -> list[str]:
    """Las que se pueden apagar cuando sobran: primero las que el servidor ya vació; después las rotas (prendidas
    hace rato y nunca se dieron de alta). Nunca más de `excess`."""
    if excess <= 0:
        return []
    online = set(plan.get("online_instances") or [])
    ids = {r.id for r in running}
    drained = [i for i in (plan.get("stop") or []) if i in ids]
    broken = [r.id for r in running if r.id not in online and r.state == "running" and r.launched
              and now - r.launched > BROKEN_AFTER and r.id not in drained]
    return (drained + broken)[:excess]


def _active(plan: dict, instances: list[Instance]) -> list[Instance]:
    """Prendidas (o prendiéndose) que van a seguir: las que la nube ya reclamó (Spot) no cuentan."""
    leaving = set(plan.get("leaving_instances") or [])
    return [i for i in instances if i.state in ("pending", "running") and i.id not in leaving]


def decide(plan: dict, instances: list[Instance], max_machines: int, can_launch: bool,
           now: datetime | None = None) -> Decision:
    """Qué prender y qué apagar con máquinas EC2 sueltas (sin tocar nada: solo decide)."""
    now = now or datetime.now(timezone.utc)
    desired = max(0, min(int(plan.get("desired") or 0), max_machines))
    running = _active(plan, instances)
    stopped = sorted((i for i in instances if i.state == "stopped"), key=lambda i: i.id)
    start: list[str] = []
    launch = missing = 0
    if len(running) < desired:
        need = desired - len(running)
        start = [i.id for i in stopped[:need]]
        rest = need - len(start)
        if rest > 0:
            launch, missing = (rest, 0) if can_launch else (0, rest)
    stop = _excess_to_stop(plan, running, len(running) - desired, now)
    return Decision(start, launch, stop, missing)


def decide_asg(plan: dict, group_desired: int, group_max: int, instances: list[Instance], max_machines: int,
               now: datetime | None = None) -> AsgDecision:
    """Con un grupo de Auto Scaling: subir la cantidad deseada si faltan máquinas; si sobran, terminar las que el
    servidor ya vació (una por una, bajando la cantidad deseada). Las que la nube reclamó las repone el grupo."""
    now = now or datetime.now(timezone.utc)
    want = max(0, min(int(plan.get("desired") or 0), max_machines, group_max))
    if want > group_desired:
        return AsgDecision(want)
    running = _active(plan, instances)
    terminate = _excess_to_stop(plan, running, min(group_desired, len(running)) - want, now)
    # pedía más de las que hay (AWS no tenía lugar) y ya no hacen falta: bajar sin tocar las que andan
    total = sum(1 for i in instances if i.state in ("pending", "running"))
    lower = max(want, total)
    return AsgDecision(lower if lower < group_desired else None, terminate)


def _error_code(exc: Exception) -> str | None:
    return ((getattr(exc, "response", None) or {}).get("Error") or {}).get("Code")


class AwsPool:
    """Máquinas EC2 sueltas con la etiqueta `vigia-grupo`."""

    def __init__(self, region: str, group: str, template: str | None = None, *, spot: bool = False,
                 types: list[str] | None = None, fallback: bool = True, ec2=None):
        if ec2 is None:
            import boto3  # solo en la máquina del encendedor (pip install boto3)

            ec2 = boto3.client("ec2", region_name=region)
        self.ec2 = ec2
        self.group = group
        self.template = template
        self.spot = spot
        self.types = [t for t in (types or []) if t]
        self.fallback = fallback

    def instances(self) -> list[Instance]:
        out = []
        pages = self.ec2.get_paginator("describe_instances").paginate(Filters=[
            {"Name": "tag:vigia-grupo", "Values": [self.group]},
            {"Name": "instance-state-name", "Values": ["pending", "running", "stopping", "stopped"]}])
        for page in pages:
            for res in page["Reservations"]:
                for i in res["Instances"]:
                    out.append(Instance(i["InstanceId"], i["State"]["Name"], i.get("LaunchTime"),
                                        i.get("InstanceLifecycle") == "spot"))
        return out

    def start(self, ids: list[str]) -> None:
        self.ec2.start_instances(InstanceIds=ids)

    def stop(self, ids: list[str], spot: set[str] | None = None) -> None:
        """Detiene las de a demanda (el disco queda: arrancan más rápido) y termina las Spot."""
        spot = spot or set()
        if [i for i in ids if i in spot]:
            self.ec2.terminate_instances(InstanceIds=[i for i in ids if i in spot])
        if [i for i in ids if i not in spot]:
            self.ec2.stop_instances(InstanceIds=[i for i in ids if i not in spot])

    def launch(self, n: int) -> int:
        """Crea `n` máquinas con la plantilla: Spot primero (si está pedido), probando cada tipo; si AWS no tiene
        lugar, a demanda (con respaldo). Devuelve cuántas se crearon."""
        tmpl = {"LaunchTemplateId": self.template} if self.template.startswith("lt-") else \
            {"LaunchTemplateName": self.template}
        tags = [{"ResourceType": "instance", "Tags": [{"Key": "vigia-grupo", "Value": self.group},
                                                      {"Key": "Name", "Value": f"vigia-{self.group}"}]}]
        markets = ["spot"] if self.spot else []
        if not self.spot or self.fallback:
            markets.append("demanda")
        left = n
        for market in markets:
            for t in self.types or [None]:
                if left <= 0:
                    return n
                kw = {"InstanceType": t} if t else {}
                if market == "spot":
                    kw["InstanceMarketOptions"] = {"MarketType": "spot", "SpotOptions": {
                        "SpotInstanceType": "one-time", "InstanceInterruptionBehavior": "terminate"}}
                try:
                    r = self.ec2.run_instances(LaunchTemplate=tmpl, MinCount=1, MaxCount=left, TagSpecifications=tags,
                                               **kw)
                except Exception as exc:  # noqa: BLE001 - botocore ClientError; sin lugar se prueba otro tipo
                    if _error_code(exc) not in CAPACITY_ERRORS:
                        raise
                    log.warning("AWS no tiene %s %s ahora (%s); pruebo otra opción", market, t or "del tipo de la "
                                "plantilla", _error_code(exc))
                    continue
                got = len(r.get("Instances") or [])
                left -= got
                log.info("Creada(s) %d máquina(s) %s %s", got, market, t or "")
        if left:
            log.warning("No se pudieron crear %d máquina(s): AWS no tiene lugar en ningún tipo ni mercado", left)
        return n - left

    def apply(self, plan: dict, max_machines: int, now: datetime | None = None) -> Decision:
        instances = self.instances()
        d = decide(plan, instances, max_machines, bool(self.template), now)
        if d.start:
            log.info("Prendiendo %d máquina(s): %s", len(d.start), ", ".join(d.start))
            self.start(d.start)
        if d.launch:
            log.info("Creando %d máquina(s) nueva(s) con la plantilla", d.launch)
            self.launch(d.launch)
        if d.missing:
            log.warning("Faltan %d máquina(s) en el grupo y no hay plantilla para crearlas", d.missing)
        if d.stop:
            log.info("Apagando %d máquina(s) sin cámaras: %s", len(d.stop), ", ".join(d.stop))
            self.stop(d.stop, {i.id for i in instances if i.spot})
        return d


class AsgPool:
    """Un grupo de Auto Scaling (deploy/aws/vigia-analisis.yaml): Spot de varios tipos, con protección contra
    achicarse solo (AWS no termina por su cuenta una máquina con cámaras)."""

    def __init__(self, region: str, name: str, *, asg=None, ec2=None):
        if asg is None or ec2 is None:
            import boto3

            asg = asg or boto3.client("autoscaling", region_name=region)
            ec2 = ec2 or boto3.client("ec2", region_name=region)
        self.asg, self.ec2, self.name = asg, ec2, name

    def group(self) -> tuple[int, int, list[Instance]]:
        g = self.asg.describe_auto_scaling_groups(AutoScalingGroupNames=[self.name])["AutoScalingGroups"]
        if not g:
            raise RuntimeError(f"No existe el grupo de Auto Scaling {self.name!r}")
        g = g[0]
        states = {}
        for i in g.get("Instances") or []:
            st = i.get("LifecycleState") or ""
            if st.startswith("Pending"):
                states[i["InstanceId"]] = "pending"
            elif st == "InService":
                states[i["InstanceId"]] = "running"
        launched: dict[str, datetime] = {}
        if states:
            for res in self.ec2.describe_instances(InstanceIds=list(states))["Reservations"]:
                for i in res["Instances"]:
                    launched[i["InstanceId"]] = i.get("LaunchTime")
        return (int(g.get("DesiredCapacity") or 0), int(g.get("MaxSize") or 0),
                [Instance(i, s, launched.get(i), True) for i, s in sorted(states.items())])

    def apply(self, plan: dict, max_machines: int, now: datetime | None = None) -> AsgDecision:
        desired, top, instances = self.group()
        d = decide_asg(plan, desired, top, instances, max_machines, now)
        if d.desired is not None:  # (antes de terminar: cada terminada baja la cantidad deseada en 1)
            log.info("Grupo %s: de %d a %d máquina(s)", self.name, desired, d.desired)
            self.asg.set_desired_capacity(AutoScalingGroupName=self.name, DesiredCapacity=d.desired,
                                          HonorCooldown=False)
        for iid in d.terminate:
            log.info("Grupo %s: terminando %s (sin cámaras)", self.name, iid)
            self.asg.terminate_instance_in_auto_scaling_group(InstanceId=iid, ShouldDecrementDesiredCapacity=True)
        return d


class NoCloud:
    """Sin nube manejable por programa: solo avisa cuántas máquinas harían falta."""

    def apply(self, plan: dict, max_machines: int, now: datetime | None = None) -> None:
        log.info("Hacen falta %s máquinas (carga %s, %s en línea) — sin nube conectada, no se toca nada",
                 plan.get("desired"), plan.get("units"), plan.get("online"))
        return None


def tick(server: str, key: str, cloud, max_machines: int, http: httpx.Client):
    r = http.get(server.rstrip("/") + "/api/agent/scale", headers={"X-Pool-Key": key})
    r.raise_for_status()
    plan = r.json()
    p = {**(plan.get("plan") or plan), "online_instances": plan.get("online_instances"),
         "leaving_instances": plan.get("leaving_instances")}
    return cloud.apply(p, max_machines)


def read_env_file(path: Path) -> dict[str, str]:
    """`CLAVE=valor` por línea (copia de la del agente: el encendedor no instala OpenCV ni el modelo)."""
    out: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def yes(v: str | None) -> bool:
    return (v or "").strip().lower() in ("1", "si", "sí", "true", "yes")


def main() -> None:
    env = {**read_env_file(Path(os.getenv("VIGIA_ESCALADOR_CONFIG", "/etc/vigia/escalador.env"))),
           **{k: v for k, v in os.environ.items() if k.startswith(("VIGIA_", "AWS_"))}}
    p = argparse.ArgumentParser(description="Prende y apaga las máquinas de análisis de Vigía")
    p.add_argument("--server", default=env.get("VIGIA_SERVER", "https://vigia-three-ashy.vercel.app"))
    p.add_argument("--clave-maquinas", default=env.get("VIGIA_POOL_KEY"))
    p.add_argument("--nube", default=env.get("VIGIA_NUBE", "aws"), choices=["aws-asg", "aws", "ninguna"])
    p.add_argument("--region", default=env.get("AWS_REGION", "sa-east-1"))
    p.add_argument("--asg", default=env.get("VIGIA_ASG"), help="Grupo de Auto Scaling (nube aws-asg)")
    p.add_argument("--grupo", default=env.get("VIGIA_GRUPO", "analisis"))
    p.add_argument("--plantilla", default=env.get("VIGIA_PLANTILLA"), help="Plantilla de lanzamiento (opcional)")
    p.add_argument("--spot", default=env.get("VIGIA_SPOT", "no"), help="si: las máquinas nuevas, Spot (nube aws)")
    p.add_argument("--tipos", default=env.get("VIGIA_TIPOS", ""), help="tipos a probar, separados por coma")
    p.add_argument("--respaldo", default=env.get("VIGIA_RESPALDO", "si"),
                   help="si: sin lugar para Spot, crear a demanda (nube aws)")
    p.add_argument("--max-maquinas", type=int, default=int(env.get("VIGIA_MAX_MAQUINAS", "10")),
                   help="Tope de máquinas prendidas (protege la factura)")
    p.add_argument("--cada", type=int, default=60, help="Segundos entre revisiones")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not a.clave_maquinas:
        p.error("Falta la clave de máquinas (VIGIA_POOL_KEY)")
    if a.nube == "aws-asg":
        if not a.asg:
            p.error("Falta el nombre del grupo de Auto Scaling (VIGIA_ASG)")
        cloud = AsgPool(a.region, a.asg)
    elif a.nube == "aws":
        cloud = AwsPool(a.region, a.grupo, a.plantilla, spot=yes(a.spot), fallback=yes(a.respaldo),
                        types=[t.strip() for t in a.tipos.split(",")])
    else:
        cloud = NoCloud()
    log.info("Encendedor de máquinas: nube %s, %s, tope %d", a.nube,
             f"grupo de Auto Scaling {a.asg}" if a.nube == "aws-asg" else f"grupo {a.grupo}", a.max_maquinas)
    with httpx.Client(timeout=30.0) as http:
        while True:
            try:
                tick(a.server, a.clave_maquinas, cloud, a.max_maquinas, http)
            except Exception as exc:  # noqa: BLE001 - sin plan no se toca nada; se reintenta
                log.warning("No se pudo revisar (%s); no se toca nada", exc)
            time.sleep(a.cada)


if __name__ == "__main__":
    main()
