"""Encendedor de máquinas de análisis: prende y apaga las máquinas en la nube según los horarios de los locales.

    python -m agent.escalador            # lee /etc/vigia/escalador.env o variables VIGIA_* / AWS_*

Cada minuto pregunta al servidor de Vigía (`GET /api/agent/scale`, con la clave de máquinas) cuántas
máquinas tienen que estar prendidas, mirando también los próximos 20 minutos (así una máquina está lista
antes de que abra el local). Si faltan, prende máquinas apagadas del grupo (o crea nuevas con la plantilla);
si sobran, el servidor primero les saca las cámaras y recién entonces aparecen en `stop` y se apagan.

Corre en una máquina chiquita siempre prendida (en AWS una t4g.nano, ~US$ 3 por mes) o en cualquier PC que
quede prendida. Las claves de la nube viven solo acá (mejor: un rol de IAM), nunca en Vigía.

Nubes:
- `aws`: EC2. Las máquinas del grupo llevan la etiqueta `vigia-grupo=<grupo>`. Permisos que necesita:
  ec2:DescribeInstances, ec2:StartInstances, ec2:StopInstances (y ec2:RunInstances + ec2:CreateTags si se
  usa plantilla para crear máquinas nuevas).
- `ninguna`: no prende ni apaga nada, solo registra cuántas máquinas harían falta (por ejemplo en Tigo
  mientras no sepamos cómo se manejan sus máquinas por programa).
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

log = logging.getLogger("vigia.escalador")
BROKEN_AFTER = timedelta(minutes=20)  # prendida hace más de esto y sin darse de alta en Vigía = rota


@dataclass
class Instance:
    id: str
    state: str  # pending | running | stopping | stopped
    launched: datetime | None = None


@dataclass
class Decision:
    start: list[str]
    launch: int  # máquinas nuevas a crear con la plantilla
    stop: list[str]
    missing: int  # faltan máquinas en el grupo y no hay plantilla


def decide(plan: dict, instances: list[Instance], max_machines: int, can_launch: bool,
           now: datetime | None = None) -> Decision:
    """Qué prender y qué apagar (sin tocar nada: solo decide)."""
    now = now or datetime.now(timezone.utc)
    desired = max(0, min(int(plan.get("desired") or 0), max_machines))
    running = [i for i in instances if i.state in ("pending", "running")]
    stopped = sorted((i for i in instances if i.state == "stopped"), key=lambda i: i.id)
    start: list[str] = []
    launch = missing = 0
    if len(running) < desired:
        need = desired - len(running)
        start = [i.id for i in stopped[:need]]
        rest = need - len(start)
        if rest > 0:
            launch, missing = (rest, 0) if can_launch else (0, rest)
    stop: list[str] = []
    excess = len(running) - desired
    if excess > 0:
        online = set(plan.get("online_instances") or [])
        drained = [i for i in (plan.get("stop") or []) if i in {r.id for r in running}]
        broken = [r.id for r in running if r.id not in online and r.state == "running" and r.launched
                  and now - r.launched > BROKEN_AFTER and r.id not in drained]
        stop = (drained + broken)[:excess]
    return Decision(start, launch, stop, missing)


class AwsPool:
    def __init__(self, region: str, group: str, template: str | None = None):
        import boto3  # solo en la máquina del encendedor (pip install boto3)

        self.ec2 = boto3.client("ec2", region_name=region)
        self.group = group
        self.template = template

    def instances(self) -> list[Instance]:
        out = []
        pages = self.ec2.get_paginator("describe_instances").paginate(Filters=[
            {"Name": "tag:vigia-grupo", "Values": [self.group]},
            {"Name": "instance-state-name", "Values": ["pending", "running", "stopping", "stopped"]}])
        for page in pages:
            for res in page["Reservations"]:
                for i in res["Instances"]:
                    out.append(Instance(i["InstanceId"], i["State"]["Name"], i.get("LaunchTime")))
        return out

    def start(self, ids: list[str]) -> None:
        self.ec2.start_instances(InstanceIds=ids)

    def stop(self, ids: list[str]) -> None:
        self.ec2.stop_instances(InstanceIds=ids)

    def launch(self, n: int) -> None:
        tmpl = {"LaunchTemplateId": self.template} if self.template.startswith("lt-") else \
            {"LaunchTemplateName": self.template}
        self.ec2.run_instances(LaunchTemplate=tmpl, MinCount=n, MaxCount=n, TagSpecifications=[
            {"ResourceType": "instance", "Tags": [{"Key": "vigia-grupo", "Value": self.group},
                                                  {"Key": "Name", "Value": f"vigia-{self.group}"}]}])


class NoCloud:
    """Sin nube manejable por programa: solo avisa cuántas máquinas harían falta."""

    def instances(self) -> list[Instance]:
        return []

    def start(self, ids: list[str]) -> None:
        pass

    def stop(self, ids: list[str]) -> None:
        pass

    def launch(self, n: int) -> None:
        pass


def tick(server: str, key: str, cloud, max_machines: int, http: httpx.Client) -> Decision | None:
    r = http.get(server.rstrip("/") + "/api/agent/scale", headers={"X-Pool-Key": key})
    r.raise_for_status()
    plan = r.json()
    p = plan.get("plan") or plan
    if isinstance(cloud, NoCloud):
        log.info("Hacen falta %s máquinas (carga %s, %s en línea) — sin nube conectada, no se toca nada",
                 p.get("desired"), p.get("units"), p.get("online"))
        return None
    d = decide({**p, "online_instances": plan.get("online_instances")}, cloud.instances(), max_machines,
               bool(getattr(cloud, "template", None)))
    if d.start:
        log.info("Prendiendo %d máquina(s): %s", len(d.start), ", ".join(d.start))
        cloud.start(d.start)
    if d.launch:
        log.info("Creando %d máquina(s) nueva(s) con la plantilla", d.launch)
        cloud.launch(d.launch)
    if d.missing:
        log.warning("Faltan %d máquina(s) en el grupo y no hay plantilla para crearlas", d.missing)
    if d.stop:
        log.info("Apagando %d máquina(s) sin cámaras: %s", len(d.stop), ", ".join(d.stop))
        cloud.stop(d.stop)
    return d


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


def main() -> None:
    env = {**read_env_file(Path(os.getenv("VIGIA_ESCALADOR_CONFIG", "/etc/vigia/escalador.env"))),
           **{k: v for k, v in os.environ.items() if k.startswith(("VIGIA_", "AWS_"))}}
    p = argparse.ArgumentParser(description="Prende y apaga las máquinas de análisis de Vigía")
    p.add_argument("--server", default=env.get("VIGIA_SERVER", "https://vigia-three-ashy.vercel.app"))
    p.add_argument("--clave-maquinas", default=env.get("VIGIA_POOL_KEY"))
    p.add_argument("--nube", default=env.get("VIGIA_NUBE", "aws"), choices=["aws", "ninguna"])
    p.add_argument("--region", default=env.get("AWS_REGION", "sa-east-1"))
    p.add_argument("--grupo", default=env.get("VIGIA_GRUPO", "analisis"))
    p.add_argument("--plantilla", default=env.get("VIGIA_PLANTILLA"), help="Plantilla de lanzamiento (opcional)")
    p.add_argument("--max-maquinas", type=int, default=int(env.get("VIGIA_MAX_MAQUINAS", "10")),
                   help="Tope de máquinas prendidas (protege la factura)")
    p.add_argument("--cada", type=int, default=60, help="Segundos entre revisiones")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not a.clave_maquinas:
        p.error("Falta la clave de máquinas (VIGIA_POOL_KEY)")
    cloud = AwsPool(a.region, a.grupo, a.plantilla) if a.nube == "aws" else NoCloud()
    log.info("Encendedor de máquinas: nube %s, grupo %s, tope %d", a.nube, a.grupo, a.max_maquinas)
    with httpx.Client(timeout=30.0) as http:
        while True:
            try:
                tick(a.server, a.clave_maquinas, cloud, a.max_maquinas, http)
            except Exception as exc:  # noqa: BLE001 - sin plan no se toca nada; se reintenta
                log.warning("No se pudo revisar (%s); no se toca nada", exc)
            time.sleep(a.cada)


if __name__ == "__main__":
    main()
