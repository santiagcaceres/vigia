"""Agente local: corre en una PC dentro de la red del cliente, junto al NVR Hikvision.

    python -m agent --server https://vigia.tuempresa.com --token <TOKEN_DEL_AGENTE>

Descarga del servidor la lista de cámaras de su cliente, las analiza en tiempo real y sube
los eventos con foto y clip. Si se corta internet sigue analizando y guarda todo en disco.

Las cámaras en modo "alarmas" (Hik-Partner Pro) o "alarmas_equipo" (el grabador avisa solo)
no se miran en vivo: solo se analiza el tramo de video de cada alarma (ver agent/alarms.py y
agent/device_alarms.py).
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from core.detectors import Detector, build_detector
from core.scorer import Scorer

from .alarms import AlarmRunner
from .device_alarms import DeviceAlarmListener
from .imports import ImportJobRunner
from .outbox import Outbox, ServerClient
from .worker import CameraWorker

log = logging.getLogger("vigia.agent")
VERSION = "2.0.0"
CONFIG_EVERY_SECONDS = 60
STATUS_EVERY_SECONDS = 30


class Agent:
    def __init__(self, server: str, token: str, data_dir: Path, model_path: str | None, verify: bool | str = True,
                 process_imports: bool = True, alarm_workers: int = 2):
        self.client = ServerClient(server, token, verify=verify)
        self.process_imports = process_imports
        self.imports: ImportJobRunner | None = None
        self.alarms: AlarmRunner | None = None
        self.alarm_workers = alarm_workers
        self.data_dir = Path(data_dir)
        self.outbox = Outbox(self.data_dir / "outbox", self.client)
        self.model_path = model_path
        self.clip_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="clips")
        self.workers: dict[int, CameraWorker] = {}
        self.listeners: dict[int, DeviceAlarmListener] = {}
        self.scorer: Scorer | None = None
        self._scorer_id = None
        self._stop = threading.Event()

    def _detector(self, name: str) -> Detector:
        return build_detector(name, self.model_path)

    def apply_config(self, cfg: dict) -> None:
        if self.process_imports and cfg.get("process_imports", True) and self.imports is None:
            self.imports = ImportJobRunner(self.client, self.model_path, self.data_dir / "imports")
            self.imports.start()
            log.info("Procesamiento de videos importados: activado")
        model = cfg.get("filter_model")
        if (model or {}).get("id") != self._scorer_id:
            self.scorer = Scorer.maybe(model)
            self._scorer_id = (model or {}).get("id")
            for w in self.workers.values():  # nuevo modelo: reiniciar workers para usarlo
                w.fingerprint = ""
            log.info("Modelo de filtro: %s", self._scorer_id or "ninguno (se alertan todos los eventos)")

        cams = [c for c in cfg.get("cameras", []) if c.get("enabled", True)]
        if self.alarms is None and any(c.get("mode") in ("alarmas", "alarmas_equipo") for c in cams):
            self.alarms = AlarmRunner(self.client, self.model_path, self.data_dir / "alarmas", self.alarm_workers)
            self.alarms.start()
            log.info("Análisis de alarmas: activado (%d en paralelo)", self.alarm_workers)
        # cámaras cuyo grabador avisa las alarmas: un hilo escuchando por cámara
        listen = {c["id"]: c for c in cams if c.get("mode") == "alarmas_equipo"}
        for cid in list(self.listeners):
            lst = self.listeners[cid]
            if cid not in listen or lst.fingerprint != DeviceAlarmListener.fingerprint_of(listen[cid]):
                log.info("Dejando de escuchar alarmas de %s", lst.name)
                lst.stop()
                del self.listeners[cid]
        for cid, c in listen.items():
            if cid not in self.listeners:
                lst = DeviceAlarmListener(c, self.client)
                self.listeners[cid] = lst
                lst.start()
        wanted = {c["id"]: c for c in cams if c.get("mode", "vivo") == "vivo"}
        for cid in list(self.workers):
            w = self.workers[cid]
            if cid not in wanted or w.fingerprint != CameraWorker.fingerprint_of(wanted[cid]):
                log.info("Deteniendo cámara %s", w.name)
                w.stop()
                w.thread.join(15)
                del self.workers[cid]
        for cid, c in wanted.items():
            if cid in self.workers:
                continue
            try:
                det = self._detector(c.get("detector", "yolo"))
            except Exception as exc:
                log.error("Cámara %s: no se pudo crear el detector: %s", c["name"], exc)
                continue
            # modo central: cada cámara trae el modelo de su cliente; si no, se usa el del agente
            scorer = Scorer.maybe(c["filter_model"]) if c.get("filter_model") is not None else self.scorer
            w = CameraWorker(c, det, scorer, self.outbox, self.client, self.clip_pool, cfg.get("settings", {}))
            self.workers[cid] = w
            w.start()
            log.info("Iniciada cámara %s", c["name"])

    def run(self) -> None:
        self.outbox.start()
        last_cfg = last_status = -1e9
        while not self._stop.is_set():
            now = time.monotonic()
            if now - last_cfg >= CONFIG_EVERY_SECONDS:
                try:
                    self.apply_config(self.client.config())
                    last_cfg = now
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code == 401:
                        log.error("Token de agente inválido. Revisalo en el panel (Administración > Agentes).")
                    else:
                        log.warning("No se pudo obtener la configuración: %s", exc)
                    last_cfg = now - CONFIG_EVERY_SECONDS + 15
                except httpx.HTTPError as exc:
                    log.warning("Servidor no disponible (%s); las cámaras siguen funcionando", exc)
                    last_cfg = now - CONFIG_EVERY_SECONDS + 15
            if now - last_status >= STATUS_EVERY_SECONDS:
                last_status = now
                try:
                    self.client.status({"cameras": [w.status() for w in self.workers.values()]
                                        + [l.status() for l in self.listeners.values()],
                                        "outbox_pending": self.outbox.pending(), "version": VERSION,
                                        **(self.alarms.status() if self.alarms else {})})
                except httpx.HTTPError:
                    pass
            self._stop.wait(1.0)
        self.shutdown()

    def stop(self) -> None:
        self._stop.set()

    def shutdown(self) -> None:
        if self.imports:
            self.imports.stop()
        if self.alarms:
            self.alarms.stop()
        for lst in self.listeners.values():
            lst.stop()
        for w in self.workers.values():
            w.stop()
        for w in self.workers.values():
            w.thread.join(15)
        self.clip_pool.shutdown(wait=True)
        self.outbox.stop()


SERVIDOR_POR_DEFECTO = "https://vigia-three-ashy.vercel.app"
ARCHIVO_CONFIG = "agente.env"


def base_dir() -> Path:
    """Carpeta donde vive el programa: la del .exe si está empaquetado, si no la del repo."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def bundled_model() -> Path | None:
    """Modelo ONNX que viene dentro del .exe (PyInstaller lo descomprime en sys._MEIPASS)."""
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        cand = Path(meipass) / "models" / "personas.onnx"
        if cand.exists():
            return cand
    return None


def read_env_file(path: Path) -> dict[str, str]:
    """Lee `CLAVE=valor` por línea (sin comillas ni comentarios complicados)."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def write_env_file(path: Path, values: dict[str, str]) -> None:
    lines = ["# Configuración del agente Vigía. Borrá este archivo para volver a pedir los datos."]
    lines += [f"{k}={v}" for k, v in values.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def pedir_datos_interactivo(cfg_path: Path, server: str | None, token: str | None) -> tuple[str, str]:
    """Si faltan servidor o token y hay una consola, los pide y los guarda en agente.env."""
    if not sys.stdin or not sys.stdin.isatty():
        return server or "", token or ""
    print("Agente Vigía: configuración inicial (se guarda en %s)" % cfg_path)
    if not server:
        server = input(f"Servidor [{SERVIDOR_POR_DEFECTO}]: ").strip() or SERVIDOR_POR_DEFECTO
    while not token:
        token = input("Token del agente (Administración → Agentes en el panel): ").strip()
    write_env_file(cfg_path, {"VIGIA_SERVER": server, "VIGIA_AGENT_TOKEN": token})
    print("Guardado. La próxima vez arranca solo.\n")
    return server, token


def main() -> None:
    base = base_dir()
    cfg_path = Path(os.getenv("VIGIA_AGENT_CONFIG", base / ARCHIVO_CONFIG))
    env = {**read_env_file(cfg_path), **{k: v for k, v in os.environ.items() if k.startswith("VIGIA_")}}
    modelo_defecto = env.get("VIGIA_PERSON_MODEL") or str(bundled_model() or base / "models" / "personas.onnx")
    p = argparse.ArgumentParser(description="Agente Vigía para cámaras Hikvision y Dahua")
    p.add_argument("--server", default=env.get("VIGIA_SERVER"), help="URL del servidor central")
    p.add_argument("--token", default=env.get("VIGIA_AGENT_TOKEN"), help="Token del agente (del panel)")
    p.add_argument("--data", default=env.get("VIGIA_AGENT_DATA", str(base / "agent_data")))
    p.add_argument("--model", default=modelo_defecto, help="Modelo ONNX de detección de personas")
    p.add_argument("--insecure", action="store_true", help="No verificar HTTPS (solo pruebas)")
    p.add_argument("--no-imports", action="store_true", help="No analizar videos importados en esta PC")
    p.add_argument("--alarm-workers", type=int, default=int(env.get("VIGIA_ALARM_WORKERS", "2")),
                   help="Alarmas que se bajan y analizan en paralelo")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not a.server or not a.token:
        a.server, a.token = pedir_datos_interactivo(cfg_path, a.server, a.token)
    if not a.server or not a.token:
        p.error(f"Faltan --server y --token (o VIGIA_SERVER y VIGIA_AGENT_TOKEN, o el archivo {cfg_path})")
    agent = Agent(a.server, a.token, Path(a.data), a.model, verify=not a.insecure, process_imports=not a.no_imports,
                  alarm_workers=max(1, a.alarm_workers))
    signal.signal(signal.SIGTERM, lambda *_: agent.stop())
    signal.signal(signal.SIGINT, lambda *_: agent.stop())
    try:
        agent.run()
    except Exception:
        log.exception("El agente se detuvo por un error")
        if getattr(sys, "frozen", False) and sys.stdin and sys.stdin.isatty():
            input("Apretá Enter para cerrar...")
        raise


if __name__ == "__main__":
    main()
