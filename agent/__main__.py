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
from .live import LiveWorker
from .outbox import Outbox, ServerClient
from .ventana import ESTADOS, Ventana
from .worker import CameraWorker

log = logging.getLogger("vigia.agent")
VERSION = "2.0.0"
CONFIG_EVERY_SECONDS = 60
STATUS_EVERY_SECONDS = 10  # la respuesta trae qué cámaras se están mirando en vivo
LIVE_EXTEND_SECONDS = 45.0  # cuánto sigue el vivo después del último aviso del servidor


class Agent:
    def __init__(self, server: str, token: str, data_dir: Path, model_path: str | None, verify: bool | str = True,
                 process_imports: bool = True, alarm_workers: int = 2, window: bool = False):
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
        self.live_workers: dict[int, LiveWorker] = {}  # vivo de cámaras en modo alarmas
        self._cams_cfg: dict[int, dict] = {}
        self.window = Ventana(self.data_dir / "ventana.json") if window else None
        if self.window is not None and self.window.failed:
            self.window = None
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
        self._cams_cfg = {c["id"]: c for c in cams}
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
                    resp = self.client.status({"cameras": [w.status() for w in self.workers.values()]
                                               + [l.status() for l in self.listeners.values()],
                                               "live": self.live_ids(),
                                               "outbox_pending": self.outbox.pending(), "version": VERSION,
                                               **(self.alarms.status() if self.alarms else {})})
                    self.apply_live(resp.get("live") or [])
                except httpx.HTTPError:
                    pass
            if self.window is not None:
                self.window.set_cameras(self.window_cameras())
                self.window.refresh(self.window_tiles(self.window.selected_ids()))
                if not self.window.pump(40):
                    log.info("Ventana cerrada; el agente sigue trabajando (Ctrl+C para detenerlo)")
                    self.window = None
                    for w in list(self.workers.values()) + list(self.live_workers.values()):
                        w.show_window = False
            else:
                self._stop.wait(1.0)
        self.shutdown()

    # ---- ventana de la PC central ----
    def _cam_state(self, cid: int) -> str:
        w = self.workers.get(cid)
        lw = self.live_workers.get(cid)
        if w is not None:
            return ESTADOS.get(w.state, w.state) + (f" · {w.fps:.0f} fps" if w.state == "en_linea" and w.fps else "")
        if lw is not None and lw.thread.is_alive():
            return "En vivo (panel)"
        if cid in self.listeners:
            lst = self.listeners[cid]
            return ESTADOS.get(lst.state, lst.state)
        return "Escuchando alarmas"

    def window_cameras(self) -> list[tuple[int, str, str, str]]:
        """Todas las cámaras asignadas, para la lista de la izquierda: (id, cliente, nombre, estado)."""
        return [(cid, c.get("client_name") or "", c["name"], self._cam_state(cid))
                for cid, c in sorted(self._cams_cfg.items(), key=lambda kv: (kv[1].get("client_name") or "", kv[1]["name"]))]

    def window_tiles(self, ids: set[int]) -> list:
        """Celdas del mosaico: solo las cámaras elegidas; las demás no gastan en dibujar recuadros."""
        tiles = []
        for cid, c in sorted(self._cams_cfg.items()):
            w = self.workers.get(cid)
            lw = self.live_workers.get(cid)
            target = w if w is not None else (lw if lw is not None and lw.thread.is_alive() else None)
            if target is not None:
                target.show_window = cid in ids
            if cid not in ids:
                continue
            name = f"{c.get('client_name', '')} · {c['name']}" if c.get("client_name") else c["name"]
            if w is not None:
                tiles.append((name, w.display, {"en_linea": "Conectando...", "sin_conexion": "Sin conexion con el grabador",
                                                 "reconectando": "Reconectando...", "error": "Error: " + str(w.error or "")}
                              .get(w.state, w.state)))
            elif lw is not None and lw.thread.is_alive():
                tiles.append((name, lw.display, "Abriendo el stream..."))
            else:
                tiles.append((name, None, "Escuchando alarmas del grabador (video solo cuando alguien mira desde el panel)"))
        return tiles

    # ---- vista en vivo ----
    def live_ids(self) -> list[int]:
        ids = [w.id for w in self.workers.values() if w.live.active]
        ids += [lw.id for lw in self.live_workers.values() if lw.active]
        return ids

    def apply_live(self, ids: list[int]) -> None:
        """El servidor dice qué cámaras tienen a alguien mirando: se extiende el vivo de cada una."""
        for cid in ids:
            if cid in self.workers:  # modo vivo: el worker ya tiene el cuadro analizado
                self.workers[cid].live.extend(LIVE_EXTEND_SECONDS)
                continue
            lw = self.live_workers.get(cid)
            if lw is not None and lw.thread.is_alive():
                lw.extend(LIVE_EXTEND_SECONDS)
                continue
            c = self._cams_cfg.get(cid)
            if c is None:
                continue
            try:
                lw = LiveWorker(c, self._detector(c.get("detector", "yolo")), self.client)
            except Exception as exc:
                log.error("Vivo de %s: no se pudo crear el detector: %s", c["name"], exc)
                continue
            lw.extend(LIVE_EXTEND_SECONDS)
            self.live_workers[cid] = lw
            lw.start()
            log.info("Vivo de %s: alguien lo está mirando, abriendo el stream", lw.name)
        for cid in list(self.live_workers):
            if not self.live_workers[cid].thread.is_alive():
                del self.live_workers[cid]

    def stop(self) -> None:
        self._stop.set()

    def shutdown(self) -> None:
        if self.imports:
            self.imports.stop()
        if self.alarms:
            self.alarms.stop()
        for lst in self.listeners.values():
            lst.stop()
        for lw in self.live_workers.values():
            lw.stop()
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
    p.add_argument("--ventana", dest="ventana", action="store_true", default=None,
                   help="Mostrar la ventana con las cámaras y los recuadros (en el .exe viene activada)")
    p.add_argument("--sin-ventana", dest="ventana", action="store_false", help="Sin ventana (servidor sin pantalla)")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args()
    if a.ventana is None:
        a.ventana = env.get("VIGIA_VENTANA", "1" if getattr(sys, "frozen", False) else "0") not in ("0", "no", "false")
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not a.server or not a.token:
        a.server, a.token = pedir_datos_interactivo(cfg_path, a.server, a.token)
    if not a.server or not a.token:
        p.error(f"Faltan --server y --token (o VIGIA_SERVER y VIGIA_AGENT_TOKEN, o el archivo {cfg_path})")
    agent = Agent(a.server, a.token, Path(a.data), a.model, verify=not a.insecure, process_imports=not a.no_imports,
                  alarm_workers=max(1, a.alarm_workers), window=a.ventana)
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
