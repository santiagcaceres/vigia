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
import logging.handlers
from collections.abc import Callable
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from core.detectors import Detector, build_detector
from core.rules import parse_zones
from core.scorer import Scorer

from . import actualizar, hikconnect
from .alarms import AlarmRunner
from .device_alarms import DeviceAlarmListener
from .imports import ImportJobRunner
from .live import LiveWorker
from .outbox import Outbox, ServerClient, login_pc
from .ventana import ESTADOS, Ventana, default_pc_name
from .worker import CameraWorker

log = logging.getLogger("vigia.agent")
VERSION = "2.0.0"
CONFIG_EVERY_SECONDS = 20  # cada cuánto se piden cámaras, zonas y reglas (además se pide al guardar en la app)
STATUS_EVERY_SECONDS = 10  # la respuesta trae qué cámaras se están mirando en vivo
LIVE_EXTEND_SECONDS = 45.0  # cuánto sigue el vivo después del último aviso del servidor


class Agent:
    def __init__(self, server: str, token: str, data_dir: Path, model_path: str | None, verify: bool | str = True,
                 process_imports: bool = True, alarm_workers: int = 2, window: "bool | Ventana" = False,
                 relogin: "Callable[[str | None], str | None] | None" = None, pc_name: str = ""):
        self.client = ServerClient(server, token, verify=verify)
        # cámaras por Hik-Connect: el link de video lo da el servidor (siempre con el cliente vigente)
        hikconnect.set_url_provider(lambda cid, kind, quality, start, end:
                                    self.client.stream_url(cid, kind, quality, start, end))
        self.relogin = relogin  # pide usuario y clave de nuevo (ventana) y devuelve el token nuevo
        self.pc_name = pc_name
        self._logout_requested = False
        self._force_config = False
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
        if isinstance(window, Ventana):
            self.window: Ventana | None = window
        else:
            self.window = Ventana(self.data_dir / "ventana.json") if window else None
        if self.window is not None and self.window.failed:
            self.window = None
        if self.window is not None and self.window.on_logout is None:
            self.window.on_logout = self._request_logout
        self.scorer: Scorer | None = None
        self._scorer_id = None
        self._stop = threading.Event()

    def _detector(self, name: str, cam_cfg: dict | None = None) -> Detector:
        vehicles = bool(((cam_cfg or {}).get("analysis") or {}).get("vehicles"))
        return build_detector(name, self.model_path, vehicles=vehicles)

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
            elif w.analysis_fp != CameraWorker.analysis_of(wanted[cid]):  # solo cambiaron zonas o reglas
                w.analysis_fp = CameraWorker.analysis_of(wanted[cid])
                w.update_analysis(wanted[cid].get("analysis") or {})
        for cid, lw in self.live_workers.items():  # vivo de cámaras en modo alarmas: zonas nuevas en el dibujo
            if cid in self._cams_cfg:
                try:
                    lw.streamer.zones = parse_zones((self._cams_cfg[cid].get("analysis") or {}).get("zones"))
                except ValueError:
                    pass
        for cid, c in wanted.items():
            if cid in self.workers:
                continue
            try:
                det = self._detector(c.get("detector", "yolo"), c)
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
            if self._force_config:
                self._force_config = False
                last_cfg = -1e9
            if now - last_cfg >= CONFIG_EVERY_SECONDS:
                try:
                    self.apply_config(self.client.config())
                    last_cfg = now
                    if self.window is not None:
                        n = len(self._cams_cfg)
                        self.window.set_status(f"{self.pc_name} · analizando {n} cámara{'s' if n != 1 else ''}")
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code == 401:
                        log.error("Esta PC ya no está vinculada (la borraron del panel o cambió la clave).")
                        if self._ask_login("Esta PC ya no está vinculada. Entrá de nuevo con tu usuario."):
                            last_cfg = -1e9
                            continue
                    else:
                        log.warning("No se pudo obtener la configuración: %s", exc)
                    last_cfg = now - CONFIG_EVERY_SECONDS + 15
                except httpx.HTTPError as exc:
                    log.warning("Servidor no disponible (%s); las cámaras siguen funcionando", exc)
                    if self.window is not None:
                        self.window.set_status(f"{self.pc_name} · sin conexión con el servidor (reintentando)", ok=False)
                    last_cfg = now - CONFIG_EVERY_SECONDS + 15
            if self._logout_requested:
                self._logout_requested = False
                self.apply_config({"cameras": []})  # deja de mirar las cámaras hasta volver a entrar
                if self._ask_login(None):
                    last_cfg = -1e9
                    continue
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
                note = "" if self._cams_cfg else ("Esta PC todavía no tiene cámaras asignadas. En «Clientes y cámaras», "
                                                  f"al cargar o editar una cámara, elegí esta PC ({self.pc_name}).")
                self.window.refresh(self.window_tiles(self.window.selected_ids()), note or " ")
                if not self.window.pump(40):
                    log.info("Ventana cerrada: Vigía se detiene")
                    self.window = None
                    self.stop()
            else:
                self._stop.wait(1.0)
        self.shutdown()

    def refresh_config(self) -> None:
        """Pedir ya la configuración (se guardaron cámaras, zonas o reglas en la app)."""
        self._force_config = True

    def local_probe(self, cam_id: int) -> dict | None:
        """Prueba la cámara DESDE ESTA PC (página del grabador + video). None si no la analiza esta PC."""
        c = self._cams_cfg.get(cam_id)
        if c is None:
            return None
        from .cameras import build_camera, probe, test_stream
        cam = build_camera(c["connection"])
        info = probe(cam)
        try:
            video = test_stream(cam)
        except Exception as exc:  # noqa: BLE001
            video = {"ok": False, "error": str(exc)}
        return {"http": info, "video": video, "rtsp_url": cam.redacted(cam.live_url())}

    def _request_logout(self) -> None:
        self._logout_requested = True

    def _ask_login(self, error: str | None) -> bool:
        """Vuelve a pedir usuario y clave (si hay ventana). True si quedó vinculada con un token nuevo."""
        if self.relogin is None or self.window is None:
            return False
        token = self.relogin(error)
        if not token:
            self.stop()
            return False
        self.client.http.headers["X-Agent-Token"] = token
        log.info("PC vinculada de nuevo")
        return True

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
                lw = LiveWorker(c, self._detector(c.get("detector", "yolo"), c), self.client)
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


def pedir_datos_interactivo(cfg_path: Path, server: str | None, token: str | None,
                            verify: bool | str = True) -> tuple[str, str]:
    """Sin ventana: si falta el token y hay una consola, pide usuario y clave de administrador (o un
    token pegado a mano) y guarda lo obtenido en agente.env."""
    if not sys.stdin or not sys.stdin.isatty():
        return server or "", token or ""
    import getpass

    print("Vigía · PC central: configuración inicial (se guarda en %s)" % cfg_path)
    if not server:
        server = input(f"Servidor [{SERVIDOR_POR_DEFECTO}]: ").strip() or SERVIDOR_POR_DEFECTO
    pc_name = default_pc_name()
    while not token:
        user = input("Usuario administrador (o pegá un token de agente): ").strip()
        if not user:
            continue
        if len(user) > 30 and " " not in user:  # pegaron un token
            token = user
            break
        pwd = getpass.getpass("Clave: ")
        try:
            token = login_pc(server, user, pwd, pc_name, verify=verify)["token"]
        except ValueError as exc:
            print(f"No se pudo: {exc}")
    write_env_file(cfg_path, {"VIGIA_SERVER": server, "VIGIA_AGENT_TOKEN": token, "VIGIA_PC_NAME": pc_name})
    print("Guardado. La próxima vez arranca solo.\n")
    return server, token


def setup_logging(data_dir: Path, verbose: bool) -> None:
    """Registro a archivo (agent_data/vigia.log, rota en 3 x 1 MB) y a la consola si la hay."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(data_dir / "vigia.log", maxBytes=1_000_000, backupCount=3,
                                                  encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except OSError:
        pass
    if sys.stderr is not None:  # el .exe sin consola no tiene stderr
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)


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
    data_dir = Path(a.data)
    setup_logging(data_dir, a.verbose)
    threading.Thread(target=actualizar.cleanup_old, daemon=True).start()
    a.server = a.server or SERVIDOR_POR_DEFECTO
    verify = not a.insecure
    if a.ventana:
        ventana = Ventana(data_dir / "ventana.json")
        if not ventana.failed:
            return run_app(a, env, cfg_path, data_dir, ventana, verify)
        log.warning("Sin ventana: sigue como servicio (solo análisis)")
    # sin ventana (servidor): solo analiza, con el token del agente
    pc_name = env.get("VIGIA_PC_NAME") or default_pc_name()
    if not a.token:
        a.server, a.token = pedir_datos_interactivo(cfg_path, a.server, a.token, verify=verify)
    if not a.token:
        p.error(f"Falta el token (entrá con --ventana, o VIGIA_AGENT_TOKEN, o el archivo {cfg_path})")
    agent = Agent(a.server, a.token, data_dir, a.model, verify=verify, process_imports=not a.no_imports,
                  alarm_workers=max(1, a.alarm_workers), window=False, pc_name=pc_name)
    signal.signal(signal.SIGTERM, lambda *_: agent.stop())
    signal.signal(signal.SIGINT, lambda *_: agent.stop())
    agent.run()


def restart_app() -> None:
    """Vuelve a abrir el programa (para cerrar sesión y entrar con otro usuario)."""
    import subprocess

    if getattr(sys, "frozen", False):
        cmd = [sys.executable] + sys.argv[1:]
    else:
        cmd = [sys.executable, "-m", "agent"] + sys.argv[1:]
    # el .exe (PyInstaller de un solo archivo) nuevo tiene que desempaquetarse de cero y no usar la
    # carpeta temporal de este proceso, que se borra al cerrarlo
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("_MEI", "_PYI")) and k not in ("TCL_LIBRARY", "TK_LIBRARY")}
    env["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
    subprocess.Popen(cmd, close_fds=True, env=env, cwd=str(Path(sys.executable).parent)
                     if getattr(sys, "frozen", False) else None)


def run_app(a, env: dict, cfg_path: Path, data_dir: Path, ventana: Ventana, verify: bool | str) -> None:
    """La app de escritorio: inicio de sesión de usuario (administrador o empleado), pestañas de gestión y
    el análisis de las cámaras que el admin le asignó a ese usuario (pestaña «En vivo»). No hay que elegir
    ni vincular PCs: la PC de cada usuario se crea sola en el servidor."""
    from .oficina import Api, ApiError, Oficina

    saved = dict(read_env_file(cfg_path))
    api = Api(a.server, verify=verify)

    def save_env() -> None:
        write_env_file(cfg_path, {"VIGIA_SERVER": a.server, "VIGIA_SESSION": api.token or ""})

    def do_login(user: str, pwd: str, *_ignored) -> bool:
        try:
            res = api.login(user, pwd)
        except ApiError as exc:
            raise ValueError(str(exc)) from exc
        if res["user"]["role"] == "cliente":
            api.set_token(None)
            raise ValueError("Este usuario es de un cliente: los clientes entran por la web.")
        save_env()
        log.info("Sesión iniciada como %s", user)
        return True

    # sesión guardada
    logged = False
    if saved.get("VIGIA_SESSION"):
        api.set_token(saved["VIGIA_SESSION"])
        try:
            logged = api.me().get("role") in ("admin", "entrenador")
        except ApiError:
            logged = False
    if not logged:
        api.set_token(None)
        if not ventana.login(do_login, None, None, True):
            return  # cerraron la ventana sin entrar

    def logout() -> None:
        try:
            api.post("/api/logout")
        except ApiError:
            pass
        write_env_file(cfg_path, {"VIGIA_SERVER": a.server})
        restart_app()
        ventana.close()

    # la PC de este usuario: analiza y muestra las cámaras que el admin le asignó
    try:
        session = api.post("/api/agent/session")
    except ApiError as exc:
        ventana.messagebox.showerror("Vigía", f"No se pudo preparar el análisis de cámaras: {exc}")
        return
    user = api.user or {}
    n = len(session.get("camera_ids") or [])
    title = (f"{user.get('username')} ({'administrador' if user.get('role') == 'admin' else 'empleado'})"
             f" · {n} cámara{'s' if n != 1 else ''} asignada{'s' if n != 1 else ''}")
    ventana.on_logout = logout
    ventana.build_main(title, analyze=True)
    holder: dict = {}

    def reopen() -> None:
        """Después de instalar una versión nueva: abrir el .exe nuevo (con la misma sesión) y cerrar este."""
        restart_app()
        ventana.close()

    oficina = Oficina(ventana, api, on_logout=logout, on_restart=reopen,
                      local_probe=lambda cid: holder["agent"].local_probe(cid) if "agent" in holder else None,
                      on_config_saved=lambda: holder["agent"].refresh_config() if "agent" in holder else None)
    oficina.build(ventana.notebook)
    ventana.hooks.append(oficina.bg.drain)

    def relogin(_error: str | None) -> str | None:
        """El servidor rechazó el token de esta PC (el mismo usuario entró en otra PC): volver a entrar."""
        ventana.messagebox.showwarning("Vigía", "Tu usuario entró en otra PC, así que esta dejó de analizar "
                                                "las cámaras. Entrá de nuevo para retomarlas acá.")
        logout()
        return None

    agent = Agent(a.server, session["token"], data_dir, a.model, verify=verify, process_imports=not a.no_imports,
                  alarm_workers=max(1, a.alarm_workers), window=ventana, relogin=relogin,
                  pc_name=f"PC de {user.get('username')}")
    holder["agent"] = agent
    signal.signal(signal.SIGTERM, lambda *_: agent.stop())
    signal.signal(signal.SIGINT, lambda *_: agent.stop())
    try:
        agent.run()
    except Exception as exc:
        log.exception("El agente se detuvo por un error")
        if ventana.open:
            try:
                ventana.messagebox.showerror("Vigía", f"Vigía se detuvo por un error:\n{exc}\n\n"
                                                      f"Detalle en {data_dir / 'vigia.log'}")
            except Exception:
                pass
        raise


if __name__ == "__main__":
    main()
