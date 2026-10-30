"""«En vivo» de la app de escritorio: mirar cámaras sin analizarlas.

- `ViewWorker`: la cámara no se analiza. Abre el video (igual que el análisis: RTSP o Hik-Connect) y deja el
  último cuadro para la ventana. No usa el detector, no guarda nada ni sube nada.
- `AIViewWorker` («Ver con la IA de esta PC», pedido del dueño 30 oct: «que se le aplique el YOLO y la postura y
  todo lo que tengamos, pero que no mida»): abre el video del grabador (aunque la analice una máquina: es otra
  transmisión) y le aplica EN ESTA PC lo mismo que el análisis (YOLO con objetos y bolsos, seguimiento, postura,
  zonas, segundos en cada zona, la marcha de los gestos), solo para mostrar cómo la capta. No guarda ni manda nada:
  ni alertas, ni métricas, ni imágenes. Si una regla o un gesto salta, se ve un cartel de prueba en la imagen.
- `RemoteView`: la cámara la analiza una máquina de análisis. No se le pide otra transmisión al grabador (por
  Hik-Connect da pocas a la vez): se muestra la imagen que sube la máquina, 1 por segundo y con los recuadros de
  la IA, con el mismo latido que usa el panel web (`POST /api/cameras/{id}/live`).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

from dataclasses import replace
from datetime import datetime

import cv2
import httpx
import numpy as np

from core.imagen import pixel_aspect, square_pixels

from .cameras import build_camera

log = logging.getLogger("vigia.vista")
SHOW_INTERVAL = 0.1  # la ventana no necesita más de 10 cuadros por segundo
BEAT_SECONDS = 10.0  # latido «alguien mira»: la máquina sube cuadros mientras lleguen


def _name(cfg: dict) -> str:
    return f"{cfg['client_name']} · {cfg['name']}" if cfg.get("client_name") else cfg["name"]


class ViewWorker:
    """Mirar una cámara que no se analiza."""

    remote = False

    def __init__(self, cfg: dict):
        self.id: int = cfg["id"]
        self.name = _name(cfg)
        self.hik = build_camera(cfg["connection"])
        self._sar: float | None = None
        self.state = "conectando"
        self.error: str | None = None
        self.fps = 0.0
        self.display: np.ndarray | None = None
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self.run, name=f"ver-{self.id}", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self._stop.set()

    def text(self) -> str:
        if self.state == "en_linea":
            return f"Viendo · {self.fps:.0f} fps"
        if self.state == "sin_conexion":
            return "Sin conexión" + (f": {self.error}" if self.error else "")
        return "Conectando..."

    def run(self) -> None:
        backoff = 2.0
        while not self._stop.is_set():
            try:
                url = self.hik.live_url()  # los links de Hik-Connect vencen: uno nuevo en cada conexión
            except Exception as exc:  # noqa: BLE001 - sin link no hay video; se reintenta
                self.state, self.error = "sin_conexion", str(exc)
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 60.0)
                continue
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                why = self.hik.open_failure(url)
                self.state, self.error = "sin_conexion", why or "no se pudo abrir el video"
                log.warning("[%s] Para mirar: %s", self.name, self.error)
                cap.release()
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 60.0)
                continue
            self.state, self.error, backoff = "en_linea", None, 2.0
            self._sar = pixel_aspect(cap)  # «1080p lite»: con píxeles cuadrados, como en las máquinas
            try:
                self._read(cap)
            finally:
                cap.release()
            if not self._stop.is_set():
                self.state = "conectando"
                self._stop.wait(2.0)
        self.state = "detenida"

    def _read(self, cap: cv2.VideoCapture) -> None:
        last_show = -1e9
        shown: list[float] = []
        fails = 0
        while not self._stop.is_set():
            if not cap.grab():  # tomar todos los cuadros (no acumular retraso); convertir solo los que se muestran
                fails += 1
                if fails >= 3:
                    return
                continue
            fails = 0
            now = time.monotonic()
            if now - last_show < SHOW_INTERVAL:
                continue
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                continue
            last_show = now
            if self._sar:
                frame = square_pixels(frame, self._sar, shrink=True)
            self._show(frame, now)
            shown = [t for t in shown if now - t < 5] + [now]
            self.fps = len(shown) / 5.0

    def _show(self, frame: np.ndarray, now: float) -> None:
        self.display = frame


class AIViewWorker(ViewWorker):
    """Mirar una cámara con la IA aplicada en esta PC, solo para ver cómo la capta (ver el docstring del módulo).

    El video se lee en un hilo (sin retrasarse) y la IA corre en otro, sobre el último cuadro, a los cuadros por
    segundo de la cámara (como en la máquina); si la PC no da abasto, analiza menos cuadros sin trabar el video.
    `detector_factory` y `pose_factory` arman (y comparten) el YOLO y la postura de la app."""

    def __init__(self, cfg: dict, detector_factory: Callable[[], object], pose_factory: Callable[[], object] | None):
        super().__init__(cfg)
        from core.rules import parse_config
        from core.tracker import CentroidTracker

        self.analysis = dict(cfg.get("analysis") or {})
        self.zones, self.rule_cfg = parse_config(self.analysis)
        if self.analysis.get("metrics_only"):  # como en la máquina: solo métricas no mira gestos ni reglas
            from core.gestos import GestureConfig
            self.rule_cfg.gestures = GestureConfig()
        self.target_fps = max(1.0, float(cfg.get("process_fps") or 5.0))
        self._detector_factory, self._pose_factory = detector_factory, pose_factory
        self.detector = self.pose = None
        self.tracker = CentroidTracker()
        self.engine = None
        self.ai_fps = 0.0
        self.people = 0
        self._latest: tuple[np.ndarray, float] | None = None
        self._new = threading.Event()
        self._hot: list = []  # carteles de prueba (reglas o gestos que saltaron), solo en pantalla
        self.ai_error: str | None = None
        self.ai_thread = threading.Thread(target=self._ai_loop, name=f"ver-ia-{self.id}", daemon=True)

    def start(self) -> None:
        super().start()
        self.ai_thread.start()

    def stop(self) -> None:
        super().stop()
        self._new.set()

    def text(self) -> str:
        if self.state != "en_linea":
            return super().text()
        if self.ai_error:
            return f"Viendo sin IA: {self.ai_error}"
        if self.detector is None:
            return "Viendo · cargando la IA..."
        return f"IA en esta PC · {self.ai_fps:.1f} cuadros/s · {self.people} persona{'s' if self.people != 1 else ''}"

    def _show(self, frame: np.ndarray, now: float) -> None:
        self._latest = (frame, now)
        self._new.set()
        if self.detector is None or self.ai_error:  # mientras carga la IA (o si falló), el video tal cual
            self.display = frame

    def _ai_loop(self) -> None:
        from core.rules import RuleEngine
        try:
            self.detector = self._detector_factory()
            self.pose = self._pose_factory() if self._pose_factory else None
        except Exception as exc:  # noqa: BLE001 - sin modelo se sigue viendo el video, sin IA
            log.warning("[%s] Ver con la IA: no se pudo cargar el detector (%s)", self.name, exc)
            self.ai_error = str(exc)
            return
        self.engine = RuleEngine(self.zones, self.rule_cfg)
        done: list[float] = []
        last_ts = -1e9
        while not self._stop.is_set():
            self._new.wait(1.0)
            self._new.clear()
            item = self._latest
            if item is None or self._stop.is_set():
                continue
            frame, ts = item
            if ts <= last_ts or ts - last_ts < 1.0 / self.target_fps - 0.02:
                continue
            last_ts = ts
            try:
                self.display = self.process(frame, ts, datetime.now().astimezone())
            except Exception as exc:  # noqa: BLE001 - un cuadro raro no corta la vista
                log.debug("[%s] Ver con la IA: %s", self.name, exc)
                continue
            now = time.monotonic()
            done = [t for t in done if now - t < 5] + [now]
            self.ai_fps = len(done) / 5.0

    def process(self, frame: np.ndarray, ts: float, wall: datetime) -> np.ndarray:
        """Un cuadro: detectar, seguir, mirar la postura, pasar las reglas (sin guardar nada) y dibujar."""
        from core.detectors import Detection
        from core.pose import look

        from .ajuste import WIDTH, Sample, TestAlert, draw, view_detect

        h, w = frame.shape[:2]
        gest = bool(self.rule_cfg.gestures.enabled)
        dets, objs = view_detect(self.detector, frame, gest)
        tracks = self.tracker.update(dets, ts, frame)
        posed = self.pose is not None
        if posed:  # la postura de cada persona, aunque la cámara no tenga gestos: para ver cómo la capta
            look(self.pose, frame, tracks, self.rule_cfg.gestures.min_height(h))
        else:
            for t in tracks:
                t.kpts = None
        bags = [Detection(o.bbox, o.label) for o in objs if o.label in ("mochila", "cartera", "valija")]
        events = self.engine.update(tracks, ts, (w, h), wall, bags)
        for e in events:  # no se guardan ni se mandan: solo el cartel en la imagen
            self._hot.append(TestAlert(0, 0, ts, wall, e.type, e.description, e.track_id))
        self._hot = [a for a in self._hot if ts - a.ts <= 3.0]
        people = [t for t in tracks if t.visible and t.label != "vehiculo"]
        self.people = len(people)
        sample = Sample(0, ts, wall, b"", (w, h), [replace(t, hist=None) for t in tracks],
                        [(o.label, o.conf, o.bbox) for o in objs], posed)
        small = frame if w <= WIDTH else cv2.resize(frame, (WIDTH, int(h * WIDTH / w)), interpolation=cv2.INTER_AREA)
        return draw(small, sample, self.engine.progress(ts), self.zones, self.rule_cfg, self._hot,
                    f"{wall:%H:%M:%S}  IA en esta PC (no guarda nada)")


class RemoteView:
    """Mirar una cámara que analiza una máquina: la imagen que sube la máquina (1 por segundo, con recuadros)."""

    remote = True

    def __init__(self, cfg: dict, beat: Callable[[int], dict], server: str):
        self.id: int = cfg["id"]
        self.name = _name(cfg)
        self.analyzer = cfg.get("analyzer")
        self.beat = beat
        self.server = server.rstrip("/")
        self.state = "esperando"
        self.error: str | None = None
        self.display: np.ndarray | None = None
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self.run, name=f"ver-maquina-{self.id}", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self._stop.set()

    def text(self) -> str:
        if self.state == "en_linea":
            return "Imagen de la máquina (1 por segundo, con recuadros)"
        if self.state == "maquina_apagada":
            return "La máquina que la analiza no está en línea"
        if self.state == "sin_conexion":
            return "Sin conexión con el servidor" + (f": {self.error}" if self.error else "")
        return "Esperando la imagen de la máquina..."

    def run(self) -> None:
        url, last_beat = None, -1e9
        with httpx.Client(timeout=10.0, follow_redirects=True) as http:
            while not self._stop.is_set():
                now = time.monotonic()
                if now - last_beat >= BEAT_SECONDS:
                    last_beat = now
                    try:
                        info = self.beat(self.id)
                        url = info.get("frame_url")
                        if not info.get("agent_online"):
                            self.state = "maquina_apagada"
                    except Exception as exc:  # noqa: BLE001 - se reintenta en el próximo latido
                        self.state, self.error = "sin_conexion", str(exc)
                if url and self.state != "maquina_apagada":
                    full = url if url.startswith("http") else self.server + url
                    try:
                        r = http.get(full + ("&" if "?" in full else "?") + f"t={int(time.time() * 1000)}")
                        img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR) \
                            if r.status_code == 200 else None
                        if img is not None:
                            self.display, self.state = img, "en_linea"
                    except httpx.HTTPError as exc:
                        self.error = str(exc)
                self._stop.wait(1.0)
        self.state = "detenida"
