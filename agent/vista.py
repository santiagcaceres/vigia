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
from collections import deque
from collections.abc import Callable

from dataclasses import replace
from datetime import datetime

import cv2
import httpx
import numpy as np

from core.imagen import pixel_aspect, square_pixels

from .cameras import build_camera
from .worker import MediaClock

log = logging.getLogger("vigia.vista")
SHOW_INTERVAL = 0.1  # la ventana no necesita más de 10 cuadros por segundo
BEAT_SECONDS = 10.0  # latido «alguien mira»: la máquina sube cuadros mientras lleguen
PLAY_DELAY = 0.6  # Hik-Connect manda el video en ráfagas de 1-4 s: un colchón para mostrarlo a su ritmo
LATE = 1.0  # la IA saltea un cuadro que se termina de analizar más tarde que esto (si ya hay otros esperando)
WARM = 1.5  # al conectar, Hik-Connect manda varios segundos viejos de golpe: en este rato se salta a lo más nuevo
AHEAD = 12.0  # un cuadro con más adelanto que esto: algo raro (el reloj del video saltó): se vuelve a anclar


class Pacer:
    """Muestra los cuadros al ritmo del video aunque lleguen en ráfagas.

    Por Hik-Connect el video llega de a 1-4 s de golpe. Cada cuadro trae su marca de tiempo del video (`MediaClock`);
    acá se le pone una hora para mostrarse (marca + ancla) y la ventana ve el último cuyo momento ya llegó. Es un
    colchón que se agranda solo: si un cuadro llega tarde (la red se atrasó), se vuelve a anclar más tarde y desde ahí
    sigue parejo. Al conectar, la ráfaga inicial (segundos viejos) se saltea para quedar cerca del vivo."""

    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self._anchor: float | None = None
        self._warm_until = 0.0
        self._q: deque = deque(maxlen=200)
        self._lock = threading.Lock()
        self._current: np.ndarray | None = None

    def restart(self, delay: float) -> None:
        """Al (re)conectar: las marcas de tiempo del video nuevo empiezan de otro lado."""
        with self._lock:
            self.delay, self._anchor = delay, None

    def due(self, ts: float, now: float) -> float:
        """Cuándo mostrar el cuadro con marca `ts` que llegó (o se leyó) en `now`."""
        with self._lock:
            if self._anchor is None:
                self._warm_until = now + WARM
            due = None if self._anchor is None else ts + self._anchor
            if (due is None or due < now or due > now + AHEAD
                    or (now < self._warm_until and due > now + self.delay)):
                self._anchor = now - ts + self.delay
                due = ts + self._anchor
            return due

    def put(self, frame: np.ndarray, due: float) -> None:
        with self._lock:
            self._q.append((due, frame))

    def get(self, now: float) -> np.ndarray | None:
        """El último cuadro cuyo momento ya llegó (los de antes se descartan: al volver a anclar quedan atrás)."""
        with self._lock:
            last = max((i for i, (due, _f) in enumerate(self._q) if due <= now), default=-1)
            for _ in range(last + 1):
                self._current = self._q.popleft()[1]
            return self._current


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
        self.pacer = Pacer()
        self._t0 = time.monotonic()
        self._last_ts = -1.0  # el reloj de la vista no vuelve atrás al reconectar (el seguimiento lo necesita)
        self._shown: list[float] = []
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self.run, name=f"ver-{self.id}", daemon=True)

    @property
    def display(self) -> np.ndarray | None:
        """El cuadro que toca mostrar ahora (lo pide la ventana)."""
        return self.pacer.get(time.monotonic())

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
        # Hik-Connect manda el video en ráfagas: los cuadros se eligen con el reloj del video (como en las máquinas y
        # en «Configurar…»), no con el de la PC; si no, de cada ráfaga quedaba uno solo (0,5 cuadros por segundo)
        clock = MediaClock(cap) if getattr(self.hik, "brand", "") == "hikconnect" else None
        self.pacer.restart(PLAY_DELAY if clock else 0.0)
        offset = max(self._last_ts + 0.2, time.monotonic() - self._t0)
        fails = 0
        while not self._stop.is_set():
            if not cap.grab():  # tomar todos los cuadros (no acumular retraso); convertir solo los que se usan
                fails += 1
                if fails >= 3:
                    return
                continue
            fails = 0
            ts = offset + clock.tick() if clock else time.monotonic() - self._t0
            if not self._wants(ts):
                continue
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                continue
            self._last_ts = ts
            if self._sar:
                frame = square_pixels(frame, self._sar, shrink=True)
            now = time.monotonic()
            self._frame(frame, ts, self.pacer.due(ts, now))
            self._shown = [t for t in self._shown if now - t < 5] + [now]
            self.fps = len(self._shown) / 5.0

    _picked = -1e9

    def _every(self) -> float:
        """Cada cuántos segundos de video se usa un cuadro: para mirar alcanzan 10 por segundo."""
        return SHOW_INTERVAL

    def _wants(self, ts: float) -> bool:
        if ts - self._picked < self._every() - 1e-3:
            return False
        self._picked = ts
        return True

    def _frame(self, frame: np.ndarray, ts: float, due: float) -> None:
        self.pacer.put(frame, due)


class AIViewWorker(ViewWorker):
    """Mirar una cámara con la IA aplicada en esta PC, solo para ver cómo la capta (ver el docstring del módulo).

    El video se lee en un hilo (sin retrasarse) y elige los cuadros por el reloj del video, a los cuadros por segundo
    de la cámara (como la máquina); la IA corre en otro hilo, en orden, y cada cuadro analizado se muestra a su ritmo
    (`Pacer`). Si la PC no da abasto, saltea los cuadros que ya quedaron viejos sin trabar el video.
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
        self.ai_ms = 0.0  # lo que tarda la IA por cuadro (promedio de los últimos)
        self.people = 0
        self._q: deque = deque(maxlen=max(4, int(self.target_fps * 4)))  # cuadros esperando a la IA (≤ 4 s)
        self._qlock = threading.Lock()
        self._new = threading.Event()
        self._hot: list = []  # carteles de prueba (reglas o gestos que saltaron), solo en pantalla
        self.ai_error: str | None = None
        self._summary_at: float | None = None
        self._done: list[float] = []
        self._took: deque = deque(maxlen=20)
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
        who = f"{self.people} persona{'s' if self.people != 1 else ''}"
        if self.ai_fps and self.ai_fps < 0.7 * self.target_fps and self.ai_ms > 1000.0 / self.target_fps:
            return f"IA · {who} · la PC no da abasto ({self.ai_ms:.0f} ms por cuadro)"
        return f"IA · {who} · {min(self.ai_fps, self.target_fps):.1f} de {self.target_fps:.0f} cuadros/s"

    def _ready(self) -> bool:
        return self.detector is not None and not self.ai_error

    def _every(self) -> float:
        # con la IA lista, los cuadros que analiza la máquina; mientras carga (o si falló), el video para mirar
        return 1.0 / self.target_fps if self._ready() else SHOW_INTERVAL

    def _frame(self, frame: np.ndarray, ts: float, due: float) -> None:
        if not self._ready():
            self.pacer.put(frame, due)
            return
        with self._qlock:
            self._q.append((frame, ts, due))
        self._new.set()

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
        while not self._stop.is_set():
            self._new.wait(1.0)
            self._new.clear()
            self._drain()

    def _drain(self) -> None:
        """Analizar, en orden, los cuadros que esperan; cada uno se muestra en su momento (`Pacer`)."""
        while not self._stop.is_set():
            with self._qlock:
                if not self._q:
                    return
                frame, ts, due = self._q.popleft()
                newer = len(self._q)
            if newer and due < time.monotonic() - LATE:  # quedó viejo (la PC no da abasto): al siguiente
                continue
            t0 = time.monotonic()
            try:
                out = self.process(frame, ts, datetime.now().astimezone())
            except Exception as exc:  # noqa: BLE001 - un cuadro raro no corta la vista
                log.debug("[%s] Ver con la IA: %s", self.name, exc)
                continue
            now = time.monotonic()
            self.pacer.put(out, due)
            self._took.append(now - t0)
            self.ai_ms = 1000.0 * sum(self._took) / len(self._took)
            self._done = [t for t in self._done if now - t < 5] + [now]
            self.ai_fps = len(self._done) / 5.0
            self._summary(now)

    def _summary(self, now: float) -> None:
        """Una línea en el registro a los 30 s: para saber si la PC da abasto y con qué corre."""
        if self._summary_at is None:
            self._summary_at = now + 30.0
        elif now >= self._summary_at > 0:
            self._summary_at = -1.0
            log.info("[%s] Ver con la IA: %.1f de %.0f cuadros/s analizados, %.0f ms por cuadro (detector en %s), "
                     "le llegan %.1f cuadros/s del video", self.name, self.ai_fps, self.target_fps, self.ai_ms,
                     getattr(self.detector, "device", "?"), self.fps)

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
