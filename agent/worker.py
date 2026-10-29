from __future__ import annotations

import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import cv2

from core.analyzer import Analyzer, ClipReady, Finding
from core.clips import write_mp4
from core.detectors import Detector
from core.gestos import GESTURE_NAMES
from core.imagen import pixel_aspect, square_pixels
from core.metrics import MetricsCollector
from core.motion import MotionGate
from core.pose import build_pose
from core.rules import RuleEngine, parse_config
from core.scorer import Scorer

from .cameras import build_camera, download_nvr_clip
from .live import LiveStreamer, hud
from .outbox import Outbox, ServerClient

log = logging.getLogger("vigia.worker")
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;10000000|timeout;10000000")

PREVIEW_EVERY_SECONDS = 300
METRICS_EVERY_SECONDS = 300  # cada cuánto se mandan las métricas (gente, tiempos, vehículos) al servidor
# el NVR tarda unos segundos en tener disponible lo recién grabado
NVR_WRITE_DELAY_SECONDS = float(os.getenv("VIGIA_NVR_DELAY", "15"))


DISPLAY_INTERVAL = 0.1  # la ventana de la PC central no necesita más de 10 cuadros por segundo

_pose_lock = threading.Lock()
_pose_state: dict = {}  # "est": el estimador de postura del proceso (o None), una vez que se intentó armarlo


def shared_pose():
    """La postura de las personas (core/pose.py) para los gestos de ocultamiento: un solo estimador para todas las
    cámaras del proceso (y las alarmas), armado la primera vez que una cámara con gestos lo pide. Sin el modelo
    (models/postura.onnx) o si no arranca, None: esas cámaras siguen sin gestos, con todo lo demás igual, y queda
    una sola línea en el registro (no una por cámara ni por cada vez que se pide)."""
    with _pose_lock:
        if "est" not in _pose_state:
            try:
                _pose_state["est"] = build_pose()
            except Exception as exc:  # noqa: BLE001 - un modelo roto no puede dejar las cámaras sin analizar
                log.warning("No se pudo cargar la postura de las personas (%s): los gestos quedan apagados", exc)
                _pose_state["est"] = None
        return _pose_state["est"]


def wants_gestures(analysis: dict | None, metrics_only: bool = False) -> bool:
    """La cámara tiene algún gesto de ocultamiento activado (`analysis["gestures"]`, aparte de `rules`). Nunca en
    una cámara de solo métricas: ahí no hay alertas ni clips."""
    g = (analysis or {}).get("gestures")
    return not metrics_only and isinstance(g, dict) and bool(g.get("enabled"))


class MetricsQueue:
    """Junta las métricas de todas las cámaras de la PC y las manda en un solo pedido (lo llama el agente
    cada 5 min). Antes cada cámara hacía su pedido: con 600 cámaras eran 5 millones de llamadas por mes."""

    MAX_PER_REQUEST = 500  # lo que acepta el servidor por pedido

    def __init__(self, client: ServerClient | None):
        self.client = client
        self._lock = threading.Lock()
        self._items: dict[tuple[int, str], dict] = {}

    def add(self, buckets: list[dict], older: bool = False) -> None:
        with self._lock:
            for b in buckets:
                key = (b["camera_id"], b["bucket_start"])
                if older and key in self._items:
                    continue  # un reintento no pisa un cubo más nuevo de la misma hora
                self._items[key] = b

    def pending(self) -> int:
        return len(self._items)

    def flush(self) -> None:
        with self._lock:
            items, self._items = list(self._items.values()), {}
        if not items or self.client is None:
            return
        for i in range(0, len(items), self.MAX_PER_REQUEST):
            batch = items[i:i + self.MAX_PER_REQUEST]
            try:
                self.client.post_metrics(batch)
            except Exception as exc:  # noqa: BLE001 - se reintenta en el próximo envío
                log.warning("No se pudieron enviar las métricas (%s); se reintenta", exc)
                self.add(batch, older=True)


class MediaClock:
    """Segundos de video transcurridos según las marcas de tiempo del stream. Si el stream no las trae
    (o vuelven atrás), cuenta cuadros a los fps que declara. Nunca retrocede."""

    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        self.step = 1.0 / fps if 1.0 <= fps <= 60.0 else 1.0 / 12
        self.first: float | None = None
        self.t = 0.0

    def tick(self) -> float:
        pos = (self.cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
        if pos > 0:
            if self.first is None:
                self.first = pos - self.t
            cand = pos - self.first
            if self.t <= cand < self.t + 10:  # avanza de forma razonable: usar la marca del video
                self.t = cand
                return self.t
            self.first = pos - self.t - self.step  # salto o vuelta atrás (otro segmento): reanclar
        self.t += self.step
        return self.t


class CameraWorker:
    """Un hilo por cámara: lee el stream, analiza, guarda eventos y clips en la cola."""

    def __init__(self, cam_cfg: dict, detector: Detector, scorer: Scorer | None, outbox: Outbox,
                 client: ServerClient | None, clip_pool: ThreadPoolExecutor, settings: dict,
                 metrics_queue: MetricsQueue | None = None):
        self.id: int = cam_cfg["id"]
        # en modo central conviven cámaras de varios clientes: el log las distingue por cliente
        self.name: str = self.display_name(cam_cfg)
        self.fingerprint = self.fingerprint_of(cam_cfg)
        self.hik = build_camera(cam_cfg["connection"])
        # solo métricas (contar gente, tiempos): menos cuadros y sin clips; para hurto, mínimo 5 fps
        self.metrics_only = bool(cam_cfg.get("metrics_only"))
        self.process_fps = max(1.0 if self.metrics_only else 5.0, float(cam_cfg.get("process_fps") or 5))
        self.clip_pre = float(settings.get("clip_pre_seconds", 6))
        self.clip_post = float(settings.get("clip_post_seconds", 6))
        self.clip_max = float(settings.get("clip_max_seconds", 180))
        # la postura (gestos) se engancha en el hilo de la cámara (`_sync_pose`): cargar el modelo la primera vez
        # tarda un momento y no tiene que trabar a quien arma la cámara (la ventana)
        self._analysis: dict = cam_cfg.get("analysis") or {}
        self._sar_logged: float | None = None
        self.analyzer = Analyzer(self._analysis, detector, scorer, self.clip_pre, self.clip_post,
                                 clip_max=self.clip_max, clips=not self.metrics_only)
        # sin movimiento y sin nadie en escena no se usa el detector (ahorra placa de video)
        self.gate = MotionGate()
        self.idle_pct: float | None = None  # parte de los cuadros salteados por quietud (último minuto)
        self.metrics_queue = metrics_queue
        self.outbox = outbox
        self.client = client
        self.clip_pool = clip_pool
        self._findings: dict[str, Finding] = {}
        # reloj del análisis: sigue corriendo entre reconexiones. Si volviera a cero, el buffer de clips, los clips
        # abiertos y el seguimiento esperarían a que alcanzara al viejo, guardando cuadros mientras tanto (27 oct:
        # la máquina de Google llegó a 13,7 GB con 5 cámaras de Hik-Connect, que se reconectan seguido)
        self._last_ts: float | None = None
        self._last_ts_at = 0.0
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self.run, name=f"cam-{self.id}", daemon=True)
        self.state = "iniciando"
        self.error: str | None = None
        self.fps = 0.0
        self.events = 0
        self.last_frame_at: float | None = None
        # vista en vivo: sube cuadros con recuadros mientras alguien mira la cámara en el panel
        self.live = LiveStreamer(self.id, self.name, client, self.analyzer.zones)
        # métricas del local (personas, tiempos, destinos, vehículos): mismos tracks, sin costo extra
        self.metrics = MetricsCollector(self.analyzer.zones, self.analyzer.engine.cfg.schedule)
        self._metrics_pending: list[dict] = []
        self.analysis_fp = self.analysis_of(cam_cfg)
        self._pending_analysis: dict | None = None
        # ventana de la PC central: último cuadro con recuadros (se arma solo si la ventana está abierta)
        self.show_window = False
        self.display = None  # np.ndarray | None

    @staticmethod
    def display_name(cam_cfg: dict) -> str:
        return f"{cam_cfg['client_name']} · {cam_cfg['name']}" if cam_cfg.get("client_name") else cam_cfg["name"]

    def rename(self, cam_cfg: dict) -> None:
        """Le cambiaron el nombre a la cámara (o al cliente): sin cortar el video."""
        self.name = self.live.name = self.display_name(cam_cfg)

    @staticmethod
    def fingerprint_of(cam_cfg: dict) -> str:
        """Lo que obliga a reconectar: conexión, modo, detector… Las zonas y reglas NO (se aplican en
        caliente con `update_analysis`); solo cuenta si hay que detectar vehículos (cambia el detector).
        El nombre tampoco (`rename`)."""
        base = {k: v for k, v in cam_cfg.items() if k not in ("analysis", "name", "client_name")}
        base["vehicles"] = bool((cam_cfg.get("analysis") or {}).get("vehicles"))
        return json.dumps(base, sort_keys=True, default=str)

    @staticmethod
    def analysis_of(cam_cfg: dict) -> str:
        return json.dumps(cam_cfg.get("analysis") or {}, sort_keys=True, default=str)

    def update_analysis(self, analysis: dict) -> None:
        """Zonas, reglas o gestos nuevos: se aplican en el próximo cuadro, sin cortar el video."""
        self._pending_analysis = analysis

    def _sync_pose(self) -> None:
        """Con algún gesto activado el analizador mira la postura (el estimador compartido del proceso); sin gestos,
        no la mira (no gasta). Corre en el hilo de la cámara, al conectar y al cambiar el análisis."""
        want = wants_gestures(self._analysis, self.metrics_only)
        if want and self.analyzer.pose is not None:
            return  # ya la tiene (al reconectar no hace falta pedirla de nuevo)
        pose = shared_pose() if want else None
        if pose is self.analyzer.pose:
            return
        self.analyzer.pose = pose
        if pose is not None:
            log.info("[%s] gestos de ocultamiento activados: %s", self.name,
                     ", ".join(GESTURE_NAMES[g].lower() for g in self.analyzer.engine.cfg.gestures.enabled))
        else:  # (sin el modelo de postura no se llega acá: `shared_pose` ya lo dejó en el registro, una vez)
            log.info("[%s] gestos de ocultamiento apagados", self.name)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self._stop.set()

    def status(self) -> dict:
        return {"camera_id": self.id, "state": self.state, "error": self.error, "fps": round(self.fps, 2),
                "events": self.events, "live": self.live.active,
                "idle_pct": None if self.idle_pct is None else round(self.idle_pct * 100),
                "metrics_only": self.metrics_only, "gestures": self.analyzer.gestures, "last_frame_age_s":
                    None if self.last_frame_at is None else round(time.time() - self.last_frame_at, 1)}

    def run(self) -> None:
        backoff = 2.0
        while not self._stop.is_set():
            try:
                url = self.hik.live_url()  # en cada conexión: los links de Hik-Connect vencen
            except Exception as exc:  # noqa: BLE001 - sin link no hay video; se reintenta
                self.state, self.error = "sin_conexion", str(exc)
                log.warning("[%s] %s; reintento en %.0fs", self.name, self.error, backoff)
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 60.0)
                continue
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                self.state, self.error = "sin_conexion", f"No se pudo abrir {self.hik.redacted(url)}"
                why = self.hik.open_failure(url)
                if why:
                    self.error += f" — {why}"
                log.warning("[%s] %s; reintento en %.0fs", self.name, self.error, backoff)
                cap.release()
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 60.0)
                continue
            self.state, self.error = "en_linea", None
            log.info("[%s] conectada", self.name)
            connected_at = time.monotonic()
            try:
                self._read_loop(cap)
            except Exception as exc:
                log.exception("[%s] error procesando", self.name)
                self.state, self.error = "error", str(exc)
                self._stop.wait(5.0)
            finally:
                cap.release()
            if not self._stop.is_set():
                self.state = "reconectando"
                # si el video se corta enseguida una y otra vez, esperar cada vez más (no martillar al equipo
                # ni a Hik-Connect pidiendo links); si anduvo un rato, reconectar rápido
                backoff = 2.0 if time.monotonic() - connected_at > 30 else min(backoff * 2, 60.0)
                self._stop.wait(backoff)
        self._finish_clips(self.analyzer.flush())
        self._send_metrics(self.metrics.close())
        self.state = "detenida"

    def _read_loop(self, cap: cv2.VideoCapture) -> None:
        self._sync_pose()
        t0 = time.monotonic()
        # sigue desde donde quedó la conexión anterior, más lo que duró el corte
        base = 0.0 if self._last_ts is None else self._last_ts + max(t0 - self._last_ts_at, 0.1)
        # Hik-Connect (HLS) manda el video en ráfagas de 1-4 s: con el reloj de la PC se analizaría un solo
        # cuadro por ráfaga. Ahí se usa el reloj del video (marca de tiempo de cada cuadro).
        clock = MediaClock(cap) if self.hik.brand == "hikconnect" else None
        # «1080p lite» (960x1080 con píxeles 2:1): se analiza con píxeles cuadrados (960x540), igual que en
        # «Configurar…» y en las alarmas; si no, las personas quedan flacas y los gestos no andan
        sar = pixel_aspect(cap)
        if sar and sar != self._sar_logged:
            self._sar_logged = sar
            log.info("[%s] el video tiene píxeles de %.2f:1: se analiza con píxeles cuadrados", self.name, sar)
        last_proc = -1e9
        last_show = -1e9
        last_preview = -1e9
        last_metrics = last_idle = time.monotonic()
        window: list[float] = []
        fails = 0
        while not self._stop.is_set():
            if self._pending_analysis is not None:
                # se saca antes de aplicarlo: si falla, no se reintenta en cada cuadro
                pending, self._pending_analysis = self._pending_analysis, None
                self._apply_analysis(pending)
            # tomar todos los cuadros evita acumular retraso; solo se convierten a imagen los que se usan
            # (analizar o mostrar en la ventana), que es lo que más procesador gasta
            ok = cap.grab()
            if not ok:
                fails += 1
                if fails >= 3:
                    self.error = "Se perdió la señal"
                    return
                continue
            fails = 0
            now = time.monotonic()
            ts = base + (clock.tick() if clock else now - t0)  # segundos del video (o de la PC)
            self._last_ts, self._last_ts_at = ts, now
            self.last_frame_at = time.time()
            analyze = ts - last_proc >= 1.0 / self.process_fps - 1e-3
            show = self.show_window and now - last_show >= DISPLAY_INTERVAL
            if not (analyze or show):
                continue
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                continue
            if sar:
                frame = square_pixels(frame, sar, shrink=True)
            if show:  # la ventana se dibuja a 10 cuadros por segundo como máximo
                last_show = now
                self.display = hud(frame, self.analyzer.zones, self.analyzer.last_tracks, ts, self.name,
                                   datetime.now(self.hik.tz), self._gesture_engine())
            if not analyze:
                continue
            last_proc = ts
            wall = datetime.now(self.hik.tz)
            if self.gate.should_analyze(frame, ts, self.analyzer.busy):
                findings = self.analyzer.process(frame, ts, wall)
                self.metrics.update(self.analyzer.last_tracks, ts, frame.shape[1::-1], wall)
            else:  # escena quieta y vacía: no hace falta buscar personas
                self.analyzer.idle(frame, ts)
                findings = []
            if now - last_idle > 60:
                last_idle = now
                self.idle_pct = self.gate.take_saved_fraction()
            if self.live.active:  # (la imagen en vivo la puede ver el cliente: sin la marcha de los gestos)
                self.live.push(frame, self.analyzer.last_tracks, ts, wall)
            for f in findings:
                self._on_finding(f)
            if now - last_metrics > METRICS_EVERY_SECONDS:
                last_metrics = now
                self._send_metrics(self.metrics.flush())
            self._finish_clips(self.analyzer.take_ready_clips())
            window = [t for t in window if now - t < 10] + [now]
            self.fps = len(window) / 10.0
            if self.client and now - last_preview > PREVIEW_EVERY_SECONDS:
                last_preview = now
                ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    self.clip_pool.submit(self._send_preview, jpg.tobytes())

    def _on_finding(self, f: Finding) -> None:
        self.events += 1
        log.info("[%s] %s (puntaje=%s, alerta=%s)", self.name, f.description, f.score, f.alert)
        self.outbox.add_event(self.id, f.meta(), f.snapshot_jpg)
        self._findings[f.uuid] = f

    def _finish_clips(self, ready: list[ClipReady]) -> None:
        for c in ready:
            f = self._findings.pop(c.uuid, None)
            self.clip_pool.submit(self._make_clip, c, f)

    def _make_clip(self, c: ClipReady, f: Finding | None) -> None:
        out = self.outbox.clip_path(c.uuid)
        try:
            if self.hik.nvr_clips and f is not None and f.wall_time is not None and not self.hik.source_url:
                # el mismo tramo que cubre el clip del buffer: desde que apareció la persona hasta que se fue
                start = f.wall_time - timedelta(seconds=max(0.0, f.ts - c.start_ts))
                end = f.wall_time + timedelta(seconds=max(self.clip_post, c.end_ts - f.ts))
                wait = (end - datetime.now(self.hik.tz)).total_seconds() + NVR_WRITE_DELAY_SECONDS
                if wait > 0:
                    self._stop.wait(wait)
                if download_nvr_clip(self.hik, start, end, out):
                    self.outbox.clip_done(c.uuid, "nvr")
                    return
            if write_mp4(c.frames, out):
                self.outbox.clip_done(c.uuid, "buffer")
            else:
                self.outbox.clip_done(c.uuid, None)
        except Exception:
            log.exception("[%s] no se pudo generar el clip %s", self.name, c.uuid)
            self.outbox.clip_done(c.uuid, None)

    def _gesture_engine(self):
        """El motor de gestos, si la cámara los está mirando (para escribir el gesto en curso sobre la imagen)."""
        return self.analyzer.engine.gestures if self.analyzer.gestures else None

    def _apply_analysis(self, analysis: dict) -> None:
        try:
            zones, rule_cfg = parse_config(analysis)
        except Exception as exc:  # noqa: BLE001 - (un número en null, una clave rara): seguir con lo anterior
            log.warning("[%s] zonas, reglas o gestos inválidos, sigo con los anteriores: %s", self.name, exc)
            return
        self.analyzer.zones = zones
        # motor nuevo: los tiempos de las reglas y de los gestos rigen desde este cuadro
        self.analyzer.engine = RuleEngine(zones, rule_cfg)
        self._analysis = analysis
        self._sync_pose()  # se prendieron o apagaron los gestos: enganchar o soltar la postura
        self._send_metrics(self.metrics.close())  # lo medido hasta ahora queda con las zonas viejas
        self.metrics = MetricsCollector(zones, rule_cfg.schedule)
        self.live.zones = zones
        log.info("[%s] zonas, reglas y gestos actualizados (%d zonas)", self.name, len(zones))

    def _send_metrics(self, buckets: list[dict]) -> None:
        """Deja los cubos en la cola de la PC (un solo envío para todas las cámaras). Sin cola (pruebas,
        uso suelto) los manda directo en otro hilo; si falla, se reintentan con el próximo envío."""
        if self.metrics_queue is not None:
            self.metrics_queue.add([{"camera_id": self.id, **b} for b in buckets])
            return
        pending = self._metrics_pending + [{"camera_id": self.id, **b} for b in buckets]
        self._metrics_pending = []
        if not pending or self.client is None:
            return
        latest: dict[str, dict] = {}
        for b in pending:  # si un cubo quedó repetido por reintentos, vale el último
            latest[b["bucket_start"]] = b
        batch = list(latest.values())[-200:]

        def send() -> None:
            try:
                self.client.post_metrics(batch)  # type: ignore[union-attr]
            except Exception as exc:
                log.warning("[%s] no se pudieron enviar las métricas (%s); se reintenta", self.name, exc)
                self._metrics_pending = batch + self._metrics_pending
        self.clip_pool.submit(send)

    def _send_preview(self, jpg: bytes) -> None:
        try:
            self.client.preview(self.id, jpg)  # type: ignore[union-attr]
        except Exception as exc:
            log.debug("No se pudo enviar la vista previa: %s", exc)
