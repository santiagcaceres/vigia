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
from core.scorer import Scorer

from .cameras import build_camera, download_nvr_clip
from .live import LiveStreamer
from .outbox import Outbox, ServerClient

log = logging.getLogger("vigia.worker")
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;10000000|timeout;10000000")

PREVIEW_EVERY_SECONDS = 300
# el NVR tarda unos segundos en tener disponible lo recién grabado
NVR_WRITE_DELAY_SECONDS = float(os.getenv("VIGIA_NVR_DELAY", "15"))


class CameraWorker:
    """Un hilo por cámara: lee el stream, analiza, guarda eventos y clips en la cola."""

    def __init__(self, cam_cfg: dict, detector: Detector, scorer: Scorer | None, outbox: Outbox,
                 client: ServerClient | None, clip_pool: ThreadPoolExecutor, settings: dict):
        self.id: int = cam_cfg["id"]
        # en modo central conviven cámaras de varios clientes: el log las distingue por cliente
        self.name: str = (f"{cam_cfg['client_name']} · {cam_cfg['name']}" if cam_cfg.get("client_name")
                          else cam_cfg["name"])
        self.fingerprint = self.fingerprint_of(cam_cfg)
        self.hik = build_camera(cam_cfg["connection"])
        self.process_fps = float(cam_cfg.get("process_fps", 5))
        self.clip_pre = float(settings.get("clip_pre_seconds", 6))
        self.clip_post = float(settings.get("clip_post_seconds", 6))
        self.analyzer = Analyzer(cam_cfg.get("analysis", {}), detector, scorer, self.clip_pre, self.clip_post)
        self.outbox = outbox
        self.client = client
        self.clip_pool = clip_pool
        self._findings: dict[str, Finding] = {}
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self.run, name=f"cam-{self.id}", daemon=True)
        self.state = "iniciando"
        self.error: str | None = None
        self.fps = 0.0
        self.events = 0
        self.last_frame_at: float | None = None
        # vista en vivo: sube cuadros con recuadros mientras alguien mira la cámara en el panel
        self.live = LiveStreamer(self.id, self.name, client, self.analyzer.zones)

    @staticmethod
    def fingerprint_of(cam_cfg: dict) -> str:
        return json.dumps(cam_cfg, sort_keys=True, default=str)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self._stop.set()

    def status(self) -> dict:
        return {"camera_id": self.id, "state": self.state, "error": self.error, "fps": round(self.fps, 2),
                "events": self.events, "live": self.live.active, "last_frame_age_s":
                    None if self.last_frame_at is None else round(time.time() - self.last_frame_at, 1)}

    def run(self) -> None:
        url = self.hik.live_url()
        backoff = 2.0
        while not self._stop.is_set():
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                self.state, self.error = "sin_conexion", f"No se pudo abrir {self.hik.redacted(url)}"
                log.warning("[%s] %s; reintento en %.0fs", self.name, self.error, backoff)
                cap.release()
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 60.0)
                continue
            self.state, self.error, backoff = "en_linea", None, 2.0
            log.info("[%s] conectada", self.name)
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
                self._stop.wait(2.0)
        self._finish_clips(self.analyzer.flush())
        self.state = "detenida"

    def _read_loop(self, cap: cv2.VideoCapture) -> None:
        t0 = time.monotonic()
        last_proc = -1e9
        last_preview = -1e9
        window: list[float] = []
        fails = 0
        while not self._stop.is_set():
            ok, frame = cap.read()  # leer todos los frames evita acumular retraso en el stream
            if not ok:
                fails += 1
                if fails >= 3:
                    self.error = "Se perdió la señal"
                    return
                continue
            fails = 0
            now = time.monotonic()
            self.last_frame_at = time.time()
            if now - last_proc < 1.0 / self.process_fps - 1e-3:
                continue
            last_proc = now
            wall = datetime.now(self.hik.tz)
            findings = self.analyzer.process(frame, now - t0, wall)
            if self.live.active:
                self.live.push(frame, self.analyzer.last_tracks, now - t0, wall)
            for f in findings:
                self._on_finding(f)
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
                start = f.wall_time - timedelta(seconds=self.clip_pre)
                end = f.wall_time + timedelta(seconds=self.clip_post)
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

    def _send_preview(self, jpg: bytes) -> None:
        try:
            self.client.preview(self.id, jpg)  # type: ignore[union-attr]
        except Exception as exc:
            log.debug("No se pudo enviar la vista previa: %s", exc)
