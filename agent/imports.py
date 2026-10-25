"""Análisis de videos importados (grabaciones exportadas de iVMS-4200) dentro del agente.

El servidor en Vercel no puede procesar video, así que los agentes de cada cliente toman
los videos que se suben desde el panel, los analizan con las zonas de la cámara y suben
los eventos para etiquetar.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import httpx

from core.analyzer import Analyzer, ClipReady, Finding
from core.clips import video_duration, write_mp4
from core.detectors import build_detector
from core.scorer import Scorer

from .outbox import ServerClient

log = logging.getLogger("vigia.imports")
IDLE_POLL_SECONDS = 120  # videos subidos para entrenar: no son urgentes y cada pregunta es un pedido al servidor


class ImportJobRunner:
    def __init__(self, client: ServerClient, model_path: str | None, work_dir: Path):
        self.client = client
        self.model_path = model_path
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, name="imports", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                worked = self.run_once()
            except httpx.HTTPError as exc:
                log.warning("No se pudo consultar videos pendientes: %s", exc)
                worked = False
            except Exception:
                log.exception("Error en el procesamiento de videos")
                worked = False
            if not worked:
                self._stop.wait(IDLE_POLL_SECONDS)

    def run_once(self) -> bool:
        """Procesa un video pendiente si hay. Devuelve True si procesó alguno."""
        job = self.client.next_job()
        if not job:
            return False
        iid = job["import_id"]
        log.info("Analizando video importado #%s (%s)", iid, job["filename"])
        tmp = Path(tempfile.mkdtemp(dir=self.work_dir))
        try:
            video = tmp / ("video" + Path(job["filename"]).suffix.lower())
            self.client.download(job["download_url"], video)
            duration = self._analyze(job, video, tmp)
            self.client.import_done(iid, duration)
            log.info("Video #%s listo", iid)
        except _Cancelled:
            log.info("El video #%s fue reasignado o cancelado", iid)
        except Exception as exc:
            log.exception("Falló el análisis del video #%s", iid)
            try:
                self.client.import_fail(iid, f"{type(exc).__name__}: {exc}")
            except httpx.HTTPError:
                pass
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return True

    def _analyze(self, job: dict, video: Path, tmp: Path) -> float | None:
        cam = job["camera"]
        iid = job["import_id"]
        detector = build_detector(cam.get("detector", "yolo"), self.model_path)
        analyzer = Analyzer(cam["analysis"], detector, Scorer.maybe(job.get("filter_model")),
                            clip_pre=6, clip_post=6, clip_width=960)
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError("No se pudo abrir el video. Si viene de iVMS-4200, exportalo en formato MP4.")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        fps = fps if 0 < fps < 120 else 25.0
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        step = max(1, int(round(fps / float(cam.get("process_fps") or 5))))
        start = None
        if job.get("recorded_at"):
            start = datetime.fromisoformat(job["recorded_at"]).replace(tzinfo=ZoneInfo(job["timezone"]))
        idx, last_report = 0, time.monotonic()
        try:
            while not self._stop.is_set():
                if not cap.grab():
                    break
                if idx % step == 0:
                    ok, frame = cap.retrieve()
                    if ok:
                        ts = idx / fps
                        wall = start + timedelta(seconds=ts) if start else None
                        for f in analyzer.process(frame, ts, wall):
                            self._send_finding(f, cam["id"], iid)
                        self._send_clips(analyzer.take_ready_clips(), tmp)
                idx += 1
                if time.monotonic() - last_report > 10 and total:
                    last_report = time.monotonic()
                    if not self.client.import_progress(iid, min(idx / total, 0.99)):
                        raise _Cancelled()
        finally:
            cap.release()
        self._send_clips(analyzer.flush(), tmp)
        return video_duration(video) or (idx / fps)

    def _send_finding(self, f: Finding, camera_id: int, iid: int) -> None:
        meta = {**f.meta(), "camera_id": camera_id, "source": "importado", "import_id": iid}
        if not f.wall_time:
            meta["occurred_at"] = None
        self.client.post_event(meta, f.snapshot_jpg)

    def _send_clips(self, ready: list[ClipReady], tmp: Path) -> None:
        for c in ready:
            path = tmp / f"{c.uuid}.mp4"
            if write_mp4(c.frames, path):
                self.client.put_clip(c.uuid, path, "video")
                path.unlink(missing_ok=True)


class _Cancelled(Exception):
    pass
