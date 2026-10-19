"""Alarmas de Hik-Partner Pro en la PC central.

Por cada alarma que el servidor recibió de HPP (movimiento, intrusión, cruce de línea):
1. baja del DVR el tramo de video [alarma - antes, alarma + después];
2. lo analiza entero (personas, zonas, reglas, modelo del cliente);
3. si no hay ninguna persona, la descarta (nadie la ve); si hay, sube un evento con foto y
   clip para que los empleados lo revisen y etiqueten.

Hoy el video se baja por VPN con la grabación del DVR (RTSP). Cuando Hikvision habilite la
clave VSaaS se puede agregar otra fuente (HPNetSDK) sin tocar el resto: ver `ClipFetcher`.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable

import httpx

from core.clip_analysis import analyze_clip
from core.detectors import Detector, build_detector
from core.scorer import Scorer

from .cameras import BaseCamera, build_camera, download_nvr_clip
from .outbox import ServerClient

log = logging.getLogger("vigia.alarms")
IDLE_POLL_SECONDS = 10

# (cámara, inicio, fin, archivo destino) -> True si dejó el clip
ClipFetcher = Callable[[BaseCamera, datetime, datetime, Path], bool]


def fetch_from_dvr(cam: BaseCamera, start: datetime, end: datetime, out: Path) -> bool:
    return download_nvr_clip(cam, start, end, out, timeout=180)


class AlarmRunner:
    def __init__(self, client: ServerClient, model_path: str | None, work_dir: Path, workers: int = 2,
                 fetch: ClipFetcher = fetch_from_dvr):
        self.client = client
        self.model_path = model_path
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.fetch = fetch
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._detectors: dict[str, Detector] = {}
        self.done = self.discarded = self.failed = 0
        self.threads = [threading.Thread(target=self._loop, name=f"alarmas-{i}", daemon=True) for i in range(workers)]

    def start(self) -> None:
        for t in self.threads:
            t.start()

    def stop(self) -> None:
        self._stop.set()

    def status(self) -> dict:
        return {"alarms_with_person": self.done, "alarms_discarded": self.discarded, "alarms_failed": self.failed}

    def _detector(self, name: str) -> Detector:
        if name != "yolo":  # el de movimiento guarda estado: uno nuevo por clip
            return build_detector(name, self.model_path)
        with self._lock:
            if name not in self._detectors:
                self._detectors[name] = build_detector(name, self.model_path)
            return self._detectors[name]

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                worked = self.run_once()
            except httpx.HTTPError as exc:
                log.warning("No se pudo consultar alarmas pendientes: %s", exc)
                worked = False
            except Exception:
                log.exception("Error procesando alarmas")
                worked = False
            if not worked:
                self._stop.wait(IDLE_POLL_SECONDS)

    def run_once(self) -> bool:
        """Procesa una alarma si hay. Devuelve True si procesó alguna."""
        job = self.client.next_alarm()
        if not job:
            return False
        aid, cam = job["alarm_id"], job["camera"]
        name = f"{cam.get('client_name', '')} · {cam['name']}"
        tmp = Path(tempfile.mkdtemp(dir=self.work_dir))
        try:
            hik = build_camera(cam["connection"])
            start, end = datetime.fromisoformat(job["start"]), datetime.fromisoformat(job["end"])
            clip = tmp / "alarma.mp4"
            if not self.fetch(hik, start, end, clip):
                self.failed += 1
                log.warning("[%s] alarma #%s: no se pudo bajar el video del DVR", name, aid)
                self.client.alarm_fail(aid, "No se pudo bajar el video del DVR (¿VPN o DVR caídos? ¿canal o "
                                            "clave mal cargados?)")
                return True
            res = analyze_clip(str(clip), cam.get("analysis", {}), self._detector(cam.get("detector", "yolo")),
                               Scorer.maybe(job.get("filter_model")), float(cam.get("process_fps") or 5), start)
            if not res.has_person:
                self.discarded += 1
                log.info("[%s] alarma #%s (%s): sin personas, descartada", name, aid, job.get("event_type"))
                self.client.alarm_done(aid, "sin_persona", res.stats())
                return True
            ev_uuid = uuid.uuid4().hex
            meta = {**res.meta(), "uuid": ev_uuid, "camera_id": cam["id"], "source": "alarma", "alarm_id": aid,
                    "clip_stats": res.stats()}
            if not meta["occurred_at"]:
                meta["occurred_at"] = start.isoformat()
            self.client.post_event(meta, res.snapshot_jpg)
            self.client.put_clip(ev_uuid, clip, "nvr")
            self.client.alarm_done(aid, "con_persona", res.stats(), ev_uuid)
            self.done += 1
            log.info("[%s] alarma #%s: %s (puntaje=%s)", name, aid, res.description, res.score)
        except httpx.HTTPError:
            raise
        except Exception as exc:
            self.failed += 1
            log.exception("[%s] falló la alarma #%s", name, aid)
            try:
                self.client.alarm_fail(aid, f"{type(exc).__name__}: {exc}")
            except httpx.HTTPError:
                pass
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return True
