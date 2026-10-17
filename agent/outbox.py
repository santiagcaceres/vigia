"""Cola en disco: los eventos se guardan primero localmente y se suben cuando hay conexión.

Cada evento es una carpeta:  outbox/<uuid>/meta.json, snapshot.jpg, clip.mp4, state.json
Si internet se corta, los eventos se acumulan y se envían al volver.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from pathlib import Path

import httpx

log = logging.getLogger("vigia.outbox")

CLIP_GIVE_UP_SECONDS = 15 * 60


class Outbox:
    def __init__(self, root: Path, client: "ServerClient"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.client = client
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- escritura (desde los workers de cámara) ----
    def add_event(self, camera_id: int, meta: dict, snapshot: bytes) -> Path:
        d = self.root / meta["uuid"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "snapshot.jpg").write_bytes(snapshot)
        _write_json(d / "meta.json", {**meta, "camera_id": camera_id, "queued_at": time.time()})
        self._wake.set()
        return d

    def clip_path(self, uuid: str) -> Path:
        return self.root / uuid / "clip.mp4"

    def clip_done(self, uuid: str, source: str | None) -> None:
        """source: 'nvr', 'buffer' o None si no hubo clip."""
        d = self.root / uuid
        if d.exists():
            _write_json(d / "clip.json", {"source": source})
            self._wake.set()

    # ---- envío ----
    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="outbox", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(10)

    def pending(self) -> int:
        return sum(1 for d in self.root.iterdir() if d.is_dir())

    def _loop(self) -> None:
        backoff = 2.0
        while not self._stop.is_set():
            try:
                self.flush_once()
                backoff = 2.0
                self._wake.wait(5.0)
            except httpx.HTTPError as exc:
                log.warning("Servidor no disponible (%s); reintento en %.0fs", exc, backoff)
                self._wake.wait(backoff)
                backoff = min(backoff * 2, 120.0)
            except Exception:
                log.exception("Error en la cola de envío")
                self._wake.wait(10.0)
            self._wake.clear()

    def flush_once(self) -> int:
        """Procesa toda la cola una vez. Devuelve cuántos eventos quedaron completos."""
        done = 0
        dirs = sorted((d for d in self.root.iterdir() if d.is_dir() and (d / "meta.json").exists()),
                      key=lambda d: d.stat().st_mtime)
        for d in dirs:
            state = _read_json(d / "state.json") or {}
            meta = _read_json(d / "meta.json")
            if not state.get("posted"):
                self.client.post_event(meta, (d / "snapshot.jpg").read_bytes())
                state["posted"] = True
                _write_json(d / "state.json", state)
            clip_info = _read_json(d / "clip.json")
            if clip_info is None:
                if time.time() - meta.get("queued_at", time.time()) < CLIP_GIVE_UP_SECONDS:
                    continue
                clip_info = {"source": None}
            clip = d / "clip.mp4"
            if clip_info.get("source") and clip.exists() and not state.get("clip_uploaded"):
                self.client.put_clip(meta["uuid"], clip, clip_info["source"])
                state["clip_uploaded"] = True
                _write_json(d / "state.json", state)
            shutil.rmtree(d, ignore_errors=True)
            done += 1
        return done


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False))
    tmp.replace(path)


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


class ServerClient:
    """Cliente HTTP del agente contra la API de Vigía."""

    def __init__(self, base_url: str, token: str, timeout: float = 30.0, verify: bool | str = True,
                 http: httpx.Client | None = None):
        self.http = http or httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout, verify=verify)
        self.http.headers["X-Agent-Token"] = token
        # Para subir y bajar archivos de Supabase: sin el token del agente.
        self.files = httpx.Client(timeout=httpx.Timeout(60.0, read=600.0, write=600.0), verify=verify,
                                  follow_redirects=True)

    def config(self) -> dict:
        r = self.http.get("/api/agent/config")
        r.raise_for_status()
        return r.json()

    def status(self, payload: dict) -> None:
        self.http.post("/api/agent/status", json=payload).raise_for_status()

    def _rejected(self, r: httpx.Response, what: str) -> bool:
        """True si el servidor rechazó el dato (no reintentar); lanza si es un error temporal."""
        if r.status_code >= 500 or r.status_code in (401, 429):
            r.raise_for_status()
        if r.status_code >= 400:
            log.error("El servidor rechazó %s: %s", what, r.text[:300])
            return True
        return False

    def post_event(self, meta: dict, snapshot: bytes) -> dict:
        r = self.http.post("/api/agent/events", data={"meta": json.dumps(meta, ensure_ascii=False)},
                           files={"snapshot": ("snapshot.jpg", snapshot, "image/jpeg")})
        if self._rejected(r, f"el evento {meta.get('uuid')}"):
            return {}
        return r.json()

    def put_clip(self, uuid: str, path: Path, source: str) -> None:
        r = self.http.post(f"/api/agent/events/{uuid}/clip-upload")
        if self._rejected(r, f"el clip {uuid}"):
            return
        info = r.json()
        self.upload(info["upload_url"], path, info.get("content_type", "video/mp4"))
        r = self.http.post(f"/api/agent/events/{uuid}/clip-done", json={"source": source})
        self._rejected(r, f"el clip {uuid}")

    def upload(self, url: str, path: Path, content_type: str) -> None:
        with open(path, "rb") as fh:
            if url.startswith("/"):  # almacenamiento local del servidor (desarrollo)
                r = self.http.put(url, content=fh.read(), headers={"Content-Type": content_type})
            else:
                r = self.files.put(url, content=fh, headers={"Content-Type": content_type, "x-upsert": "true"})
        r.raise_for_status()

    def download(self, url: str, dest: Path) -> None:
        client = self.http if url.startswith("/") else self.files
        with client.stream("GET", url) as r:
            r.raise_for_status()
            with open(dest, "wb") as out:
                for chunk in r.iter_bytes(1024 * 1024):
                    out.write(chunk)

    def preview(self, camera_id: int, jpg: bytes) -> None:
        self.http.post(f"/api/agent/cameras/{camera_id}/preview",
                       files={"image": ("preview.jpg", jpg, "image/jpeg")}).raise_for_status()

    # ---- videos importados ----
    def next_job(self) -> dict | None:
        r = self.http.get("/api/agent/jobs")
        r.raise_for_status()
        return None if r.status_code == 204 else r.json()

    def import_progress(self, iid: int, progress: float) -> bool:
        r = self.http.post(f"/api/agent/imports/{iid}/progress", json={"progress": progress})
        return r.status_code < 400

    def import_done(self, iid: int, duration_s: float | None) -> None:
        self.http.post(f"/api/agent/imports/{iid}/done", json={"duration_s": duration_s}).raise_for_status()

    def import_fail(self, iid: int, error: str) -> None:
        self.http.post(f"/api/agent/imports/{iid}/fail", json={"error": error[:500]})

    # ---- alarmas de Hik-Partner Pro ----
    def next_alarm(self) -> dict | None:
        r = self.http.get("/api/agent/alarms/next")
        r.raise_for_status()
        return None if r.status_code == 204 else r.json()

    def alarm_done(self, aid: int, result: str, stats: dict, event_uuid: str | None = None) -> None:
        r = self.http.post(f"/api/agent/alarms/{aid}/done",
                           json={"result": result, "stats": stats, "event_uuid": event_uuid})
        self._rejected(r, f"el resultado de la alarma {aid}")

    def alarm_fail(self, aid: int, error: str, retry: bool = True) -> None:
        self.http.post(f"/api/agent/alarms/{aid}/fail", json={"error": error[:500], "retry": retry})
