"""Los trabajos de «Entrenar objetos» en una máquina de análisis (18 nov; el dueño: «¿podemos subir los videos y fotos
desde la app y que los analicen las máquinas virtuales?»). Un hilo pregunta cada minuto si hay algo
(`/api/agent/objects/next`) y lo hace sin frenar el análisis de las cámaras:

- **video**: baja el video que se subió desde la app, saca una foto cada tantos segundos salteando las repetidas
  (core/objetos.video_frames), las sube al Storage y, si la máquina tiene Ultralytics, la IA propone recuadros.
- **premarcar**: la IA propone recuadros en las fotos sin marcar (la versión activa del objeto o YOLOE con las palabras en
  inglés de las clases).
- **entrenar**: arma la carpeta de fotos y etiquetas como la pide YOLO, entrena, mide con las fotos de prueba, exporta a
  ONNX y sube el modelo.

Lo que usa torch va en otro proceso (agent/objetos_ia.py): si se cae, se cancela o se queda sin memoria, las cámaras
siguen. Cancelar desde la app corta ese proceso en el próximo aviso de avance (cada 10 s como mucho).
Pre-marcar y entrenar solo en máquinas instaladas con ENTRENAR=si (Ultralytics); `VIGIA_ENTRENAR=no` lo apaga.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

import httpx

from core import objetos as ob

from .outbox import ServerClient

log = logging.getLogger("vigia.objetos")
IDLE_POLL_SECONDS = 60
PROGRESS_EVERY_S = 10.0
REPO = Path(__file__).resolve().parents[1]


class Cancelled(Exception):
    pass


def training_python() -> str | None:
    """El Python con Ultralytics: el entorno aparte que arma instalar.sh con ENTRENAR=si (VIGIA_PY_ENTRENAR) o, si no,
    este mismo si lo tiene. None = esta máquina no entrena."""
    if os.getenv("VIGIA_ENTRENAR", "").lower() in ("no", "0", "false"):
        return None
    py = os.getenv("VIGIA_PY_ENTRENAR")
    if py:
        return py if Path(py).exists() else None
    return sys.executable if importlib.util.find_spec("ultralytics") is not None else None


def capabilities() -> dict:
    """Qué trabajos de objetos puede hacer esta máquina (va en cada estado al servidor)."""
    ultra = training_python() is not None
    gpu, device = False, ""
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.run([smi, "-L"], capture_output=True, text=True, timeout=10).stdout
            first = next((ln for ln in out.splitlines() if ln.startswith("GPU ")), "")
            if first:
                gpu, device = True, first.split(":", 1)[1].split("(UUID")[0].strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return {"video": True, "ultra": ultra, "gpu": gpu, "device": device or ("procesador" if ultra else "")}


class ObjectJobRunner:
    def __init__(self, client: ServerClient, work_dir: Path, caps: dict | None = None, python: str | None = None):
        self.client = client
        self.work_dir = Path(work_dir).absolute()  # (el otro proceso corre parado ahí)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.caps = caps if caps is not None else capabilities()
        self.python = python or training_python() or sys.executable
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, name="objetos", daemon=True)
        self._last_progress = 0.0

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------ red
    def _get(self, path: str) -> dict | None:
        r = self.client.http.get(path)
        if r.status_code == 409:
            raise Cancelled()
        r.raise_for_status()
        return None if r.status_code == 204 else r.json()

    def _post(self, path: str, body: dict | None = None) -> dict:
        r = self.client.http.post(path, json=body or {})
        if r.status_code == 409:
            raise Cancelled()
        r.raise_for_status()
        return r.json()

    def progress(self, jid: int, value: float, message: str | None = None, force: bool = False) -> None:
        """Avisa cuánto va (cada 10 s como mucho) y corta si una persona lo canceló."""
        now = time.monotonic()
        if not force and now - self._last_progress < PROGRESS_EVERY_S:
            return
        self._last_progress = now
        try:
            resp = self._post(f"/api/agent/objects/jobs/{jid}/progress",
                              {"progress": max(0.0, min(1.0, value)), "message": (message or "")[:300] or None})
        except httpx.HTTPError as exc:
            log.warning("No se pudo avisar el avance del trabajo %s: %s", jid, exc)
            return
        if resp.get("cancel"):
            raise Cancelled()

    # ------------------------------------------------------------ vuelta
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                worked = self.run_once()
            except httpx.HTTPError as exc:
                log.warning("No se pudo preguntar por trabajos de objetos: %s", exc)
                worked = False
            except Exception:  # noqa: BLE001
                log.exception("Error en los trabajos de objetos")
                worked = False
            if not worked:
                self._stop.wait(IDLE_POLL_SECONDS)

    def run_once(self) -> bool:
        job = self._get("/api/agent/objects/next")
        if not job:
            return False
        jid, kind = job["job_id"], job["kind"]
        log.info("Trabajo de objetos %s: %s («%s»)", jid, ob.JOB_NAMES.get(kind, kind), job["model"]["name"])
        tmp = Path(tempfile.mkdtemp(prefix=f"obj{jid}-", dir=self.work_dir))
        self._last_progress = 0.0
        try:
            result = {"video": self._video, "premarcar": self._prelabel, "entrenar": self._train}[kind](job, tmp)
            self._post(f"/api/agent/objects/jobs/{jid}/done", {"result": result})
            log.info("Trabajo de objetos %s listo: %s", jid, {k: v for k, v in result.items() if k != "metrics"})
        except Cancelled:
            log.info("El trabajo de objetos %s se canceló o pasó a otra máquina", jid)
        except Exception as exc:  # noqa: BLE001
            log.exception("Falló el trabajo de objetos %s", jid)
            try:
                self._post(f"/api/agent/objects/jobs/{jid}/fail", {"error": str(exc)[:500] or type(exc).__name__})
            except (httpx.HTTPError, Cancelled):
                pass
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return True

    # ------------------------------------------------------------ fotos de un video
    def upload_photos(self, jid: int, shots: list[tuple[float, bytes, int, int]], name: str) -> list[tuple[int, bytes]]:
        """Registra y sube fotos (segundo, JPEG, ancho, alto). Devuelve (id, JPEG) de las nuevas."""
        body = {"images": []}
        by_uid = {}
        for ts, jpg, w, h in shots:
            uid = ob.image_uid(jpg)
            if uid in by_uid:
                continue
            by_uid[uid] = jpg
            body["images"].append({"uuid": uid, "name": name, "width": w, "height": h, "video_ts": ts})
        items = self._post(f"/api/agent/objects/jobs/{jid}/images", body)["items"]
        done = []
        for it in items:
            if it.get("upload_url"):
                self.client.upload_bytes(it["upload_url"], by_uid[it["uuid"]], "image/jpeg")
                done.append((it["id"], by_uid[it["uuid"]]))
        if done:
            self._post(f"/api/agent/objects/jobs/{jid}/images/uploaded", {"ids": [i for i, _ in done]})
        return done

    def _video(self, job: dict, tmp: Path) -> dict:
        jid, p = job["job_id"], job.get("params") or {}
        video = tmp / ("video" + Path(job.get("filename") or "v.mp4").suffix.lower())
        self.progress(jid, 0.02, "Bajando el video", force=True)
        self.client.download(job["download_url"], video)
        duration = _duration(video)
        kept: list[tuple[int, Path]] = []
        batch: list[tuple[float, bytes, int, int]] = []
        name = job.get("filename") or video.name
        total = 0

        def flush() -> None:
            for iid, jpg in self.upload_photos(jid, batch, name):
                path = tmp / f"{iid}.jpg"
                path.write_bytes(jpg)
                kept.append((iid, path))
            batch.clear()

        for ts, frame in ob.video_frames(video, float(p.get("every_s") or ob.VIDEO_EVERY_S),
                                         int(p.get("max_images") or ob.VIDEO_MAX_IMAGES),
                                         progress=lambda f: self.progress(jid, 0.05 + 0.6 * f,
                                                                          f"Sacando fotos ({len(kept) + len(batch)})"),
                                         stop=self._stop.is_set):
            img = ob.fit_photo(frame)
            batch.append((ts, ob.encode_photo(img), img.shape[1], img.shape[0]))
            total += 1
            if len(batch) >= 20:
                flush()
        if batch:
            flush()
        proposed = 0
        spec = job.get("prelabel")
        if kept and spec and self.caps.get("ultra"):
            proposed = self._propose(job, spec, kept, tmp, 0.7)
        return {"images": len(kept), "skipped": total - len(kept), "duration_s": duration, "proposed": proposed}

    # ------------------------------------------------------------ pre-marcar
    def _prelabel(self, job: dict, tmp: Path) -> dict:
        jid = job["job_id"]
        spec = job.get("prelabel")
        if not spec:
            raise RuntimeError("No hay con qué pre-marcar (ni versión activa ni palabras en inglés)")
        todo = self._get(f"/api/agent/objects/jobs/{jid}/todo")["items"]
        kept = []
        for n, it in enumerate(todo):
            path = tmp / f"{it['id']}.jpg"
            self.client.download(it["url"], path)
            kept.append((it["id"], path))
            self.progress(jid, 0.3 * (n + 1) / max(1, len(todo)), f"Bajando fotos ({n + 1} de {len(todo)})")
        proposed = self._propose(job, spec, kept, tmp, 0.3) if kept else 0
        return {"images": len(kept), "proposed": proposed}

    def _propose(self, job: dict, spec: dict, kept: list[tuple[int, Path]], tmp: Path, start: float) -> int:
        jid = job["job_id"]
        cfg = {"images": [{"id": i, "path": str(p)} for i, p in kept], "conf": ob.PROPOSE_CONF,
               "out": str(tmp / "propuestas.json"), "imgsz": ob.IMGSZ}
        if spec.get("method") == "version":
            pt = tmp / "version.pt"
            self.client.download(spec["pt_url"], pt)
            cfg.update(method="version", weights=str(pt), keys=spec["keys"])
            by = f"Versión {spec.get('number')}"
        else:
            cfg.update(method="yoloe", prompts=spec["prompts"], weights=str(self.work_dir / "yoloe-11s-seg.pt"))
            by = "IA de palabras (YOLOE)"
        self.run_ia("predict", cfg, tmp, jid, start, 0.95)
        found = json.loads(Path(cfg["out"]).read_text())
        items = [{"id": int(k), "boxes": v} for k, v in found.items()]
        n = 0
        for i in range(0, len(items), 200):
            n += self._post(f"/api/agent/objects/jobs/{jid}/proposals", {"by": by, "items": items[i:i + 200]})["ok"]
        return n

    # ------------------------------------------------------------ entrenar
    def _train(self, job: dict, tmp: Path) -> dict:
        jid, p = job["job_id"], job.get("params") or {}
        self.progress(jid, 0.01, "Bajando las fotos marcadas", force=True)
        ds = self._get(f"/api/agent/objects/jobs/{jid}/dataset")
        keys = [c["key"] for c in ds["classes"]]
        root = tmp / "datos"
        items = ds["items"]
        val = [it for it in items if it["split"] == "prueba"]
        if not val:  # sin fotos de prueba marcadas: se aparta una de cada cinco para medir algo
            val = sorted(items, key=lambda it: it["uuid"])[::5]
        val_ids = {it["id"] for it in val}
        n_boxes = 0
        for n, it in enumerate(items):
            split = "val" if it["id"] in val_ids else "train"
            (root / "images" / split).mkdir(parents=True, exist_ok=True)
            (root / "labels" / split).mkdir(parents=True, exist_ok=True)
            self.client.download(it["url"], root / "images" / split / f"{it['uuid']}.jpg")
            lines = ob.yolo_lines(it["boxes"], keys)
            n_boxes += len(lines)
            (root / "labels" / split / f"{it['uuid']}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            self.progress(jid, 0.05 * (n + 1) / max(1, len(items)), f"Bajando fotos ({n + 1} de {len(items)})")
        if not (root / "images" / "train").exists():  # todas quedaron de prueba (pocas fotos): se entrena con ellas
            shutil.copytree(root / "images" / "val", root / "images" / "train")
            shutil.copytree(root / "labels" / "val", root / "labels" / "train")
        data = root / "data.yaml"
        data.write_text(json.dumps({"path": str(root), "train": "images/train", "val": "images/val",
                                    "names": {i: k for i, k in enumerate(keys)}}))  # (JSON es YAML válido)
        base = job.get("base")
        if base and base.get("pt_url"):
            weights = tmp / "anterior.pt"
            self.client.download(base["pt_url"], weights)
        else:
            weights = self.work_dir / str(p.get("base") or "yolo11n.pt")  # Ultralytics lo baja la primera vez
        cfg = {"data": str(data), "out": str(tmp / "corrida"), "base": str(weights), "keys": keys,
               "epochs": int(p.get("epochs") or ob.auto_epochs(len(items))), "imgsz": int(p.get("imgsz") or ob.IMGSZ),
               "patience": int(p.get("patience") or 25), "test_images": len(val)}
        res = self.run_ia("train", cfg, tmp, jid, 0.06, 0.96)
        files = self._post(f"/api/agent/objects/jobs/{jid}/files")
        self.progress(jid, 0.97, "Subiendo el modelo", force=True)
        self.client.upload(files["onnx_url"], Path(res["onnx"]), "application/octet-stream")
        if res.get("pt") and Path(res["pt"]).exists():
            self.client.upload(files["pt_url"], Path(res["pt"]), "application/octet-stream")
        return {"metrics": res.get("metrics") or {}, "n_images": len(items), "n_boxes": n_boxes,
                "device": res.get("device"), "epochs_run": res.get("epochs_run"), "minutes": res.get("minutes")}

    # ------------------------------------------------------------ el otro proceso
    def run_ia(self, cmd: str, cfg: dict, tmp: Path, jid: int, lo: float, hi: float) -> dict:
        """Corre agent/objetos_ia.py y va pasando su avance al servidor (entre `lo` y `hi`). Devuelve lo que dice al
        terminar; si falla, el error con las últimas líneas que escribió."""
        path = tmp / f"{cmd}.json"
        path.write_text(json.dumps(cfg))
        env = {**os.environ, "PYTHONPATH": str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", ""),
               "YOLO_VERBOSE": "False", "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen([self.python, "-m", "agent.objetos_ia", cmd, str(path)], cwd=str(self.work_dir),
                                env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        tail: deque[str] = deque(maxlen=25)
        done, error = None, None
        try:
            for line in proc.stdout:  # type: ignore[union-attr]
                if not line.startswith("@@VIGIA "):
                    if line.strip():
                        tail.append(line.rstrip()[:300])
                    continue
                try:
                    msg = json.loads(line[8:])
                except ValueError:
                    continue
                if "done" in msg:
                    done = msg["done"]
                elif "error" in msg:
                    error = msg["error"]
                elif "progress" in msg:
                    self.progress(jid, lo + (hi - lo) * float(msg["progress"]), msg.get("message"))
                if self._stop.is_set():
                    raise Cancelled()
        except Cancelled:
            proc.terminate()
            try:
                proc.wait(30)
            except subprocess.TimeoutExpired:
                proc.kill()
            raise
        proc.wait()
        if done is None:
            detail = error or (tail[-1] if tail else f"terminó con código {proc.returncode}")
            log.error("Lo último que escribió Ultralytics:\n%s", "\n".join(tail))
            raise RuntimeError(f"La IA no pudo {'entrenar' if cmd == 'train' else 'pre-marcar'}: {detail}")
        return done


def _duration(video: Path) -> float | None:
    import cv2

    cap = cv2.VideoCapture(str(video))
    try:
        fps, n = cap.get(cv2.CAP_PROP_FPS) or 0, cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        return round(n / fps, 1) if fps > 0 and n > 0 else None
    finally:
        cap.release()
