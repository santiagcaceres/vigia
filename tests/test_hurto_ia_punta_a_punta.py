"""La IA de movimientos de hurto de punta a punta (pedido del dueño, 9 nov), con los módulos de verdad de cada parte y
solo dos dobles: el detector de personas (la persona es lo rojo del video) y la postura (el esqueleto sale de una marca
de color en una esquina del video sintético, que dice en qué parte del movimiento está: así el mismo video «muestra»
el movimiento a la app y a la máquina). Nada de video real (el repo es público).

1. La app (pestaña «Entrenar IA hurtos»): `agent.ia_clips.analyze` analiza cada clip en la PC (seguimiento, postura,
   registros de cuadro), `core.hurto_ia.person_windows` arma las ventanas y `EntrenarIA._upload_work` las sube
   comprimidas con gzip por `agent.oficina.Api` al servidor de verdad; el .json.gz de los cuadros va al almacenamiento.
2. Marcar: el clic en la persona (`entrenar_ia.person_at` sobre el registro del momento) y el `PUT` con inicio, fin,
   persona y cómo; los clips normales, «normal».
3. Entrenar (`POST /api/training/ia/run`) y activar el modelo.
4. Una máquina de análisis (dada de alta con la clave de máquinas) pide `/api/agent/config`: la cámara que mide hurtos
   trae `ia_hurto`; la de solo métricas, no.
5. `agent.worker.CameraWorker` (con su `core.analyzer.Analyzer`) mira en vivo a alguien que hace el mismo movimiento
   → evento `hurto_ia`; la cola (`agent.outbox.Outbox` + `ServerClient`) lo manda a `/api/agent/events` y el servidor
   lo guarda con sus rasgos. A alguien que no hace nada raro no le salta."""

from __future__ import annotations

import gzip
import json
import queue
import threading
import types
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import agent.worker as worker_mod
from agent import entrenar_ia as ei
from agent import ia_clips
from agent.oficina import Api
from agent.outbox import Outbox, ServerClient
from core import hurto_ia
from core.detectors import Detection
from core.rules import HURTO_IA
from core.scorer import Scorer
from server.main import Settings, create_app

ADMIN = ("admin", "clave-admin-123")
W, H, VIDEO_FPS = 640, 360, 10.0
PERSON = (270, 30, 373, 330)  # la persona (roja), quieta frente al estante: 300 px de alto
SHELF = {"name": "Góndola", "type": "shelf", "polygon": [[0.0, 0.7], [1.0, 0.7], [1.0, 1.0], [0.0, 1.0]]}
# la marca de la esquina: en qué parte del movimiento está la persona (BGR)
PHASE_COLORS = {"alcanza": (0, 255, 0), "vuelve": (0, 255, 255), "bolsillo": (255, 0, 255)}
SECONDS = 20.0


def phase(t: float, t_hurto: float | None) -> str | None:
    """El hurto: estira el brazo al estante 1 s, lo trae (0,4 s) y se lleva la mano al bolsillo (2,6 s)."""
    if t_hurto is None:
        return None
    rel = t - t_hurto
    return "alcanza" if 0 <= rel < 1 else "vuelve" if 1 <= rel < 1.4 else "bolsillo" if 1.4 <= rel < 4 else None


def draw(t: float, t_hurto: float | None, rng: np.random.Generator) -> np.ndarray:
    img = np.full((H, W, 3), (90, 110, 120), np.uint8)
    cv2.rectangle(img, (0, 40), (W, 110), (40, 160, 200), -1)  # estante
    x1, y1, x2, y2 = PERSON
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), -1)
    p = phase(t, t_hurto)
    if p:
        cv2.rectangle(img, (0, 0), (40, 40), PHASE_COLORS[p], -1)
    # ruido gris abajo (que no parezca rojo): la escena «se mueve» y el filtro de movimiento no saltea cuadros
    img[H - 20:, :] = rng.integers(60, 200, (20, W, 1), dtype=np.uint8)
    return img


def make_video(path: Path, t_hurto: float | None, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, (W, H))
    assert out.isOpened()
    for i in range(int(SECONDS * VIDEO_FPS)):
        out.write(draw(i / VIDEO_FPS, t_hurto, rng))
    out.release()
    return path


class RedPerson:
    """Detector doble: la persona es la mancha roja más grande (sin bolsos)."""

    def detect(self, frame, conf=None, far=None):
        b, g, r = (frame[..., i].astype(np.int16) for i in range(3))
        mask = ((r > 180) & (g < 80) & (b < 80)).astype(np.uint8)
        n, _lab, stats, _c = cv2.connectedComponentsWithStats(mask)
        if n < 2:
            return []
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, w, h, area = stats[i]
        if area < 500:
            return []
        return [Detection((float(x), float(y), float(x + w), float(y + h)), "persona", 0.9)]

    def detect_also(self, frame, classes, conf, person_conf=None, far=None):
        return self.detect(frame), []


# el esqueleto de frente dentro de un recuadro de 120x350 en (40, 50) (como tests/test_hurto_ia_servidor.esqueletos)
BASE_BOX = (40.0, 50.0, 160.0, 400.0)
BASE = {0: (100, 70), 1: (95, 65), 2: (105, 65), 5: (120, 100), 6: (80, 100), 11: (115, 200), 12: (85, 200),
        13: (115, 280), 14: (85, 280), 15: (115, 360), 16: (85, 360), 7: (128, 150), 8: (72, 150),
        9: (130, 205), 10: (70, 205)}
MOVES = {"alcanza": {10: (-10, 120), 8: (30, 112)}, "vuelve": {10: (95, 150), 8: (70, 140)},
         "bolsillo": {10: (83, 205), 8: (70, 160)}}


class PhasePose:
    """Postura doble: lee la marca de la esquina del cuadro y devuelve el esqueleto de esa parte del movimiento,
    llevado al recuadro de la persona (con un poco de ruido, como una postura de verdad)."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.calls = 0

    @staticmethod
    def which(frame) -> str | None:
        b, g, r = (int(v) for v in frame[20, 20])
        for name, (cb, cg, cr) in PHASE_COLORS.items():
            if abs(b - cb) < 70 and abs(g - cg) < 70 and abs(r - cr) < 70:
                return name
        return None

    def estimate(self, frame, bbox):
        self.calls += 1
        pts = {**BASE, **MOVES.get(self.which(frame), {})}
        bx1, by1, bx2, by2 = BASE_BOX
        x1, y1, x2, y2 = bbox
        sx, sy = (x2 - x1) / (bx2 - bx1), (y2 - y1) / (by2 - by1)
        kp = np.zeros((17, 3), np.float32)
        for j, (x, y) in pts.items():
            nx, ny = self.rng.normal(0, 2, 2)
            kp[j] = (x1 + (x + nx - bx1) * sx, y1 + (y + ny - by1) * sy, 0.9)
        return kp


class FakeTime:
    def __init__(self):
        self.t = 5000.0

    def monotonic(self):
        return self.t

    def time(self):
        return self.t


class LiveCap:
    """La cámara en vivo: el mismo dibujo a 25 cuadros por segundo (con el reloj de la prueba) y después se corta."""

    def __init__(self, clock: FakeTime, t_hurto: float | None, seconds: float, seed: int):
        self.clock, self.t_hurto, self.left = clock, t_hurto, int(seconds * 25)
        self.start = clock.t
        self.rng = np.random.default_rng(seed)

    def grab(self):
        self.clock.t += 0.04
        self.left -= 1
        return self.left >= 0

    def retrieve(self):
        return True, draw(self.clock.t - self.start, self.t_hurto, self.rng)

    def get(self, prop):
        return 0.0

    def release(self):
        pass


class FakePool:
    """En lugar del grupo de hilos de los clips (los clips no se arman en esta prueba: el evento igual se manda)."""

    def __init__(self):
        self.jobs = []

    def submit(self, fn, *args):
        self.jobs.append((fn, args))


def test_de_punta_a_punta(tmp_path, db_url, monkeypatch):
    app = create_app(Settings(data_dir=tmp_path / "data", database_url=db_url, admin_password=ADMIN[1],
                              serve_public=False))
    with TestClient(app) as admin:
        assert admin.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]}).status_code == 200
        cid = admin.post("/api/clients", json={"name": "Almacén Demo", "plans": ["hurto"]}).json()["id"]
        cam_id = admin.post("/api/cameras", json={"client_id": cid, "name": "Pasillo", "brand": "ip", "host": "dvr",
                                                  "password": "x", "profile": "gondola"}).json()["id"]
        # zonas, y sin gestos: lo que avise acá es de la IA de movimientos
        r = admin.put(f"/api/cameras/{cam_id}/analysis", json={"zones": [SHELF], "rules": {},
                                                               "gestures": {"enabled": []}})
        assert r.status_code == 200, r.text
        conteo = admin.post("/api/cameras", json={"client_id": cid, "name": "Depósito", "brand": "ip",
                                                  "host": "dvr", "password": "x", "profile": "metricas"}).json()["id"]

        # ---- 1. la app sube los clips analizados en la PC (comprimidos), con el detector y la postura dobles
        api = Api("http://testserver")
        api.http = admin  # (la sesión de la prueba)
        api.user = {"username": "admin", "role": "admin"}
        cam = next(c for c in api.get("/api/cameras") if c["id"] == cam_id)
        assert cam["zones"][0]["type"] == "shelf"
        sent = []
        real_call = api.call

        def call(method, path, **kw):
            if (method, path) == ("POST", "/api/training/clips"):
                sent.append(kw)
            return real_call(method, path, **kw)
        api.call = call
        tab = ei.EntrenarIA.__new__(ei.EntrenarIA)  # (solo la subida, sin ventana)
        tab.app = types.SimpleNamespace(api=api, agent_of=None)
        tab.analyze_fn, tab.windows_fn = ia_clips.analyze, None
        pose = PhasePose(1)
        tab.ai_setup = lambda: (RedPerson(), pose)
        tab.data_dir, tab._local_obj, tab._q = tmp_path / "pc", None, queue.Queue()
        rng = np.random.default_rng(7)
        thefts = {}
        for i in range(6):
            t_hurto = round(float(rng.uniform(5, 12)), 1)
            thefts[str(make_video(tmp_path / f"hurto {i}.mp4", t_hurto, seed=i))] = t_hurto
        normals = [str(make_video(tmp_path / f"normal {i}.mp4", None, seed=100 + i)) for i in range(4)]

        def upload(paths, verdict):
            tab._upload_work(paths, cid, cam, verdict, {}, threading.Event())
            msgs = [tab._q.get() for _ in range(tab._q.qsize())]
            done = [m for m in msgs if m[0] == "done"]
            assert len(done) == len(paths) and all(m[3] == "" for m in done), msgs
            return [m[1] for m in done]
        theft_clips = upload(list(thefts), "sin_marcar")
        upload(normals, "normal")
        assert pose.calls > 0
        assert all(kw["headers"]["Content-Encoding"] == "gzip" for kw in sent) and len(sent) == 10
        body = json.loads(gzip.decompress(sent[0]["content"]))
        assert body["feature_version"] == hurto_ia.FEATURE_VERSION and body["windows"]
        assert all(len(w["f"]) == len(hurto_ia.FEATURE_NAMES) for w in body["windows"])
        listed = {c["uuid"]: c for c in api.get("/api/training/clips", client_id=cid)}
        assert len(listed) == 10 and all(c["has_data"] and c["camera_name"] == "Pasillo" for c in listed.values())

        # ---- 2. marcar cada hurto: inicio, fin, clic en la persona y cómo
        for clip, (path, t_hurto) in zip(theft_clips, thefts.items()):
            frames = tab.local.load_data(clip["uuid"])["frames"]
            assert frames == ia_clips.unpack(api.fetch(api.get(f"/api/training/clips/{clip['id']}/data")["url"]))[
                "frames"]  # (lo que ve otra PC)
            rec = frames[ei.nearest([f["t"] for f in frames], t_hurto + 1)]
            x1, y1, x2, y2 = PERSON
            who = ei.person_at(rec, (x1 + x2) / 2, (y1 + y2) / 2)
            assert who is not None
            res = api.put(f"/api/training/clips/{clip['id']}",
                          {"verdict": "hurto", "marks": [{"start": t_hurto, "end": round(t_hurto + 4, 1),
                                                          "track_id": who, "kind": "bolsillo"}]})
            assert res["verdict"] == "hurto" and res["marks_without_windows"] == [] and res["positive_windows"] > 0
        summary = api.get("/api/training/ia/summary", client_id=cid)
        assert (summary["theft_clips"], summary["marks"], summary["normal_clips"]) == (6, 6, 4)

        # ---- 3. entrenar y activar
        model = api.post("/api/training/ia/run", {"client_id": cid, "target_recall": ei.TARGET_RECALL})
        met = model["metrics"]
        assert model["usable"] and not model["active"] and met["marks"] == 6 and met["clips"] == 10
        assert met["marks_seen"] >= 5 and met["auc"] > 0.9 and met["validation"] == "por_clip", met
        assert api.post(f"/api/training/ia/models/{model['id']}/activate")["active"] is True

        # ---- 4. una máquina de análisis recibe la IA en la cámara que mide hurtos
        key = admin.post("/api/analysis/pool-key").json()["key"]
        tok = TestClient(app).post("/api/agent/enroll", json={"key": key, "name": "m1", "instance_id": "i-1"}).json()
        client = ServerClient("http://testserver", tok["token"], http=TestClient(app))
        client.status({"cameras": [], "instance_id": "i-1", "fresh": True})
        assert admin.post("/api/cameras/analyze", json={"camera_ids": [cam_id, conteo], "analyze": True}
                          ).status_code == 200
        cfg = client.config()
        cams = {c["id"]: c for c in cfg["cameras"]}
        assert set(cams) == {cam_id, conteo}
        spec = cams[cam_id]["analysis"]["ia_hurto"]
        assert spec["id"] == model["id"] and hurto_ia.TheftModel(spec).ok
        assert "ia_hurto" not in cams[conteo]["analysis"] and cams[conteo]["metrics_only"] is True
        assert "gestures" not in cams[cam_id]["analysis"] and cams[cam_id]["process_fps"] >= 5

        # ---- 5. la máquina ve a alguien que hace el mismo movimiento: evento hurto_ia, al servidor con sus rasgos
        clock = FakeTime()
        monkeypatch.setattr(worker_mod, "time", clock)
        live_pose = PhasePose(2)
        monkeypatch.setattr(worker_mod, "shared_pose", lambda: live_pose)

        def machine_run(t_hurto, seed, folder):
            c = cams[cam_id]
            scorer = Scorer.maybe(c["filter_model"]) if c.get("filter_model") is not None else None
            outbox = Outbox(tmp_path / folder, client)
            w = worker_mod.CameraWorker(c, RedPerson(), scorer, outbox, client, FakePool(), cfg.get("settings", {}))
            w._read_loop(LiveCap(clock, t_hurto, 16.0, seed))
            return w, outbox
        w, outbox = machine_run(8.0, 11, "cola")
        assert w.analyzer.theft_ai and not w.analyzer.gestures and w.status()["ia_hurto"] is True
        metas = [json.loads(p.read_text(encoding="utf-8")) for p in (tmp_path / "cola").glob("*/meta.json")]
        assert [m["type"] for m in metas] == [HURTO_IA], metas
        meta = metas[0]
        assert meta["severity"] == "alta" and meta["zone"] == "Góndola" and "posible hurto" in meta["description"]
        assert spec["threshold"] <= meta["features"]["ia_score"] <= 1.0 and meta["track_id"] is not None
        outbox.flush_once()  # (el clip no se arma en la prueba: el evento se manda igual)
        items = admin.get("/api/events", params={"camera_id": cam_id}).json()["items"]
        assert len(items) == 1 and items[0]["type"] == HURTO_IA
        ev = admin.get(f"/api/events/{items[0]['id']}").json()
        assert ev["severity"] == "alta" and ev["source"] == "vivo" and ev["alert"] is True
        assert ev["features"]["ia_score"] >= spec["threshold"]
        assert set(hurto_ia.FEATURE_NAMES) <= set(ev["features"]) and ev["features"]["track_age_s"] > 0

        # alguien parado sin hacer nada raro: no salta
        w2, _ = machine_run(None, 12, "cola2")
        assert w2.analyzer.theft_ai and w2.events == 0
        assert not list((tmp_path / "cola2").glob("*/meta.json"))


@pytest.mark.parametrize("t", [0.0, 0.5, 1.2, 2.0, 5.0])
def test_la_marca_del_video_se_lee_igual_despues_de_comprimirlo(tmp_path, t):
    """(La prueba de arriba depende de esto: la marca de color de la esquina sobrevive al MP4.)"""
    path = make_video(tmp_path / "v.mp4", 0.0, seed=1)
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * VIDEO_FPS))
    ok, frame = cap.read()
    cap.release()
    assert ok and PhasePose.which(frame) == phase(t, 0.0)
    assert RedPerson().detect(frame)[0].bbox == pytest.approx(PERSON, abs=3)
