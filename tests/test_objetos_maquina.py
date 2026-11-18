"""Los trabajos de «Entrenar objetos» en una máquina de análisis (agent/objetos_maquina.py) contra el servidor de verdad:
sacar fotos de un video subido y pre-marcarlas, pre-marcar las sin marcar, entrenar (bajar fotos, armar la carpeta de
YOLO, subir el modelo), cancelar a mitad y una falla. Lo de Ultralytics lo hace un programa de mentira (escribe lo mismo
que agent/objetos_ia.py); con VIGIA_ULTRA_PY = un Python con Ultralytics, la última prueba entrena de verdad."""

import json
import os
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from agent import objetos_maquina as om
from agent.outbox import ServerClient
from core import objetos as ob
from server.main import Settings, create_app

ROOT = Path(__file__).resolve().parents[1]
ADMIN = ("admin", "clave-admin-123")

FAKE = r'''
import json, sys, time, shutil
from pathlib import Path
cmd, cfg = sys.argv[3], json.loads(Path(sys.argv[4]).read_text())
say = lambda **d: print("@@VIGIA " + json.dumps(d), flush=True)
print("Ultralytics de mentira", flush=True)
mode = Path("modo.txt")  # (corre parado en la carpeta de trabajo de la máquina)
mode = mode.read_text() if mode.exists() else ""
if "falla" in mode:
    print("CUDA out of memory", flush=True)
    say(error="RuntimeError: CUDA out of memory")
    sys.exit(1)
if cmd == "predict":
    out = {str(x["id"]): [{"k": list(cfg.get("prompts") or {"c1": 1})[0] if cfg["method"] == "yoloe" else cfg["keys"][0],
                          "x1": .1, "y1": .1, "x2": .4, "y2": .5, "p": .55}] for x in cfg["images"]}
    Path(cfg["out"]).write_text(json.dumps(out))
    say(progress=1.0, message="listo")
    say(done={"images": len(out)})
else:
    root = Path(cfg["data"]).parent
    labels = sorted((root / "labels" / "train").glob("*.txt")) + sorted((root / "labels" / "val").glob("*.txt"))
    n = 200 if "lento" in mode else 3
    for e in range(1, n + 1):
        say(progress=e / n, message=f"Vuelta {e} de {n}")
        time.sleep(0.05)
    out = Path(cfg["out"]); out.mkdir(parents=True, exist_ok=True)
    (out / "best.onnx").write_bytes(b"onnx-" + str(len(labels)).encode())
    (out / "best.pt").write_bytes(b"pt")
    say(done={"metrics": {"map50": 0.72, "test_images": cfg["test_images"],
                          "per_class": {cfg["keys"][0]: {"precision": 0.8, "recall": 0.7, "map50": 0.72}}},
              "onnx": str(out / "best.onnx"), "pt": str(out / "best.pt"), "device": "placa de mentira",
              "epochs_run": n, "minutes": 0.1, "base": cfg["base"], "labels": len(labels)})
'''


@pytest.fixture
def setup(tmp_path, db_url, monkeypatch):
    monkeypatch.setattr(om, "PROGRESS_EVERY_S", 0.0)
    app = create_app(Settings(data_dir=tmp_path / "data", database_url=db_url, admin_password=ADMIN[1],
                              serve_public=False))
    admin = TestClient(app)
    admin.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]})
    key = admin.post("/api/analysis/pool-key").json()["key"]
    tok = TestClient(app).post("/api/agent/enroll", json={"key": key, "name": "tigo-1",
                                                          "instance_id": "i-1"}).json()["token"]
    http = TestClient(app)
    http.headers["X-Agent-Token"] = tok
    http.post("/api/agent/status", json={"cameras": [], "objects": {"video": True, "ultra": True, "gpu": True}})
    fake = tmp_path / "fakepy.py"
    fake.write_text(FAKE)
    py = tmp_path / "python-falso"
    py.write_text(f"#!/bin/sh\nexec {sys.executable} {fake} \"$@\"\n")
    py.chmod(0o755)
    runner = om.ObjectJobRunner(ServerClient("http://testserver", tok, http=http), tmp_path / "work",
                                caps={"video": True, "ultra": True, "gpu": True}, python=str(py))
    return app, admin, runner, tmp_path


def _video(path: Path) -> Path:
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 180))
    for i in range(150):  # 15 s con una caja que cruza
        img = np.full((180, 320, 3), 90, np.uint8)
        x = i * 2
        cv2.rectangle(img, (x, 40), (x + 90, 140), (230, 230, 230), -1)
        out.write(img)
    out.release()
    return path


def _put(client, url, data):
    u = urlsplit(url)
    assert client.put(u.path + "?" + u.query, content=data).status_code < 300


def test_video_premarcar_y_entrenar(setup):
    app, admin, runner, tmp = setup
    mid = admin.post("/api/objects", json={"name": "Cajas", "classes": [{"name": "Caja", "prompt": "box"},
                                                                         {"name": "Caja abierta"}]}).json()["id"]
    # un video subido desde la app: la máquina saca las fotos y la IA (YOLOE) las pre-marca
    j = admin.post(f"/api/objects/{mid}/videos", json={"filename": "deposito.mp4", "every_s": 1.5}).json()
    _put(admin, j["upload_url"], _video(tmp / "v.mp4").read_bytes())
    admin.post(f"/api/objects/jobs/{j['id']}/uploaded")
    assert runner.run_once()
    job = admin.get(f"/api/objects/{mid}/jobs").json()[0]
    assert job["status"] == "listo", job
    n = job["result"]["images"]
    assert 5 <= n <= 11 and job["result"]["proposed"] == n and job["machine"] == "tigo-1"
    imgs = admin.get(f"/api/objects/{mid}/images?limit=100").json()["items"]
    assert len(imgs) == n and all(i["proposed_by"] == "IA de palabras (YOLOE)" and i["proposed"] for i in imgs)
    assert all(i["width"] == 320 and i["height"] == 180 and i["source_name"] == "deposito.mp4" for i in imgs)
    assert not runner.run_once()  # no queda nada
    # 30 marcadas → entrenar: baja las fotos, arma las etiquetas de YOLO y sube el modelo
    for i, im in enumerate(imgs):
        admin.put(f"/api/objects/images/{im['id']}", json={"boxes": [{"k": "c1", "x1": .1, "y1": .1, "x2": .5,
                                                                       "y2": .6}] * 3})
    extra = []
    for i in range(30 - n):
        img = np.full((90, 160, 3), (i * 7) % 255, np.uint8)
        jpg = ob.encode_photo(img)
        extra.append((ob.image_uid(jpg), jpg))
    got = admin.post(f"/api/objects/{mid}/images", json={"images": [{"uuid": u, "width": 160, "height": 90}
                                                                     for u, _ in extra]}).json()["items"]
    for g in got:
        _put(admin, g["upload_url"], dict(extra)[g["uuid"]])
    admin.post(f"/api/objects/{mid}/images/uploaded", json={"ids": [g["id"] for g in got]})
    for g in got:
        admin.put(f"/api/objects/images/{g['id']}", json={"boxes": []})  # «no hay ninguno»
    r = admin.post(f"/api/objects/{mid}/train", json={})
    assert r.status_code == 201, r.text
    assert runner.run_once()
    d = admin.get(f"/api/objects/{mid}").json()
    assert d["jobs"][0]["status"] == "listo", d["jobs"][0]
    v = d["version_list"][0]
    assert v["number"] == 1 and v["n_images"] == 30 and v["n_boxes"] == 3 * n and v["trained_on"] == "placa de mentira"
    assert v["grade"] == "Bien"
    with app.state.Session() as s:
        from server.db import ObjectVersion
        ver = s.get(ObjectVersion, v["id"])
        assert app.state.storage.get(ver.onnx_path) == b"onnx-30" and app.state.storage.get(ver.pt_path) == b"pt"
    # la versión activa pre-marca lo nuevo (con su .pt)
    admin.post(f"/api/objects/versions/{v['id']}/activate")
    admin.put(f"/api/objects/images/{imgs[0]['id']}", json={"status": "sin_marcar"})
    assert admin.post(f"/api/objects/{mid}/prelabel").status_code == 201
    assert runner.run_once()
    first = admin.get(f"/api/objects/{mid}/images?status=sin_marcar").json()["items"][0]
    assert first["proposed_by"] == "Versión 1"


def test_cancelar_a_mitad_corta_el_otro_proceso(setup):
    app, admin, runner, tmp = setup
    mid = admin.post("/api/objects", json={"name": "Pallets", "classes": [{"name": "Pallet"}]}).json()["id"]
    items = []
    for i in range(30):
        jpg = ob.encode_photo(np.full((90, 160, 3), i * 8, np.uint8))
        items.append((ob.image_uid(jpg), jpg))
    got = admin.post(f"/api/objects/{mid}/images", json={"images": [{"uuid": u, "width": 160, "height": 90}
                                                                     for u, _ in items]}).json()["items"]
    for g in got:
        _put(admin, g["upload_url"], dict(items)[g["uuid"]])
    admin.post(f"/api/objects/{mid}/images/uploaded", json={"ids": [g["id"] for g in got]})
    for g in got:
        admin.put(f"/api/objects/images/{g['id']}", json={"boxes": [{"k": "c1", "x1": .1, "y1": .1, "x2": .5,
                                                                      "y2": .5}]})
    (runner.work_dir / "modo.txt").write_text("lento")  # 200 vueltas: da tiempo a cancelar
    jid = admin.post(f"/api/objects/{mid}/train", json={}).json()["id"]
    t = threading.Thread(target=runner.run_once)
    t.start()
    t0 = time.monotonic()
    while "Vuelta" not in (admin.get(f"/api/objects/{mid}/jobs").json()[0].get("message") or ""):
        assert time.monotonic() - t0 < 30
        time.sleep(0.1)
    admin.post(f"/api/objects/jobs/{jid}/cancel")
    t.join(20)
    assert not t.is_alive()
    job = admin.get(f"/api/objects/{mid}/jobs").json()[0]
    assert job["status"] == "cancelado" and not admin.get(f"/api/objects/{mid}").json()["version_list"]
    # una falla de la IA queda con el motivo
    (runner.work_dir / "modo.txt").write_text("falla")
    admin.post(f"/api/objects/{mid}/train", json={})
    assert runner.run_once()
    job = admin.get(f"/api/objects/{mid}/jobs").json()[0]
    assert job["status"] == "error" and "CUDA out of memory" in job["error"]


def test_capacidades(monkeypatch, tmp_path):
    monkeypatch.setenv("VIGIA_PY_ENTRENAR", str(tmp_path / "no-existe"))
    assert om.capabilities()["ultra"] is False and om.capabilities()["video"] is True
    monkeypatch.setenv("VIGIA_PY_ENTRENAR", sys.executable)
    assert om.capabilities()["ultra"] is True
    monkeypatch.setenv("VIGIA_ENTRENAR", "no")
    assert om.capabilities()["ultra"] is False


@pytest.mark.skipif(not os.getenv("VIGIA_ULTRA_PY"), reason="VIGIA_ULTRA_PY no definido (un Python con Ultralytics)")
def test_entrena_de_verdad_con_ultralytics(setup):
    """Ultralytics de verdad (YOLO11n afinado con fotos inventadas de pallets, 3 vueltas en el procesador) y el .onnx que
    sale lo lee el detector de la app."""
    app, admin, runner, tmp = setup
    runner.python = os.environ["VIGIA_ULTRA_PY"]
    mid = admin.post("/api/objects", json={"name": "Pallets", "classes": [{"name": "Pallet"},
                                                                           {"name": "Con film"}]}).json()["id"]
    rng = np.random.default_rng(0)
    truth = {}
    shots = []
    for i in range(60):
        img = rng.integers(90, 140, (240, 320, 3)).astype(np.uint8)
        boxes = []
        for _ in range(int(rng.integers(1, 4))):
            w, h = int(rng.integers(50, 90)), int(rng.integers(30, 60))
            x, y = int(rng.integers(0, 320 - w)), int(rng.integers(0, 240 - h))
            film = rng.random() < 0.5
            cv2.rectangle(img, (x, y), (x + w, y + h), (200, 120, 60) if film else (40, 90, 150), -1)
            for k in range(y + 4, y + h, 8):
                cv2.line(img, (x, k), (x + w, k), (230, 200, 180) if film else (20, 40, 70), 2)
            boxes.append({"k": "c2" if film else "c1", "x1": x / 320, "y1": y / 240, "x2": (x + w) / 320,
                          "y2": (y + h) / 240})
        jpg = ob.encode_photo(img)
        truth[ob.image_uid(jpg)] = boxes
        shots.append((ob.image_uid(jpg), jpg))
    for k in range(0, 60, 20):
        part = shots[k:k + 20]
        got = admin.post(f"/api/objects/{mid}/images", json={"images": [{"uuid": u, "width": 320, "height": 240}
                                                                         for u, _ in part]}).json()["items"]
        for g in got:
            _put(admin, g["upload_url"], dict(part)[g["uuid"]])
        admin.post(f"/api/objects/{mid}/images/uploaded", json={"ids": [g["id"] for g in got]})
    for im in admin.get(f"/api/objects/{mid}/images?limit=100").json()["items"]:
        admin.put(f"/api/objects/images/{im['id']}", json={"boxes": truth[im["uuid"]]})
    assert admin.post(f"/api/objects/{mid}/train", json={"epochs": 3}).status_code == 201
    assert runner.run_once()
    d = admin.get(f"/api/objects/{mid}").json()
    assert d["jobs"][0]["status"] == "listo", d["jobs"][0]
    v = d["version_list"][0]
    assert v["metrics"]["map50"] is not None and v["metrics"]["test_images"] > 0
    with app.state.Session() as s:
        from server.db import ObjectVersion
        onnx = tmp / "v1.onnx"
        onnx.write_bytes(app.state.storage.get(s.get(ObjectVersion, v["id"]).onnx_path))
    det = ob.ObjectDetector(onnx, ["c1", "c2"], conf=0.25)
    img = np.full((240, 320, 3), 115, np.uint8)
    cv2.rectangle(img, (30, 40), (110, 90), (40, 90, 150), -1)
    for k in range(44, 90, 8):
        cv2.line(img, (30, k), (110, k), (20, 40, 70), 2)
    found = det.detect(img)
    assert any(abs(b["x1"] - 30 / 320) < 0.06 and abs(b["y2"] - 90 / 240) < 0.08 for b in found), found
