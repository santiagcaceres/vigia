"""Pestaña «Entrenar objetos» de la app (18 nov, el dueño: «hacé otra pestaña… primero añadir el nuevo objeto y ponerle
para qué plan de métricas sería… hacelo completo con una guía de qué hacer y para qué en cada paso»).

Sin ventana: la frase de «qué sigue», lo que falta para entrenar, el texto de cada trabajo y el detector de una versión.
Con ventana (xvfb-run), con el servidor de verdad: crear el objeto con sus clases y el plan nuevo, subir fotos, marcar
con el mouse (dibujar, mover, cambiar la clase con el teclado, «No hay ninguno», descartar, aceptar lo que propone la
IA), pedir el entrenamiento, que una máquina lo termine y probar y activar la versión; todo dentro de 1366x768."""

import os
import queue
import threading
import time
import types
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from agent import entrenar_objetos as eo
from core import objetos as ob

ROOT = Path(__file__).resolve().parents[1]
ADMIN = ("admin", "clave-admin-123")
SHOTS = os.getenv("VIGIA_CAPTURAS")  # carpeta: si está, deja capturas de cada paso


def pallet_photo(seed: int, n: int = 2) -> tuple[np.ndarray, list[dict]]:
    """Una foto inventada: piso gris y pallets marrones (con o sin film celeste)."""
    rng = np.random.default_rng(seed)
    img = np.full((720, 1280, 3), (110, 115, 120), np.uint8)
    img += rng.integers(0, 20, img.shape, dtype=np.uint8)
    boxes = []
    for _ in range(n):
        w, h = int(rng.integers(160, 260)), int(rng.integers(100, 160))
        x, y = int(rng.integers(0, 1280 - w)), int(rng.integers(300, 720 - h))
        film = rng.random() < 0.5
        cv2.rectangle(img, (x, y), (x + w, y + h), (200, 150, 90) if film else (40, 90, 150), -1)
        for k in range(y + 8, y + h, 14):
            cv2.line(img, (x, k), (x + w, k), (230, 210, 190) if film else (20, 50, 80), 3)
        boxes.append({"k": "c2" if film else "c1", "x1": x / 1280, "y1": y / 720, "x2": (x + w) / 1280,
                      "y2": (y + h) / 720})
    return img, boxes


# ------------------------------------------------------------------ sin ventana
def test_la_guia_dice_que_sigue_en_cada_momento():
    cl = [{"key": "c1", "name": "Pallet"}]
    assert eo.guide_text(None)[1] == 0
    base = {"classes": cl, "boxes": {"c1": 0}, "jobs": [], "version_list": [],
            "counts": {"total": 5, "marcada": 0, "sin_marcar": 5}, "check": {"ok": False, "problems": ["Faltan"]}}
    text, step = eo.guide_text(base)
    assert step == 1 and "hay 5" in text
    m = {**base, "counts": {"total": 80, "marcada": 10, "sin_marcar": 70}}
    text, step = eo.guide_text(m)
    assert step == 2 and "70 fotos sin marcar" in text
    m = {**m, "check": {"ok": True}, "counts": {"total": 80, "marcada": 60, "sin_marcar": 20}}
    assert eo.guide_text(m)[1] == 3
    m["jobs"] = [{"kind": "entrenar", "status": "procesando", "message": "Vuelta 3 de 100"}]
    text, step = eo.guide_text(m)
    assert step == 3 and "Vuelta 3 de 100" in text
    m["jobs"] = []
    m["version_list"] = [{"number": 1, "grade": "Bien", "active": False}]
    assert eo.guide_text(m)[1] == 4
    m["version_list"] = [{"number": 2, "grade": "Muy bien", "active": False}, {"number": 1, "grade": "Bien",
                                                                                 "active": True}]
    assert "versión nueva (2" in eo.guide_text(m)[0]
    m["version_list"] = [{"number": 1, "grade": "Bien", "active": True}]
    assert "está activa" in eo.guide_text(m)[0]
    lines = eo.check_lines({"classes": cl, "boxes": {"c1": 25}, "counts": {"marcada": 31, "test_marked": 6}})
    assert all(ok for _, ok in lines) and "25 recuadros" in lines[1][0]
    assert eo.job_text({"kind": "video", "status": "listo", "result": {"images": 40, "proposed": 12}}) == \
        "Listo: 40 fotos (12 pre-marcadas)"
    assert "Error: sin placa" in eo.job_text({"kind": "entrenar", "status": "error", "error": "sin placa"})
    assert eo.box_at([{"x1": 0, "y1": 0, "x2": 1, "y2": 1}, {"x1": .4, "y1": .4, "x2": .6, "y2": .6}], .5, .5) == 1
    assert eo.detections_summary([], cl) == "No encontró ninguno."


def test_el_detector_de_una_version_y_un_objeto_una_sola_clase():
    """Un modelo ONNX de Ultralytics con N clases (salida (1, 4+N, 8400)): acá el de personas, mirando solo sus 2
    primeras clases como si fueran «c1» y «c2»."""
    det = ob.ObjectDetector(ROOT / "models" / "personas.onnx", ["c1", "c2"], conf=0.25)
    img = np.full((360, 640, 3), 120, np.uint8)
    assert det.detect(img) == []
    # el mismo recuadro visto con dos clases queda una vez, con la más segura
    a = {"k": "c1", "x1": .1, "y1": .1, "x2": .3, "y2": .3, "p": .6}
    b = {**a, "k": "c2", "p": .9, "x2": .31}
    c = {**a, "x1": .6, "x2": .8}
    assert ob.one_class_each([a, b, c]) == [b, c]


def test_las_fotos_de_un_video_saltean_las_repetidas(tmp_path):
    path = tmp_path / "v.mp4"
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 180))
    for i in range(200):  # 20 s: 10 quietos y 10 con algo que se mueve
        img = np.full((180, 320, 3), 100, np.uint8)
        if i >= 100:
            x = (i - 100) * 2  # (un autoelevador que cruza: algo grande y claro)
            cv2.rectangle(img, (x, 30), (x + 110, 150), (240, 240, 240), -1)
        out.write(img)
    out.release()
    shots = list(ob.video_frames(path, every_s=2.0))
    ts = [t for t, _ in shots]
    assert ts[0] == 0 and all(t >= 10 for t in ts[1:]) and len(shots) == 6  # 1 de los quietos y los 5 que cambian
    assert len(list(ob.video_frames(path, every_s=1.0, max_images=3))) == 3


# ------------------------------------------------------------------ con ventana y el servidor de verdad
def _pump(v, until, timeout=30.0):
    t0 = time.monotonic()
    while not until():
        v.root.update()
        time.sleep(0.02)
        assert time.monotonic() - t0 < timeout, "no terminó a tiempo"
    v.root.update()


def _sync_run(fn, done=None, fail=None):
    try:
        res = fn()
    except Exception as exc:  # noqa: BLE001
        if fail:
            return fail(exc)
        raise
    return done(res) if done else None


def _walk(w):
    out = [w]
    for c in w.winfo_children():
        out += _walk(c)
    return out


def _shot(v, name):
    if SHOTS:
        Path(SHOTS).mkdir(parents=True, exist_ok=True)
        v.root.update()
        time.sleep(0.3)
        v.root.update()
        os.system(f"import -window root {Path(SHOTS) / name}")


class Msgs:
    def __init__(self):
        self.seen = []

    def showinfo(self, *a, **k):
        self.seen.append(("info", a[1] if len(a) > 1 else ""))

    def showwarning(self, *a, **k):
        self.seen.append(("warning", a[1] if len(a) > 1 else ""))

    def showerror(self, *a, **k):
        self.seen.append(("error", a[1] if len(a) > 1 else ""))

    def askyesno(self, *a, **k):
        self.seen.append(("ask", a[1] if len(a) > 1 else ""))
        return True


def _click(tab, kind, fx, fy):
    """Un evento del mouse en el lugar (fx, fy) de la foto (fracciones)."""
    _, ox, oy, dw, dh = tab._geo
    ev = types.SimpleNamespace(x=ox + fx * dw, y=oy + fy * dh)
    {"press": tab._press, "move": tab._motion, "release": tab._release, "right": tab._right_click}[kind](ev)


def _key(tab, keysym):
    tab._key(types.SimpleNamespace(keysym=keysym))


def test_de_punta_a_punta_en_la_app(tmp_path, db_url):
    pytest.importorskip("tkinter")
    if os.name != "nt" and not os.environ.get("DISPLAY"):
        pytest.skip("sin pantalla (correr con xvfb-run)")
    from agent.oficina import Api, Oficina
    from agent.ventana import Ventana
    from server.main import Settings, create_app

    app = create_app(Settings(data_dir=tmp_path / "data", database_url=db_url, admin_password=ADMIN[1],
                              serve_public=False))
    c = TestClient(app)
    assert c.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]}).status_code == 200
    cid = c.post("/api/clients", json={"name": "Depósito Demo", "plans": ["objetos"]}).json()["id"]
    api = Api("http://testserver")
    api.http = c
    api.user = {"username": "admin", "role": "admin"}
    v = Ventana(tmp_path / "ventana.json")
    if v.failed or not v.open:
        pytest.skip("sin Tk")
    v.root.attributes("-zoomed", False)
    v.root.geometry("1366x698+0+0")
    v.build_main("admin (administrador)", analyze=True, on_demand=True)
    agent = types.SimpleNamespace(data_dir=tmp_path / "datos")
    off = Oficina(v, api, agent=lambda: agent)
    off.bg = types.SimpleNamespace(run=_sync_run, q=queue.Queue(), drain=lambda: None)
    off.mb = Msgs()
    off.clients = [{"id": cid, "name": "Depósito Demo", "plans": ["objetos"]}]
    off.cameras = []
    off.build(v.notebook)
    tab = off.objects_tab
    try:
        v.notebook.select(off.tab_objects)
        _pump(v, lambda: tab.guide.cget("text"))
        assert "Paso 1" in tab.guide.cget("text") and tab.model is None
        _shot(v, "1-sin-objeto.png")
        # 1 · crear el objeto: nombre, dos clases (una con palabra en inglés), el plan nuevo y el cliente
        tab.new_object()
        top = tab._dialog_top
        top.update()
        entries = [w for w in _walk(top) if w.winfo_class() == "TEntry"]
        entries[0].insert(0, "Pallets")
        rows = tab._dialog_rows
        rows[0]["name"].set("Pallet vacío")
        rows[0]["prompt"].set("pallet")
        next(w for w in _walk(top) if w.winfo_class() == "TButton" and w.cget("text") == "＋ Otra clase").invoke()
        rows[1]["name"].set("Terminado con film")
        combos = [w for w in _walk(top) if w.winfo_class() == "TCombobox"]
        assert combos[0].get().startswith("Conteo de objetos")  # el plan nuevo viene elegido
        help_text = " ".join(w.cget("text") for w in _walk(top) if w.winfo_class() == "TLabel")
        assert "Plan nuevo para contar cosas" in help_text
        combos[1].set("Depósito Demo")
        top.update_idletasks()
        assert top.winfo_reqheight() <= 698 - 30 and top.winfo_reqwidth() <= 1366
        _shot(v, "1-nuevo-objeto.png")
        next(w for w in _walk(top) if w.winfo_class() == "TButton" and w.cget("text") == "Guardar").invoke()
        _pump(v, lambda: tab.model is not None and tab.model.get("jobs") is not None)
        m = tab.model
        assert m["name"] == "Pallets" and m["plan"] == "objetos" and m["client_name"] == "Depósito Demo"
        assert [x["name"] for x in m["classes"]] == ["Pallet vacío", "Terminado con film"]
        assert tab.step() == 1 and "Paso 2" in tab.guide.cget("text")
        _shot(v, "1-objeto-creado.png")
        # 2 · subir fotos (34 inventadas y un archivo que no es foto)
        truth = {}
        paths = []
        for i in range(34):
            img, boxes = pallet_photo(i)
            p = tmp_path / f"pallets-{i:02d}.jpg"
            cv2.imwrite(str(p), img)
            paths.append(str(p))
            truth[ob.image_uid(ob.encode_photo(ob.read_photo(p)))] = boxes
        (tmp_path / "nota.txt").write_text("no soy una foto")
        tab.start_work(lambda stop: tab._photos_work(paths + [str(tmp_path / "nota.txt")], stop), "Subiendo")
        _pump(v, lambda: not tab.busy, 60)
        assert "34 fotos nuevas" in tab.work_line.cget("text") and "no es una imagen" in tab.work_line.cget("text")
        _pump(v, lambda: (tab.model.get("counts") or {}).get("total") == 34)
        assert "34 fotos" in tab.photo_counts.cget("text")
        _shot(v, "2-fotos.png")
        # 3 · marcar
        tab.go(2)
        _pump(v, lambda: tab.current() is not None and tab.current()["id"] in tab._pix)
        _shot(v, "3-marcar-vacia.png")
        first = tab.current()
        assert len(tab.images) == 34
        # dibujar un recuadro (clase 1), elegir la clase 2 con el teclado y moverlo
        tab.set_class("c1")
        _click(tab, "press", 0.10, 0.50)
        _click(tab, "move", 0.25, 0.70)
        _click(tab, "release", 0.25, 0.70)
        assert len(tab.boxes) == 1 and tab.boxes[0]["k"] == "c1" and tab.dirty
        _key(tab, "2")
        assert tab.boxes[0]["k"] == "c2"
        _click(tab, "press", 0.15, 0.60)  # adentro del recuadro: se mueve
        _click(tab, "move", 0.20, 0.60)
        _click(tab, "release", 0.20, 0.60)
        assert abs(tab.boxes[0]["x1"] - 0.15) < 0.01
        # un clic suelto no deja un recuadro; clic derecho borra
        _click(tab, "press", 0.80, 0.10)
        _click(tab, "release", 0.80, 0.10)
        assert len(tab.boxes) == 1
        _shot(v, "3-marcar.png")
        _key(tab, "Return")  # guarda y pasa a la siguiente
        assert tab.current()["id"] != first["id"]
        with app.state.Session() as s:
            from server.db import ObjectImage
            row = s.get(ObjectImage, first["id"])
            assert row.status == "marcada" and row.boxes[0]["k"] == "c2"
        _key(tab, "n")  # no hay ninguno
        _key(tab, "d")  # descartar
        d = c.get(f"/api/objects/{m['id']}").json()
        assert d["counts"]["marcada"] == 2 and d["counts"]["descartada"] == 1
        # el resto, como lo marcaría una persona (por la API, para no dibujar 31 recuadros)
        imgs = c.get(f"/api/objects/{m['id']}/images?status=sin_marcar&limit=500").json()["items"]
        for im in imgs:
            assert c.put(f"/api/objects/images/{im['id']}", json={"boxes": truth[im["uuid"]]}).status_code == 200
        # lo que propone la IA se acepta con A (una foto más, pre-marcada por una «máquina»)
        img, _ = pallet_photo(99, 1)
        extra = tmp_path / "pallets-99.jpg"
        cv2.imwrite(str(extra), img)
        tab.start_work(lambda stop: tab._photos_work([str(extra)], stop), "Subiendo")
        _pump(v, lambda: not tab.busy, 30)
        new = c.get(f"/api/objects/{m['id']}/images?status=sin_marcar").json()["items"][0]
        with app.state.Session() as s:
            row = s.get(ObjectImage, new["id"])
            row.proposed = [{"k": None, "x1": .3, "y1": .5, "x2": .5, "y2": .7, "p": .52}]
            row.proposed_by, row.doubt = "IA de palabras (YOLOE)", 0.96
            s.commit()
        tab.load_images()
        _pump(v, lambda: tab.current() is not None and tab.current()["id"] == new["id"] and tab.proposed)
        _shot(v, "3-propuesta-ia.png")
        assert "IA de palabras (YOLOE) propone 1 recuadro" in tab.label_line.cget("text")
        tab.set_class("c1")
        _key(tab, "a")
        assert tab.boxes == [{"k": "c1", "x1": .3, "y1": .5, "x2": .5, "y2": .7}] and not tab.proposed
        _key(tab, "Return")
        # 4 · entrenar: alcanza y queda en cola; sin máquina que entrene, lo dice
        tab.load_model(m["id"])
        _pump(v, lambda: (tab.model.get("check") or {}).get("ok"))
        tab.go(3)
        v.root.update()
        assert str(tab.train_btn.cget("state")) == "normal"
        assert "ninguna máquina que pueda entrenar" in tab.machines_line.cget("text")
        tab.train()
        assert any("quedó pedido" in t for k, t in off.mb.seen if k == "info")
        _pump(v, lambda: eo.active_job(tab.model, "entrenar") is not None)
        assert "En cola" in tab.train_state.cget("text") and str(tab.train_btn.cget("state")) == "disabled"
        _shot(v, "4-entrenar-en-cola.png")
        # una máquina lo toma y termina (con un .onnx de verdad: el de personas, que tiene 80 clases)
        key = c.post("/api/analysis/pool-key").json()["key"]
        tok = TestClient(app).post("/api/agent/enroll", json={"key": key, "name": "tigo-1",
                                                              "instance_id": "i-1"}).json()["token"]
        mq = TestClient(app)
        mq.headers["X-Agent-Token"] = tok
        mq.post("/api/agent/status", json={"cameras": [], "objects": {"video": True, "ultra": True, "gpu": True,
                                                                      "device": "Tesla T4"}})
        job = mq.get("/api/agent/objects/next").json()
        mq.post(f"/api/agent/objects/jobs/{job['job_id']}/progress", json={"progress": 0.4,
                                                                         "message": "Vuelta 40 de 100"})
        tab.load_model(m["id"], images=False)
        _pump(v, lambda: "Vuelta 40 de 100" in tab.train_state.cget("text"))
        assert "tigo-1" in tab.train_state.cget("text") and int(tab.train_bar.cget("value")) > 30
        _shot(v, "4-entrenando.png")
        f = mq.post(f"/api/agent/objects/jobs/{job['job_id']}/files").json()
        for url, data in ((f["onnx_url"], (ROOT / "models" / "personas.onnx").read_bytes()), (f["pt_url"], b"pt")):
            u = urlsplit(url)
            mq.put(u.path + "?" + u.query, content=data)
        mq.post(f"/api/agent/objects/jobs/{job['job_id']}/done", json={"result": {
            "metrics": {"map50": 0.87, "test_images": 7, "per_class": {
                "c1": {"precision": 0.9, "recall": 0.8}, "c2": {"precision": 0.7, "recall": 0.9}}},
            "n_images": 34, "n_boxes": 60, "device": "tigo-1 (Tesla T4)"}})
        # 5 · probar y activar
        tab.load_model(m["id"], images=False)
        _pump(v, lambda: (tab.model.get("version_list") or []))
        tab.go(4)
        v.root.update()
        assert "Paso 5" in tab.guide.cget("text")
        txt = tab.ver_text.get("1.0", "end")
        assert "Resultado: Muy bien" in txt and "encuentra 8 de cada 10" in txt
        _shot(v, "5-version.png")
        tab.set_version(True)
        _pump(v, lambda: tab.model.get("active_version_id"))
        assert "está activa" in tab.guide.cget("text")
        # probar con una foto: el modelo se baja una vez a esta PC y corre acá
        vv = tab._version()
        img, _ = pallet_photo(7)
        tab._test(vv, lambda: img, "pallets-07.jpg")
        _pump(v, lambda: getattr(tab, "_test_top", None) is not None)
        assert list((tmp_path / "datos" / "objetos").glob("*.onnx"))
        _shot(v, "5-probar.png")
        tab._test_top.destroy()
        # un video: a la máquina virtual (se sube entero y queda en cola) o en esta PC (sube solo las fotos)
        vid = tmp_path / "deposito.mp4"
        out = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 360))
        for i in range(120):
            img, _ = pallet_photo(500 + i // 20, 2)  # cambia cada 2 s
            out.write(cv2.resize(img, (640, 360)))
        out.release()
        msg = tab._video_vm(str(vid), 2.0, 50, threading.Event())
        assert msg.startswith("✔ Video subido") and "ninguna máquina" not in msg  # (tigo-1 está en línea)
        job = c.get(f"/api/objects/{m['id']}/jobs").json()[0]
        assert job["kind"] == "video" and job["status"] == "en_cola" and job["filename"] == "deposito.mp4"
        tab.load_model(m["id"], images=False)
        _pump(v, lambda: eo.active_job(tab.model, "video"))
        tab.cancel_job(eo.active_job(tab.model, "video"))
        assert c.get(f"/api/objects/{m['id']}/jobs").json()[0]["status"] == "cancelado"
        before = c.get(f"/api/objects/{m['id']}").json()["counts"]["total"]
        msg = tab._video_work(str(vid), 2.0, 50, threading.Event())
        assert msg.startswith("✔ 6 fotos nuevas"), msg
        imgs = c.get(f"/api/objects/{m['id']}/images?limit=10").json()["items"]
        assert c.get(f"/api/objects/{m['id']}").json()["counts"]["total"] == before + 6
        assert imgs[0]["source"] == "video" and imgs[0]["source_name"].startswith("deposito.mp4 0:")
        # todo entra en 1366x768 en cada paso, y lo importante de cada uno se ve
        must = {0: [], 1: [tab.next2_btn], 2: [tab.btn_ok, tab.cls_bar, tab.canvas, tab.img_list, tab.label_line],
                3: [tab.train_btn, tab.train_bar, tab.check_box], 4: [tab.ver_list, tab.ver_text]}
        for i in range(5):
            tab.go(i)
            v.root.update()
            for w in must[i]:
                assert w.winfo_ismapped(), (i, w)
            if i == 2:
                assert tab.canvas.winfo_height() >= 300, tab.canvas.winfo_height()
            p = tab.panels[i]
            for w in _walk(p):
                if w.winfo_ismapped() and w.winfo_class() in ("TButton", "Treeview", "Canvas", "Text"):
                    assert w.winfo_rootx() + w.winfo_width() <= v.root.winfo_rootx() + 1366 + 1, (i, w)
                    assert w.winfo_rooty() + w.winfo_height() <= v.root.winfo_rooty() + 698 + 1, (i, w)
        # los avisos verdes de lo que se hizo
        msgs = []
        while not api.done.empty():
            msgs.append(api.done.get_nowait())
        assert "Objeto creado" in msgs and "Entrenamiento pedido" in msgs and "Versión activada" in msgs
    finally:
        v.close()
