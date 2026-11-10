"""Pestaña «Entrenar IA hurtos» de la app (pedido del dueño, 9 nov: «subiremos clips y pondremos si hubo hurto y
dónde; aplicarle la IA de YOLO y la postura a estos videos y ahí entrenaremos a la IA»).

Sin ventana: los textos, el identificador del video, el análisis de un clip en la PC (con un detector y una postura
dobles sobre un video sintético) y lo que queda guardado en la PC. Con ventana (xvfb-run): la pestaña dentro de la
ventana de verdad en 1366x768, subir un clip (con un servidor doble), marcar con clic en la persona, guardar,
entrenar las dos IA y activarlas, un clip cuyo video está en otra PC y «Probar con las reglas»."""

import gzip
import json
import os
import queue
import threading
import time
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

from agent import entrenar_ia as ei
from agent import ia_clips
from core.detectors import Detection

RED = (0, 0, 255)  # la «persona» del video sintético (el detector doble la encuentra por el color)
BAG = (255, 0, 0)  # y su bolso


# ------------------------------------------------------------------ video sintético y dobles
def make_video(path: Path, size=(960, 1080), fps: float = 10.0, seconds: float = 4.0) -> Path:
    """Un video «1080p lite» (960x1080, píxeles el doble de anchos): una persona roja que camina de izquierda a
    derecha con un bolso azul. Nada de video real (el repo es público)."""
    w, h = size
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert out.isOpened()
    for i in range(int(fps * seconds)):
        img = np.full((h, w, 3), (90, 110, 120), np.uint8)
        cv2.rectangle(img, (0, int(h * 0.15)), (w, int(h * 0.35)), (40, 160, 200), -1)  # estante
        x = int(w * (0.15 + 0.5 * i / (fps * seconds)))
        cv2.rectangle(img, (x, int(h * 0.3)), (x + int(w * 0.08), int(h * 0.9)), RED, -1)
        cv2.rectangle(img, (x + int(w * 0.08), int(h * 0.6)), (x + int(w * 0.12), int(h * 0.72)), BAG, -1)
        out.write(img)
    out.release()
    return path


def _blob(frame: np.ndarray, color) -> tuple | None:
    b, g, r = (frame[..., i].astype(np.int16) for i in range(3))
    if color == RED:
        mask = (r > 180) & (g < 80) & (b < 80)
    else:
        mask = (b > 180) & (g < 80) & (r < 80)
    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


class ColorDetector:
    """Detector doble: la persona es lo rojo, el bolso lo azul. Guarda el tamaño de los cuadros que recibe."""

    def __init__(self):
        self.sizes = []

    def detect(self, frame, conf=None, far=None):
        self.sizes.append(frame.shape[1::-1])
        box = _blob(frame, RED)
        return [Detection(box, "persona", 0.9)] if box else []

    def detect_also(self, frame, classes, conf, person_conf=None, far=None):
        people = self.detect(frame)
        bag = _blob(frame, BAG) if 26 in classes else None
        return people, ([Detection(bag, "cartera", 0.6)] if bag else [])


class FakePose:
    """Postura doble: un esqueleto parado dentro del recuadro (17 puntos COCO), con la muñeca derecha en la cadera."""

    def estimate(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        cx = (x1 + x2) / 2

        def p(fx, fy):
            return [cx + fx * w, y1 + fy * h, 0.9]
        pts = [p(0, 0.05), p(-0.05, 0.04), p(0.05, 0.04), p(-0.1, 0.05), p(0.1, 0.05),
               p(-0.3, 0.2), p(0.3, 0.2), p(-0.35, 0.35), p(0.35, 0.35), p(-0.35, 0.5), p(0.25, 0.5),
               p(-0.15, 0.52), p(0.15, 0.52), p(-0.15, 0.75), p(0.15, 0.75), p(-0.15, 0.97), p(0.15, 0.97)]
        return np.array(pts, dtype=np.float32)


# ------------------------------------------------------------------ sin ventana
def test_los_textos_de_la_pestana():
    assert ei.verdict_text({"verdict": "hurto", "marks": [{}, {}]}) == "Hurto: 2 marcas"
    assert ei.verdict_text({"verdict": "hurto", "marks": [{}]}) == "Hurto: 1 marca"
    assert ei.verdict_text({"verdict": "normal"}) == "Normal"
    assert ei.verdict_text({}) == "Sin marcar"
    assert ei.verdict_text({"verdict": "normal", "feature_ok": False}) == "Normal · subir de nuevo"
    assert (ei.ia_result_text({"marks": 12, "marks_seen": 9, "false_per_hour": 1.5})
            == "De 12 hurtos marcados vio 9 · 1,5 falsas alarmas por hora de video normal")
    assert ei.ia_result_text({"marks": 1, "marks_seen": 1}) == "De 1 hurto marcado vio 1"
    assert ei.fmt_t(72.44, True) == "1:12.4" and ei.fmt_t(5, True) == "0:05.0" and ei.fmt_t(72.6) == "1:13"
    assert ei.mark_text({"start": 1, "end": 2.5, "track_id": 3, "kind": "bolsillo"}) == "0:01.0 a 0:02.5 · #3 · Al bolsillo"
    assert "cualquier persona" in ei.mark_text({"start": 1, "end": 2, "track_id": None, "kind": None})


def test_cuanto_hay_para_entrenar_y_que_se_dice():
    clips = [{"verdict": "hurto", "marks": [{}, {}], "duration_s": 30}, {"verdict": "hurto", "marks": [{}]},
             {"verdict": "normal", "duration_s": 600}, {"verdict": "sin_marcar"}]
    c = ei.summary_counts(None, clips)  # (sin el resumen del servidor: contado de la lista)
    assert (c["clips"], c["theft_clips"], c["marks"], c["normal_clips"], c["unmarked"]) == (4, 2, 3, 1, 1)
    assert c["normal_minutes"] == 10
    text, enough = ei.ia_advice(c)
    assert not enough and "al menos 5 hurtos marcados" in text and "30 a 50" in text
    assert ei.ia_counts_text(c) == "4 clips: 2 con hurto (3 marcas) · 1 normal (10 min) · 1 sin marcar."
    # el resumen del servidor manda (y sus mínimos)
    server = {"clips": 9, "theft_clips": 4, "marks": 8, "normal_clips": 3, "normal_minutes": 25.0, "unmarked": 2,
              "old_version": 1, "min_marks": 5, "min_normal_clips": 3, "min_normal_minutes": 10.0,
              "recommended_marks": 30}
    c = ei.summary_counts(server, clips)
    text, enough = ei.ia_advice(c)
    assert enough and "primera prueba" in text and "8 hurtos marcados" in text
    assert "1 clip de una versión vieja de la app no cuenta (volvé a subirlo)" in ei.ia_counts_text(c)
    assert ei.ia_advice({**c, "marks": 40})[0].startswith("Hay bastantes")
    assert not ei.ia_advice({**c, "theft_clips": 1})[1]  # todos los hurtos en un solo clip: no se puede validar
    # el filtro de alertas, de un cliente o de todos
    summary = {"clients": [{"client_id": 1, "labels": {"hurto": 5, "normal": 20, "sin_etiqueta": 3}, "from_tuning": 2},
                           {"client_id": 2, "labels": {"sospechoso": 4, "falso_positivo": 10}}],
               "min_total": 30, "min_per_class": 8}
    one = ei.filter_counts(summary, 1)
    assert (one["pos"], one["neg"], one["tuning"], one["unlabeled"]) == (5, 20, 2, 3)
    assert not ei.filter_text(one)[2] and "Hacen falta 30 en total" in ei.filter_text(one)[1]
    every = ei.filter_counts(summary, None)
    assert (every["pos"], every["neg"]) == (9, 30) and ei.filter_text(every)[2]


def test_el_identificador_del_video(tmp_path):
    a, b, c = tmp_path / "a.mp4", tmp_path / "b.mp4", tmp_path / "c.mp4"
    data = os.urandom(3 * 1024 * 1024)
    a.write_bytes(data)
    b.write_bytes(data)  # una copia (otra PC, otra carpeta): el mismo clip
    c.write_bytes(data[:-1] + b"x")  # cambia el final
    ua = ia_clips.clip_uuid(a)
    assert len(ua) == 40 and ua.isalnum()
    assert ia_clips.clip_uuid(b) == ua and ia_clips.clip_uuid(c) != ua
    small = tmp_path / "chico.mp4"
    small.write_bytes(b"hola")
    assert len(ia_clips.clip_uuid(small)) == 40


def test_analizar_un_clip_en_la_pc(tmp_path):
    """El video «lite» se analiza a 5 cuadros por segundo con píxeles cuadrados; cada registro trae la persona con su
    número, recuadro y esqueleto, y el bolso. Lo que sale sirve para core/hurto_ia.person_windows."""
    video = make_video(tmp_path / "clip.mp4")
    det, seen = ColorDetector(), []
    res = ia_clips.analyze(video, det, FakePose(), progress=lambda f, s: seen.append((f, s)))
    assert det.sizes and set(det.sizes) == {(1920, 1080)}  # píxeles cuadrados, como «🎞 Clips» y las máquinas
    assert (res["width"], res["height"], res["fps"]) == (1920, 1080, 5.0)
    frames = res["frames"]
    assert 17 <= len(frames) <= 21, len(frames)  # 4 s a 5 cuadros por segundo
    ts = [f["t"] for f in frames]
    assert ts == sorted(ts) and all(0.15 <= b - a <= 0.25 for a, b in zip(ts, ts[1:]))
    with_people = [f for f in frames if f["people"]]
    assert len(with_people) >= len(frames) - 3  # (el seguimiento confirma a la persona al tercer cuadro)
    p = with_people[-1]["people"][0]
    assert p["id"] == 1 and len(p["kp"]) == 17 and len(p["kp"][0]) == 3 and len(p["box"]) == 4
    assert all(len(f["bags"]) == 1 for f in with_people)
    assert res["people"] == 1 and res["posed"] == len(with_people) and not res["truncated"]
    assert 3.7 <= res["duration_s"] <= 4.1
    assert seen and seen[-1][0] > 0.9
    # sin postura (una PC sin el modelo): las personas igual, sin esqueleto
    res2 = ia_clips.analyze(video, ColorDetector(), None)
    assert res2["posed"] == 0 and all(p["kp"] is None for f in res2["frames"] for p in f["people"])
    # las ventanas de la IA de movimientos salen de esto
    from core import hurto_ia
    wins = hurto_ia.person_windows(frames)
    assert wins and all(w["track_id"] == 1 and len(w["f"]) == len(hurto_ia.FEATURE_NAMES) for w in wins)
    assert hurto_ia.person_windows(res2["frames"]) == []  # sin postura no hay ventanas
    # frenar
    stop = threading.Event()
    stop.set()
    with pytest.raises(ia_clips.ClipError, match="se frenó"):
        ia_clips.analyze(video, ColorDetector(), None, stop=stop)
    # lo que no es video
    bad = tmp_path / "no.mp4"
    bad.write_bytes(b"no es un video")
    with pytest.raises(ia_clips.ClipError, match="exportalo"):
        ia_clips.analyze(bad, ColorDetector(), None)
    # tope de duración
    res3 = ia_clips.analyze(video, ColorDetector(), None, max_seconds=2.0)
    assert res3["truncated"] and res3["frames"][-1]["t"] <= 2.0


def test_la_sensibilidad_de_la_camara_se_aplica(tmp_path):
    """Con una sensibilidad distinta de la de siempre, el detector compartido se usa con el umbral de la cámara (sin
    tocarlo para las demás)."""
    video = make_video(tmp_path / "clip.mp4", seconds=1.0)

    class Tunable(ColorDetector):
        tunable = True

        def __init__(self):
            super().__init__()
            self.confs = []

        def detect_also(self, frame, classes, conf, person_conf=None, far=None):
            self.confs.append((person_conf, far))
            return super().detect_also(frame, classes, conf)
    det = Tunable()
    ia_clips.analyze(video, det, None, detection={"sensitivity": 8, "far": True})
    assert det.confs and all(abs(c - 0.23) < 1e-6 and far is True for c, far in det.confs)


def test_lo_que_queda_en_la_pc(tmp_path):
    local = ia_clips.LocalClips(tmp_path / "entrenamiento")
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    assert local.path_of("abc") is None and not local.known("abc")
    local.link("abc", video)
    data = ia_clips.pack({"uuid": "abc"}, [{"t": 0.0, "w": 10, "h": 10, "people": [], "bags": []}])
    local.save_data("abc", data)
    again = ia_clips.LocalClips(tmp_path / "entrenamiento")  # (otra vez la app)
    assert again.path_of("abc") == str(video.resolve()) and again.load_data("abc")["frames"][0]["w"] == 10
    video.unlink()
    assert again.path_of("abc") is None and again.known("abc")  # se movió el archivo: «no está en esta PC»
    again.forget("abc")
    assert not again.known("abc") and again.load_data("abc") is None
    # el .json.gz se lee también sin comprimir (si el almacenamiento lo manda ya descomprimido) o como lista sola
    assert ia_clips.unpack(json.dumps([{"t": 1}]).encode())["frames"] == [{"t": 1}]
    assert ia_clips.unpack(gzip.compress(b'{"frames": []}'))["frames"] == []
    with pytest.raises(ValueError):
        ia_clips.unpack(b"basura")


def test_clic_en_la_persona_y_dibujo():
    rec = {"t": 1.0, "w": 1920, "h": 1080, "bags": [[700, 600, 760, 700]],
           "people": [{"id": 1, "box": [100, 200, 400, 1000], "kp": FakePose().estimate(None, (100, 200, 400, 1000)).tolist()},
                      {"id": 2, "box": [300, 300, 500, 900], "kp": None}]}
    assert ei.person_at(rec, 350, 500) == 2  # en las dos: la de recuadro más chico (la de adelante)
    assert ei.person_at(rec, 150, 250) == 1
    assert ei.person_at(rec, 1500, 500) is None
    assert ei.person_at(rec, 405, 1005, slack=10) == 1
    img, k = ei.render(rec, None, 1, [{"start": 0, "end": 2, "track_id": 2, "kind": "bolsillo"}], "0:01.0")
    assert img.shape[:2] == (ei.PLAYER_H, ei.PLAYER_W) and abs(k - ei.PLAYER_W / 1920) < 1e-6
    x1, y1 = int(100 * k), int(200 * k)
    assert tuple(img[y1 + 40, x1]) == ei.ORANGE  # la elegida, en naranja
    x2 = int(500 * k)
    assert tuple(img[int(600 * k), x2 - 1]) == ei.RED or tuple(img[int(600 * k), x2]) == ei.RED  # la de la marca
    assert tuple(img[5, 300]) == (0, 0, 160)  # el cartel de la marca
    # un cuadro de otro formato (4:3) entra igual, con bandas
    k2, w2, h2 = ei.fit(640, 480)
    assert h2 == ei.PLAYER_H and w2 == round(640 * ei.PLAYER_H / 480)
    assert ei.nearest([0.0, 0.2, 0.4], 0.3) == 1 and ei.nearest([0.0, 0.2], -1) == 0 and ei.nearest([], 1) == -1


def test_el_tipo_de_alerta_nuevo_tiene_nombre_en_la_app():
    import agent.oficina as of
    assert of.TYPE_NAMES["hurto_ia"] == "Posible hurto (IA de movimientos)"
    from core.rules import EVENT_TYPES, HURTO_IA
    assert HURTO_IA in EVENT_TYPES and HURTO_IA in of.TYPE_NAMES


def test_de_punta_a_punta_con_el_servidor_de_verdad(tmp_path, db_url):
    """Lo que hace la pestaña al subir un clip, contra el servidor de verdad (SQLite): el análisis en la PC (con un
    detector y una postura dobles), las ventanas de core/hurto_ia, el alta del clip, los cuadros al almacenamiento,
    bajarlos de vuelta, guardar marcas y el «faltan datos» al entrenar con un solo hurto."""
    from fastapi.testclient import TestClient

    from agent.oficina import Api, ApiError
    from server.main import Settings, create_app
    app = create_app(Settings(data_dir=tmp_path / "data", database_url=db_url, admin_password="clave-admin-123",
                              serve_public=False))
    with TestClient(app) as c:
        assert c.post("/api/login", json={"username": "admin", "password": "clave-admin-123"}).status_code == 200
        cid = c.post("/api/clients", json={"name": "Almacén Demo", "plans": ["hurto"]}).json()["id"]
        r = c.post("/api/cameras", json={"client_id": cid, "name": "Pasillo", "brand": "ip", "host": "dvr",
                                         "password": "x"})
        assert r.status_code == 201, r.text
        cam = next(x for x in c.get("/api/cameras").json() if x["id"] == r.json()["id"])
        api = Api("http://testserver")
        api.http = c  # (la sesión de la prueba)
        api.user = {"username": "admin", "role": "admin"}
        tab = ei.EntrenarIA.__new__(ei.EntrenarIA)  # (solo la subida, sin ventana)
        tab.app = types.SimpleNamespace(api=api, agent_of=None)
        tab.analyze_fn, tab.windows_fn = ia_clips.analyze, None
        tab.ai_setup = lambda: (ColorDetector(), FakePose())
        tab.data_dir, tab._local_obj, tab._q = tmp_path / "pc", None, queue.Queue()
        video = make_video(tmp_path / "hurto.mp4")

        def run(paths, verdict="sin_marcar"):
            tab._upload_work(paths, cid, cam, verdict, {}, threading.Event())
            out = []
            while not tab._q.empty():
                out.append(tab._q.get())
            return out
        msgs = run([str(video)])
        done = [m for m in msgs if m[0] == "done"]
        assert len(done) == 1 and done[0][3] == "", msgs
        clip = done[0][1]
        listed = c.get("/api/training/clips", params={"client_id": cid}).json()
        assert [x["uuid"] for x in listed] == [ia_clips.clip_uuid(video)] and listed[0]["n_windows"] > 0
        assert listed[0]["camera_name"] == "Pasillo" and listed[0]["created_by_name"] == "admin"
        assert listed[0]["has_data"] and listed[0]["feature_ok"]
        # los cuadros vuelven de la nube como se subieron (lo que ve otra PC)
        url = api.get(f"/api/training/clips/{clip['id']}/data")["url"]
        frames = ia_clips.unpack(api.fetch(url))["frames"]
        assert frames == tab.local.load_data(clip["uuid"])["frames"] and frames[-1]["people"][0]["kp"]
        # marcar: la persona 1, mientras se la ve
        res = api.put(f"/api/training/clips/{clip['id']}",
                      {"verdict": "hurto", "marks": [{"start": 1.0, "end": 2.6, "track_id": 1, "kind": "bolsillo"}]})
        assert res["verdict"] == "hurto" and res["marks_without_windows"] == [] and res["positive_windows"] > 0
        # una persona que no está: el servidor avisa que esa marca no enseña nada
        res = api.put(f"/api/training/clips/{clip['id']}",
                      {"marks": [{"start": 1.0, "end": 2.6, "track_id": 7, "kind": None}]})
        assert res["marks_without_windows"] == [0]
        # con un solo hurto no se entrena: el motivo, en palabras
        with pytest.raises(ApiError) as e:
            api.post("/api/training/ia/run", {"client_id": cid, "target_recall": ei.TARGET_RECALL})
        assert e.value.status == 422 and "Faltan hurtos marcados" in str(e.value)
        counts = ei.summary_counts(api.get("/api/training/ia/summary", client_id=cid), [])
        assert (counts["clips"], counts["theft_clips"], counts["marks"]) == (1, 1, 1)
        assert not ei.ia_advice(counts)[1]
        # el mismo video otra vez (por ejemplo desde otra PC): no se vuelve a analizar ni pisa las marcas
        msgs = run([str(video)])
        assert [m[0] for m in msgs if m[0] in ("done", "linked")] == ["linked"]
        assert c.get("/api/training/clips").json()[0]["marks"][0]["track_id"] == 7


def test_un_clip_largo_viaja_comprimido(tmp_path, db_url):
    """Un clip de 28 min con 12 personas a la vez: en JSON pasa los 4,5 MB que acepta Vercel; la app lo manda
    comprimido con gzip («Content-Encoding: gzip») y el servidor lo guarda entero. «Cortalo en partes» solo si
    comprimido pasa el tope."""
    from fastapi.testclient import TestClient

    from agent.oficina import Api, gzip_json
    from core import hurto_ia
    from server.main import Settings, create_app
    app = create_app(Settings(data_dir=tmp_path / "data", database_url=db_url, admin_password="clave-admin-123",
                              serve_public=False))
    rng = np.random.default_rng(3)
    n_f, people, n = len(hurto_ia.FEATURE_NAMES), 12, 19_800
    windows = [{"track_id": i % people, "t0": float(i // people), "t1": float(i // people) + 3.0,
                "f": [float(x) for x in np.round(rng.uniform(10, 99, n_f), 3)]} for i in range(n)]
    duration = n // people + 4.0
    with TestClient(app) as c:
        assert c.post("/api/login", json={"username": "admin", "password": "clave-admin-123"}).status_code == 200
        cid = c.post("/api/clients", json={"name": "Almacén Demo", "plans": ["hurto"]}).json()["id"]
        api = Api("http://testserver")
        api.http = c
        api.user = {"username": "admin", "role": "admin"}
        sent = []
        real_call = api.call

        def call(method, path, **kw):
            if path == "/api/training/clips" and method == "POST":
                sent.append(kw)
            return real_call(method, path, **kw)
        api.call = call
        tab = ei.EntrenarIA.__new__(ei.EntrenarIA)
        tab.app = types.SimpleNamespace(api=api, agent_of=None)
        tab.analyze_fn = lambda path, *a, **k: {"frames": [], "duration_s": duration, "fps": 5.0, "width": 640,
                                                "height": 360, "people": people, "video_fps": 25.0}
        tab.windows_fn = lambda frames, zones: windows
        tab.ai_setup = lambda: (object(), object())
        tab.data_dir, tab._local_obj, tab._q = tmp_path / "pc", None, queue.Queue()
        video = tmp_path / "largo.mp4"
        video.write_bytes(b"un video largo" * 1000)
        tab._upload_work([str(video)], cid, None, "normal", {}, threading.Event())
        msgs = [tab._q.get() for _ in range(tab._q.qsize())]
        assert [m[0] for m in msgs if m[0] in ("done", "error")] == ["done"], msgs
        # comprimido, con el encabezado, y mucho más chico que en JSON (que no entraría en Vercel)
        (kw,) = sent
        assert kw["headers"]["Content-Encoding"] == "gzip" and kw["headers"]["Content-Type"] == "application/json"
        plain = len(json.dumps(json.loads(gzip.decompress(kw["content"])), separators=(",", ":")))
        assert plain > 4_500_000 and len(kw["content"]) < ei.MAX_BODY < plain
        listed = c.get("/api/training/clips", params={"client_id": cid}).json()
        assert len(listed) == 1 and listed[0]["n_windows"] == n and listed[0]["verdict"] == "normal"
        # un cuerpo que ni comprimido entra: lo dice antes de mandarlo
        before = len(sent)
        old = ei.MAX_BODY
        ei.MAX_BODY = len(gzip_json({"windows": windows})) // 2
        try:
            other = tmp_path / "otro.mp4"
            other.write_bytes(b"otro video" * 1000)
            tab._upload_work([str(other)], cid, None, "normal", {}, threading.Event())
        finally:
            ei.MAX_BODY = old
        msgs = [tab._q.get() for _ in range(tab._q.qsize())]
        err = [m for m in msgs if m[0] == "error"]
        assert len(sent) == before and err and "cortalo en partes" in err[0][2]


# ------------------------------------------------------------------ con ventana
class Msgs:
    def __init__(self, answer=True):
        self.seen, self.answer = [], answer

    def _add(self, kind, a):
        self.seen.append((kind, a[1] if len(a) > 1 else ""))

    def showinfo(self, *a, **k):
        self._add("info", a)

    def showwarning(self, *a, **k):
        self._add("warning", a)

    def showerror(self, *a, **k):
        self._add("error", a)

    def askyesno(self, *a, **k):
        self._add("ask", a)
        return self.answer


class FakeServer:
    """El servidor doble: lo que la pestaña pide y manda (los clips, los modelos, las marcas)."""

    def __init__(self, user="lucia", role="admin"):
        from agent.oficina import ApiError
        self.ApiError = ApiError
        self.user = {"id": 4, "username": user, "role": role}
        self.calls: list[tuple] = []
        self.clips: list[dict] = []
        self.files: dict[str, bytes] = {}
        self.ia_models: list[dict] = []
        self.filter_models = [{"id": 3, "client_id": 1, "client_name": "Almacén Demo", "n_samples": 40,
                               "metrics": {"recall": 0.95, "falsas_alarmas_eliminadas": 0.4}, "active": False,
                               "created_at": "2026-11-08T10:00:00Z"}]
        self.ia_fail: Exception | None = None
        self.clients = [{"id": 1, "name": "Almacén Demo", "plans": ["hurto", "comercio"]},
                        {"id": 2, "name": "Kiosco Sur", "plans": ["hurto"]}]
        self.cameras = [{"id": 11, "client_id": 1, "name": "Pasillo bebidas", "enabled": True,
                         "zones": [{"name": "Góndola", "type": "shelf",
                                    "polygon": [[0.1, 0.5], [0.9, 0.5], [0.9, 0.99], [0.1, 0.99]]}],
                         "detection": {"sensitivity": 5, "far": False}},
                        {"id": 12, "client_id": 1, "name": "Caja", "enabled": True, "zones": []},
                        {"id": 21, "client_id": 2, "name": "Mostrador", "enabled": True, "zones": []}]

    def get(self, path, **params):
        self.calls.append(("GET", path, params))
        if path == "/api/clients":
            return self.clients
        if path == "/api/cameras":
            return self.cameras
        if path in ("/api/agents", "/api/users"):
            return []
        if path == "/api/stats/alerts":
            return {"clients": {}, "cameras": {}}
        if path == "/api/events":
            return {"items": [], "total": 0}
        if path == "/api/training/clips":
            cid = params.get("client_id")
            return [c for c in self.clips if cid is None or c.get("client_id") == cid]
        if path == "/api/training/ia/summary":
            cid = params.get("client_id")
            mine = [c for c in self.clips if cid is None or c.get("client_id") == cid]
            return {"clips": len(mine), "theft_clips": sum(c["verdict"] == "hurto" for c in mine),
                    "marks": sum(len(c.get("marks") or []) for c in mine if c["verdict"] == "hurto"),
                    "normal_clips": sum(c["verdict"] == "normal" for c in mine), "normal_minutes": 0.0,
                    "unmarked": sum(c["verdict"] == "sin_marcar" for c in mine), "old_version": 0,
                    "min_marks": 5, "min_normal_clips": 3, "min_normal_minutes": 10.0, "recommended_marks": 30,
                    "models": [m for m in self.ia_models if cid is None or m["client_id"] in (cid, None)]}
        if path == "/api/training/summary":
            return {"clients": [{"client_id": 1, "labels": {"hurto": 10, "normal": 30}, "from_tuning": 4}],
                    "min_total": 30, "min_per_class": 8}
        if path == "/api/models":
            return self.filter_models
        if path.startswith("/api/training/clips/") and path.endswith("/data"):
            cid = int(path.split("/")[4])
            clip = next(c for c in self.clips if c["id"] == cid)
            if f"/files/{clip['uuid']}" not in self.files:
                raise self.ApiError("Los cuadros de este clip todavía no llegaron a la nube", 404)
            return {"url": f"/files/{clip['uuid']}", "expires_in": 3600}
        if path == "/api/app/update":
            return {}
        raise AssertionError(f"GET inesperado {path}")

    def post(self, path, body=None):
        self.calls.append(("POST", path, body))
        if path == "/api/training/clips":
            row = {k: v for k, v in body.items() if k != "windows"}
            row.update(id=100 + len(self.clips), n_windows=len(body["windows"]), created_by=4,
                       created_by_name=self.user["username"], client_name="Almacén Demo",
                       camera_name="Pasillo bebidas" if body.get("camera_id") == 11 else None, feature_ok=True)
            self.clips.insert(0, row)
            return {**row, "data_upload": f"/files/{body['uuid']}", "data_content_type": "application/gzip"}
        if path == "/api/training/ia/run":
            if self.ia_fail:
                raise self.ia_fail
            m = {"id": 7, "client_id": body["client_id"], "client_name": "Almacén Demo" if body["client_id"] else None,
                 "metrics": {"marks": 12, "marks_seen": 9, "false_per_hour": 1.5}, "n_clips": 6, "active": False,
                 "usable": True, "created_at": "2026-11-09T12:00:00Z"}
            self.ia_models.insert(0, m)
            return m
        if path == "/api/training/run":
            m = {"id": 4 if len(self.filter_models) == 1 else 3 + len(self.filter_models), "client_id": body["client_id"],
                 "client_name": "Almacén Demo" if body["client_id"] else None, "n_samples": 40,
                 "metrics": {"recall": 0.9, "falsas_alarmas_eliminadas": 0.55}, "active": False,
                 "created_at": "2026-11-09T12:00:00Z"}
            self.filter_models.insert(0, m)
            return m
        for models, base in ((self.ia_models, "/api/training/ia/models/"), (self.filter_models, "/api/models/")):
            if path.startswith(base):
                mid, action = path[len(base):].split("/")
                m = next(x for x in models if x["id"] == int(mid))
                m["active"] = action == "activate"
                return m
        if path == "/api/agent/session":
            return {"token": "t"}
        raise AssertionError(f"POST inesperado {path}")

    def post_gzip(self, path, data):
        """El clip viaja comprimido (como lo manda Api.post_gzip): se descomprime y sigue como un POST común."""
        assert isinstance(data, bytes) and data[:2] == b"\x1f\x8b", "el clip tiene que ir comprimido con gzip"
        return self.post(path, json.loads(gzip.decompress(data)))

    def put(self, path, body):
        self.calls.append(("PUT", path, body))
        cid = int(path.rsplit("/", 1)[1])
        clip = next(c for c in self.clips if c["id"] == cid)
        clip.update(body)
        # (como el servidor: las marcas sin ninguna ventana de esa persona no enseñan nada)
        empty = [i for i, m in enumerate(body.get("marks") or []) if m.get("track_id") == 99]
        return {**clip, "marks_without_windows": empty}

    def delete(self, path):
        self.calls.append(("DELETE", path, None))
        cid = int(path.rsplit("/", 1)[1])
        self.clips = [c for c in self.clips if c["id"] != cid]

    def patch(self, path, body):
        raise AssertionError("PATCH inesperado")

    def upload(self, url, data, content_type):
        self.calls.append(("UPLOAD", url, content_type))
        self.files[url] = data

    def fetch(self, url):
        self.calls.append(("FETCH", url, None))
        return self.files[url]

    def sent(self, method, path):
        return [c[2] for c in self.calls if c[0] == method and c[1] == path]


def _sync_run(fn, done=None, fail=None):
    try:
        res = fn()
    except Exception as exc:  # noqa: BLE001
        if fail:
            return fail(exc)
        raise
    return done(res) if done else None


def _ventana(tmp_path):
    pytest.importorskip("tkinter")
    if os.name != "nt" and not os.environ.get("DISPLAY"):
        pytest.skip("sin pantalla (correr con xvfb-run)")
    from agent.ventana import Ventana
    v = Ventana(tmp_path / "ventana.json")
    if v.failed or not v.open:
        pytest.skip("sin Tk")
    return v


def _app(tmp_path, role="admin", user="lucia", size=(1366, 698)):
    """La ventana de verdad (cabecera, pestañas, registro abajo) con la Oficina y un servidor doble. 1366x698 =
    una pantalla de 1366x768 menos la barra de tareas y el título de la ventana (maximizada)."""
    from agent.oficina import Oficina
    v = _ventana(tmp_path)
    v.root.attributes("-zoomed", False)
    v.root.geometry(f"{size[0]}x{size[1]}+0+0")
    v.build_main(f"{user} ({'administrador' if role == 'admin' else 'operador'})", analyze=True, on_demand=True)
    server = FakeServer(user, role)
    agent = types.SimpleNamespace(data_dir=tmp_path / "datos")
    off = Oficina(v, server, agent=lambda: agent)
    off.bg = types.SimpleNamespace(run=_sync_run, q=queue.Queue(), drain=lambda: None)
    off.mb = Msgs()
    off.build(v.notebook)
    tab = off.trainer
    tab.ai_setup = lambda: (ColorDetector(), FakePose())
    return v, off, tab, server


def _pump(v, until, timeout=20.0):
    t0 = time.monotonic()
    while not until():
        v.root.update()
        time.sleep(0.02)
        assert time.monotonic() - t0 < timeout, "no terminó a tiempo"
    v.root.update()


def _click_person(tab, pid):
    rec = tab._shown_rec
    p = next(x for x in rec["people"] if x["id"] == pid)
    x1, y1, x2, y2 = p["box"]
    ev = types.SimpleNamespace(x=tab._ox + (x1 + x2) / 2 * tab._k, y=tab._oy + (y1 + y2) / 2 * tab._k)
    tab.on_click(ev)


def test_la_pestana_esta_entre_alertas_y_maquinas_y_entra_en_1366x768(tmp_path):
    for role, tabs in (("admin", ["En vivo", "Clientes", "Alertas", "Entrenar IA hurtos", "Máquinas",
                                  "Administración"]),
                       ("entrenador", ["En vivo", "Clientes", "Alertas", "Entrenar IA hurtos"])):
        v, off, tab, server = _app(tmp_path / role, role=role)
        try:
            names = [v.notebook.tab(t, "text").strip() for t in v.notebook.tabs()]
            assert names == tabs, names
            # con datos: un clip con hurto, uno normal; los dos modelos; elegido el primero
            server.clips = [
                {"id": 1, "uuid": "a" * 40, "name": "hurto-pasillo-1.mp4", "client_id": 1, "client_name": "Almacén Demo",
                 "camera_id": 11, "camera_name": "Pasillo bebidas", "duration_s": 48.0, "verdict": "hurto",
                 "marks": [{"start": 12.0, "end": 18.5, "track_id": 3, "kind": "bolsillo"}], "people": 2,
                 "created_by_name": "lucia", "width": 1920, "height": 1080, "feature_ok": True},
                {"id": 2, "uuid": "b" * 40, "name": "rato-normal-con-un-nombre-largo-de-verdad.mp4", "client_id": 1,
                 "client_name": "Almacén Demo", "camera_id": 12, "camera_name": "Caja", "duration_s": 600.0,
                 "verdict": "normal", "marks": [], "people": 5, "created_by_name": "marcos", "feature_ok": True}]
            v.notebook.select(off.tab_train)
            v.root.update()
            assert len(tab.tree.get_children()) == 2
            tab.tree.selection_set("1")
            v.root.update()
            assert tab.clip["id"] == 1 and len(tab.mtree.get_children()) == 1
            v.root.update_idletasks()
            need_w, need_h = off.tab_train.winfo_reqwidth(), off.tab_train.winfo_reqheight()
            have_w, have_h = off.tab_train.winfo_width(), off.tab_train.winfo_height()
            assert need_w <= have_w and need_h <= have_h, ((need_w, need_h), (have_w, have_h))
            # nada de la pestaña queda afuera de la ventana (con las dos IA de entrenar a la vista)
            for panel in (tab.ia, tab.flt):
                tab.train_nb.select(panel["tree"].master)
                panel["result"].configure(text="Faltan hurtos marcados: hay 1 y se necesitan al menos 5 (cada uno "
                                               "con la persona a la vista y su postura). Empieza a servir de verdad "
                                               "con 30 a 50.")  # (el texto más largo que llega del servidor)
                v.root.update()
                for w in (tab.canvas, tab.add_btn, tab.normal_btn, tab.guide, tab.step_lbl, panel["tree"], panel["btn"],
                          tab.tree, tab.mtree):
                    assert w.winfo_ismapped(), w
                    assert w.winfo_rootx() + w.winfo_width() <= v.root.winfo_rootx() + 1366, w
                    assert w.winfo_rooty() + w.winfo_height() <= v.root.winfo_rooty() + 698, w
                bottom = panel["tree"].master.winfo_children()[-1]  # «Activar» / «Apagar»
                assert bottom.winfo_rooty() + bottom.winfo_height() <= off.tab_train.winfo_rooty() + have_h
            # el diálogo de subir clips también entra
            seen = {}

            def answer():
                top = tab._ask_top
                top.update_idletasks()
                seen["size"] = (top.winfo_reqwidth(), top.winfo_reqheight())
                ok = next(w for w in _walk(top) if w.winfo_class() == "TButton" and w.cget("text") == "Analizar y subir")
                ok.invoke()
            v.root.after(300, answer)
            res = tab.ask_upload([str(tmp_path / "x.mp4")])
            assert seen["size"][0] <= 1366 and seen["size"][1] <= 698 - 30, seen
            assert res[0] == 1 and res[2] == "sin_marcar"  # (el cliente elegido arriba; sin decir nada, «sin marcar»)
        finally:
            tab.close()
            v.close()


def _walk(w):
    yield w
    for c in w.winfo_children():
        yield from _walk(c)


def test_subir_marcar_guardar_entrenar_y_activar(tmp_path):
    v, off, tab, server = _app(tmp_path)
    try:
        v.notebook.select(off.tab_train)
        v.root.update()
        assert tab.ia["info"].cget("text").startswith("0 clips")
        video = make_video(tmp_path / "Almacen - hurto.mp4")
        cam = server.cameras[0]
        tab.start_upload([str(video)], 1, cam, "sin_marcar")
        assert tab.uploading
        _pump(v, lambda: not tab.uploading)
        body = server.sent("POST", "/api/training/clips")[0]
        assert body["uuid"] == ia_clips.clip_uuid(video) and body["name"] == "Almacen - hurto.mp4"
        assert (body["client_id"], body["camera_id"], body["verdict"], body["marks"]) == (1, 11, "sin_marcar", [])
        assert (body["width"], body["height"], body["fps"], body["people"]) == (1920, 1080, 5.0, 1)
        from core import hurto_ia
        assert body["windows"] and body["feature_version"] == hurto_ia.FEATURE_VERSION
        assert all(len(w["f"]) == len(hurto_ia.FEATURE_NAMES) for w in body["windows"])
        # los cuadros, directo al almacenamiento y también en esta PC (el video no se sube)
        up = [c for c in server.calls if c[0] == "UPLOAD"]
        assert up == [("UPLOAD", f"/files/{body['uuid']}", "application/gzip")]
        frames = ia_clips.unpack(server.files[f"/files/{body['uuid']}"])["frames"]
        assert len(frames) >= 17 and frames[-1]["people"][0]["kp"]
        assert (tmp_path / "datos" / "entrenamiento" / f"{body['uuid']}.json.gz").is_file()
        assert ia_clips.LocalClips(tmp_path / "datos" / "entrenamiento").path_of(body["uuid"]) == str(video.resolve())
        assert not any(len(json.dumps(c, default=str)) > 2_000_000 for c in server.calls)
        assert "1 clip subido" in tab.up_lbl.cget("text")
        # queda elegido y se ve con el video de esta PC
        row = tab.tree.item(str(server.clips[0]["id"]))
        assert row["values"][-1] == "✔" and row["values"][1] == "Sin marcar" and "subió lucia" in tab.info.cget("text")
        # arriba dice qué sigue; al lado del clip, el primer paso para marcar
        assert tab.guide.cget("text").startswith("Ahora:  Falta decir qué pasa en 1 clip")
        assert "Donde EMPIEZA" in tab.step_lbl.cget("text")
        assert tab.clip["id"] == server.clips[0]["id"] and tab.video == str(video.resolve())
        assert "el video está en esta PC" in tab.info.cget("text")
        _pump(v, lambda: tab.base is not None)
        assert tab.base.shape[:2] == (ei.PLAYER_H, ei.PLAYER_W)

        # marcar: inicio, fin, clic en la persona, cómo → la marca a la lista
        tab._goto(1.0)
        _pump(v, lambda: abs(tab.base_t - 1.0) < 1e-6 and tab._shown_rec is not None)
        tab.set_start()
        assert tab.start_lbl.cget("text") == "0:01.0" and "TERMINA" in tab.step_lbl.cget("text")
        _click_person(tab, 1)
        assert tab.sel_track == 1 and "#1" in tab.who_lbl.cget("text")
        tab._goto(2.6)
        tab.set_end()
        assert "Guardar este hurto" in tab.step_lbl.cget("text")  # ya está todo: falta el botón
        tab.kind.set("Al bolsillo")
        tab.add_mark()
        # «✔ Guardar este hurto» lo guarda en el momento (no hay otro botón de guardar)
        assert tab.marks == [{"start": 1.0, "end": 2.6, "track_id": 1, "kind": "bolsillo"}]
        put = server.sent("PUT", f"/api/training/clips/{server.clips[0]['id']}")[-1]
        assert put == {"verdict": "hurto", "marks": [{"start": 1.0, "end": 2.6, "track_id": 1, "kind": "bolsillo"}]}
        assert not tab.dirty and len(tab.mtree.get_children()) == 1
        assert tab.step_lbl.cget("text").startswith("✔ Hurto guardado")
        assert tab.tree.item(str(server.clips[0]["id"]))["values"][1] == "Hurto: 1 marca"
        assert not [m for m in off.mb.seen if m[0] == "warning"]
        # mientras dura la marca, la persona se dibuja en rojo con el cartel
        tab._goto(2.0)
        _pump(v, lambda: abs(tab.base_t - 2.0) < 1e-6)
        assert tab._active(2.0) and tab._active(3.0) == []
        # una marca de alguien sin postura en ese rato: el servidor lo dice y la app avisa
        tab.marks.append({"start": 3.0, "end": 3.5, "track_id": 99, "kind": None})
        tab._mark_changed()
        assert any("no le enseña nada" in m[1] for m in off.mb.seen if m[0] == "warning")
        # «Quitar» también se guarda solo
        tab.mtree.selection_set("1")
        tab.remove_mark()
        assert len(server.sent("PUT", f"/api/training/clips/{server.clips[0]['id']}")[-1]["marks"]) == 1
        # sin persona elegida pregunta; sin fin, explica
        tab.m_start, tab.m_end = 0.5, None
        tab.add_mark()
        assert "Fin del hurto" in off.mb.seen[-1][1]

        # entrenar la IA de movimientos: el resultado en palabras y, ahí mismo, la pregunta para activarla
        tab.scope.set("Almacén Demo")
        tab.load()
        off.mb.answer = False  # «No»: queda entrenada y apagada, hasta «Activar»
        tab.train_ia()
        assert server.sent("POST", "/api/training/ia/run")[-1] == {"client_id": 1, "target_recall": 0.8}
        assert tab.ia["result"].cget("text").startswith(
            "De 12 hurtos marcados vio 9 · 1,5 falsas alarmas por hora de video normal")
        assert "falta «Activar»" in tab.ia["result"].cget("text")
        assert any("IA de movimientos para Almacén Demo" in m[1] for m in off.mb.seen if m[0] == "ask")
        rows = tab.ia["tree"].get_children()
        assert rows == ("7",) and tab.ia["tree"].item("7")["values"][2:] == ["vio 9 de 12 · 1,5 falsas/hora", ""]
        assert ("POST", "/api/training/ia/models/7/activate", None) not in server.calls
        assert "apagada" in tab.guide.cget("text")
        # con «Sí» (o después, con «Activar») las máquinas la empiezan a usar
        off.mb.answer = True
        tab.ia["tree"].selection_set("7")
        tab.set_model(True, True)
        assert ("POST", "/api/training/ia/models/7/activate", None) in server.calls
        assert tab.ia["tree"].item("7")["values"][-1] == "sí" and "✔ En uso" in tab.ia["result"].cget("text")
        assert "está activada" in tab.guide.cget("text")
        tab.ia["tree"].selection_set("7")
        tab.set_model(True, False)
        assert ("POST", "/api/training/ia/models/7/deactivate", None) in server.calls
        # pocos datos: el motivo que da el servidor, en la pestaña (sin ventanas que tapen)
        server.ia_fail = server.ApiError("Faltan hurtos marcados: hay 1 y se necesitan al menos 5.", 422)
        tab.train_ia()
        assert "Faltan hurtos marcados" in tab.ia["result"].cget("text")
        assert str(tab.ia["btn"].cget("state")) == "normal"

        # el filtro de alertas del cliente (el de antes, ahora acá)
        assert tab.flt["info"].cget("text").startswith("40 alertas etiquetadas: 10 hurto o sospechoso")
        tab.train_filter()
        assert server.sent("POST", "/api/training/run")[-1] == {"client_id": 1}
        assert tab.flt["result"].cget("text").startswith("Con 40 ejemplos: Deja pasar el 90 % de los hurtos")
        # recién entrenado, pregunta si se activa (con cómo anduvo): «Sí» lo deja en uso
        assert any("¿Activar el filtro de alertas para Almacén Demo?" in m[1] for m in off.mb.seen if m[0] == "ask")
        assert ("POST", "/api/models/4/activate", None) in server.calls
        # para todos los clientes
        tab.scope.set(ei.ALL_CLIENTS)
        tab.load()
        tab.train_filter()
        assert server.sent("POST", "/api/training/run")[-1] == {"client_id": None}

        # «Este clip es normal» borra las marcas (pregunta) y lo guarda
        tab.tree.selection_set(str(server.clips[0]["id"]))
        v.root.update()
        tab.set_normal()
        assert server.sent("PUT", f"/api/training/clips/{server.clips[0]['id']}")[-1] == {"verdict": "normal",
                                                                                            "marks": []}
        # subir el mismo video otra vez no lo vuelve a analizar: queda vinculado (aunque se esté mirando otro cliente)
        n_posts = len(server.sent("POST", "/api/training/clips"))
        tab.scope.set("Kiosco Sur")
        tab.load()
        assert not tab.tree.get_children()
        tab.start_upload([str(video)], 1, cam, "sin_marcar")
        _pump(v, lambda: not tab.uploading)
        assert len(server.sent("POST", "/api/training/clips")) == n_posts
        assert "ya estaba subido" in " ".join(w for _n, w in tab.upload_log)
        tab.scope.set(ei.ALL_CLIENTS)
        tab.load()
        tab.tree.selection_set(str(server.clips[0]["id"]))
        v.root.update()
        # borrar
        tab.delete_clip()
        assert server.clips == [] and tab.clip is None and not tab.tree.get_children()
        assert not (tmp_path / "datos" / "entrenamiento" / f"{body['uuid']}.json.gz").exists()
    finally:
        tab.close()
        v.close()


def test_frenar_y_un_video_que_no_abre(tmp_path):
    v, off, tab, server = _app(tmp_path)
    try:
        v.notebook.select(off.tab_train)
        bad = tmp_path / "grabador.dav"
        bad.write_bytes(b"no es un video")
        good = make_video(tmp_path / "bien.mp4", seconds=1.0)
        tab.start_upload([str(bad), str(good)], 1, None, "normal")
        _pump(v, lambda: not tab.uploading)
        assert len(server.sent("POST", "/api/training/clips")) == 1
        body = server.sent("POST", "/api/training/clips")[0]
        assert body["verdict"] == "normal" and body["camera_id"] is None
        warn = [m[1] for m in off.mb.seen if m[0] == "warning"][-1]
        assert "grabador.dav" in warn and "exportalo" in warn
        assert "bien.mp4" in warn and "no vio bien la postura de nadie" in warn  # (1 s: no alcanza para una ventana)
        # frenar: el que se estaba analizando no se sube
        slow = threading.Event()

        def slow_analyze(path, *a, stop=None, **k):
            slow.set()
            while not stop.is_set():
                time.sleep(0.01)
            raise ia_clips.ClipError("se frenó")
        tab.analyze_fn = slow_analyze
        other = make_video(tmp_path / "otro.mp4", seconds=1.5)
        tab.start_upload([str(other)], 1, None, "sin_marcar")
        _pump(v, slow.is_set)
        tab.stop_upload()
        _pump(v, lambda: not tab.uploading)
        assert len(server.sent("POST", "/api/training/clips")) == 1
        assert tab.up_lbl.cget("text") == "Frenado: no se subió nada."
        # un clip demasiado largo para un solo pedido: lo dice antes de mandarlo
        tab.analyze_fn = ia_clips.analyze
        monkey = ei.MAX_BODY
        ei.MAX_BODY = 20  # (comprimido, un clip de 4 s pesa unos cientos de bytes)
        try:
            n = len(server.sent("POST", "/api/training/clips"))
            other2 = make_video(tmp_path / "largo.mp4", seconds=4.0)
            tab.start_upload([str(other2)], 1, None, "sin_marcar")
            _pump(v, lambda: not tab.uploading)
        finally:
            ei.MAX_BODY = monkey
        assert len(server.sent("POST", "/api/training/clips")) == n
        assert "cortalo en partes" in [m[1] for m in off.mb.seen if m[0] == "warning"][-1]
        # sin la IA de esta PC (todavía arrancando): lo dice
        tab.ai_setup = None
        off.agent_of = lambda: None
        tab.start_upload([str(other)], 1, None, "sin_marcar")
        _pump(v, lambda: not tab.uploading)
        assert "todavía está arrancando" in off.mb.seen[-1][1]
    finally:
        tab.close()
        v.close()


def test_un_clip_cuyo_video_esta_en_otra_pc(tmp_path):
    """Lo subió otra persona desde su PC: acá se bajan sus cuadros (recuadros y esqueletos) y se ven sobre fondo gris;
    se puede marcar igual. «Probar con las reglas» necesita el video: lo dice."""
    v, off, tab, server = _app(tmp_path)
    try:
        frames = [{"t": i * 0.2, "w": 1280, "h": 720, "bags": [],
                   "people": [{"id": 5, "box": [400, 100, 600, 700], "kp": FakePose().estimate(None, (400, 100, 600, 700)).tolist()}]}
                  for i in range(50)]
        server.files["/files/" + "c" * 40] = ia_clips.pack({"uuid": "c" * 40}, frames)
        server.clips = [{"id": 9, "uuid": "c" * 40, "name": "de-marcos.mp4", "client_id": 1, "client_name": "Almacén Demo",
                         "camera_id": 11, "camera_name": "Pasillo bebidas", "duration_s": 10.0, "verdict": "sin_marcar",
                         "marks": [], "people": 1, "created_by_name": "marcos", "width": 1280, "height": 720}]
        v.notebook.select(off.tab_train)
        v.root.update()
        assert tab.tree.item("9")["values"][-1] == ""  # no está en esta PC
        tab.tree.selection_set("9")
        v.root.update()
        assert tab.video is None and len(tab.frames) == 50
        assert "el video está en la PC de marcos" in tab.info.cget("text")
        assert (tmp_path / "datos" / "entrenamiento" / ("c" * 40 + ".json.gz")).is_file()  # queda para la próxima
        tab._goto(3.0)
        assert tab._shown_rec["t"] == pytest.approx(3.0)
        _click_person(tab, 5)
        assert tab.sel_track == 5
        tab.try_rules()
        assert "no está en esta PC" in off.mb.seen[-1][1] and "marcos" in off.mb.seen[-1][1]
        # un clip sin los cuadros en la nube: lo dice
        server.clips.append({"id": 10, "uuid": "d" * 40, "name": "sin-datos.mp4", "client_id": 1, "duration_s": 5.0,
                             "verdict": "sin_marcar", "marks": [], "created_by_name": "marcos"})
        tab.load()
        tab.tree.selection_set("10")
        v.root.update()
        assert "sin los cuadros" in tab.info.cget("text") and tab._frames_error
        # ubicar el video en esta PC: si no es el mismo archivo, lo dice
        wrong = tmp_path / "otro.mp4"
        wrong.write_bytes(b"otro")
        tab.link_video(str(wrong))
        assert "no es el de este clip" in off.mb.seen[-1][1]
    finally:
        tab.close()
        v.close()


def test_probar_con_las_reglas_abre_la_ventana_de_la_camara(tmp_path):
    v, off, tab, server = _app(tmp_path)
    try:
        video = make_video(tmp_path / "hurto.mp4", seconds=2.0)
        v.notebook.select(off.tab_train)
        tab.start_upload([str(video)], 1, server.cameras[0], "sin_marcar")
        _pump(v, lambda: not tab.uploading)
        tab.marks = [{"start": 1.5, "end": 1.9, "track_id": 1, "kind": "bolsa"},
                     {"start": 0.4, "end": 1.0, "track_id": 1, "kind": "bolsillo"}]
        tab.dirty = False
        opened = []
        off.tune_dialog = lambda cam, clips=None, mode="vivo", pick=True: opened.append((cam["id"], clips, mode, pick))
        tab.try_rules()
        assert len(opened) == 1
        cam_id, clips, mode, pick = opened[0]
        assert (cam_id, mode, pick) == (11, "clips", False)
        assert clips[0].path == str(video.resolve()) and clips[0].kind == "hurto" and clips[0].marks == [0.4]
        assert off.library().videos(11)[0].kind == "hurto"  # quedó entre los clips de la cámara
        # sin cámara no se puede (las reglas son de cada cámara)
        tab.clip = {**tab.clip, "camera_id": None}
        tab.try_rules()
        assert "no tiene cámara" in off.mb.seen[-1][1]
    finally:
        tab.close()
        v.close()


def test_desde_el_cliente_se_llega_a_la_pestana_con_ese_cliente(tmp_path):
    """El botón «Entrenar modelo…» del cliente se fue: «Más» → «Entrenar su IA de hurtos…» lleva a la pestaña."""
    v, off, tab, server = _app(tmp_path, role="entrenador")
    try:
        labels = [off.cl_menu.entrycget(i, "label") for i in range(off.cl_menu.index("end") + 1)
                  if off.cl_menu.type(i) == "command"]
        assert labels == ["Entrenar su IA de hurtos…"]  # (el operador no borra clientes)
        assert not any(w.winfo_class() == "TButton" and w.cget("text") == "Entrenar modelo…"
                       for w in _walk(off.tab_clients))
        off.cl_tree.selection_set("2")
        v.root.update()
        off.training_dialog()
        v.root.update()
        assert v.notebook.select() == str(off.tab_train) and tab.scope.get() == "Kiosco Sur"
        assert ("GET", "/api/training/clips", {"client_id": 2}) in server.calls
        assert ("GET", "/api/training/ia/summary", {"client_id": 2}) in server.calls
    finally:
        tab.close()
        v.close()
