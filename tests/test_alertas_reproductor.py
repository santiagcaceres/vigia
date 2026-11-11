"""El reproductor de «Alertas» (11 nov, el dueño: «cuando miro las alertas, que se muestre con la IA, y en el reproductor
una barra para ver por dónde va y dónde sería el hurto, y opciones de iniciar y pausar el video; sacá la opción de ver en
el reproductor»): agent/alerta_video.py, y el segundo de la alerta que mandan las máquinas."""

import os
import tempfile
import threading
import time
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

import agent.alerta_video as av
from core.clips import clip_fps, video_offset


# ---------------------------------------------------------------- sin ventana
def test_en_que_segundo_del_clip_queda_la_alerta():
    # el MP4 reparte los cuadros parejo: 3 s quietos (1 cuadro por segundo) y después 5 por segundo
    frames = [(float(t), b"") for t in (0, 1, 2)] + [(3 + i * 0.2, b"") for i in range(21)]
    fps = clip_fps(frames)
    assert fps == pytest.approx(23 / 7)
    assert video_offset(frames, 5.0) == round(13 / fps, 2)  # el cuadro 13 (no 5 s: los quietos pasan más rápido)
    assert video_offset(frames, 0.0) == 0.0 and video_offset([], 1.0) is None
    assert video_offset(frames, 99.0) == round(23 / fps, 2)  # (después del último: el final)


def test_donde_esta_la_alerta():
    # la máquina mandó el segundo: exacto
    assert av.alert_offset({"clip_offset_s": 12.4, "video_ts": 900.0}, 60.0) == (12.4, True)
    assert av.alert_offset({"clip_offset_s": 99.0}, 30.0) == (pytest.approx(29.9), True)  # (no más que el clip)
    # alarma o importado: `video_ts` ya es el segundo del clip
    assert av.alert_offset({"source": "alarma", "video_ts": 8.5}, 20.0) == (8.5, True)
    # en vivo, sin el segundo: desde que apareció la persona, con 3 s de margen y nunca menos de 6
    assert av.alert_offset({"source": "vivo", "video_ts": 900.0, "features": {"track_age_s": 20.0}}, 60.0) == (23.0,
                                                                                                                 False)
    assert av.alert_offset({"source": "vivo", "features": {"track_age_s": 1.0}}, 60.0) == (6.0, False)
    assert av.alert_offset({"source": "vivo", "features": {"track_age_s": 900.0}}, 300.0) == (174.0, False)
    assert av.alert_offset({}, None) == (6.0, False)


def _rec(t, box, pid=1, w=640, h=360):
    return {"t": t, "w": w, "h": h, "people": [{"id": pid, "box": box, "kp": None}], "bags": []}


def test_la_ia_corrige_el_momento_con_el_recuadro_de_la_alerta():
    # la persona camina de izquierda a derecha; la alerta la vio a los 7 s (en un cuadro de 960x540)
    recs = [_rec(round(i * 0.2, 1), [20 + i * 8, 100, 80 + i * 8, 300]) for i in range(60)]
    at7 = recs[35]["people"][0]["box"]
    bbox = [v * 1.5 for v in at7]
    assert av.refine_offset(recs, 6.0, bbox, [960, 540]) == pytest.approx(7.0)
    assert av.match_person(recs[35], bbox, [960, 540]) == (1, pytest.approx(1.0))
    # lejos de lo estimado (más de 8 s) o sin recuadro: no se toca
    assert av.refine_offset(recs, 30.0, bbox, [960, 540]) is None
    assert av.refine_offset(recs, 6.0, None, None) is None


def test_el_dibujo_marca_a_la_persona_de_la_alerta():
    img = np.zeros((360, 640, 3), np.uint8)
    rec = _rec(1.0, [100, 50, 200, 300], pid=4)
    zones = [{"name": "Góndola", "type": "shelf", "polygon": [[0.5, 0.5], [0.9, 0.5], [0.9, 0.9]]}]
    av.draw_overlay(img, rec, zones, "Mano al bolsillo", 4)
    assert tuple(img[5, 5]) == (0, 0, 170)  # el cartel «ALERTA» arriba
    assert tuple(img[175, 100]) == av.RED  # la persona de la alerta, en rojo
    img2 = np.zeros((360, 640, 3), np.uint8)
    av.draw_overlay(img2, rec, [], None, None)
    assert tuple(img2[175, 100]) == av.GREEN and not img2[5, 5].any()  # lejos de la alerta: verde y sin cartel


# ---------------------------------------------------------------- la pestaña, con Tk
def _tk():
    tk = pytest.importorskip("tkinter")
    if os.name != "nt" and not os.environ.get("DISPLAY"):
        pytest.skip("sin pantalla (correr con xvfb-run)")
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"sin Tk: {exc}")
    return tk, root


def _walk(w):
    yield w
    for c in w.winfo_children():
        yield from _walk(c)


def _video(path: Path, seconds: float = 10.0, fps: int = 10) -> bytes:
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 360))
    for i in range(int(seconds * fps)):
        img = np.full((360, 640, 3), 60, np.uint8)
        x = 20 + int(i / fps * 40)
        cv2.rectangle(img, (x, 100), (x + 60, 300), (255, 255, 255), -1)
        vw.write(img)
    vw.release()
    return path.read_bytes()


def _pump(root, cond, secs=8.0):
    end = time.time() + secs
    while time.time() < end and not cond():
        root.update()
        time.sleep(0.02)
    root.update()
    return cond()


def test_el_reproductor_de_alertas(tmp_path):
    tk, root = _tk()
    from tkinter import ttk

    import agent.oficina as of
    from agent import estilo
    eid = 900000 + os.getpid() % 90000
    cached = Path(tempfile.gettempdir()) / f"vigia_clip_{eid}.mp4"
    cached.unlink(missing_ok=True)
    try:
        root.geometry("1366x698+0+0")
        estilo.aplicar(root)
        nb = ttk.Notebook(root)
        nb.pack(fill="both", expand=True)
        clip = _video(tmp_path / "c.mp4")
        downloads = []

        def download(url):
            downloads.append(url)
            return clip
        o = of.Oficina.__new__(of.Oficina)
        o.tk, o.ttk = tk, ttk
        o.v = types.SimpleNamespace(root=root, notebook=nb)
        o.clients, o.cameras = [], [{"id": 3, "zones": [{"name": "Pasillo", "type": "aisle",
                                                          "polygon": [[0.1, 0.6], [0.9, 0.6], [0.9, 0.95]]}]}]
        o.api = types.SimpleNamespace(download=download)
        o.bg = types.SimpleNamespace(run=lambda fn, done=None, fail=None: done(fn()) if done else fn())
        o.load_alerts = lambda *a, **k: None
        o.agent_of = None
        o.tab_alerts = ttk.Frame(nb, padding=8)
        nb.add(o.tab_alerts, text="Alertas")
        o._build_alerts()
        nb.select(o.tab_alerts)
        root.update()
        p = o.player
        # la IA doble: lo mismo que ve el cuadro (el rectángulo), de a poco y en otro hilo
        calls = []

        def analyze(path, detector, pose, fps=5.0, detection=None, stop=None, on_frame=None, **kw):
            calls.append(path)
            recs = []
            for i in range(50):
                if stop is not None and stop.is_set():
                    raise RuntimeError("se frenó")
                t = round(i * 0.2, 1)
                x = 20 + int(t * 40)
                rec = _rec(t, [x, 100, x + 60, 300], pid=7)
                recs.append(rec)
                on_frame(rec)
                time.sleep(0.005)
            return {"frames": recs, "duration_s": 10.0}
        p.analyze_fn, p.ai_setup = analyze, lambda: (None, None)

        # sin «Abrir en el reproductor»; ▶ y ±5 s a la vista, y ningún botón se queda con el foco
        btns = [w for w in _walk(o.tab_alerts) if w.winfo_class() == "TButton"]
        texts = [str(b.cget("text")) for b in btns]
        assert "Abrir en el reproductor" not in texts
        assert {"▶ Reproducir", "−5 s", "+5 s", "Avisar al cliente…", "4 · Falsa alarma"} <= set(texts)
        assert all(str(b.cget("takefocus")) == "0" for b in btns)

        # la alerta: sin el segundo exacto (máquina vieja), la persona llevaba 2 s a la vista; la vio a los 7 s
        box7 = [20 + 280, 100, 80 + 280, 300]
        e = {"id": eid, "type": "mano_bolsillo", "client_name": "Minas", "camera_name": "Entrada", "camera_id": 3,
             "occurred_at": "2026-11-10T18:57:00", "description": "Persona #1082: mano al bolsillo", "source": "vivo",
             "video_ts": 4000.0, "features": {"track_age_s": 2.0}, "bbox": box7, "frame_size": [640, 360],
             "clip_url": "/api/events/1/clip", "snapshot_url": None, "label": None}
        o.alerts = [e]
        o.al_tree.insert("", "end", iid=str(eid), values=("", "", "", "", ""))
        o.al_tree.selection_set(str(eid))
        assert _pump(root, lambda: p.reader is not None and p.ai_done)
        assert downloads == ["/api/events/1/clip"] and len(calls) == 1
        # la barra: el largo del clip y la alerta, corregida con lo que vio la IA (6 s estimado → 7 s)
        assert p.duration == pytest.approx(10.0, abs=0.2)
        assert p.offset == pytest.approx(7.0) and p.exact and p.bar.alert == pytest.approx(7.0)
        assert "La alerta, en el 0:07" in str(p.status.cget("text"))
        assert p.pos == pytest.approx(2.0)  # (abre 4 s antes de lo estimado)
        # reproducir y pausar
        o.play_clip()
        assert p.playing and str(p.play_btn.cget("text")) == "❚❚ Pausa"
        _pump(root, lambda: p.pos > 2.6, 3)
        o.play_clip()
        assert not p.playing and str(p.play_btn.cget("text")) == "▶ Reproducir" and p.pos > 2.6
        # clic en la barra, sobre la alerta: el cuadro de ese momento, con el cartel y la persona en rojo
        p.bar._click(types.SimpleNamespace(x=p.bar.x_of(7.0)))
        assert p.pos == pytest.approx(7.0, abs=0.1)
        assert _pump(root, lambda: p.base_t is not None and abs(p.base_t - 7.0) < 0.15)
        assert tuple(p.shown[5, 5]) == (0, 0, 170)
        # las flechas: 5 s para adelante (hasta el final) y para atrás
        ev = lambda keysym: types.SimpleNamespace(char="", keysym=keysym, widget=o.al_tree)  # noqa: E731
        o._on_key(ev("Right"))
        assert p.pos == pytest.approx(p.duration)
        o._on_key(ev("Left"))
        assert p.pos == pytest.approx(p.duration - 5.0)
        # otra alerta y de vuelta: no se vuelve a bajar ni a analizar
        o._alerts_empty()
        assert p.reader is None and p.bar.duration == 0
        o._al_shown = None
        o.show_alert()
        assert _pump(root, lambda: p.reader is not None)
        assert len(downloads) == 1 and len(calls) == 1 and p.ai_done and p.offset == pytest.approx(7.0)
        # entra en la pantalla
        last = max(btns, key=lambda b: b.winfo_rootx() + b.winfo_width())
        assert last.winfo_rootx() + last.winfo_width() <= root.winfo_rootx() + 1366
    finally:
        cached.unlink(missing_ok=True)
        root.destroy()


def test_la_maquina_manda_el_segundo_de_la_alerta(tmp_path):
    """El clip le dice al servidor en qué segundo está la alerta y la app lo recibe con la alerta."""
    from fastapi.testclient import TestClient

    from agent.outbox import Outbox
    from server.main import Settings, create_app
    app = create_app(Settings(data_dir=tmp_path / "data", admin_password="clave-admin-123", serve_public=False))
    with TestClient(app) as c:
        c.post("/api/login", json={"username": "admin", "password": "clave-admin-123"})
        cl = c.post("/api/clients", json={"name": "Minas"}).json()
        ag = c.post("/api/agents", json={"name": "central", "client_id": None}).json()
        cam = c.post("/api/cameras", json={"client_id": cl["id"], "agent_id": ag["id"], "name": "Entrada", "host": "h",
                                           "password": "x", "mode": "vivo"}).json()
        sent = []

        class Client:
            def post_event(self, meta, jpg):
                sent.append(("event", meta))
                c.post("/api/agent/events", headers={"X-Agent-Token": ag["token"]},
                       data={"meta": __import__("json").dumps(meta)},
                       files={"snapshot": ("s.jpg", cv2.imencode(".jpg", np.zeros((8, 8, 3), np.uint8))[1].tobytes(),
                                           "image/jpeg")})

            def put_clip(self, uuid, path, source, offset=None):
                sent.append(("clip", source, offset))
                h = {"X-Agent-Token": ag["token"]}
                up = c.post(f"/api/agent/events/{uuid}/clip-upload", headers=h).json()
                c.put(up["upload_url"], content=Path(path).read_bytes(), headers={"Content-Type": "video/mp4"})
                body = {"source": source, **({"offset_s": offset} if offset is not None else {})}
                assert c.post(f"/api/agent/events/{uuid}/clip-done", headers=h, json=body).status_code == 200
        ob = Outbox(tmp_path / "outbox", Client())
        meta = {"uuid": "a" * 32, "camera_id": cam["id"], "type": "mano_bolsillo", "severity": "alta",
                "description": "x", "track_id": 3, "video_ts": 4000.0, "occurred_at": "2026-11-10T18:57:00",
                "bbox": [1, 2, 3, 4], "frame_size": [640, 360], "features": {}, "alert": True}
        ob.add_event(cam["id"], meta, b"jpg")
        (tmp_path / "outbox" / meta["uuid"] / "clip.mp4").write_bytes(b"mp4")
        ob.clip_done(meta["uuid"], "buffer", 12.4)
        ob.flush_once()
        assert ("clip", "buffer", 12.4) in sent
        ev = c.get("/api/events").json()["items"][0]
        assert ev["clip_offset_s"] == 12.4 and ev["bbox"] == [1, 2, 3, 4] and ev["frame_size"] == [640, 360]
        # una máquina vieja no lo manda: sigue andando, sin el segundo
        assert c.post(f"/api/agent/events/{'a' * 32}/clip-done", headers={"X-Agent-Token": ag["token"]},
                      json={"source": "buffer"}).status_code == 200
        assert c.get("/api/events").json()["items"][0]["clip_offset_s"] == 12.4  # (no lo borra)
        assert c.post(f"/api/agent/events/{'a' * 32}/clip-done", headers={"X-Agent-Token": ag["token"]},
                      json={"source": "buffer", "offset_s": -3}).status_code == 422
