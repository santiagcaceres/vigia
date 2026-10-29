"""Videos de prueba por cámara («Ajustar…»): quedan guardados en la PC, se vuelven a analizar con un clic, se mide si
la cámara ve los hurtos marcados y sus alertas etiquetadas sirven para entrenar el modelo del cliente."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from agent.biblioteca import Library, evaluate, live_uuid, sample_uuid, suggested_label
from core import synthetic
from core.detectors import MotionDetector
from server.main import Settings, create_app

ADMIN = ("admin", "clave-admin-123")


def alert(ts: float, type_: str = "merodeo", track: int = 1):
    return SimpleNamespace(ts=ts, type=type_, track_id=track, wall=None)


def test_la_lista_queda_guardada_en_la_pc(tmp_path):
    f = tmp_path / "datos" / "videos_prueba.json"
    lib = Library(f)
    a, b = tmp_path / "hurto 1.mp4", tmp_path / "normal.mp4"
    a.write_bytes(b"x")
    b.write_bytes(b"y")
    assert [v.name for v in lib.add(7, [str(a), str(b)])] == ["hurto 1.mp4", "normal.mp4"]
    assert lib.add(7, [str(a)]) == []  # no se repite
    v = lib.videos(7)[0]
    v.kind, v.marks = "hurto", [12.5]
    lib.update(7, v)
    # otra apertura de la app: está todo
    again = Library(f)
    assert [(x.name, x.kind, x.marks) for x in again.videos(7)] == [("hurto 1.mp4", "hurto", [12.5]),
                                                                    ("normal.mp4", "", [])]
    assert again.videos(8) == []
    again.remove(7, str(b))
    assert [x.name for x in Library(f).videos(7)] == ["hurto 1.mp4"]
    assert b.exists()  # sacarlo de la lista no borra el archivo
    b.unlink()
    assert Library(f).videos(7)[0].exists()  # el que sigue en la lista sí existe


def test_cruzar_alertas_con_lo_marcado():
    from agent.biblioteca import TestVideo

    hurto = TestVideo("/v/h.mp4", kind="hurto", marks=[60.0])
    r = evaluate(hurto, [alert(95.0)], {"merodeo": "Merodeo"})  # 35 s después del momento: la vio
    assert r["ok"] is True and r["text"].startswith("✔ vio el hurto (merodeo 1:35)")
    r = evaluate(hurto, [alert(300.0)])
    assert r["ok"] is False and "no vio el hurto de 1:00" in r["text"] and len(r["extra"]) == 1
    dos = TestVideo("/v/h2.mp4", kind="hurto", marks=[10.0, 200.0])
    r = evaluate(dos, [alert(15.0)])
    assert r["ok"] is False and "vio 1 de 2 hurtos" in r["text"] and "3:20" in r["text"]
    normal = TestVideo("/v/n.mp4", kind="normal")
    assert evaluate(normal, [])["ok"] is True
    r = evaluate(normal, [alert(5.0), alert(9.0)])
    assert r["ok"] is False and "2 falsas alarmas" in r["text"]
    assert evaluate(TestVideo("/v/x.mp4"), [alert(1.0)])["ok"] is None  # sin clasificar: no se puede medir
    # etiquetas para entrenar que surgen de lo marcado
    assert suggested_label(hurto, alert(70.0)) == "hurto"
    assert suggested_label(hurto, alert(500.0)) is None
    assert suggested_label(normal, alert(3.0)) == "falso_positivo"
    # identificadores estables: el mismo momento manda la misma muestra (se actualiza, no se repite)
    assert sample_uuid(7, hurto, alert(70.0)) == sample_uuid(7, hurto, alert(70.04))
    assert sample_uuid(7, hurto, alert(70.0)) != sample_uuid(7, hurto, alert(71.0))
    assert live_uuid(7, alert(70.0)).isalnum() and len(live_uuid(7, alert(70.0))) == 32


def test_analizar_la_biblioteca_mide_si_la_camara_ve_el_hurto(tmp_path):
    from agent.ajuste import Tuner

    lib = Library(tmp_path / "videos_prueba.json")
    paths = [str(synthetic.write_video(synthetic.render(n), tmp_path / f"{n}.mp4")) for n in ("hurto", "cliente_paga")]
    lib.add(7, paths)
    h, n = lib.videos(7)
    h.kind, n.kind = "hurto", "normal"
    cfg = {"id": 7, "name": "Góndola 1", "client_name": "Minas", "process_fps": synthetic.FPS, "connection": {},
           "analysis": {"zones": synthetic.STORE_ZONES, "rules": {}}}
    t = Tuner(cfg, MotionDetector(), [h.path, n.path])
    t.run()
    seg_alerts = [[a for a in t.alerts if a.seg == i] for i in range(2)]
    (salida,) = seg_alerts[0]
    h.marks = [max(0.0, salida.ts - 5)]  # el momento del hurto, unos segundos antes de que salga
    assert evaluate(h, seg_alerts[0])["ok"] is True
    assert evaluate(n, seg_alerts[1])["ok"] is True  # el que pagó no da alerta
    assert salida.features and salida.bbox  # lo que hace falta para entrenar
    assert suggested_label(h, salida) == "hurto"


def test_yolo_completo_ve_objetos_sin_cambiar_las_personas():
    """Una sola pasada del modelo: las personas salen idénticas a las de siempre y aparte los objetos."""
    from core.detectors import OnnxYoloDetector

    class Runner:  # salida de YOLO armada a mano: (4 + 80, N)
        device, on_gpu, size, session = "prueba", False, (640, 640), None
        col_of = {c: 4 + c for c in range(80)}

        def run(self, canvas):
            out = np.zeros((84, 100), np.float32)  # como el modelo: más cajas que clases
            out[:4, 0] = (200, 300, 80, 240)  # una persona
            out[4 + 0, 0] = 0.9
            out[4 + 24, 0] = 0.2  # (algo de «mochila» en la misma caja: no le gana a la persona)
            out[:4, 1] = (230, 290, 40, 50)  # una mochila
            out[4 + 24, 1] = 0.8
            out[:4, 2] = (500, 100, 30, 80)  # una botella con poca confianza: no se muestra
            out[4 + 39, 2] = 0.3
            return out

    det = OnnxYoloDetector(session=Runner())
    frame = np.zeros((640, 640, 3), np.uint8)
    people = det.detect(frame)
    same, objects = det.detect_also(frame, {0: "persona", 24: "mochila", 39: "botella"}, 0.45)
    assert [(d.label, d.bbox) for d in people] == [(d.label, d.bbox) for d in same] and len(people) == 1
    assert [(o.label, round(o.conf, 2)) for o in objects] == [("mochila", 0.8)]


@pytest.fixture
def app(tmp_path, db_url):
    return create_app(Settings(data_dir=tmp_path / "data", database_url=db_url, admin_password=ADMIN[1],
                               serve_public=False))


def test_alertas_etiquetadas_van_al_entrenamiento_sin_llegarle_a_nadie(app):
    with TestClient(app) as c:
        c.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]})
        cid = c.post("/api/clients", json={"name": "Minas", "web_username": "minas", "web_password": "clave-minas-1"}
                     ).json()["id"]
        cam = c.post("/api/cameras", json={"client_id": cid, "name": "Góndolas ventana", "brand": "ip",
                                           "host": "dvr.example", "password": "x"}).json()["id"]
        meta = {"camera_id": cam, "uuid": "a" * 32, "type": "salida_sin_pago", "label": "hurto", "description": "salió",
                "features": {"dwell_shelf": 48.0, "otra_cosa": 3}, "track_id": 4, "video_ts": 95.0, "bbox": [1, 2, 3, 4],
                "frame_size": [368, 416], "video": "VIDEO CAMARA 1.mp4"}
        r = c.post("/api/training/samples", data={"meta": json.dumps(meta)},
                   files={"snapshot": ("a.jpg", b"\xff\xd8jpeg", "image/jpeg")})
        assert r.status_code == 201, r.text
        eid = r.json()["id"]
        ev = c.get(f"/api/events/{eid}").json()
        assert ev["source"] == "ajuste" and ev["label"] == "hurto" and ev["alert"] is False
        assert "otra_cosa" not in ev["features"]
        # no se mezcla con las alertas (ni en «Revisar pendientes»): solo se lista si se pide
        assert c.get("/api/events").json()["total"] == 0
        assert c.get("/api/events?source=ajuste").json()["total"] == 1
        assert c.get("/api/stats").json()["total"] == 0
        # mandarla otra vez cambia la etiqueta, no la repite
        r = c.post("/api/training/samples", data={"meta": json.dumps({**meta, "label": "falso_positivo"})},
                   files={"snapshot": ("a.jpg", b"\xff\xd8jpeg", "image/jpeg")})
        assert r.json() == {"id": eid, "updated": True}
        assert c.get(f"/api/events/{eid}").json()["label"] == "falso_positivo"
        summary = c.get("/api/training/summary").json()["clients"]
        assert summary == [{"client_id": cid, "client_name": "Minas", "labels": {"falso_positivo": 1}, "total": 1,
                            "from_tuning": 1}]
        bad = c.post("/api/training/samples", data={"meta": json.dumps({**meta, "uuid": "b" * 32, "label": "x"})},
                     files={"snapshot": ("a.jpg", b"j", "image/jpeg")})
        assert bad.status_code == 422
        # el cliente no la ve nunca
        c.post("/api/logout")
        c.post("/api/login", json={"username": "minas", "password": "clave-minas-1"})
        assert c.get("/api/events").json()["total"] == 0
        assert c.get("/api/events?source=ajuste").json()["total"] == 0
        assert c.get(f"/api/events/{eid}").status_code == 404


def test_videos_del_grabador_con_pixeles_anchos_se_ven_como_en_el_grabador(tmp_path):
    """«1080p lite»: el grabador graba 960x1080 con píxeles 2:1. Sin corregir, las personas quedan flacas."""
    import shutil
    import subprocess

    import cv2

    from agent.ajuste import pixel_aspect, square_pixels
    if not shutil.which("ffmpeg"):
        pytest.skip("sin ffmpeg")
    path = tmp_path / "lite.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=96x108:rate=10", "-t", "1",
                    "-vf", "setsar=2", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    assert ok and pixel_aspect(cap) == 2.0
    assert square_pixels(frame, 2.0).shape[:2] == (108, 192)
    assert pixel_aspect(cv2.VideoCapture(str(synthetic.write_video(synthetic.render("hurto"), tmp_path / "h.mp4")))) is None
    # el de Góndolas ventana (Minas) no declara la forma de los píxeles: se reconoce por el tamaño «1080p lite»
    sin_dato = tmp_path / "lite-sin-dato.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=960x1080:rate=5", "-t",
                    "0.4", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(sin_dato)], check=True)
    assert pixel_aspect(cv2.VideoCapture(str(sin_dato))) == 2.0
