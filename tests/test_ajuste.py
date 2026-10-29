"""«Ajustar cámara»: una cámara se analiza en la PC del usuario (en vivo o con videos) con alertas de prueba que no
salen de la app; los tiempos se prueban y se recalcula todo sin volver a detectar; «Aprobar» la manda a las máquinas."""

import pytest
from fastapi.testclient import TestClient

from core import synthetic
from core.detectors import MotionDetector
from server.main import Settings, create_app

ADMIN = ("admin", "clave-admin-123")


@pytest.fixture
def app(tmp_path, db_url):
    return create_app(Settings(data_dir=tmp_path / "data", database_url=db_url, admin_password=ADMIN[1],
                               serve_public=False))


class Counting(MotionDetector):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return super().detect(frame)


def test_probar_una_camara_con_videos_y_cambiar_tiempos(tmp_path):
    from agent.ajuste import Tuner

    videos = [synthetic.write_video(synthetic.render(n), tmp_path / f"{n}.mp4") for n in ("hurto", "cliente_paga")]
    cfg = {"id": 7, "name": "Góndola 1", "client_name": "Minas", "process_fps": synthetic.FPS, "connection": {},
           "analysis": {"zones": synthetic.STORE_ZONES, "rules": {}}}
    det = Counting()
    t = Tuner(cfg, det, [str(v) for v in videos])
    t.run()  # en la app corre en otro hilo
    assert t.state == "listo" and t.progress == 1.0
    segs = t.segments()
    assert [s["name"] for s in segs] == ["hurto.mp4", "cliente_paga.mp4"]
    assert [a.type for a in segs[0]["alerts"]] == ["salida_sin_pago"]  # se llevó algo sin pasar por caja
    assert segs[1]["alerts"] == []  # pagó: sin alerta

    # otros tiempos: se recalcula todo lo analizado sin volver a detectar
    calls = det.calls
    t.set_times({"loiter_seconds": 3})
    segs = t.segments()
    assert sorted(a.type for a in segs[0]["alerts"]) == ["merodeo", "salida_sin_pago"]
    assert [a.type for a in segs[1]["alerts"]] == ["merodeo"]
    assert det.calls == calls

    # se puede ir al momento de cada alerta: la imagen tiene zonas, recuadros y el cartel
    a = segs[0]["alerts"][-1]
    img = t.render(a.index)
    assert img.shape[:2] == (synthetic.H, synthetic.W)
    assert t.clock_text(a).startswith("hurto.mp4 00:")
    st = t.stats()
    assert st["persons"] >= 2 and st["alerts"] == 3 and st["detections"] > 0
    assert not any("zona" in h.lower() and "sin" in h.lower() for h in t.hints())

    # sin zonas: no hay alertas y lo avisa
    t.set_analysis({"zones": [], "rules": {}})
    assert t.alerts == [] and any("No hay zonas" in h for h in t.hints())
    # sin caja: avisa que cualquiera que salga de una góndola dispara «salida sin pago»
    t.set_analysis({"zones": [z for z in synthetic.STORE_ZONES if z["type"] != "checkout"], "rules": {}})
    assert any("Sin zona «Caja»" in h for h in t.hints())
    assert [a.type for a in t.segments()[1]["alerts"]] == ["salida_sin_pago"]  # el que pagó, ahora da alerta


def test_un_video_que_no_se_puede_abrir(tmp_path):
    from agent.ajuste import Tuner

    bad = tmp_path / "grabacion.dav"
    bad.write_bytes(b"no es video")
    t = Tuner({"id": 1, "name": "x", "process_fps": 5, "connection": {}, "analysis": {}}, MotionDetector(), [str(bad)])
    t.run()
    seg = t.segments()[0]
    assert t.state == "listo" and "MP4" in seg["error"] and seg["alerts"] == []


def test_ajustar_y_aprobar_desde_la_app(app):
    with TestClient(app) as a:
        a.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]})
        cl = a.post("/api/clients", json={"name": "Minas"}).json()
        cam = a.post("/api/cameras", json={"client_id": cl["id"], "name": "Góndola 1", "host": "10.0.0.1",
                                           "profile": "gondola"}).json()
        assert cam["tuning"] == "sin_ajustar" and not cam["analyzed"]
        pc = a.post("/api/agent/session").json()
        h = {"X-Agent-Token": pc["token"]}

        info = a.post("/api/agent/tune", headers=h, json={"camera_id": cam["id"]}).json()
        assert info["camera"]["connection"]["host"] == "10.0.0.1" and info["tuning"] == "sin_ajustar"
        assert info["camera"]["analysis"]["rules"]["enabled_rules"]  # lo que agrega el tipo de cámara
        assert info["raw"] == {"zones": [], "rules": {}, "gestures": None}  # lo guardado, para volver a guardarlo
        # al ajustar en vivo, la PC deja la imagen de la cámara (aunque no la analice)
        r = a.post(f"/api/agent/cameras/{cam['id']}/preview", headers=h,
                   files={"image": ("p.jpg", b"\xff\xd8jpg", "image/jpeg")})
        assert r.status_code == 200
        assert a.get("/api/agent/config", headers=h).json()["cameras"] == []  # la PC no la analiza

        a.put(f"/api/cameras/{cam['id']}/analysis", json={"zones": synthetic.STORE_ZONES, "rules": {"loiter_seconds": 30}})
        res = a.post(f"/api/cameras/{cam['id']}/tuning", json={"status": "aprobada", "notes": "probada con 3 hurtos",
                                                                 "analyze": True}).json()
        assert res["tuning"] == "aprobada" and res["tuned_by"] == "admin" and res["analyzed"]
        assert res["tuning_notes"] == "probada con 3 hurtos" and res["has_preview"]
        assert a.post("/api/agent/tune", headers=h, json={"camera_id": cam["id"]}).json()["raw"]["rules"] == \
            {"loiter_seconds": 30}

        # una PC vieja (sin usuario) o una máquina no ajustan cámaras
        vieja = a.post("/api/agents", json={"name": "PC vieja", "client_id": None}).json()
        assert a.post("/api/agent/tune", headers={"X-Agent-Token": vieja["token"]},
                      json={"camera_id": cam["id"]}).status_code == 403


def test_ajuste_en_vivo_reconecta_y_recuerda_lo_ultimo(tmp_path):
    import time

    from agent.ajuste import Tuner

    video = synthetic.write_video(synthetic.render("hurto"), tmp_path / "camara.mp4")
    cfg = {"id": 3, "name": "Caja", "process_fps": 10, "analysis": {"zones": synthetic.STORE_ZONES},
           "connection": {"brand": "hikvision", "host": "127.0.0.1", "source_url": str(video),
                          "timezone": "America/Montevideo"}}
    t = Tuner(cfg, MotionDetector(), keep_seconds=1.0)
    t.start()
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline and not (t.cuts >= 1 and t.bounds()[1] > 0):
        time.sleep(0.05)
    t.stop()
    t.thread.join(5)
    assert t.state == "detenido" and t.cuts >= 1  # el video «se cortó» (terminó el archivo) y volvió a conectar
    first, end = t.bounds()
    assert end > first and t.sample_at(end - 1).wall is not None  # en vivo cada cuadro tiene la hora real
    assert t.render() is not None and t.stats()["fps"] is not None


def test_el_ajuste_muestra_que_alertas_descartaria_el_modelo_entrenado(tmp_path):
    from agent.ajuste import Tuner
    from core.scorer import VECTOR_NAMES

    n = len(VECTOR_NAMES)
    spec = {"vector_names": list(VECTOR_NAMES), "mean": [0.0] * n, "scale": [1.0] * n, "coef": [0.0] * n,
            "intercept": -5.0, "threshold": 0.5, "id": 3}  # un modelo que descarta todo
    video = synthetic.write_video(synthetic.render("hurto"), tmp_path / "hurto.mp4")
    cfg = {"id": 7, "name": "Góndola", "process_fps": synthetic.FPS, "connection": {}, "filter_model": spec,
           "analysis": {"zones": synthetic.STORE_ZONES, "rules": {}}}
    t = Tuner(cfg, MotionDetector(), [str(video)])
    t.run()
    assert [a.type for a in t.alerts] == ["salida_sin_pago"]
    a = t.alerts[0]
    assert a.score is not None and a.score < 0.01 and not a.kept
    assert t.stats()["dropped"] == 1


def test_probar_clip_explica_por_que_no_dio_alerta(tmp_path):
    """«Probar clip»: además de si dio alerta, qué midió de cada persona contra lo que pide cada regla."""
    from agent.ajuste import Tuner

    video = synthetic.write_video(synthetic.render("hurto"), tmp_path / "hurto.mp4")
    zones = [z for z in synthetic.STORE_ZONES if z["type"] != "exit"]
    t = Tuner({"id": 7, "name": "Góndola 1", "process_fps": synthetic.FPS, "connection": {},
               "analysis": {"zones": zones, "rules": {}}}, MotionDetector(), [str(video)])
    t.run()
    assert t.alerts == []
    why = t.explain(0)
    assert why[0].startswith("Persona #") and "«tomó algo»" in why[0] and "merodeo pide 45 s" in why[0]
    assert any("no tiene Salida" in x for x in why) and any("Para «merodeo»" in x for x in why)
    # con la salida: da la alerta y ya no dice que falta
    t.set_analysis({"zones": synthetic.STORE_ZONES, "rules": {}})
    assert [a.type for a in t.alerts] == ["salida_sin_pago"]
    assert not any("Salida" in x for x in t.explain(0))


def test_aprobar_dice_en_que_maquina_quedo(app):
    with TestClient(app) as a:
        a.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]})
        cl = a.post("/api/clients", json={"name": "Minas"}).json()
        cam = a.post("/api/cameras", json={"client_id": cl["id"], "name": "Pasillo bebidas", "host": "10.0.0.1"}).json()
        body = {"status": "aprobada", "analyze": True}
        res = a.post(f"/api/cameras/{cam['id']}/tuning", json=body).json()
        assert res["analyzed"] and res["analyzer_name"] is None  # ninguna máquina en línea todavía
        key = a.post("/api/analysis/pool-key").json()["key"]
        tok = TestClient(app).post("/api/agent/enroll", json={"key": key, "name": "vm-1", "instance_id": "i-1"}).json()
        TestClient(app, headers={"X-Agent-Token": tok["token"]}).post("/api/agent/status", json={"cameras": []})
        res = a.post(f"/api/cameras/{cam['id']}/tuning", json=body).json()
        assert res["analyzer_name"] == "vm-1"
