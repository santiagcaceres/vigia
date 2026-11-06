"""De punta a punta (pedido del dueño, 6 nov: «¿cómo puedo aumentar la facilidad o tolerancia para detectar personas?
si están lejos no las cuenta»): la «Sensibilidad» y «Personas lejos» que se guardan en una cámara en el servidor viajan
compactas a la máquina (/api/agent/config), a la ventana de la cámara (/api/agent/tune) y a «Ver con la IA»
(/api/agent/view); cada camino arma el detector de la cámara sobre el compartido (conf 0,23 con sensibilidad 8, más el
mosaico) y la gente del fondo que antes no se contaba ahora se cuenta, sin repetir a nadie.

Sin modelos ni cámaras: el «modelo» es el de manchas de tests/test_sensibilidad_core.py (una mancha blanca = persona,
con más puntaje cuanto más grande le llega al modelo, como el YOLO de verdad), sobre un cuadro de 1280x720 de un
«depósito» con 5 personas: una cerca y 4 cada vez más lejos o chicas."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

pytest.importorskip("core.deteccion")

import agent.worker as worker_mod  # noqa: E402
from agent.worker import CameraWorker  # noqa: E402
from core import synthetic  # noqa: E402
from core.clip_analysis import analyze_clip  # noqa: E402
from core.detectors import FAR_PASSES, OnnxYoloDetector, TunedDetector, conf_for, tuned  # noqa: E402
from tests.test_gestos_agente import FakePool, FakeTime  # noqa: E402
from tests.test_sensibilidad_core import BlobRunner  # noqa: E402
from server.main import Settings, create_app  # noqa: E402
from tests.test_sensibilidad_servidor import ADMIN, cam_body, config_of, setup  # noqa: E402

W, H = 1280, 720
# (x1, y1, x2, y2). El modelo ve el cuadro entero a la mitad (640 de 1280) y cada parte de «Personas lejos» a 0,875;
# el puntaje de una mancha es (alto que le llega al modelo / 100)²
BIG = (900, 100, 1020, 400)  # cerca: 0,95 siempre (la ven también 1 o 2 partes: tiene que salir una sola vez)
MID = (300, 100, 350, 210)  # imagen entera 0,30: la ve con sensibilidad 8 (0,23) y no con la de siempre (0,35)
CENTER = (610, 320, 650, 400)  # imagen entera 0,16: solo en las partes (0,49), donde se pisan las 4: una sola vez
FAR_ONE = (100, 600, 140, 680)  # imagen entera 0,16; en su parte 0,49 (abajo a la izquierda)
TINY = (1150, 620, 1180, 680)  # imagen entera 0,09; en su parte 0,28: solo con «Lejos» Y sensibilidad 8
EVERYONE = (BIG, MID, CENTER, FAR_ONE, TINY)
ZONE = [{"name": "Depósito", "type": "aisle", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}]
SETTINGS = {"clip_pre_seconds": 1, "clip_post_seconds": 1, "clip_max_seconds": 5}


@pytest.fixture
def app(tmp_path, db_url):
    return create_app(Settings(data_dir=tmp_path / "data", database_url=db_url, admin_password=ADMIN[1],
                               serve_public=False))


@pytest.fixture
def admin(app):
    with TestClient(app) as c:
        assert c.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]}).status_code == 200
        yield c


def scene(k: int = 0) -> np.ndarray:
    """El depósito, con un cuadrado gris que se mueve (no es una persona: el modelo de manchas solo ve el blanco) para
    que el filtro de movimiento no saltee cuadros, como pasa con gente caminando."""
    img = np.zeros((H, W, 3), np.uint8)
    x = 40 + (k * 37) % 1000
    img[250:330, x:x + 80] = 120
    for x1, y1, x2, y2 in EVERYONE:
        img[y1:y2, x1:x2] = 255
    return img


def near(box, want, tol=6) -> bool:
    return all(abs(a - b) <= tol for a, b in zip(box, want))


def same_people(dets, want=EVERYONE) -> bool:
    """Exactamente esas personas, una vez cada una (en coordenadas del cuadro)."""
    people = [d for d in dets if d.label == "persona"]
    return len(people) == len(want) and all(sum(near(d.bbox, b) for d in people) == 1 for b in want)


class SceneCap:
    """Hace de cv2.VideoCapture: `seconds` de la escena a 25 cuadros por segundo y se corta."""

    def __init__(self, clock: FakeTime, seconds: float):
        self.clock, self.left, self.k = clock, int(seconds * 25), 0

    def grab(self):
        self.clock.t += 0.04
        self.left -= 1
        self.k += 1
        return self.left >= 0

    def retrieve(self):
        return True, scene(self.k)

    def get(self, prop):
        return 0.0

    def release(self):
        pass


def worker_for(tmp_path, cfg: dict, detector) -> CameraWorker:
    return CameraWorker(cfg, detector, None, worker_mod.Outbox(tmp_path, None), None, FakePool(), SETTINGS)


def visits(w: CameraWorker) -> int:
    return w.metrics.live_counts()["zones"]["Depósito"]["visits"]


def test_el_modelo_de_prueba_ve_lo_que_dice_la_escena():
    """(Si esto falla, las pruebas de abajo no prueban nada: la escena está pensada para este «modelo».)"""
    det = OnnxYoloDetector(session=BlobRunner())
    frame = scene()
    assert same_people(det.detect(frame), [BIG])  # lo de siempre: solo el de adelante
    assert same_people(det.detect(frame, conf=conf_for(8)), [BIG, MID])  # más sensible: el del medio también
    assert same_people(det.detect(frame, far=True), [BIG, MID, CENTER, FAR_ONE])  # por partes: los del fondo
    assert same_people(det.detect(frame, conf=conf_for(8), far=True))  # las dos cosas: todos, una vez cada uno


def test_de_punta_a_punta_la_gente_del_fondo_se_cuenta(app, admin, tmp_path, monkeypatch):
    clock = FakeTime()
    monkeypatch.setattr(worker_mod, "time", clock)

    # ---- servidor: la cámara de un depósito de «solo métricas», con sensibilidad 8 y «Personas lejos»
    cid, ag, h, user_h = setup(admin, plans=["comercio"])
    r = admin.post("/api/cameras", json=cam_body(cid, "Depósito fondo", agent_id=ag["id"],
                                                 detection={"sensitivity": 8, "far": True}))
    assert r.status_code == 201, r.text
    cam_id = r.json()["id"]
    other = admin.post("/api/cameras", json=cam_body(cid, "Depósito frente", agent_id=ag["id"], channel=2)).json()
    for c in (cam_id, other["id"]):  # las zonas desde la ventana (sin `detection`: no la toca)
        r = admin.put(f"/api/cameras/{c}/analysis", json={"zones": ZONE, "rules": {}})
        assert r.status_code == 200, r.text
    assert admin.get(f"/api/cameras/{cam_id}").json()["detection"] == {"sensitivity": 8, "far": True}

    want = {"sensitivity": 8, "far": True}
    cfg = config_of(app, h, cam_id)
    assert cfg["analysis"]["detection"] == want and cfg["analysis"]["metrics_only"] is True
    assert "detection" not in config_of(app, h, other["id"])["analysis"]  # la otra, como siempre: no viaja nada
    tune = TestClient(app).post("/api/agent/tune", json={"camera_id": cam_id}, headers=user_h)
    assert tune.status_code == 200 and tune.json()["camera"]["analysis"]["detection"] == want
    view = TestClient(app).post("/api/agent/view", json={"camera_ids": [cam_id]}, headers=user_h)
    assert view.status_code == 200 and view.json()["cameras"][0]["analysis"]["detection"] == want

    # ---- la máquina: un solo detector compartido para las dos cámaras
    runner = BlobRunner()
    shared = OnnxYoloDetector(session=runner)
    w = worker_for(tmp_path / "fondo", cfg, shared)
    plain = worker_for(tmp_path / "frente", config_of(app, h, other["id"]), shared)
    det = w.analyzer.detector
    assert isinstance(det, TunedDetector) and det.base is shared and det.cfg.far is True
    assert det.conf == 0.23 and shared.conf == 0.35  # el compartido no cambia
    assert plain.analyzer.detector is shared  # la de siempre: el detector tal cual
    assert w.status()["detection"] == "sensibilidad 8 de 10 · personas lejos"
    runner.calls = 0
    assert same_people(det.detect(scene()))  # las 5, una vez cada una...
    assert runner.calls == FAR_PASSES  # ...con la imagen entera y sus 4 partes

    # analizan el mismo depósito unos segundos (a 2 cuadros/s, como una cámara de solo métricas)
    w._read_loop(SceneCap(clock, 6))
    plain._read_loop(SceneCap(clock, 6))
    assert w.process_fps == plain.process_fps == 2
    assert visits(w) == 5  # se cuenta a la gente del fondo...
    assert visits(plain) == 1  # ...que con la detección de siempre no se contaba
    assert len(w.analyzer.last_tracks) == 5  # (sin repetidos: el seguimiento tiene 5 personas, no más)

    # ---- la ventana de la cámara («🎞 Clips») con un clip del depósito, con lo que manda /api/agent/tune
    from agent.ajuste import Tuner

    video = str(synthetic.write_video([scene(k) for k in range(30)], tmp_path / "deposito.mp4", fps=5))
    t = Tuner(tune.json()["camera"], shared, [video])
    assert isinstance(t.detector, TunedDetector) and t.detector.conf == 0.23 and t.detector.cfg.far
    t.run()
    st = t.stats()
    assert t.state == "listo" and st["persons"] == 5 and st["sensitivity"] == 8 and st["far"] is True
    t0 = Tuner(tune.json()["camera"] | {"analysis": {"zones": ZONE, "rules": {}}}, shared, [video])
    t0.run()
    assert t0.detector is shared and t0.stats()["persons"] == 1

    # ---- «Ver con la IA» con lo que manda /api/agent/view
    from agent.vista import AIViewWorker
    from core.rules import RuleEngine

    vw = AIViewWorker(view.json()["cameras"][0], lambda: shared, None)
    vw.hik = SimpleNamespace(brand="ip")
    vw.detector, vw.pose = shared, None  # (en la app lo hace su hilo)
    vw.engine = RuleEngine(vw.zones, vw.rule_cfg)
    vdet = vw._detector_for(vw.detection)
    assert isinstance(vdet, TunedDetector) and vdet.conf == 0.23 and vdet.cfg.far
    for k in range(6):
        vw.process(scene(k), k / 2, datetime(2026, 11, 6, 10, 0, k))
    vw.state, vw.ai_fps, vw.target_fps = "en_linea", 2.0, 2.0
    assert vw.people == 5 and "5 personas · sensibilidad 8 · lejos" in vw.text()

    # ---- un clip de alarma de esa cámara, con su configuración (agent/alarms.py → analyze_clip)
    assert analyze_clip(video, cfg["analysis"], shared, process_fps=2).max_people == 5
    assert analyze_clip(video, config_of(app, h, other["id"])["analysis"], shared, process_fps=2).max_people == 1

    # ---- volver a lo de siempre desde la ventana: la máquina lo toma en caliente, sin reconectar
    rev = int(TestClient(app).get("/api/agent/config", headers=h).json()["rev"])
    r = admin.put(f"/api/cameras/{cam_id}/analysis",
                  json={"zones": ZONE, "rules": {}, "detection": {"sensitivity": 5, "far": False}})
    assert r.status_code == 200 and r.json()["detection_default"] is True
    new = config_of(app, h, cam_id)
    assert "detection" not in new["analysis"]
    assert int(TestClient(app).get("/api/agent/config", headers=h).json()["rev"]) > rev  # la máquina se entera ya
    assert CameraWorker.fingerprint_of(new) == w.fingerprint  # no hay que volver a abrir el video
    engine = w.analyzer.engine
    w.update_analysis(new["analysis"])
    w._read_loop(SceneCap(clock, 1))
    assert w.analyzer.detector is shared and w.analyzer.engine is engine  # lo medido y las reglas siguen como venían
    assert tuned(shared, new["analysis"].get("detection")) is shared
