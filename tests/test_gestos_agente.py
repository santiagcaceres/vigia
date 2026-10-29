"""Gestos de ocultamiento en el agente (máquinas de análisis y la app): el `CameraWorker` mira la postura solo si la
cámara tiene gestos activados, prenderlos o apagarlos rige sin cortar el video, y una cámara con configuración
inválida no tira abajo a las demás. Con dobles del detector y de la postura (sin modelos ni cámaras)."""

from __future__ import annotations

import json
import logging

import numpy as np

import agent.worker as worker_mod
from agent.live import gesture_text, hud
from agent.worker import CameraWorker, wants_gestures
from core.detectors import Detection
from core.gestos import MANO_BOLSA, MANO_BOLSILLO, MANO_ROPA

BBOX = (40.0, 50.0, 160.0, 400.0)  # una persona de 350 px de alto (más que el mínimo para mirarle la postura)
ALL = [MANO_BOLSILLO, MANO_ROPA, MANO_BOLSA]
CONN = {"host": "dvr", "brand": "hikvision"}


def pose(rw=(70, 205), re=(72, 150)):
    """Persona de frente con torso de 100 px (como en tests/test_gestos.py); por defecto los brazos cuelgan."""
    kp = np.zeros((17, 3), np.float32)
    pts = {0: (100, 70), 1: (95, 65), 2: (105, 65), 5: (120, 100), 6: (80, 100), 11: (115, 200), 12: (85, 200),
           13: (115, 280), 14: (85, 280), 15: (115, 360), 16: (85, 360), 7: (128, 150), 9: (130, 205),
           8: re, 10: rw}
    for i, (x, y) in pts.items():
        kp[i] = (x, y, 0.9)
    return kp


POCKET_R = dict(rw=(83, 205), re=(70, 160))  # muñeca derecha junto a la cadera derecha, codo doblado


class FakeTime:
    def __init__(self):
        self.t = 1000.0

    def monotonic(self):
        return self.t

    def time(self):
        return self.t


class FakeCap:
    """`frames` cuadros a 25 por segundo (siempre «se mueve»: no los saltea el filtro de movimiento) y se corta."""

    def __init__(self, clock: FakeTime, frames: int):
        self.clock, self.left = clock, frames
        self.rng = np.random.default_rng(frames)

    def grab(self):
        self.clock.t += 0.04
        self.left -= 1
        return self.left >= 0

    def retrieve(self):
        return True, self.rng.integers(0, 255, (420, 200, 3), dtype=np.uint8)

    def get(self, prop):
        return 0.0

    def release(self):
        pass


class OnePerson:
    """Detector de prueba: siempre la misma persona (sin `detect_also`: no da bolsos)."""

    def detect(self, frame):
        return [Detection(BBOX, "persona", 0.9)]


class FakePose:
    """Postura de prueba: siempre la misma pose (o la de `kp`)."""

    def __init__(self, kp=None):
        self.kp = pose() if kp is None else kp
        self.calls = 0

    def estimate(self, frame, bbox):
        self.calls += 1
        return self.kp


class FakePool:
    """En lugar del grupo de hilos de los clips: anota lo que se le pide (los clips no se arman en estas pruebas)."""

    def __init__(self):
        self.jobs = []

    def submit(self, fn, *args):
        self.jobs.append((fn, args))


def provider(monkeypatch, est):
    """`shared_pose` del agente devuelve `est` y cuenta cuántas veces se lo pidieron."""
    asked = []

    def fake():
        asked.append(1)
        return est
    monkeypatch.setattr(worker_mod, "shared_pose", fake)
    return asked


def make_worker(tmp_path, analysis, **extra) -> CameraWorker:
    cfg = {"id": 1, "name": "Pasillo", "client_name": "Minas", "connection": CONN, "analysis": analysis, **extra}
    return CameraWorker(cfg, OnePerson(), None, worker_mod.Outbox(tmp_path, None), None, FakePool(),
                        {"clip_pre_seconds": 1, "clip_post_seconds": 1, "clip_max_seconds": 5})


def run(w: CameraWorker, clock: FakeTime, seconds: float = 2.0) -> None:
    w._read_loop(FakeCap(clock, int(seconds * 25)))


def test_el_worker_mira_la_postura_solo_si_la_camara_tiene_gestos(tmp_path, monkeypatch):
    clock = FakeTime()
    monkeypatch.setattr(worker_mod, "time", clock)
    est = FakePose()
    asked = provider(monkeypatch, est)

    w = make_worker(tmp_path, {"zones": [], "gestures": {"enabled": ALL}})
    assert w.analyzer.pose is None  # se engancha en el hilo de la cámara, no al armarla (no traba la ventana)
    run(w, clock)
    assert w.analyzer.pose is est and w.analyzer.gestures and w.status()["gestures"] is True
    assert est.calls > 0  # se le miró la postura a la persona
    assert any(t.kpts is not None for t in w.analyzer.last_tracks)
    n = len(asked)
    run(w, clock)  # reconectar no vuelve a pedir el estimador ni lo cambia
    assert len(asked) == n and w.analyzer.pose is est

    # sin gestos: ni se pide el estimador ni se mira la postura
    est2 = FakePose()
    asked2 = provider(monkeypatch, est2)
    w2 = make_worker(tmp_path / "b", {"zones": [], "rules": {}})
    run(w2, clock)
    assert w2.analyzer.pose is None and not w2.analyzer.gestures and asked2 == [] and est2.calls == 0
    # con gestos apagados a propósito, igual
    w3 = make_worker(tmp_path / "c", {"zones": [], "gestures": {"enabled": []}})
    run(w3, clock)
    assert w3.analyzer.pose is None and asked2 == []
    # cámara de solo métricas: nunca gestos (no hay alertas ni clips), aunque la configuración los traiga
    w4 = make_worker(tmp_path / "d", {"zones": [], "gestures": {"enabled": ALL}}, metrics_only=True)
    run(w4, clock)
    assert w4.analyzer.pose is None and not w4.analyzer.gestures and asked2 == [] and est2.calls == 0


def test_sin_modelo_de_postura_la_camara_sigue_sin_gestos(tmp_path, monkeypatch):
    clock = FakeTime()
    monkeypatch.setattr(worker_mod, "time", clock)
    provider(monkeypatch, None)  # esta máquina no tiene models/postura.onnx
    w = make_worker(tmp_path, {"zones": [], "gestures": {"enabled": ALL}})
    run(w, clock)
    assert w.analyzer.pose is None and not w.analyzer.gestures
    assert w.analyzer.frames > 0  # el resto del análisis sigue igual


def test_el_estimador_compartido_se_arma_una_sola_vez(monkeypatch, caplog):
    built = []

    def missing(model_path=None):
        built.append(1)
        logging.getLogger("vigia.pose").warning("No está el modelo de postura")
        return None
    monkeypatch.setattr(worker_mod, "build_pose", missing)
    monkeypatch.setattr(worker_mod, "_pose_state", {})
    with caplog.at_level(logging.WARNING):
        assert [worker_mod.shared_pose() for _ in range(5)] == [None] * 5
    assert built == [1]  # una sola vez por proceso: una sola línea en el registro
    assert sum("postura" in r.getMessage() for r in caplog.records) == 1

    def broken(model_path=None):
        raise RuntimeError("modelo roto")
    monkeypatch.setattr(worker_mod, "build_pose", broken)
    monkeypatch.setattr(worker_mod, "_pose_state", {})
    assert worker_mod.shared_pose() is None  # un modelo roto tampoco frena a las cámaras


def test_prender_cambiar_y_apagar_los_gestos_sin_cortar_el_video(tmp_path, monkeypatch):
    clock = FakeTime()
    monkeypatch.setattr(worker_mod, "time", clock)
    est = FakePose()
    provider(monkeypatch, est)
    cam = {"id": 1, "name": "Pasillo", "connection": CONN, "analysis": {"zones": []}}
    w = make_worker(tmp_path, {"zones": []})
    fp = w.fingerprint
    run(w, clock)
    assert w.analyzer.pose is None

    # los gestos viajan dentro de `analysis`: no obligan a reconectar (mismo fingerprint), se aplican en caliente
    on = {"zones": [], "gestures": {"enabled": [MANO_BOLSILLO], "pocket_seconds": 0.6}}
    assert CameraWorker.fingerprint_of({**cam, "analysis": on}) == CameraWorker.fingerprint_of(cam)
    assert CameraWorker.analysis_of({**cam, "analysis": on}) != CameraWorker.analysis_of(cam)
    w.update_analysis(on)
    run(w, clock)
    assert w.analyzer.pose is est and w.analyzer.gestures
    assert w.analyzer.engine.gestures.cfg.enabled == (MANO_BOLSILLO,) and w.fingerprint == fp

    # otros tiempos: rigen enseguida, con la misma postura
    w.update_analysis({"zones": [], "gestures": {"enabled": ALL, "pocket_seconds": 1.5}})
    run(w, clock, 0.4)
    assert w.analyzer.engine.cfg.gestures.pocket_seconds == 1.5 and w.analyzer.pose is est
    assert w.analyzer.engine.gestures.cfg.enabled == tuple(ALL)

    # un gesto o una clave que esta versión no conoce (de un servidor más nuevo): se ignoran, sigue con lo demás
    w.update_analysis({"zones": [], "gestures": {"enabled": ["mano_magica", MANO_ROPA], "pocket_seconds": 1.5,
                                                 "clave_nueva": 1}})
    run(w, clock, 0.4)
    assert w.analyzer.engine.cfg.gestures.enabled == (MANO_ROPA,) and w.analyzer.gestures
    # una configuración rota de verdad (un número en null): sigue con la anterior, sin caerse ni reintentar
    w.update_analysis({"zones": [], "rules": {"loiter_seconds": None}, "gestures": {"enabled": ALL}})
    run(w, clock, 0.4)
    assert w.analyzer.engine.cfg.gestures.enabled == (MANO_ROPA,) and w._pending_analysis is None
    w.update_analysis({"zones": [], "gestures": {"enabled": ALL, "pocket_seconds": 1.5}})
    run(w, clock, 0.4)
    assert w.analyzer.engine.cfg.gestures.pocket_seconds == 1.5 and w.analyzer.gestures

    # apagados: se suelta la postura (no se gasta en mirarla)
    calls = est.calls
    w.update_analysis({"zones": [], "gestures": {"enabled": []}})
    run(w, clock)
    assert w.analyzer.pose is None and not w.analyzer.gestures and w.analyzer.engine.gestures is None
    assert est.calls == calls


def test_el_gesto_llega_como_evento_y_se_ve_en_la_imagen(tmp_path, monkeypatch):
    clock = FakeTime()
    monkeypatch.setattr(worker_mod, "time", clock)
    provider(monkeypatch, FakePose(pose(**POCKET_R)))
    # sin pedir una toma antes (como en un depósito): la mano derecha en el bolsillo alcanza
    w = make_worker(tmp_path, {"zones": [], "gestures": {"enabled": ALL, "require_take": False}})
    run(w, clock, 4.0)
    assert w.events == 1 and w.outbox.pending() == 1
    meta = json.loads(next(tmp_path.glob("*/meta.json")).read_text(encoding="utf-8"))
    assert meta["type"] == MANO_BOLSILLO and meta["camera_id"] == 1
    assert "posible ocultamiento" in meta["description"]  # un indicio para revisar, no una acusación
    assert meta["features"]["gesture_dwell_s"] >= 0.6
    eng = w._gesture_engine()
    assert eng is not None
    tid = w.analyzer.last_tracks[0].id
    text, hot = gesture_text(eng, tid, w._last_ts)
    assert text.startswith("mano der bolsillo") and hot and text.isascii()  # OpenCV no dibuja tildes
    img = np.zeros((420, 200, 3), np.uint8)
    assert hud(img, [], w.analyzer.last_tracks, w._last_ts, "Minas", None, eng).shape == img.shape
    assert gesture_text(None, tid, 0.0) == ("", False)


def test_que_camaras_miran_gestos():
    assert wants_gestures({"gestures": {"enabled": [MANO_ROPA]}})
    assert not wants_gestures({"gestures": {"enabled": []}})  # apagados a propósito
    assert not wants_gestures({"rules": {}}) and not wants_gestures(None)
    assert not wants_gestures({"gestures": {"enabled": ALL}}, metrics_only=True)
    assert not wants_gestures({"gestures": None})


def test_una_camara_mal_configurada_no_tira_abajo_a_las_demas(tmp_path, monkeypatch, caplog):
    from agent.__main__ import Agent

    started = []
    monkeypatch.setattr(CameraWorker, "start", lambda self: started.append(self.id))
    provider(monkeypatch, FakePose())
    ag = Agent("http://127.0.0.1:9", "token", tmp_path, None)
    ag._detector = lambda name, cam_cfg=None: OnePerson()

    def cam(cid, analysis, **extra):
        return {"id": cid, "name": f"Cámara {cid}", "client_name": "Minas", "connection": CONN, "mode": "vivo",
                "analysis": analysis, **extra}
    good = cam(1, {"zones": [], "gestures": {"enabled": ALL}})
    bad_gesture = cam(2, {"zones": [], "rules": {"merodeo_magico": 1}})  # una regla que esta versión no conoce
    bad_zone = cam(3, {"zones": [{"name": "x", "type": "inventada", "polygon": [[0, 0], [1, 0], [1, 1]]}]})
    no_conn = {"id": 4, "name": "Cámara 4", "mode": "vivo", "analysis": {}}  # sin datos de conexión
    cfg = {"cameras": [good, bad_gesture, bad_zone, no_conn], "settings": {}, "process_imports": False}
    with caplog.at_level(logging.ERROR, logger="vigia.agent"):
        ag.apply_config(cfg)  # no explota
        assert set(ag.workers) == {1} and started == [1]
        assert set(ag._broken) == {2, 3, 4}
        assert ag._cam_state(2).startswith("Error: Configuración inválida") and "merodeo_magico" in ag._cam_state(2)
        errors = len([r for r in caplog.records if r.levelno >= logging.ERROR])
        assert errors == 3
        ag.apply_config(cfg)  # la misma configuración: no se reintenta ni se repite el error en el registro
        assert len([r for r in caplog.records if r.levelno >= logging.ERROR]) == errors
        assert set(ag.workers) == {1} and started == [1]

    # la corrigen: arranca; la que dejan de mandar sale de la lista de rotas
    fixed = cam(2, {"zones": [], "gestures": {"enabled": [MANO_BOLSA]}})
    ag.apply_config({"cameras": [good, fixed, bad_zone], "settings": {}, "process_imports": False})
    assert set(ag.workers) == {1, 2} and started == [1, 2] and set(ag._broken) == {3}
    ag.clip_pool.shutdown(wait=False)
