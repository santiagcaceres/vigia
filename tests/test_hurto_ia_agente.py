"""IA de movimientos de hurto en el agente (`analysis["ia_hurto"]`, core/hurto_ia.py): las máquinas cargan la postura
aunque la cámara no tenga gestos y la prenden, cambian o apagan sin cortar el video; «Ver con la IA» y la ventana de la
cámara («● En vivo» / «🎞 Clips») escriben el puntaje sobre cada persona («IA 0,72») y dan alertas de prueba
`hurto_ia` como las de las reglas (que «Probar estos tiempos» recalcula con la postura guardada); las alarmas le pasan
la postura y el análisis completo a `analyze_clip`. Una cámara sin modelo (o un agente viejo) sigue igual.

Con dobles del detector y de la postura (sin modelos ni cámaras) y modelos armados a mano: con el término
independiente alto, toda ventana con postura pasa el umbral; con uno bajo, ninguna."""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime

import numpy as np
import pytest

import agent.worker as worker_mod
from agent.worker import CameraWorker, theft_spec, wants_pose
from core import synthetic
from core.detectors import Detection
from core.gestos import MANO_BOLSILLO
from core.hurto_ia import FEATURE_NAMES, FEATURE_VERSION
from core.rules import HURTO_IA, RuleEngine
from core.tracker import Track
from tests.test_gestos_agente import BBOX, CONN, FakePose, FakeTime, make_worker, pose, provider, run

H, W = 420, 200  # el cuadro de las pruebas de gestos: la persona (BBOX) ocupa 350 px de alto
SHELF = {"name": "Góndola", "type": "shelf", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}


def spec(score: float = 0.99, threshold: float = 0.5, **extra) -> dict:
    """Un modelo como los que arma core/hurto_ia.train, con todos los coeficientes en 0: toda ventana con postura vale
    `score`."""
    n = len(FEATURE_NAMES)
    return {"kind": "ia_hurto", "version": FEATURE_VERSION, "names": list(FEATURE_NAMES), "mean": [0.0] * n,
            "scale": [1.0] * n, "coef": [0.0] * n, "intercept": math.log(score / (1 - score)),
            "threshold": threshold, "window_s": 3.0, "step_s": 1.0, "consecutive": 2, "cooldown_s": 30.0, **extra}


class _Det:
    """Siempre la misma persona (y cuenta las pasadas)."""

    def __init__(self):
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return [Detection(BBOX, "persona", 0.9)]


class _Pose:
    """Postura de prueba: siempre la misma persona de frente (y cuenta las llamadas)."""

    def __init__(self):
        self.calls = 0

    def estimate(self, frame, bbox):
        self.calls += 1
        return pose()


def _video(path, seconds: float = 7.0) -> str:
    path.mkdir(parents=True, exist_ok=True)
    return str(synthetic.write_video([np.zeros((H, W, 3), np.uint8)] * int(seconds * 5), path / "clip.mp4", fps=5))


def _tuner(path, analysis: dict, seconds: float = 7.0, **cfg):
    from agent.ajuste import Tuner

    det, est = _Det(), _Pose()
    t = Tuner({"id": 44, "name": "Pasillo bebidas", "client_name": "Minas", "process_fps": 5, "connection": {},
               "analysis": analysis, **cfg}, det, [_video(path, seconds)], pose=est)
    t.run()  # (en la app corre en otro hilo)
    return t, det, est


# ---------------------------------------------------------------------------------------------- qué cámaras la usan
def test_que_camaras_usan_la_ia_de_hurtos(caplog):
    good = {"zones": [], "ia_hurto": spec()}
    assert theft_spec(good) == good["ia_hurto"] and wants_pose(good)  # mira la postura aunque no tenga gestos
    assert theft_spec(good, metrics_only=True) is None and not wants_pose(good, metrics_only=True)
    assert theft_spec({**good, "metrics_only": True}) is None  # solo métricas: ni alertas ni clips
    assert theft_spec({"zones": []}) is None and theft_spec(None) is None and theft_spec({"ia_hurto": "x"}) is None
    assert not wants_pose({"zones": []})
    assert wants_pose({"gestures": {"enabled": [MANO_BOLSILLO]}})  # (los gestos, como siempre)
    # un modelo de otra versión de los rasgos (de un servidor más nuevo): no se usa y queda dicho una sola vez
    other = {"zones": [], "ia_hurto": spec(version=FEATURE_VERSION + 1)}
    with caplog.at_level(logging.WARNING, logger="vigia.worker"):
        assert [theft_spec(other) for _ in range(4)] == [None] * 4
    assert sum("IA de hurtos" in r.getMessage() for r in caplog.records) == 1
    assert not wants_pose(other)


# ---------------------------------------------------------------------------------------------- máquinas
def test_la_maquina_mira_la_postura_con_la_ia_y_avisa_como_las_reglas(tmp_path, monkeypatch):
    clock = FakeTime()
    monkeypatch.setattr(worker_mod, "time", clock)
    est = FakePose()
    provider(monkeypatch, est)
    w = make_worker(tmp_path, {"zones": [SHELF], "rules": {}, "ia_hurto": spec()})
    assert w.analyzer.pose is None  # (se engancha en el hilo de la cámara, no al armarla)
    run(w, clock, 6.0)
    assert w.analyzer.pose is est and not w.analyzer.gestures and w.analyzer.theft_ai
    assert w.status()["ia_hurto"] is True and w.status()["gestures"] is False
    assert est.calls > 0
    assert w.events == 1  # la misma persona: un aviso (después, enfriamiento)
    meta = json.loads(next(tmp_path.glob("*/meta.json")).read_text(encoding="utf-8"))
    assert meta["type"] == HURTO_IA and meta["severity"] == "alta" and meta["zone"] == "Góndola"
    assert "posible hurto" in meta["description"]  # un indicio para revisar, no una acusación
    assert meta["features"]["ia_score"] > 0.9 and set(FEATURE_NAMES) <= set(meta["features"])
    assert "track_age_s" in meta["features"]  # (y las medidas de siempre, para el filtro de alertas)


def test_prender_cambiar_y_apagar_la_ia_sin_cortar_el_video(tmp_path, monkeypatch, caplog):
    clock = FakeTime()
    monkeypatch.setattr(worker_mod, "time", clock)
    est = FakePose()
    asked = provider(monkeypatch, est)
    plain = {"zones": [SHELF], "rules": {}}
    w = make_worker(tmp_path, plain)
    cam = {"id": 1, "name": "Pasillo", "connection": CONN, "analysis": plain}
    fp = w.fingerprint
    run(w, clock)
    assert w.analyzer.pose is None and asked == [] and w.status()["ia_hurto"] is False  # sin modelo: como siempre

    # se activó un modelo en la app: viaja dentro de `analysis` (no obliga a reconectar) y rige desde el cuadro siguiente
    on = {**plain, "ia_hurto": spec()}
    assert CameraWorker.fingerprint_of({**cam, "analysis": on}) == CameraWorker.fingerprint_of(cam)
    assert CameraWorker.analysis_of({**cam, "analysis": on}) != CameraWorker.analysis_of(cam)
    engine, metrics = w.analyzer.engine, w.metrics
    with caplog.at_level(logging.INFO, logger="vigia.worker"):
        w.update_analysis(on)
        run(w, clock)
    assert w.analyzer.pose is est and w.analyzer.theft_ai and w.fingerprint == fp
    assert any("IA de movimientos de hurto activada" in r.getMessage() for r in caplog.records)
    # solo cambió el modelo: las reglas, el seguimiento y lo medido siguen como venían
    assert w.analyzer.engine is engine and w.metrics is metrics
    first = w.analyzer.theft

    # otro modelo (se entrenó y activó otro): se cambia sin soltar la postura; el mismo de nuevo no reinicia nada
    w.update_analysis({**plain, "ia_hurto": spec(threshold=0.7)})
    run(w, clock, 0.4)
    assert w.analyzer.theft is not first and w.analyzer.theft.threshold == 0.7 and w.analyzer.pose is est
    second = w.analyzer.theft
    w.update_analysis({**plain, "ia_hurto": spec(threshold=0.7), "detection": {"sensitivity": 7}})
    run(w, clock, 0.4)
    assert w.analyzer.theft is second and w.detection.sensitivity == 7

    # zonas nuevas: la IA las toma (mira «frente a la góndola») y sigue con el mismo modelo
    w.update_analysis({"zones": [], "rules": {}, "ia_hurto": spec(threshold=0.7), "detection": {"sensitivity": 7}})
    run(w, clock, 0.4)
    assert w.analyzer.theft is second and w.analyzer.theft.zones == []

    # un modelo que esta versión no puede usar: la IA queda apagada, lo demás sigue
    w.update_analysis({**plain, "ia_hurto": spec(version=FEATURE_VERSION + 1)})
    run(w, clock, 0.4)
    assert w.analyzer.theft is None and w.analyzer.pose is None and w.analyzer.frames > 0

    # con gestos y la IA: apagar la IA no suelta la postura (la siguen usando los gestos)
    both = {**plain, "gestures": {"enabled": [MANO_BOLSILLO]}, "ia_hurto": spec()}
    w.update_analysis(both)
    run(w, clock, 0.4)
    assert w.analyzer.pose is est and w.analyzer.gestures and w.analyzer.theft_ai
    w.update_analysis({k: v for k, v in both.items() if k != "ia_hurto"})
    run(w, clock, 0.4)
    assert w.analyzer.pose is est and w.analyzer.gestures and not w.analyzer.theft_ai
    calls = est.calls
    with caplog.at_level(logging.INFO, logger="vigia.worker"):
        w.update_analysis(plain)  # sin nada que mire la postura: se suelta (no se gasta en mirarla)
        run(w, clock)
    assert w.analyzer.pose is None and est.calls == calls
    assert any("gestos de ocultamiento apagados" in r.getMessage() for r in caplog.records)


def test_camara_de_solo_metricas_no_usa_la_ia(tmp_path, monkeypatch):
    clock = FakeTime()
    monkeypatch.setattr(worker_mod, "time", clock)
    asked = provider(monkeypatch, FakePose())
    w = make_worker(tmp_path, {"zones": [], "ia_hurto": spec()}, metrics_only=True)
    run(w, clock)
    assert w.analyzer.pose is None and not w.analyzer.theft_ai and asked == [] and w.events == 0


def test_las_alarmas_le_pasan_la_postura_y_el_analisis_completo_a_analyze_clip(tmp_path, monkeypatch):
    import agent.alarms as alarms_mod
    from agent.alarms import AlarmRunner

    video = _video(tmp_path / "v")
    est = _Pose()
    monkeypatch.setattr(alarms_mod, "shared_pose", lambda: est)

    class Client:
        def __init__(self, analysis):
            self.job = {"alarm_id": 9, "start": "2026-11-09T10:00:00-03:00", "end": "2026-11-09T10:00:07-03:00",
                        "event_type": "VMD",
                        "camera": {"id": 1, "name": "Pasillo", "client_name": "Minas", "connection": CONN,
                                   "process_fps": 5, "analysis": analysis}}
            self.events, self.done = [], []

        def next_alarm(self):
            job, self.job = self.job, None
            return job

        def post_event(self, meta, jpg):
            self.events.append(meta)

        def put_clip(self, uuid, path, source):
            pass

        def alarm_done(self, aid, result, stats, ev=None):
            self.done.append(result)

        def alarm_fail(self, aid, why):
            raise AssertionError(why)

    def fetch(cam, start, end, out):
        out.write_bytes(open(video, "rb").read())
        return True

    client = Client({"zones": [SHELF], "rules": {}, "ia_hurto": spec()})
    runner = AlarmRunner(client, None, tmp_path / "w", fetch=fetch)
    monkeypatch.setattr(runner, "_detector", lambda name: _Det())
    assert runner.run_once()
    assert client.done == ["con_persona"] and est.calls > 0
    assert client.events[0]["type"] == HURTO_IA and client.events[0]["features"]["ia_score"] > 0.9

    # sin modelo (y sin gestos): ni se pide la postura
    est.calls = 0
    client = Client({"zones": [SHELF], "rules": {}})
    runner = AlarmRunner(client, None, tmp_path / "w2", fetch=fetch)
    monkeypatch.setattr(runner, "_detector", lambda name: _Det())
    assert runner.run_once()
    assert est.calls == 0 and client.events[0]["type"] != HURTO_IA


# ---------------------------------------------------------------------------------------------- ventana de la cámara
def test_la_ventana_de_la_camara_da_alertas_de_prueba_de_la_ia(tmp_path):
    t, det, est = _tuner(tmp_path, {"zones": [SHELF], "rules": {}, "ia_hurto": spec()})
    assert t.state == "listo" and t.ia_on and not t.gestures_on and t.stats()["ia"] is True
    ia = [a for a in t.alerts if a.type == HURTO_IA]
    assert len(ia) == 1 and len(t.alerts) == 1  # sin reglas que salten: solo la IA (y una vez: enfriamiento)
    a = ia[0]
    assert a.features["ia_score"] > 0.9 and set(FEATURE_NAMES) <= set(a.features) and "track_age_s" in a.features
    assert "IA de movimientos" in a.description and a.bbox is not None and a.kept and a.score is None
    smp = t.sample_at(a.index)
    assert smp.posed and smp.tracks[0].kpts is not None  # la postura queda guardada con el cuadro
    tid = a.track_id
    assert t.marks[a.index - t.base][tid]["ia"] == (pytest.approx(0.99, abs=1e-3), True)  # para escribir «IA 0,99»
    assert t.render(a.index) is not None
    assert any("IA de movimientos: puntaje más alto 0,99 (avisa desde 0,50, 2 veces seguidas)" in x
               for x in t.explain(0))

    # «Probar estos tiempos»: se recalcula la IA con la postura guardada, sin volver a correr YOLO ni la postura
    calls = (det.calls, est.calls)
    assert t.set_times({"checkout_min_seconds": 2.0}) is False
    assert [x.type for x in t.alerts] == [HURTO_IA] and (det.calls, est.calls) == calls
    # zonas nuevas: también (la alerta dice en qué zona estaba la persona)
    assert t.alerts[0].features["ia_gondola"] > 0.9
    t.set_analysis({"zones": [], "rules": {}, "ia_hurto": spec()})
    assert [x.type for x in t.alerts] == [HURTO_IA] and t.alerts[0].features["ia_gondola"] == 0.0
    # apagaron el modelo: sin alertas de la IA ni puntajes, sin volver a analizar
    t.set_analysis({"zones": [], "rules": {}})
    assert not t.ia_on and t.alerts == [] and all("ia" not in v for m in t.marks for v in m.values())
    assert (det.calls, est.calls) == calls


def test_la_ventana_muestra_el_puntaje_aunque_no_llegue_al_umbral(tmp_path):
    t, _det, _est = _tuner(tmp_path, {"zones": [], "rules": {}, "ia_hurto": spec(0.72, threshold=0.8)})
    assert t.alerts == []
    shown = [v["ia"] for m in t.marks for v in m.values() if "ia" in v]
    assert shown and all(s == (pytest.approx(0.72, abs=1e-3), False) for s in shown)
    why = " ".join(t.explain(0))
    assert "puntaje más alto 0,72 (avisa desde 0,80" in why
    assert any("La IA de movimientos de hurto también mira sin zonas" in h for h in t.hints())


def test_con_la_ia_activada_despues_pide_volver_a_analizar(tmp_path):
    t, _det, est = _tuner(tmp_path, {"zones": [], "rules": {}})
    assert est.calls == 0 and all(not s.posed for s in t.samples)  # sin gestos ni IA no se mira la postura
    t.set_analysis({"zones": [], "rules": {}, "ia_hurto": spec()})
    assert t.ia_on and t.needs_pose()  # lo analizado no tiene la postura: la ventana lo vuelve a analizar
    assert t.alerts == []
    assert any("sin mirar la postura: para ver la IA de movimientos de hurto" in x for x in t.explain(0))


def test_sin_modelo_o_en_solo_metricas_la_ventana_sigue_igual(tmp_path):
    t, _det, est = _tuner(tmp_path / "a", {"zones": [], "rules": {}, "metrics_only": True, "ia_hurto": spec()})
    assert not t.ia_on and est.calls == 0 and t.alerts == [] and t.stats()["ia"] is False
    t2, _det, est2 = _tuner(tmp_path / "b", {"zones": [], "rules": {},
                                             "ia_hurto": spec(version=FEATURE_VERSION + 1)})
    assert not t2.ia_on and est2.calls == 0 and t2.alerts == []
    assert all("ia" not in v for m in t2.marks for v in m.values())


def test_el_puntaje_se_escribe_sobre_el_recuadro():
    from agent.ajuste import draw_ia, ia_score_text

    assert ia_score_text(0.724) == "IA 0,72"
    tr = Track(3, (100.0, 120.0, 200.0, 330.0), "persona", 0.0, 1.0, hits=10)

    def red_and_white(mark):
        img = np.zeros((360, 640, 3), np.uint8)
        draw_ia(img, [tr], mark)
        top = img[95:118, 100:260].astype(int)  # arriba del recuadro, al lado del número
        red = ((top[..., 2] > 200) & (top[..., 1] < 80) & (top[..., 0] < 80)).sum()
        white = ((top > 200).all(axis=-1)).sum()
        return red, white

    red, white = red_and_white({3: {"ia": (0.81, True)}})
    assert red > 10 and white == 0  # pasa el umbral: en rojo
    red, white = red_and_white({3: {"ia": (0.31, False)}})
    assert red == 0 and white > 10
    assert red_and_white({}) == (0, 0) and red_and_white({3: {"zones": []}}) == (0, 0)


# ---------------------------------------------------------------------------------------------- «Ver con la IA»
CFG = {"id": 7, "name": "Pasillo", "client_name": "Minas", "process_fps": 5,
       "connection": {"brand": "ip", "host": "dvr", "channel": 1},
       "analysis": {"zones": [SHELF], "rules": {}, "ia_hurto": spec()}}


def _viewer(cfg=CFG):
    from agent.vista import AIViewWorker

    det = _Det()
    w = AIViewWorker(cfg, lambda: det, _Pose)
    w.detector, w.pose = w._detector_factory(), w._pose_factory()  # (en la app lo hace su hilo)
    w.engine = RuleEngine(w.zones, w.rule_cfg)
    return w


def test_ver_con_la_ia_muestra_el_puntaje_y_el_cartel_sin_mandar_nada():
    w = _viewer()
    frame = np.full((H, W, 3), 60, np.uint8)
    hot, out = [], None
    for i in range(30):
        out = w.process(frame, i / 5, datetime(2026, 11, 9, 10, 0, i // 5))
        hot += [a for a in w._hot if a.type == HURTO_IA and a not in hot]
    assert len(hot) == 1 and hot[0].track_id == 1 and "IA de movimientos" in hot[0].description
    assert out is not None and w.theft.shown()[1][1] is True  # (el puntaje pasa el umbral: en rojo sobre la persona)
    w.state = "en_linea"
    assert "IA de hurtos" in w.text()
    # sin el modelo de la cámara: como siempre (ni puntajes ni carteles de la IA)
    plain = _viewer({**CFG, "analysis": {"zones": [SHELF], "rules": {}}})
    assert plain.theft is None
    for i in range(30):
        plain.process(frame, i / 5, datetime(2026, 11, 9, 10, 0, i // 5))
        assert not any(a.type == HURTO_IA for a in plain._hot)


def test_ver_con_la_ia_cambia_el_modelo_sin_perder_lo_contado():
    w = _viewer()
    frame = np.full((H, W, 3), 60, np.uint8)
    for i in range(12):
        w.process(frame, i / 5, datetime(2026, 11, 9, 10, 0, 0))
    engine, metrics, theft = w.engine, w.metrics, w.theft
    w.set_analysis({**CFG["analysis"], "ia_hurto": spec(threshold=0.7)})  # se activó otro modelo
    assert w.theft is not theft and w.theft.threshold == 0.7 and w.engine is engine and w.metrics is metrics
    w.set_analysis({k: v for k, v in CFG["analysis"].items() if k != "ia_hurto"})  # se apagó
    assert w.theft is None and w.engine is engine and w.metrics is metrics
    w.set_analysis({**CFG["analysis"], "metrics_only": True})  # solo métricas: nunca
    assert w.theft is None
