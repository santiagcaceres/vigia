"""«Sensibilidad para detectar personas» y «Personas lejos» en el agente (máquinas de análisis y la app): cada cámara
usa el detector compartido con SU detección (`analysis["detection"]`, core/deteccion.py) en todos los caminos (la
máquina, el vivo de las cámaras en modo alarmas, «Ver con la IA», la ventana de la cámara y los videos importados), y
cambiarla rige en caliente, sin volver a abrir el video ni empezar de cero las reglas. Con dobles del detector (sin
modelos ni cámaras)."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

import agent.worker as worker_mod
from agent.worker import CameraWorker, without_detection
from core import synthetic
from core.detectors import Detection, DetectionConfig, TunedDetector, conf_for, tuned
from tests.test_gestos_agente import CONN, FakeCap, FakePool, FakeTime, OnePerson

W, H = 640, 360
BIG = (100.0, 40.0, 180.0, 330.0)  # 290 px de 360: una persona cerca
SMALL = (500.0, 60.0, 512.0, 90.0)  # 30 px de 360 (8 %): una persona al fondo del depósito


class Recorder:
    """Como OnnxYoloDetector (acepta el umbral de personas y el mosaico por llamada): anota con qué se lo llamó y
    devuelve siempre la misma persona. `tunable` = el envoltorio de la cámara le pasa su detección."""

    tunable = True
    name, device, on_gpu, size, conf = "YOLO11n", "procesador", False, (640, 640), 0.35

    def __init__(self, bbox=BIG, score: float = 0.9):
        self.bbox, self.score = bbox, score
        self.calls: list[tuple[str, float, bool]] = []

    def _people(self):
        return [Detection(self.bbox, "persona", self.score)]

    def detect(self, frame, conf=None, far=False):
        self.calls.append(("detect", self.conf if conf is None else conf, far))
        return self._people()

    def detect_also(self, frame, classes, conf, person_conf=None, far=False):
        self.calls.append(("also", self.conf if person_conf is None else person_conf, far))
        return self._people(), []

    def detect_groups(self, frame, groups, person_conf=None, far=False):
        self.calls.append(("groups", self.conf if person_conf is None else person_conf, far))
        return self._people(), [[] for _ in groups]

    def last(self) -> tuple[float, bool]:
        return self.calls[-1][1:]


def make_worker(tmp_path, analysis, det, **extra) -> CameraWorker:
    cfg = {"id": 1, "name": "Depósito", "client_name": "Almacén Demo", "connection": CONN, "analysis": analysis, **extra}
    return CameraWorker(cfg, det, None, worker_mod.Outbox(tmp_path, None), None, FakePool(),
                        {"clip_pre_seconds": 1, "clip_post_seconds": 1, "clip_max_seconds": 5})


def run(w: CameraWorker, clock: FakeTime, seconds: float = 1.0) -> None:
    w._read_loop(FakeCap(clock, int(seconds * 25)))


def test_lo_que_se_usa_del_contrato():
    # (lo de core: si cambia, las pruebas de abajo dejan de tener sentido)
    assert conf_for(5) == 0.35 and conf_for(10) == 0.15 and conf_for(1) == 0.51 and conf_for(9) == 0.19
    assert DetectionConfig.from_dict({"sensitivity": 9, "far": True}).to_dict(compact=True) == {"sensitivity": 9,
                                                                                              "far": True}


def test_la_maquina_usa_la_sensibilidad_de_la_camara_sin_tocar_el_detector_compartido(tmp_path, monkeypatch):
    clock = FakeTime()
    monkeypatch.setattr(worker_mod, "time", clock)
    base = Recorder()
    w = make_worker(tmp_path, {"zones": [], "rules": {}, "detection": {"sensitivity": 9}}, base)
    assert isinstance(w.analyzer.detector, TunedDetector) and w.analyzer.detector.base is base
    assert w.analyzer.detector.conf == 0.19 and w.status()["detection"] == "sensibilidad 9 de 10"
    run(w, clock)
    assert base.calls and all(c[1:] == (0.19, False) for c in base.calls)
    assert base.conf == 0.35  # el compartido no se toca (otras cámaras lo usan con lo suyo)

    # otra cámara con el mismo detector, sin detección guardada: el detector tal cual, con lo de siempre
    other = make_worker(tmp_path / "b", {"zones": [], "rules": {}}, base)
    assert other.analyzer.detector is base and other.status()["detection"] == "normal"
    base.calls.clear()
    run(other, clock)
    assert all(c[1:] == (0.35, False) for c in base.calls)

    # una cámara de solo métricas (el caso del depósito del pedido) también la usa
    m = make_worker(tmp_path / "c", {"zones": [], "rules": {}, "detection": {"far": True}}, base, metrics_only=True)
    base.calls.clear()
    run(m, clock)
    assert m.analyzer.detector.base is base and all(c[1:] == (0.35, True) for c in base.calls)
    assert m.gate.min_fraction == worker_mod.FAR_MOTION_FRACTION  # con «Personas lejos», el filtro de movimiento ve lo chico
    assert other.gate.min_fraction == worker_mod.MOTION_FRACTION


def test_cambiar_la_deteccion_rige_en_caliente_sin_reconectar_ni_empezar_de_cero(tmp_path, monkeypatch):
    clock = FakeTime()
    monkeypatch.setattr(worker_mod, "time", clock)
    base = Recorder()
    zones = [{"name": "Pasillo", "type": "aisle", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}]
    analysis = {"zones": zones, "rules": {}}
    cam = {"id": 1, "name": "Depósito", "connection": CONN, "analysis": analysis}
    w = make_worker(tmp_path, analysis, base)
    fp = w.fingerprint
    run(w, clock)
    assert w.analyzer.detector is base and base.last() == (0.35, False)
    engine, metrics, tracks = w.analyzer.engine, w.metrics, list(w.analyzer.last_tracks)
    assert tracks

    # la detección viaja dentro de `analysis`: no obliga a reconectar, pero sí a aplicar el análisis nuevo
    new = {**analysis, "detection": {"sensitivity": 10, "far": True}}
    assert CameraWorker.fingerprint_of({**cam, "analysis": new}) == CameraWorker.fingerprint_of(cam)
    assert CameraWorker.analysis_of({**cam, "analysis": new}) != CameraWorker.analysis_of(cam)
    assert without_detection(new) == analysis
    w.update_analysis(new)
    run(w, clock, 0.4)
    assert w.fingerprint == fp and isinstance(w.analyzer.detector, TunedDetector)
    assert base.last() == (0.15, True) and w.gate.min_fraction == worker_mod.FAR_MOTION_FRACTION
    # solo cambió la detección: las reglas, lo medido y el seguimiento siguen como venían
    assert w.analyzer.engine is engine and w.metrics is metrics
    assert {t.id for t in w.analyzer.last_tracks} == {t.id for t in tracks}

    # cambiar también las zonas: se rehacen las reglas y se conserva la detección
    w.update_analysis({**new, "zones": []})
    run(w, clock, 0.4)
    assert w.analyzer.engine is not engine and base.last() == (0.15, True)

    # volver a lo de siempre: el detector compartido tal cual (sin envoltorio)
    w.update_analysis({"zones": [], "rules": {}})
    run(w, clock, 0.4)
    assert w.analyzer.detector is base and base.last() == (0.35, False)
    assert w.gate.min_fraction == worker_mod.MOTION_FRACTION


def test_una_deteccion_rara_no_traba_la_camara(tmp_path, monkeypatch):
    """Un servidor más nuevo (o un dato roto): el agente la lee tolerante y sigue con lo de siempre."""
    clock = FakeTime()
    monkeypatch.setattr(worker_mod, "time", clock)
    base = Recorder()
    w = make_worker(tmp_path, {"zones": [], "rules": {},
                               "detection": {"sensitivity": "mucho", "far": "si", "clave_nueva": 1}}, base)
    assert w.analyzer.detector is base
    run(w, clock)
    assert base.last() == (0.35, False)
    w.update_analysis({"zones": [], "rules": {}, "detection": [1, 2]})
    run(w, clock, 0.4)
    assert w.analyzer.detector is base and w.analyzer.frames > 0
    w.update_analysis({"zones": [], "rules": {}, "detection": {"sensitivity": 11, "far": True}})
    run(w, clock, 0.4)
    assert base.last() == (0.35, True)  # la sensibilidad fuera de 1..10 queda en 5; lo demás vale


def test_un_detector_sin_ajuste_queda_igual(tmp_path, monkeypatch):
    """El detector de movimiento (y los viejos) no aceptan sensibilidad: la cámara anda igual que siempre."""
    from core.detectors import MotionDetector

    clock = FakeTime()
    monkeypatch.setattr(worker_mod, "time", clock)
    det = OnePerson()
    w = make_worker(tmp_path, {"zones": [], "rules": {}, "detection": {"sensitivity": 10, "far": True}}, det)
    assert w.analyzer.detector is det
    run(w, clock)
    assert w.analyzer.frames > 0 and w.analyzer.last_tracks
    motion = MotionDetector()
    assert tuned(motion, {"sensitivity": 10, "far": True}) is motion


def test_el_agente_no_reinicia_la_camara_al_cambiar_la_deteccion(tmp_path, monkeypatch):
    from agent.__main__ import Agent

    clock = FakeTime()
    monkeypatch.setattr(worker_mod, "time", clock)
    started = []
    monkeypatch.setattr(CameraWorker, "start", lambda self: started.append(self.id))
    base = Recorder()
    ag = Agent("http://127.0.0.1:9", "token", tmp_path, None)
    ag._detector = lambda name, cam_cfg=None: base
    cam = {"id": 1, "name": "Depósito", "client_name": "Almacén Demo", "connection": CONN, "mode": "vivo",
           "analysis": {"zones": [], "rules": {}}}
    ag.apply_config({"cameras": [cam], "settings": {}, "process_imports": False})
    w = ag.workers[1]
    assert started == [1] and w.analyzer.detector is base
    ag.apply_config({"cameras": [{**cam, "analysis": {**cam["analysis"], "detection": {"sensitivity": 8}}}],
                     "settings": {}, "process_imports": False})
    assert ag.workers[1] is w and started == [1]  # la misma cámara, sin cortar el video
    run(w, clock, 0.4)
    assert base.last() == (conf_for(8), False)
    ag.clip_pool.shutdown(wait=False)


def test_el_vivo_de_las_camaras_en_modo_alarmas_usa_la_deteccion(tmp_path):
    from agent.live import LiveWorker

    base = Recorder()
    cam = {"id": 5, "name": "Depósito", "connection": CONN, "mode": "alarmas_equipo",
           "analysis": {"zones": [], "detection": {"far": True}}}
    lw = LiveWorker(cam, base, None)
    assert isinstance(lw.detector, TunedDetector) and lw.detector.base is base
    lw.detector.detect(np.zeros((H, W, 3), np.uint8))
    assert base.last() == (0.35, True)
    assert lw.set_detection({"zones": [], "detection": {"sensitivity": 3}}) is True
    lw.detector.detect(np.zeros((H, W, 3), np.uint8))
    assert base.last() == (conf_for(3), False) and lw.detector.base is base  # (sin envolver dos veces)
    assert lw.set_detection({"zones": [], "detection": {"sensitivity": 3}}) is False  # lo mismo: nada
    assert lw.set_detection({"zones": []}) is True and lw.detector is base

    # el agente se la pasa al vivo cuando cambia la configuración
    from agent.__main__ import Agent
    ag = Agent("http://127.0.0.1:9", "token", tmp_path, None)
    ag.live_workers[5] = lw
    ag.apply_config({"cameras": [{**cam, "analysis": {"zones": [], "detection": {"sensitivity": 10}}}],
                     "settings": {}, "process_imports": False})
    assert isinstance(lw.detector, TunedDetector) and lw.detector.conf == 0.15
    ag.clip_pool.shutdown(wait=False)
    for lst in ag.listeners.values():
        lst.stop()
    if ag.alarms is not None:  # (con una cámara en modo alarmas arranca la cola de alarmas: que no quede preguntando)
        ag.alarms.stop()


def test_cambiar_la_deteccion_no_corta_la_escucha_de_alarmas_del_grabador(tmp_path, monkeypatch):
    from agent.__main__ import Agent
    from agent.device_alarms import DeviceAlarmListener

    started = []
    monkeypatch.setattr(DeviceAlarmListener, "start", lambda self: started.append(self.id))
    cam = {"id": 6, "name": "Depósito", "client_name": "Almacén Demo", "connection": CONN, "mode": "alarmas_equipo",
           "alarm_cfg": {"event_types": ["VMD"]}, "analysis": {"zones": []}}
    ag = Agent("http://127.0.0.1:9", "token", tmp_path, None)
    try:
        ag.apply_config({"cameras": [cam], "settings": {}, "process_imports": False})
        lst = ag.listeners[6]
        assert started == [6]
        # la detección (y las zonas, el modelo o el nombre) son para analizar el clip: la escucha sigue conectada
        ag.apply_config({"cameras": [{**cam, "name": "Depósito fondo", "filter_model": {"x": 1},
                                      "analysis": {"zones": [], "detection": {"sensitivity": 8, "far": True}}}],
                         "settings": {}, "process_imports": False})
        assert ag.listeners[6] is lst and started == [6]
        assert lst.name == "Almacén Demo · Depósito fondo"
        # otros tipos de alarma sí: se vuelve a conectar con lo nuevo
        ag.apply_config({"cameras": [{**cam, "alarm_cfg": {"event_types": ["VMD", "linedetection"]}}],
                         "settings": {}, "process_imports": False})
        assert ag.listeners[6] is not lst and started == [6, 6]
    finally:
        ag.clip_pool.shutdown(wait=False)
        for lst in ag.listeners.values():
            lst.stop()
        if ag.alarms is not None:
            ag.alarms.stop()


def _ai_worker(det, analysis):
    from types import SimpleNamespace

    from agent.vista import AIViewWorker
    from core.rules import RuleEngine

    cfg = {"id": 7, "name": "Depósito", "client_name": "Almacén Demo", "process_fps": 2,
           "connection": {"brand": "ip", "host": "dvr", "channel": 1}, "analysis": analysis}
    w = AIViewWorker(cfg, lambda: det, None)
    w.hik = SimpleNamespace(brand="ip")
    w.detector, w.pose = det, None  # (en la app lo hace su hilo)
    w.engine = RuleEngine(w.zones, w.rule_cfg)
    return w


def test_ver_con_la_ia_usa_la_deteccion_y_la_cambia_sin_perder_lo_contado():
    base = Recorder()
    zones = [{"name": "Pasillo", "type": "aisle", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}]
    analysis = {"zones": zones, "rules": {}, "detection": {"sensitivity": 8}}
    w = _ai_worker(base, analysis)
    frame = np.full((H, W, 3), 60, np.uint8)
    for i in range(6):
        w.process(frame, i / 2, datetime(2026, 11, 6, 10, 0, i))
    assert base.last() == (conf_for(8), False) and base.conf == 0.35
    metrics, engine = w.metrics, w.engine

    # solo cambió la detección (se guardó otra sensibilidad en la app): rige desde el próximo cuadro, sin empezar de
    # cero el conteo por zona ni las reglas
    w.set_analysis({**analysis, "detection": {"sensitivity": 10, "far": True}})
    assert w.metrics is metrics and w.engine is engine
    w.process(frame, 3.5, datetime(2026, 11, 6, 10, 0, 7))
    assert base.last() == (0.15, True)
    w.state, w.ai_fps, w.target_fps = "en_linea", 2.0, 2.0
    assert "sensibilidad 10" in w.text() and "lejos" in w.text()

    # zonas nuevas: el conteo empieza de cero y la detección sigue
    w.set_analysis({**analysis, "zones": [], "detection": {"far": True}})
    assert w.metrics is not metrics
    w.process(frame, 4.0, datetime(2026, 11, 6, 10, 0, 8))
    assert base.last() == (0.35, True)

    # sin detección: el compartido tal cual
    w.set_analysis({"zones": [], "rules": {}})
    w.process(frame, 4.5, datetime(2026, 11, 6, 10, 0, 9))
    assert w._detector_for(w.detection) is base and base.last() == (0.35, False)


def test_ver_con_la_ia_con_gestos_pasa_la_deteccion_a_los_bolsos():
    from core.gestos import MANO_BOLSA

    base = Recorder()
    w = _ai_worker(base, {"zones": [], "rules": {}, "gestures": {"enabled": [MANO_BOLSA]},
                          "detection": {"sensitivity": 7}})
    w.process(np.full((H, W, 3), 60, np.uint8), 0.0, datetime(2026, 11, 6, 10))
    assert base.calls[-1] == ("groups", conf_for(7), False)


def _video(tmp_path, n: int = 30) -> str:
    return str(synthetic.write_video([np.zeros((H, W, 3), np.uint8)] * n, tmp_path / "clip.mp4", fps=5))


def _tune_cfg(**analysis) -> dict:
    return {"id": 9, "name": "Depósito", "client_name": "Almacén Demo", "process_fps": 5, "connection": {},
            "analysis": {"zones": [], "rules": {}, **analysis}}


def test_la_ventana_de_la_camara_analiza_los_clips_con_la_deteccion_guardada(tmp_path):
    from agent.ajuste import Tuner

    base = Recorder()
    t = Tuner(_tune_cfg(detection={"sensitivity": 9, "far": True}), base, [_video(tmp_path)])
    assert t.detection == DetectionConfig(9, True) and t.base_detector is base
    t.run()  # en la app corre en otro hilo
    assert t.state == "listo" and base.calls and all(c[1:] == (0.19, True) for c in base.calls)
    st = t.stats()
    assert st["sensitivity"] == 9 and st["far"] is True

    # otros tiempos o gestos (set_analysis con lo que ya tenía) no pierden la detección
    t.set_times({"checkout_min_seconds": 20})
    assert t.detection == DetectionConfig(9, True) and t.analysis["detection"] == {"sensitivity": 9, "far": True}


def test_probar_otra_sensibilidad_en_vivo_rige_para_los_cuadros_nuevos(tmp_path):
    from agent.ajuste import Tuner

    base = Recorder(SMALL, 0.4)
    t = Tuner(_tune_cfg(), base)  # sin videos = en vivo (sin arrancarlo: los cuadros se le dan a mano)
    assert t.live and t.detector is base
    frame = np.zeros((H, W, 3), np.uint8)
    wall = datetime(2026, 11, 6, 10)
    for i in range(25):
        t._analyze(frame, i / 5, wall, 0)
    assert base.last() == (0.35, False) and t.stats()["detections"] == 25
    n_samples, alerts = len(t.samples), list(t.alerts)

    assert t.set_detection(DetectionConfig(sensitivity=8)) is True
    assert t.analysis["detection"] == {"sensitivity": 8} and t.detector.base is base
    # «Cómo anda la cámara» vuelve a medir el detector de cero (para ver el efecto); lo analizado queda como estaba
    st = t.stats()
    assert st["detections"] == 0 and st["sensitivity"] == 8 and st["far"] is False
    assert len(t.samples) == n_samples and t.alerts == alerts
    t._analyze(frame, 5.2, wall, 0)
    assert base.last() == (conf_for(8), False) and t.stats()["detections"] == 1
    assert t.set_detection({"sensitivity": 8}) is False  # lo mismo: nada
    assert t.set_detection(None) is True and t.detector is base and "detection" not in t.analysis


def test_los_avisos_sugieren_sensibilidad_o_lejos_si_las_personas_son_chicas(tmp_path):
    from agent.ajuste import Tuner

    video = _video(tmp_path)
    t = Tuner(_tune_cfg(), Recorder(SMALL, 0.4), [video])
    t.run()
    hints = " ".join(t.hints())
    assert "se ven chicas" in hints and "«Sensibilidad» (ahora 5)" in hints and "prender «Personas lejos»" in hints
    assert "1280x720" in hints and "El detector duda" in hints
    assert t.stats()["frame_size"] == (W, H)
    assert "Este clip es de" not in hints  # (un clip chico se ve como en la máquina)

    # con la sensibilidad ya en 10, no sugiere subirla
    t = Tuner(_tune_cfg(detection={"sensitivity": 10}), Recorder(SMALL, 0.4), [video])
    t.run()
    hints = " ".join(t.hints())
    assert "La sensibilidad ya está en 10: probá prender «Personas lejos»" in hints
    assert "Probá subir la «Sensibilidad»" not in hints and "(ahora 10)" not in hints

    # con más sensibilidad la confianza media baja sola: no dice «duda», dice qué mirar
    t = Tuner(_tune_cfg(detection={"sensitivity": 8}), Recorder(SMALL, 0.4), [video])
    t.run()
    hints = " ".join(t.hints())
    assert "El detector duda" not in hints and "acepta personas desde 0.23" in hints
    assert "(ahora 8)" in hints

    # con «Personas lejos» ya prendido, no lo vuelve a sugerir: subir la sensibilidad o mejor imagen
    t = Tuner(_tune_cfg(detection={"far": True}), Recorder(SMALL, 0.6), [video])
    t.run()
    hints = " ".join(t.hints())
    assert "Ya está «Personas lejos»" in hints and "prender «Personas lejos»" not in hints

    t = Tuner(_tune_cfg(detection={"sensitivity": 10, "far": True}), Recorder(SMALL, 0.6), [video])
    t.run()
    assert "todo al máximo" in " ".join(t.hints())

    # personas grandes y seguras: nada de esto
    t = Tuner(_tune_cfg(), Recorder(BIG, 0.9), [video])
    t.run()
    hints = " ".join(t.hints())
    assert "chicas" not in hints and "Sensibilidad" not in hints

    # el caso del pedido (6 nov): un depósito, las que ve ocupan el 14 % del alto con confianza 0.49 (las del fondo no
    # aparecen y no bajan el promedio): el aviso de «duda» también dice qué probar
    t = Tuner(_tune_cfg(), Recorder((300.0, 100.0, 320.0, 150.4), 0.49), [video])
    t.run()
    hints = t.hints()
    duda = next(h for h in hints if h.startswith("El detector duda"))
    assert "confianza media 0.49" in duda and "«Personas lejos»" in duda
    assert "Si no ve a las personas del fondo, probá subir la «Sensibilidad» (ahora 5)" in duda
    assert not any("chicas" in h for h in hints)


def test_un_clip_en_alta_con_lejos_avisa_que_la_maquina_mira_la_secundaria(tmp_path):
    """Los clips de «🎞 Clips» suelen venir en 1920x1080 (bajados del grabador) y la máquina mira la calidad
    secundaria (640x360): con «Personas lejos» acá se ve mucho mejor que lo que va a dar allá. Y en vivo, con una
    imagen de 640x360, «Personas lejos» ayuda poco."""
    from agent.ajuste import Tuner

    big = str(synthetic.write_video([np.zeros((720, 1280, 3), np.uint8)] * 30, tmp_path / "alta.mp4", fps=5))
    cfg = _tune_cfg(detection={"far": True})
    cfg["connection"] = {"detect_stream": "sub"}
    t = Tuner(cfg, Recorder(BIG, 0.9), [big])
    t.run()
    hints = " ".join(t.hints())
    assert "Este clip es de 1280x720 y la máquina mira la calidad secundaria" in hints
    assert "con «Personas lejos» puede no pasar allá" in hints and "Calidad del video" in hints
    cfg["connection"] = {"detect_stream": "main"}  # la máquina mira la principal: es lo mismo
    t = Tuner(cfg, Recorder(BIG, 0.9), [big])
    t.run()
    assert "Este clip es de" not in " ".join(t.hints())
    t = Tuner(_tune_cfg(), Recorder(BIG, 0.9), [big])  # sin tocar la detección y con personas grandes: nada
    t.run()
    assert "Este clip es de" not in " ".join(t.hints())

    # en vivo, con 640x360 y «Personas lejos»
    t = Tuner(_tune_cfg(detection={"far": True}), Recorder(BIG, 0.9))
    assert t.live
    t._analyze(np.zeros((H, W, 3), np.uint8), 0.0, None, 0)
    assert "La imagen llega en 640x360: «Personas lejos» ayuda poco" in t._resolution_hint(t.stats()["frame_size"])


def test_with_detection_arma_lo_que_manda_el_servidor():
    from agent.ajuste import with_detection

    a = {"zones": [], "rules": {}}
    out = with_detection(a, DetectionConfig(8, True))
    assert out == {**a, "detection": {"sensitivity": 8, "far": True}} and "detection" not in a  # (copia nueva)
    assert with_detection(out, {"far": True})["detection"] == {"far": True}
    assert "detection" not in with_detection(out, DetectionConfig())
    assert "detection" not in with_detection(out, None)


def test_los_videos_importados_se_analizan_con_la_deteccion(tmp_path, monkeypatch):
    import agent.imports as imports_mod

    base = Recorder()
    monkeypatch.setattr(imports_mod, "build_detector", lambda name, model_path=None, **kw: base)

    class Client:
        def import_progress(self, iid, frac):
            return True

        def post_event(self, meta, jpg):
            pass

        def put_clip(self, *a):
            pass

    runner = imports_mod.ImportJobRunner(Client(), None, tmp_path / "imports")
    job = {"import_id": 3, "camera": {"id": 9, "process_fps": 5,
                                      "analysis": {"zones": [], "rules": {}, "detection": {"sensitivity": 9}}}}
    from pathlib import Path
    runner._analyze(job, Path(_video(tmp_path)), tmp_path)
    assert base.calls and all(c[1:] == (0.19, False) for c in base.calls)


@pytest.mark.parametrize("analysis", [{}, {"detection": None}, {"detection": {}}])
def test_sin_deteccion_todo_queda_como_antes(tmp_path, monkeypatch, analysis):
    clock = FakeTime()
    monkeypatch.setattr(worker_mod, "time", clock)
    base = Recorder()
    w = make_worker(tmp_path, {"zones": [], "rules": {}, **analysis}, base)
    assert w.analyzer.detector is base and w.gate.min_fraction == worker_mod.MOTION_FRACTION


def test_ver_con_la_ia_dice_que_lejos_hace_mas_trabajo_si_la_pc_no_da_abasto():
    w = _ai_worker(Recorder(), {"zones": [], "rules": {}, "detection": {"far": True}})
    w.state, w.target_fps, w.ai_fps, w.ai_ms = "en_linea", 2.0, 0.5, 900.0
    assert "no da abasto" in w.text() and "con «Personas lejos» hace 5 veces el trabajo" in w.text()
    w.set_analysis({"zones": [], "rules": {}})
    assert "lejos" not in w.text()


def test_con_el_yolo_de_verdad_lejos_corre_el_modelo_5_veces_por_cuadro(tmp_path):
    """Con el modelo real (si está): la ventana de la cámara y la máquina pasan por el envoltorio de la cámara hasta el
    detector de verdad, sin romper nada (bolsos, objetos, gestos) y con el compartido intacto."""
    pytest.importorskip("onnxruntime")
    from pathlib import Path

    from agent.ajuste import Tuner
    from core.detectors import OnnxSession, OnnxYoloDetector
    from core.gestos import MANO_BOLSA

    model = Path(__file__).resolve().parent.parent / "models" / "personas.onnx"
    if not model.exists():
        pytest.skip("sin models/personas.onnx")
    base = OnnxYoloDetector(session=OnnxSession(model, threads=1, max_batch=1))
    video = _video(tmp_path, 10)  # 2 s a 5 cuadros por segundo
    for detection, passes in (({}, 1), ({"far": True, "sensitivity": 9}, 5)):
        base.runner.take_stats()
        # (con un gesto: pasa por `detect_groups`, como en la máquina con bolsos)
        t = Tuner(_tune_cfg(detection=detection, gestures={"enabled": [MANO_BOLSA]}), base, [video], pose=None)
        t._pose_built = True  # (sin postura: acá solo importa el detector)
        t.run()
        assert t.state == "listo", t.error
        frames, _ = base.runner.take_stats()
        assert frames == passes * t.stats()["frames"]
    assert base.conf == 0.35
