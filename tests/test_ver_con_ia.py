"""«Ver con la IA de esta PC» (pestaña «En vivo» de la app): a la cámara se le aplica lo mismo que el análisis (YOLO,
seguimiento, postura, zonas, gestos) solo para mostrar cómo la capta. No guarda ni manda nada."""

import numpy as np

from core.detectors import Detection
from tests.test_ajuste_gestos import ALL, BBOX, H, POCKET_R, TAKE, W, pose

CFG = {"id": 7, "name": "Entrada", "client_name": "Minas", "process_fps": 5,
       "connection": {"brand": "ip", "host": "dvr", "channel": 1},
       "analysis": {"zones": [{"name": "Góndola", "type": "shelf", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}],
                    "rules": {}, "gestures": {"enabled": ALL}}}


class _Det:
    def __init__(self):
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return [Detection(BBOX, "persona", 0.9)]


class _Pose:
    def __init__(self, frames):
        self.frames = list(frames)

    def estimate(self, frame, bbox):
        return self.frames.pop(0) if self.frames else pose()


def test_dibuja_lo_que_ve_la_ia_sin_guardar_nada():
    from agent.vista import AIViewWorker

    det = _Det()
    w = AIViewWorker(CFG, lambda: det, lambda: _Pose(TAKE + [pose(**POCKET_R)] * 8))
    w.detector, w.pose = w._detector_factory(), w._pose_factory()  # (en la app lo hace su hilo)
    from core.rules import RuleEngine
    w.engine = RuleEngine(w.zones, w.rule_cfg)
    frame = np.full((H, W, 3), 60, np.uint8)
    out = None
    for i in range(len(TAKE) + 10):
        out = w.process(frame, i / 5, __import__("datetime").datetime(2026, 10, 30, 10, 0, i % 60))
    assert out is not None and out.shape[0] > 0 and not np.array_equal(out[: frame.shape[0]], frame[: out.shape[0]])
    assert w.people == 1 and det.calls == len(TAKE) + 10
    # el gesto saltó: se ve el cartel de prueba en la imagen (y no se manda a ningún lado: no hay servidor)
    assert any(a.type == "mano_bolsillo" for a in w._hot)
    assert w.text().startswith("Conectando")  # (sin video todavía: el estado es el de la conexión)


def test_camara_solo_metricas_no_mira_gestos():
    from agent.vista import AIViewWorker

    cfg = {**CFG, "analysis": {**CFG["analysis"], "metrics_only": True}}
    w = AIViewWorker(cfg, lambda: _Det(), None)
    assert not w.rule_cfg.gestures.enabled


def test_la_app_elige_como_abrir_cada_camara():
    from agent.__main__ import Agent
    from agent.vista import AIViewWorker, RemoteView, ViewWorker

    a = Agent.__new__(Agent)
    a.beat, a.server = (lambda cid: {}), "https://x"
    a._detector = lambda name, cfg=None: _Det()
    analyzed = {**CFG, "analyzed": True}
    assert isinstance(a._viewer(analyzed, True), AIViewWorker)  # con la IA: el video del grabador, aunque la analice una máquina
    assert isinstance(a._viewer(analyzed, False), RemoteView)  # sin la IA: la imagen que sube la máquina
    assert type(a._viewer(CFG, False)) is ViewWorker
