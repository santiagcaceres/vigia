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


class _BurstCap:
    """Un video como el de Hik-Connect: llega de golpe (ráfagas de 2 s), a 15 cuadros por segundo."""

    def __init__(self, seconds: float = 2.0, fps: float = 15.0):
        self.n, self.fps, self.i = int(seconds * fps), fps, -1
        self.fails = 0

    def grab(self):
        if self.i + 1 >= self.n:
            self.fails += 1
            return False
        self.i += 1
        return True

    def retrieve(self):
        return True, np.full((H, W, 3), 60, np.uint8)

    def get(self, prop):
        import cv2
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        if prop == cv2.CAP_PROP_POS_MSEC:
            return 1000.0 * self.i / self.fps
        return 0.0


def _ready_worker(det):
    from types import SimpleNamespace

    from agent.vista import AIViewWorker
    from core.rules import RuleEngine

    w = AIViewWorker(CFG, lambda: det, None)
    w.hik = SimpleNamespace(brand="hikconnect")
    w.detector, w.pose = det, None
    w.engine = RuleEngine(w.zones, w.rule_cfg)
    return w


def test_con_hik_connect_en_rafagas_la_ia_analiza_5_cuadros_por_segundo_de_video():
    """En la Dell (30 oct) iba a 0,6 cuadros/s y «personas: 0»: los cuadros se elegían con el reloj de la PC y de
    cada ráfaga quedaba uno; con un cuadro cada 1,6 s el seguimiento no llegaba a confirmar a nadie."""
    det = _Det()
    w = _ready_worker(det)
    w._read(_BurstCap(seconds=2.0))  # 30 cuadros que llegan de golpe
    ts = [t for _f, t, _d in w._q]
    assert len(ts) == 10 and all(abs((b - a) - 0.2) < 0.01 for a, b in zip(ts, ts[1:]))  # cada 0,2 s del video
    dues = [d for _f, _t, d in w._q]
    assert dues == sorted(dues)  # (al conectar, la ráfaga se muestra desde lo más nuevo: ver los del `Pacer`)
    w._drain()
    assert det.calls == 10 and w.people == 1  # ahora sí: la persona queda confirmada y con su recuadro
    assert w.ai_fps > 0 and w.text().startswith("Conectando")  # (el estado de la conexión lo pone `run`)


def test_el_video_se_muestra_a_su_ritmo_aunque_llegue_de_golpe():
    from agent.vista import Pacer

    p = Pacer(delay=0.5)
    img = lambda i: np.full((2, 2, 3), i, np.uint8)  # noqa: E731
    for i in range(10):  # al conectar llega 1 s de video de golpe: se salta a lo más nuevo
        p.put(img(i), p.due(i * 0.1, 100.0))
    assert p.get(100.4) is None and p.get(100.5)[0, 0, 0] == 9
    for i in range(10, 20):  # la ráfaga siguiente llega tarde (la red se atrasó): el colchón se agranda y sigue parejo
        p.put(img(i), p.due(i * 0.1, 102.0))
    assert p.get(102.45)[0, 0, 0] == 9
    assert p.get(102.5)[0, 0, 0] == 10
    assert p.get(102.85)[0, 0, 0] == 13
    for i in range(20, 30):  # la otra llega a tiempo (con el colchón nuevo): ni salta ni se acumula atraso
        p.put(img(i), p.due(i * 0.1, 103.0))
    assert p.get(103.45)[0, 0, 0] == 19 and p.get(103.55)[0, 0, 0] == 20 and p.get(104.45)[0, 0, 0] == 29


def test_al_conectar_salta_a_lo_mas_reciente():
    """Al abrir, Hik-Connect manda varios segundos viejos de golpe: no hay que mostrarlos con atraso."""
    from agent.vista import Pacer

    p = Pacer(delay=0.5)
    for i in range(80):  # 8 s de video que llegan todos en el segundo 100
        p.put(np.full((2, 2, 3), i, np.uint8), p.due(i * 0.1, 100.0))
    assert p.get(100.5)[0, 0, 0] == 79


def test_si_la_pc_no_da_abasto_saltea_los_cuadros_viejos():
    import time

    det = _Det()
    w = _ready_worker(det)
    now = time.monotonic()
    for i in range(8):  # 8 cuadros que tenían que mostrarse hace rato
        w._q.append((np.full((H, W, 3), 60, np.uint8), i * 0.2, now - 10 + i * 0.2))
    w._drain()
    assert det.calls == 1  # solo el último: no se queda atrasada analizando lo viejo
