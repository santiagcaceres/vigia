"""Conteo por zona a la vista (pedido del dueño, 2 nov: «cuando guarde las zonas automáticamente se tiene que quedar
marcado y contando cuando una persona entre a ese pasillo o sector»): en «● En vivo», en «🎞 Clips» y en «Ver con la IA»
se ve cuántas personas entraron a cada zona y cuántas hay ahora, con las mismas reglas que las métricas que guardan las
máquinas (visitas de 2 s o más, sin el personal). Y «⚑ Marcar hurto» guarda el pedazo del vivo como clip."""

from datetime import datetime, timedelta

import cv2
import numpy as np

from core import synthetic
from core.detectors import Detection, MotionDetector
from core.metrics import MetricsCollector
from core.rules import parse_zones
from core.tracker import Track

W, H = 1000, 1000
ZONES = parse_zones([
    {"name": "Pasillo bebidas", "type": "aisle", "polygon": [[0, 0.4], [0.5, 0.4], [0.5, 0.8], [0, 0.8]]},
    {"name": "Mostrador", "type": "staff", "polygon": [[0.6, 0.4], [1, 0.4], [1, 0.8], [0.6, 0.8]]},
    {"name": "Puerta", "type": "entrance", "direction": "abajo", "polygon": [[0, 0], [1, 0], [1, 0.3], [0, 0.3]]},
])


def person(tid, x, y, first, ts):
    return Track(tid, (x - 20, y - 80, x + 20, y), "persona", first, ts, hits=10)


def test_cuenta_en_vivo_lo_mismo_que_guardan_las_metricas():
    m = MetricsCollector(ZONES)
    wall0 = datetime(2026, 11, 2, 11, 0, 0)
    seen = []
    for i in range(60):  # 2 cuadros por segundo, 30 s
        ts = i / 2
        tracks = []
        if ts < 12:  # 1: entra por la puerta (de arriba hacia abajo) y va al pasillo
            tracks.append(person(1, 200, 50 + ts * 100 if ts < 4 else 600, 0.0, ts))
        if 2 <= ts < 3:  # 2: pasa 1 s por el pasillo (ruido del detector: no cuenta)
            tracks.append(person(2, 300, 600, 2.0, ts))
        if 5 <= ts < 20:  # 3: empleada en el mostrador (no cuenta como cliente)
            tracks.append(person(3, 800, 600, 5.0, ts))
        m.update(tracks, ts, (W, H), wall0 + timedelta(seconds=ts))
        seen.append(m.live_counts())
    at = {round(i / 2, 1): c for i, c in enumerate(seen)}
    assert at[1.0]["zones"]["Pasillo bebidas"]["visits"] == 0  # todavía en la puerta
    mid = at[6.0]
    assert mid["zones"]["Pasillo bebidas"] == {"type": "aisle", "visits": 1, "now": 1}  # entró y está
    assert mid["entries"] == 1 and mid["exits"] == 0 and mid["door"] is True
    assert at[8.0]["staff"] == 1 and at[8.0]["zones"]["Mostrador"]["now"] == 1  # la empleada no cuenta como visita
    end = seen[-1]
    assert end["zones"]["Pasillo bebidas"]["visits"] == 1 and end["zones"]["Pasillo bebidas"]["now"] == 0
    assert end["since"] == wall0
    # lo mismo que queda en las métricas al cerrar las visitas (lo que mandan las máquinas)
    buckets = m.close()
    zones = buckets[0]["data"]["zones"]
    assert zones["Pasillo bebidas"]["visits"] == end["zones"]["Pasillo bebidas"]["visits"]
    assert buckets[0]["data"]["entries"] == end["entries"] and buckets[0]["data"]["staff"] == end["staff"]


def _tuner(tmp_path, zones=None):
    from agent.ajuste import Tuner

    video = synthetic.write_video(synthetic.render("hurto"), tmp_path / "hurto.mp4")
    t = Tuner({"id": 7, "name": "Góndola 1", "process_fps": synthetic.FPS, "connection": {},
               "analysis": {"zones": zones or synthetic.STORE_ZONES, "rules": {}}}, MotionDetector(), [str(video)])
    t.run()
    return t


def test_el_ajuste_cuenta_por_zona_cuadro_a_cuadro_y_recuenta_con_zonas_nuevas(tmp_path):
    from agent.ajuste import count_text, draw_counts

    t = _tuner(tmp_path)
    first, end = t.bounds()
    start = t.counts_at(first)
    assert all(z["visits"] == 0 for z in start["zones"].values())  # al empezar el video, nadie
    final = t.counts_at()
    assert final["zones"]["gondola"]["visits"] == 1 and final["zones"]["salida"]["visits"] == 1
    assert final["zones"]["caja"]["visits"] == 0 and final["since"] is None  # (un video no tiene hora)
    # mientras está en la góndola: «ahora 1»
    assert any(t.counts_at(i)["zones"]["gondola"]["now"] == 1 for i in range(first, end))
    # se ve en la imagen: debajo del nombre de cada zona
    zone = next(z for z in t.zones if z.name == "gondola")
    assert count_text(zone, final) == "entraron 1"
    plain = np.zeros((360, 640, 3), np.uint8)
    drawn = plain.copy()
    draw_counts(drawn, t.zones, final)
    assert drawn.any() and t.render(end - 1) is not None
    # zonas nuevas (se guardaron en el editor): se recuenta todo lo analizado sin volver a detectar
    t.set_analysis({"zones": [{"name": "Pasillo", "type": "aisle", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}],
                    "rules": {}})
    assert list(t.counts_at()["zones"]) == ["Pasillo"] and t.counts_at()["zones"]["Pasillo"]["visits"] >= 1


def test_marcar_hurto_en_vivo_guarda_el_pedazo_como_clip(tmp_path):
    from agent.ajuste import Tuner, clip_name

    t = _tuner(tmp_path)
    first, end = t.bounds()
    mid = first + (end - first) // 2
    path, mark, dur = t.export_clip(tmp_path / "clips" / clip_name("Minas", "Góndola 1", datetime(2026, 11, 2, 11, 50)),
                                    mid, before=4.0, after=3.0)
    assert path.name == "Minas - Góndola 1 - 2026-11-02 11.50.00 hurto.mp4" and path.is_file()
    assert abs(mark - 4.0) < 0.2 and abs(dur - 7.0) < 0.3
    cap = cv2.VideoCapture(str(path))
    n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    assert abs(n - (dur * t.process_fps + 1)) <= 1 and abs(fps - t.process_fps) < 0.5
    # el clip se vuelve a analizar igual (a los mismos cuadros por segundo)
    again = Tuner({**t.cfg, "analysis": {"zones": synthetic.STORE_ZONES, "rules": {}}}, MotionDetector(), [str(path)])
    again.run()
    assert again.state == "listo" and again.bounds()[1] > 0


def test_ver_con_la_ia_cuenta_por_zona_y_toma_las_zonas_nuevas_sin_reabrir():
    from agent.vista import AIViewWorker
    from core.rules import RuleEngine

    class Det:
        def detect(self, frame):
            return [Detection((280, 300, 360, 340), "persona", 0.9)]  # pies en (320, 340): adentro de la góndola

    cfg = {"id": 7, "name": "Pasillo", "process_fps": 5, "connection": {"brand": "ip", "host": "dvr", "channel": 1},
           "analysis": {"zones": [{"name": "Góndola", "type": "shelf",
                                   "polygon": [[0.25, 0.5], [0.75, 0.5], [0.75, 1], [0.25, 1]]}], "rules": {}}}
    w = AIViewWorker(cfg, Det, None)
    w.detector, w.pose = Det(), None
    w.engine = RuleEngine(w.zones, w.rule_cfg)
    frame = np.full((360, 640, 3), 60, np.uint8)
    wall = datetime(2026, 11, 2, 11, 0, 0)
    for i in range(20):
        w.process(frame, i / 5, wall + timedelta(seconds=i / 5))
    c = w.metrics.live_counts()
    assert c["zones"]["Góndola"]["visits"] == 1 and c["zones"]["Góndola"]["now"] == 1
    # se guardaron zonas nuevas en la app: las toma sin volver a abrir el video y cuenta de cero con ellas
    w.set_analysis({"zones": [{"name": "Caja", "type": "checkout", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}],
                    "rules": {}}, 8)
    assert [z.name for z in w.zones] == ["Caja"] and w.target_fps == 8
    for i in range(20, 35):
        w.process(frame, i / 5, wall + timedelta(seconds=i / 5))
    c = w.metrics.live_counts()
    assert list(c["zones"]) == ["Caja"] and c["zones"]["Caja"]["visits"] == 1
    # zonas con error: siguen las de antes
    w.set_analysis({"zones": [{"name": "x", "type": "nada", "polygon": [[0, 0], [1, 0], [1, 1]]}]})
    assert [z.name for z in w.zones] == ["Caja"]


def test_al_guardar_zonas_la_app_se_las_pasa_a_las_camaras_que_se_miran_con_la_ia():
    import queue
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace

    from agent.__main__ import Agent
    from agent.vista import AIViewWorker, ViewWorker

    got = []

    class FakeAI(AIViewWorker):
        def __init__(self):  # (sin video ni IA: solo se mira qué le llega)
            pass

        def set_analysis(self, analysis, process_fps=None):
            got.append((analysis, process_fps))

    asked = []
    a = Agent.__new__(Agent)
    a.client = SimpleNamespace(view=lambda ids: asked.append(ids) or [{"id": 7, "analysis": {"zones": ["nuevas"]},
                                                                       "process_fps": 5}])
    a.clip_pool = ThreadPoolExecutor(1)
    a._view_updates, a._views_stale = queue.Queue(), False
    a.viewers = {7: FakeAI(), 9: ViewWorker.__new__(ViewWorker)}
    a._force_config = False
    a.refresh_config()  # se guardaron zonas en la app
    a._refresh_views()
    a.clip_pool.shutdown(wait=True)
    assert asked == [[7]]  # solo las que se miran con la IA
    a._refresh_views()
    assert got == [({"zones": ["nuevas"]}, 5)]
