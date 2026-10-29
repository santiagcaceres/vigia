from datetime import datetime

import numpy as np
import pytest

from core import synthetic
from core.analyzer import Analyzer
from core.detectors import MotionDetector
from core.geometry import point_in_polygon
from core.rules import (FUERA_DE_HORARIO, MERODEO, SALIDA_SIN_PAGO, ZONA_RESTRINGIDA, RuleConfig, RuleEngine,
                        parse_config, parse_zones)
from core.scorer import FEATURE_NAMES, VECTOR_NAMES, Scorer, vectorize
from core.tracker import Track

SIZE = (synthetic.W, synthetic.H)


def engine(rules=None):
    zones, cfg = parse_config({"zones": synthetic.STORE_ZONES, "rules": rules or {}})
    return RuleEngine(zones, cfg)


def run(eng, name, track_id=1, wall=None, ts0=0.0):
    out = []
    for i, (nx, ny) in enumerate(synthetic.positions(name)):
        ts = ts0 + i / synthetic.FPS
        fx, fy = nx * synthetic.W, ny * synthetic.H
        t = Track(id=track_id, bbox=(fx - 12, fy - 60, fx + 12, fy), label="persona", first_ts=ts0, last_ts=ts, hits=10)
        out += eng.update([t], ts, SIZE, wall)
    return out


def test_polygon():
    assert point_in_polygon(0.5, 0.5, [(0, 0), (1, 0), (1, 1), (0, 1)])
    assert not point_in_polygon(1.5, 0.5, [(0, 0), (1, 0), (1, 1), (0, 1)])


def test_hurto_y_features():
    evs = run(engine(), "hurto")
    assert [e.type for e in evs] == [SALIDA_SIN_PAGO]
    f = evs[0].features
    assert set(f) == set(FEATURE_NAMES)
    assert f["shelf_total_s"] >= 4.0 and f["checkout_total_s"] == 0.0
    assert f["shelf_visits"] == 1 and f["path_len"] > 0.5 and f["has_time"] == 0.0


@pytest.mark.parametrize("name", ["cliente_paga", "mira_rapido", "pasa_de_largo"])
def test_normales_no_alertan(name):
    assert run(engine(), name) == []


def test_restringida_horario_y_reglas_desactivadas():
    assert [e.type for e in run(engine(), "intruso_deposito")] == [ZONA_RESTRINGIDA]
    # el merodeo ya no existe: lo que todavía mandan apps y servidores viejos se ignora, sin error
    assert MERODEO not in [e.type for e in run(engine({"loiter_seconds": 3, "shelf_dwell_seconds": 99}), "hurto")]
    night = run(engine({"opening_hours": {"start": "09:00", "end": "21:00"}}), "pasa_de_largo",
                wall=datetime(2026, 10, 16, 3, 0))
    assert FUERA_DE_HORARIO in [e.type for e in night]
    assert night[0].features["has_time"] == 1.0
    assert run(engine({"enabled_rules": ["merodeo"]}), "hurto") == []  # (sin reglas que quedan: nada)
    assert run(engine({"enabled_rules": ["zona_restringida"]}), "hurto") == []


def test_cooldown():
    eng = engine()
    assert len(run(eng, "hurto", 1, ts0=0)) == 1
    assert run(eng, "hurto", 2, ts0=3) == []
    assert len(run(eng, "hurto", 3, ts0=60)) == 1


@pytest.mark.parametrize("zones,msg", [
    ([{"name": "a", "type": "x", "polygon": [[0, 0], [1, 0], [1, 1]]}], "Tipo"),
    ([{"name": "a", "type": "shelf", "polygon": [[0, 0], [1, 0]]}], "3 puntos"),
    ([{"name": "a", "type": "shelf", "polygon": [[0, 0], [2, 0], [1, 1]]}], "normalizadas"),
])
def test_zonas_invalidas(zones, msg):
    with pytest.raises(ValueError, match=msg):
        parse_zones(zones)


def test_reglas_invalidas():
    with pytest.raises(ValueError):
        RuleConfig.from_dict({"inventada": 1})
    with pytest.raises(ValueError):
        RuleConfig.from_dict({"enabled_rules": ["robo_magico"]})


def test_scorer_coincide_con_sklearn():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, len(VECTOR_NAMES)))
    y = (X[:, 1] + 0.5 * X[:, 3] > 0).astype(int)
    sc = StandardScaler().fit(X)
    lr = LogisticRegression().fit(sc.transform(X), y)
    s = Scorer({"vector_names": list(VECTOR_NAMES), "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
                "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]), "threshold": 0.5})
    feats = {"shelf_total_s": 5.0, "path_len": 0.8, "track_age_s": 12}
    v = vectorize("salida_sin_pago", feats)
    expected = lr.predict_proba(sc.transform(v[None]))[0, 1]
    assert abs(s.score("salida_sin_pago", feats) - expected) < 1e-9


def test_scorer_con_variables_distintas_se_ignora():
    assert Scorer.maybe({"vector_names": ["otra"], "mean": [0], "scale": [1], "coef": [1],
                         "intercept": 0, "threshold": 0.5}) is None
    assert Scorer.maybe(None) is None


def test_analyzer_genera_hallazgo_con_foto_y_clip():
    a = Analyzer({"zones": synthetic.STORE_ZONES}, MotionDetector(), clip_pre=2, clip_post=1)
    found = []
    for i, f in enumerate(synthetic.render("hurto")):
        found += a.process(f, i / synthetic.FPS)
    clips = a.flush()
    assert [x.type for x in found] == ["salida_sin_pago"]
    assert found[0].snapshot_jpg[:2] == b"\xff\xd8" and found[0].alert is True and found[0].score is None
    assert len(clips) == 1 and clips[0].uuid == found[0].uuid and len(clips[0].frames) >= 25


def test_detector_yolo_onnx_encuentra_personas():
    from pathlib import Path

    import cv2

    model = Path(__file__).resolve().parents[1] / "models" / "personas.onnx"
    try:
        import ultralytics

        img_path = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
    except ImportError:
        img_path = None
    if not model.exists() or img_path is None or not img_path.exists():
        pytest.skip("modelo o imagen de prueba no disponibles")
    from core.detectors import OnnxYoloDetector

    dets = OnnxYoloDetector(model).detect(cv2.imread(str(img_path)))
    assert len(dets) == 4  # bus.jpg: cuatro personas
    assert all(d.conf > 0.35 and d.bbox[3] > d.bbox[1] for d in dets)


def test_el_clip_cubre_toda_la_estadia_de_la_persona():
    """El clip arranca cuando apareció la persona (no solo clip_pre antes de la alerta) y sigue
    mientras se la vea, hasta clip_post después de que se fue, con tope clip_max."""
    from core.detectors import Detection
    import numpy as np

    class Fija:
        """Una persona quieta en la góndola los primeros 40 s; después nadie."""
        def __init__(self):
            self.t = 0.0

        def detect(self, frame):
            return [Detection((100, 100, 160, 300), "persona", 0.9)] if self.t < 40 else []

    det = Fija()
    zones = [{"name": "g", "type": "restricted", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}]
    rules = {"restricted_min_seconds": 20, "cooldown_seconds": {"zona_restringida": 100}}
    a = Analyzer({"zones": zones, "rules": rules}, det, clip_pre=2, clip_post=3, clip_max=60)
    frame = np.zeros((360, 640, 3), np.uint8)
    found = []
    for i in range(50 * 5):  # 50 s a 5 fps
        det.t = i / 5
        found += a.process(frame, i / 5)
    clips = a.flush()
    assert [f.type for f in found] == ["zona_restringida"] and 19 <= found[0].ts <= 22
    c = clips[0]
    assert c.start_ts <= 1.0  # arrancó cuando apareció (0 s), no 2 s antes de la alerta (~18 s)
    assert 42 <= c.end_ts <= 44  # siguió hasta 3 s después de que se fue (40 s)
    assert isinstance(c.frames[0][1], bytes) and c.frames[0][1][:2] == b"\xff\xd8"  # buffer en JPEG
    assert sum(1 for _ in c.decoded()) == len(c.frames)


def test_worker_aplica_zonas_nuevas_sin_reconectar():
    """Cambiar solo zonas/reglas no cambia la huella de conexión: se aplican en caliente."""
    from agent.worker import CameraWorker
    base = {"id": 1, "name": "A", "connection": {"host": "h", "channel": 1}, "process_fps": 5,
            "analysis": {"zones": [], "rules": {}, "vehicles": False}}
    otra = {**base, "analysis": {"zones": [{"name": "g", "type": "shelf", "polygon": [[0, 0], [1, 0], [1, 1]]}],
                                 "rules": {"loiter_seconds": 15}, "vehicles": False}}
    assert CameraWorker.fingerprint_of(base) == CameraWorker.fingerprint_of(otra)
    assert CameraWorker.analysis_of(base) != CameraWorker.analysis_of(otra)
    veh = {**base, "analysis": {**base["analysis"], "vehicles": True}}  # vehículos cambia el detector: reconecta
    assert CameraWorker.fingerprint_of(base) != CameraWorker.fingerprint_of(veh)
    nuevo = {**base, "name": "Caja 1", "client_name": "Minas"}  # renombrar no corta el video
    assert CameraWorker.fingerprint_of(base) == CameraWorker.fingerprint_of(nuevo)
    assert CameraWorker.display_name(nuevo) == "Minas · Caja 1"


def test_misma_persona_perdida_unos_segundos_conserva_el_numero():
    import numpy as np

    from core.detectors import Detection
    from core.tracker import CentroidTracker

    frame = np.zeros((360, 640, 3), np.uint8)
    frame[100:260, 300:340] = (0, 0, 200)  # persona de ropa roja en el medio
    frame[100:260, 30:70] = (200, 0, 0)  # otra de azul, lejos
    tr = CentroidTracker()
    roja = Detection((300, 100, 340, 260))
    ts = 0.0
    for _ in range(10):  # 2 s viéndola
        tracks = tr.update([roja], ts, frame)
        ts += 0.2
    pid = tracks[0].id
    for _ in range(25):  # 5 s sin verla (detrás de la góndola)
        tracks = tr.update([], ts, frame)
        ts += 0.2
    assert tracks and not tracks[0].visible  # se la recuerda
    # reaparece un poco corrida: es la misma, con el mismo tiempo en escena
    frame2 = np.zeros_like(frame)
    frame2[100:260, 340:380] = (0, 0, 200)
    tracks = tr.update([Detection((340, 100, 380, 260))], ts, frame2)
    assert tracks[0].id == pid and tracks[0].visible and tracks[0].first_ts == 0.0


def test_quien_entra_por_el_borde_es_nuevo_y_la_ropa_distinta_no_se_confunde():
    import numpy as np

    from core.detectors import Detection
    from core.tracker import CentroidTracker

    frame = np.zeros((360, 640, 3), np.uint8)
    frame[100:260, 300:340] = (0, 0, 200)
    tr = CentroidTracker()
    ts = 0.0
    for _ in range(10):
        tracks = tr.update([Detection((300, 100, 340, 260))], ts, frame)
        ts += 0.2
    pid = tracks[0].id
    for _ in range(15):
        tr.update([], ts, frame)
        ts += 0.2
    # alguien de ropa azul aparece en el mismo lugar: otra persona
    azul = np.zeros_like(frame)
    azul[100:260, 300:340] = (200, 0, 0)
    tracks = tr.update([Detection((300, 100, 340, 260))], ts, azul)
    ts += 0.2
    for _ in range(3):
        tracks = tr.update([Detection((300, 100, 340, 260))], ts, azul)
        ts += 0.2
    assert pid not in [t.id for t in tracks if t.visible]
    # entra alguien por el borde izquierdo: nuevo, aunque la persona perdida siga en memoria
    borde = np.zeros_like(frame)
    borde[100:260, 0:40] = (0, 0, 200)
    for _ in range(4):
        tracks = tr.update([Detection((0, 100, 40, 260))], ts, borde)
        ts += 0.2
    assert all(t.id != pid for t in tracks if t.visible)
