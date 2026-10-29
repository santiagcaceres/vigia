"""Gestos de ocultamiento (core/gestos.py): toma + mano al bolsillo, dentro de la ropa o en la bolsa, con poses
armadas a mano (persona de frente, torso de 100 px)."""

import numpy as np
import pytest

from core.analyzer import Analyzer
from core.detectors import Detection
from core.gestos import (MANO_BOLSA, MANO_BOLSILLO, MANO_ROPA, GestureConfig, GestureEngine, body_of)
from core.rules import EVENT_TYPES, RULE_PRIORITY, SEVERITY, RuleConfig, RuleEngine, parse_config
from core.scorer import FEATURE_NAMES, VECTOR_NAMES, Scorer
from core.tracker import Track

BBOX = (40.0, 50.0, 160.0, 400.0)  # 350 px de alto
ALL = (MANO_BOLSILLO, MANO_ROPA, MANO_BOLSA)


def pose(lw=None, rw=None, le=None, re=None, conf=0.9, wconf=None):
    """Persona de frente: hombros (80,100)-(120,100), caderas (85,200)-(115,200) → torso de 100 px. Por defecto los
    brazos cuelgan. lw/rw: muñeca izquierda/derecha (de la persona: la izquierda queda a la derecha de la imagen)."""
    kp = np.zeros((17, 3), np.float32)
    pts = {0: (100, 70), 1: (95, 65), 2: (105, 65), 5: (120, 100), 6: (80, 100), 11: (115, 200), 12: (85, 200),
           13: (115, 280), 14: (85, 280), 15: (115, 360), 16: (85, 360),
           7: le or (128, 150), 8: re or (72, 150), 9: lw or (130, 205), 10: rw or (70, 205)}
    for i, (x, y) in pts.items():
        kp[i] = (x, y, conf)
    if wconf is not None:
        kp[9, 2] = kp[10, 2] = wconf
    return kp


REACH_R = dict(rw=(-10, 120), re=(30, 112))  # brazo derecho estirado hacia el costado, a la altura del pecho
POCKET_R = dict(rw=(83, 205), re=(70, 160))  # muñeca derecha junto a la cadera derecha, codo doblado
POCKET_L = dict(lw=(117, 205), le=(130, 160))


def feed(eng, frames, t0=0.0, fps=5.0, tid=1, **kw):
    hits, ts = [], t0
    for kp in frames:
        hits += eng.step(tid, kp, BBOX, ts, near_shelf=kw.get("near_shelf", True),
                         at_checkout=kw.get("at_checkout", False), staff=kw.get("staff", False), bags=kw.get("bags"))
        ts += 1.0 / fps
    return hits, ts


def take_then(eng, after: dict, n_after: int, **kw):
    """Alcance al estante (3 cuadros), vuelve al cuerpo, y después `after` durante n_after cuadros."""
    frames = [pose(**REACH_R)] * 3 + [pose(rw=(95, 150), re=(70, 140))] + [pose(**after)] * n_after
    return feed(eng, frames, **kw)


def test_cuerpo_en_largos_de_torso():
    b = body_of(pose(), BBOX)
    assert b is not None and b.T == pytest.approx(100.0, abs=0.5)
    along, across = b.local((100, 200))  # mitad de las caderas
    assert along == pytest.approx(1.0, abs=0.01) and across == pytest.approx(0.0, abs=0.01)
    assert body_of(pose(conf=0.2), BBOX) is None  # sin hombros ni caderas a la vista: no se mira


def test_toma_y_mano_al_bolsillo():
    eng = GestureEngine(GestureConfig(enabled=ALL))
    hits, ts = take_then(eng, POCKET_R, 6)
    assert [h.type for h in hits] == [MANO_BOLSILLO]
    h = hits[0]
    assert h.side == "der" and h.dwell >= 0.59 and h.since_take is not None and h.since_take < 1.5
    assert "bolsillo" in h.description and "tomar algo" in h.description
    assert h.features["takes"] == 1 and h.features["gesture_dwell_s"] >= 0.6
    # la misma persona sigue con la mano ahí: no se repite (enfriamiento)
    more, _ = feed(eng, [pose(**POCKET_R)] * 10, t0=ts)
    assert more == []


def test_bolsillo_sin_toma_no_avisa():
    eng = GestureEngine(GestureConfig(enabled=ALL))
    hits, _ = feed(eng, [pose(**POCKET_R)] * 10)
    assert hits == []
    assert eng.summary(1)["suppressed"].get("sin toma vista antes")
    # si se pide sin toma (cámara de depósito, por ejemplo), avisa
    eng2 = GestureEngine(GestureConfig(enabled=ALL, require_take=False))
    hits2, _ = feed(eng2, [pose(**POCKET_R)] * 10)
    assert [h.type for h in hits2] == [MANO_BOLSILLO] and "no se vio que tomara" in hits2[0].description


def test_descartes_caja_personal_y_dos_manos():
    for kw, why in (({"at_checkout": True}, "en la caja (pagar)"), ({"staff": True}, "zona de personal")):
        eng = GestureEngine(GestureConfig(enabled=ALL))
        hits, _ = take_then(eng, POCKET_R, 6, **kw)
        assert hits == [] and eng.summary(1)["suppressed"].get(why), why
    eng = GestureEngine(GestureConfig(enabled=ALL))
    hits, _ = take_then(eng, {**POCKET_R, **POCKET_L}, 6)
    assert hits == [] and eng.summary(1)["suppressed"].get("las dos manos a la vez (acomodarse)")


def test_toma_lejos_de_la_gondola_no_cuenta():
    eng = GestureEngine(GestureConfig(enabled=ALL))
    hits, _ = take_then(eng, POCKET_R, 6, near_shelf=False)
    assert hits == [] and eng.summary(1)["takes"] == 0


def test_toma_vieja_no_cuenta():
    eng = GestureEngine(GestureConfig(enabled=ALL, take_window=10.0))
    _, ts = feed(eng, [pose(**REACH_R)] * 3 + [pose(rw=(95, 150), re=(70, 140))])
    _, ts = feed(eng, [pose()] * 60, t0=ts)  # 12 s con los brazos colgando
    hits, _ = feed(eng, [pose(**POCKET_R)] * 6, t0=ts)
    assert hits == [] and eng.summary(1)["suppressed"].get("la toma fue hace mucho")


def test_mano_dentro_de_la_ropa():
    # la muñeca derecha cruza el cuerpo, dentro del torso, con el codo doblado (debajo de la campera)
    eng = GestureEngine(GestureConfig(enabled=ALL))
    hits, _ = take_then(eng, dict(rw=(115, 160), re=(80, 175)), 6)
    assert [h.type for h in hits] == [MANO_ROPA] and "ropa" in hits[0].description


def test_sostener_el_producto_delante_no_es_ropa():
    eng = GestureEngine(GestureConfig(enabled=ALL))
    hits, _ = take_then(eng, dict(rw=(92, 140), re=(78, 165)), 8)  # delante del pecho, del mismo lado
    assert hits == []


def test_mano_en_la_bolsa_con_bolso_detectado():
    eng = GestureEngine(GestureConfig(enabled=ALL))
    bag = (20.0, 190.0, 75.0, 260.0)  # cartera colgada del lado derecho
    hits, _ = take_then(eng, dict(rw=(55, 225), re=(62, 165)), 6, bags=[bag])
    assert [h.type for h in hits] == [MANO_BOLSA] and hits[0].features["bag_seen"] == 1.0


def test_persona_chica_o_sin_postura_no_se_mira():
    eng = GestureEngine(GestureConfig(enabled=ALL))
    small = (40.0, 50.0, 60.0, 100.0)
    assert eng.step(1, pose(), small, 0.0, near_shelf=True, at_checkout=False, staff=False) == []
    assert eng.step(1, None, BBOX, 0.2, near_shelf=True, at_checkout=False, staff=False) == []


def test_configuracion_estricta_y_viaja_aparte_de_las_reglas():
    cfg = GestureConfig.from_dict({"enabled": ["mano_bolsa", "mano_bolsillo"], "pocket_seconds": 0.8})
    assert cfg.enabled == (MANO_BOLSILLO, MANO_BOLSA) and cfg.pocket_seconds == 0.8
    assert GestureConfig.from_dict(cfg.to_dict()) == cfg
    with pytest.raises(ValueError):
        GestureConfig.from_dict({"enabled": ["mano_magica"]})
    with pytest.raises(ValueError):
        GestureConfig.from_dict({"bolsillo_ms": 400})
    # un agente viejo solo mira `rules`: los gestos no pueden ir ahí (RuleConfig es estricto)
    with pytest.raises(ValueError):
        RuleConfig.from_dict({"enabled_rules": ["mano_bolsillo"]})
    zones, rc = parse_config({"zones": [], "rules": {"loiter_seconds": 30}, "gestures": {"enabled": ["mano_ropa"]}})
    assert rc.gestures.enabled == (MANO_ROPA,) and rc.loiter_seconds == 30
    assert RuleEngine(zones, rc).gestures is not None
    assert RuleEngine(*parse_config({"zones": []})).gestures is None  # sin gestos: como siempre


def test_tipos_severidad_y_prioridad():
    for g in ALL:
        assert g in EVENT_TYPES and g in RULE_PRIORITY and SEVERITY[g] in ("alta", "media")
        assert f"regla={g}" in VECTOR_NAMES
    assert {"gesture_dwell_s", "since_take_s", "takes", "wrist_conf_drop", "bag_seen"} <= set(FEATURE_NAMES)


def test_un_modelo_viejo_sigue_valiendo_y_no_filtra_gestos():
    old_names = list(FEATURE_NAMES[:13]) + [f"regla={r}" for r in EVENT_TYPES[:5]]
    n = len(old_names)
    spec = {"vector_names": old_names, "mean": [0.0] * n, "scale": [1.0] * n, "coef": [0.0] * n,
            "intercept": -5.0, "threshold": 0.5}
    s = Scorer(spec)
    assert s.score("merodeo", {}) < 0.5  # lo que conoce, lo puntúa
    assert s.score(MANO_BOLSILLO, {}) == 1.0 and s.is_alert(s.score(MANO_BOLSILLO, {}))  # lo nuevo pasa
    with pytest.raises(ValueError):
        Scorer({**spec, "vector_names": old_names[:-1] + ["variable_rara"]})


def _track(kp, ts):
    t = Track(1, BBOX, "persona", 0.0, ts, hits=5)
    t.kpts = kp
    return t


def test_motor_de_reglas_emite_el_gesto_y_marca_que_tomo_algo():
    zones, rc = parse_config({"zones": [{"name": "Góndola", "type": "shelf",
                                         "polygon": [[0, 0.8], [1, 0.8], [1, 1], [0, 1]]}],
                              "gestures": {"enabled": list(ALL)}})
    eng = RuleEngine(zones, rc)
    frames = [pose(**REACH_R)] * 3 + [pose(rw=(95, 150), re=(70, 140))] + [pose(**POCKET_R)] * 6
    events, ts = [], 0.0
    for kp in frames:
        events += eng.update([_track(kp, ts)], ts, (200, 420))
        ts += 0.2
    assert [e.type for e in events] == [MANO_BOLSILLO]
    e = events[0]
    assert e.zone == "Góndola" and e.severity == "alta" and e.features["gesture_dwell_s"] >= 0.6
    assert set(FEATURE_NAMES) <= set(e.features)
    assert eng.progress(ts)[1]["took"] is True  # la toma frente a la góndola cuenta como «tomó algo»
    assert eng.progress(ts)[1]["gesture"]["hands"]["der"][0] == "bolsillo"


class _Det:
    """Detector de prueba: siempre la misma persona."""

    def detect(self, frame):
        return [Detection(BBOX, "persona", 0.9)]


class _Pose:
    """Postura de prueba: sigue un guion de poses, una por cuadro."""

    def __init__(self, frames):
        self.frames = list(frames)

    def estimate(self, frame, bbox):
        return self.frames.pop(0) if self.frames else pose()


def test_analyzer_mira_la_postura_y_dibuja_el_esqueleto():
    frames = [pose()] * 3 + [pose(**REACH_R)] * 3 + [pose(rw=(95, 150), re=(70, 140))] + [pose(**POCKET_R)] * 8
    cfg = {"zones": [], "gestures": {"enabled": list(ALL)}}
    an = Analyzer(cfg, _Det(), pose=_Pose(frames))
    img = np.zeros((420, 200, 3), np.uint8)
    found = []
    for i in range(len(frames) + 2):
        found += an.process(img, i * 0.2)
    assert [f.type for f in found] == [MANO_BOLSILLO]
    assert found[0].snapshot_jpg[:2] == b"\xff\xd8"
    # sin modelo de postura (o sin gestos activados) no se mira la postura
    an2 = Analyzer(cfg, _Det())
    assert not an2.gestures
    an3 = Analyzer({"zones": []}, _Det(), pose=_Pose(frames))
    assert not an3.gestures
