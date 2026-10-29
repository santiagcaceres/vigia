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
        hits += eng.step(tid, kp, BBOX, ts, near_shelf=kw.get("near_shelf", True), others=kw.get("others"),
                         at_checkout=kw.get("at_checkout", False), staff=kw.get("staff", False), bags=kw.get("bags"))
        ts += 1.0 / fps
    return hits, ts


def take_then(eng, after: dict, n_after: int, **kw):
    """Parada 2 s, alcance al estante (3 cuadros), vuelve al cuerpo, y después `after` durante n_after cuadros."""
    frames = [pose()] * 10 + [pose(**REACH_R)] * 3 + [pose(rw=(95, 150), re=(70, 140))] + [pose(**after)] * n_after
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
    _, ts = feed(eng, [pose()] * 10 + [pose(**REACH_R)] * 3 + [pose(rw=(95, 150), re=(70, 140))])
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
    hits, _ = take_then(eng, dict(rw=(55, 225), re=(45, 170)), 6, bags=[bag])  # codo doblado, mano en la cartera
    assert [h.type for h in hits] == [MANO_BOLSA] and hits[0].features["bag_seen"] == 1.0


def test_la_mano_que_cuelga_junto_a_la_cartera_no_es_guardar():
    # el brazo estirado hacia abajo con la mano al lado de la cartera (apoyada, llevando algo): no es «en la bolsa»
    eng = GestureEngine(GestureConfig(enabled=ALL))
    bag = (20.0, 190.0, 75.0, 260.0)
    hits, _ = take_then(eng, dict(rw=(55, 225), re=(62, 165)), 10, bags=[bag])
    assert hits == []


def test_persona_chica_o_sin_postura_no_se_mira():
    eng = GestureEngine(GestureConfig(enabled=ALL))
    small = (40.0, 50.0, 60.0, 100.0)
    assert eng.step(1, pose(), small, 0.0, near_shelf=True, at_checkout=False, staff=False) == []
    assert eng.step(1, None, BBOX, 0.2, near_shelf=True, at_checkout=False, staff=False) == []


def test_mismo_gesto_en_cualquier_lugar_de_la_imagen():
    # el mismo guion corrido a la otra punta de un cuadro de 1080p: tiene que dar lo mismo (antes «parada» dependía
    # de dónde estaba la persona)
    def guion(dx, dy, frame_h):
        eng = GestureEngine(GestureConfig(enabled=ALL))
        frames = [pose()] * 10 + [pose(**REACH_R)] * 3 + [pose(rw=(95, 150), re=(70, 140))] + [pose(**POCKET_R)] * 6
        hits, ts = [], 0.0
        for kp in frames:
            k = kp.copy()
            k[:, 0] += dx
            k[:, 1] += dy
            box = (BBOX[0] + dx, BBOX[1] + dy, BBOX[2] + dx, BBOX[3] + dy)
            hits += eng.step(1, k, box, ts, near_shelf=True, at_checkout=False, staff=False, frame_h=frame_h)
            ts += 0.2
        return [h.type for h in hits]

    assert guion(0, 0, 1080) == guion(1500, 600, 1080) == guion(700, 300, 1080) == [MANO_BOLSILLO]


def test_tamano_minimo_relativo_al_cuadro():
    cfg = GestureConfig(enabled=ALL)
    assert cfg.min_height(360) == pytest.approx(90) and cfg.min_height(1080) == pytest.approx(270)
    eng = GestureEngine(cfg)
    # 350 px de alto: en 1080p alcanza (> 270), en un cuadro de 4320 de alto no (es chica para ese cuadro)
    assert eng.step(1, pose(), BBOX, 0.0, near_shelf=True, at_checkout=False, staff=False, frame_h=1080) == []
    assert eng.summary(1)["frames"] == 1
    eng2 = GestureEngine(cfg)
    eng2.step(1, pose(), BBOX, 0.0, near_shelf=True, at_checkout=False, staff=False, frame_h=4320)
    assert eng2.summary(1)["frames"] == 0


def test_tapada_un_rato_la_permanencia_arranca_de_nuevo():
    # toma, la mano 1 cuadro en el bolsillo, alguien la tapa 2 s y vuelve con la mano ahí 2 cuadros: no suma el rato
    # en que no se la vio (0,6 s seguidos no los cumplió)
    eng = GestureEngine(GestureConfig(enabled=ALL))
    _, ts = feed(eng, [pose()] * 10 + [pose(**REACH_R)] * 3 + [pose(rw=(95, 150), re=(70, 140))] + [pose(**POCKET_R)])
    for _ in range(10):  # 2 s sin verla
        assert eng.step(1, None, BBOX, ts, near_shelf=True, at_checkout=False, staff=False) == []
        ts += 0.2
    hits, _ = feed(eng, [pose(**POCKET_R)] * 2, t0=ts)
    assert hits == []


def test_configuracion_tolerante_en_los_agentes_y_compacta_al_mandar():
    nueva = {"enabled": ["mano_bolsa", "mano_manga"], "pocket_seconds": 0.9, "min_pose_ok": 0.5}
    with pytest.raises(ValueError):
        GestureConfig.from_dict(nueva)  # el servidor no la guarda
    cfg = GestureConfig.from_dict(nueva, strict=False)  # un agente viejo la lee igual: ignora lo que no conoce
    assert cfg.enabled == (MANO_BOLSA,) and cfg.pocket_seconds == 0.9
    # al mandarla, solo los gestos y lo que difiere de lo de siempre
    assert cfg.to_dict(compact=True) == {"enabled": ["mano_bolsa"], "pocket_seconds": 0.9}
    _, rc = parse_config({"zones": [], "gestures": nueva})
    assert rc.gestures.enabled == (MANO_BOLSA,)


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
    frames = [pose()] * 10 + [pose(**REACH_R)] * 3 + [pose(rw=(95, 150), re=(70, 140))] + [pose(**POCKET_R)] * 6
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


def test_un_recuadro_duplicado_de_la_misma_persona_no_la_tapa():
    # el detector da, además de la persona, otro recuadro con su torso y el brazo (el seguimiento lo toma como otra
    # persona): no tiene que contar como «otra persona» a la que se le pasa algo
    zones, rc = parse_config({"zones": [], "gestures": {"enabled": list(ALL)}})
    eng = RuleEngine(zones, rc)
    frames = [pose()] * 10 + [pose(**REACH_R)] * 3 + [pose(rw=(95, 150), re=(70, 140))] + [pose(**POCKET_R)] * 6
    events, ts = [], 0.0
    for kp in frames:
        dup = Track(2, (60.0, 90.0, 150.0, 260.0), "persona", 0.0, ts, hits=5)
        events += eng.update([_track(kp, ts), dup], ts, (200, 420))
        ts += 0.2
    assert [e.type for e in events] == [MANO_BOLSILLO]


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
    frames = [pose()] * 10 + [pose(**REACH_R)] * 3 + [pose(rw=(95, 150), re=(70, 140))] + [pose(**POCKET_R)] * 8
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


def test_pasarle_algo_a_otra_persona_no_es_tomar():
    eng = GestureEngine(GestureConfig(enabled=ALL))
    other = (-60.0, 60.0, 15.0, 400.0)  # otra persona a la izquierda de la imagen: la mano llega a ella
    hits, _ = take_then(eng, POCKET_R, 6, others=[other])
    assert hits == [] and eng.summary(1)["takes"] == 0


def test_la_mano_que_ya_estaba_en_la_cadera_no_cuenta():
    # la mano derecha queda apoyada en la cadera mientras la izquierda toma algo: no la «llevó» al bolsillo
    eng = GestureEngine(GestureConfig(enabled=ALL))
    reach_l = dict(lw=(210, 120), le=(170, 112))
    frames = ([pose(**POCKET_R)] * 10 + [pose(**reach_l, **POCKET_R)] * 3 + [pose(lw=(105, 150), le=(130, 140), **POCKET_R)]
              + [pose(**POCKET_R)] * 8)
    hits, _ = feed(eng, frames)
    assert hits == [] and eng.summary(1)["suppressed"].get("la mano ya estaba ahí antes de tomar")


def test_caminando_o_recien_aparecida_no_toma():
    eng = GestureEngine(GestureConfig(enabled=ALL))
    # recién aparecida: el alcance no cuenta
    hits, _ = feed(eng, [pose(**REACH_R)] * 3 + [pose(rw=(95, 150), re=(70, 140))] + [pose(**POCKET_R)] * 6)
    assert hits == [] and eng.summary(1)["takes"] == 0
    # caminando (las caderas se corren 60 px por cuadro, ~0,9 altos por segundo): tampoco
    eng2 = GestureEngine(GestureConfig(enabled=ALL))
    frames = [pose()] * 10 + [pose(**REACH_R)] * 3
    hits, ts = [], 0.0
    for i, kp in enumerate(frames):
        k = kp.copy()
        k[:, 0] += 60 * i
        hits += eng2.step(1, k, (BBOX[0] + 60 * i, BBOX[1], BBOX[2] + 60 * i, BBOX[3]), ts,
                          near_shelf=True, at_checkout=False, staff=False)
        ts += 0.2
    assert eng2.summary(1)["reaches"] == 0


def test_si_la_placa_de_video_falla_la_postura_sigue_en_el_procesador(tmp_path):
    from pathlib import Path

    from core.pose import PoseEstimator, _CpuRunner

    model = Path(__file__).resolve().parent.parent / "models" / "postura.onnx"
    if not model.exists():
        pytest.skip("sin modelo de postura")

    class Rota:
        def run(self, canvas):
            raise RuntimeError("DmlExecutionProvider: falló la primera pasada")

    est = PoseEstimator(model)
    est.runner, est.on_gpu, est.device = Rota(), True, "GPU (Dml)"
    est.estimate(np.zeros((360, 640, 3), np.uint8), (100.0, 50.0, 200.0, 340.0))  # no corta: sigue
    assert isinstance(est.runner, _CpuRunner) and est.device == "procesador"


def test_sin_ejemplos_de_gestos_se_entrena_con_las_variables_de_antes():
    # un modelo entrenado sin gestos usa exactamente las variables de antes: las máquinas sin actualizar lo siguen usando
    from core.scorer import LEGACY_NAMES
    from server.training import vector_names

    assert len(LEGACY_NAMES) == 18 and not any("gesture" in n or "mano_" in n for n in LEGACY_NAMES)
    assert vector_names([{"type": "merodeo"}, {"type": "salida_sin_pago"}]) == LEGACY_NAMES
    assert vector_names([{"type": "merodeo"}, {"type": MANO_BOLSA}]) == VECTOR_NAMES
