"""IA de movimientos de hurto (core/hurto_ia.py): rasgos por ventana con esqueletos armados a mano (persona de frente,
torso de 100 px), etiquetas por marcas, entrenamiento con clips sintéticos, validación separando por clip, el detector
en línea (ventanas seguidas y enfriamiento) y la integración con el Analyzer y el análisis de alarmas."""

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from core import hurto_ia as hi
from core.detectors import Detection
from core.rules import EVENT_TYPES, HURTO_IA, HURTO_IA_NAME, MANO_BOLSA, RULE_PRIORITY, SEVERITY, Zone, parse_zones
from core.tracker import Track

ROOT = Path(__file__).resolve().parents[1]
BOX = (40.0, 50.0, 160.0, 400.0)  # 350 px de alto
W, H = 200, 420
F = {n: i for i, n in enumerate(hi.FEATURE_NAMES)}


def pose(lw=None, rw=None, le=None, re=None, conf=0.9, dx=0.0, dy=0.0, k=1.0, rng=None, noise=0.0):
    """Persona de frente: hombros (80,100)-(120,100), caderas (85,200)-(115,200) → torso de 100 px (como en
    tests/test_gestos.py). Por defecto los brazos cuelgan. `k`: escala; `dx`, `dy`: corrimiento."""
    pts = {0: (100, 70), 1: (95, 65), 2: (105, 65), 5: (120, 100), 6: (80, 100), 11: (115, 200), 12: (85, 200),
           13: (115, 280), 14: (85, 280), 15: (115, 360), 16: (85, 360),
           7: le or (128, 150), 8: re or (72, 150), 9: lw or (130, 205), 10: rw or (70, 205)}
    kp = [[0.0, 0.0, 0.0] for _ in range(17)]
    for i, (x, y) in pts.items():
        nx, ny = (rng.normal(0, noise), rng.normal(0, noise)) if rng is not None else (0.0, 0.0)
        kp[i] = [x * k + dx + nx, y * k + dy + ny, conf]
    return kp


REACH = dict(rw=(-10, 120), re=(30, 112))  # brazo derecho estirado al costado, a la altura del pecho
BACK = dict(rw=(95, 150), re=(70, 140))  # la mano vuelve al cuerpo
POCKET = dict(rw=(83, 205), re=(70, 160))  # muñeca derecha junto a la cadera derecha, codo doblado


def box(k=1.0, dx=0.0, dy=0.0):
    return [BOX[0] * k + dx, BOX[1] * k + dy, BOX[2] * k + dx, BOX[3] * k + dy]


def rec(t, people, bags=(), w=W, h=H):
    return {"t": round(t, 3), "w": w, "h": h, "people": people, "bags": [list(b) for b in bags]}


def steady(kind="colgando", seconds=6.0, fps=5.0, k=1.0, dx=0.0, tid=1):
    """Una persona quieta, con los brazos de una manera todo el rato."""
    arms = {"colgando": {}, "bolsillo": POCKET, "alcance": REACH}[kind]
    return [rec(i / fps, [{"id": tid, "box": box(k, dx), "kp": pose(**arms, k=k, dx=dx)}], w=W * k, h=H * k)
            for i in range(int(seconds * fps))]


def last_window(frames, zones=None):
    ws = hi.person_windows(frames, zones)
    assert ws, "no salió ninguna ventana"
    return dict(zip(hi.FEATURE_NAMES, ws[-1]["f"]))


# ---------------------------------------------------------------------------------------------- lo básico
def test_el_servidor_lo_importa_sin_opencv():
    out = subprocess.run([sys.executable, "-c", "import core.hurto_ia, sys; print('cv2' in sys.modules)"],
                         cwd=ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_constantes_y_nombres_del_contrato():
    assert (hi.FEATURE_VERSION, hi.WINDOW_S, hi.STEP_S, hi.MIN_POSED, hi.MAX_CLIP_S) == (1, 3.0, 1.0, 0.5, 1800)
    assert len(set(hi.FEATURE_NAMES)) == len(hi.FEATURE_NAMES) >= 20
    assert all(n.startswith("ia_") and hi.FEATURE_HELP[n] for n in hi.FEATURE_NAMES)
    from core.scorer import FEATURE_NAMES as FILTER_NAMES

    # los rasgos van junto con las medidas del filtro en la misma alerta: no se pisan
    assert not set(hi.FEATURE_NAMES) & set(FILTER_NAMES) and "ia_score" not in hi.FEATURE_NAMES
    assert issubclass(hi.NotEnoughData, ValueError)


def test_el_tipo_de_evento_va_al_final_y_el_filtro_viejo_sigue_andando():
    from core.scorer import LEGACY_NAMES, VECTOR_NAMES, Scorer

    assert EVENT_TYPES[-1] == HURTO_IA == "hurto_ia" and HURTO_IA_NAME == "Posible hurto (IA de movimientos)"
    assert EVENT_TYPES[:8] == ("salida_sin_pago", "merodeo", "zona_restringida", "fuera_de_horario", "alarma_dvr",
                               "mano_bolsillo", "mano_ropa", "mano_bolsa")  # el orden de siempre
    assert RULE_PRIORITY.index(HURTO_IA) == RULE_PRIORITY.index(MANO_BOLSA) + 1 and SEVERITY[HURTO_IA] == "alta"
    # la lista de variables de los agentes sin actualizar no cambia (si no, ignoran el modelo entero)
    assert LEGACY_NAMES == ("track_age_s", "shelf_total_s", "checkout_total_s", "restricted_total_s",
                            "max_shelf_dwell_s", "shelf_visits", "path_len", "mean_speed", "people_in_scene", "bbox_h",
                            "has_time", "hour_sin", "hour_cos", "regla=salida_sin_pago", "regla=merodeo",
                            "regla=zona_restringida", "regla=fuera_de_horario", "regla=alarma_dvr")
    assert VECTOR_NAMES[-1] == "regla=hurto_ia" and VECTOR_NAMES[:-1] == tuple(
        n for n in VECTOR_NAMES if n != "regla=hurto_ia")
    n = len(LEGACY_NAMES)
    old = Scorer({"vector_names": list(LEGACY_NAMES), "mean": [0.0] * n, "scale": [1.0] * n, "coef": [0.0] * n,
                  "intercept": -5.0, "threshold": 0.5})
    assert old.score(HURTO_IA, {"ia_score": 0.9}) == 1.0  # un modelo viejo no conoce el tipo: siempre es alerta
    assert old.score("salida_sin_pago", {}) < 0.5


# ---------------------------------------------------------------------------------------------- rasgos
def test_rasgos_de_bolsillo_alcance_y_brazos_colgando():
    hang = last_window(steady("colgando"))
    pocket = last_window(steady("bolsillo"))
    reach = last_window(steady("alcance"))
    assert hang["ia_mano_cintura"] == 0.0 and pocket["ia_mano_cintura"] == 1.0
    assert pocket["ia_muneca_cadera_min"] < 0.1 < hang["ia_muneca_cadera_min"]
    assert pocket["ia_codo_min"] < 0.9 < hang["ia_codo_min"]  # codo doblado (154°) contra brazo que cuelga (173°)
    assert reach["ia_alcance"] == 1.0 and hang["ia_alcance"] == 0.0 and reach["ia_alcance_max"] > 1.0
    for f in (hang, pocket, reach):
        assert f["ia_con_postura"] == 1.0 and f["ia_quieto"] == 1.0 and f["ia_cadera_vel"] == 0.0
        assert f["ia_alto_rel"] == pytest.approx(350 / 420, abs=1e-3) and f["ia_zonas"] == 0.0
        assert f["ia_toma"] == 0.0 and f["ia_mano_otra_persona"] == 0.0 and f["ia_bolso_visto"] == 0.0
        assert all(math.isfinite(v) for v in f.values())


def test_toma_y_velocidad_de_la_muneca():
    fps = 5.0
    script = [{}] * 5 + [REACH] * 4 + [BACK] * 2 + [POCKET] * 4  # 3 s: parada, estira, vuelve y al bolsillo
    frames = [rec(i / fps, [{"id": 1, "box": box(), "kp": pose(**arms)}]) for i, arms in enumerate(script)]
    f = last_window(frames)
    assert f["ia_toma"] == 1.0 and f["ia_alcance"] > 0 and f["ia_mano_cintura"] > 0
    assert f["ia_muneca_vel_max"] > 3.0 > f["ia_muneca_vel_media"] > 0


def test_los_rasgos_no_dependen_del_tamano_ni_del_lugar_en_la_imagen():
    for kind in ("colgando", "bolsillo", "alcance"):
        a = last_window(steady(kind))
        b = last_window(steady(kind, k=2.5, dx=37.0))  # 2,5 veces más grande (cuadro también) y corrida
        for n in hi.FEATURE_NAMES:
            assert b[n] == pytest.approx(a[n], abs=2e-3), (kind, n)


def test_puntos_que_faltan_o_vienen_mal_no_rompen_nada():
    fps = 5.0
    frames = []
    for i in range(40):
        kp = pose(**POCKET)
        if i % 4 == 1:
            kp[10][2] = 0.1  # la muñeca derecha no se ve (tapada)
        if i % 4 == 2:
            kp = None  # no se le miró la postura
        if i % 10 == 3:
            kp = [[1, 2, 3]] * 16  # mal armada: se toma como sin postura
        if i % 10 == 7:
            kp = pose(conf=0.1)  # sin hombros ni caderas a la vista
        frames.append(rec(i / fps, [{"id": 1, "box": box(), "kp": kp}]))
    frames.append({"t": 8.1, "w": W})  # cuadros raros: se saltean
    frames.append("nada")
    frames.append(rec(8.2, [{"id": "x", "box": "y"}, {"id": 2, "box": [5, 5, 1, 1]}]))
    ws = hi.person_windows(frames)
    assert ws and all(len(w["f"]) == len(hi.FEATURE_NAMES) and all(math.isfinite(v) for v in w["f"]) for w in ws)
    for w in ws:
        f = dict(zip(hi.FEATURE_NAMES, w["f"]))
        assert hi.MIN_POSED <= f["ia_con_postura"] < 1.0 and f["ia_muneca_oculta"] > 0
    # con la postura en menos de la mitad de los cuadros, la persona no da ventanas
    sparse = [rec(i / fps, [{"id": 1, "box": box(), "kp": pose() if i % 3 == 0 else None}]) for i in range(40)]
    assert hi.person_windows(sparse) == []
    assert hi.person_windows([]) == [] and hi.person_windows(None) == []


def test_una_ventana_por_persona_cada_paso():
    fps = 5.0
    frames = [rec(i / fps, [{"id": 1, "box": box(), "kp": pose()},
                            {"id": 7, "box": box(dx=300), "kp": pose(dx=300) if i >= 50 else None}], w=600)
              for i in range(100)]  # 20 s; la persona 7 recién tiene postura desde los 10 s
    ws = hi.person_windows(frames)
    one = [w for w in ws if w["track_id"] == 1]
    seven = [w for w in ws if w["track_id"] == 7]
    assert [w["t1"] for w in one] == pytest.approx([float(t) for t in range(3, 20)], abs=1e-6)
    assert all(w["t0"] == pytest.approx(w["t1"] - hi.WINDOW_S) for w in ws)
    assert seven and min(w["t1"] for w in seven) >= 11.0
    # el reloj vuelve atrás (otro video seguido): se empieza de cero
    again = frames + [rec(i / fps, [{"id": 1, "box": box(), "kp": pose()}]) for i in range(20)]
    assert len(hi.person_windows(again)) == len(ws) + 2
    # MAX_CLIP_S: lo que pasa de 30 min no se mira
    long = [rec(t, [{"id": 1, "box": box(), "kp": pose()}]) for t in (0, 1, 2, 3, 1801, 1802, 1803, 1804, 1805)]
    assert all(w["t1"] <= hi.MAX_CLIP_S for w in hi.person_windows(long))


def test_bolso_y_mano_sobre_otra_persona():
    fps = 5.0
    bag = (70.0, 190.0, 110.0, 260.0)  # una cartera delante de la cadera: la muñeca derecha adentro
    frames = [rec(i / fps, [{"id": 1, "box": box(), "kp": pose(rw=(90, 205), re=(70, 160))}], bags=[bag])
              for i in range(20)]
    f = last_window(frames)
    assert f["ia_bolso_mano"] == 1.0 and f["ia_bolso_visto"] == 1.0
    far_bag = [rec(r["t"], r["people"], bags=[(400, 10, 420, 30)], w=500) for r in frames]
    g = last_window(far_bag)
    assert g["ia_bolso_mano"] == 0.0 and g["ia_bolso_visto"] == 0.0  # el bolso de otro (lejos) no es suyo
    other = (-60.0, 60.0, 15.0, 400.0)  # otra persona a la izquierda: la mano estirada llega a ella
    frames = [rec(i / fps, [{"id": 1, "box": box(), "kp": pose(**REACH)}, {"id": 2, "box": list(other), "kp": None}])
              for i in range(20)]
    assert last_window(frames)["ia_mano_otra_persona"] == 1.0


def test_zonas_de_la_camara():
    zone = {"name": "Góndola", "type": "shelf", "polygon": [[0, 0.8], [1, 0.8], [1, 1], [0, 1]]}
    frames = steady("colgando")
    for zones in ([zone], parse_zones([zone]), {"zones": [zone]}, [zone, {"name": "mal", "type": "raro"}]):
        f = last_window(frames, zones)
        assert f["ia_zonas"] == 1.0 and f["ia_gondola"] == 1.0 and f["ia_caja_salida"] == 0.0
    cash = {"name": "Caja", "type": "checkout", "polygon": [[0, 0.8], [1, 0.8], [1, 1], [0, 1]]}
    assert last_window(frames, [cash])["ia_caja_salida"] == 1.0
    aisle = {"name": "Pasillo", "type": "aisle", "polygon": [[0, 0.8], [1, 0.8], [1, 1], [0, 1]]}
    f = last_window(frames, [aisle])
    assert f["ia_zonas"] == 0.0 and f["ia_gondola"] == 0.0  # un pasillo no dice nada del hurto


def test_registro_de_cuadro_desde_el_seguimiento():
    t1 = Track(3, BOX, "persona", 0.0, 1.0, hits=5)
    t1.kpts = np.array(pose(), np.float32)
    t2 = Track(4, (300.0, 10.0, 340.0, 90.0), "persona", 0.0, 1.0, hits=5, missed=2)  # no se ve en este cuadro
    car = Track(5, (0.0, 0.0, 50.0, 30.0), "vehiculo", 0.0, 1.0, hits=5)
    r = hi.frame_record(1.23456, (W, H), [t1, t2, car], [Detection((1, 2, 3, 4), "cartera", 0.5), (5, 6, 7, 8)])
    assert r["t"] == 1.235 and (r["w"], r["h"]) == (W, H) and r["bags"] == [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert [p["id"] for p in r["people"]] == [3] and len(r["people"][0]["kp"]) == 17
    assert r["people"][0]["box"] == list(BOX) and json.loads(json.dumps(r)) == r
    t1.kpts = None
    assert hi.frame_record(0, (W, H), [t1])["people"][0]["kp"] is None


# ---------------------------------------------------------------------------------------------- etiquetas
def _wins(track_id, t1s):
    return [{"track_id": track_id, "t0": t - 3.0, "t1": t, "f": [0.0] * len(hi.FEATURE_NAMES)} for t in t1s]


def test_etiquetas_por_marcas():
    ws = _wins(1, range(3, 21)) + _wins(2, range(3, 21))
    assert hi.window_labels("normal", [], ws) == [0] * len(ws)
    assert hi.window_labels("sin_marcar", [{"start": 1, "end": 5, "track_id": 1}], ws) == [None] * len(ws)
    assert hi.window_labels("hurto", [], ws) == [None] * len(ws)  # un hurto sin marcas: no se sabe dónde
    marks = [{"start": 10.0, "end": 12.0, "track_id": 1, "kind": "bolsillo"}]
    lab = dict(zip([(w["track_id"], w["t1"]) for w in ws], hi.window_labels("hurto", marks, ws)))
    # persona 1: ventanas (t1-3, t1] que se pisan 1 s o más con [10, 12] → t1 de 11 a 14
    assert [t for t in range(3, 21) if lab[(1, t)] == 1] == [11, 12, 13, 14]
    # el borde (2 s alrededor de la marca) sin llegar a 1 s: no se usa
    assert [t for t in range(3, 21) if lab[(1, t)] is None] == [9, 10, 15, 16]
    assert lab[(1, 8)] == 0 and lab[(1, 17)] == 0 and lab[(1, 20)] == 0
    # otra persona, aunque sea en el mismo momento: 0
    assert all(lab[(2, t)] == 0 for t in range(3, 21))
    # una marca que no dice persona vale para todas
    both = hi.window_labels("hurto", [{"start": 10.0, "end": 12.0, "track_id": None}], ws)
    assert sum(1 for x in both if x == 1) == 8
    # marcas raras se saltean; y una marca corta (menos de 1 s) no da ventanas «sí»
    odd = [{"start": "a", "end": 3}, {"start": 9, "end": 8}, None, {"start": 10.0, "end": 10.5, "track_id": 1}]
    got = hi.window_labels("hurto", odd, ws)
    assert 1 not in got and None in got


def test_muestras_para_entrenar_desde_los_clips():
    clips = [{"id": 10, "verdict": "hurto", "marks": [{"start": 10.0, "end": 12.0, "track_id": 1}],
              "windows": _wins(1, range(3, 21)), "duration_s": 21.0},
             {"id": 11, "verdict": "normal", "marks": [], "windows": _wins(1, range(3, 9)), "duration_s": 9.0},
             {"id": 12, "verdict": "sin_marcar", "marks": [], "windows": _wins(1, range(3, 9))},
             {"id": 13, "verdict": "normal", "windows": _wins(1, range(3, 9)), "feature_version": 99}]
    samples, groups, durations = hi.build_samples(clips)
    assert set(groups) == {10, 11} and durations == {10: 21.0, 11: 9.0}
    pos = [s for s in samples if s[1] == 1]
    assert len(pos) == 4 and all(s[2]["mark"] == (10, 0) and s[2]["track_id"] == 1 for s in pos)
    assert len(samples) == 18 - 4 + 6  # sin las 4 del borde


# ---------------------------------------------------------------------------------------------- entrenamiento
def synthetic_clip(kind: str, seed: int, seconds: float = 30.0, fps: float = 5.0):
    """Un clip sintético con ruido: «hurto» (estira el brazo al estante, la mano vuelve y se va al bolsillo 2,6 s,
    en un momento al azar) o «normal» (brazos colgando; en algunos estira el brazo y lo deja ahí: mira un producto).
    Algunos con una segunda persona que no hace nada."""
    rng = np.random.default_rng(seed)
    t_theft = float(rng.uniform(6, seconds - 10))
    second = seed % 3 == 0
    frames = []
    for i in range(int(seconds * fps)):
        t = i / fps
        rel = t - t_theft
        arms = {}
        if kind == "hurto" and 0 <= rel < 1:
            arms = REACH
        elif kind == "hurto" and 1 <= rel < 1.4:
            arms = BACK
        elif kind == "hurto" and 1.4 <= rel < 4:
            arms = POCKET
        elif kind == "normal" and seed % 2 and 10 <= t < 12:
            arms = REACH
        people = [{"id": 1, "box": box(), "kp": pose(**arms, rng=rng, noise=2.0)}]
        if second:
            people.append({"id": 2, "box": box(dx=250), "kp": pose(dx=250, rng=rng, noise=2.0)})
        frames.append(rec(t, people, w=500))
    marks = ([{"start": round(t_theft, 2), "end": round(t_theft + 4, 2), "track_id": 1, "kind": "bolsillo"}]
             if kind == "hurto" else [])
    return frames, marks


def synthetic_clips(n_theft=8, n_normal=5):
    clips = []
    for i in range(n_theft + n_normal):
        kind = "hurto" if i < n_theft else "normal"
        frames, marks = synthetic_clip(kind, i)
        clips.append({"id": 100 + i, "verdict": kind, "marks": marks, "windows": hi.person_windows(frames),
                      "duration_s": 30.0, "feature_version": hi.FEATURE_VERSION})
    return clips


def test_entrenar_con_clips_sinteticos_y_usar_el_modelo():
    clips = synthetic_clips()
    samples, groups, durations = hi.build_samples(clips)
    spec, m = hi.train(samples, groups, target_recall=0.8, durations=durations)
    assert {"marks", "marks_seen", "false_per_hour", "clips", "windows", "positives", "threshold", "auc"} <= set(m)
    assert m["marks"] == 8 and m["marks_seen"] >= 7 and m["clips"] == 13 and m["validation"] == "por_clip"
    assert m["auc"] > 0.95 and m["false_per_hour"] <= 1.0 and m["positives"] > 0 and m["windows"] == len(samples)
    assert 0 < m["threshold"] < 1 and m["negative_hours"] > 0.08  # las horas de video normal: los 5 clips enteros
    assert spec["kind"] == "ia_hurto" and spec["version"] == hi.FEATURE_VERSION
    assert spec["names"] == list(hi.FEATURE_NAMES) and len(spec["coef"]) == len(spec["mean"]) == len(hi.FEATURE_NAMES)
    assert (spec["window_s"], spec["step_s"], spec["consecutive"], spec["cooldown_s"]) == (3, 1, 2, 30)
    spec = json.loads(json.dumps(spec))  # viaja como JSON (base de datos, /api/agent/config)
    model = hi.TheftModel(spec)
    assert model.ok
    # con un clip nuevo: avisa en el hurto y no en uno normal
    for kind, seed, expect in (("hurto", 991, True), ("normal", 993, False), ("normal", 994, False)):
        frames, marks = synthetic_clip(kind, seed)
        det = hi.OnlineTheftDetector(spec)
        hits = [h for r in frames for h in det.update(r["t"], r["people"], r["bags"], (r["w"], r["h"]))]
        assert bool(hits) is expect, (kind, seed, hits)
        for h in hits:
            assert h["track_id"] == 1 and marks[0]["start"] - 3 <= h["t1"] <= marks[0]["end"] + 3
            assert h["score"] >= model.threshold and set(h["features"]) == set(hi.FEATURE_NAMES)
    # sin los datos de cada ventana (solo rasgos y etiqueta) también entrena: cada clip con hurto cuenta como uno
    spec2, m2 = hi.train([(s[0], s[1]) for s in samples], groups)
    assert m2["marks"] == 8 and m2["by_mark"] is False and hi.TheftModel(spec2).ok


def test_la_validacion_separa_por_clip():
    # cada clip tiene su «firma» (todas sus ventanas casi iguales) y sus ventanas de hurto, un corrimiento al azar
    # propio de ese clip: no hay nada que sirva para otro clip. Si se validara mezclando ventanas del mismo clip, el
    # modelo se las aprendería de memoria y parecería perfecto; separando por clip, no ve nada
    rng = np.random.default_rng(5)
    n = len(hi.FEATURE_NAMES)
    samples, groups = [], []
    for g in range(16):
        center, shift = rng.normal(0, 1, n), rng.normal(0, 3, n)
        theft = g < 8
        for i in range(40):
            pos = theft and 10 <= i < 16
            f = center + (shift if pos else 0) + rng.normal(0, 0.01, n)
            samples.append((f.tolist(), int(pos), {"track_id": 1, "t0": i - 3.0, "t1": float(i),
                                                     "mark": 0 if pos else None}))
            groups.append(f"clip{g}")
    folds = hi._folds(np.array([s[1] for s in samples]), groups)
    owner = {}
    for k, idx in enumerate(folds):
        for i in idx:
            assert owner.setdefault(groups[i], k) == k  # un clip entero en un solo grupo de prueba
    assert sorted(i for f in folds for i in f) == list(range(len(samples)))
    _, m = hi.train(samples, groups)
    assert m["auc"] < 0.8 and m["false_per_hour"] > 0


def test_faltan_datos():
    clips = synthetic_clips(n_theft=4, n_normal=5)
    with pytest.raises(hi.NotEnoughData, match="Faltan hurtos marcados: hay 4"):
        hi.train(*hi.build_samples(clips)[:2])
    clips = synthetic_clips(n_theft=6, n_normal=1)
    with pytest.raises(hi.NotEnoughData, match="Faltan clips normales"):
        hi.train(*hi.build_samples(clips)[:2])
    # todos los hurtos en un mismo clip: no se puede probar con un clip que el modelo no vio
    frames = []
    marks = []
    for j in range(6):
        part, mk = synthetic_clip("hurto", 7 + j)
        frames += [dict(r, t=round(r["t"] + 30 * j, 3)) for r in part]
        marks += [dict(m, start=m["start"] + 30 * j, end=m["end"] + 30 * j) for m in mk]
    one = {"id": 1, "verdict": "hurto", "marks": marks, "windows": hi.person_windows(frames), "duration_s": 180}
    clips = [one] + [c for c in synthetic_clips(n_theft=0, n_normal=4)]
    with pytest.raises(hi.NotEnoughData, match="mismo clip"):
        hi.train(*hi.build_samples(clips)[:2])
    with pytest.raises(ValueError):
        hi.train([([0.0] * len(hi.FEATURE_NAMES), 1)], [1, 2])


# ---------------------------------------------------------------------------------------------- en vivo
def handmade(coef: dict, intercept: float, threshold: float = 0.5, **extra) -> dict:
    n = len(hi.FEATURE_NAMES)
    return {"kind": "ia_hurto", "version": hi.FEATURE_VERSION, "names": list(hi.FEATURE_NAMES), "mean": [0.0] * n,
            "scale": [1.0] * n, "coef": [float(coef.get(name, 0.0)) for name in hi.FEATURE_NAMES],
            "intercept": intercept, "threshold": threshold, "window_s": 3, "step_s": 1, "consecutive": 2,
            "cooldown_s": 30, **extra}


POCKET_SPEC = handmade({"ia_mano_cintura": 20.0}, -10.0)  # puntaje alto con la mano en el bolsillo más de la mitad


def test_el_modelo_es_tolerante_con_lo_que_le_llegue():
    good = hi.TheftModel(POCKET_SPEC)
    assert good.ok and good.threshold == 0.5 and good.consecutive == 2 and good.cooldown_s == 30
    f_pocket = [0.0] * len(hi.FEATURE_NAMES)
    f_pocket[F["ia_mano_cintura"]] = 1.0
    assert good.score(f_pocket) > 0.99 and good.score([0.0] * len(hi.FEATURE_NAMES)) < 0.01
    assert good.score([float("nan")] * len(hi.FEATURE_NAMES)) < 0.01 and good.score([1, 2]) == 0.0
    assert hi.TheftModel.maybe(POCKET_SPEC) is not None and hi.spec_ok(POCKET_SPEC)
    bad = [None, "raro", 3, [], {}, dict(POCKET_SPEC, version=2), dict(POCKET_SPEC, names=["x"]),
           dict(POCKET_SPEC, kind="otro"), dict(POCKET_SPEC, coef=[1.0]), dict(POCKET_SPEC, threshold=7),
           dict(POCKET_SPEC, mean=["a"] * len(hi.FEATURE_NAMES)), dict(POCKET_SPEC, window_s=5),
           {k: v for k, v in POCKET_SPEC.items() if k != "intercept"}, dict(POCKET_SPEC, intercept=float("inf"))]
    for spec in bad:
        m = hi.TheftModel(spec)
        assert not m.ok and m.score(f_pocket) == 0.0 and not m.is_alert(1.0) and hi.TheftModel.maybe(spec) is None
        det = hi.OnlineTheftDetector(spec)
        assert not det.ok and all(det.update(r["t"], r["people"], r["bags"], (W, H)) == []
                                  for r in steady("bolsillo"))
    # lo que sobra o es raro en consecutive y cooldown queda en lo de siempre
    m = hi.TheftModel(dict(POCKET_SPEC, consecutive=99, cooldown_s=-3, extra="algo"))
    assert m.ok and m.consecutive == 2 and m.cooldown_s == 30


def run(det, frames):
    return [h for r in frames for h in det.update(r["t"], r["people"], r["bags"], (r["w"], r["h"]))]


def scripted(segments, fps=5.0, t0=0.0, tid=1, dx=0.0, w=W):
    """`segments`: [(segundos, brazos)] seguidos."""
    out, t = [], t0
    for seconds, arms in segments:
        for _ in range(int(round(seconds * fps))):
            out.append(rec(t, [{"id": tid, "box": box(dx=dx), "kp": pose(**arms, dx=dx)}], w=w))
            t = round(t + 1.0 / fps, 6)
    return out


def test_detector_en_linea_ventanas_seguidas_y_enfriamiento():
    det = hi.OnlineTheftDetector(POCKET_SPEC)
    hits = run(det, scripted([(6, {}), (8, POCKET)]))
    # la primera ventana con la mano en el bolsillo más de la mitad termina a los ~8 s; la segunda seguida, ~9 s
    assert len(hits) == 1 and hits[0]["track_id"] == 1 and 8.5 <= hits[0]["t1"] <= 9.5
    h = hits[0]
    assert h["score"] > 0.5 and h["t0"] < h["t1"] - 3 and h["features"]["ia_mano_cintura"] > 0.5
    assert det.scores[1] > 0.5
    # una racha avisa una sola vez; cortada y vuelta a empezar dentro del enfriamiento, no avisa…
    t = 14.0
    more = run(det, scripted([(4, {}), (5, POCKET)], t0=t))
    assert more == [] and det.scores[1] > 0.5
    # …pero si la racha sigue, avisa apenas termina el enfriamiento (30 s desde el aviso anterior)
    later = run(det, scripted([(20, POCKET)], t0=23.0))
    assert len(later) == 1 and later[0]["t1"] == pytest.approx(h["t1"] + 30, abs=1.01)


def test_detector_en_linea_por_persona_memoria_y_reloj():
    det = hi.OnlineTheftDetector(dict(POCKET_SPEC, consecutive=1))
    both = []
    for a, b in zip(scripted([(10, POCKET)]), scripted([(10, POCKET)], tid=2, dx=250, w=500)):
        both.append(rec(a["t"], a["people"] + b["people"], w=500))
    hits = run(det, both)
    assert sorted(h["track_id"] for h in hits) == [1, 2]  # cada persona lo suyo (consecutive=1: la primera ventana)
    assert all(h["t1"] == pytest.approx(3.0, abs=0.21) for h in hits)
    # las personas que no se ven hace más de 10 s se olvidan
    run(det, [rec(10.0 + i * 0.5, []) for i in range(25)])
    assert det._win.people == {} and det.scores == {}
    # el reloj vuelve atrás (otra conexión de cero): empieza de nuevo sin fallar
    hits = run(det, scripted([(5, POCKET)], t0=0.0))
    assert [h["track_id"] for h in hits] == [1]
    det.zones = [{"name": "G", "type": "shelf", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}]
    assert det.zones and isinstance(det.zones[0], Zone)
    # consecutive=3: la tercera ventana seguida
    det3 = hi.OnlineTheftDetector(dict(POCKET_SPEC, consecutive=3))
    hits = run(det3, scripted([(10, POCKET)]))
    assert len(hits) == 1 and hits[0]["t1"] == pytest.approx(5.0, abs=0.21)


def test_en_vivo_da_lo_mismo_que_los_clips():
    """El detector en vivo (desde el seguimiento) y el análisis de un clip (desde los registros) miden lo mismo."""
    frames = scripted([(4, {}), (1, REACH), (0.4, BACK), (4, POCKET), (3, {})])
    model = hi.TheftModel(POCKET_SPEC)
    det = hi.OnlineTheftDetector(dict(POCKET_SPEC, threshold=0.0, consecutive=1, cooldown_s=0))
    live = {}
    for r in frames:
        tr = Track(1, tuple(r["people"][0]["box"]), "persona", 0.0, r["t"], hits=5)
        tr.kpts = np.array(r["people"][0]["kp"], np.float32)
        rr = hi.frame_record(r["t"], (W, H), [tr])
        det.update(rr["t"], rr["people"], rr["bags"], (W, H))
        if 1 in det.scores and det._win.ticked:
            live[round(rr["t"], 2)] = det.scores[1]
    offline = {round(w["t1"], 2): model.score(w["f"]) for w in hi.person_windows(frames)}
    assert live.keys() == offline.keys() and len(live) >= 8
    for k in live:
        assert live[k] == pytest.approx(offline[k], abs=1e-6)


# ---------------------------------------------------------------------------------------------- Analyzer y alarmas
class _Det:
    """Detector de prueba: siempre la misma persona."""

    def detect(self, frame):
        return [Detection(BOX, "persona", 0.9)]


class _Pose:
    """Postura de prueba: sigue un guion (una pose por cuadro) y cuenta cuántas veces se la pidió."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def estimate(self, frame, bbox):
        self.calls += 1
        return np.array(self.script.pop(0) if self.script else pose(**POCKET), np.float32)


def _run_analyzer(an, seconds=12.0, fps=5.0):
    img = np.zeros((H, W, 3), np.uint8)
    found = []
    for i in range(int(seconds * fps)):
        found += an.process(img, i / fps)
    return found


def test_analyzer_con_la_ia_de_hurtos():
    from core.analyzer import Analyzer

    est = _Pose([pose()] * 15)  # 3 s con los brazos colgando y después la mano en el bolsillo
    an = Analyzer({"zones": [], "ia_hurto": POCKET_SPEC}, _Det(), pose=est, clip_pre=1, clip_post=1)
    assert an.theft_ai and an.wants_pose and not an.gestures  # mira la postura aunque la cámara no tenga gestos
    found = _run_analyzer(an)
    assert est.calls > 0 and [f.type for f in found] == [HURTO_IA]
    f = found[0]
    assert f.severity == "alta" and f.alert is True and f.score is None and f.track_id >= 1
    assert f.snapshot_jpg[:2] == b"\xff\xd8" and "IA de movimientos" in f.description and "0," in f.description
    assert f.features["ia_score"] > 0.5 and f.features["ia_mano_cintura"] > 0.5
    assert f.features["track_age_s"] > 3  # y las medidas de siempre (para el filtro de alertas)
    clips = an.flush()
    assert [c.uuid for c in clips] == [f.uuid] and clips[0].frames
    # con el filtro de alertas de antes (no conoce el tipo): pasa igual, con puntaje 1
    from core.scorer import LEGACY_NAMES, Scorer

    n = len(LEGACY_NAMES)
    sc = Scorer({"vector_names": list(LEGACY_NAMES), "mean": [0.0] * n, "scale": [1.0] * n, "coef": [0.0] * n,
                 "intercept": -5.0, "threshold": 0.5})
    an2 = Analyzer({"zones": [], "ia_hurto": POCKET_SPEC}, _Det(), sc, pose=_Pose([pose()] * 15))
    found = _run_analyzer(an2)
    assert [(x.type, x.score, x.alert) for x in found] == [(HURTO_IA, 1.0, True)]


@pytest.mark.parametrize("spec", [None, "raro", 5, {}, dict(POCKET_SPEC, version=2), dict(POCKET_SPEC, names=[]),
                                  {"kind": "ia_hurto"}])
def test_analyzer_sin_ia_o_con_un_modelo_que_no_sirve_queda_como_siempre(spec):
    from core.analyzer import Analyzer

    est = _Pose([])
    cfg = {"zones": []} if spec is None else {"zones": [], "ia_hurto": spec}
    an = Analyzer(cfg, _Det(), pose=est)
    assert not an.theft_ai and not an.wants_pose and an.theft is None
    assert _run_analyzer(an) == [] and est.calls == 0  # ni siquiera mira la postura


def test_analyzer_solo_metricas_o_sin_postura_no_mira():
    from core.analyzer import Analyzer

    est = _Pose([])
    an = Analyzer({"zones": [], "ia_hurto": POCKET_SPEC}, _Det(), pose=est, clips=False)
    assert not an.theft_ai and not an.wants_pose
    assert _run_analyzer(an) == [] and est.calls == 0
    an2 = Analyzer({"zones": [], "ia_hurto": POCKET_SPEC}, _Det())  # sin el modelo de postura en esta máquina
    assert not an2.theft_ai and an2.wants_pose and _run_analyzer(an2) == []


def test_analyzer_prende_cambia_y_apaga_la_ia_en_caliente():
    from core.analyzer import Analyzer

    est = _Pose([pose()] * 10)
    an = Analyzer({"zones": []}, _Det(), pose=est)
    assert _run_analyzer(an, seconds=4) == [] and est.calls == 0
    assert an.set_theft(POCKET_SPEC) and an.theft_ai
    det = an.theft
    assert an.set_theft(json.loads(json.dumps(POCKET_SPEC))) and an.theft is det  # el mismo modelo: sigue lo medido
    img = np.zeros((H, W, 3), np.uint8)
    found = []
    for i in range(20, 80):
        found += an.process(img, i / 5)
    assert [f.type for f in found] == [HURTO_IA]
    # zonas nuevas en caliente (el CameraWorker reemplaza `zones`): la IA las toma sin rehacerse
    an.zones = parse_zones([{"name": "G", "type": "shelf", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}])
    an.process(img, 16.2)
    assert an.theft is det and [z.name for z in det.zones] == ["G"]
    assert an.set_theft(dict(POCKET_SPEC, threshold=0.9)) and an.theft is not det  # otro modelo: de cero
    assert not an.set_theft(None) and an.theft is None and not an.theft_ai
    calls = est.calls
    assert _run_analyzer(an, seconds=2) == [] and est.calls == calls  # apagada: no mira más la postura
    assert not an.set_theft(dict(POCKET_SPEC, version=7)) and an.theft is None


def test_los_gestos_siguen_andando_con_la_ia():
    from core.analyzer import Analyzer
    from core.gestos import MANO_BOLSILLO

    script = [pose()] * 10 + [pose(**REACH)] * 3 + [pose(**BACK)] + [pose(**POCKET)] * 8 + [pose()] * 40
    never = handmade({}, -20.0)  # una IA que nunca avisa
    cfg = {"zones": [], "gestures": {"enabled": ["mano_bolsillo", "mano_ropa", "mano_bolsa"]}, "ia_hurto": never}
    an = Analyzer(cfg, _Det(), pose=_Pose(script))
    assert an.gestures and an.theft_ai
    assert [f.type for f in _run_analyzer(an)] == [MANO_BOLSILLO]
    # y con una IA que avisa, las dos alertas
    cfg["ia_hurto"] = POCKET_SPEC
    an = Analyzer(cfg, _Det(), pose=_Pose(script))
    assert sorted(f.type for f in _run_analyzer(an)) == sorted([MANO_BOLSILLO, HURTO_IA])


def _video(path: Path, seconds: float = 10.0, fps: int = 5) -> Path:
    import cv2

    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (W, H))
    for i in range(int(seconds * fps)):
        f = np.full((H, W, 3), 40, np.uint8)
        f[50:400, 40:160] = 120 + (i % 5)
        out.write(f)
    out.release()
    return path


def test_las_alarmas_usan_la_ia_de_hurtos(tmp_path):
    from core.clip_analysis import analyze_clip

    video = _video(tmp_path / "alarma.avi")
    est = _Pose([pose()] * 15)
    res = analyze_clip(str(video), {"zones": [], "rules": {}, "ia_hurto": POCKET_SPEC}, _Det(), process_fps=5,
                       pose=est)
    assert res.has_person and res.type == HURTO_IA and res.severity == "alta" and HURTO_IA in res.rules
    assert res.features["ia_score"] > 0.5 and res.snapshot_jpg[:2] == b"\xff\xd8" and est.calls > 0
    # sin modelo (o uno que no sirve, o sin postura): como siempre
    for cfg, p in (({"zones": [], "rules": {}}, _Pose([])), ({"ia_hurto": {"version": 3}}, _Pose([])),
                   ({"ia_hurto": POCKET_SPEC}, None)):
        res = analyze_clip(str(video), cfg, _Det(), process_fps=5, pose=p)
        assert res.type == "alarma_dvr" and HURTO_IA not in res.rules
        assert p is None or p.calls == 0
