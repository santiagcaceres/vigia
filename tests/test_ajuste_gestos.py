"""«Configurar…» con gestos de ocultamiento: el Tuner mira la postura una sola vez por cuadro, da las alertas de gesto
(de prueba), recalcula al cambiar gestos o segundos sin volver a mirarla, explica por qué dio o no, avisa si la cámara
no sirve para gestos y dibuja el esqueleto. Poses armadas a mano como en tests/test_gestos.py (persona de frente,
torso de 100 px) y un detector y una postura de prueba."""

from types import SimpleNamespace

import numpy as np
import pytest

from core import synthetic
from core.detectors import Detection
from core.gestos import MANO_BOLSA, MANO_BOLSILLO, MANO_ROPA

ALL = [MANO_BOLSILLO, MANO_ROPA, MANO_BOLSA]
BBOX = (40.0, 50.0, 160.0, 400.0)  # 350 px de alto
W, H = 200, 420


def pose(lw=None, rw=None, le=None, re=None, conf=0.9):
    """Persona de frente: hombros (80,100)-(120,100), caderas (85,200)-(115,200) → torso de 100 px. Por defecto los
    brazos cuelgan. lw/rw: muñeca izquierda/derecha (de la persona: la izquierda queda a la derecha de la imagen)."""
    kp = np.zeros((17, 3), np.float32)
    pts = {0: (100, 70), 1: (95, 65), 2: (105, 65), 5: (120, 100), 6: (80, 100), 11: (115, 200), 12: (85, 200),
           13: (115, 280), 14: (85, 280), 15: (115, 360), 16: (85, 360),
           7: le or (128, 150), 8: re or (72, 150), 9: lw or (130, 205), 10: rw or (70, 205)}
    for i, (x, y) in pts.items():
        kp[i] = (x, y, conf)
    return kp


REACH_R = dict(rw=(-10, 120), re=(30, 112))  # brazo derecho estirado hacia el costado, a la altura del pecho
BACK_R = dict(rw=(95, 150), re=(70, 140))  # la mano vuelve al cuerpo
POCKET_R = dict(rw=(83, 205), re=(70, 160))  # muñeca derecha junto a la cadera derecha, codo doblado
POCKET_L = dict(lw=(117, 205), le=(130, 160))
TAKE = [pose()] * 10 + [pose(**REACH_R)] * 3 + [pose(**BACK_R)]  # parada 2 s, estira el brazo y lo trae
TAKE_POCKET = TAKE + [pose(**POCKET_R)] * 8


class _Det:
    """Detector de prueba: siempre la misma persona (y cuenta las pasadas)."""

    def __init__(self, bbox=BBOX):
        self.bbox, self.calls = bbox, 0

    def detect(self, frame):
        self.calls += 1
        return [Detection(self.bbox, "persona", 0.9)]


class _DetWithBag(_Det):
    """Como YOLO completo: en la misma pasada, una cartera colgada del lado derecho (confianza baja: 0,35)."""

    BAG = (20.0, 190.0, 75.0, 260.0)

    def detect_also(self, frame, classes, conf):
        self.conf = conf
        return self.detect(frame), [Detection(self.BAG, "cartera", 0.35)] if conf <= 0.35 else []


class _Pose:
    """Postura de prueba: sigue un guion de poses, una por llamada (y cuenta las llamadas)."""

    def __init__(self, frames):
        self.frames, self.calls = list(frames), 0

    def estimate(self, frame, bbox):
        self.calls += 1
        return self.frames.pop(0) if self.frames else pose()


def _video(path, n: int) -> str:
    path.mkdir(parents=True, exist_ok=True)
    return str(synthetic.write_video([np.zeros((H, W, 3), np.uint8)] * n, path / "clip.mp4", fps=5))


def _cfg(gestures=ALL, zones=(), **extra) -> dict:
    analysis = {"zones": list(zones), "rules": {}, **extra}
    if gestures:
        analysis["gestures"] = {"enabled": list(gestures)}
    return {"id": 44, "name": "Pasillo bebidas", "client_name": "Minas", "process_fps": 5, "connection": {},
            "analysis": analysis}


def _tuner(path, script, gestures=ALL, det=None, **extra):
    from agent.ajuste import Tuner

    est = _Pose(script)
    # las 2 primeras vueltas el seguimiento todavía no confirma a la persona: no se le mira la postura
    t = Tuner(_cfg(gestures, **extra), det or _Det(), [_video(path, len(script) + 2)], pose=est)
    t.run()  # en la app corre en otro hilo
    return t, est


def test_da_la_alerta_de_gesto_y_recalcula_sin_volver_a_mirar_la_postura(tmp_path):
    det = _Det()
    t, est = _tuner(tmp_path, TAKE_POCKET, det=det)
    assert t.state == "listo"
    assert [a.type for a in t.alerts] == [MANO_BOLSILLO]
    a = t.alerts[0]
    assert "posible ocultamiento" in a.description and a.features["gesture_dwell_s"] >= 0.6
    smp = t.sample_at(a.index)
    assert smp.posed and smp.tracks[0].kpts is not None  # la postura queda guardada con el cuadro
    assert t.render(a.index) is not None
    assert t.marks[a.index - t.base][1]["gesture"]["hands"]["der"][0] == "bolsillo"

    calls, dcalls = est.calls, det.calls
    assert t.set_gestures({"enabled": [MANO_ROPA]}) is False  # no hace falta volver a analizar
    assert t.alerts == []
    t.set_gestures({"enabled": ALL, "pocket_seconds": 5.0})
    assert t.alerts == []  # estuvo 1,4 s con la mano en el bolsillo
    # «Probar estos tiempos» manda tiempos y gestos juntos: una sola vuelta de reglas
    t.set_times({"loiter_seconds": 30}, gestures={"enabled": ALL, "pocket_seconds": 0.6})
    assert [a.type for a in t.alerts] == [MANO_BOLSILLO]
    assert (est.calls, det.calls) == (calls, dcalls)  # ni la postura ni el detector se volvieron a correr
    # los gestos nunca van dentro de las reglas (un agente viejo dejaría de analizar la cámara)
    assert t.analysis["rules"] == {"loiter_seconds": 30} and t.analysis["gestures"]["enabled"] == ALL
    t.set_gestures(None)
    assert t.alerts == [] and "gestures" not in t.analysis


def test_explica_por_que_un_gesto_no_dio_alerta(tmp_path):
    # la mano al bolsillo sin haber tomado nada antes: no avisa, y dice por qué
    t, _ = _tuner(tmp_path / "a", [pose()] * 3 + [pose(**POCKET_R)] * 12)
    assert t.alerts == []
    why = " ".join(t.explain(0))
    assert "Gestos: postura bien vista en el" in why and "mano al bolsillo hasta" in why
    assert "no se vio que tomara algo antes" in why and "«Pedir que tome algo antes»" in why
    # sin pedir la toma, la misma mano da la alerta (sin volver a mirar la postura)
    t.set_gestures({"enabled": ALL, "require_take": False})
    assert [a.type for a in t.alerts] == [MANO_BOLSILLO]
    # las dos manos a la vez al bolsillo: acomodarse
    t2, _ = _tuner(tmp_path / "b", TAKE + [pose(**POCKET_R, **POCKET_L)] * 8)
    assert t2.alerts == []
    why2 = " ".join(t2.explain(0))
    assert "1 toma" in why2 and "las dos manos a la vez" in why2


def test_mano_en_la_bolsa_con_la_cartera_que_ve_yolo(tmp_path):
    det = _DetWithBag()
    t, _ = _tuner(tmp_path, TAKE + [pose(rw=(55, 225), re=(45, 170))] * 8, det=det)  # codo doblado: guardando
    assert det.conf == 0.3  # con gestos, los bolsos se buscan con la confianza de las máquinas
    assert [a.type for a in t.alerts] == [MANO_BOLSA] and t.alerts[0].features["bag_seen"] == 1.0
    hand = t.marks[t.alerts[0].index - t.base][1]["gesture"]["hands"]["der"]
    assert hand[0] == "bolsa" and hand[2] == 1.0  # con la cartera a la vista no se pide el doble


def test_gestos_activados_despues_piden_volver_a_analizar(tmp_path):
    t, est = _tuner(tmp_path, TAKE_POCKET, gestures=())
    assert est.calls == 0 and t.alerts == []  # sin gestos no se mira la postura (cuesta)
    assert all(not s.posed for s in t.samples)
    assert t.set_gestures({"enabled": ALL}) is True  # lo analizado no tiene la postura: la ventana lo vuelve a analizar
    assert any("sin mirar la postura" in x for x in t.explain(0))


def test_sin_modelo_de_postura_sigue_sin_gestos_y_lo_dice(tmp_path, monkeypatch):
    import agent.ajuste as aj

    monkeypatch.setattr(aj, "build_pose", lambda *a, **k: None)
    t = aj.Tuner(_cfg(), _Det(), [_video(tmp_path, 15)])
    t.run()
    assert t.state == "listo" and t.alerts == [] and t.pose_missing and not t.needs_pose()
    assert any("modelo de postura" in h for h in t.hints())
    assert any("modelo de postura" in x for x in t.explain(0))


def test_avisa_si_la_camara_no_sirve_para_los_gestos(tmp_path):
    # personas chicas (70 px de alto): no se les mira la postura
    t, est = _tuner(tmp_path / "a", [pose()] * 25, det=_Det((40.0, 50.0, 80.0, 120.0)))
    assert est.calls == 0
    assert any("menos del 25 % del alto" in h for h in t.hints())
    assert t.stats()["pose"]["min_pct"] == 25.0
    # sin postura vista no tiene sentido sugerir que no pida la toma: se dice que no se vio la postura
    why = " ".join(t.explain(0))
    assert "No se le vio la postura a nadie" in why and "«Pedir que tome algo antes»" not in why
    # postura mal vista (hombros y caderas con confianza baja: tapados)
    t2, _ = _tuner(tmp_path / "b", [pose(conf=0.2)] * 25)
    assert t2.stats()["pose"]["ok_pct"] == 0
    assert any("los gestos no van a andar bien" in h for h in t2.hints())
    # con zonas pero sin góndola: cualquier brazo estirado cuenta como toma
    t3, _ = _tuner(tmp_path / "c", [pose()] * 5, zones=[{"name": "Caja", "type": "checkout",
                                                         "polygon": [[0, 0], [1, 0], [1, 0.2], [0, 0.2]]}])
    assert any("Sin zona «Góndola», cualquier brazo" in h for h in t3.hints())


def test_camara_solo_metricas_no_mira_gestos(tmp_path):
    t, est = _tuner(tmp_path, TAKE_POCKET, metrics_only=True)
    assert est.calls == 0 and t.alerts == [] and not t.gestures_on


def test_los_gestos_viajan_aparte_de_las_reglas():
    from agent.ajuste import gestures_blocked, with_gestures

    a = {"zones": [], "rules": {"enabled_rules": []}, "metrics_only": True, "plans": ["hurto"], "profile": "general"}
    b = with_gestures(a, {"enabled": [MANO_ROPA], "clothes_seconds": 1.2})
    # con las reglas apagadas a mano, mirar gestos la saca de «solo métricas» (como el servidor)
    assert b["gestures"]["enabled"] == [MANO_ROPA] and not b["metrics_only"]
    assert b["rules"] == {"enabled_rules": []}
    assert with_gestures(b, {"enabled": []})["metrics_only"] and "gestures" not in with_gestures(b, None)
    # cliente sin plan de hurto o tipo «Solo métricas»: nunca gestos
    sin_plan = {**a, "plans": ["comercio"]}
    assert gestures_blocked(sin_plan) and "gestures" not in with_gestures(sin_plan, {"enabled": ALL})
    assert gestures_blocked({**a, "profile": "metricas"}) and gestures_blocked(a) is None
    # lo que esta versión no conoce (de un servidor más nuevo) se deja pasar, como en las máquinas: «Probar estos
    # tiempos» no se traba; un gesto desconocido solo no prende nada
    c = with_gestures(a, {"enabled": [MANO_ROPA], "clave_nueva": 3})
    assert c["gestures"]["clave_nueva"] == 3 and not c["metrics_only"]
    assert "gestures" not in with_gestures(a, {"enabled": ["mano_magica"]})
    with pytest.raises(ValueError):  # un tiempo fuera de rango sí es error
        with_gestures(a, {"enabled": [MANO_ROPA], "clothes_seconds": -1})


def test_guardar_solo_manda_lo_que_cambio():
    from agent.ajuste import gestures_to_save

    secs = {"pocket_seconds": 0.6, "clothes_seconds": 0.8, "bag_seconds": 1.0, "require_take": True}
    eff_all = {"enabled": ALL}  # lo que rige: los 3 por el tipo de cámara, tiempos de siempre (compacto)
    # nada cambió: no se manda nada (antes se fijaban la lista y los tiempos de siempre)
    assert gestures_to_save(None, {"enabled": ALL, **secs}, eff_all, ALL) is None
    # un tiempo cambiado: solo ese; la lista sigue siendo la del tipo
    assert gestures_to_save(None, {"enabled": ALL, **secs, "pocket_seconds": 1.2}, eff_all, ALL) == {
        "pocket_seconds": 1.2}
    # se saca un gesto: se fija la lista
    assert gestures_to_save(None, {"enabled": [MANO_BOLSILLO, MANO_ROPA], **secs}, eff_all, ALL) == {
        "enabled": [MANO_BOLSILLO, MANO_ROPA]}
    # apagados a mano y se prende uno: la lista elegida, y lo que había guardado queda
    raw = {"enabled": [], "bag_seconds": 1.5}
    assert gestures_to_save(raw, {"enabled": [MANO_BOLSA], **secs, "bag_seconds": 1.5}, None, ALL) == {
        "enabled": [MANO_BOLSA], "bag_seconds": 1.5}
    # lo guardado que esta versión no toca (o no conoce) no se pierde
    raw = {"take_window": 8.0, "clave_nueva": 1}
    assert gestures_to_save(raw, {"enabled": ALL, **secs, "require_take": False}, {**eff_all, "take_window": 8.0},
                            ALL) == {"take_window": 8.0, "clave_nueva": 1, "require_take": False}
    # servidor viejo (sin la lista del tipo): cambiar la lista la fija
    assert gestures_to_save(None, {"enabled": [MANO_ROPA], **secs}, eff_all, None) == {"enabled": [MANO_ROPA]}


def test_tocar_un_tiempo_sin_probarlo_cuenta_como_cambio(monkeypatch):
    """«Aprobar» guarda y «Cerrar» pregunta también si se cambió un tiempo o un gesto sin «Probar estos tiempos»."""
    from agent.oficina import TuneWindow

    class _Var:
        def __init__(self, v):
            self.v = v

        def get(self):
            return self.v

        def set(self, v):
            self.v = v

    w = TuneWindow.__new__(TuneWindow)
    w.tv = {"loiter_seconds": _Var("45")}
    w.gv = {MANO_BOLSILLO: _Var(True)}
    w.gs = {"pocket_seconds": _Var("0.6")}
    w.g_take = _Var(True)
    w._tested, w._saved_view = False, None
    assert not w.dirty  # todavía no se cargó la cámara
    w._saved_view = w._view()
    assert not w.dirty
    w.gs["pocket_seconds"].set("1.2")
    assert w.dirty
    w.gs["pocket_seconds"].set("0.6")
    assert not w.dirty  # volvió a lo guardado
    w._tested = True  # probado sin guardar
    assert w.dirty


def test_los_bolsos_salen_como_en_las_maquinas(tmp_path):
    """Con gestos, el Tuner pide los bolsos aparte (la clase de más puntaje entre mochila, cartera y valija, desde
    0,3), como core/pose.detect_with_bags en las máquinas, y lo demás que ve YOLO desde 0,45."""
    from core.pose import BAG_CLASSES

    class _DetGroups(_Det):
        def detect_also(self, frame, classes, conf):
            raise AssertionError("con grupos no se usa detect_also")

        def detect_groups(self, frame, groups):
            self.groups = groups
            (rest, rest_conf), (bags, bag_conf) = groups
            assert bags == BAG_CLASSES and bag_conf == 0.3 and rest_conf == 0.45
            assert not set(rest) & set(BAG_CLASSES)  # un bolso no sale dos veces
            return self.detect(frame), [[Detection((150.0, 10.0, 190.0, 40.0), "botella", 0.6)],
                                        [Detection(_DetWithBag.BAG, "cartera", 0.35)]]

    det = _DetGroups()
    t, _ = _tuner(tmp_path, TAKE + [pose(rw=(55, 225), re=(45, 170))] * 8, det=det)
    assert [a.type for a in t.alerts] == [MANO_BOLSA] and t.alerts[0].features["bag_seen"] == 1.0
    assert {o[0] for o in t.samples[-1].objects} == {"botella", "cartera"}


def test_dibuja_el_esqueleto_escalado_y_la_marcha_del_gesto():
    from agent.ajuste import Sample, TestAlert, draw
    from core.rules import parse_config
    from core.tracker import Track

    zones, cfg = parse_config({"zones": [], "gestures": {"enabled": ALL}})
    # el cuadro original era 4 veces más grande que la imagen que se guarda
    tr = Track(1, tuple(v * 4 for v in BBOX), "persona", 0.0, 1.0, hits=5)
    tr.kpts = pose(**POCKET_R) * np.array([4, 4, 1], np.float32)
    s = Sample(0, 1.0, None, b"", (W * 4, H * 4), [tr], posed=True)
    mark = {1: {"zones": [], "took": False,
                "gesture": {"hands": {"der": ("bolsillo", 0.4, 0.6)}, "reaching": [], "took_ago": 1.0}}}
    img = np.zeros((H, W, 3), np.uint8)
    hot = [TestAlert(0, 0, 1.0, None, MANO_BOLSILLO, "Persona #1: posible ocultamiento", 1)]
    out = draw(img, s, mark, zones, cfg, hot, "Góndolas.mp4 00:01")
    assert out.shape == img.shape
    assert tuple(out[205, 83]) == (0, 0, 255)  # la muñeca derecha en el bolsillo, en rojo y en su lugar escalado
    assert tuple(out[160, 70]) == (0, 140, 255)  # el codo, naranja: es la persona de la alerta
    # sin gesto, la misma persona: esqueleto amarillo, sin la muñeca en rojo
    out2 = draw(img, s, {1: {"zones": [], "took": False}}, zones, cfg, [], "x")
    assert tuple(out2[205, 83]) == (0, 220, 255)


def test_un_gesto_tiene_que_caer_cerca_del_momento_marcado():
    from agent.biblioteca import TestVideo, evaluate, suggested_label

    def alert(ts, type_):
        return SimpleNamespace(ts=ts, type=type_, track_id=1, wall=None)

    v = TestVideo("/v/pasillo.mp4", kind="hurto", marks=[16.0])
    near, far = alert(18.0, MANO_BOLSA), alert(40.0, MANO_BOLSILLO)
    assert evaluate(v, [near])["ok"] is True
    r = evaluate(v, [far])
    assert r["ok"] is False and len(r["extra"]) == 1  # un gesto 24 s después es otro momento
    assert evaluate(v, [alert(40.0, "merodeo")])["ok"] is True  # las reglas saltan después: ventana ancha
    assert suggested_label(v, near) == "hurto" and suggested_label(v, far) is None


def test_la_app_nombra_los_gestos():
    from agent.oficina import TYPE_NAMES

    assert TYPE_NAMES[MANO_BOLSILLO] == "Mano al bolsillo" and TYPE_NAMES[MANO_BOLSA] == "Mano en la bolsa"


def test_el_texto_nombra_solo_los_gestos_que_se_miran():
    from agent.ajuste import gesture_text
    from core.gestos import GestureConfig

    sm = {"frames": 10, "pose_ok_ratio": 1.0, "takes": 0, "max_dwell": {}, "suppressed": {}}
    txt = gesture_text(sm, 10, GestureConfig(enabled=(MANO_BOLSA,)))
    assert "nunca llevó la mano a la bolsa" in txt and "bolsillo" not in txt
    assert "nunca llevó la mano al bolsillo, a la ropa ni a la bolsa" in gesture_text(
        sm, 10, GestureConfig(enabled=tuple(ALL)))


def test_guardar_compara_contra_lo_guardado_y_no_contra_lo_probado():
    """Cámara con las reglas apagadas a mano: se marca un gesto, se prueba y se guarda → se tiene que mandar (antes,
    después de «Probar estos tiempos» se comparaba contra lo probado y no se guardaba nada)."""
    from agent.oficina import TuneWindow

    w = TuneWindow.__new__(TuneWindow)
    w.analysis = {"plans": ["hurto"], "profile": "general", "gestures": {"enabled": [MANO_BOLSILLO]}}  # lo probado
    w.info = {"raw": {"rules": {"enabled_rules": []}}}
    w._eff_gestures, w._type_gestures, w._gestures_unreadable = None, [], False  # lo que rige: ningún gesto
    secs = {"pocket_seconds": 0.6, "clothes_seconds": 0.8, "bag_seconds": 1.0, "require_take": True}
    assert w._gestures_save({"enabled": [MANO_BOLSILLO], **secs}) == {"enabled": [MANO_BOLSILLO]}
    w._gestures_unreadable = True  # los gestos guardados no se entendieron: no se tocan
    assert w._gestures_save({"enabled": [MANO_BOLSILLO], **secs}) is None
