"""«Sensibilidad para detectar personas» (1 a 10) y «Personas lejos» por cámara: la parte de core.

- core/deteccion.py: la configuración (Python puro: la importa el servidor de Vercel, que no tiene OpenCV).
- core/detectors.py: el umbral por llamada (sin tocar el detector compartido), el mosaico 2x2 y `tuned`.
- core/clip_analysis.py: las alarmas y los clips usan la detección de la cámara.

Se prueba con un «modelo» de mentira que ve manchas de colores en el lienzo que le llega (blanco = persona, rojo =
vehículo, azul = mochila) y les da más puntaje cuanto más grandes llegan: así se ve que la mancha chica del fondo
aparece al subir la sensibilidad o al mirar la imagen por partes (agrandadas), sin el modelo de verdad.
"""

import subprocess
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.deteccion import DEFAULT_SENSITIVITY, FAR_PASSES, DetectionConfig, conf_for
from core.detectors import (Detection, MotionDetector, OnnxYoloDetector, TunedDetector, far_tiles, merge_far,
                            tuned)

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "personas.onnx"


# ---------------------------------------------------------------- core/deteccion.py
def test_confianza_segun_la_sensibilidad():
    assert conf_for(5) == 0.35  # exacto: como antes (0,55 - 0,2 sin redondear da 0,35000000000000003)
    assert [conf_for(s) for s in range(1, 11)] == [0.51, 0.47, 0.43, 0.39, 0.35, 0.31, 0.27, 0.23, 0.19, 0.15]
    assert conf_for(0) == conf_for(1) and conf_for(99) == conf_for(10)  # se lleva a 1..10
    assert conf_for("x") == conf_for(None) == 0.35
    assert DetectionConfig().conf == 0.35 and DetectionConfig(10).conf == 0.15 and DEFAULT_SENSITIVITY == 5


def test_configuracion_tolerante_para_los_agentes():
    d = DetectionConfig.from_dict
    assert d(None) == d({}) == d([]) == d("8") == d(3) == DetectionConfig()
    assert d({"sensitivity": 8}) == DetectionConfig(8, False)
    assert d({"sensitivity": 7.0, "far": True}) == DetectionConfig(7, True)  # un decimal entero vale
    for raro in (0, 11, -3, 7.5, "7", True, False, [5], {"a": 1}):
        assert d({"sensitivity": raro}) == DetectionConfig(), raro  # lo raro queda en lo de siempre
    for raro in ("si", 1, 0, None, [True]):
        assert d({"far": raro}).far is False, raro
    assert d({"sensitivity": None, "far": None}) == DetectionConfig()
    assert d({"sensitivity": 9, "far": True, "clave_nueva": 3}) == DetectionConfig(9, True)  # claves nuevas: ignoradas


def test_configuracion_estricta_para_el_servidor():
    d = lambda x: DetectionConfig.from_dict(x, strict=True)  # noqa: E731
    assert d(None) == d({}) == DetectionConfig()
    assert d({"sensitivity": 10, "far": True}) == DetectionConfig(10, True)
    assert d({"sensitivity": 7.0}) == DetectionConfig(7)
    with pytest.raises(ValueError, match="va un objeto"):
        d([1, 2])
    with pytest.raises(ValueError, match="va un objeto"):
        d("8")
    for raro in (0, 11, "7", True, 7.5):
        with pytest.raises(ValueError, match="la sensibilidad va de 1 a 10"):
            d({"sensitivity": raro})
    for raro in ("si", 1, 0):
        with pytest.raises(ValueError, match="va true o false"):
            d({"far": raro})
    with pytest.raises(ValueError, match="claves desconocidas"):
        d({"sensitivity": 5, "umbral": 0.2})


def test_configuracion_compacta_completa_y_textos():
    assert DetectionConfig().to_dict(compact=True) == {}
    assert DetectionConfig(8).to_dict(compact=True) == {"sensitivity": 8}
    assert DetectionConfig(5, True).to_dict(compact=True) == {"far": True}
    assert DetectionConfig(3, True).to_dict(compact=True) == {"sensitivity": 3, "far": True}
    assert DetectionConfig().to_dict(compact=False) == {"sensitivity": 5, "far": False}
    assert DetectionConfig(8, True).to_dict(compact=False) == {"sensitivity": 8, "far": True}
    for cfg in (DetectionConfig(), DetectionConfig(8), DetectionConfig(1, True)):  # ida y vuelta
        assert DetectionConfig.from_dict(cfg.to_dict(compact=True), strict=True) == cfg
        assert DetectionConfig.from_dict(cfg.to_dict(compact=False), strict=True) == cfg
    assert DetectionConfig().is_default
    assert not DetectionConfig(6).is_default and not DetectionConfig(far=True).is_default
    assert DetectionConfig().describe() == "normal"
    assert DetectionConfig(8).describe() == "sensibilidad 8 de 10"
    assert DetectionConfig(far=True).describe() == "personas lejos"
    assert DetectionConfig(8, True).describe() == "sensibilidad 8 de 10 · personas lejos"
    assert DetectionConfig(far=True).passes == FAR_PASSES == 5 and DetectionConfig(9).passes == 1


def test_el_servidor_puede_importarla_sin_opencv_ni_numpy():
    code = ("import sys, core.deteccion as d; d.DetectionConfig.from_dict({'sensitivity': 8}, strict=True); "
            "print('cv2' in sys.modules, 'numpy' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["False", "False"]


# ---------------------------------------------------------------- un «modelo» de manchas
class BlobRunner:
    """Hace de la sesión ONNX: busca manchas en el lienzo de 640x640 (blanco = persona, rojo = vehículo, azul =
    mochila) y las devuelve como YOLO (4 + 80 clases, N cajas), con puntaje (alto en el lienzo / 100)²: una persona
    que llega más grande al modelo sale más segura, como pasa con el de verdad."""

    device, on_gpu, size, session = "prueba", False, (640, 640), None
    col_of = {c: 4 + c for c in range(80)}
    COLORS = ((0, lambda c: (c[:, :, 0] > 200) & (c[:, :, 1] > 200) & (c[:, :, 2] > 200)),
              (2, lambda c: (c[:, :, 2] > 200) & (c[:, :, 1] < 60) & (c[:, :, 0] < 60)),
              (24, lambda c: (c[:, :, 0] > 200) & (c[:, :, 1] < 60) & (c[:, :, 2] < 60)))

    def __init__(self):
        self.calls = 0
        self._lock = threading.Lock()

    def run(self, canvas):
        with self._lock:
            self.calls += 1
        out = np.zeros((84, 200), np.float32)  # como el modelo: más cajas que clases
        n = 0
        for cls, mask_of in self.COLORS:
            mask = mask_of(canvas).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                out[:4, n] = (x + w / 2, y + h / 2, w, h)
                out[4 + cls, n] = min(0.95, (h / 100.0) ** 2)
                n += 1
        return out


W, H = 1280, 720  # cuadro de la cámara: el modelo lo ve a la mitad (r = 0,5); cada parte, a 0,875


def blob(img, box, color=(255, 255, 255)):
    x1, y1, x2, y2 = box
    img[y1:y2, x1:x2] = color


def scene(*boxes, color=(255, 255, 255)):
    img = np.zeros((H, W, 3), np.uint8)
    for b in boxes:
        blob(img, b, color)
    return img


FAR_ONE = (100, 600, 140, 680)  # 40x80 abajo a la izquierda: imagen entera 0,16; en su parte, 0,49
BIG = (900, 100, 1020, 400)  # 120x300: se ve siempre (0,95)
CENTER = (610, 320, 650, 400)  # chica (como FAR_ONE), en la zona donde se pisan las 4 partes


def people(dets):
    return [d for d in dets if d.label == "persona"]


def near(box, want, tol=4):
    return all(abs(a - b) <= tol for a, b in zip(box, want))


def test_el_umbral_va_por_llamada_y_el_detector_compartido_no_cambia():
    det = OnnxYoloDetector(session=BlobRunner())
    frame = scene(FAR_ONE, BIG)
    assert [near(d.bbox, BIG) for d in det.detect(frame)] == [True]  # 0,35: la chica (0,16) no
    seen = det.detect(frame, conf=0.15)  # sensibilidad 10
    assert len(seen) == 2 and any(near(d.bbox, FAR_ONE) for d in seen)
    assert det.conf == 0.35 and len(det.detect(frame)) == 1  # nada quedó cambiado
    assert det.detect(frame, conf=0.51) == det.detect(frame)  # la grande pasa igual con sensibilidad 1


def test_los_vehiculos_siguen_con_su_umbral():
    det = OnnxYoloDetector(session=BlobRunner(), classes={0: "persona", 2: "vehiculo"})
    frame = scene(FAR_ONE)
    blob(frame, (600, 100, 640, 180), (0, 0, 255))  # un vehículo chico: 0,16
    labels = sorted(d.label for d in det.detect(frame, conf=0.15))
    assert labels == ["persona"]  # la sensibilidad es de las personas: el vehículo sigue pidiendo 0,35
    blob(frame, (800, 100, 900, 300), (0, 0, 255))  # uno grande
    assert sorted(d.label for d in det.detect(frame, conf=0.15)) == ["persona", "vehiculo"]


def test_las_partes_estan_bien_armadas():
    tiles = far_tiles(W, H)
    assert len(tiles) == 4
    cover = np.zeros((H, W), np.uint8)
    for x1, y1, x2, y2 in tiles:
        assert 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H
        cover[y1:y2, x1:x2] += 1
    assert cover.min() == 1  # cubren todo el cuadro
    a, _, b, _ = tiles[0]  # arriba a la izquierda
    c, _, _, _ = tiles[1]  # arriba a la derecha
    assert abs((b - c) / (b - a) - 0.25) < 0.01  # dos vecinas se pisan el 25 % de la parte
    assert abs((b - a) / W - 0.571) < 0.01  # cada parte, el 57 % del ancho
    assert far_tiles(40, 30) == []  # un cuadro diminuto no se parte


def test_personas_lejos_suma_las_que_solo_se_ven_en_las_partes():
    runner = BlobRunner()
    det = OnnxYoloDetector(session=runner)
    frame = scene(FAR_ONE, BIG)
    runner.calls = 0
    dets = det.detect(frame, far=True)
    assert runner.calls == FAR_PASSES  # la imagen entera y las 4 partes
    assert len(dets) == 2
    assert near(dets[0].bbox, BIG)  # primero lo de la imagen entera
    lejos = dets[1]
    assert near(lejos.bbox, FAR_ONE) and lejos.conf > 0.35  # en coordenadas del cuadro, agrandada en su parte
    assert len(det.detect(frame)) == 1  # sin «lejos», como siempre


def test_la_que_esta_donde_se_pisan_las_partes_sale_una_sola_vez():
    det = OnnxYoloDetector(session=BlobRunner())
    dets = det.detect(scene(CENTER), far=True)
    assert len(dets) == 1 and near(dets[0].bbox, CENTER)  # la ven las 4 partes: una sola persona


def test_una_grande_cortada_entre_partes_sale_una_sola_vez():
    det = OnnxYoloDetector(session=BlobRunner())
    grande = (560, 150, 700, 560)  # cruza las 4 partes: en cada una queda cortada por el borde interno
    dets = det.detect(scene(grande), far=True)
    assert len(dets) == 1 and near(dets[0].bbox, grande)  # gana la de la imagen entera


def test_con_lejos_las_partes_solo_suman_personas():
    runner = BlobRunner()
    det = OnnxYoloDetector(session=runner, classes={0: "persona", 2: "vehiculo"})
    frame = scene(FAR_ONE)
    blob(frame, (1100, 600, 1140, 680), (0, 0, 255))  # vehículo chico: en su parte llegaría a 0,49
    blob(frame, (700, 600, 740, 680), (255, 0, 0))  # mochila chica: igual
    dets = det.detect(frame, far=True)
    assert [d.label for d in dets] == ["persona"]  # el vehículo y la mochila salen solo de la imagen entera
    runner.calls = 0
    main, extra = det.detect_also(frame, {24: "mochila"}, 0.3, far=True)
    assert runner.calls == FAR_PASSES and [d.label for d in main] == ["persona"]
    assert extra == det.detect_also(frame, {24: "mochila"}, 0.3)[1] == []  # lo demás, de la imagen entera
    main, (bags,) = det.detect_groups(frame, [({24: "mochila"}, 0.3)], far=True)
    assert [d.label for d in main] == ["persona"] and bags == []


def test_detect_also_y_detect_groups_con_el_umbral_de_la_camara():
    det = OnnxYoloDetector(session=BlobRunner())
    frame = scene(FAR_ONE, BIG)
    blob(frame, (300, 300, 380, 420), (255, 0, 0))  # mochila: 120 px → en el lienzo 60 → 0,36
    main, extra = det.detect_also(frame, {24: "mochila"}, 0.3, person_conf=0.15)
    assert len(people(main)) == 2 and [d.label for d in extra] == ["mochila"]
    main, extra = det.detect_also(frame, {24: "mochila"}, 0.3)
    assert len(main) == 1 and len(extra) == 1  # la mochila no depende del umbral de las personas
    main, (bags,) = det.detect_groups(frame, [({24: "mochila"}, 0.3)], person_conf=0.15)
    assert len(main) == 2 and len(bags) == 1


def test_juntar_lo_de_la_imagen_entera_con_lo_de_las_partes():
    full = [Detection((100, 100, 200, 400), "persona", 0.9), Detection((500, 300, 900, 600), "vehiculo", 0.8)]
    parts = [
        Detection((102, 98, 201, 398), "persona", 0.95),  # la misma de la imagen entera: gana la de la imagen entera
        Detection((100, 250, 200, 400), "persona", 0.6),  # medio cuerpo de la misma (cortada): adentro → no
        Detection((1000, 50, 1030, 110), "persona", 0.5),  # una nueva
        Detection((1001, 52, 1031, 111), "persona", 0.4),  # la misma nueva vista en otra parte: no se repite
        Detection((600, 350, 640, 450), "persona", 0.5),  # alguien dentro del auto: no se compara con el vehículo
    ]
    out = merge_far(full, parts)
    assert [(d.bbox, d.conf) for d in out] == [
        ((100, 100, 200, 400), 0.9), ((500, 300, 900, 600), 0.8), ((1000, 50, 1030, 110), 0.5),
        ((600, 350, 640, 450), 0.5)]
    assert merge_far(full, []) == full and merge_far([], []) == []


# ---------------------------------------------------------------- tuned
class SinAjuste:
    """Un detector de prueba como los de siempre: solo `detect(frame)`."""

    def detect(self, frame):
        return []


def test_tuned_no_envuelve_lo_que_no_hace_falta():
    base = OnnxYoloDetector(session=BlobRunner())
    assert tuned(base, None) is base and tuned(base, {}) is base and tuned(base, DetectionConfig()) is base
    assert tuned(base, {"sensitivity": 5, "far": False}) is base
    assert tuned(base, {"sensitivity": "raro"}) is base  # tolerante: lo de siempre
    assert tuned(base, 42) is base
    motion = MotionDetector()
    assert tuned(motion, {"sensitivity": 10, "far": True}) is motion  # el de movimiento no tiene umbral
    falso = SinAjuste()
    assert tuned(falso, DetectionConfig(9, True)) is falso
    assert tuned(None, DetectionConfig(9)) is None


def test_tuned_arma_sobre_el_de_abajo_y_pasa_sus_datos():
    base = OnnxYoloDetector(session=BlobRunner())
    t = tuned(base, {"sensitivity": 8})
    assert isinstance(t, TunedDetector) and t.base is base and t.conf == 0.23 and t.cfg == DetectionConfig(8)
    assert (t.name, t.device, t.on_gpu, t.size) == (base.name, base.device, base.on_gpu, base.size)
    assert t.runner is base.runner and t.classes == base.classes  # lo demás, del de abajo
    assert hasattr(t, "detect_also") and hasattr(t, "detect_groups") and t.tunable
    t2 = tuned(t, {"far": True})
    assert t2.base is base and t2.cfg == DetectionConfig(5, True)  # no se envuelve dos veces
    assert tuned(t, None) is base and tuned(t, DetectionConfig()) is base
    assert tuned(t, DetectionConfig(8)) is t  # la misma configuración: el mismo objeto
    assert "sensibilidad 8" in repr(t)


def test_tuned_aplica_el_umbral_y_el_mosaico():
    runner = BlobRunner()
    base = OnnxYoloDetector(session=runner)
    frame = scene(FAR_ONE, BIG)
    assert len(tuned(base, {"sensitivity": 10}).detect(frame)) == 2  # 0,15: ve la chica en la imagen entera
    assert len(tuned(base, {"sensitivity": 1}).detect(frame)) == 1
    runner.calls = 0
    lejos = tuned(base, {"far": True})
    assert len(lejos.detect(frame)) == 2 and runner.calls == FAR_PASSES
    main, extra = lejos.detect_also(frame, {24: "mochila"}, 0.3)
    assert len(main) == 2 and extra == []
    main, groups = lejos.detect_groups(frame, [({24: "mochila"}, 0.3)])
    assert len(main) == 2 and groups == [[]]
    assert base.conf == 0.35 and len(base.detect(frame)) == 1  # el compartido sigue igual


def test_dos_camaras_con_distinta_sensibilidad_sobre_el_mismo_detector():
    base = OnnxYoloDetector(session=BlobRunner())
    frame = scene(FAR_ONE, BIG)
    alta, baja = tuned(base, {"sensitivity": 10}), tuned(base, {"sensitivity": 1})
    got: dict[str, list[int]] = {"alta": [], "baja": []}
    errors: list = []

    def run(name, det):
        try:
            for _ in range(30):
                got[name].append(len(det.detect(frame)))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    hilos = [threading.Thread(target=run, args=(n, d)) for n, d in (("alta", alta), ("baja", baja))]
    for h in hilos:
        h.start()
    for h in hilos:
        h.join()
    assert not errors
    assert got["alta"] == [2] * 30 and got["baja"] == [1] * 30  # cada una con lo suyo, aunque corran a la vez
    assert base.conf == 0.35


def test_los_bolsos_de_los_gestos_con_el_detector_de_la_camara():
    from core.pose import detect_with_bags

    base = OnnxYoloDetector(session=BlobRunner())
    frame = scene(FAR_ONE, BIG)
    blob(frame, (300, 300, 380, 420), (255, 0, 0))  # mochila
    dets, bags = detect_with_bags(tuned(base, {"sensitivity": 10}), frame, True)
    assert len(dets) == 2 and [b.label for b in bags] == ["mochila"]
    dets, bags = detect_with_bags(tuned(base, {"sensitivity": 10}), frame, False)
    assert len(dets) == 2 and bags == []
    dets, bags = detect_with_bags(base, frame, True)
    assert len(dets) == 1 and len(bags) == 1


def test_el_envoltorio_sabe_hacer_lo_mismo_que_el_de_abajo():
    """El programa pregunta con `hasattr` si un detector da objetos o grupos (core/pose, «Configurar…», «Ver con la
    IA»): el envoltorio no tiene que decir que sabe algo que el de abajo no sabe."""

    class SoloPersonas:
        tunable = True

        def detect(self, frame, conf=None, far=False):
            return [Detection((0, 0, 1, 1), "persona", conf)]

    class ConObjetos(SoloPersonas):
        def detect_also(self, frame, classes, conf, person_conf=None, far=False):
            return self.detect(frame, person_conf, far), []

    t = tuned(SoloPersonas(), {"sensitivity": 9})
    assert not hasattr(t, "detect_also") and not hasattr(t, "detect_groups")
    assert t.detect(None)[0].conf == 0.19
    t = tuned(ConObjetos(), {"sensitivity": 9})
    assert hasattr(t, "detect_also") and not hasattr(t, "detect_groups")
    assert t.detect_also(None, {}, 0.3)[0][0].conf == 0.19
    t = tuned(OnnxYoloDetector(session=BlobRunner()), {"sensitivity": 9})
    assert hasattr(t, "detect_also") and hasattr(t, "detect_groups")


# ---------------------------------------------------------------- alarmas y clips
class Anota:
    """Detector de prueba que acepta el ajuste y anota con qué umbral y si con mosaico lo llamaron."""

    tunable = True
    name, device, on_gpu, size = "prueba", "procesador", False, (640, 640)

    def __init__(self):
        self.calls: list[tuple] = []

    def detect(self, frame, conf=None, far=False):
        self.calls.append(("detect", conf, far))
        return []

    def detect_also(self, frame, classes, conf, person_conf=None, far=False):
        self.calls.append(("detect_also", person_conf, far))
        return [], []


def _video(path: Path, n: int = 12) -> Path:
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 180))
    for i in range(n):
        f = np.zeros((180, 320, 3), np.uint8)
        f[60:120, 10 + 10 * i:30 + 10 * i] = 255
        out.write(f)
    out.release()
    return path


def test_las_alarmas_y_los_clips_usan_la_deteccion_de_la_camara(tmp_path):
    from core.clip_analysis import analyze_clip

    video = _video(tmp_path / "clip.avi")
    det = Anota()
    res = analyze_clip(str(video), {"zones": [], "rules": {}, "detection": {"sensitivity": 10, "far": True}}, det,
                       process_fps=10)
    assert res.frames > 0 and det.calls
    assert {c[1:] for c in det.calls} == {(0.15, True)}
    det2 = Anota()
    analyze_clip(str(video), {"zones": [], "rules": {}}, det2, process_fps=10)  # sin ajuste: como siempre
    assert {c for c in det2.calls} == {("detect", None, False)}
    det3 = Anota()
    analyze_clip(str(video), {"zones": [], "rules": {}, "detection": {"sensitivity": "raro"}}, det3, process_fps=10)
    assert {c for c in det3.calls} == {("detect", None, False)}  # un valor raro no traba la alarma
    # con el de movimiento, nada cambia
    res = analyze_clip(str(video), {"detection": {"sensitivity": 10, "far": True}}, MotionDetector(), process_fps=10)
    assert res.frames > 0


# ---------------------------------------------------------------- con el modelo de verdad
def test_con_el_modelo_de_verdad_lejos_corre_5_pasadas():
    pytest.importorskip("onnxruntime")
    if not MODEL.exists():
        pytest.skip("sin modelo")
    from core.detectors import OnnxSession

    sess = OnnxSession(MODEL, threads=1, max_batch=1)
    det = OnnxYoloDetector(session=sess)
    frame = np.zeros((720, 1280, 3), np.uint8)
    sess.take_stats()
    assert det.detect(frame, far=True) == [] and tuned(det, {"sensitivity": 10, "far": True}).detect(frame) == []
    assert sess.take_stats()[0] == 2 * FAR_PASSES
    assert det.detect_groups(frame, [({24: "mochila"}, 0.3)], person_conf=0.15, far=True) == ([], [[]])


def test_en_nvidia_las_5_pasadas_van_en_el_mismo_lote():
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    if not MODEL.exists():
        pytest.skip("sin modelo")
    from core.detectors import OnnxSession

    sess = OnnxSession(MODEL, threads=1, max_batch=8, max_wait=0.2)
    det = OnnxYoloDetector(session=sess)
    frame = np.full((720, 1280, 3), 120, np.uint8)
    sess.take_stats()
    uno = OnnxYoloDetector(MODEL)
    assert det.detect(frame, conf=0.15, far=True) == uno.detect(frame, conf=0.15, far=True) == []
    frames, avg = sess.take_stats()
    assert frames == FAR_PASSES and avg == FAR_PASSES  # un solo lote con la imagen entera y sus 4 partes
