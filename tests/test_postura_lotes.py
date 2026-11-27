"""La postura de todas las personas de un cuadro en una sola llamada al modelo (25 nov: en la máquina con placa de
video cada llamada cuesta procesador aunque el trabajo lo haga la placa; con 3 personas a la vista eran 3 por cuadro).
Tiene que dar lo mismo que de a una persona."""

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "postura.onnx"
PEOPLE = ROOT / "models" / "personas.onnx"

pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="sin el modelo de postura")


def bus():
    """La foto de prueba de Ultralytics (4 personas). `VIGIA_TEST_FOTO` = otra ruta a esa foto."""
    path = os.getenv("VIGIA_TEST_FOTO")
    if not path:
        try:
            import ultralytics

            path = str(Path(ultralytics.__file__).parent / "assets" / "bus.jpg")
        except ImportError:
            return None
    return cv2.imread(path) if Path(path).exists() else None


def test_el_modelo_por_lotes_da_lo_mismo_que_el_original():
    from core.detectors import OnnxSession
    from core.pose import _CpuRunner

    rng = np.random.default_rng(3)
    canvases = [cv2.GaussianBlur(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8), (0, 0), 5) for _ in range(5)]
    ref = _CpuRunner(MODEL, threads=1)
    sess = OnnxSession(MODEL, threads=1, max_batch=8, max_wait=0.05, all_outputs=True)
    assert sess.max_batch == 8 and sess.uint8 and sess.size == (256, 256)
    got = sess.run_many(canvases)
    for c, g in zip(canvases, got):
        want = ref.run(c)
        assert g.shape == want.shape == (56, 1344)  # caja, puntaje y 17 puntos: la salida entera, sin recortar
        assert float(np.abs(g - want).max()) < 0.01
    assert sess.batches == 1 and sess.frames == 5  # las cinco en una sola llamada


def test_la_sesion_de_postura_se_comprueba_sola_y_si_falla_sigue_de_a_una(monkeypatch, caplog):
    from core import pose

    monkeypatch.setenv("VIGIA_LOTE_POSTURA", "4")
    sess = pose._session(MODEL)
    assert sess.max_batch == 4 and sess.max_wait == pose.POSE_WAIT_S

    # un lote que diera otra cosa (placa o driver raro) no se usa: de a una persona, como siempre
    real = pose.OnnxSession.run_many
    monkeypatch.setattr(pose.OnnxSession, "run_many", lambda self, cs: [o + 5.0 for o in real(self, cs)])
    with caplog.at_level("WARNING", logger="vigia.pose"):
        sess = pose._session(MODEL)
    assert sess.max_batch == 1
    assert "no pasó la comprobación" in caplog.text

    monkeypatch.setenv("VIGIA_LOTE_POSTURA", "1")  # y se puede apagar a mano
    assert pose._session(MODEL).max_batch == 1


def test_varias_personas_en_una_llamada_dan_la_misma_postura(monkeypatch):
    frame = bus()
    if frame is None or not PEOPLE.exists():
        pytest.skip("sin la foto de prueba (VIGIA_TEST_FOTO) o sin el detector")
    from core import pose
    from core.detectors import OnnxSession, OnnxYoloDetector

    boxes = [d.bbox for d in OnnxYoloDetector(PEOPLE).detect(frame)]
    assert len(boxes) == 4
    one = pose.PoseEstimator(MODEL, runner=OnnxSession(MODEL, threads=1, max_batch=1))
    monkeypatch.setenv("VIGIA_LOTE_POSTURA", "8")
    sess = pose._session(MODEL)
    many = pose.PoseEstimator(MODEL, runner=sess)
    sess.take_stats()

    want = [one.estimate(frame, b) for b in boxes]
    got = many.estimate_many(frame, boxes)
    assert sess.take_stats() == (4, 4.0)  # una llamada con las cuatro
    assert sum(k is not None for k in want) >= 3
    for w, g in zip(want, got):
        assert (w is None) == (g is None)
        if w is not None:
            assert float(np.abs(w[:, :2] - g[:, :2]).max()) < 0.5  # píxeles del cuadro
            assert float(np.abs(w[:, 2] - g[:, 2]).max()) < 0.01  # confianza de cada punto

    # un recuadro que no sirve no rompe la tanda ni corre a los demás
    mixed = many.estimate_many(frame, [boxes[0], (10, 10, 11, 12), boxes[1]])
    assert mixed[1] is None
    assert np.allclose(mixed[0], got[0], atol=0.5) and np.allclose(mixed[2], got[1], atol=0.5)
    assert many.estimate_many(frame, []) == []

    # `look` (lo que usa el análisis) pide todas juntas y deja la postura en cada persona
    tracks = [SimpleNamespace(id=i, bbox=b, visible=True, label="persona", kpts=None) for i, b in enumerate(boxes)]
    tracks.append(SimpleNamespace(id=9, bbox=(0, 0, 20, 30), visible=True, label="persona", kpts="vieja"))  # chica
    sess.take_stats()
    n = pose.look(many, frame, tracks, min_px=90)
    assert n == sum(k is not None for k in want)
    assert sess.take_stats()[1] == len(boxes)
    assert tracks[-1].kpts is None  # a la que no se mira no le queda la postura de antes
    for t, w in zip(tracks, want):
        assert (t.kpts is None) == (w is None)


def test_si_el_lote_falla_andando_sigue_de_a_una(monkeypatch):
    frame = bus()
    if frame is None:
        pytest.skip("sin la foto de prueba (VIGIA_TEST_FOTO)")
    from core import pose
    from core.detectors import OnnxSession

    box = (40, 390, 240, 910)  # la persona de la izquierda de la foto
    sess = OnnxSession(MODEL, threads=1, max_batch=4, max_wait=0.002, all_outputs=True)
    est = pose.PoseEstimator(MODEL, runner=sess)
    want = est.estimate(frame, box)
    assert want is not None

    def roto(canvases):
        raise RuntimeError("la placa no quiso")

    monkeypatch.setattr(sess, "run_many", roto)
    got = est.estimate_many(frame, [box, box])
    assert all(g is not None and np.allclose(g, want, atol=0.5) for g in got)


def test_un_estimador_viejo_sin_lotes_sigue_sirviendo():
    """(los de las pruebas y el de «medir» tienen solo `estimate`)"""
    from core import pose

    class Solo:
        def __init__(self):
            self.calls = 0

        def estimate(self, frame, bbox):
            self.calls += 1
            return np.zeros((17, 3), np.float32)

    est = Solo()
    tracks = [SimpleNamespace(id=i, bbox=(0, 0, 50, 200), visible=True, label="persona", kpts=None) for i in range(3)]
    assert pose.look(est, np.zeros((360, 640, 3), np.uint8), tracks) == 3
    assert est.calls == 3 and all(t.kpts is not None for t in tracks)
    assert pose.look(None, np.zeros((360, 640, 3), np.uint8), tracks) == 0
    assert all(t.kpts is None for t in tracks)


def test_de_fabrica_solo_la_placa_nvidia_usa_lotes(monkeypatch):
    from core import detectors, pose

    monkeypatch.delenv("VIGIA_LOTE", raising=False)
    monkeypatch.setattr(detectors, "_providers", lambda ort: ["CPUExecutionProvider"])
    assert pose._auto_batch() == 1  # procesador (y la app de Windows con DirectML): de a una, como siempre
    monkeypatch.setattr(detectors, "_providers", lambda ort: ["DmlExecutionProvider", "CPUExecutionProvider"])
    assert pose._auto_batch() == 1
    monkeypatch.setattr(detectors, "_providers", lambda ort: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert pose._auto_batch() == pose.MAX_PER_FRAME  # justo los tamaños que se comprueban al arrancar
    monkeypatch.setenv("VIGIA_LOTE", "1")  # quien apagó los lotes del detector, apagó también estos
    assert pose._auto_batch() == 1


def test_si_el_proceso_se_cerro_preparando_el_lote_la_proxima_arranca_sin_el(tmp_path, monkeypatch, caplog):
    import os
    import time

    from core import pose

    mark = tmp_path / "datos" / "postura-lotes-1.intento"
    monkeypatch.setenv("VIGIA_MARCA_POSTURA", str(mark))
    monkeypatch.setenv("VIGIA_LOTE_POSTURA", "4")
    # lo normal: la marca existe solo mientras se arma y se comprueba
    seen = []
    real = pose.OnnxSession.run_many
    monkeypatch.setattr(pose.OnnxSession, "run_many", lambda self, cs: (seen.append(mark.exists()), real(self, cs))[1])
    assert pose._session(MODEL).max_batch == 4
    assert seen and all(seen) and not mark.exists()
    monkeypatch.setattr(pose.OnnxSession, "run_many", real)

    # el proceso se cerró en ese rato (quedó la marca): la vez siguiente, de a una persona, y lo dice
    mark.write_text("x", encoding="utf-8")
    with caplog.at_level("WARNING", logger="vigia.pose"):
        assert pose._session(MODEL).max_batch == 1
    assert "se cerró preparando la postura por lotes" in caplog.text
    assert mark.exists()  # sigue protegiendo en los arranques que vienen

    # pasada una semana se prueba de nuevo (por si fue otra cosa: un corte justo en ese momento)
    old = time.time() - pose.RETRY_AFTER_S - 60
    os.utime(mark, (old, old))
    assert pose._session(MODEL).max_batch == 4 and not mark.exists()

    # si la comprobación falla sin cerrar el proceso, la marca no queda (ya se sabe qué pasó: sigue de a una)
    monkeypatch.setattr(pose.OnnxSession, "run_many", lambda self, cs: [o + 5.0 for o in real(self, cs)])
    assert pose._session(MODEL).max_batch == 1 and not mark.exists()


# ------------------------------------------------- la postura con TensorRT (27 nov, `VIGIA_TENSORRT=todo`)
# En la máquina de AWS: 2,3 a 3,6 veces menos placa y procesador por persona, los puntos a 0,004 largos de torso en
# promedio. En media precisión no da exactamente lo mismo: se compara con el lote de CUDA ya comprobado.
def _fake_trt(monkeypatch, pose, change, device: str = "GPU (Tensorrt)"):
    real_init = pose.OnnxSession.__init__
    made = []

    def init(self, *a, trt=False, **k):
        real_init(self, *a, **k)
        if trt:  # (acá no hay placa: la sesión «TensorRT» es la de siempre con los números corridos)
            self.device = device
            made.append(self)
            real_many = pose.OnnxSession.run_many

            def many(cs, _self=self):
                return [change(o) for o in real_many(_self, cs)]
            self.run_many = many

    monkeypatch.setattr(pose.OnnxSession, "__init__", init)
    return made


def test_la_postura_usa_tensorrt_si_da_casi_lo_mismo_que_cuda(monkeypatch, caplog):
    from core import pose

    monkeypatch.setenv("VIGIA_LOTE_POSTURA", "4")
    monkeypatch.setenv("VIGIA_TENSORRT", "todo")
    made = _fake_trt(monkeypatch, pose, lambda o: o * 1.001)  # un milésimo distinto: como media precisión
    with caplog.at_level("INFO", logger="vigia.pose"):
        sess = pose._session(MODEL)
    assert made and sess is made[0] and sess.max_batch == 4
    assert "Postura con TensorRT" in caplog.text


def test_si_tensorrt_da_distinto_o_no_queda_sigue_con_el_lote_de_cuda(monkeypatch, caplog):
    from core import pose

    monkeypatch.setenv("VIGIA_LOTE_POSTURA", "4")
    monkeypatch.setenv("VIGIA_TENSORRT", "todo")
    made = _fake_trt(monkeypatch, pose, lambda o: o + 5.0)
    with caplog.at_level("WARNING", logger="vigia.pose"):
        sess = pose._session(MODEL)
    assert made and sess is not made[0] and sess.max_batch == 4  # el lote de CUDA, no de a una persona
    assert "TensorRT no pasó la comprobación" in caplog.text and made[0].session is None  # (la otra se soltó)
    made = _fake_trt(monkeypatch, pose, lambda o: o, device="GPU (CUDA)")  # TensorRT no quedó: tampoco
    sess = pose._session(MODEL)
    assert sess is not made[-1] and sess.max_batch == 4
    monkeypatch.setenv("VIGIA_TENSORRT", "1")  # solo el detector: la postura ni lo prueba
    n = len(made)
    pose._session(MODEL)
    assert len(made) == n


def test_una_sesion_cerrada_termina_su_hilo_de_lotes():
    import threading
    import time

    from core.detectors import OnnxSession

    sess = OnnxSession(MODEL, threads=1, max_batch=4, all_outputs=True)
    before = sum(t.name == "postura-lotes" for t in threading.enumerate())
    sess.close()
    for _ in range(50):
        if sum(t.name == "postura-lotes" for t in threading.enumerate()) < before:
            break
        time.sleep(0.02)
    assert sum(t.name == "postura-lotes" for t in threading.enumerate()) == before - 1 and sess.session is None
