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
