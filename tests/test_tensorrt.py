"""TensorRT para el detector de personas de las máquinas (26 nov, medido en la máquina de AWS con una T4 y
`tools/tensorrt_prueba.py`: 2,6-3 veces menos placa y procesador por cuadro que CUDA, las mismas 29 personas en 44
imágenes). Acá no hay placa: se prueba cuándo se pide y que, si TensorRT falla, se siga con CUDA y nunca con el
procesador."""

from __future__ import annotations

import logging
import sys
import types

import numpy as np
import pytest

from core import detectors


class FakeInput:
    name = "images"
    shape = [1, 3, 640, 640]


class FakeSession:
    def __init__(self, model, opts, providers):
        names = [p[0] if isinstance(p, tuple) else p for p in providers]
        if FakeOrt.fail_trt and "TensorrtExecutionProvider" in names:
            raise RuntimeError("libnvinfer.so.10: cannot open shared object file")
        self.providers = names
        FakeOrt.made.append(names)

    def get_providers(self):
        return self.providers

    def get_inputs(self):
        return [FakeInput()]


class FakeOrt(types.ModuleType):
    made: list = []
    fail_trt = False

    def __init__(self):
        super().__init__("onnxruntime")
        self.InferenceSession = FakeSession
        self.SessionOptions = lambda: types.SimpleNamespace()
        self.ExecutionMode = types.SimpleNamespace(ORT_SEQUENTIAL=0)

    @staticmethod
    def get_available_providers():
        return ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]


@pytest.fixture
def fake_ort(monkeypatch):
    ort = FakeOrt()
    FakeOrt.made = []
    FakeOrt.fail_trt = False
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.delenv("VIGIA_DISPOSITIVO", raising=False)  # (otras pruebas fuerzan el procesador)
    monkeypatch.setattr(detectors, "gpu_wait_mode", lambda: "")
    monkeypatch.setattr(detectors, "_trt_ok", None)
    monkeypatch.setattr(detectors, "_load_trt", lambda: True)
    return ort


def test_tensorrt_solo_para_quien_lo_pide_y_con_la_variable(fake_ort, monkeypatch):
    monkeypatch.delenv("VIGIA_TENSORRT", raising=False)
    assert "TensorrtExecutionProvider" not in detectors._providers(fake_ort, tensorrt=True)  # apagado de fábrica
    monkeypatch.setenv("VIGIA_TENSORRT", "1")
    assert detectors.tensorrt_mode() == "detector"
    assert detectors._providers(fake_ort, tensorrt=True)[0] == "TensorrtExecutionProvider"
    assert "TensorrtExecutionProvider" not in detectors._providers(fake_ort)  # (postura, objetos: no lo piden)
    monkeypatch.setenv("VIGIA_TENSORRT", "todo")
    assert detectors.tensorrt_mode() == "todo"
    monkeypatch.setenv("VIGIA_TENSORRT", "no")
    assert detectors.tensorrt_mode() == ""


def test_sin_las_librerias_sigue_con_cuda(fake_ort, monkeypatch):
    monkeypatch.setenv("VIGIA_TENSORRT", "1")
    monkeypatch.setattr(detectors, "_load_trt", lambda: False)
    assert detectors._providers(fake_ort, tensorrt=True)[0] == "CUDAExecutionProvider"


def test_si_tensorrt_falla_al_armar_la_sesion_sigue_con_cuda_y_no_con_el_procesador(fake_ort, monkeypatch, tmp_path,
                                                                                     caplog):
    monkeypatch.setenv("VIGIA_TENSORRT", "1")
    FakeOrt.fail_trt = True
    with caplog.at_level(logging.WARNING, logger="vigia"):
        s = detectors.OnnxSession(tmp_path / "x.onnx", max_batch=1, trt=True)
    assert s.device == "GPU (CUDA)" and FakeOrt.made == [["CUDAExecutionProvider", "CPUExecutionProvider"]]
    assert "TensorRT no anduvo" in caplog.text
    FakeOrt.fail_trt = False
    FakeOrt.made = []
    s = detectors.OnnxSession(tmp_path / "x.onnx", max_batch=1, trt=True)
    assert s.device == "GPU (Tensorrt)" and FakeOrt.made[0][0] == "TensorrtExecutionProvider"


def test_las_librerias_se_cargan_una_sola_vez(monkeypatch):
    calls = []
    monkeypatch.setattr(detectors, "_trt_ok", None)
    monkeypatch.setattr(detectors, "_load_trt", lambda: calls.append(1) or True)
    assert detectors.tensorrt_ready() and detectors.tensorrt_ready() and calls == [1]


def test_busca_las_librerias_donde_las_deja_la_herramienta(monkeypatch, tmp_path):
    monkeypatch.setenv("VIGIA_TRT_LIBS", str(tmp_path))
    dirs = detectors._trt_dirs()
    assert dirs[0] == tmp_path and dirs[1].as_posix() == "/var/lib/vigia/trt/tensorrt_libs"
    assert dirs[2].as_posix() == "/var/tmp/vigia-trt/tensorrt_libs"
    monkeypatch.setattr(detectors, "_trt_dirs", lambda: [tmp_path])
    (tmp_path / "libnvinfer.so.10").write_bytes(b"no es una libreria")
    assert detectors._load_trt() is False  # (no carga: se sigue con CUDA, sin romper nada)


def test_el_detector_compartido_de_las_maquinas_pide_tensorrt(monkeypatch, tmp_path):
    seen = {}

    class Sess:
        def __init__(self, path, threads=2, max_batch=None, trt=False, **kw):
            seen["trt"] = trt
            self.max_batch, self.on_gpu, self.device, self.size = 16, True, "GPU (Tensorrt)", (640, 640)
            self.uint8, self.col_of, self.session = True, {c: 4 + c for c in range(80)}, None

        def run(self, canvas):
            return np.zeros((84, 1), np.float32)
    model = tmp_path / "personas.onnx"
    model.write_bytes(b"x")
    monkeypatch.setattr(detectors, "OnnxSession", Sess)
    monkeypatch.setattr(detectors, "_shared", {})
    monkeypatch.setattr(detectors, "_sessions", {})
    monkeypatch.delenv("VIGIA_HILOS", raising=False)
    det = detectors.build_detector("yolo", model)
    assert seen["trt"] is True and det.device == "GPU (Tensorrt)"
