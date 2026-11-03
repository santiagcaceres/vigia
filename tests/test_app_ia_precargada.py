"""La app de escritorio carga la IA una sola vez, al abrir (con la pantalla de carga), y la comparte entre las
cámaras que se miran con «Ver con la IA». Antes, en una PC sin placa de video, cada cámara volvía a cargar el modelo
y la ventana quedaba «No responde» unos segundos (onnxruntime no suelta Python mientras arma la sesión)."""

import threading
import time
from pathlib import Path

import pytest

from core import detectors

MODEL = Path("models/personas.onnx")
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="sin el modelo")


def test_la_app_comparte_el_modelo_tambien_en_el_procesador(monkeypatch):
    monkeypatch.delenv("VIGIA_HILOS", raising=False)
    made = []
    real = detectors.OnnxSession

    def counting(*a, **k):
        made.append(1)
        return real(*a, **k)

    monkeypatch.setattr(detectors, "OnnxSession", counting)
    monkeypatch.setattr(detectors, "_shared", {})
    monkeypatch.setattr(detectors, "_sessions", {})
    a = detectors.build_detector("yolo", MODEL, shared=True)
    b = detectors.build_detector("yolo", MODEL, shared=True)
    c = detectors.build_detector("yolo", MODEL, vehicles=True, shared=True)
    assert a is b and c.session is a.session and len(made) == 1  # una sola carga para todas las cámaras
    # en la app, lo demás (videos subidos, alarmas) también usa el que ya está cargado
    assert detectors.build_detector("yolo", MODEL).session is a.session and len(made) == 1
    # las máquinas sin ventana siguen como siempre (en el procesador, una sesión por cámara)
    monkeypatch.setattr(detectors, "_shared", {})
    monkeypatch.setattr(detectors, "_sessions", {})
    detectors.build_detector("yolo", MODEL)
    detectors.build_detector("yolo", MODEL)
    assert len(made) == 3


def test_precargar_la_ia_al_abrir(monkeypatch):
    from agent import __main__ as app
    monkeypatch.setattr(detectors, "_shared", {})
    monkeypatch.setattr(detectors, "_sessions", {})
    app.preload_ai(MODEL)
    assert (str(MODEL), False) in detectors._shared and (str(MODEL), True) in detectors._shared
    # después, abrir una cámara no vuelve a cargar nada: la ventana no se frena
    gaps, done = [], threading.Event()

    def open_camera():
        detectors.build_detector("yolo", MODEL, vehicles=True, shared=True)
        done.set()

    threading.Thread(target=open_camera).start()
    last = time.monotonic()
    while not done.wait(0.01):
        now = time.monotonic()
        gaps.append(now - last)
        last = now
    assert max(gaps, default=0) < 0.2
    # sin el modelo, la app igual abre
    app.preload_ai(Path("no-existe.onnx"))
