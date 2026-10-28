"""«Bajar grabación…»: tramos de la grabación del grabador a MP4 en la PC, sin ffmpeg (OpenCV lee el link de
reproducción y escribe el archivo)."""

import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from agent import grabacion as g
from core import synthetic


class FakeCam:
    """Como una cámara del grabador: `playback_url` da un video (acá, un archivo)."""

    brand = "ip"

    def __init__(self, src):
        self.src = str(src)
        self.asked = []

    def playback_url(self, start, end):
        self.asked.append((start, end))
        return self.src

    def open_failure(self, url):
        return None

    def redacted(self, url):
        return Path(url).name


def test_tramos_y_nombres():
    t = lambda h, m: datetime(2026, 10, 28, h, m)  # noqa: E731
    assert g.pieces(t(8, 0), t(9, 30), 30) == [(t(8, 0), t(8, 30)), (t(8, 30), t(9, 0)), (t(9, 0), t(9, 30))]
    assert g.pieces(t(8, 0), t(8, 45), 30)[-1] == (t(8, 30), t(8, 45))
    with pytest.raises(ValueError):
        g.pieces(t(9, 0), t(8, 0))
    assert g.file_name("Minas", "Pasillo: bebidas/2", t(8, 0), t(8, 30)) == \
        "Minas - Pasillo- bebidas-2 - 2026-10-28 08.00-08.30.mp4"
    cams = [{"id": 44, "name": "Pasillo bebidas", "client_name": "Minas"}, {"id": 7, "name": "Góndolas ventana"}]
    p = g.plan(cams, t(8, 0), t(9, 30), Path("/tmp/x"), 30)
    assert len(p) == 6 and p[0].label == "Pasillo bebidas 08:00-08:30" and p[3].camera_id == 7


def test_baja_un_tramo_a_mp4(tmp_path):
    src = synthetic.write_video(synthetic.render("hurto"), tmp_path / "grabador.mp4")
    frames = int(cv2.VideoCapture(str(src)).get(cv2.CAP_PROP_FRAME_COUNT))
    seconds = frames / synthetic.FPS
    start = datetime(2026, 10, 28, 8, 0)
    end = start.replace(second=int(seconds))
    piece = g.plan([{"id": 1, "name": "Caja", "client_name": "Minas"}], start, end, tmp_path / "salida", 30)[0]
    seen = []
    cam = FakeCam(src)
    g.download(cam, piece, on_progress=lambda p: seen.append(p.progress))
    assert piece.state == "listo", piece.detail
    assert cam.asked == [(start, end)]
    cap = cv2.VideoCapture(str(piece.path))
    ok, frame = cap.read()
    assert ok and frame.shape[:2] == (synthetic.H, synthetic.W)
    assert abs(cap.get(cv2.CAP_PROP_FRAME_COUNT) - frames) <= 1
    assert seen and seen[-1] == pytest.approx(1.0, abs=0.05)
    assert not list((tmp_path / "salida").glob("*.bajando.mp4"))

    # si el grabador tiene menos grabación que el tramo pedido: queda el archivo, marcado incompleto
    long = g.plan([{"id": 1, "name": "Caja", "client_name": "Minas"}], start, start.replace(minute=5),
                  tmp_path / "salida2", 30)[0]
    g.download(FakeCam(src), long)
    assert long.state == "incompleto" and long.path.exists() and "llegaron" in long.detail


def test_errores_y_detener(tmp_path):
    start = datetime(2026, 10, 28, 8, 0)
    piece = g.plan([{"id": 1, "name": "Caja"}], start, start.replace(minute=1), tmp_path, 30)[0]
    g.download(FakeCam(tmp_path / "no-existe.mp4"), piece)
    assert piece.state == "error" and "No se pudo abrir" in piece.detail and not piece.path.exists()

    src = synthetic.write_video([np.zeros((synthetic.H, synthetic.W, 3), np.uint8)] * 30, tmp_path / "g.mp4")
    stop = threading.Event()
    stop.set()
    piece2 = g.plan([{"id": 1, "name": "Caja"}], start, start.replace(minute=1), tmp_path / "s", 30)[0]
    g.download(FakeCam(src), piece2, stop)
    assert piece2.state == "detenido" and not piece2.path.exists()

    class SinLink(FakeCam):
        def playback_url(self, start, end):
            raise RuntimeError("Hik-Connect: el grabador no está en línea")
    piece3 = g.plan([{"id": 1, "name": "Caja"}], start, start.replace(minute=1), tmp_path / "t", 30)[0]
    g.download(SinLink(src), piece3)
    assert piece3.state == "error" and "no está en línea" in piece3.detail


def test_guarda_como_mucho_15_cuadros_por_segundo(tmp_path):
    frame = np.zeros((synthetic.H, synthetic.W, 3), np.uint8)
    src = synthetic.write_video([frame] * 100, tmp_path / "g25.mp4", fps=25)  # 4 s a 25 cuadros/s
    start = datetime(2026, 10, 28, 8, 0)
    piece = g.plan([{"id": 1, "name": "Caja"}], start, start.replace(second=4), tmp_path / "s", 30)[0]
    g.download(FakeCam(src), piece)
    cap = cv2.VideoCapture(str(piece.path))
    assert piece.state == "listo" and cap.get(cv2.CAP_PROP_FPS) == pytest.approx(12.5)
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == pytest.approx(50, abs=1)
    assert piece.seconds == pytest.approx(4.0, abs=0.1)
