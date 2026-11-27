"""Leer el video con la placa dentro del proceso (agent/video_av.py, `VIGIA_VIDEO_PLACA=proceso`). Acá no hay placa:
se prueba el camino entero con PyAV decodificando con el procesador (`video_av.USE_GPU = False`); la placa se mide en la
máquina con tools/placa_prueba.py."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("av")

import agent.worker as worker_mod
from agent import video_av
from agent.worker import CameraWorker, video_info
from tests.test_video_placa import (  # noqa: F401
    Counter,
    rtsp_camera,
    run_worker,
)


def make_video(tmp_path: Path, codec: str = "libx264", size: str = "640x480", rate: int = 25,
               seconds: int = 4) -> Path:
    out = tmp_path / f"v_{codec}_{size}.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    f"testsrc2=size={size}:rate={rate}:duration={seconds}", "-c:v", codec, "-g", str(rate),
                    "-pix_fmt", "yuv420p", str(out)], check=True)
    return out


def test_las_opciones_de_ffmpeg_pasan_a_pyav():
    assert video_av.input_options(["-rtsp_transport", "tcp", "-timeout", "5000000"]) == {
        "rtsp_transport": "tcp", "timeout": "5000000"}
    assert video_av.input_options([]) == {"rtsp_transport": "tcp", "timeout": "10000000"}


@pytest.mark.parametrize("codec", ["libx264", "libx265"])
def test_lee_como_opencv(tmp_path, codec):
    """Mismos cuadros, tamaño, cuadros por segundo y códec que OpenCV (lo que el worker informa y usa)."""
    src = make_video(tmp_path, codec)
    cap = video_av.AvCapture(str(src), [], use_gpu=False)
    ref = cv2.VideoCapture(str(src))
    assert cap.isOpened() and not cap.gpu
    assert video_info(cap) == video_info(ref)
    n, diffs, last = 0, [], -1.0
    while cap.grab():
        ok, want = ref.read()
        assert ok
        n += 1
        pos = cap.get(cv2.CAP_PROP_POS_MSEC)
        assert pos > last
        last = pos
        if n % 7 == 0:
            ok, img = cap.retrieve()
            assert ok and img.shape == want.shape
            diffs.append(np.abs(img.astype(np.int16) - want.astype(np.int16)).mean())
    assert n == 100 and not ref.read()[0]
    assert max(diffs) < 1.5  # (OpenCV pasa a colores con swscale; acá OpenCV: redondeos distintos)
    cap.release()
    assert not cap.isOpened() and not cap.grab()


def test_si_no_abre_dice_por_que_sin_la_clave(tmp_path):
    cap = video_av.AvCapture("rtsp://vigia:Clave123@127.0.0.1:9/Streaming/Channels/101", [], use_gpu=False)
    assert not cap.isOpened() and cap.why and "Clave123" not in cap.why


def test_el_worker_lo_usa_si_se_pide_y_si_no_abre_vuelve_a_opencv(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("VIGIA_VIDEO_PLACA", "proceso")
    monkeypatch.setattr(video_av, "available", lambda: (True, ""))
    opened = []

    class Fake:
        def __init__(self, url, args, threads=1, ok=True):
            opened.append((url, threads))
            self.gpu, self.why = True, "" if ok else "ConnectionRefusedError"

        def isOpened(self):
            return not self.why

    monkeypatch.setattr(video_av, "AvCapture", Fake)
    cfg = {"id": 7, "name": "Puerta", "process_fps": 5, "analysis": {},
           "connection": {"brand": "hikvision", "host": "127.0.0.1", "rtsp_port": 9, "username": "vigia",
                          "password": "x", "channel": 3}}
    w = CameraWorker(cfg, Counter(), None, worker_mod.Outbox(tmp_path / "outbox", None), None, None,
                     {"clip_pre_seconds": 1, "clip_max_seconds": 30})
    url = "rtsp://vigia:x@127.0.0.1:9/Streaming/Channels/302"
    assert isinstance(w._open_av(url), Fake) and opened == [(url, w._video_threads)]
    assert w._open_gpu(url) is None  # (el de un ffmpeg por cámara no se usa: no se aprende nada para él)
    assert w._open_av("https://x/live.m3u8") is None  # solo RTSP
    monkeypatch.setattr(video_av, "AvCapture", lambda url, args, threads=1: Fake(url, args, threads, ok=False))
    with caplog.at_level(logging.WARNING, logger="vigia"):
        assert w._open_av(url) is None
    assert "con OpenCV por 10 minutos" in caplog.text and w._gpu_off_until > 0
    n = len(opened)
    assert w._open_av(url) is None and len(opened) == n  # (por 10 minutos ni lo intenta)
    monkeypatch.delenv("VIGIA_VIDEO_PLACA")
    w._gpu_off_until = 0.0
    assert w._open_av(url) is None  # sin pedirlo, nunca


def test_la_camara_rtsp_se_analiza_igual_leyendo_con_pyav(tmp_path, rtsp_camera, monkeypatch, caplog):  # noqa: F811
    """De punta a punta con una cámara RTSP simulada: el worker abre con PyAV (acá sin placa), informa el video y
    analiza a los cuadros por segundo pedidos, como con OpenCV."""
    monkeypatch.setenv("VIGIA_VIDEO_PLACA", "proceso")
    monkeypatch.setattr(video_av, "available", lambda: (True, ""))
    monkeypatch.setattr(video_av, "USE_GPU", False)
    used = []
    real = video_av.AvCapture

    def spy(*a, **k):
        cap = real(*a, **k)
        used.append(cap)
        return cap

    monkeypatch.setattr(video_av, "AvCapture", spy)
    with caplog.at_level(logging.INFO, logger="vigia"):
        w, det = run_worker(tmp_path, rtsp_camera, 8.0)
    assert used and used[0].isOpened() is False  # (se cerró al terminar)
    st = w.last_status
    assert st["state"] == "en_linea" and st["video_codec"] == "H.264" and (st["video_w"], st["video_h"]) == (640, 360)
    assert 4.0 <= det.rate(4.0) <= 5.6
    assert "la placa no puede con este video" in caplog.text  # (acá no hay placa: PyAV con el procesador)
