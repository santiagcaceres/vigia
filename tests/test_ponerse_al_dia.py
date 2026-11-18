"""Una máquina que no da abasto no se atrasa (18 nov, la farmacia: «va muy lento, es como varias fotos más que
video»). Con 12 cámaras en una máquina de 4, analizar un cuadro con gente tardaba más que lo que la cámara tarda en
mandar el siguiente: los cuadros se apilaban y se analizaban todos en orden, cada vez más atrasados, con la hora de la
PC. Los clips salían en cámara lenta (un rato de 3 s duraba 17) y los gestos se medían lentos. Ahora, después de un
cuadro lento, se saltean los que se apilaron (salen al instante) y se sigue con el vivo."""

from __future__ import annotations

import numpy as np

import agent.worker as worker_mod
from agent.worker import CameraWorker
from core.detectors import Detection


class FakeTime:
    def __init__(self):
        self.t = 1000.0

    def monotonic(self):
        return self.t

    def time(self):
        return self.t

    def sleep(self, s):
        self.t += s


class LiveCam:
    """Una cámara en vivo a `fps`: el cuadro k sale a los k/fps segundos. Si nadie lo lee, queda esperando (como en el
    buffer de la conexión): leerlo después sale al instante (solo se decodifica)."""

    def __init__(self, clock: FakeTime, fps: float, seconds: float):
        self.clock, self.fps = clock, fps
        self.start, self.k, self.total = clock.t, 0, int(seconds * fps)
        self.rng = np.random.default_rng(1)
        self.seen: list[tuple[float, float]] = []  # (cuándo se analizó, cuándo lo grabó la cámara)

    def grab(self):
        if self.k >= self.total:
            return False
        avail = self.start + self.k / self.fps
        if self.clock.t < avail:
            self.clock.t = avail  # el vivo: espera a que la cámara lo mande
        else:
            self.clock.t += 0.002  # ya estaba esperando: solo decodificarlo
        self.cur = avail
        self.k += 1
        return True

    def retrieve(self):
        self.seen.append((self.clock.t, self.cur))
        return True, self.rng.integers(0, 255, (72, 128, 3), dtype=np.uint8)  # siempre «se mueve»

    def get(self, prop):
        import cv2
        return self.fps if prop == cv2.CAP_PROP_FPS else 0.0

    def release(self):
        pass


class SlowDetector:
    """Una persona siempre a la vista, y cada cuadro tarda 0,3 s en analizarse (la máquina sobrecargada)."""

    def __init__(self, clock: FakeTime):
        self.clock = clock

    def detect(self, frame, *a, **k):
        self.clock.t += 0.3
        return [Detection((30, 10, 60, 70), "persona", 0.9)]


def _run(tmp_path, monkeypatch, catchup: bool):
    clock = FakeTime()
    monkeypatch.setattr(worker_mod, "time", clock)
    if not catchup:
        monkeypatch.setattr(worker_mod, "CATCHUP_MAX_S", 0.0)  # (como antes: analizar todo en orden)
    cfg = {"id": 1, "name": "Canal 8", "connection": {"host": "dvr", "brand": "hikvision"}, "analysis": {}}
    w = CameraWorker(cfg, SlowDetector(clock), None, worker_mod.Outbox(tmp_path, None), None, None,
                     {"clip_pre_seconds": 1, "clip_post_seconds": 1, "clip_max_seconds": 30})
    cam = LiveCam(clock, 15.0, 20.0)
    w._read_loop(cam)
    lags = [now - shot for now, shot in cam.seen]
    return w, cam, lags


def test_sin_ponerse_al_dia_se_atrasa_cada_vez_mas(tmp_path, monkeypatch):
    _, cam, lags = _run(tmp_path, monkeypatch, catchup=False)
    assert lags[-1] > 30  # (lo de antes: al final analiza lo de hace más de medio minuto)


def test_la_maquina_que_no_da_abasto_sigue_con_el_vivo(tmp_path, monkeypatch):
    w, cam, lags = _run(tmp_path, monkeypatch, catchup=True)
    # cada cuadro que se analiza es de ahora (a lo sumo lo que tardó el último análisis más un cuadro)
    assert max(lags) < 0.3 + 2 / 15 + 0.01, max(lags)
    # analiza lo que puede (unos 3 por segundo), repartidos en los 20 s (no los primeros 60 cuadros seguidos)
    shots = [shot - cam.start for _, shot in cam.seen]
    assert 40 <= len(shots) <= 70 and shots[-1] > 19
    # el clip queda con la hora de verdad: lo que dice el buffer coincide con cuándo lo grabó la cámara
    buf = w.analyzer._buffer
    span = buf[-1][0] - buf[0][0]
    assert abs(span - (shots[-1] - shots[len(shots) - len(buf)])) < 0.5
    assert w.skipped_pct is None or w.skipped_pct >= 0  # (se calcula cada minuto; la prueba dura 20 s)


def test_una_camara_al_dia_no_saltea_nada(tmp_path, monkeypatch):
    """Si analizar es más rápido que la cámara, no se saltea ningún cuadro de los que tocaba analizar."""
    clock = FakeTime()
    monkeypatch.setattr(worker_mod, "time", clock)

    class Fast(SlowDetector):
        def detect(self, frame, *a, **k):
            self.clock.t += 0.01
            return [Detection((30, 10, 60, 70), "persona", 0.9)]
    cfg = {"id": 1, "name": "Caja", "connection": {"host": "dvr", "brand": "hikvision"}, "analysis": {}}
    w = CameraWorker(cfg, Fast(clock), None, worker_mod.Outbox(tmp_path, None), None, None,
                     {"clip_pre_seconds": 1, "clip_post_seconds": 1, "clip_max_seconds": 30})
    cam = LiveCam(clock, 15.0, 10.0)
    w._read_loop(cam)
    assert 45 <= len(cam.seen) <= 52  # 5 por segundo, como siempre
    assert max(now - shot for now, shot in cam.seen) < 0.05
