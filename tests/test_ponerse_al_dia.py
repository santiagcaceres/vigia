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


class Takes(SlowDetector):
    """Una persona siempre a la vista; cada cuadro tarda `seconds` en analizarse."""

    def __init__(self, clock: FakeTime, seconds: float):
        super().__init__(clock)
        self.seconds = seconds

    def detect(self, frame, *a, **k):
        self.clock.t += self.seconds
        return [Detection((30, 10, 60, 70), "persona", 0.9)]


def _minute(tmp_path, monkeypatch, seconds: float, video_fps: float = 15.0):
    clock = FakeTime()
    monkeypatch.setattr(worker_mod, "time", clock)
    cfg = {"id": 1, "name": "Caja", "connection": {"host": "dvr", "brand": "hikvision"}, "analysis": {}}
    w = CameraWorker(cfg, Takes(clock, seconds), None, worker_mod.Outbox(tmp_path, None), None, None,
                     {"clip_pre_seconds": 1, "clip_post_seconds": 1, "clip_max_seconds": 30})
    cam = LiveCam(clock, video_fps, 125.0)  # (la cuenta se cierra una vez por minuto)
    w._read_loop(cam)
    return w, cam


def test_no_da_abasto_es_analizar_menos_de_lo_que_toca(tmp_path, monkeypatch, caplog):
    """24 nov, la máquina de AWS: 200 avisos de «no da abasto» en 15 minutos con un tercio del procesador libre. El
    aviso contaba los cuadros que se saltean al ponerse al día, y eso pasa siempre que un análisis tarda más que UN
    cuadro del video (normal con gente a la vista): esos cuadros igual no se iban a analizar."""
    caplog.set_level("INFO", logger="vigia.worker")
    # 100 ms por cuadro: más que un cuadro del video (67 ms), menos que lo que hay entre análisis (200 ms): llega bien
    w, cam = _minute(tmp_path, monkeypatch, 0.10)
    assert 4.8 <= len(cam.seen) / 125 <= 5.1  # los 5 cuadros por segundo de siempre
    assert w.skipped_pct == 0.0
    assert not w.status().get("skipped_pct")
    assert "no da abasto" not in caplog.text
    # 300 ms por cuadro: de los 5 por segundo que tocaban analiza unos 3 (un tercio menos)
    w, cam = _minute(tmp_path, monkeypatch, 0.30)
    assert 3.0 <= len(cam.seen) / 125 <= 3.6
    assert 0.28 <= w.skipped_pct <= 0.38, w.skipped_pct
    assert 28 <= w.status()["skipped_pct"] <= 38
    assert "no da abasto" in caplog.text and "menos de los que tocaba" in caplog.text


def test_con_el_video_a_8_cuadros_toca_analizar_uno_de_cada_dos(tmp_path, monkeypatch, caplog):
    """La segunda calidad a 8 cuadros/s (lo que se pide en los grabadores): con 5 pedidos toca uno de cada dos, cada
    250 ms. Un análisis de 220 ms llega a tiempo aunque pase los 200 ms."""
    caplog.set_level("INFO", logger="vigia.worker")
    w, cam = _minute(tmp_path, monkeypatch, 0.22, video_fps=8.0)
    assert 3.9 <= len(cam.seen) / 125 <= 4.1  # uno de cada dos
    assert w.skipped_pct == 0.0 and "no da abasto" not in caplog.text
    # y la máquina informa los cuadros por segundo que de verdad llegan (contados, no lo que dice el video)
    assert 7.8 <= w.status()["video_fps"] <= 8.2


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
