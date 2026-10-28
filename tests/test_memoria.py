"""La memoria de una cámara no crece con las reconexiones (27 oct: la máquina de Google pasó de 3,8 a 13,7 GB).

Hik-Connect corta el video cada tanto y el agente reconecta. Si el reloj del análisis volvía a cero en cada
conexión, el buffer de los clips (que descarta lo de hace más de 3 min) y los clips abiertos esperaban a que el
reloj nuevo alcanzara al viejo: mientras tanto guardaban todos los cuadros.
"""

from __future__ import annotations

import numpy as np

import agent.worker as worker_mod
from agent.worker import CameraWorker
from core.analyzer import Analyzer
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


class FakeCap:
    """Un video de `frames` cuadros a 25 por segundo que después se corta (grab falla)."""

    def __init__(self, clock: FakeTime, frames: int):
        self.clock, self.left = clock, frames
        self.rng = np.random.default_rng(len(str(frames)))

    def grab(self):
        self.clock.t += 0.04
        self.left -= 1
        return self.left >= 0

    def retrieve(self):
        return True, self.rng.integers(0, 255, (72, 128, 3), dtype=np.uint8)  # siempre «se mueve»

    def get(self, prop):
        return 0.0

    def release(self):
        pass


class NoOne:
    def detect(self, frame):
        return []


def test_las_reconexiones_no_hacen_crecer_la_memoria(tmp_path, monkeypatch):
    clock = FakeTime()
    monkeypatch.setattr(worker_mod, "time", clock)
    cfg = {"id": 1, "name": "Caja", "connection": {"host": "dvr", "brand": "hikvision"}, "analysis": {}}
    w = CameraWorker(cfg, NoOne(), None, worker_mod.Outbox(tmp_path, None), None, None,
                     {"clip_pre_seconds": 1, "clip_post_seconds": 1, "clip_max_seconds": 5})
    buf = w.analyzer._buffer
    for _ in range(6):  # 6 conexiones de 10 s cada una, con cortes de 2 s
        w._read_loop(FakeCap(clock, 250))
        clock.t += 2.0
        ts = [t for t, _ in buf]
        assert ts == sorted(ts)  # el reloj del análisis nunca vuelve atrás
        assert ts[-1] - ts[0] <= 5.0 + 1e-6  # solo los últimos 5 s (clip_max)
        assert len(buf) <= 5 * 5 + 2  # a 5 cuadros por segundo
    # el reloj siguió corriendo: 6 conexiones de 10 s más los cortes
    assert ts[-1] > 6 * 10


def test_un_clip_abierto_se_cierra_si_el_reloj_vuelve_atras():
    class OneAlways:
        def detect(self, frame):
            return [Detection((10, 10, 40, 70), 0.9)]

    zone = {"name": "deposito", "type": "restricted", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}
    a = Analyzer({"zones": [zone], "rules": {}}, OneAlways(), clip_pre=1, clip_post=1, clip_max=5)
    frame = np.zeros((72, 128, 3), np.uint8)
    ts = 100.0
    while not a._pending:  # hasta que salte la alerta y quede un clip abierto
        ts += 0.2
        a.process(frame, ts)
        assert ts < 200
    for i in range(50):  # el reloj volvió a cero (otra fuente que reinicia): el clip abierto no sigue juntando
        a.process(frame, i * 0.2)
    assert not a._pending
    assert len(a.take_ready_clips()) >= 1
    assert all(t <= 50 * 0.2 for t, _ in a._buffer)
