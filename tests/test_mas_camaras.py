"""Más cámaras por máquina sin perder calidad (25 nov, el dueño: «hacé lo que tengas que hacer para que entren más
cámaras por PC sin perder calidad»): el video de cada cámara se decodifica con un solo hilo, OpenCV sin su grupo de
hilos en las máquinas, y cada proceso informa en qué se le va el procesador."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

import agent.worker as worker_mod
from agent import consumo
from agent.oficina import cpu_text
from agent.worker import CameraWorker, heavy_video, open_video, video_threads
from core.detectors import Detection
from server.main import Settings, create_app

cv2 = pytest.importorskip("cv2")
ADMIN = ("admin", "clave-admin-123")


# ---------------------------------------------------------------- el video con un solo hilo
def test_cuantos_hilos_decodifican(monkeypatch):
    monkeypatch.delenv("VIGIA_HILOS_VIDEO", raising=False)
    assert video_threads() == 1  # lo de fábrica en las máquinas
    for value, want in (("auto", None), ("0", None), ("2", 2), ("cualquiera", 1), ("", 1)):
        monkeypatch.setenv("VIGIA_HILOS_VIDEO", value)
        assert video_threads() == want
    # hasta 1080p a 30 cuadros/s alcanza con uno; más que eso, varios
    assert not heavy_video({"video_w": 960, "video_h": 1088, "video_fps": 15.0})
    assert not heavy_video({"video_w": 1920, "video_h": 1080, "video_fps": 25.0})
    assert heavy_video({"video_w": 3840, "video_h": 2160, "video_fps": 15.0})
    assert heavy_video({"video_w": 2560, "video_h": 1440})  # (sin el dato, 25 cuadros/s)
    assert not heavy_video({})


def _video(path, frames: int = 40):
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 240))
    if not out.isOpened():
        pytest.skip("este OpenCV no puede escribir el video de prueba")
    rng = np.random.default_rng(5)
    base = cv2.GaussianBlur(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8), (0, 0), 4)
    for i in range(frames):
        img = base.copy()
        cv2.rectangle(img, (10 + 5 * i, 60), (50 + 5 * i, 200), (255, 255, 255), -1)
        out.write(img)
    out.release()


def test_con_un_hilo_el_video_sale_identico(tmp_path, monkeypatch):
    monkeypatch.setattr(worker_mod, "_threads_ok", None)
    path = tmp_path / "prueba.mp4"
    _video(path)

    def frames(threads):
        cap = open_video(str(path), threads)
        assert cap.isOpened()
        n_threads, out = cap.get(cv2.CAP_PROP_N_THREADS), []
        while cap.grab():
            out.append(cap.retrieve()[1].copy())
        cap.release()
        return n_threads, out

    one_n, one = frames(1)
    _, auto = frames(None)
    assert one_n == 1.0
    assert len(one) == len(auto) == 40
    assert all(np.array_equal(a, b) for a, b in zip(one, auto))  # ni un píxel de diferencia


def test_un_opencv_que_no_recibe_opciones_abre_como_antes(monkeypatch):
    calls = []

    def viejo(url, api):  # (sin el tercer dato)
        calls.append((url, api))
        return "cap"

    monkeypatch.setattr(worker_mod, "_threads_ok", None)
    monkeypatch.setattr(worker_mod.cv2, "VideoCapture", viejo)
    assert open_video("rtsp://x", 1) == "cap" and calls == [("rtsp://x", cv2.CAP_FFMPEG)]
    assert worker_mod._threads_ok is False  # y el proceso no lo intenta más


OPENS: list = []


class Capture:
    """Un OpenCV de mentira: abre o no según la dirección y según si se le pidió elegir los hilos."""

    def __init__(self, url, api, params=None):
        OPENS.append((url, bool(params)))
        self.ok = ("anda" in url) and not (params and "sin-opcion" in url)

    def isOpened(self):
        return self.ok

    def release(self):
        pass


def test_si_la_opcion_de_hilos_no_abre_y_sin_ella_si_se_deja_de_usar(monkeypatch, caplog):
    monkeypatch.setattr(worker_mod.cv2, "VideoCapture", Capture)
    for name, value in (("_threads_ok", None), ("_plain_tries", 0)):
        monkeypatch.setattr(worker_mod, name, value)
    OPENS.clear()
    # una cámara caída: se prueba también sin la opción, pero pocas veces (no golpear dos veces al grabador siempre)
    for _ in range(6):
        assert not open_video("rtsp://caida", 1).isOpened()
    assert OPENS.count(("rtsp://caida", False)) == worker_mod.PLAIN_TRIES_MAX
    assert worker_mod._threads_ok is None
    # en cuanto una abre con la opción, queda claro que anda: las caídas ya no se prueban dos veces
    assert open_video("rtsp://anda", 1).isOpened() and worker_mod._threads_ok is True
    OPENS.clear()
    assert not open_video("rtsp://caida", 1).isOpened() and OPENS == [("rtsp://caida", True)]

    # un OpenCV donde con la opción no abre ninguna y sin ella sí: se usa sin la opción de ahí en más
    for name, value in (("_threads_ok", None), ("_plain_tries", 0)):
        monkeypatch.setattr(worker_mod, name, value)
    OPENS.clear()
    with caplog.at_level("WARNING", logger="vigia.worker"):
        assert open_video("rtsp://anda-sin-opcion", 1).isOpened()
    assert worker_mod._threads_ok is False and "como antes" in caplog.text
    OPENS.clear()
    assert open_video("rtsp://anda-sin-opcion", 1).isOpened() and OPENS == [("rtsp://anda-sin-opcion", False)]


class FakeTime:
    def __init__(self):
        self.t = 1000.0

    def monotonic(self):
        return self.t

    def time(self):
        return self.t

    def sleep(self, s):
        self.t += s


class Cam:
    """Una cámara en vivo: el cuadro k sale a los k/fps segundos; leerlo «gasta» `read_cpu` segundos de procesador."""

    def __init__(self, clock: FakeTime, cpu: list, fps: float, seconds: float, size=(640, 360), read_cpu: float = 0.0):
        self.clock, self.cpu, self.fps, self.size, self.read_cpu = clock, cpu, fps, size, read_cpu
        self.start, self.k, self.total = clock.t, 0, int(seconds * fps)
        self.rng = np.random.default_rng(1)

    def grab(self):
        if self.k >= self.total:
            return False
        self.clock.t = max(self.clock.t, self.start + self.k / self.fps)
        self.cpu[0] += self.read_cpu
        self.k += 1
        return True

    def retrieve(self):
        return True, self.rng.integers(0, 255, (72, 128, 3), dtype=np.uint8)

    def get(self, prop):
        return {cv2.CAP_PROP_FPS: self.fps, cv2.CAP_PROP_FRAME_WIDTH: self.size[0],
                cv2.CAP_PROP_FRAME_HEIGHT: self.size[1]}.get(prop, 0.0)

    def release(self):
        pass


class Persona:
    def __init__(self, cpu: list):
        self.cpu = cpu

    def detect(self, frame, *a, **k):
        self.cpu[0] += 0.004  # analizar gasta 4 ms de este hilo
        return [Detection((30, 10, 60, 70), "persona", 0.9)]


def _worker(tmp_path, monkeypatch):
    clock, cpu = FakeTime(), [0.0]
    monkeypatch.setattr(worker_mod, "time", clock)
    monkeypatch.setattr(worker_mod, "thread_time", lambda: cpu[0])
    monkeypatch.delenv("VIGIA_HILOS_VIDEO", raising=False)
    cfg = {"id": 1, "name": "Caja", "connection": {"host": "dvr", "brand": "hikvision"}, "analysis": {}}
    w = CameraWorker(cfg, Persona(cpu), None, worker_mod.Outbox(tmp_path, None), None, None,
                     {"clip_pre_seconds": 1, "clip_post_seconds": 1, "clip_max_seconds": 30})
    return w, clock, cpu


def test_la_camara_informa_cuanto_procesador_gasta(tmp_path, monkeypatch):
    w, clock, cpu = _worker(tmp_path, monkeypatch)
    w._read_loop(Cam(clock, cpu, 15.0, 125.0, read_cpu=0.002))  # leer cada cuadro: 2 ms → 3 % de un procesador
    st = w.status()
    assert st["cpu_video"] == 3 and st["cpu_ia"] == 2  # (5 análisis por segundo de 4 ms)
    assert st["cpu"] == 5
    assert w._video_threads == 1 and w.cpu_read_s == pytest.approx(125 * 15 * 0.002, rel=0.02)


def test_un_video_enorme_o_que_no_alcanza_con_un_hilo_pasa_a_varios(tmp_path, monkeypatch, caplog):
    caplog.set_level("INFO", logger="vigia.worker")
    # 4K: ni se intenta con uno (se corta enseguida para reconectar con varios)
    w, clock, cpu = _worker(tmp_path, monkeypatch)
    cam = Cam(clock, cpu, 15.0, 30.0, size=(3840, 2160))
    w._read_loop(cam)
    assert w._video_threads is None and cam.k == 0
    assert "video grande" in caplog.text
    cam = Cam(clock, cpu, 15.0, 5.0, size=(3840, 2160))
    w._read_loop(cam)  # ya con varios hilos: lee normal
    assert cam.k == cam.total

    # un video que al decodificarlo se lleva el 75 % de un procesador: al minuto pasa a varios hilos
    w, clock, cpu = _worker(tmp_path, monkeypatch)
    cam = Cam(clock, cpu, 15.0, 200.0, read_cpu=0.05)
    w._read_loop(cam)
    assert w._video_threads is None and 60 * 15 <= cam.k <= 62 * 15
    assert "varios hilos" in caplog.text
    # y uno normal no se toca
    w, clock, cpu = _worker(tmp_path, monkeypatch)
    w._read_loop(Cam(clock, cpu, 15.0, 130.0, read_cpu=0.01))
    assert w._video_threads == 1


def test_en_las_maquinas_opencv_va_sin_su_grupo_de_hilos(monkeypatch):
    before = cv2.getNumThreads()
    try:
        monkeypatch.delenv("VIGIA_HILOS_OPENCV", raising=False)
        cv2.setNumThreads(4)
        worker_mod.lean_opencv()
        assert cv2.getNumThreads() == 1
        monkeypatch.setenv("VIGIA_HILOS_OPENCV", "0")  # como antes: no toca nada
        cv2.setNumThreads(3)
        worker_mod.lean_opencv()
        assert cv2.getNumThreads() == 3
    finally:
        cv2.setNumThreads(before)


# ---------------------------------------------------------------- en qué se va el procesador
def test_los_hilos_se_juntan_por_grupo():
    g = consumo.group_of
    assert g("av:hevc:df0", None) == "video" and g("av:h264:df3", "cam-7") == "video"
    assert g("python", "cam-12") == "camaras"
    assert g("python", "detector-lotes") == "modelo" and g("python", "postura-lotes") == "modelo"
    assert g("python", "clips_0") == "clips"
    assert g("python", "outbox") == "otros" and g("python", "MainThread") == "otros"
    assert g("python", None) == "nativos"  # un hilo de las librerías (no lo creó Python)


def _burn(seconds: float) -> None:
    end = time.thread_time() + seconds
    while time.thread_time() < end:
        sum(range(2000))


def test_el_proceso_mide_su_procesador_por_grupo():
    if not consumo.thread_cpu():
        pytest.skip("sin /proc (no es Linux)")
    meter = consumo.CpuMeter(min_span=0)
    assert meter.sample() == {}  # la primera vez no hay con qué comparar
    done = threading.Event()
    hilos = [threading.Thread(target=lambda: (_burn(0.25), done.wait(5)), name=n) for n in ("cam-1", "detector-lotes")]
    for h in hilos:
        h.start()
    time.sleep(0.6)
    cpu = meter.sample(read_s=0.10)  # de lo de la cámara, 0,10 s fueron leyendo el video
    done.set()
    for h in hilos:
        h.join()
    span = 0.6
    assert cpu["modelo"] == pytest.approx(0.25 / span, abs=0.15)
    assert cpu["video"] == pytest.approx(0.10 / span, abs=0.08)
    assert cpu["camaras"] == pytest.approx(0.15 / span, abs=0.15)
    assert cpu["total"] >= cpu["modelo"] + cpu["video"] and cpu["cores"] >= 1
    # con poco tiempo entre consultas devuelve lo último (el sistema cuenta de a 10 ms)
    slow = consumo.CpuMeter(min_span=30)
    slow.sample()
    assert slow.sample() == {} and slow.last == {}


def test_la_app_dice_cuanto_procesador_usa_el_analisis():
    m = {"cpu": {"total": 2.48, "cores": 4, "video": 0.9, "camaras": 0.62, "modelo": 0.71, "nativos": 0.21,
                 "clips": 0.01, "otros": 0.03}}
    assert cpu_text(m, short=True) == "proc. 62 %"
    largo = cpu_text(m)
    assert largo.startswith("El análisis usa el 62 % del procesador (2,5 de 4): ")
    assert "decodificar el video 0,9" in largo and "darle el trabajo a la placa 0,7" in largo
    assert "clips" not in largo and "otros" not in largo  # lo que casi no gasta no se nombra
    for nada in (None, {}, {"cpu": None}, {"cpu": {"total": 1.0}}, {"cpu": {"total": "x", "cores": 4}}):
        assert cpu_text(nada) == "" and cpu_text(nada, short=True) == ""  # una máquina sin actualizar


# ---------------------------------------------------------------- el servidor lo guarda y lo muestra por máquina
@pytest.fixture
def app(tmp_path, db_url):
    return create_app(Settings(data_dir=tmp_path / "data", database_url=db_url, admin_password=ADMIN[1],
                               serve_public=False))


def test_el_servidor_guarda_el_procesador_de_cada_proceso_y_lo_suma_por_maquina(app):
    with TestClient(app) as a:
        a.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]})
        key = a.post("/api/analysis/pool-key").json()["key"]
        procs = []
        for n in (1, 2):
            r = TestClient(app).post("/api/agent/enroll", json={"key": key, "name": f"fija-{n}", "capacity": 8,
                                                                "instance_id": "i-fija"})
            assert r.status_code == 200, r.text
            procs.append(TestClient(app, headers={"X-Agent-Token": r.json()["token"]}))
        base = {"cameras": [], "capacity": 8, "instance_id": "i-fija"}
        procs[0].post("/api/agent/status", json={**base, "cpu": {"total": 0.6, "video": 0.25, "camaras": 0.2,
                                                                 "modelo": 0.15, "cores": 4}})
        # lo raro no se guarda: textos, claves que no son de las nuestras, números imposibles
        procs[1].post("/api/agent/status", json={**base, "cpu": {
            "total": 0.5, "video": 0.3, "modelo": 0.2, "cores": 4, "<script>": 1, "otros": "mucho", "clips": -3,
            "nativos": 1e9, "camaras": True}})
        m = a.get("/api/analysis").json()["machines"]
        assert len(m) == 1 and m[0]["instance_id"] == "i-fija"
        assert m[0]["cpu"] == {"total": 1.1, "video": 0.55, "camaras": 0.2, "modelo": 0.35, "cores": 4}
        # un dato que no sirve no rompe nada: ese proceso queda sin informar y se ve lo del otro
        assert procs[1].post("/api/agent/status", json={**base, "cpu": "mucho"}).status_code == 200
        assert a.get("/api/analysis").json()["machines"][0]["cpu"]["total"] == 0.6
        # y una máquina que todavía no lo manda (sin actualizar) va sin el dato
        for p in procs:
            p.post("/api/agent/status", json=base)
        assert a.get("/api/analysis").json()["machines"][0]["cpu"] is None


# ---------------------------------------------------------------- esperar a la placa durmiendo (a prueba)
class Cuda:
    def __init__(self, set_device=0, set_flags=0):
        self.calls, self._dev, self._flags = [], set_device, set_flags

    def cudaSetDevice(self, n):
        self.calls.append(("device", n))
        return self._dev

    def cudaSetDeviceFlags(self, flags):
        self.calls.append(("flags", flags))
        return self._flags


def test_esperar_a_la_placa_durmiendo_solo_si_se_pide(monkeypatch, caplog):
    from core import detectors

    monkeypatch.setattr(detectors, "_wait_mode_done", False)
    monkeypatch.delenv("VIGIA_PLACA_ESPERA", raising=False)
    lib = Cuda()
    assert detectors.gpu_wait_mode(lib) is None and lib.calls == []  # de fábrica no toca nada

    monkeypatch.setenv("VIGIA_PLACA_ESPERA", "dormir")
    assert detectors.gpu_wait_mode(lib) == "dormir"
    assert lib.calls == [("device", 0), ("flags", 0x04)]  # primero elegir la placa, después cómo se espera
    assert detectors.gpu_wait_mode(lib) is None and len(lib.calls) == 2  # una sola vez por proceso

    # si CUDA no quiere (ya estaba en uso, otra versión), queda como siempre y lo dice
    monkeypatch.setattr(detectors, "_wait_mode_done", False)
    with caplog.at_level("WARNING", logger="vigia.detectors"):
        assert detectors.gpu_wait_mode(Cuda(set_flags=36)) == "error"
    assert "queda como siempre" in caplog.text


# ---------------------------------------------------------------- filtro de movimiento más barato (mismo resultado)
def _gate_before(frame):
    """Cómo achicaba el filtro de movimiento hasta el 25 nov: primero achicar y después pasar a gris."""
    from core.motion import SIZE

    small = cv2.cvtColor(cv2.resize(frame, SIZE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(small, (3, 3), 0)


def test_el_filtro_de_movimiento_achica_mas_barato_y_decide_lo_mismo():
    from core.motion import MotionGate

    rng = np.random.default_rng(11)
    base = cv2.GaussianBlur(rng.integers(0, 255, (544, 960, 3), dtype=np.uint8), (0, 0), 3)
    gate = MotionGate()
    assert int(np.abs(gate._small(base).astype(int) - _gate_before(base)).max()) <= 1

    # una escena con ruido de cámara y una persona que entra y sale: decide igual cuadro por cuadro
    def frames():
        for i in range(120):
            img = base.copy()
            if 40 <= i < 80:
                x = 100 + 8 * (i - 40)
                cv2.rectangle(img, (x, 200), (x + 60, 500), (40, 40, 200), -1)
            noise = rng.integers(-3, 4, img.shape, dtype=np.int16)
            yield np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    old, new = MotionGate(), MotionGate()
    old._small = _gate_before  # (lo de antes, para comparar)
    got_old, got_new = [], []
    for i, f in enumerate(frames()):
        got_old.append(old.moved(f, i * 0.2))
        got_new.append(new.moved(f, i * 0.2))
    assert got_old == got_new
    assert any(got_new[40:80]) and not any(got_new[5:35])


def test_con_alguien_a_la_vista_no_compara_y_despues_se_pone_al_dia(monkeypatch):
    from core.motion import MotionGate

    gate = MotionGate(every=2.0)
    calls = []
    real = gate.moved
    monkeypatch.setattr(gate, "moved", lambda f, ts: (calls.append(ts), real(f, ts))[1])
    still = np.full((54, 96, 3), 120, np.uint8)
    assert gate.should_analyze(still, 0.0, busy=False)  # (el primero siempre)
    for i in range(1, 20):
        assert gate.should_analyze(still, i * 0.2, busy=True)  # se analiza siempre, sin achicar ni comparar
    assert calls == [0.0]
    changed = still.copy()
    changed[10:40, 20:60] = 250  # se fue la persona y quedó algo distinto: el primer cuadro sin nadie lo ve
    assert gate.should_analyze(changed, 4.0, busy=False)
    assert not gate.should_analyze(changed, 4.2, busy=False)  # y desde ahí compara contra lo nuevo
    assert gate.take_saved_fraction() == 1 / 22


# ---------------------------------------------------------------- ffmpeg de los clips no le saca procesador al análisis
# (26 nov: 10 cámaras sin ajustar dieron 51 alertas en 20 minutos; cada clip lanzaba un ffmpeg con todos los hilos y la
# misma prioridad que el análisis, y la máquina quedó con carga 51 en 4 procesadores: todas las cámaras a 1-3 cuadros/s)
def test_ffmpeg_arma_el_clip_con_un_hilo_y_prioridad_baja(monkeypatch, tmp_path):
    from core import clips

    seen = []

    def run(cmd, **kw):
        seen.append(cmd)
        (tmp_path / "out.mp4").write_bytes(b"x")
        return type("R", (), {"returncode": 0, "stderr": b""})()

    monkeypatch.setattr(clips.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(clips.subprocess, "run", run)
    assert clips.transcode_h264("rtsp://grabador/tramo", tmp_path / "out.mp4", duration=30)
    cmd = seen[-1]
    if clips.os.name != "nt":
        assert cmd[:4] == ["/usr/bin/nice", "-n", "15", "/usr/bin/ffmpeg"]  # prioridad baja: el análisis va primero
    i = cmd.index("-i")
    assert cmd[cmd.index("-threads"):cmd.index("-threads") + 2] == ["-threads", "1"] and cmd.index("-threads") < i
    after = cmd[i:]
    assert after[after.index("-threads") + 1] == "1" and after[after.index("-filter_threads") + 1] == "1"
    assert after[after.index("-c:v") + 1] == "libx264" and after[-1].endswith("out.mp4")
    # se puede volver a como era (los hilos de fábrica y la misma prioridad)
    monkeypatch.setattr(clips, "FFMPEG_THREADS", 0)
    monkeypatch.setattr(clips, "FFMPEG_NICE", 0)
    clips.transcode_h264("a.mp4", tmp_path / "out.mp4")
    assert seen[-1][0] == "/usr/bin/ffmpeg" and "-threads" not in seen[-1] and "-filter_threads" not in seen[-1]


def test_un_clip_de_verdad_sale_con_ffmpeg_de_un_hilo(tmp_path):
    import shutil

    from core.clips import video_duration, write_mp4

    if not shutil.which("ffmpeg"):
        pytest.skip("sin ffmpeg")
    import cv2

    frames = []
    for i in range(25):
        img = np.zeros((180, 320, 3), np.uint8)
        cv2.circle(img, (10 + 12 * i, 90), 12, (0, 200, 255), -1)
        frames.append((i * 0.2, cv2.imencode(".jpg", img)[1].tobytes()))
    out = tmp_path / "clip.mp4"
    assert write_mp4(frames, out) and (video_duration(out) or 0) > 4
    assert not (tmp_path / "clip.raw.mp4").exists()  # se pasó a H.264 (si no, queda el de OpenCV con otro nombre)


def test_el_medidor_cuenta_ffmpeg_la_carga_y_la_placa(monkeypatch):
    import subprocess
    import sys

    if not consumo.thread_cpu():
        pytest.skip("sin /proc (no es Linux)")
    meter = consumo.CpuMeter(min_span=0)
    meter.sample()
    t0 = time.monotonic()
    subprocess.run([sys.executable, "-c", "import time\nt = time.process_time()\nwhile time.process_time() - t < 0.4: pass"],
                   check=True)
    cpu = meter.sample()
    span = time.monotonic() - t0
    assert cpu["externos"] == pytest.approx(0.4 / span, abs=0.2)  # el programa que lanzó (como ffmpeg), al terminar
    assert cpu["total"] >= cpu["externos"] and "load" in cpu
    if consumo.machine_load().get("wait_pct") is not None:
        assert 0 <= cpu["wait_pct"] <= 100
    # la placa: lo que dice nvidia-smi (sin placa, nada)
    monkeypatch.setattr(consumo.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(consumo.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": "31, 2523\n"})())
    assert consumo.gpu_usage() == {"gpu": 31, "gpu_gb": 2.46}
    monkeypatch.setattr(consumo.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": "[N/A]\n"})())
    assert consumo.gpu_usage() == {}
    monkeypatch.setattr(consumo.shutil, "which", lambda name: None)
    assert consumo.gpu_usage() == {}


def test_la_app_avisa_cuando_la_maquina_no_da_abasto():
    m = {"cpu": {"total": 1.6, "cores": 4, "video": 0.9, "camaras": 0.4, "modelo": 0.3, "externos": 2.1,
                 "load": 51.6, "wait_pct": 98.9, "gpu": 31, "gpu_gb": 2.46}}
    assert cpu_text(m, short=True) == "proc. 40 % ⚠"
    largo = cpu_text(m)
    assert "ffmpeg (clips) 2,1" in largo and "Carga de la máquina 51,6" in largo
    assert "el 99 % del tiempo hay procesos esperando procesador: no da abasto" in largo
    assert largo.endswith("Placa de video: 31 %, 2,5 GB de memoria.")
    tranquila = cpu_text({"cpu": {"total": 1.7, "cores": 4, "load": 2.1, "wait_pct": 3.0}})
    assert "no da abasto" not in tranquila and "Carga de la máquina 2,1." in tranquila
    assert cpu_text({"cpu": {"total": 1.7, "cores": 4, "wait_pct": 3.0}}, short=True) == "proc. 42 %"


def test_lo_de_la_maquina_entera_no_se_suma_entre_sus_procesos(app):
    with TestClient(app) as a:
        a.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]})
        key = a.post("/api/analysis/pool-key").json()["key"]
        base = {"cameras": [], "capacity": 8, "instance_id": "i-fija"}
        for n, ext in ((1, 0.8), (2, 1.3)):
            r = TestClient(app).post("/api/agent/enroll", json={"key": key, "name": f"fija-{n}", "capacity": 8,
                                                                "instance_id": "i-fija"})
            TestClient(app, headers={"X-Agent-Token": r.json()["token"]}).post("/api/agent/status", json={
                **base, "cpu": {"total": 0.5, "externos": ext, "cores": 4, "load": 51.6, "wait_pct": 98.9,
                                "gpu": 31, "gpu_gb": 2.46}})
        cpu = a.get("/api/analysis").json()["machines"][0]["cpu"]
        assert cpu == {"total": 1.0, "externos": 2.1, "cores": 4, "load": 51.6, "wait_pct": 98.9, "gpu": 31,
                       "gpu_gb": 2.46}
