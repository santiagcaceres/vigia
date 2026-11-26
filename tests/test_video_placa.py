"""Decodificar el video de las cámaras en la placa NVIDIA (26 nov, el dueño: «¿no se puede usar más GPU, que está libre,
en lugar de CPU?»). En la máquina de AWS, decodificar con la placa al tamaño original gastó 0,33 s de procesador por
minuto de video contra 2,30-2,54 s con el procesador.

Acá no hay placa: las pruebas usan el mismo camino (ffmpeg por un caño en NV12, la conversión a colores, el paso de
OpenCV a la placa en el CameraWorker) con el decodificador del procesador (`decoder=""`). Lo único que no se prueba acá
es `h264_cuvid` / `hevc_cuvid` en sí, que se probó en la máquina."""

from __future__ import annotations

import logging
import os
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
if not shutil.which("ffmpeg"):
    pytest.skip("sin ffmpeg", allow_module_level=True)

from agent import consumo, video_placa  # noqa: E402
from agent import worker as worker_mod  # noqa: E402
from agent.worker import CameraWorker, video_info  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MEDIAMTX = os.getenv("MEDIAMTX") or str(ROOT.parent / "tools" / "mediamtx")


def video(tmp_path: Path, name: str, pix_fmt: str = "yuv420p", rng: str = "tv", size: str = "960x576",
          seconds: int = 4, codec: str = "libx264", source: str = "testsrc2") -> Path:
    out = tmp_path / f"{name}.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    f"{source}=size={size}:rate=12", "-t", str(seconds), "-vf", f"format={pix_fmt}", "-color_range",
                    rng, "-c:v", codec, "-g", "12", "-pix_fmt", pix_fmt, str(out)], check=True)  # (un cuadro completo
    # por segundo, como una cámara: al conectarse se espera el próximo)
    return out


def opencv_frames(path: Path, n: int) -> tuple[list[np.ndarray], dict]:
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    frames = [cap.read()[1] for _ in range(n)]
    info = video_info(cap)
    cap.release()
    return frames, info


def diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


# ---------------------------------------------------------------- el tamaño, el comando y los colores
def test_el_tamano_que_se_pide_es_el_del_video_con_pixeles_cuadrados():
    assert video_placa.target_size(960, 576, None) == (960, 576)  # como llega: sin achicar (no se pierde calidad)
    assert video_placa.target_size(960, 1080, 2.0) == (960, 540)  # «1080p lite»: como square_pixels(shrink=True)
    assert video_placa.target_size(1920, 1080, 0.5) == (960, 1080)
    assert video_placa.target_size(3840, 2160, None) == (1920, 1080)  # (más de 1920 de ancho se achica en la placa)
    assert video_placa.target_size(705, 481, None) == (704, 480)  # números pares (NV12)


def test_el_ffmpeg_decodifica_en_la_placa_y_achica_solo_si_hace_falta():
    args = ["-rtsp_transport", "tcp", "-timeout", "10000000"]
    cmd = video_placa.command("rtsp://x/1", args, "hevc_cuvid", (960, 576), (960, 576))
    assert cmd[:6] == ["ffmpeg", "-nostdin", "-hide_banner", "-nostats", "-loglevel", "level+info"]
    assert args == cmd[6:10]
    assert cmd[cmd.index("-c:v") + 1] == "hevc_cuvid" and cmd.index("-c:v") < cmd.index("-i")
    # siempre al tamaño pedido, aunque sea el mismo: si la cámara cambia de tamaño, el caño no se desarma
    assert cmd[cmd.index("-resize") + 1] == "960x576" and cmd.index("-resize") < cmd.index("-i")
    assert cmd[-5:] == ["-f", "rawvideo", "-pix_fmt", "nv12", "pipe:1"]  # cada cuadro por el caño, sin pasar a colores
    assert cmd[cmd.index("-pix_fmt") + 1] == "nv12" and cmd[cmd.index("-map") + 1] == "0:v:0"
    lite = video_placa.command("rtsp://x/1", args, "h264_cuvid", (960, 540), (960, 1080))
    assert lite[lite.index("-resize") + 1] == "960x540" and lite.index("-resize") < lite.index("-i")
    cpu = video_placa.command("a.mp4", [], None, (960, 540), (960, 1080))
    assert "-c:v" not in cpu and cpu[cpu.index("-vf") + 1] == "scale=960:540" and "-threads" in cpu
    same = video_placa.command("a.mp4", [], None, (640, 360), (640, 360))
    assert same[same.index("-vf") + 1] == "scale=640:360"
    for c in (cmd, lite, cpu):  # cada cuadro como llega: ffmpeg no inventa copias para llenar un ritmo fijo
        assert c[c.index("-fps_mode") + 1] == "passthrough" and c.index("-fps_mode") > c.index("-i")
    assert "-hwaccel" not in cmd and "-vf" not in cmd  # (sin `min_gap`, como antes: se trae cada cuadro)


def test_de_lo_que_dice_ffmpeg_quedan_los_errores_sin_la_clave_y_el_tamano_del_video():
    """Con `-loglevel level+info`: los errores (para decir por qué no anduvo) sin usuario ni clave, y el tamaño del
    video que llega. El renglón «Input #0 … from 'rtsp://usuario:clave@…'» no se guarda."""
    line = video_placa.log_line
    assert line("[error] Error opening input file rtsp://vigia:Abc@2024@10.0.0.1:554/x.", False) == (
        "error", "Error opening input file rtsp://***@10.0.0.1:554/x.")
    assert line("[tcp @ 0x56] [error] Connection refused", True) == ("error", "Connection refused")
    assert line("[fatal] Error opening input files: Connection refused", False)[0] == "error"
    assert line("[info] Input #0, rtsp, from 'rtsp://vigia:clave@10.0.0.1/x':", False) == ("", None)
    assert line("[warning] RTP: missed 3 packets", False) == ("", None)
    assert line("[info]   Stream #0:0: Video: hevc (Main), yuv420p(tv), 960x1088, 25 fps, 90k tbn", False) == (
        "size", (960, 1088))
    assert line("[info]   Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(progressive), "
                "320x240 [SAR 1:1 DAR 4:3], 76 kb/s", False) == ("size", (320, 240))
    assert line("[info] Output #0, rawvideo, to 'pipe:1':", False) == ("output", None)
    # (lo de la salida no es el video que llega)
    assert line("[info]   Stream #0:0: Video: rawvideo (NV12 / 0x3231564E), nv12, 640x360, q=2-31", True) == ("", None)


def test_si_la_camara_cambio_de_tamano_no_se_abre_y_lo_dice(tmp_path):
    """La placa se abre con lo que se aprendió con OpenCV. Si la cámara ahora manda otro tamaño (un técnico cambió la
    segunda calidad), no se abre: el CameraWorker vuelve a mirar el video con OpenCV y después pasa a la placa con el
    tamaño nuevo. Si OpenCV vuelve a decir lo mismo (lo cuentan distinto), se abre igual al tamaño pedido."""
    path = video(tmp_path, "otro", size="640x480", seconds=2)
    viejo = {"video_codec": "H.264", "video_w": 640, "video_h": 360, "video_fps": 12.0, "sar": None}
    pc = video_placa.PlacaCapture(str(path), [], viejo, None, decoder=None)
    assert not pc.isOpened() and pc.stale and pc.proc is None
    assert pc.why == "el video ahora llega de 640x480 (se esperaba 640x360)"
    igual = video_placa.PlacaCapture(str(path), [], viejo, None, decoder=None, check_size=False)
    try:
        assert igual.isOpened() and not igual.stale
        ok, img = igual.read()
        assert ok and img.shape == (360, 640, 3)  # al tamaño pedido: los cuadros no salen corridos
    finally:
        igual.release()
    bien = video_placa.PlacaCapture(str(path), [], {**viejo, "video_h": 480}, None, decoder=None)
    try:
        assert bien.isOpened() and bien.input_size == (640, 480)
    finally:
        bien.release()


def test_con_hueco_los_cuadros_quedan_en_la_placa_y_se_baja_solo_los_que_pasan():
    """26 nov, en la máquina de AWS: traer cada cuadro de la placa 0,53 s de procesador por minuto; dejarlos allá y
    bajar uno de cada dos, 0,41 s (con ffmpeg todavía rellenando los que faltaban)."""
    gap = video_placa.MIN_GAP_S
    assert gap == 0.045
    cmd = video_placa.command("rtsp://x/1", [], "hevc_cuvid", (960, 544), (960, 1088), min_gap=gap)
    assert cmd[cmd.index("-hwaccel") + 1] == "cuda" and cmd[cmd.index("-hwaccel_output_format") + 1] == "cuda"
    assert cmd.index("-hwaccel") < cmd.index("-c:v") < cmd.index("-resize") < cmd.index("-i") < cmd.index("-vf")
    keep, *rest = cmd[cmd.index("-vf") + 1].split("',")
    assert keep.startswith("select='") and "gte(t-prev_selected_t,0.045)" in keep and "lt(t,prev_selected_t)" in keep
    assert rest == ["hwdownload,format=nv12"]  # (el filtro no toca la imagen: anda con los cuadros en la placa)
    cpu = video_placa.command("a.mp4", [], None, (960, 540), (960, 1080), min_gap=gap)
    assert "-hwaccel" not in cpu and cpu[cpu.index("-vf") + 1].endswith("',scale=960:540")
    assert video_placa.thin_factor(25, gap) == 2 and video_placa.thin_factor(30, gap) == 2
    assert [video_placa.thin_factor(f, gap) for f in (12, 15, 20)] == [1, 1, 1]  # (esos llegan enteros)
    assert video_placa.thin_factor(60, gap) == 3 and video_placa.thin_factor(25, 0) == 1
    assert video_placa.thin_factor(0, gap) == 1  # (sin saber cuántos manda la cámara)


def _count(pc) -> int:
    n = 0
    while pc.grab():
        n += 1
    return n


def test_llega_uno_de_cada_dos_de_un_video_de_25_y_todos_de_uno_de_12(tmp_path):
    rapido = tmp_path / "rapido.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=640x360:rate=25", "-t", "4", "-c:v", "libx264", "-g", "25", "-pix_fmt", "yuv420p",
                    str(rapido)], check=True)
    hint = {"video_codec": "H.264", "video_w": 640, "video_h": 360, "video_fps": 25.0, "sar": None}
    pc = video_placa.PlacaCapture(str(rapido), [], hint, None, decoder=None, min_gap=video_placa.MIN_GAP_S)
    try:
        assert pc.thin == 2 and pc.get(cv2.CAP_PROP_FPS) == 12.5 and pc.source_fps == 25
        assert _count(pc) == 50  # de los 100 cuadros de 4 s, uno de cada dos
    finally:
        pc.release()
    lento = video(tmp_path, "lento", seconds=3)
    _, info = opencv_frames(lento, 1)
    pc = video_placa.PlacaCapture(str(lento), [], {**info, "sar": None}, None, decoder=None,
                                  min_gap=video_placa.MIN_GAP_S)
    try:
        assert pc.thin == 1 and _count(pc) == 36  # 12 cuadros/s: llegan todos
    finally:
        pc.release()


def test_un_corte_del_video_no_se_rellena_con_cuadros_repetidos(tmp_path):
    """Con el ritmo fijo de fábrica de ffmpeg, un hueco en el video (un corte de la cámara, una que manda menos de lo
    que dice) salía relleno de copias del último cuadro (en la máquina: `dup=359` en un minuto). OpenCV no lo hace."""
    corte = tmp_path / "corte.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=320x240:rate=10:duration=2", "-vf", "setpts='if(gte(N,10),PTS+3/TB,PTS)'",
                    "-fps_mode", "vfr", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(corte)], check=True)
    ref = cv2.VideoCapture(str(corte), cv2.CAP_FFMPEG)
    n_ref = 0
    while ref.grab():
        n_ref += 1
    ref.release()
    assert n_ref == 20
    hint = {"video_codec": "H.264", "video_w": 320, "video_h": 240, "video_fps": 10.0, "sar": None}
    for gap in (0.0, video_placa.MIN_GAP_S):
        pc = video_placa.PlacaCapture(str(corte), [], hint, None, decoder=None, min_gap=gap)
        try:
            assert _count(pc) == n_ref  # los mismos 20 cuadros que OpenCV, sin los 30 repetidos del corte
        finally:
            pc.release()


@pytest.mark.parametrize("name,pix_fmt,rng", [("limitado", "yuv420p", "tv"), ("completo", "yuvj420p", "pc")])
def test_las_imagenes_salen_iguales_que_con_opencv(tmp_path, name, pix_fmt, rng):
    path = video(tmp_path, name, pix_fmt, rng)
    ref, info = opencv_frames(path, 10)
    pc = video_placa.PlacaCapture(str(path), [], {**info, "sar": None}, ref[0], decoder=None)
    try:
        assert pc.isOpened() and pc.get(cv2.CAP_PROP_FRAME_WIDTH) == 960 and pc.get(cv2.CAP_PROP_FPS) == 12
        assert video_info(pc) == info  # el CameraWorker ve el mismo video (códec, tamaño, cuadros por segundo)
        got = [pc.read()[1] for _ in range(10)]
    finally:
        pc.release()
    assert all(g.shape == (576, 960, 3) for g in got)
    assert max(diff(a, b) for a, b in zip(ref, got)) < 1.5  # (de 255: el redondeo de dos conversiones)
    assert pc.proc is None and not pc.isOpened()


def test_el_rango_completo_se_reconoce_comparando_con_la_imagen_de_opencv(tmp_path, monkeypatch):
    """El decodificador de la placa entrega el video tal cual viene: si la cámara lo manda en el rango completo
    (0-255), la conversión de OpenCV (que espera 16-235) daría la imagen lavada. Se reconoce comparando con una imagen
    de OpenCV de un momento antes. Acá se le pide a ffmpeg que no convierta el rango, como hace la placa."""
    path = video(tmp_path, "completo", "yuvj420p", "pc", source="gradients")  # (tonos medios: ahí se nota el rango)
    ref, info = opencv_frames(path, 10)
    real = video_placa.command

    def as_the_card(*a, **k):  # NV12 en el rango completo, como lo entrega el decodificador de la placa
        c = real(*a, **k)
        c[c.index("-vf") + 1] += ":out_range=pc"
        return c
    monkeypatch.setattr(video_placa, "command", as_the_card)
    raw = video_placa.PlacaCapture(str(path), [], {**info, "sar": None}, None, decoder=None)
    plain = raw.read()[1]
    raw.release()
    assert diff(ref[0], plain) > 3  # sin darse cuenta del rango, la imagen sale distinta
    pc = video_placa.PlacaCapture(str(path), [], {**info, "sar": None}, ref[0], decoder=None)
    try:
        assert pc.full_range
        got = [pc.read()[1] for _ in range(10)]
    finally:
        pc.release()
    assert max(diff(a, b) for a, b in zip(ref, got)) < 1.5


def test_si_ffmpeg_no_da_video_no_queda_abierto_y_dice_por_que(tmp_path):
    pc = video_placa.PlacaCapture(str(tmp_path / "no-existe.mp4"), [], {"video_codec": "H.264", "video_w": 640,
                                                                       "video_h": 360}, None, decoder=None,
                                  first_timeout=5)
    assert not pc.isOpened() and pc.proc is None and "no dio video" in pc.why and not pc.grab()
    otro = video_placa.PlacaCapture("rtsp://x/1", [], {"video_codec": "MJPEG", "video_w": 640, "video_h": 360})
    assert not otro.isOpened() and "MJPEG" in otro.why  # (la placa solo decodifica H.264 y H.265 acá)


def test_sin_la_libreria_de_nvidia_no_se_usa_y_lo_dice(monkeypatch):
    monkeypatch.setattr(video_placa, "_available", None)
    monkeypatch.setattr(video_placa.shutil, "which", lambda name: "/usr/bin/ffmpeg")

    def no_lib(name):
        raise OSError("libnvcuvid.so.1: cannot open shared object file")
    monkeypatch.setattr(video_placa.ctypes, "CDLL", no_lib)
    ok, why = video_placa.available()
    assert not ok and "libnvcuvid" in why and "instalar.sh" in why
    monkeypatch.setattr(video_placa, "_available", None)  # (que la próxima prueba vuelva a averiguar)


def test_el_medidor_cuenta_el_ffmpeg_de_la_placa_en_video_y_no_dos_veces(tmp_path):
    if not consumo.thread_cpu():
        pytest.skip("sin /proc (no es Linux)")
    path = video(tmp_path, "largo", seconds=30, codec="libx264")
    _ref, info = opencv_frames(path, 1)
    meter = consumo.CpuMeter(min_span=0)
    meter.sample()
    pc = video_placa.PlacaCapture(str(path), [], {**info, "sar": None}, None, decoder=None)
    n = 0
    while pc.grab() and n < 300:  # (leer todo lo que pueda: el ffmpeg decodifica a toda velocidad)
        n += 1
    mid = meter.sample()
    pc.release()
    time.sleep(1.0)  # (con un rato entre las dos lecturas: en 10 ms, el redondeo de 10 ms del sistema pesaría)
    end = meter.sample()
    assert n == 300 and mid["video"] > 0.05  # lo que gasta el ffmpeg mientras anda cuenta como decodificar video
    assert end.get("externos", 0.0) < 0.05  # y al terminar no aparece otra vez como programa externo


# ---------------------------------------------------------------- de punta a punta: una cámara RTSP simulada
def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_port(port: int, timeout: float = 20) -> None:
    end = time.time() + timeout
    while time.time() < end:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.2)
    raise TimeoutError(f"puerto {port} no abrió")


class Counter:
    def __init__(self):
        self.frames: list[tuple[int, int]] = []
        self.at: list[float] = []

    def detect(self, frame, *a, **k):
        self.frames.append(frame.shape[:2])
        self.at.append(time.monotonic())
        return []

    def rate(self, seconds: float = 5.0) -> float:
        """Cuadros analizados por segundo en los últimos `seconds`."""
        end = self.at[-1] if self.at else 0.0
        return sum(1 for t in self.at if t > end - seconds) / seconds


@pytest.fixture
def rtsp_camera(tmp_path):
    if not Path(MEDIAMTX).exists():
        pytest.skip("sin mediamtx (tools/mediamtx o env MEDIAMTX)")
    port = free_port()
    (tmp_path / "mtx.yml").write_text(f"logLevel: warn\nrtspAddress: :{port}\nprotocols: [tcp]\nrtmp: no\nhls: no\n"
                                      "webrtc: no\nsrt: no\napi: no\nmetrics: no\nplayback: no\npaths:\n  all_others:\n")
    procs = [subprocess.Popen([MEDIAMTX, str(tmp_path / "mtx.yml")], stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)]
    wait_port(port)
    src = video(tmp_path, "camara", size="640x360", seconds=6)
    procs.append(subprocess.Popen(["ffmpeg", "-nostdin", "-loglevel", "error", "-re", "-stream_loop", "-1", "-i",
                                   str(src), "-c", "copy", "-f", "rtsp", "-rtsp_transport", "tcp",
                                   f"rtsp://127.0.0.1:{port}/Streaming/Channels/302"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    time.sleep(1.5)
    yield port
    for p in procs:
        p.terminate()
    for p in procs:
        try:
            p.wait(10)
        except subprocess.TimeoutExpired:
            p.kill()


def run_worker(tmp_path, port: int, seconds: float) -> tuple[CameraWorker, Counter]:
    cfg = {"id": 7, "name": "Puerta", "process_fps": 5, "analysis": {},
           "connection": {"brand": "hikvision", "host": "127.0.0.1", "rtsp_port": port, "username": "vigia",
                          "password": "x", "channel": 3}}
    det = Counter()
    w = CameraWorker(cfg, det, None, worker_mod.Outbox(tmp_path / "outbox", None), None, None,
                     {"clip_pre_seconds": 1, "clip_post_seconds": 1, "clip_max_seconds": 30})
    t = threading.Thread(target=w.run, daemon=True)
    t.start()
    time.sleep(seconds)
    st = w.status()
    w._stop.set()
    t.join(20)
    w.last_status = st
    return w, det


def test_la_camara_pasa_sola_de_opencv_a_la_placa_y_analiza_igual(tmp_path, rtsp_camera, monkeypatch, caplog):
    monkeypatch.setenv("VIGIA_VIDEO_PLACA", "si")
    monkeypatch.setattr(video_placa, "available", lambda: (True, ""))
    monkeypatch.setattr(video_placa, "DECODERS", {"H.264": "", "H.265": ""})  # (acá: el decodificador del procesador)
    with caplog.at_level(logging.INFO, logger="vigia"):
        w, det = run_worker(tmp_path, rtsp_camera, 16)  # (pasar a la placa espera el próximo cuadro completo)
    st = w.last_status
    assert "video por la placa" in caplog.text, caplog.text[-2000:]
    assert st["state"] == "en_linea" and st["video_codec"] == "H.264 · placa"
    assert (st["video_w"], st["video_h"]) == (640, 360)
    # los cuadros por segundo que manda la cámara, contados con OpenCV antes de pasar a la placa (no lo que dice el video)
    assert 10.5 <= w._gpu_hint["video_fps"] <= 13.5 and 10.5 <= st["video_fps"] <= 13.5
    assert det.rate() >= 3.5  # analiza a los cuadros por segundo de siempre (a 5 por segundo, con video de 12)
    assert det.frames and all(f == (360, 640) for f in det.frames)
    assert not video_placa._live  # al cerrar la cámara no queda ningún ffmpeg andando
    assert "no anduvo" not in caplog.text  # (cerrar la cámara no es una falla de la placa)


def test_sin_pedirlo_sigue_con_opencv_y_si_la_placa_falla_tambien(tmp_path, rtsp_camera, monkeypatch, caplog):
    monkeypatch.delenv("VIGIA_VIDEO_PLACA", raising=False)
    with caplog.at_level(logging.INFO, logger="vigia"):
        w, det = run_worker(tmp_path / "a", rtsp_camera, 5)
    assert w.last_status["video_codec"] == "H.264" and "placa" not in caplog.text and det.frames
    # pedida pero el ffmpeg no da video: sigue con OpenCV, sin cortar el análisis
    monkeypatch.setenv("VIGIA_VIDEO_PLACA", "si")
    monkeypatch.setattr(video_placa, "available", lambda: (True, ""))
    monkeypatch.setattr(video_placa, "FIRST_FRAME_TIMEOUT_S", 3)
    real = video_placa.command
    monkeypatch.setattr(video_placa, "command", lambda *a, **k: real(*a, **k)[:-1] + ["/proc/no-se-puede"])
    monkeypatch.setattr(video_placa, "DECODERS", {"H.264": "", "H.265": ""})
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="vigia"):
        w, det = run_worker(tmp_path / "b", rtsp_camera, 9)
    assert "el video por la placa no anduvo" in caplog.text and "10 minutos con el procesador" in caplog.text
    assert w.last_status["state"] == "en_linea" and w.last_status["video_codec"] == "H.264" and det.frames


def test_si_la_camara_cambio_de_tamano_vuelve_a_mirarla_con_opencv_sin_castigar_a_la_placa(tmp_path, monkeypatch, caplog):
    """`_open_gpu` con una cámara que cambió de tamaño: borra lo aprendido (la conexión siguiente es con OpenCV, que lo
    vuelve a aprender y pasa a la placa enseguida), sin los 10 minutos con el procesador de una falla. Si OpenCV vuelve
    a decir el mismo tamaño, la placa se abre sin comparar."""
    monkeypatch.setenv("VIGIA_VIDEO_PLACA", "si")
    monkeypatch.setattr(video_placa, "available", lambda: (True, ""))
    seen = []

    class Fake:
        def __init__(self, url, args, hint, ref, min_gap=0.0, check_size=True):
            seen.append(check_size)
            self.stale = check_size
            self.why = "el video ahora llega de 640x480 (se esperaba 640x360)"
            self.decoder, self.w, self.h, self.full_range, self.thin = "h264_cuvid", 640, 360, False, 1

        def isOpened(self):  # noqa: N802
            return not self.stale

        def release(self):
            pass
    monkeypatch.setattr(video_placa, "PlacaCapture", Fake)
    cfg = {"id": 7, "name": "Puerta", "process_fps": 5, "analysis": {},
           "connection": {"brand": "hikvision", "host": "127.0.0.1", "rtsp_port": 9, "username": "vigia",
                          "password": "x", "channel": 3}}
    w = CameraWorker(cfg, Counter(), None, worker_mod.Outbox(tmp_path / "outbox", None), None, None,
                     {"clip_pre_seconds": 1, "clip_max_seconds": 30})
    hint = {"video_codec": "H.264", "video_w": 640, "video_h": 360, "video_fps": 12.0, "sar": None}
    w._gpu_hint = dict(hint)
    url = "rtsp://vigia:x@127.0.0.1:9/Streaming/Channels/302"
    with caplog.at_level(logging.INFO, logger="vigia"):
        assert w._open_gpu(url) is None
    assert w._gpu_hint is None and w._gpu_relearn == (640, 360) and w._gpu_off_until == 0.0
    assert "se vuelve a mirar el video con OpenCV" in caplog.text and "no anduvo" not in caplog.text
    w._gpu_hint = dict(hint)  # OpenCV dijo lo mismo: ahora se abre sin comparar el tamaño
    assert isinstance(w._open_gpu(url), Fake) and seen == [True, False]
