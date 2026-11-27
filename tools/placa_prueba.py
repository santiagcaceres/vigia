"""¿Qué más se puede pasar a la placa de video? (27 nov: con TensorRT el modelo gasta la mitad de procesador; lo que
sigue pesando en el procesador es leer el video —un ffmpeg por cámara con la placa tiene un costo fijo de ~0,03
procesadores, lo mismo que decodificar un video liviano con el procesador—, comprimir los clips de las alertas con
libx264 y la postura, que sigue en CUDA). Esto mide EN LA MÁQUINA, en un proceso aparte, las tres cosas antes de
cambiarlas en producción. No toca las cámaras (usa un poco de procesador y de placa unos 3 minutos).

    curl -fsSL https://raw.githubusercontent.com/santiagcaceres/vigia/main/tools/placa_prueba.py -o /tmp/pp.py
    cd /opt/vigia && sudo -u vigia -H nice -n 10 /opt/vigia/.venv/bin/python /tmp/pp.py

1. **Video:** 8 «cámaras» leyendo a la velocidad real el video de prueba (/tmp/prueba.mp4, H.265 960x576; y una
   copia en H.264 640x480 a 25 cuadros/s, como las de Bazar) dentro de UN proceso, como lo haría la máquina: con el
   procesador (un hilo por cámara, como hoy) y con la placa compartiendo el contexto de CUDA del proceso (PyAV; un solo
   «enchufe» a la placa para todas, en vez de un ffmpeg por cámara). En las dos se pasan a colores 4,5 cuadros por
   segundo (los que se analizan). Dice cuánto procesador y cuánta memoria de la placa gasta cada cámara y si las
   imágenes salen iguales. PyAV se baja aparte (pip, ~40 MB, a /var/tmp/vigia-av): no toca el programa.
2. **Clips:** el mismo clip de 30 s comprimido como hoy (libx264) y con el compresor de la placa (NVENC): procesador,
   tiempo, tamaño y parecido con el original.
3. **Postura con TensorRT:** los puntos del cuerpo de las personas de dos fotos con CUDA y con TensorRT (media
   precisión): cuánto se mueven los puntos y cuánto tarda cada uno. La primera vez arma el motor (unos minutos) y lo
   deja guardado donde lo buscaría el programa.

`--solo video|clips|postura` corre una sola parte. No imprime links ni claves."""

from __future__ import annotations

import argparse
import os
import resource
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

AV_DIR = Path("/var/tmp/vigia-av")
SRC = Path("/tmp/prueba.mp4")
SRC_H264 = Path("/var/tmp/vigia-prueba-h264.mp4")
TRT_LIBS = (Path("/var/lib/vigia/trt/tensorrt_libs"), Path("/var/tmp/vigia-trt/tensorrt_libs"))


def say(*a) -> None:
    print(*a, flush=True)


def gpu_mem_of(pid: int) -> int | None:
    """MB de la placa que usa este proceso (según nvidia-smi), o None."""
    if not shutil.which("nvidia-smi"):
        return None
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=15, check=False)
    for line in r.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0] == str(pid) and parts[1].isdigit():
            return int(parts[1])
    return 0


# ---------------------------------------------------------------------------------------------------------- 1. video

def load_av():
    try:
        import av
        return av
    except ImportError:
        pass
    if not (AV_DIR / "av").exists():
        say("Bajando PyAV (una sola vez, ~40 MB)…")
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--target", str(AV_DIR), "av>=19,<20"],
                           capture_output=True, text=True, check=False)
        if r.returncode != 0:
            say("   No se pudo bajar PyAV:", (r.stderr.strip().splitlines() or ["?"])[-1][:300])
            return None
    sys.path.insert(0, str(AV_DIR))
    import av
    return av


def make_h264_copy() -> bool:
    if SRC_H264.exists() and SRC_H264.stat().st_size > 0:
        return True
    say("Armando una copia del video en H.264 640x480 a 25 cuadros/s (una sola vez)…")
    r = subprocess.run(["nice", "-n", "15", "ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(SRC), "-an",
                        "-vf", "scale=640:480,fps=25", "-c:v", "libx264", "-preset", "veryfast", "-g", "50",
                        "-threads", "2", str(SRC_H264)], capture_output=True, text=True, timeout=600, check=False)
    return r.returncode == 0 and SRC_H264.exists()


def to_bgr(frame, hw: bool):
    import cv2
    if hw:
        frame = frame.reformat(format="nv12")  # (baja ESTE cuadro de la placa; los demás quedan allá)
    fmt = frame.format.name
    if fmt == "nv12":
        return cv2.cvtColor(frame.to_ndarray(), cv2.COLOR_YUV2BGR_NV12)
    if fmt in ("yuv420p", "yuvj420p"):
        return cv2.cvtColor(frame.reformat(format="yuv420p").to_ndarray(), cv2.COLOR_YUV2BGR_I420)
    return frame.to_ndarray(format="bgr24")


def camera(av, path: Path, mode: str, seconds: float, stats: dict, idx: int, stop: threading.Event) -> None:
    """Una «cámara»: lee el archivo a la velocidad real (en vuelta) y pasa a colores 4,5 cuadros por segundo."""
    hw = mode == "placa"
    accel = None
    if hw:
        from av.codec.hwaccel import HWAccel
        accel = HWAccel("cuda", allow_software_fallback=False, options={"primary_ctx": "1"}, is_hw_owned=True)
    got = used = 0
    t0 = time.monotonic()
    next_use = 0.0
    offset = 0.0
    try:
        while not stop.is_set() and time.monotonic() - t0 < seconds:
            with av.open(str(path), hwaccel=accel) as cont:
                st = cont.streams.video[0]
                if not hw:
                    st.codec_context.thread_count = 1
                last = 0.0
                for frame in cont.decode(st):
                    if stop.is_set():
                        break
                    ts = offset + float(frame.time or 0.0)
                    last = ts
                    wait = t0 + ts - time.monotonic()
                    if wait > 0:
                        time.sleep(wait)
                    if time.monotonic() - t0 >= seconds:
                        break
                    got += 1
                    if ts >= next_use:
                        next_use = ts + 1 / 4.5
                        img = to_bgr(frame, hw)
                        used += 1
                        if idx == 0 and used == 20:
                            stats["sample"] = img
                offset = last + 0.05
    except Exception as exc:  # noqa: BLE001
        stats.setdefault("errors", []).append(f"{type(exc).__name__}: {exc}"[:200])
    stats.setdefault("frames", []).append(got)
    stats.setdefault("used", []).append(used)


def run_video_mode(av, path: Path, mode: str, cams: int, seconds: float) -> dict:
    stats: dict = {}
    stop = threading.Event()
    c0 = time.process_time()
    w0 = time.monotonic()
    threads = [threading.Thread(target=camera, args=(av, path, mode, seconds, stats, i, stop), daemon=True)
               for i in range(cams)]
    for t in threads:
        t.start()
    time.sleep(min(seconds * 0.6, 12))
    stats["gpu_mb"] = gpu_mem_of(os.getpid())
    stats["threads"] = len(os.listdir("/proc/self/task"))
    for t in threads:
        t.join(seconds + 30)
    stop.set()
    stats["cpu"] = (time.process_time() - c0) / max(time.monotonic() - w0, 1e-6)
    return stats


def video_part(cams: int, seconds: float) -> None:
    say("\n== 1. Leer el video de varias cámaras en un solo proceso")
    if not SRC.exists():
        say(f"   Falta el video de prueba {SRC} (el de la prueba del 26 nov). Se puede volver a bajar de una cámara.")
        return
    av = load_av()
    if av is None:
        return
    say(f"   PyAV {av.__version__}; la placa: {'sí' if 'cuda' in av.codec.hwaccel.hwdevices_available() else 'NO'}")
    sources = [("H.265 960x576 a 12", SRC)]
    if make_h264_copy():
        sources.append(("H.264 640x480 a 25", SRC_H264))
    for label, path in sources:
        results = {}
        for mode in ("procesador", "placa"):
            s = run_video_mode(av, path, mode, cams, seconds)
            results[mode] = s
            fps = np.mean(s.get("frames") or [0]) / seconds
            err = f" · ERRORES: {s['errors'][0]}" if s.get("errors") else ""
            mb = s.get("gpu_mb")
            mem = "" if mb is None or mode == "procesador" else \
                f" · placa {mb} MB en total (incluye ~300 del contexto de CUDA, que en la máquina ya lo tiene el modelo)"
            say(f"   {label}, {mode:<10}: {s['cpu']:.3f} procesadores para {cams} cámaras = "
                f"**{s['cpu'] / cams:.4f} por cámara** · {fps:.1f} cuadros/s leídos por cámara · "
                f"{s.get('threads')} hilos{mem}{err}")
        a, b = results["procesador"].get("sample"), results["placa"].get("sample")
        if a is not None and b is not None and a.shape == b.shape:
            diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
            say(f"   las imágenes: diferencia media {diff.mean():.2f} de 255 (máxima {int(diff.max())})")
    say("   (Hoy, con un ffmpeg por cámara por la placa: ~0,031 por cámara; con el procesador, 0,02-0,045 según el "
        "video.)")


# ---------------------------------------------------------------------------------------------------------- 2. clips

def children_cpu() -> float:
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime + r.ru_stime


def psnr(a, b) -> float:
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if mse <= 1e-9 else 10 * np.log10(255 ** 2 / mse)


def frame_at(path: Path, sec: float):
    import cv2
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    ok, f = cap.read()
    cap.release()
    return f if ok else None


def clips_part() -> None:
    say("\n== 2. Comprimir el clip de una alerta (30 s)")
    if not SRC.exists() or not shutil.which("ffmpeg"):
        say("   Falta el video de prueba o ffmpeg.")
        return
    out = Path("/var/tmp/vigia-clip-prueba")
    out.mkdir(parents=True, exist_ok=True)
    common_in = ["-nostdin", "-y", "-loglevel", "error"]
    scale = ["-vf", "scale='min(1280,iw)':-2"]
    tail = ["-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    variants = {
        "hoy (libx264, 1 hilo)": ["-threads", "1"] + common_in + ["-i", str(SRC), "-t", "30", "-an"] + scale + [
            "-threads", "1", "-filter_threads", "1", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26"] + tail,
        "NVENC": ["-threads", "1"] + common_in + ["-i", str(SRC), "-t", "30", "-an"] + scale + [
            "-filter_threads", "1", "-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "27", "-b:v", "0"]
        + tail,
        "placa entera (cuvid + NVENC)": common_in + ["-c:v", "hevc_cuvid", "-i", str(SRC), "-t", "30", "-an"]
        + scale + ["-filter_threads", "1", "-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "27",
                   "-b:v", "0"] + tail,
    }
    ref = frame_at(SRC, 12.0)
    for i, (label, args) in enumerate(variants.items()):
        dst = out / f"clip{i}.mp4"
        c0, w0 = children_cpu(), time.monotonic()
        r = subprocess.run(["nice", "-n", "15", "ffmpeg"] + args + [str(dst)], capture_output=True, text=True,
                           timeout=300, check=False)
        cpu, wall = children_cpu() - c0, time.monotonic() - w0
        if r.returncode != 0:
            say(f"   {label:<30}: no anduvo ({(r.stderr.strip().splitlines() or ['?'])[-1][:200]})")
            continue
        got = frame_at(dst, 12.0)
        q = f" · parecido al original {psnr(ref, got):.1f} dB" if ref is not None and got is not None and \
            got.shape == ref.shape else ""
        say(f"   {label:<30}: {cpu:5.2f} s de procesador · {wall:5.1f} s de reloj · "
            f"{dst.stat().st_size / 1e6:.1f} MB{q}")
    shutil.rmtree(out, ignore_errors=True)


# -------------------------------------------------------------------------------------------------------- 3. postura

def preload_trt() -> Path | None:
    import ctypes
    for d in TRT_LIBS:
        if (d / "libnvinfer.so.10").exists():
            for name in ("libnvinfer.so.10", "libnvinfer_plugin.so.10", "libnvonnxparser.so.10"):
                if (d / name).exists():
                    ctypes.CDLL(str(d / name), mode=ctypes.RTLD_GLOBAL)
            return d
    return None


def photos():
    import urllib.request

    import cv2
    out = []
    for name in ("bus.jpg", "zidane.jpg"):
        url = f"https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/{name}"
        try:
            img = cv2.imdecode(np.frombuffer(urllib.request.urlopen(url, timeout=20).read(), np.uint8),
                               cv2.IMREAD_COLOR)
        except OSError:
            img = None
        if img is not None:
            out += [img, img[:, ::-1].copy(), cv2.resize(img, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)]
    return out


def bench(sess, batch: int, runs: int = 40) -> tuple[float, float]:
    rng = np.random.default_rng(3)
    arr = rng.integers(0, 255, (batch, sess.size[1], sess.size[0], 3), dtype=np.uint8)
    feed = {sess.input_name: arr}
    for _ in range(5):
        sess.session.run(None, feed)
    w0, c0 = time.perf_counter(), time.thread_time()
    for _ in range(runs):
        sess.session.run(None, feed)
    return 1000 * (time.perf_counter() - w0) / (runs * batch), 1000 * (time.thread_time() - c0) / (runs * batch)


def pose_part() -> None:
    say("\n== 3. Postura con TensorRT")
    import onnxruntime as ort
    if "TensorrtExecutionProvider" not in ort.get_available_providers():
        say("   Este onnxruntime no trae TensorRT.")
        return
    libs = preload_trt()
    if not libs:
        say("   No están las librerías de TensorRT (las bajó tools/tensorrt_prueba.py --instalar).")
        return
    os.environ["VIGIA_TRT_LIBS"] = str(libs)
    os.environ["VIGIA_TENSORRT"] = "todo"
    from core.detectors import OnnxSession, OnnxYoloDetector
    from core.pose import PoseEstimator, default_model
    path = default_model()
    cuda = OnnxSession(path, threads=2, max_batch=8, all_outputs=True, trt=False)
    say(f"   CUDA: {cuda.device}. Armando el motor de TensorRT (la primera vez, unos minutos)…")
    t0 = time.monotonic()
    trt = OnnxSession(path, threads=2, max_batch=8, all_outputs=True, trt=True)
    say(f"   TensorRT: {trt.device}, listo en {time.monotonic() - t0:.0f} s")
    if "Tensorrt" not in trt.device:
        say("   No quedó en TensorRT: no se compara.")
        return
    det = OnnxYoloDetector(Path("models/personas.onnx"), session=OnnxSession(Path("models/personas.onnx"),
                                                                             threads=2, max_batch=1))
    a_est = PoseEstimator(path, runner=cuda)
    b_est = PoseEstimator(path, runner=trt)
    moved: list[float] = []
    dconf: list[float] = []
    people = 0
    for img in photos():
        boxes = [d.bbox for d in det.detect(img)]
        if not boxes:
            continue
        a = a_est.estimate_many(img, boxes)
        b = b_est.estimate_many(img, boxes)
        for ka, kb, box in zip(a, b, boxes):
            if ka is None or kb is None:
                continue
            people += 1
            torso = max(abs(box[3] - box[1]) / 3, 1.0)  # (un largo de torso ≈ un tercio del alto de la persona)
            ok = (ka[:, 2] > 0.3) & (kb[:, 2] > 0.3)
            moved += list(np.hypot(*(ka[ok, :2] - kb[ok, :2]).T) / torso)
            dconf += list(np.abs(ka[:, 2] - kb[:, 2]))
    if moved:
        say(f"   {people} personas: los puntos se mueven {np.mean(moved):.3f} largos de torso en promedio (el peor "
            f"{max(moved):.3f}; los gestos usan umbrales de 0,35 a 0,5) · confianza {np.mean(dconf):.3f} (peor "
            f"{max(dconf):.3f})")
    else:
        say("   No hubo personas para comparar (¿no se pudieron bajar las fotos?).")
    say("   lote   CUDA ms/persona (procesador)   TensorRT ms/persona (procesador)")
    for n in (1, 4, 8):
        cw, cc = bench(cuda, n)
        tw, tc = bench(trt, n)
        say(f"   {n:>4}   {cw:8.2f} ({cc:5.2f})                {tw:8.2f} ({tc:5.2f})        ×{cw / tw:.1f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo", choices=("video", "clips", "postura"))
    ap.add_argument("--camaras", type=int, default=8)
    ap.add_argument("--segundos", type=float, default=20.0)
    args = ap.parse_args()
    with open("/proc/loadavg", encoding="ascii") as f:
        load = f.read().split()[:3]
    say(f"Carga de la máquina ahora: {' '.join(load)} (con las cámaras andando)")
    parts = {"video": lambda: video_part(args.camaras, args.segundos), "clips": clips_part, "postura": pose_part}
    for name, fn in parts.items():
        if args.solo and args.solo != name:
            continue
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - una parte que falla no corta las otras
            say(f"   Falló: {type(exc).__name__}: {str(exc)[:300]}")


if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    main()
