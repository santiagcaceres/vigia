from __future__ import annotations

import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("vigia.clips")
# hasta el último «@» antes del host (una clave con «@» sin codificar, como «Abc@2024», no queda a la vista)
_CREDENTIALS = re.compile(r"(\w+://)[^/\s]*@")


def redact(text: str) -> str:
    """Quita usuario y clave de cualquier URL (rtsp://user:clave@host -> rtsp://***@host)."""
    return _CREDENTIALS.sub(r"\1***@", text)


def _decode(f) -> np.ndarray | None:
    if isinstance(f, (bytes, bytearray, memoryview)):
        return cv2.imdecode(np.frombuffer(f, np.uint8), cv2.IMREAD_COLOR)
    return f


# Un hueco largo entre dos cuadros (un corte de la cámara dentro del clip) no se muestra entero: la imagen quieta
# dura como mucho esto y el clip sigue (en los ratos quietos el buffer guarda 1 cuadro por segundo: no llegan acá)
MAX_HOLD_S = 2.5
MAX_CLIP_FPS = 15.0


def clip_fps(frames: list[tuple[float, object]]) -> float:
    """Los cuadros por segundo del MP4 que arma `write_mp4`: el ritmo al que llegaron los cuadros con gente (el
    intervalo entre cuadros seguidos del cuarto más corto; en los ratos quietos el buffer guarda 1 por segundo y no
    cuentan), redondeado para arriba para no perder cuadros, entre 1 y 15."""
    gaps = sorted(b[0] - a[0] for a, b in pairwise(frames) if b[0] > a[0])
    if not gaps:
        return 5.0
    return float(max(1, min(int(MAX_CLIP_FPS), math.ceil(1.0 / gaps[len(gaps) // 4] - 0.05))))


def clip_timeline(frames: list[tuple[float, object]]) -> tuple[float, list[int]]:
    """Cómo queda el clip a la velocidad real (24 nov; antes los cuadros se repartían parejo y, con ratos quietos de 1
    cuadro por segundo y ratos con gente de 5, lo que tenía gente se veía en cámara lenta): (cuadros por segundo del
    MP4, qué cuadro de `frames` se ve en cada cuadro del MP4). Un cuadro se repite hasta que llega el siguiente; un
    hueco de más de `MAX_HOLD_S` se acorta a eso."""
    if not frames:
        return 5.0, []
    fps = clip_fps(frames)
    vt = [0.0]  # la hora de cada cuadro en el clip (con los huecos largos acortados)
    for a, b in pairwise(frames):
        vt.append(vt[-1] + min(MAX_HOLD_S, max(0.0, b[0] - a[0])))
    slots = math.floor(vt[-1] * fps + 1e-6) + 1
    out, i = [], 0
    for k in range(slots):
        t = k / fps + 1e-6
        while i + 1 < len(frames) and vt[i + 1] <= t:
            i += 1
        out.append(i)
    if out[-1] != len(frames) - 1:  # (el último cuadro se ve aunque caiga entre dos del MP4)
        out.append(len(frames) - 1)
    return fps, out


def video_offset(frames: list[tuple[float, object]], ts: float) -> float | None:
    """En qué segundo del MP4 que arma `write_mp4` con `frames` queda el momento `ts` del stream (el de la alerta): el
    primer cuadro del MP4 que muestra ese momento o uno posterior. None si no hay cuadros."""
    if not frames:
        return None
    idx = next((i for i, (t, _) in enumerate(frames) if t >= ts - 1e-6), len(frames) - 1)
    fps, out = clip_timeline(frames)
    k = next((k for k, i in enumerate(out) if i >= idx), len(out) - 1)
    return round(k / fps, 2)


def write_mp4(frames: list[tuple[float, np.ndarray | bytes]], path: str | Path) -> bool:
    """Escribe frames (ts, imagen BGR o JPEG en bytes) como MP4 H.264 reproducible en navegadores, a la velocidad real
    (`clip_timeline`: cada cuadro dura hasta que llega el siguiente).

    Los JPEG se decodifican de a uno, así un clip de 3 minutos no ocupa cientos de MB en memoria.
    Usa ffmpeg si está disponible (H.264). Si no, deja el MP4 de OpenCV (mp4v), que se
    descarga bien pero algunos navegadores no reproducen.
    """
    path = Path(path)
    if len(frames) < 2:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fps, order = clip_timeline(frames)
    first = _decode(frames[0][1])
    if first is None:
        return False
    h, w = first.shape[:2]
    if _pipe_mp4(frames, order, fps, (w, h), path):  # (ffmpeg recibe los cuadros por un caño: sin el MP4 intermedio)
        return True
    tmp = path.with_name(path.stem + ".raw.mp4")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():
        log.warning("No se pudo abrir VideoWriter para %s", path)
        return False
    last_i, img = 0, first
    for i in order:
        if i != last_i:  # (un cuadro repetido no se vuelve a decodificar)
            f = _decode(frames[i][1])
            if f is not None:
                img = f if f.shape[:2] == (h, w) else cv2.resize(f, (w, h))
            last_i = i
        vw.write(img)
    vw.release()
    if transcode_h264(tmp, path):
        tmp.unlink(missing_ok=True)
    else:
        os.replace(tmp, path)
    return path.exists() and path.stat().st_size > 0


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


# ffmpeg arma cada clip (lo pasa a H.264 y, con `nvr_clips`, lo baja del grabador). Por defecto usaba un hilo por
# procesador para decodificar y otros tantos para comprimir, con la misma prioridad que el análisis: el 26 nov, en la
# máquina de AWS, 10 cámaras nuevas sin ajustar dieron 51 alertas en 20 minutos, cada una con su ffmpeg, y la máquina
# quedó con carga 51 en 4 procesadores (98 % del tiempo había procesos esperando procesador): las cámaras de todos los
# clientes bajaron a 1-3 cuadros/s. Ahora cada ffmpeg usa un hilo (`VIGIA_FFMPEG_HILOS`; 0 = los de fábrica) y corre
# con prioridad baja (`VIGIA_FFMPEG_NICE`, de 0 a 19; 0 = la misma que el análisis): el clip tarda más en estar listo
# cuando la máquina está ocupada, pero nunca le saca procesador al análisis en vivo
FFMPEG_THREADS = _env_int("VIGIA_FFMPEG_HILOS", 1)
FFMPEG_NICE = min(max(_env_int("VIGIA_FFMPEG_NICE", 15), 0), 19)


def ffmpeg_command(ffmpeg: str, args: list[str]) -> list[str]:
    """ffmpeg con prioridad baja (`nice`, donde lo hay: no en Windows) y la cantidad de hilos de `FFMPEG_THREADS`."""
    threads = ["-threads", str(FFMPEG_THREADS)] if FFMPEG_THREADS > 0 else []
    nice = shutil.which("nice") if FFMPEG_NICE > 0 and os.name != "nt" else None
    return ([nice, "-n", str(FFMPEG_NICE)] if nice else []) + [ffmpeg] + threads + args


# Comprimir el clip con la placa de video (27 nov: en la máquina de AWS el procesador es el límite y la placa sobra; con
# muchas alertas juntas los clips se hacían cola esperando procesador). Con una placa NVIDIA, ffmpeg comprime con NVENC
# (el compresor de la placa: casi no usa procesador) y decodifica lo que entra con la placa (`-hwaccel cuda`, que si no
# puede vuelve solo al procesador). `VIGIA_CLIPS_PLACA`: «auto» (de fábrica: si hay `nvidia-smi` y ffmpeg puede
# comprimir una prueba con NVENC), «si» (sin la prueba) o «no». Si la placa falla al comprimir, ese clip se rehace con
# libx264 como antes y la placa no se vuelve a usar para clips en una hora.
# (cq 30: medido en la máquina de AWS el 27 nov, con 27 el clip salía con más calidad que libx264 —39,4 dB contra
# 35,9— pero 60 % más pesado; con 30 queda más o menos del mismo peso, y el Storage cobra por lo que se guarda)
NVENC_ARGS = ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "30", "-b:v", "0"]
X264_ARGS = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "26"]
NVENC_RETRY_S = 3600.0
_GPU_ERROR = re.compile(r"nvenc|cuda|cuvid|nvdec|OpenEncodeSession|No capable devices|libnvidia|hwaccel|"
                        r"hardware|device", re.IGNORECASE)
_nvenc: dict = {"ok": None, "off_until": 0.0}
_nvenc_lock = threading.Lock()


def _gpu_mode() -> str:
    mode = (os.getenv("VIGIA_CLIPS_PLACA") or "auto").strip().lower()
    return "no" if mode in ("no", "0", "false", "off") else ("si" if mode in ("si", "sí", "1", "true") else "auto")


def use_gpu(ffmpeg: str | None = None) -> bool:
    """¿Comprimir los clips con la placa? (se averigua una vez por proceso)."""
    mode = _gpu_mode()
    if mode == "no" or time.monotonic() < _nvenc["off_until"]:
        return False
    with _nvenc_lock:
        if _nvenc["ok"] is None:
            ffmpeg = ffmpeg or shutil.which("ffmpeg")
            if not ffmpeg:
                return False
            if mode == "si":
                _nvenc["ok"] = True
            elif not shutil.which("nvidia-smi"):
                _nvenc["ok"] = False
            else:
                _nvenc["ok"] = _nvenc_works(ffmpeg)
                log.info("Clips: %s", "se comprimen con la placa de video (NVENC)" if _nvenc["ok"] else
                         "la placa no los puede comprimir: siguen con el procesador")
        return bool(_nvenc["ok"])


def _nvenc_works(ffmpeg: str) -> bool:
    try:
        r = subprocess.run([ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                            "color=c=gray:s=320x240:d=0.4:r=10", "-c:v", "h264_nvenc", "-f", "null", "-"],
                           capture_output=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    return getattr(r, "returncode", 1) == 0


def _gpu_failed(err: str) -> None:
    _nvenc["off_until"] = time.monotonic() + NVENC_RETRY_S
    log.warning("Clips: la placa no pudo comprimir (%s); se usa el procesador por una hora", redact(err)[:200])


def _even_size(w: int, h: int, max_width: int = 1280) -> tuple[int, int]:
    ow = max(2, min(max_width, w) // 2 * 2)
    oh = max(2, round(h * ow / max(w, 1) / 2) * 2)
    return ow, oh


def _pipe_mp4(frames: list, order: list[int], fps: float, size: tuple[int, int], path: Path,
              timeout: int = 180) -> bool:
    """El clip en una sola pasada: los cuadros van a ffmpeg por un caño (los JPEG del buffer tal cual, sin
    decodificarlos acá) y sale el MP4 H.264. Antes se decodificaba cada JPEG, se escribía un MP4 intermedio con OpenCV
    y ffmpeg lo volvía a leer y comprimir. False si no anduvo (y se hace como antes)."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    w, h = size
    jpeg = all(isinstance(f, (bytes, bytearray, memoryview)) for _, f in frames)
    ow, oh = _even_size(w, h)
    for gpu in ([True, False] if use_gpu(ffmpeg) else [False]):
        if jpeg:
            inp = (["-hwaccel", "cuda"] if gpu else []) + ["-f", "image2pipe", "-framerate", f"{fps:.4f}",
                                                           "-c:v", "mjpeg", "-i", "pipe:0"]
        else:
            inp = ["-f", "rawvideo", "-pix_fmt", "bgr24", "-video_size", f"{w}x{h}", "-framerate", f"{fps:.4f}",
                   "-i", "pipe:0"]
        threads = ["-filter_threads", str(FFMPEG_THREADS)] if FFMPEG_THREADS > 0 else []
        threads += ["-threads", str(FFMPEG_THREADS)] if FFMPEG_THREADS > 0 and not gpu else []
        args = ["-y", "-loglevel", "error"] + inp + ["-an", "-vf", f"scale={ow}:{oh}"] + threads + \
            (NVENC_ARGS if gpu else X264_ARGS) + ["-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
        cmd = ffmpeg_command(ffmpeg, args)
        ok, err = _feed(cmd, frames, order, (w, h), jpeg, timeout)
        if ok and path.exists() and path.stat().st_size > 0:
            return True
        if gpu and _GPU_ERROR.search(err or ""):
            _gpu_failed(err)
            continue
        if err:
            log.warning("ffmpeg no armó el clip %s: %s", path.name, redact(err)[:200])
        return False
    return False


def _feed(cmd: list[str], frames: list, order: list[int], size: tuple[int, int], jpeg: bool,
          timeout: int) -> tuple[bool, str]:
    with tempfile.TemporaryFile() as errf:
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errf)
        except OSError as exc:
            return False, str(exc)
        deadline = time.monotonic() + timeout
        ok = True
        try:
            last_i, data = None, b""
            for i in order:
                if time.monotonic() > deadline:
                    ok = False
                    break
                if i != last_i:
                    f = frames[i][1]
                    if jpeg:
                        data = bytes(f)
                    else:
                        img = _decode(f)
                        if img is not None:
                            if img.shape[:2] != (size[1], size[0]):
                                img = cv2.resize(img, size)
                            data = np.ascontiguousarray(img).tobytes()
                    last_i = i
                if data:
                    proc.stdin.write(data)
            proc.stdin.close()
        except (BrokenPipeError, OSError, ValueError):
            ok = False
        try:
            proc.wait(timeout=max(5.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            ok = False
        errf.seek(0)
        err = errf.read().decode(errors="replace").strip()
        last = err.splitlines()[-1] if err else ""
        return ok and proc.returncode == 0, (err if _GPU_ERROR.search(err) else last)


def transcode_h264(src: str | Path, dst: str | Path, max_width: int = 1280, timeout: int = 180,
                   extra_input: list[str] | None = None, duration: float | None = None) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    for gpu in ([True, False] if use_gpu(ffmpeg) else [False]):
        args = ["-y", "-loglevel", "error"] + (["-hwaccel", "cuda"] if gpu else []) + (extra_input or []) + \
            ["-i", str(src)]
        if duration:
            args += ["-t", f"{duration:.2f}"]
        threads = ["-threads", str(FFMPEG_THREADS), "-filter_threads", str(FFMPEG_THREADS)] if FFMPEG_THREADS > 0 \
            else []
        args += ["-an", "-vf", f"scale='min({max_width},iw)':-2"] + threads + (NVENC_ARGS if gpu else X264_ARGS) + [
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)]
        cmd = ffmpeg_command(ffmpeg, args)  # (el primer «-threads», antes de «-i», es para decodificar)
        try:
            r = subprocess.run(cmd, timeout=timeout, capture_output=True, check=False)
        except subprocess.TimeoutExpired:
            log.warning("ffmpeg excedió el tiempo procesando %s", redact(str(src)))
            return False
        if r.returncode != 0:
            text = r.stderr.decode(errors="replace").strip() if r.stderr else ""
            if gpu and _GPU_ERROR.search(text):
                _gpu_failed(text.splitlines()[-1] if text else "")
                continue
            err = redact(text.splitlines()[-1] if text else "")
            log.warning("ffmpeg falló (%s): %s", redact(str(src)), err)
            return False
        return Path(dst).exists() and Path(dst).stat().st_size > 0
    return False


def video_duration(path: str | Path) -> float | None:
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        try:
            r = subprocess.run([ffprobe, "-v", "error", "-show_entries", "format=duration",
                                "-of", "default=nw=1:nk=1", str(path)], capture_output=True, timeout=30)
            return float(r.stdout.decode().strip())
        except Exception:
            pass
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        return n / fps if fps > 0 else None
    finally:
        cap.release()
