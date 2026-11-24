from __future__ import annotations

import logging
import math
import os
import re
import shutil
import subprocess
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


def transcode_h264(src: str | Path, dst: str | Path, max_width: int = 1280, timeout: int = 180,
                   extra_input: list[str] | None = None, duration: float | None = None) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    cmd = [ffmpeg, "-y", "-loglevel", "error"] + (extra_input or []) + ["-i", str(src)]
    if duration:
        cmd += ["-t", f"{duration:.2f}"]
    cmd += ["-an", "-vf", f"scale='min({max_width},iw)':-2", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "26", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)]
    try:
        r = subprocess.run(cmd, timeout=timeout, capture_output=True)
    except subprocess.TimeoutExpired:
        log.warning("ffmpeg excedió el tiempo procesando %s", redact(str(src)))
        return False
    if r.returncode != 0:
        err = redact(r.stderr.decode(errors="replace").strip().splitlines()[-1] if r.stderr.strip() else "")
        log.warning("ffmpeg falló (%s): %s", redact(str(src)), err)
        return False
    return Path(dst).exists() and Path(dst).stat().st_size > 0


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
