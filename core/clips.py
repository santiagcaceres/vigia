from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
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


def clip_fps(frames: list[tuple[float, object]]) -> float:
    """Los cuadros por segundo con que `write_mp4` escribe esos cuadros (los reparte parejo en el tiempo que cubren)."""
    span = frames[-1][0] - frames[0][0] if len(frames) > 1 else 0.0
    return max(1.0, min(30.0, (len(frames) - 1) / span)) if span > 0 else 5.0


def video_offset(frames: list[tuple[float, object]], ts: float) -> float | None:
    """En qué segundo del MP4 que arma `write_mp4` con `frames` queda el momento `ts` del stream (el de la alerta).
    No es `ts` menos el primero: en los ratos quietos el buffer guarda 1 cuadro por segundo y el MP4 los pasa a la
    misma velocidad que los demás. None si no hay cuadros."""
    if not frames:
        return None
    idx = next((i for i, (t, _) in enumerate(frames) if t >= ts - 1e-6), len(frames) - 1)
    return round(idx / clip_fps(frames), 2)


def write_mp4(frames: list[tuple[float, np.ndarray | bytes]], path: str | Path) -> bool:
    """Escribe frames (ts, imagen BGR o JPEG en bytes) como MP4 H.264 reproducible en navegadores.

    Los JPEG se decodifican de a uno, así un clip de 3 minutos no ocupa cientos de MB en memoria.
    Usa ffmpeg si está disponible (H.264). Si no, deja el MP4 de OpenCV (mp4v), que se
    descarga bien pero algunos navegadores no reproducen.
    """
    path = Path(path)
    if len(frames) < 2:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fps = clip_fps(frames)
    first = _decode(frames[0][1])
    if first is None:
        return False
    h, w = first.shape[:2]
    tmp = path.with_name(path.stem + ".raw.mp4")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():
        log.warning("No se pudo abrir VideoWriter para %s", path)
        return False
    for _, f in frames:
        f = _decode(f)
        if f is None:
            continue
        if f.shape[:2] != (h, w):
            f = cv2.resize(f, (w, h))
        vw.write(f)
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
