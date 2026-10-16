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
_CREDENTIALS = re.compile(r"(\w+://)[^@/\s]+@")


def redact(text: str) -> str:
    """Quita usuario y clave de cualquier URL (rtsp://user:clave@host -> rtsp://***@host)."""
    return _CREDENTIALS.sub(r"\1***@", text)


def write_mp4(frames: list[tuple[float, np.ndarray]], path: str | Path) -> bool:
    """Escribe frames (ts, imagen BGR) como MP4 H.264 reproducible en navegadores.

    Usa ffmpeg si está disponible (H.264). Si no, deja el MP4 de OpenCV (mp4v), que se
    descarga bien pero algunos navegadores no reproducen.
    """
    path = Path(path)
    if len(frames) < 2:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    span = frames[-1][0] - frames[0][0]
    fps = max(1.0, min(30.0, (len(frames) - 1) / span)) if span > 0 else 5.0
    h, w = frames[0][1].shape[:2]
    tmp = path.with_name(path.stem + ".raw.mp4")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():
        log.warning("No se pudo abrir VideoWriter para %s", path)
        return False
    for _, f in frames:
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
