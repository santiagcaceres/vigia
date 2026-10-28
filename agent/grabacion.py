"""«Bajar grabación…» (Clientes): tramos de la grabación de una cámara (el disco del grabador) a archivos MP4 en esta
PC, para probarlos con «Probar clip» o en «Ajustar…», o para guardarlos.

- Sin ffmpeg (en las PCs con Windows no suele estar): OpenCV abre el link de reproducción y escribe el MP4. Por
  Hik-Connect el link lo da el servidor (HLS, `type` 3 de la API: la grabación del grabador); por IP y puertos es el
  RTSP de reproducción del grabador (`playback_url` de cada marca).
- Se baja de a tramos (30 min por defecto): si uno falla, los demás siguen, y cada archivo queda manejable.
- Se guarda con la resolución y los cuadros por segundo que manda el grabador (los videos «lite» de 960x1080 quedan
  así; «Ajustar…» los estira solo).
"""

from __future__ import annotations

import logging
import math
import re
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import cv2

log = logging.getLogger("vigia.grabacion")

CHUNK_MINUTES = 30
MAX_HOURS = 12  # tope de un pedido (horas de grabación por cámara)
MAX_FAILS = 25  # lecturas fallidas seguidas antes de dar el tramo por terminado (el HLS llega en ráfagas)
MAX_FPS = 15  # se guardan como mucho 15 cuadros por segundo (el análisis usa 5): archivos más chicos


@dataclass
class Piece:
    camera_id: int
    camera: str
    client: str
    start: datetime
    end: datetime
    path: Path
    state: str = "pendiente"  # pendiente | bajando | listo | incompleto | error | detenido
    progress: float = 0.0
    seconds: float = 0.0
    detail: str = ""

    @property
    def label(self) -> str:
        return f"{self.camera} {self.start:%H:%M}-{self.end:%H:%M}"


def pieces(start: datetime, end: datetime, minutes: int = CHUNK_MINUTES) -> list[tuple[datetime, datetime]]:
    """[start, end) en tramos de `minutes` (el último puede ser más corto)."""
    if end <= start:
        raise ValueError("«Hasta» tiene que ser después de «Desde»")
    if end - start > timedelta(hours=MAX_HOURS):
        raise ValueError(f"Como mucho {MAX_HOURS} horas por pedido")
    step = timedelta(minutes=max(1, int(minutes)))
    out, t = [], start
    while t < end:
        out.append((t, min(t + step, end)))
        t += step
    return out


def file_name(client: str, camera: str, start: datetime, end: datetime) -> str:
    """«Minas - Pasillo bebidas - 2026-10-28 08.00-08.30.mp4» (sin caracteres que Windows no acepta)."""
    def clean(text: str) -> str:
        return re.sub(r"[\\/:*?\"<>|]+", "-", text).strip(" .") or "camara"
    return f"{clean(client)} - {clean(camera)} - {start:%Y-%m-%d %H.%M}-{end:%H.%M}.mp4"


def plan(cameras: list[dict], start: datetime, end: datetime, folder: Path,
         minutes: int = CHUNK_MINUTES) -> list[Piece]:
    """Los tramos a bajar: por cada cámara ({id, name, client_name}), cada tramo de [start, end)."""
    out = []
    for c in cameras:
        for s, e in pieces(start, end, minutes):
            out.append(Piece(c["id"], c["name"], c.get("client_name") or "", s, e,
                             Path(folder) / file_name(c.get("client_name") or "", c["name"], s, e)))
    return out


def playback_urls(cam, start: datetime, end: datetime) -> list[str]:
    """Links para probar en orden. Por Hik-Connect, primero la calidad principal y, si no abre (¿H.265?), la
    secundaria."""
    urls = [cam.playback_url(start, end)]
    if getattr(cam, "brand", "") == "hikconnect":
        try:
            urls.append(cam._url("playback", "sub", cam._utc_or_local(start), cam._utc_or_local(end)))
        except Exception as exc:  # noqa: BLE001 - la principal ya está; la otra es por si acaso
            log.info("Sin link de calidad secundaria: %s", exc)
    return urls


def open_writer(path: Path, fps: float, size: tuple[int, int]) -> tuple[cv2.VideoWriter | None, str]:
    """El MP4 más chico que se pueda escribir en esta PC: H.264 con Media Foundation en Windows (viene con el
    sistema), H.264 con el FFmpeg de OpenCV si lo trae, y si no MPEG-4 (anda siempre; unas 3 veces más grande)."""
    tries = []
    if sys.platform == "win32":
        tries.append((cv2.CAP_MSMF, "H264", "H.264"))
    tries += [(cv2.CAP_FFMPEG, "avc1", "H.264"), (cv2.CAP_FFMPEG, "mp4v", "MPEG-4")]
    for api, fourcc, name in tries:
        try:
            w = cv2.VideoWriter(str(path), api, cv2.VideoWriter_fourcc(*fourcc), fps, size)
        except cv2.error:
            continue
        if w.isOpened():
            return w, name
        w.release()
    return None, ""


def download(cam, piece: Piece, stop: threading.Event | None = None,
             on_progress: Callable[[Piece], None] | None = None) -> Piece:
    """Baja un tramo a `piece.path`. Deja en `piece` el estado, los segundos bajados y el motivo si falló."""
    stop = stop or threading.Event()
    duration = (piece.end - piece.start).total_seconds()
    piece.state, piece.progress, piece.detail = "bajando", 0.0, ""
    if on_progress:
        on_progress(piece)
    try:
        urls = playback_urls(cam, piece.start, piece.end)
    except Exception as exc:  # noqa: BLE001 - p. ej. Hik-Connect no da el link
        piece.state, piece.detail = "error", str(exc)
        return piece
    cap, used = None, None
    for url in urls:
        c = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if c.isOpened():
            cap, used = c, url
            break
        c.release()
    if cap is None:
        why = None
        try:
            why = cam.open_failure(urls[0])
        except Exception:  # noqa: BLE001
            pass
        piece.state = "error"
        piece.detail = why or "No se pudo abrir la grabación (¿hay grabación en ese horario? ¿el grabador está en línea?)"
        return piece
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if not 1 <= fps <= 60:
        fps = 15.0
    step = max(1, math.ceil(fps / MAX_FPS - 1e-6))  # de 25 cuadros por segundo se guarda 1 de cada 2
    out_fps = fps / step
    piece.path.parent.mkdir(parents=True, exist_ok=True)
    tmp = piece.path.with_name(piece.path.stem + ".bajando.mp4")
    writer, codec, frames, read, fails, last_report = None, "", 0, 0, 0, -1.0
    try:
        while not stop.is_set():
            if not cap.grab():
                fails += 1
                if fails >= MAX_FAILS:
                    break
                continue
            read += 1
            if (read - 1) % step:
                fails = 0
                continue
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                fails += 1
                if fails >= MAX_FAILS:
                    break
                continue
            fails = 0
            if writer is None:
                h, w = frame.shape[:2]
                writer, codec = open_writer(tmp, out_fps, (w, h))
                if writer is None:
                    piece.state, piece.detail = "error", f"No se pudo escribir {tmp.name}"
                    return piece
            writer.write(frame)
            frames += 1
            piece.seconds = frames / out_fps
            piece.progress = min(piece.seconds / duration, 1.0) if duration > 0 else 1.0
            if on_progress and piece.progress - last_report >= 0.01:
                last_report = piece.progress
                on_progress(piece)
            if piece.seconds >= duration + 1.0:  # algunos links siguen de largo después del fin pedido
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if stop.is_set():
        tmp.unlink(missing_ok=True)
        piece.state, piece.detail = "detenido", ""
        return piece
    if frames == 0:
        tmp.unlink(missing_ok=True)
        piece.state, piece.detail = "error", "La grabación abrió pero no llegó ningún cuadro (¿no hay grabación a esa hora?)"
        return piece
    tmp.replace(piece.path)
    if piece.seconds < duration * 0.9:
        piece.state = "incompleto"
        piece.detail = (f"llegaron {fmt_minutes(piece.seconds)} de {fmt_minutes(duration)} (¿sin grabación en parte "
                        "del tramo?)")
    else:
        piece.state, piece.detail = "listo", ""
    log.info("Grabación %s: %s (%.0f s, %.0f cuadros/s, %s, %s)", piece.label, piece.state, piece.seconds, out_fps,
             codec, cam.redacted(used) if hasattr(cam, "redacted") else "")
    return piece


def fmt_minutes(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m} min {s:02d} s" if m else f"{s} s"
