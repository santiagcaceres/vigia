"""Leer el video de una cámara con el decodificador de la placa NVIDIA (NVDEC) en vez del procesador (26 nov, el dueño:
«¿no se puede usar más GPU, que está libre, en lugar de CPU?»).

Medido en la máquina de AWS (T4; 1 minuto de H.265 960x576 a 12 cuadros/s, pasando 5 por segundo a colores):
decodificando con el procesador, como OpenCV, 2,30 s de procesador; con el decodificador de la placa («hevc_cuvid»),
0,54 s (77 % menos). Para eso la máquina necesita las librerías de video de NVIDIA del mismo driver
(`libnvidia-decode-*` y `libnvidia-encode-*`: el driver «para cálculo» no las trae; `instalar.sh` las suma).

Cómo: un ffmpeg por cámara con `h264_cuvid` / `hevc_cuvid` que entrega cada cuadro por un caño en NV12 (sin pasarlo a
colores: 1,5 bytes por píxel) y `retrieve` lo pasa a BGR con OpenCV solo en los cuadros que se usan, como
`grab` / `retrieve` de OpenCV. Lo que el CameraWorker usa de cv2.VideoCapture está igual (`grab`, `retrieve`, `read`,
`get`, `isOpened`, `release`).

Qué video se pide: el mismo que ya se estaba leyendo con OpenCV (códec, tamaño, píxeles no cuadrados). La primera
conexión de cada cámara es con OpenCV, como siempre; con lo que se aprendió de ella (`hint`), la siguiente va por la
placa con el mismo tamaño (los «1080p lite» ya con píxeles cuadrados, como `square_pixels(shrink=True)`). Si algo no
anda (sin la librería, otro códec, ffmpeg que no abre), sigue con OpenCV, como siempre.

Qué cuadros se traen (26 nov, medido en la máquina de AWS con 1 minuto de H.265 960x576 a 12 cuadros/s): traer de la
placa cada cuadro entero gasta procesador aunque después no se use (0,53 s); dejarlos en la placa y bajar solo uno de
cada dos, 0,41 s aun con ffmpeg rellenando los que faltaban. Por eso: (1) `-fps_mode passthrough`: cada cuadro sale como
llega, sin que ffmpeg invente copias para llenar un ritmo fijo (salía `dup=359`: con un corte de la cámara o un video que
manda menos de lo que dice, por el caño pasaban cuadros repetidos, y OpenCV no hace eso); (2) con la cámara analizada a
5 cuadros/s o menos, solo se baja de la placa un cuadro cada `MIN_GAP_S` (45 ms) del video: un video de 25 cuadros/s
llega a 12,5 (lo mismo que una cámara de 12, que hoy analiza igual: 4,0-4,2 cuadros/s) y uno de 15 o menos, entero.
`VIGIA_PLACA_HUECO=0` trae todos, como antes.

Apagado de fábrica: `VIGIA_VIDEO_PLACA=si` lo prende (para probarlo en una parte de la máquina y comparar). Solo RTSP:
Hik-Connect (HLS/FLV) llega en ráfagas y necesita las marcas de tiempo del video, que por el caño no vienen."""

from __future__ import annotations

import collections
import ctypes
import logging
import math
import os
import select
import shutil
import subprocess
import threading
import time

import cv2
import numpy as np

log = logging.getLogger("vigia.video_placa")

DECODERS = {"H.264": "h264_cuvid", "H.265": "hevc_cuvid"}
FIRST_FRAME_TIMEOUT_S = 20.0  # hasta el primer cuadro (conectar con el grabador y arrancar el decodificador)
FRAME_TIMEOUT_S = 20.0  # sin cuadros este rato = se cortó (ffmpeg también corta a los 10 s con `-timeout`)
MAX_WIDTH = 1920  # más ancho que esto se achica en la placa (hoy no hay cámaras así: no cambia nada)
# como mucho un cuadro cada tanto (segundos del video) se trae de la placa, si la cámara se analiza a 5 cuadros/s o menos
# (ver arriba). Más de 45 ms: un video de 20 cuadros/s (50 ms) llega entero aunque los cuadros vengan un poco corridos
def _gap_from_env(default: float = 0.045) -> float:
    try:
        return max(0.0, min(float(os.getenv("VIGIA_PLACA_HUECO") or default), 0.2))
    except ValueError:
        return default


MIN_GAP_S = _gap_from_env()
THIN_MAX_PROCESS_FPS = 5.5
PIPE_SIZE = 1 << 20  # el caño de 1 MB: un cuadro de 960x576 en NV12 (830 KB) pasa de una vez
# rango de colores «completo» (0-255, el de los JPEG; muchas cámaras lo mandan así) → el «de video» (16-235) que espera
# la conversión de OpenCV. El decodificador de la placa entrega lo que viene en el video, sin convertir el rango
_FULL_Y = np.round(16 + np.arange(256) * 219 / 255).astype(np.uint8)
_FULL_C = np.round(128 + (np.arange(256) - 128) * 224 / 255).astype(np.uint8)

_lock = threading.Lock()
_available: bool | None = None
_why_not = ""
# programas ffmpeg de este proceso que están andando (para que el medidor de procesador los cuente en «video") y lo que
# gastaron los que ya terminaron (para no contarlo dos veces en «externos»)
_live: dict[int, subprocess.Popen] = {}
reaped_cpu_s = 0.0


def wanted() -> bool:
    return (os.getenv("VIGIA_VIDEO_PLACA") or "").strip().lower() in ("si", "sí", "1", "true", "yes")


def available() -> tuple[bool, str]:
    """¿Se puede decodificar en la placa en esta máquina? (ffmpeg con los decodificadores cuvid y la librería de NVIDIA
    para decodificar). Se averigua una vez por proceso. Devuelve (sí o no, por qué no)."""
    global _available, _why_not
    with _lock:
        if _available is None:
            _available, _why_not = _check()
            if wanted():
                if _available:
                    log.info("El video de las cámaras por RTSP se decodifica en la placa de video (NVDEC)")
                else:
                    log.warning("Se pidió decodificar el video en la placa y no se puede: %s. Sigue con el procesador",
                                _why_not)
        return _available, _why_not


def _check() -> tuple[bool, str]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False, "no está ffmpeg"
    try:
        ctypes.CDLL("libnvcuvid.so.1")
    except OSError:
        return False, ("falta la librería de video de NVIDIA (libnvcuvid): instalar libnvidia-decode del mismo driver "
                       "(lo hace instalar.sh)")
    try:
        out = subprocess.run([ffmpeg, "-hide_banner", "-decoders"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"ffmpeg no responde ({exc})"
    missing = [d for d in DECODERS.values() if d not in out]
    if missing:
        return False, "este ffmpeg no tiene " + ", ".join(missing)
    return True, ""


def target_size(w: int, h: int, sar: float | None, max_w: int = MAX_WIDTH) -> tuple[int, int]:
    """El tamaño en que se pide cada cuadro: el del video, con píxeles cuadrados achicando el lado que sobra (como
    `core.imagen.square_pixels(shrink=True)`) y como mucho `max_w` de ancho. Números pares (lo pide NV12)."""
    fw, fh = float(w), float(h)
    if sar and abs(sar - 1.0) > 0.05:
        if sar > 1.0:
            fh = fh / sar
        else:
            fw = fw * sar
    if fw > max_w:
        fh, fw = fh * max_w / fw, float(max_w)
    return max(2, int(round(fw / 2)) * 2), max(2, int(round(fh / 2)) * 2)


def nv12_to_bgr(buf, w: int, h: int, full_range: bool = False) -> np.ndarray:
    """Un cuadro NV12 (Y entero y después U y V intercalados a la mitad) a BGR, como lo da OpenCV."""
    yuv = np.frombuffer(buf, np.uint8, count=w * h * 3 // 2).reshape(h * 3 // 2, w)
    if full_range:
        yuv = np.concatenate([cv2.LUT(yuv[:h], _FULL_Y), cv2.LUT(yuv[h:], _FULL_C)])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)


def keep_filter(min_gap: float) -> str:
    """El filtro que deja pasar como mucho un cuadro cada `min_gap` segundos del video (no toca la imagen: sirve con los
    cuadros todavía en la placa). Si el reloj del video vuelve atrás (el grabador cambió la hora), deja pasar enseguida:
    si no, no saldría nada hasta que el reloj alcanzara al de antes."""
    if min_gap <= 0:
        return ""
    return f"select='isnan(prev_selected_t)+gte(t-prev_selected_t,{min_gap:g})+lt(t,prev_selected_t)'"


def thin_factor(fps: float, min_gap: float) -> int:
    """De cada cuántos cuadros de un video parejo de `fps` llega uno con `keep_filter(min_gap)` (para informar los
    cuadros por segundo que manda la cámara, no los que llegan)."""
    if min_gap <= 0 or not 1.0 <= fps <= 120.0:
        return 1
    return max(1, math.ceil(min_gap * fps - 1e-6))


def command(url: str, input_args: list[str], decoder: str | None, size: tuple[int, int],
            native: tuple[int, int] | None, ffmpeg: str = "ffmpeg", min_gap: float = 0.0) -> list[str]:
    """El ffmpeg que lee la cámara. Con `decoder` (h264_cuvid / hevc_cuvid) decodifica y achica en la placa; sin él
    (las pruebas, en una máquina sin placa) lo mismo con el procesador. Con `min_gap`, los cuadros quedan en la placa y
    se baja solo uno cada `min_gap` segundos del video."""
    w, h = size
    resize = native is None or tuple(native) != (w, h)
    keep = keep_filter(min_gap)
    cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error"] + list(input_args)
    if decoder:
        if keep:  # (los cuadros decodificados quedan en la memoria de la placa hasta `hwdownload`)
            cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        cmd += ["-c:v", decoder]
        if resize:
            cmd += ["-resize", f"{w}x{h}"]
        cmd += ["-i", url]
        if keep:
            cmd += ["-vf", f"{keep},hwdownload,format=nv12"]
    else:
        cmd += ["-threads", "1", "-i", url]
        filters = [f for f in (keep, f"scale={w}:{h}" if resize else "") if f]
        if filters:
            cmd += ["-vf", ",".join(filters)]
    # cada cuadro como llega: sin copias inventadas para llenar un ritmo fijo (lo que hace ffmpeg de fábrica)
    return cmd + ["-an", "-sn", "-dn", "-map", "0:v:0", "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt",
                  "nv12", "pipe:1"]


class PlacaCapture:
    """Lo que usa el CameraWorker de cv2.VideoCapture, leyendo de un ffmpeg que decodifica en la placa. `hint`: lo que se
    sabe del video de la última conexión (`video_codec`, `video_w`, `video_h`, `video_fps` y `sar`); `ref`: una imagen
    de esa conexión (BGR) para saber en qué rango de colores viene el video."""

    gpu = True

    def __init__(self, url: str, input_args: list[str], hint: dict, ref: np.ndarray | None = None,
                 decoder: str | None = "auto", first_timeout: float = FIRST_FRAME_TIMEOUT_S, min_gap: float = 0.0):
        self.why = ""
        self.proc: subprocess.Popen | None = None
        self.decoder = DECODERS.get(hint.get("video_codec") or "") if decoder == "auto" else decoder
        native = (int(hint.get("video_w") or 0), int(hint.get("video_h") or 0))
        self.source_fps = float(hint.get("video_fps") or 0.0)  # lo que manda la cámara
        # de cada cuántos llega uno (con `min_gap`) y los cuadros por segundo que llegan por el caño
        self.thin = thin_factor(self.source_fps, min_gap)
        self.fps = self.source_fps / self.thin
        self.codec = hint.get("video_codec") or ""
        if self.decoder is None and decoder == "auto":
            self.why = f"el códec {self.codec or '(desconocido)'} no se decodifica en la placa"
            return
        if not all(native):
            self.why = "no se sabe el tamaño del video"
            return
        self.w, self.h = target_size(*native, hint.get("sar"))
        self.frame_bytes = self.w * self.h * 3 // 2
        self.full_range = False
        self._buf = bytearray(self.frame_bytes)
        self._view = memoryview(self._buf)
        self._have_frame = False
        self._pending_first = False
        self._err: collections.deque[str] = collections.deque(maxlen=8)
        cmd = command(url, input_args, self.decoder, (self.w, self.h), native, shutil.which("ffmpeg") or "ffmpeg",
                      min_gap)
        try:
            self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, bufsize=0)
        except OSError as exc:
            self.why = f"no arrancó ffmpeg ({exc})"
            self.proc = None
            return
        with _lock:
            _live[self.proc.pid] = self.proc
        try:  # (Linux: el caño más grande, así un cuadro entero pasa de una vez)
            import fcntl
            fcntl.fcntl(self.proc.stdout.fileno(), getattr(fcntl, "F_SETPIPE_SZ", 1031), PIPE_SIZE)
        except (ImportError, OSError, ValueError):
            pass
        threading.Thread(target=self._drain, daemon=True, name="placa-errores").start()
        if not self._read(first_timeout):
            msg = "; ".join(self._err) or "no llegó ningún cuadro"
            self.why = f"ffmpeg con {self.decoder or 'el procesador'} no dio video: {msg}"[:300]
            self.release()
            return
        self._pending_first = True
        if ref is not None:
            self.full_range = self._looks_full_range(ref)

    # ------------------------------------------------------------------ cv2.VideoCapture
    def isOpened(self) -> bool:  # noqa: N802 - el nombre de OpenCV
        return self.proc is not None and self._have_frame

    def grab(self) -> bool:
        if self.proc is None:
            return False
        if self._pending_first:
            self._pending_first = False
            return True
        return self._read(FRAME_TIMEOUT_S)

    def retrieve(self, image=None, flag=None):  # noqa: ARG002 - la firma de OpenCV
        if not self._have_frame:
            return False, None
        return True, nv12_to_bgr(self._view, self.w, self.h, self.full_range)

    def read(self, image=None):  # noqa: ARG002
        if not self.grab():
            return False, None
        return self.retrieve()

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.w)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.h)
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        if prop == cv2.CAP_PROP_FOURCC:
            code = {"H.264": "h264", "H.265": "hevc"}.get(self.codec, "")
            return float(sum(ord(c) << 8 * i for i, c in enumerate(code))) if code else 0.0
        if prop in (getattr(cv2, "CAP_PROP_SAR_NUM", -1), getattr(cv2, "CAP_PROP_SAR_DEN", -1)):
            return 1.0  # (los píxeles ya vienen cuadrados)
        return 0.0

    def getBackendName(self) -> str:  # noqa: N802
        return "NVDEC"

    def release(self) -> None:
        global reaped_cpu_s
        proc, self.proc = self.proc, None
        self._have_frame = False
        if proc is None:
            return
        try:
            proc.kill()
        except OSError:
            pass
        try:  # (con wait4: lo que gastó en total, para que el medidor no lo cuente dos veces)
            _pid, status, usage = os.wait4(proc.pid, 0)
            proc.returncode = status
            with _lock:
                reaped_cpu_s += usage.ru_utime + usage.ru_stime
        except (ChildProcessError, OSError, AttributeError):
            try:
                proc.wait(timeout=5)
            except (subprocess.SubprocessError, OSError):
                pass
        with _lock:
            _live.pop(proc.pid, None)
        for f in (proc.stdout, proc.stderr):
            try:
                f.close()
            except (OSError, AttributeError):
                pass

    # ------------------------------------------------------------------ por dentro
    def _read(self, timeout: float) -> bool:
        """Lee un cuadro entero del caño. False si ffmpeg terminó o no mandó nada en `timeout` segundos."""
        fd = self.proc.stdout.fileno()
        got = 0
        deadline = time.monotonic() + timeout
        while got < self.frame_bytes:
            left = deadline - time.monotonic()
            if left <= 0:
                self._err.append("no llegaron cuadros")
                return False
            try:
                ready, _, _ = select.select([fd], [], [], left)
            except (OSError, ValueError):
                return False
            if not ready:
                continue
            try:
                n = os.readv(fd, [self._view[got:]])
            except OSError:
                return False
            if n <= 0:
                self._have_frame = False
                return False
            got += n
        self._have_frame = True
        return True

    def _drain(self) -> None:
        proc = self.proc
        if proc is None or proc.stderr is None:
            return
        try:
            for raw in iter(proc.stderr.readline, b""):
                line = raw.decode(errors="replace").strip()
                if line:
                    self._err.append(line[:160])
        except (OSError, ValueError):
            pass

    def _looks_full_range(self, ref: np.ndarray) -> bool:
        """¿El video viene en el rango completo? Se compara el primer cuadro, convertido de las dos maneras, contra una
        imagen de OpenCV de un momento antes (OpenCV convierte el rango bien): la que se parece más."""
        try:
            if ref.shape[:2] != (self.h, self.w):
                ref = cv2.resize(ref, (self.w, self.h), interpolation=cv2.INTER_AREA)
            small = (max(16, self.w // 8), max(16, self.h // 8))
            r = cv2.resize(ref, small, interpolation=cv2.INTER_AREA).astype(np.int16)
            a = cv2.resize(nv12_to_bgr(self._view, self.w, self.h, False), small, interpolation=cv2.INTER_AREA)
            b = cv2.resize(nv12_to_bgr(self._view, self.w, self.h, True), small, interpolation=cv2.INTER_AREA)
            da, db = float(np.abs(a.astype(np.int16) - r).mean()), float(np.abs(b.astype(np.int16) - r).mean())
            return db + 0.5 < da
        except (cv2.error, ValueError):
            return False


def live_cpu() -> dict[int, float]:
    """Procesador que lleva gastado cada ffmpeg que sigue andando: pid → segundos (Linux; si no, nada)."""
    with _lock:
        pids = list(_live)
    out: dict[int, float] = {}
    try:
        hz = float(os.sysconf("SC_CLK_TCK"))
    except (OSError, ValueError, AttributeError):
        return out
    for pid in pids:
        try:
            with open(f"/proc/{pid}/stat", "rb") as f:
                data = f.read()
            rest = data[data.rindex(b")") + 2:].split()
            out[pid] = (int(rest[11]) + int(rest[12])) / hz
        except (OSError, ValueError, IndexError):
            continue
    return out
