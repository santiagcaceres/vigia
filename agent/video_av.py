"""Leer el video de las cámaras con la placa DENTRO del proceso del análisis (27 nov; `VIGIA_VIDEO_PLACA=proceso`).

Por qué: con `VIGIA_VIDEO_PLACA=si` (agent/video_placa.py) cada cámara que va por la placa es un ffmpeg aparte, con su
propia conexión a la placa (un «contexto» de CUDA). Medido en la máquina de AWS: ~0,03 procesadores por cámara sea el
video chico o grande, casi todo del controlador de NVIDIA atendiendo ese contexto (`cuda-EvtHandlr` se despierta ~670
veces por segundo por ffmpeg) y de pasar cada cuadro por el caño. Es lo mismo que cuesta decodificar un video liviano
con el procesador, así que no rendía y solo se usa para los pesados.

Acá el video se lee con PyAV (las mismas librerías de FFmpeg, como módulo de Python) en el hilo de cada cámara, y la
placa decodifica con NVDEC usando el contexto PRINCIPAL de CUDA del proceso (`primary_ctx`): el mismo que ya usa
onnxruntime para el detector. Un solo «enchufe» a la placa para todas las cámaras del proceso, sin caño y sin otro
programa. Los cuadros quedan en la placa (`is_hw_owned`) y se baja a la memoria solo el que se va a usar
(`retrieve`), que se pasa a colores con OpenCV como los de `video_placa`.

Tiene la misma cara que `cv2.VideoCapture` (isOpened, grab, retrieve, read, get, release), así el `CameraWorker` no
cambia: entrega los cuadros del tamaño que manda la cámara (los «1080p lite» los cuadra el worker, como con OpenCV).
Si la placa no puede con ese video, PyAV decodifica con el procesador (un hilo, como hoy) y `gpu` queda en False.
Solo RTSP (Hik-Connect llega en ráfagas: sigue con OpenCV y su reloj del video).

Sin PyAV instalado, o sin la placa, `available()` dice por qué y todo sigue con OpenCV."""

from __future__ import annotations

import logging
import os
import threading
import time
from fractions import Fraction

import cv2
import numpy as np

from core.clips import redact

from .video_placa import nv12_to_bgr

log = logging.getLogger("vigia.video_av")

OPEN_TIMEOUT_S = 15.0  # conectar con el grabador y leer de qué es el video
READ_TIMEOUT_S = 10.0  # sin datos este rato = se cortó (como el `timeout` de OpenCV)
FOURCC = {"h264": "h264", "hevc": "hevc", "mjpeg": "mjpg", "mpeg4": "mp4v"}
USE_GPU = True  # (las pruebas sin placa lo apagan para probar el resto del camino con el procesador)

_lock = threading.Lock()
_available: bool | None = None
_why_not = ""


def wanted() -> bool:
    return (os.getenv("VIGIA_VIDEO_PLACA") or "").strip().lower() in ("proceso", "pyav", "av")


def available() -> tuple[bool, str]:
    """¿Se puede leer el video con PyAV y la placa en esta máquina? Se averigua una vez por proceso."""
    global _available, _why_not
    with _lock:
        if _available is None:
            _available, _why_not = _check()
            if wanted():
                if _available:
                    log.info("El video de las cámaras por RTSP se lee con la placa dentro del proceso (PyAV, NVDEC, "
                             "contexto de CUDA compartido)")
                else:
                    log.warning("Se pidió leer el video con la placa dentro del proceso y no se puede: %s. Sigue con "
                                "el procesador", _why_not)
        return _available, _why_not


def _check() -> tuple[bool, str]:
    try:
        import av
        from av.codec.hwaccel import hwdevices_available
    except ImportError:
        return False, "falta PyAV (pip install av)"
    try:
        if "cuda" not in hwdevices_available():
            return False, f"este PyAV ({av.__version__}) no trae CUDA"
    except Exception as exc:  # noqa: BLE001
        return False, f"PyAV no dice qué placas puede usar ({exc})"
    try:  # abrir la placa una vez (con el contexto principal: el mismo que usa el detector)
        from av.codec.hwaccel import HWDevice
        HWDevice("cuda", options={"primary_ctx": "1"})
    except ImportError:
        pass  # (un PyAV sin HWDevice: se sabrá al abrir la primera cámara)
    except Exception as exc:  # noqa: BLE001
        return False, f"no se pudo abrir la placa ({type(exc).__name__}: {exc})"
    return True, ""


def input_options(input_args: list[str]) -> dict[str, str]:
    """Las opciones de entrada de ffmpeg («-rtsp_transport tcp -timeout 10000000») como las pide PyAV."""
    out: dict[str, str] = {}
    i = 0
    while i < len(input_args):
        key = input_args[i]
        if key.startswith("-") and i + 1 < len(input_args):
            out[key[1:]] = str(input_args[i + 1])
            i += 2
        else:
            i += 1
    out.setdefault("rtsp_transport", "tcp")
    out.setdefault("timeout", "10000000")  # µs: sin datos 10 s, se corta
    return out


class AvCapture:
    """El video de una cámara leído con PyAV, con la placa si se puede. Ver el comentario del módulo."""

    thin = 1

    def __init__(self, url: str, input_args: list[str] | None = None, use_gpu: bool | None = None,
                 threads: int | None = 1):
        self.url = url
        self.gpu = False
        self.why = ""
        self.source_fps = 0.0
        self._cont = None
        self._stream = None
        self._frames = None
        self._frame = None
        self._pending = None  # el primer cuadro (se lee al abrir, para saber si quedó en la placa)
        self._t = 0.0
        self._open(input_options(input_args or []), USE_GPU if use_gpu is None else use_gpu, threads)

    def _open(self, options: dict[str, str], use_gpu: bool, threads: int | None) -> None:
        import av
        accel = None
        if use_gpu:
            from av.codec.hwaccel import HWAccel
            accel = HWAccel("cuda", allow_software_fallback=True, options={"primary_ctx": "1"}, is_hw_owned=True)
        try:
            self._cont = av.open(self.url, options=options, timeout=(OPEN_TIMEOUT_S, READ_TIMEOUT_S), hwaccel=accel)
            self._stream = self._cont.streams.video[0]
            ctx = self._stream.codec_context
            # (si la placa no puede y decodifica el procesador: los hilos de siempre, uno salvo un video muy grande;
            # 0 = los que elija FFmpeg)
            ctx.thread_count = int(threads or 0)
            self._frames = self._cont.decode(self._stream)
            self._pending = next(self._frames)
        except Exception as exc:  # noqa: BLE001 - no abrió: el worker lo trata como OpenCV que no abre
            self.why = redact(f"{type(exc).__name__}: {exc}")[:300]
            self.release()
            return
        self.gpu = getattr(self._pending.format, "name", "") == "cuda"
        rate = self._stream.average_rate or self._stream.guessed_rate or self._stream.base_rate
        try:
            self.source_fps = float(rate) if rate else 0.0
        except (TypeError, ValueError, ZeroDivisionError):
            self.source_fps = 0.0

    # -- la cara de cv2.VideoCapture
    def isOpened(self) -> bool:
        return self._cont is not None and (self._pending is not None or self._frame is not None)

    def grab(self) -> bool:
        if self._cont is None:
            return False
        if self._pending is not None:
            self._frame, self._pending = self._pending, None
        else:
            try:
                self._frame = next(self._frames)
            except StopIteration:
                self._frame = None
                return False
            except Exception as exc:  # noqa: BLE001 - se cortó (red, grabador, placa): el worker reconecta
                self.why = redact(f"{type(exc).__name__}: {exc}")[:300]
                self._frame = None
                return False
        t = self._frame.time
        if t is not None:
            self._t = float(t)
        return True

    def retrieve(self, image=None, flag=None):
        f = self._frame
        if f is None:
            return False, None
        try:
            return True, to_bgr(f)
        except Exception as exc:  # noqa: BLE001
            self.why = f"no se pudo pasar el cuadro a colores: {exc}"[:300]
            return False, None

    def read(self, image=None):
        if not self.grab():
            return False, None
        return self.retrieve()

    def get(self, prop: int) -> float:
        st = self._stream
        if st is None:
            return 0.0
        ctx = st.codec_context
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(ctx.width or 0)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(ctx.height or 0)
        if prop == cv2.CAP_PROP_FPS:
            return self.source_fps
        if prop == cv2.CAP_PROP_POS_MSEC:
            return self._t * 1000.0
        if prop == cv2.CAP_PROP_FOURCC:
            code = FOURCC.get(ctx.name or "", "")
            return float(cv2.VideoWriter_fourcc(*code)) if len(code) == 4 else 0.0
        sar = ctx.sample_aspect_ratio
        if prop in (cv2.CAP_PROP_SAR_NUM, cv2.CAP_PROP_SAR_DEN):
            if not sar:
                return 0.0
            sar = Fraction(sar)
            return float(sar.numerator if prop == cv2.CAP_PROP_SAR_NUM else sar.denominator)
        return 0.0

    def getBackendName(self) -> str:
        return "PyAV (placa)" if self.gpu else "PyAV"

    def release(self) -> None:
        cont, self._cont = self._cont, None
        self._frames = self._pending = self._frame = None
        if cont is not None:
            try:
                cont.close()
            except Exception:  # noqa: BLE001, S110 - cerrar algo que ya se cortó
                pass


def to_bgr(frame) -> np.ndarray:
    """Un cuadro de PyAV a BGR. Si está en la placa, baja ese cuadro (NV12) y lo pasa a colores con OpenCV, como los de
    `video_placa`; el rango de colores «completo» (el de las cámaras que mandan JPEG) se ajusta igual que allá."""
    full = frame.color_range == 2  # AVCOL_RANGE_JPEG
    if getattr(frame.format, "name", "") == "cuda":
        frame = frame.reformat(format="nv12")
    name = frame.format.name
    if name == "nv12":
        return nv12_to_bgr(np.ascontiguousarray(frame.to_ndarray()), frame.width, frame.height, full)
    if name == "yuv420p" and not full and frame.width % 2 == 0 and frame.height % 2 == 0:
        return cv2.cvtColor(np.ascontiguousarray(frame.to_ndarray()), cv2.COLOR_YUV2BGR_I420)
    return frame.to_ndarray(format="bgr24")  # (lo demás, con swscale: como lo hace OpenCV)


def warm() -> None:
    """Importar PyAV una vez fuera de los hilos de las cámaras (el primer import tarda y carga las librerías)."""
    t0 = time.monotonic()
    ok, _why = available()
    if ok:
        log.debug("PyAV listo en %.1f s", time.monotonic() - t0)
