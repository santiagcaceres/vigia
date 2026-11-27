from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from time import thread_time  # (aparte de `time`: las pruebas cambian ese reloj por uno de mentira)

import cv2

from core.analyzer import Analyzer, ClipReady, Finding
from core.clips import video_offset, write_mp4
from core.detectors import DetectionConfig, Detector, tuned
from core.gestos import GESTURE_NAMES
from core.imagen import pixel_aspect, square_pixels
from core.metrics import MetricsCollector
from core.motion import MotionGate
from core.pose import build_pose
from core.rules import RuleEngine, parse_config
from core.scorer import Scorer

from .cameras import build_camera, download_nvr_clip, retry_wait
from .live import LiveStreamer, hud
from .outbox import Outbox, ServerClient

log = logging.getLogger("vigia.worker")
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;10000000|timeout;10000000")

PREVIEW_EVERY_SECONDS = 300
METRICS_EVERY_SECONDS = 300  # cada cuánto se mandan las métricas (gente, tiempos, vehículos) al servidor
# el NVR tarda unos segundos en tener disponible lo recién grabado
NVR_WRITE_DELAY_SECONDS = float(os.getenv("VIGIA_NVR_DELAY", "15"))


DISPLAY_INTERVAL = 0.1  # la ventana de la PC central no necesita más de 10 cuadros por segundo
# Ponerse al día (18 nov, la farmacia: los clips de la máquina salían en cámara lenta, «como fotos»): si analizar un
# cuadro tarda más que lo que tarda la cámara en mandar el siguiente, los cuadros se apilan; antes se analizaban todos
# en orden, cada vez más atrasados y con la hora de la PC (un clip de 3 s quedaba de 17 s y los gestos se medían
# lentos). Ahora, después de un cuadro lento, se saltean sin analizar los que ya estaban esperando (salen al instante:
# solo se decodifican) hasta llegar al vivo. Solo con RTSP: Hik-Connect y los HLS llegan en ráfagas y van con el reloj
# del video (`MediaClock`).
# «No da abasto» (24 nov): se medía por los cuadros salteados al ponerse al día, pero eso pasa cada vez que un análisis
# tarda más que UN cuadro del video (40 ms a 25 cuadros/s: normal con gente a la vista) y esos cuadros igual no se iban a
# analizar: la máquina de AWS avisaba 200 veces en 15 min con un tercio del procesador libre. Ahora es la parte de los
# cuadros que tocaba analizar y no se analizó porque los análisis tardaron más que el tiempo entre uno y otro.
SLOW_LOG_FRACTION = 0.10  # el registro lo avisa si en el minuto se analizó un 10 % menos de lo que tocaba
WAITING_GRAB_MAX_S = 0.02  # un cuadro que tardó menos que esto en salir ya estaba esperando (es viejo)
# con la placa pedida, la primera conexión (con OpenCV) cuenta los cuadros que manda la cámara este rato antes de pasar a
# la placa: lo que dice el video al abrir a veces es el doble de lo que manda (30 con 15), y con eso la placa no sabría
# cuántos baja ni cuántos informar
GPU_LEARN_S = 6.0
GPU_LEARN_SKIP_S = 1.5  # (lo primero no se cuenta: al conectar, algunos grabadores mandan de golpe lo que tenían)
# Video con la placa dentro del proceso (agent/video_av.py): si una cámara no abrió así por algo que no es la placa (el
# grabador no contestó, mandó algo raro al conectar), esa vez abre con OpenCV y a los 2 minutos se vuelve a probar con
# la placa (y si vuelve a fallar, el doble, hasta una hora). 27 nov, en la máquina de AWS: un corte del grabador de
# Cristino dejaba sus cámaras con el procesador hasta el próximo corte. Si el problema es de la placa, una hora.
AV_BACK_S = 120.0
AV_BACK_MAX_S = 3600.0
_AV_HW_ERROR = re.compile(r"cuda|hwaccel|hw_device|nvdec|cuvid|not supported|no capable|libnvcuvid", re.IGNORECASE)
CATCHUP_MAX_S = 5.0  # (por las dudas, como mucho 5 s seguidos salteando: después se analiza igual)
# filtro de movimiento (core/motion.py): qué parte de la imagen chica (96x54) tiene que cambiar para mirar el cuadro.
# Con «Personas lejos» se baja: alguien al fondo de un depósito (menos del ~8 % del alto) mueve menos puntos que eso y,
# con la escena quieta, se analizaría un cuadro cada 2 s hasta que el detector lo viera
MOTION_FRACTION = 0.0015
FAR_MOTION_FRACTION = 0.0005

_pose_lock = threading.Lock()
_pose_state: dict = {}  # "est": el estimador de postura del proceso (o None), una vez que se intentó armarlo


def shared_pose():
    """La postura de las personas (core/pose.py) para los gestos de ocultamiento y la IA de movimientos de hurto: un
    solo estimador para todas las cámaras del proceso (y las alarmas), armado la primera vez que una cámara lo pide.
    Sin el modelo (models/postura.onnx) o si no arranca, None: esas cámaras siguen sin gestos ni IA de hurtos, con
    todo lo demás igual, y queda una sola línea en el registro (no una por cámara ni por cada vez que se pide)."""
    with _pose_lock:
        if "est" not in _pose_state:
            try:
                _pose_state["est"] = build_pose()
            except Exception as exc:  # noqa: BLE001 - un modelo roto no puede dejar las cámaras sin analizar
                log.warning("No se pudo cargar la postura de las personas (%s): los gestos quedan apagados", exc)
                _pose_state["est"] = None
        return _pose_state["est"]


def without_detection(analysis: dict | None) -> dict:
    """El análisis sin la detección de personas (`analysis["detection"]`): para saber si lo único que cambió es la
    sensibilidad o «Personas lejos», que se aplican sin rehacer las reglas ni el conteo."""
    return without_keys(analysis, "detection")


def wants_gestures(analysis: dict | None, metrics_only: bool = False) -> bool:
    """La cámara tiene algún gesto de ocultamiento activado (`analysis["gestures"]`, aparte de `rules`). Nunca en
    una cámara de solo métricas: ahí no hay alertas ni clips."""
    g = (analysis or {}).get("gestures")
    return not metrics_only and isinstance(g, dict) and bool(g.get("enabled"))


_bad_specs: set[str] = set()  # modelos de la IA de hurtos que esta versión no puede usar (ya dichos en el registro)


def theft_spec(analysis: dict | None, metrics_only: bool = False) -> dict | None:
    """El detector de movimientos de hurto entrenado en la app (`analysis["ia_hurto"]`, core/hurto_ia.py), si esta
    versión lo puede usar; si no, None y la cámara sigue con todo lo demás igual: sin modelo, en una cámara de solo
    métricas (ahí no hay alertas ni clips), o con un modelo de otra versión de los rasgos o roto (de un servidor más
    nuevo: queda una línea en el registro, una sola vez por modelo)."""
    a = analysis or {}
    spec = a.get("ia_hurto")
    if metrics_only or a.get("metrics_only") or not isinstance(spec, dict):
        return None
    try:
        from core.hurto_ia import TheftModel

        if TheftModel(spec).ok:
            return spec
        why = "es de otra versión de los rasgos o está mal armado"
    except Exception as exc:  # noqa: BLE001 - un modelo raro no puede dejar la cámara sin analizar
        why = f"no se pudo leer ({type(exc).__name__}: {exc})"
    key = json.dumps(spec, sort_keys=True, default=str)[:4000]
    if key not in _bad_specs:
        _bad_specs.add(key)
        log.warning("El modelo de la IA de hurtos %s: esta versión no lo usa (actualizá la app o la máquina); el "
                    "resto del análisis sigue igual", why)
    return None


def wants_pose(analysis: dict | None, metrics_only: bool = False) -> bool:
    """Hay que mirar la postura de las personas: por los gestos de ocultamiento o por la IA de movimientos de hurto
    (que aprende con los esqueletos)."""
    return wants_gestures(analysis, metrics_only) or theft_spec(analysis, metrics_only) is not None


def without_keys(analysis: dict | None, *keys: str) -> dict:
    """El análisis sin esas claves: para saber si lo único que cambió es algo que se aplica sin rehacer las reglas ni
    el conteo (la detección de personas, el modelo de la IA de hurtos)."""
    return {k: v for k, v in (analysis or {}).items() if k not in keys}


class MetricsQueue:
    """Junta las métricas de todas las cámaras de la PC y las manda en un solo pedido (lo llama el agente
    cada 5 min). Antes cada cámara hacía su pedido: con 600 cámaras eran 5 millones de llamadas por mes."""

    MAX_PER_REQUEST = 500  # lo que acepta el servidor por pedido

    def __init__(self, client: ServerClient | None):
        self.client = client
        self._lock = threading.Lock()
        self._items: dict[tuple[int, str], dict] = {}

    def add(self, buckets: list[dict], older: bool = False) -> None:
        with self._lock:
            for b in buckets:
                key = (b["camera_id"], b["bucket_start"])
                if older and key in self._items:
                    continue  # un reintento no pisa un cubo más nuevo de la misma hora
                self._items[key] = b

    def pending(self) -> int:
        return len(self._items)

    def flush(self) -> None:
        with self._lock:
            items, self._items = list(self._items.values()), {}
        if not items or self.client is None:
            return
        for i in range(0, len(items), self.MAX_PER_REQUEST):
            batch = items[i:i + self.MAX_PER_REQUEST]
            try:
                self.client.post_metrics(batch)
            except Exception as exc:  # noqa: BLE001 - se reintenta en el próximo envío
                log.warning("No se pudieron enviar las métricas (%s); se reintenta", exc)
                self.add(batch, older=True)


def bursty(brand: str | None, url: str | None) -> bool:
    """El video llega en ráfagas: Hik-Connect, o un link HLS (…m3u8, pegado en una cámara Hikvision o Dahua). Ahí los
    cuadros se eligen con el reloj del video: con el de la PC quedaría uno por ráfaga (0,6 cuadros por segundo). Con
    RTSP (o un MJPEG por http) el video llega parejo y se sigue usando el reloj de la PC: un MJPEG no trae marcas de
    tiempo y, si declara mal sus cuadros por segundo, el reloj del video correría a otra velocidad."""
    u = str(url or "").strip().lower()
    return brand == "hikconnect" or (u.startswith(("http://", "https://")) and ".m3u8" in u)


CODEC_NAMES = {"h264": "H.264", "avc1": "H.264", "x264": "H.264", "hevc": "H.265", "hev1": "H.265", "hvc1": "H.265",
               "h265": "H.265", "mjpg": "MJPEG", "jpeg": "MJPEG"}


def video_info(cap) -> dict:
    """Qué video manda la cámara: códec, tamaño y cuadros por segundo (24 nov: decodificar se lleva un tercio del
    procesador de la máquina y lo que pesa es el tamaño por los cuadros por segundo; con esto la app marca cuáles
    conviene pasar a la segunda calidad chica). Solo datos simples, para el estado."""
    out: dict = {}
    try:
        four = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    except Exception:  # noqa: BLE001 - es solo para informar
        return out
    code = "".join(chr((four >> 8 * i) & 0xFF) for i in range(4)).strip("\x00 ").lower()
    name = CODEC_NAMES.get(code) or (code.upper() if code.isalnum() else "")
    if name:
        out["video_codec"] = name
    if 0 < w <= 20000 and 0 < h <= 20000:
        out["video_w"], out["video_h"] = w, h
    if 1.0 <= fps <= 120.0:
        out["video_fps"] = round(fps, 1)
    return out


# Decodificar con un solo hilo por cámara (25 nov). FFmpeg usa por defecto un hilo por procesador para cada video; con
# 20 cámaras son 80 hilos peleando por 4 procesadores, y repartir un video chico entre varios hilos cuesta más que
# hacerlo en uno. Medido: leyendo a toda velocidad un hilo gasta entre 15 y 35 % menos procesador (segunda calidad
# H.265 a 8 cuadros/s: 8,6 ms por segundo de video contra 13,2); en vivo, con 10 cámaras, bastante menos (de 0,70 a
# 0,66 procesadores), y no demora los cuadros (con varios hilos cada cuadro sale un par de cuadros más tarde). El video
# sale idéntico. Con un hilo se decodifica en el hilo de la cámara. `VIGIA_HILOS_VIDEO`: otro número, o «auto» para
# dejarlo como antes.
ONE_THREAD_MAX_PX_S = 1920 * 1080 * 30  # hasta acá un hilo sobra (1080p H.265 a 25: 18 % de un procesador)
DECODE_BUSY = 0.6  # si leer el video se lleva más de esto de un procesador durante un minuto, un hilo no alcanza


def lean_opencv() -> None:
    """Máquina de análisis: OpenCV sin su grupo de hilos propio (25 nov). Achicar una imagen o pasarla a gris la
    reparte entre todos los procesadores y deja esos hilos dando vueltas un rato por si llega más trabajo; con muchas
    cámaras, cada una en su hilo, eso no gana tiempo y gasta (medido con 10 cámaras: 0,11 procesadores en esos hilos;
    haciéndolo cada cámara en el suyo, 0,04). El resultado es el mismo. `VIGIA_HILOS_OPENCV`: otro número (0 = los
    que elija OpenCV, como antes). La app de escritorio no lo usa: ahí hay pocas cámaras y procesador de sobra."""
    v = (os.getenv("VIGIA_HILOS_OPENCV") or "1").strip()
    try:
        n = int(v)
    except ValueError:
        n = 1
    if n > 0:
        cv2.setNumThreads(n)


def video_threads() -> int | None:
    """Hilos para decodificar cada cámara según `VIGIA_HILOS_VIDEO` (None = los que elija FFmpeg)."""
    v = (os.getenv("VIGIA_HILOS_VIDEO") or "1").strip().lower()
    if v in ("auto", "0"):
        return None
    try:
        return max(1, int(v))
    except ValueError:
        return 1


_threads_ok: bool | None = None  # ¿abrir con la opción de hilos anda en este proceso? (None = todavía no se sabe)
_plain_tries = 0
PLAIN_TRIES_MAX = 3


def open_video(url: str, threads: int | None = None):
    """Abre el video con FFmpeg. Con `threads`, decodificando con esa cantidad de hilos. Por si un OpenCV no abriera
    con esa opción: mientras ninguna cámara del proceso haya abierto así, una que no abre se prueba también sin la
    opción (hasta 3 veces en total: no es cuestión de golpear dos veces a un grabador que no contesta); si sin la
    opción abre, el problema es la opción y el proceso no la usa más."""
    global _threads_ok, _plain_tries
    prop = getattr(cv2, "CAP_PROP_N_THREADS", None)
    if not threads or prop is None or _threads_ok is False:
        return cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    try:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG, [prop, int(threads)])
    except (TypeError, cv2.error):  # un OpenCV viejo que no recibe opciones al abrir: como antes
        _threads_ok = False
        return cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if cap.isOpened():
        _threads_ok = True
    elif _threads_ok is None and _plain_tries < PLAIN_TRIES_MAX:
        _plain_tries += 1
        plain = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if plain.isOpened():
            _threads_ok = False
            log.warning("Este OpenCV no abre el video eligiendo los hilos: se decodifica como antes")
            cap.release()
            return plain
        plain.release()
    return cap


def heavy_video(info: dict) -> bool:
    """Un video tan grande que un solo hilo podría no alcanzar para decodificarlo (más que 1080p a 30 cuadros/s)."""
    w, h = info.get("video_w") or 0, info.get("video_h") or 0
    return w * h * (info.get("video_fps") or 25.0) > ONE_THREAD_MAX_PX_S


class MediaClock:
    """Segundos de video transcurridos según las marcas de tiempo del stream. Si el stream no las trae
    (o vuelven atrás), cuenta cuadros a los fps que declara. Nunca retrocede."""

    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        self.step = 1.0 / fps if 1.0 <= fps <= 60.0 else 1.0 / 12
        self.first: float | None = None
        self.t = 0.0

    def tick(self) -> float:
        pos = (self.cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
        if pos > 0:
            if self.first is None:
                self.first = pos - self.t
            cand = pos - self.first
            if self.t <= cand < self.t + 10:  # avanza de forma razonable: usar la marca del video
                self.t = cand
                return self.t
            self.first = pos - self.t - self.step  # salto o vuelta atrás (otro segmento): reanclar
        self.t += self.step
        return self.t


class CameraWorker:
    """Un hilo por cámara: lee el stream, analiza, guarda eventos y clips en la cola."""

    def __init__(self, cam_cfg: dict, detector: Detector, scorer: Scorer | None, outbox: Outbox,
                 client: ServerClient | None, clip_pool: ThreadPoolExecutor, settings: dict,
                 metrics_queue: MetricsQueue | None = None):
        self.id: int = cam_cfg["id"]
        # en modo central conviven cámaras de varios clientes: el log las distingue por cliente
        self.name: str = self.display_name(cam_cfg)
        self.fingerprint = self.fingerprint_of(cam_cfg)
        self.hik = build_camera(cam_cfg["connection"])
        # solo métricas (contar gente, tiempos): menos cuadros y sin clips; para hurto, mínimo 5 fps
        self.metrics_only = bool(cam_cfg.get("metrics_only"))
        self.process_fps = max(1.0 if self.metrics_only else 5.0, float(cam_cfg.get("process_fps") or 5))
        self.clip_pre = float(settings.get("clip_pre_seconds", 6))
        self.clip_post = float(settings.get("clip_post_seconds", 6))
        self.clip_max = float(settings.get("clip_max_seconds", 180))
        # la postura (gestos, IA de hurtos) se engancha en el hilo de la cámara (`_sync_pose`): cargar el modelo la
        # primera vez tarda un momento y no tiene que trabar a quien arma la cámara (la ventana)
        self._analysis: dict = cam_cfg.get("analysis") or {}
        self._pose_uses = (False, False)  # (gestos, IA de hurtos) que miran la postura ahora (para el registro)
        self._sar_logged: float | None = None
        # sensibilidad y «Personas lejos» de esta cámara (`analysis["detection"]`, core/deteccion.py): el detector es
        # el compartido de la PC; el envoltorio le pasa el umbral y el mosaico en cada llamada, sin cambiarlo para las
        # demás cámaras. Con lo de siempre (o un agente que no lo conoce) es el detector tal cual
        self.detection = DetectionConfig.from_dict(self._analysis.get("detection"))
        self.analyzer = Analyzer(self._analysis, tuned(detector, self.detection), scorer, self.clip_pre, self.clip_post,
                                 clip_max=self.clip_max, clips=not self.metrics_only)
        # la IA de movimientos de hurto (`analysis["ia_hurto"]`): la arma el analizador con el análisis; acá solo se
        # avisa en el registro si el modelo no sirve para esta versión (`theft_spec`)
        self.analyzer.set_theft(theft_spec(self._analysis, self.metrics_only))
        # sin movimiento y sin nadie en escena no se usa el detector (ahorra placa de video)
        self.gate = MotionGate(min_fraction=self._motion_fraction())
        if not self.detection.is_default:
            log.info("[%s] detección de personas: %s", self.name, self.detection.describe())
        self.idle_pct: float | None = None  # parte de los cuadros salteados por quietud (último minuto)
        self.skipped_pct: float | None = None  # parte de lo que tocaba analizar y no se llegó (último minuto)
        self.video: dict = {}  # qué video manda la cámara (`video_info`): códec, tamaño y cuadros por segundo
        self._video_threads = video_threads()  # hilos para decodificar (None = los que elija FFmpeg)
        # decodificar en la placa (agent/video_placa.py, `VIGIA_VIDEO_PLACA=si`): lo que se sabe del video de la última
        # conexión con OpenCV (códec, tamaño, píxeles) y una imagen suya; tras la primera conexión se pasa a la placa
        self._gpu_hint: dict | None = None
        self._gpu_ref = None
        self._gpu_switch = False  # se cortó a propósito la conexión de OpenCV para volver a abrir por la placa
        self._av_back_at: float | None = None  # cuándo volver a probar la placa dentro del proceso (abrió con OpenCV)
        self._av_fails = 0
        self._gpu_fails = 0
        self._gpu_off_until = 0.0
        self._gpu_relearn: tuple | None = None  # el tamaño que la placa dijo que no era (ya se volvió a mirar)
        self._url = ""
        # procesador que gasta el hilo de esta cámara (último minuto, en % de un procesador): todo, leer el video
        # (con un hilo, ahí se decodifica) y analizar. `cpu_read_s`: lo gastado leyendo, acumulado (agent/consumo.py)
        self.cpu: dict = {}
        self.cpu_read_s = 0.0
        self.metrics_queue = metrics_queue
        self.outbox = outbox
        self.client = client
        self.clip_pool = clip_pool
        self._findings: dict[str, Finding] = {}
        # reloj del análisis: sigue corriendo entre reconexiones. Si volviera a cero, el buffer de clips, los clips
        # abiertos y el seguimiento esperarían a que alcanzara al viejo, guardando cuadros mientras tanto (27 oct:
        # la máquina de Google llegó a 13,7 GB con 5 cámaras de Hik-Connect, que se reconectan seguido)
        self._last_ts: float | None = None
        self._last_ts_at = 0.0
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self.run, name=f"cam-{self.id}", daemon=True)
        self.state = "iniciando"
        self.error: str | None = None
        self.fps = 0.0
        self.events = 0
        self.last_frame_at: float | None = None
        # vista en vivo: sube cuadros con recuadros mientras alguien mira la cámara en el panel
        self.live = LiveStreamer(self.id, self.name, client, self.analyzer.zones)
        # métricas del local (personas, tiempos, destinos, vehículos): mismos tracks, sin costo extra
        self.metrics = MetricsCollector(self.analyzer.zones, self.analyzer.engine.cfg.schedule)
        self._metrics_pending: list[dict] = []
        self.analysis_fp = self.analysis_of(cam_cfg)
        self._pending_analysis: dict | None = None
        # ventana de la PC central: último cuadro con recuadros (se arma solo si la ventana está abierta)
        self.show_window = False
        self.display = None  # np.ndarray | None

    @staticmethod
    def display_name(cam_cfg: dict) -> str:
        return f"{cam_cfg['client_name']} · {cam_cfg['name']}" if cam_cfg.get("client_name") else cam_cfg["name"]

    def rename(self, cam_cfg: dict) -> None:
        """Le cambiaron el nombre a la cámara (o al cliente): sin cortar el video."""
        self.name = self.live.name = self.display_name(cam_cfg)

    @staticmethod
    def fingerprint_of(cam_cfg: dict) -> str:
        """Lo que obliga a reconectar: conexión, modo, detector… Las zonas y reglas NO (se aplican en
        caliente con `update_analysis`); solo cuenta si hay que detectar vehículos (cambia el detector).
        El nombre tampoco (`rename`)."""
        base = {k: v for k, v in cam_cfg.items() if k not in ("analysis", "name", "client_name")}
        base["vehicles"] = bool((cam_cfg.get("analysis") or {}).get("vehicles"))
        conn = base.get("connection")
        if isinstance(conn, dict) and conn.get("brand") == "hikconnect":
            # por Hik-Connect el video sale del link que da el servidor (por el id de la cámara): el canal no lo cambia.
            # Si no, corregir un canal mal guardado (6 nov) reiniciaba la cámara y se perdía lo medido en esa hora
            base["connection"] = {k: v for k, v in conn.items() if k != "channel"}
        return json.dumps(base, sort_keys=True, default=str)

    @staticmethod
    def analysis_of(cam_cfg: dict) -> str:
        return json.dumps(cam_cfg.get("analysis") or {}, sort_keys=True, default=str)

    def update_analysis(self, analysis: dict) -> None:
        """Zonas, reglas, gestos o la detección de personas nuevos: se aplican en el próximo cuadro, sin cortar el
        video."""
        self._pending_analysis = analysis

    def _motion_fraction(self) -> float:
        return FAR_MOTION_FRACTION if self.detection.far else MOTION_FRACTION

    def _apply_detection(self, analysis: dict) -> bool:
        """Otra sensibilidad o «Personas lejos» (corre en el hilo de la cámara: sin carreras con el análisis, y sin
        reconectar). Lo raro (de un servidor más nuevo) queda en lo de siempre. Devuelve si cambió."""
        det = DetectionConfig.from_dict(analysis.get("detection"))
        if det == self.detection:
            return False
        self.detection = det
        self.analyzer.detector = tuned(self.analyzer.detector, det)  # (sobre el compartido: no se envuelve dos veces)
        self.gate.min_fraction = self._motion_fraction()
        log.info("[%s] detección de personas: %s", self.name, det.describe())
        return True

    def _sync_pose(self) -> None:
        """Con algún gesto activado o con la IA de movimientos de hurto el analizador mira la postura (el estimador
        compartido del proceso); sin ninguno de los dos, no la mira (no gasta). Corre en el hilo de la cámara, al
        conectar y al cambiar el análisis."""
        gestures = wants_gestures(self._analysis, self.metrics_only)
        ia = theft_spec(self._analysis, self.metrics_only) is not None
        pose = self.analyzer.pose if (gestures or ia) and self.analyzer.pose is not None else (
            shared_pose() if gestures or ia else None)  # (al reconectar no hace falta pedirla de nuevo)
        self.analyzer.pose = pose
        now = (gestures and pose is not None, ia and pose is not None)
        before, self._pose_uses = self._pose_uses, now
        if now == before:
            return
        if now[0] != before[0]:
            if now[0]:
                log.info("[%s] gestos de ocultamiento activados: %s", self.name,
                         ", ".join(GESTURE_NAMES[g].lower() for g in self.analyzer.engine.cfg.gestures.enabled))
            elif before[0]:  # (sin el modelo de postura no se llega acá: `shared_pose` ya lo dijo, una vez)
                log.info("[%s] gestos de ocultamiento apagados", self.name)
        if now[1] != before[1]:
            log.info("[%s] IA de movimientos de hurto %s", self.name, "activada" if now[1] else "apagada")

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self._stop.set()

    @property
    def state(self) -> str:
        return self._state

    @state.setter
    def state(self, value: str) -> None:
        # desde cuándo está así: la pestaña «Inicio» de la app dice «sin conexión hace 40 min» aunque se haya abierto
        # después (y avisa recién cuando una caída dura; las reconexiones de un momento no cuentan)
        if value != getattr(self, "_state", None):
            self._state, self._state_at = value, time.time()

    def status(self) -> dict:
        return {"camera_id": self.id, "state": self.state, "error": self.error, "fps": round(self.fps, 2),
                "state_age_s": round(max(0.0, time.time() - self._state_at)),
                "events": self.events, "live": self.live.active,
                "idle_pct": None if self.idle_pct is None else round(self.idle_pct * 100),
                **({"protocol": self.hik.protocol} if self.hik.brand == "hikconnect" else {}),
                "metrics_only": self.metrics_only, "gestures": self.analyzer.gestures,
                # qué parte de los cuadros que tocaba analizar no se analizó en el último minuto porque los análisis
                # tardaron de más (la máquina no da abasto)
                **({"skipped_pct": round(self.skipped_pct * 100)} if self.skipped_pct else {}),
                # la IA de movimientos de hurto está mirando (hay modelo activo, la postura cargada y mide hurtos)
                "ia_hurto": self.analyzer.theft_ai,
                # el video que llega (códec, tamaño, cuadros por segundo): la app marca los que pesan de más
                **self.video,
                # procesador del hilo de la cámara en el último minuto (% de uno): todo, leer el video y analizar
                **self.cpu,
                # («normal», «sensibilidad 8 de 10 · personas lejos»): para saber si la máquina ya la aplica
                "detection": self.detection.describe(), "last_frame_age_s":
                    None if self.last_frame_at is None else round(time.time() - self.last_frame_at, 1)}

    def run(self) -> None:
        backoff = 2.0
        while not self._stop.is_set():
            try:
                url = self.hik.live_url()  # en cada conexión: los links de Hik-Connect vencen
            except Exception as exc:  # noqa: BLE001 - sin link no hay video; se reintenta
                self.state, self.error = "sin_conexion", str(exc)
                log.warning("[%s] %s; reintento en %.0fs", self.name, self.error, backoff)
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 60.0)
                continue
            self._url = url
            cap = self._open_gpu(url) or self._open_av(url) or open_video(url, self._video_threads)
            if not cap.isOpened():
                why = self.hik.open_failure(url)
                wait = retry_wait(why, backoff)  # (clave rechazada: no insistir, el grabador bloquearía a la máquina)
                if self.hik.brand == "hikconnect":  # (el estado se corta a 300 letras: el motivo primero)
                    self.state, self.error = "sin_conexion", f"No abrió el video de Hik-Connect — {why}"
                    log.warning("[%s] %s (%s); reintento en %.0fs", self.name, self.error, self.hik.redacted(url),
                                wait)
                elif getattr(why, "hold", False):  # (clave: el motivo entero, el estado se corta a 300 letras)
                    self.state, self.error = "sin_conexion", str(why)
                    log.warning("[%s] %s (%s); reintento en %.0fs", self.name, self.error, self.hik.redacted(url),
                                wait)
                else:
                    self.state, self.error = "sin_conexion", f"No se pudo abrir {self.hik.redacted(url)}"
                    if why:
                        self.error += f" — {why}"
                    log.warning("[%s] %s; reintento en %.0fs", self.name, self.error, wait)
                cap.release()
                self._stop.wait(wait)
                backoff = min(backoff * 2, 60.0)
                continue
            self.state, self.error = "en_linea", None
            self.hik.opened()  # (Hik-Connect: este formato anda)
            log.info("[%s] conectada", self.name)
            connected_at = time.monotonic()
            self._bursty = bursty(self.hik.brand, url)
            try:
                self._read_loop(cap)
            except Exception as exc:
                log.exception("[%s] error procesando", self.name)
                self.state, self.error = "error", str(exc)
                self._stop.wait(5.0)
            finally:
                cap.release()
            if getattr(cap, "gpu", False) and time.monotonic() - connected_at < 30 and not self._stop.is_set():
                self._gpu_failed("el video por la placa se cortó enseguida")
            if self._gpu_switch and not self._stop.is_set():
                self._gpu_switch = False  # (ya se sabe qué video manda: se vuelve a abrir enseguida, por la placa)
                continue
            if not self._stop.is_set():
                self.state = "reconectando"
                # si el video se corta enseguida una y otra vez, esperar cada vez más (no martillar al equipo
                # ni a Hik-Connect pidiendo links); si anduvo un rato, reconectar rápido
                backoff = 2.0 if time.monotonic() - connected_at > 30 else min(backoff * 2, 60.0)
                self._stop.wait(backoff)
        self._finish_clips(self.analyzer.flush())
        self._send_metrics(self.metrics.close())
        self.state = "detenida"

    def _gpu_wanted(self, url: str) -> bool:
        """¿Esta cámara se puede leer por la placa? (pedido, la máquina puede, RTSP y no apagado por fallas)."""
        from . import video_placa
        return (video_placa.wanted() and url.lower().startswith("rtsp://")
                and not bursty(self.hik.brand, url) and time.monotonic() >= self._gpu_off_until
                and video_placa.available()[0])

    def _open_av(self, url: str):
        """El video con la placa dentro del proceso (agent/video_av.py, `VIGIA_VIDEO_PLACA=proceso`). None = con
        OpenCV: no se pidió, no es RTSP, llega en ráfagas, la máquina no puede o falló hace poco. Si no abre por algo
        que no es la placa, esta vez abre con OpenCV y se vuelve a probar la placa en 2 minutos (`AV_BACK_S`, el doble
        cada vez que vuelve a fallar); si es la placa, una hora con OpenCV."""
        from . import video_av
        if (not video_av.wanted() or not url.lower().startswith("rtsp://") or bursty(self.hik.brand, url)
                or time.monotonic() < self._gpu_off_until or not video_av.available()[0]):
            return None
        cap = video_av.AvCapture(url, self.hik.ffmpeg_input_args(), threads=self._video_threads)
        if cap.isOpened():
            if not cap.gpu:
                log.info("[%s] la placa no puede con este video: lo decodifica el procesador", self.name)
            self._av_fails, self._av_back_at = 0, None
            return cap
        why = cap.why or "?"
        if _AV_HW_ERROR.search(why):
            self._gpu_off_until = time.monotonic() + 3600
            log.warning("[%s] la placa no pudo abrir este video (%s): con OpenCV por una hora", self.name, why)
            return None
        self._av_fails += 1
        wait = min(AV_BACK_S * 2 ** (self._av_fails - 1), AV_BACK_MAX_S)
        self._av_back_at = time.monotonic() + wait
        log.warning("[%s] no abrió con la placa (%s): se prueba con OpenCV y se vuelve a la placa en %d min", self.name,
                    why, round(wait / 60))
        return None

    def _open_gpu(self, url: str):
        """El video por la placa, si se pidió y ya se sabe qué video manda la cámara. None = con OpenCV."""
        from . import video_placa
        if not self._gpu_hint or not self._gpu_wanted(url):
            return None
        h = self._gpu_hint
        if not video_placa.worth_it(h.get("video_w"), h.get("video_h"), h.get("video_fps")):
            return None  # (video liviano: con el procesador cuesta lo mismo; ver video_placa.MIN_PIXEL_RATE)
        # (analizando a 5 cuadros/s o menos se baja de la placa como mucho un cuadro cada 45 ms: ver video_placa)
        gap = video_placa.MIN_GAP_S if self.process_fps <= video_placa.THIN_MAX_PROCESS_FPS else 0.0
        size = (self._gpu_hint.get("video_w"), self._gpu_hint.get("video_h"))
        # si ya se volvió a mirar con OpenCV y dijo lo mismo, el tamaño que dice ffmpeg no se compara (no cambió la
        # cámara: los dos lo cuentan distinto; los cuadros salen igual al tamaño pedido)
        cap = video_placa.PlacaCapture(url, self.hik.ffmpeg_input_args(), self._gpu_hint, self._gpu_ref, min_gap=gap,
                                       check_size=size != self._gpu_relearn)
        if cap.isOpened():
            log.info("[%s] video por la placa (%s %sx%s%s%s)", self.name, cap.decoder, cap.w, cap.h,
                     ", rango completo" if cap.full_range else "",
                     f", llega 1 de cada {cap.thin} cuadros" if cap.thin > 1 else "")
            return cap
        cap.release()
        if cap.stale:  # la cámara cambió de tamaño: se vuelve a mirar con OpenCV (enseguida pasa otra vez a la placa)
            log.info("[%s] %s: se vuelve a mirar el video con OpenCV", self.name, cap.why)
            self._gpu_relearn = size
            self._gpu_hint = None
            return None
        self._gpu_failed(cap.why)
        return None

    def _gpu_failed(self, why: str) -> None:
        """La placa no anduvo: 10 minutos con el procesador (si no, cada conexión de OpenCV volvería a pasarse a la
        placa enseguida y no se analizaría nunca); a la tercera vez seguida, una hora."""
        self._gpu_fails += 1
        if self._gpu_fails >= 3:
            self._gpu_fails = 0
            self._gpu_off_until = time.monotonic() + 3600
            log.warning("[%s] el video por la placa falló 3 veces (%s): una hora con el procesador", self.name, why)
        else:
            self._gpu_off_until = time.monotonic() + 600
            log.warning("[%s] el video por la placa no anduvo (%s): 10 minutos con el procesador", self.name, why)

    def _read_loop(self, cap: cv2.VideoCapture) -> None:
        self._sync_pose()
        t0 = time.monotonic()
        # sigue desde donde quedó la conexión anterior, más lo que duró el corte
        base = 0.0 if self._last_ts is None else self._last_ts + max(t0 - self._last_ts_at, 0.1)
        # Hik-Connect (HLS) manda el video en ráfagas de 1-4 s: con el reloj de la PC se analizaría un solo
        # cuadro por ráfaga. Ahí se usa el reloj del video (marca de tiempo de cada cuadro). Igual con un link HLS.
        clock = MediaClock(cap) if getattr(self, "_bursty", self.hik.brand == "hikconnect") else None
        # «1080p lite» (960x1080 con píxeles 2:1): se analiza con píxeles cuadrados (960x540), igual que en
        # «Configurar…» y en las alarmas; si no, las personas quedan flacas y los gestos no andan
        self.video = video_info(cap)
        gpu = getattr(cap, "gpu", False)
        thin = int(getattr(cap, "thin", 1) or 1)  # (por la placa puede llegar 1 de cada 2: se informa lo que manda)
        if gpu:
            self._gpu_fails = 0
            self.video["video_codec"] = f"{self.video.get('video_codec') or 'video'} · placa"  # (la app lo muestra)
            if getattr(cap, "source_fps", 0):
                self.video["video_fps"] = round(cap.source_fps, 1)
        if not gpu and self._video_threads == 1 and heavy_video(self.video):
            self._video_threads = None  # (se reconecta una vez: `run` vuelve a abrir enseguida)
            log.info("[%s] video grande (%sx%s): se decodifica con varios hilos", self.name,
                     self.video.get("video_w"), self.video.get("video_h"))
            return
        sar = pixel_aspect(cap)
        # con OpenCV y la placa pedida: se anota qué video manda (para pedir el mismo por la placa) y, con la primera
        # imagen, se vuelve a abrir por la placa
        learn_gpu = not gpu and self.video.get("video_codec") in ("H.264", "H.265") and self._gpu_wanted(
            getattr(self, "_url", "") or "")
        if learn_gpu:
            self._gpu_hint = {**self.video, "sar": sar}
        if sar and sar != self._sar_logged:
            self._sar_logged = sar
            log.info("[%s] el video tiene píxeles de %.2f:1: se analiza con píxeles cuadrados", self.name, sar)
        last_proc = -1e9
        last_show = -1e9
        last_preview = -1e9
        last_metrics = last_idle = time.monotonic()
        window: list[float] = []
        fails = 0
        # un cuadro que «sale al instante» ya estaba esperando: el vivo tarda lo que tarda la cámara en mandarlo
        vfps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_s = 1.0 / vfps if 1.0 <= vfps <= 60.0 else 1.0 / 25
        # (con tope: a 8 cuadros/s medio cuadro son 62 ms, y un cuadro que llegó en vivo 30 ms después de terminar el
        # análisis se tiraba por «viejo»: si era el que tocaba, se analizaba un tercio menos. 20 ms es lo que siempre
        # tuvo un video de 25 cuadros/s; decodificar uno que ya estaba esperando lleva menos)
        waiting_grab = min(0.5 * frame_s, WAITING_GRAB_MAX_S)
        behind_since: float | None = None  # desde cuándo se está poniendo al día (None = al día)
        # cada cuánto toca analizar de verdad: el primer cuadro del video que cae a 1/process_fps o más del anterior
        # (con la segunda calidad a 8 cuadros/s y 5 pedidos, uno de cada dos: 250 ms)
        step_s = 1.0 / self.process_fps
        if clock is None:
            step_s = max(step_s, math.ceil((step_s - 1e-3) / frame_s - 1e-6) * frame_s)
        analyzed = slow = grabbed = 0
        learn_since: float | None = None  # (pasar a la placa: desde cuándo se cuentan los cuadros de la cámara)
        # abrió con OpenCV porque la placa dentro del proceso no pudo esa vez: cuándo volver a probarla
        av_back = None if gpu or getattr(cap, "av", False) else self._av_back_at
        learn_times: list[float] = []  # cuándo llegó cada cuadro mientras tanto
        late_s = span_s = 0.0  # lo que los análisis se pasaron de step_s / lo que ocupó cada uno (o step_s si fue menos)
        last_skip_log = time.monotonic()
        # procesador de este hilo por etapa (segundos en el minuto): leer el video y analizar
        cpu0 = thread_time()
        cpu_read = cpu_ai = 0.0
        while not self._stop.is_set():
            if av_back is not None and time.monotonic() >= av_back:
                self._av_back_at = None
                self._gpu_switch = True  # (se vuelve a abrir enseguida: con la placa dentro del proceso)
                log.info("[%s] se vuelve a probar el video con la placa", self.name)
                return
            if self._pending_analysis is not None:
                # se saca antes de aplicarlo: si falla, no se reintenta en cada cuadro
                pending, self._pending_analysis = self._pending_analysis, None
                self._apply_analysis(pending)
            # tomar todos los cuadros evita acumular retraso; solo se convierten a imagen los que se usan
            # (analizar o mostrar en la ventana), que es lo que más procesador gasta
            g0 = time.monotonic()
            c0 = thread_time()
            ok = cap.grab()
            c1 = thread_time()
            cpu_read += c1 - c0
            self.cpu_read_s += c1 - c0
            if not ok:
                fails += 1
                if fails >= 3:
                    self.error = "Se perdió la señal"
                    return
                continue
            fails = 0
            now = time.monotonic()
            grabbed += 1
            if learn_since is not None and len(learn_times) < 2000:
                learn_times.append(now)
            ts = base + (clock.tick() if clock else now - t0)  # segundos del video (o de la PC)
            self._last_ts, self._last_ts_at = ts, now
            self.last_frame_at = time.time()
            if behind_since is not None:
                if now - g0 < waiting_grab and now - behind_since < CATCHUP_MAX_S:
                    continue  # todavía es de los que se apilaron mientras se analizaba: no sirve (es viejo)
                behind_since = None
            if now - last_skip_log > 60:
                # los cuadros por segundo de verdad (lo que dice el video al abrir a veces no es cierto)
                self.video = {**self.video, "video_fps": round(thin * grabbed / (now - last_skip_log), 1)}
                grabbed = 0
                self.skipped_pct = late_s / span_s if span_s else 0.0
                if self.skipped_pct >= SLOW_LOG_FRACTION:
                    log.info("[%s] la máquina no da abasto: analizó %d cuadros en el último minuto, un %.0f %% menos de "
                             "los que tocaba (%d análisis tardaron más de %.0f ms)", self.name, analyzed,
                             100 * self.skipped_pct, slow, 1000 * step_s)
                span = now - last_skip_log
                self.cpu = {"cpu": round(100 * (c1 - cpu0) / span), "cpu_video": round(100 * cpu_read / span),
                            "cpu_ia": round(100 * cpu_ai / span)}
                if not gpu and self._video_threads == 1 and cpu_read / span > DECODE_BUSY:
                    self._video_threads = None
                    log.info("[%s] decodificar el video se lleva el %.0f %% de un procesador: se reconecta para "
                             "repartirlo entre varios hilos", self.name, 100 * cpu_read / span)
                    return
                cpu0, cpu_read, cpu_ai = c1, 0.0, 0.0
                analyzed = slow = 0
                late_s = span_s = 0.0
                last_skip_log = now
            analyze = ts - last_proc >= 1.0 / self.process_fps - 1e-3
            show = self.show_window and now - last_show >= DISPLAY_INTERVAL
            if not (analyze or show):
                continue
            c0 = thread_time()
            ok, frame = cap.retrieve()
            if learn_gpu and ok and frame is not None:
                if learn_since is None:
                    learn_since = now
                elif now - learn_since >= GPU_LEARN_S:
                    start = learn_since + GPU_LEARN_SKIP_S
                    rate = sum(1 for t in learn_times if t >= start) / max(now - start, 1e-3)
                    if 1.0 <= rate <= 120.0:  # los cuadros por segundo que manda de verdad
                        self._gpu_hint["video_fps"] = round(rate, 1)
                    from . import video_placa
                    hint = self._gpu_hint
                    if video_placa.worth_it(hint.get("video_w"), hint.get("video_h"), hint.get("video_fps")):
                        self._gpu_ref = frame.copy()  # (sin achicar: la de la placa se compara contra esta)
                        self._gpu_switch = True
                        return
                    learn_gpu = False  # video liviano: sigue con OpenCV (por la placa costaría lo mismo)
                    log.info("[%s] video liviano (%sx%s a %s cuadros/s): se sigue leyendo con el procesador",
                             self.name, hint.get("video_w"), hint.get("video_h"), hint.get("video_fps"))
            if ok and frame is not None and sar:
                frame = square_pixels(frame, sar, shrink=True)
            c1 = thread_time()
            cpu_read += c1 - c0
            self.cpu_read_s += c1 - c0
            if not ok or frame is None:
                continue
            if show:  # la ventana se dibuja a 10 cuadros por segundo como máximo
                last_show = now
                self.display = hud(frame, self.analyzer.zones, self.analyzer.last_tracks, ts, self.name,
                                   datetime.now(self.hik.tz), self._gesture_engine())
            if not analyze:
                continue
            last_proc = ts
            wall = datetime.now(self.hik.tz)
            c0 = thread_time()
            if self.gate.should_analyze(frame, ts, self.analyzer.busy):
                findings = self.analyzer.process(frame, ts, wall)
                self.metrics.update(self.analyzer.last_tracks, ts, frame.shape[1::-1], wall)
            else:  # escena quieta y vacía: no hace falta buscar personas
                self.analyzer.idle(frame, ts)
                findings = []
            cpu_ai += thread_time() - c0
            if now - last_idle > 60:
                last_idle = now
                self.idle_pct = self.gate.take_saved_fraction()
            if self.live.active:  # (la imagen en vivo la puede ver el cliente: sin la marcha de los gestos)
                self.live.push(frame, self.analyzer.last_tracks, ts, wall)
            for f in findings:
                self._on_finding(f)
            if now - last_metrics > METRICS_EVERY_SECONDS:
                last_metrics = now
                self._send_metrics(self.metrics.flush())
            self._finish_clips(self.analyzer.take_ready_clips())
            window = [t for t in window if now - t < 10] + [now]
            self.fps = len(window) / 10.0
            if self.client and now - last_preview > PREVIEW_EVERY_SECONDS:
                last_preview = now
                ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    self.clip_pool.submit(self._send_preview, jpg.tobytes())
            took = time.monotonic() - now
            analyzed += 1
            span_s += max(took, step_s)
            if took > step_s:  # tardó más que lo que hay hasta el próximo que tocaba: ese ya no se analiza a tiempo
                slow += 1
                late_s += took - step_s
            if clock is None and took > frame_s:
                behind_since = time.monotonic()  # este cuadro tardó más que uno de la cámara: se apilaron cuadros

    def _on_finding(self, f: Finding) -> None:
        self.events += 1
        log.info("[%s] %s (puntaje=%s, alerta=%s)", self.name, f.description, f.score, f.alert)
        self.outbox.add_event(self.id, f.meta(), f.snapshot_jpg)
        self._findings[f.uuid] = f

    def _finish_clips(self, ready: list[ClipReady]) -> None:
        for c in ready:
            f = self._findings.pop(c.uuid, None)
            self.clip_pool.submit(self._make_clip, c, f)

    def _make_clip(self, c: ClipReady, f: Finding | None) -> None:
        out = self.outbox.clip_path(c.uuid)
        try:
            if self.hik.nvr_clips and f is not None and f.wall_time is not None and not self.hik.source_url:
                # el mismo tramo que cubre el clip del buffer: desde que apareció la persona hasta que se fue
                start = f.wall_time - timedelta(seconds=max(0.0, f.ts - c.start_ts))
                end = f.wall_time + timedelta(seconds=max(self.clip_post, c.end_ts - f.ts))
                wait = (end - datetime.now(self.hik.tz)).total_seconds() + NVR_WRITE_DELAY_SECONDS
                if wait > 0:
                    self._stop.wait(wait)
                if download_nvr_clip(self.hik, start, end, out):
                    # (el tramo del grabador arranca en `start`: la alerta, tantos segundos después)
                    self.outbox.clip_done(c.uuid, "nvr", round(max(0.0, f.ts - c.start_ts), 2))
                    return
            if write_mp4(c.frames, out):
                # en qué segundo del clip está la alerta: la app lo marca en la barra del reproductor
                self.outbox.clip_done(c.uuid, "buffer", video_offset(c.frames, f.ts) if f is not None else None)
            else:
                self.outbox.clip_done(c.uuid, None)
        except Exception:
            log.exception("[%s] no se pudo generar el clip %s", self.name, c.uuid)
            self.outbox.clip_done(c.uuid, None)

    def _gesture_engine(self):
        """El motor de gestos, si la cámara los está mirando (para escribir el gesto en curso sobre la imagen)."""
        return self.analyzer.engine.gestures if self.analyzer.gestures else None

    def _apply_theft(self) -> None:
        """La IA de movimientos de hurto (`analysis["ia_hurto"]`): prenderla, cambiarle el modelo (se activó otro en la
        app) o apagarla, en el hilo de la cámara y sin reconectar. Con el mismo modelo, el analizador sigue con lo que
        venía midiendo de cada persona (y toma las zonas nuevas). Un modelo que esta versión no puede usar queda
        apagado (y una línea en el registro, una vez)."""
        self.analyzer.set_theft(theft_spec(self._analysis, self.metrics_only))

    def _apply_analysis(self, analysis: dict) -> None:
        if without_keys(analysis, "detection", "ia_hurto") == without_keys(self._analysis, "detection", "ia_hurto"):
            # solo cambió la detección de personas (se probó otra sensibilidad) o el modelo de la IA de hurtos (se
            # activó otro): las reglas, el seguimiento y lo medido siguen como venían (no se empieza de cero a contar
            # los segundos de cada persona en su zona)
            self._analysis = analysis
            self._apply_detection(analysis)
            self._apply_theft()
            self._sync_pose()  # (la IA de hurtos mira la postura: engancharla o soltarla)
            return
        try:
            zones, rule_cfg = parse_config(analysis)
        except Exception as exc:  # noqa: BLE001 - (un número en null, una clave rara): seguir con lo anterior
            log.warning("[%s] zonas, reglas o gestos inválidos, sigo con los anteriores: %s", self.name, exc)
            return
        self.analyzer.zones = zones
        # motor nuevo: los tiempos de las reglas y de los gestos rigen desde este cuadro
        self.analyzer.engine = RuleEngine(zones, rule_cfg)
        self._analysis = analysis
        self._apply_detection(analysis)  # otra sensibilidad o «Personas lejos»: desde este cuadro
        self._apply_theft()  # la IA de hurtos, con las zonas nuevas (o con otro modelo, o apagada)
        self._sync_pose()  # se prendieron o apagaron los gestos o la IA de hurtos: enganchar o soltar la postura
        # lo medido hasta ahora queda con las zonas viejas y la hora en curso sigue sumando (sin pisarla en el servidor)
        old = self.metrics
        self.metrics = old.with_zones(zones, rule_cfg.schedule)
        self._send_metrics(old.pending)
        self.live.zones = zones
        log.info("[%s] zonas, reglas y gestos actualizados (%d zonas)", self.name, len(zones))

    def _send_metrics(self, buckets: list[dict]) -> None:
        """Deja los cubos en la cola de la PC (un solo envío para todas las cámaras). Sin cola (pruebas,
        uso suelto) los manda directo en otro hilo; si falla, se reintentan con el próximo envío."""
        if self.metrics_queue is not None:
            self.metrics_queue.add([{"camera_id": self.id, **b} for b in buckets])
            return
        pending = self._metrics_pending + [{"camera_id": self.id, **b} for b in buckets]
        self._metrics_pending = []
        if not pending or self.client is None:
            return
        latest: dict[str, dict] = {}
        for b in pending:  # si un cubo quedó repetido por reintentos, vale el último
            latest[b["bucket_start"]] = b
        batch = list(latest.values())[-200:]

        def send() -> None:
            try:
                self.client.post_metrics(batch)  # type: ignore[union-attr]
            except Exception as exc:
                log.warning("[%s] no se pudieron enviar las métricas (%s); se reintenta", self.name, exc)
                self._metrics_pending = batch + self._metrics_pending
        self.clip_pool.submit(send)

    def _send_preview(self, jpg: bytes) -> None:
        try:
            self.client.preview(self.id, jpg)  # type: ignore[union-attr]
        except Exception as exc:
            log.debug("No se pudo enviar la vista previa: %s", exc)
