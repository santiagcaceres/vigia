"""«Ajustar cámara» en la app de escritorio: se analiza UNA cámara en esta PC para ponerla a punto antes de mandarla
a las máquinas de análisis. Sirve en vivo o con videos grabados (por ejemplo, hurtos reales de esa cámara).

- Se ve lo que ve el análisis: los cuadros que se analizan (a los fps de la cámara), las zonas, cada persona con su
  número y, por zona, cuánto lleva («gondola 32s»; caja y restringida contra su tiempo), y si ya «tomó algo».
- Las alertas son **de prueba**: quedan en la ventana; no van a los empleados ni al cliente ni sirven para entrenar.
- Cambiar zonas o tiempos recalcula al instante todo lo analizado (los últimos 10 min en vivo, o los videos
  enteros): detectar y seguir personas (lo caro) no depende de las zonas ni de los tiempos, así que se guarda lo
  que salió de cada cuadro y solo se vuelven a pasar las reglas.
- `stats` y `hints`: números y avisos para decidir si la cámara sirve (tamaño de las personas, confianza del
  detector, apariciones breves, cuadros por segundo, cortes, zonas que faltan).
- YOLO completo: en la misma pasada del modelo se ven también los objetos (mochilas, carteras, botellas…). Se
  dibujan y se cuentan, pero no entran al seguimiento ni a las reglas: las personas son las mismas que en las
  máquinas (`OnnxYoloDetector.detect_also`). Los bolsos, carteras y mochilas sí cuentan para «mano en la bolsa».
- Gestos de ocultamiento (core/gestos.py): si la cámara los tiene, se mira la postura de cada persona (core/pose.py,
  el mismo modelo que las máquinas) y se guarda con el cuadro. Cambiar qué gestos o sus segundos (`set_gestures`)
  recalcula todo sin volver a mirar la postura (lo caro); se dibuja el esqueleto y, bajo cada persona, la mano en el
  bolsillo, la ropa o la bolsa contra lo que pide («BOLSILLO 0.4/0.6s»). Es un indicio de posible ocultamiento para
  revisar, nunca una prueba.
- IA de movimientos de hurto (core/hurto_ia.py, `analysis["ia_hurto"]`: el modelo que el personal entrenó y activó en
  «Entrenar IA hurtos»): si la cámara lo tiene, se mira la postura de todos (aunque no tenga gestos), sobre cada persona
  se escribe su puntaje («IA 0,72», en rojo si pasa el umbral) y, cuando avisa, sale una alerta de prueba de tipo
  `hurto_ia` como las de las reglas (se puede mandar al filtro igual). Cambiar zonas o tiempos la recalcula con la
  postura ya guardada en cada cuadro (`TheftWatch`). Como todo, es un indicio para revisar.
- Detección de personas (core/deteccion.py): la «Sensibilidad» y «Personas lejos» de la cámara (`analysis["detection"]`)
  se aplican sobre el detector compartido de la app (`tuned`). `set_detection` prueba otras sin guardarlas: en vivo
  rigen para los cuadros nuevos; con clips hay que volver a analizar (detectar es justamente lo que cambia).
"""

from __future__ import annotations

import logging
import threading
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from core.analyzer import ZONE_COLORS, annotate
from core.deteccion import DEFAULT_SENSITIVITY, FAR_PASSES, MAX_SENSITIVITY, DetectionConfig
from core.detectors import Detection, Detector, tuned
from core.escena import COCO_ES, OBJECT_CONF, light
from core.gestos import (GESTURE_NAMES, GESTURE_OF, GESTURE_TYPES, GestureConfig, MANO_BOLSA, MANO_BOLSILLO, MANO_ROPA,
                         REF_HEIGHT, REGION_OF, body_of)
from core.metrics import MetricsCollector, door_counts
from core.pisadas import FootMap
from core.pose import BAG_CLASSES, build_pose, look
from core.pose import draw as draw_pose
from core.rules import (BAG_LABELS, HURTO_IA, SALIDA_SIN_PAGO, VEHICLE_LABEL, RuleConfig, RuleEngine, Zone,
                        parse_config)
from core.scorer import Scorer
from core.tracker import CentroidTracker, Track

from .cameras import build_camera
from .worker import MediaClock, bursty, theft_spec

log = logging.getLogger("vigia.ajuste")

WIDTH = 640  # los cuadros se guardan achicados (JPEG, ~30 KB): 10 min a 5 fps ≈ 90 MB
JPEG_QUALITY = 60
MIN_GESTURE_FPS = 5.0  # con gestos, la máquina analiza a 5 cuadros por segundo o más
LIVE_KEEP_SECONDS = 600.0  # en vivo se recuerdan (y se recalculan) los últimos 10 minutos
MAX_VIDEO_SECONDS = 30 * 60  # tope entre todos los videos de una prueba
SHORT_TRACK_SECONDS = 2.0
SMALL_PERSON = 0.10  # persona de menos del 10 % del alto de la imagen: el detector falla más
SMALL_SHARE = 30.0  # si el 30 % de las personas que ve son chicas, las del fondo se están perdiendo (aunque el promedio no)
BANNER_SECONDS = 2.0  # cuánto se ve el cartel de una alerta de prueba
VIDEO_ORIGIN = datetime(2000, 1, 1, 12)  # videos sin hora real: la hora es esta + segundos (el horario no se aplica)
CLIP_BEFORE = 60.0  # «⚑ Marcar hurto» en vivo: el clip va de 1 minuto antes del momento marcado...
CLIP_AFTER = 30.0  # ...hasta 30 s después (lo que haya)
ZONE_TEXT = {"shelf": "gondola", "checkout": "caja", "exit": "salida", "restricted": "restringida",
             "entrance": "entrada", "pump": "surtidor", "shop": "tienda", "staff": "personal",
             "aisle": "pasillo", "table": "mesa"}
ALERT_TEXT = {"salida_sin_pago": "SALIDA SIN PAGO", "zona_restringida": "ZONA RESTRINGIDA",
              "fuera_de_horario": "FUERA DE HORARIO", MANO_BOLSILLO: "POSIBLE OCULTAMIENTO (BOLSILLO)",
              MANO_ROPA: "POSIBLE OCULTAMIENTO (ROPA)", MANO_BOLSA: "POSIBLE OCULTAMIENTO (BOLSA)",
              HURTO_IA: "POSIBLE HURTO (IA DE MOVIMIENTOS)"}
REGION_TEXT = {"bolsillo": "BOLSILLO", "ropa": "ROPA", "bolsa": "BOLSA"}
REGION_PLACE = {"bolsillo": "al bolsillo", "ropa": "a la ropa", "bolsa": "a la bolsa"}
BAG_CONF = 0.3  # bolsos para «mano en la bolsa» con la misma confianza que las máquinas (core/pose.detect_with_bags)
POSE_OK_MIN = 60.0  # postura bien vista en menos de este % de las veces: los gestos no andan bien en esa cámara
# por qué un gesto no dio alerta (los motivos de core/gestos.py, dichos para el que ajusta la cámara)
SUPPRESS_TEXT = {"sin toma vista antes": "no se vio que tomara algo antes",
                 "las dos manos a la vez (acomodarse)": "las dos manos a la vez (acomodarse la ropa)",
                 "la mano ya estaba ahí antes de tomar": "la mano ya estaba ahí antes de tomar",
                 "postura (mano quieta mucho rato)": "la mano se quedó quieta mucho rato (es postura, no un gesto)",
                 "en la caja (pagar)": "estaba en la caja (pagar)",
                 "zona de personal": "estaba en la zona de personal"}
_KEEP = object()


def clip_name(client: str, camera: str, wall: datetime | None, what: str = "hurto") -> str:
    """«Minas - Pasillo bebidas - 2026-11-02 11.50.26 hurto.mp4» (sin caracteres que Windows no acepta)."""
    import re

    def clean(text: str) -> str:
        return re.sub(r"[\\/:*?\"<>|]+", "-", text or "").strip(" .") or "camara"
    stamp = (wall or datetime.now()).strftime("%Y-%m-%d %H.%M.%S")
    return " - ".join(x for x in (clean(client) if client else "", clean(camera)) if x) + f" - {stamp} {what}.mp4"


def gestures_blocked(analysis: dict) -> str | None:
    """Por qué la cámara no puede mirar gestos aunque se elijan (como el servidor, `camera_analysis`), o None."""
    if (analysis.get("profile") or "general") == "metricas":
        return "la cámara es del tipo «Solo métricas»"
    plans = analysis.get("plans")  # (lo que mide esta cámara; `client_plans`, lo que contrató el cliente)
    if plans is not None and "hurto" not in plans:
        if "hurto" in (analysis.get("client_plans") or ()):
            return "esta cámara no mide hurtos: se elige en «Editar» → «Qué mide esta cámara»"
        return "el cliente no tiene el plan «Detección de hurtos»"
    return None


def with_gestures(analysis: dict, gestures: dict | None) -> dict:
    """La configuración con otros gestos (sin guardar), armada como la arma el servidor: `gestures` aparte de `rules`
    y solo si hay alguno activado (un agente viejo rechaza nombres que no conoce en `rules`), y una cámara con las
    reglas apagadas deja de ser «solo métricas» si mira gestos. ValueError si un tiempo está mal; lo que esta versión
    no conoce (de un servidor más nuevo) se deja pasar, como en las máquinas."""
    a = dict(analysis)
    blocked = gestures_blocked(a)
    on = bool(GestureConfig.from_dict(gestures, strict=False).enabled) and blocked is None
    if on:
        a["gestures"] = dict(gestures)
    else:
        a.pop("gestures", None)
    if blocked is None and (a.get("rules") or {}).get("enabled_rules") == []:
        a["metrics_only"] = not on
    return a


def with_detection(analysis: dict, cfg: DetectionConfig | dict | None) -> dict:
    """La configuración con otra detección de personas (sin guardar), armada como la arma el servidor: una copia nueva
    con `analysis["detection"]` compacto (solo lo que difiere de lo de siempre), o sin la clave si es lo de siempre."""
    det = cfg if isinstance(cfg, DetectionConfig) else DetectionConfig.from_dict(cfg)
    a = dict(analysis)
    if det.is_default:
        a.pop("detection", None)
    else:
        a["detection"] = det.to_dict(compact=True)
    return a


GESTURE_FORM_KEYS = ("pocket_seconds", "clothes_seconds", "bag_seconds", "require_take")  # lo que se toca en la app


def gestures_to_save(raw: dict | None, shown: dict, effective: dict | None, type_enabled: list | None) -> dict | None:
    """Lo que «Guardar» manda de los gestos (PUT …/analysis, aparte de `rules`), o None si no cambió nada.

    `raw`: lo guardado en la cámara (lo elegido a mano; sin "enabled" = los gestos de su tipo). `shown`: lo de las
    casillas (`enabled`, los segundos y `require_take`). `effective`: lo que rige (`analysis["gestures"]` del
    servidor; None si no mira gestos). `type_enabled`: los gestos del tipo de cámara (None si no se saben).
    Solo se guarda lo que cambió contra lo que rige: no se fijan los valores de siempre (una versión nueva puede
    mejorarlos) ni la lista de gestos mientras sea la del tipo (si después se le cambia el tipo, cambian con él)."""
    raw = dict(raw or {})
    now = GestureConfig.from_dict(effective if effective else {k: v for k, v in raw.items() if k != "enabled"},
                                  strict=False)
    out = dict(raw)
    for key in GESTURE_FORM_KEYS:
        if key not in shown:
            continue
        v, cur = shown[key], getattr(now, key)
        if (bool(v) != cur) if key == "require_take" else abs(float(v) - cur) > 1e-6:
            out[key] = bool(v) if key == "require_take" else float(v)
    enabled = [g for g in GESTURE_TYPES if g in (shown.get("enabled") or ())]
    if set(enabled) != set(now.enabled) and not (
            "enabled" not in raw and type_enabled is not None and set(enabled) == set(type_enabled)):
        out["enabled"] = enabled
    return out if out != raw else None


def view_detect(detector, frame: np.ndarray, gest: bool) -> tuple[list, list]:
    """Personas y lo demás que ve YOLO, en una sola pasada (la ventana de la cámara y «Ver con la IA de esta PC»). Con gestos,
    los bolsos salen como en las máquinas (core/pose.detect_with_bags: la clase de más puntaje entre mochila, cartera y
    valija, desde 0,3) y lo demás aparte, desde 0,45: así «mano en la bolsa» ve acá los mismos bolsos que en la
    máquina. Un detector sin objetos (movimiento) da solo personas."""
    if not hasattr(detector, "detect_also"):
        return detector.detect(frame), []
    extra = dict(COCO_ES)
    if gest and hasattr(detector, "detect_groups"):
        rest = {c: n for c, n in extra.items() if c not in BAG_CLASSES}
        dets, (objs, bags) = detector.detect_groups(frame, [(rest, OBJECT_CONF), (BAG_CLASSES, BAG_CONF)])
        return dets, objs + bags
    dets, objs = detector.detect_also(frame, extra, BAG_CONF if gest else OBJECT_CONF)
    if gest:  # un detector sin grupos: los bolsos con la confianza de las máquinas; lo demás, como siempre
        objs = [o for o in objs if o.conf >= OBJECT_CONF or o.label in BAG_LABELS]
    return dets, objs


@dataclass
class Sample:
    """Un cuadro analizado: lo necesario para volver a pasar las reglas y para mostrarlo."""
    seg: int  # qué video (en vivo: 0)
    ts: float  # segundos desde el inicio del video (o desde que se abrió la cámara)
    wall: datetime | None  # hora real (solo en vivo)
    jpg: bytes
    size: tuple[int, int]  # tamaño del cuadro original: los recuadros están en esa escala
    tracks: list[Track]  # copia de las personas de ese cuadro (sin el histograma de color; con la postura, si se miró)
    # (qué, confianza, recuadro): se muestran; los bolsos, carteras y mochilas cuentan para «mano en la bolsa»
    objects: list[tuple[str, float, tuple]] = field(default_factory=list)
    posed: bool = False  # se miró la postura en este cuadro (la cámara tenía gestos)


@dataclass
class TestAlert:
    index: int  # número de cuadro analizado (cuenta desde que se abrió el ajuste)
    seg: int
    ts: float
    wall: datetime | None
    type: str
    description: str
    track_id: int
    score: float | None = None  # puntaje del modelo entrenado del cliente (si tiene)
    kept: bool = True  # False = el modelo la descartaría (no llegaría a los empleados)
    features: dict[str, float] = field(default_factory=dict)  # lo que usa el modelo (para mandar a entrenar)
    bbox: tuple | None = None


def ia_score_text(score: float) -> str:
    """«IA 0,72» (con coma, como se escriben los decimales acá)."""
    return f"IA {score:.2f}".replace(".", ",")


class TheftWatch:
    """La IA de movimientos de hurto (core/hurto_ia.py) sobre lo que analiza esta PC (la ventana de la cámara y «Ver
    con la IA»), con el mismo camino que las máquinas (core/analyzer.theft_events): ventanas de 3 s cada 1 s, el aviso
    cuando el puntaje pasa el umbral en ventanas seguidas, el enfriamiento por persona y cada aviso como un evento
    `hurto_ia` con las mismas medidas, descripción y zona que en la máquina. Además, `shown` da el último puntaje de
    cada persona, aunque no llegue al umbral, para escribirlo sobre su recuadro («IA 0,72») y explicar por qué un clip
    dio o no la alerta. No guarda video ni manda nada. ValueError si el modelo no sirve para esta versión."""

    def __init__(self, spec: dict, zones: list[Zone] | None = None):
        from core.analyzer import theft_detector

        self.zones = list(zones or [])
        self.detector = theft_detector(spec, self.zones)
        if self.detector is None:
            raise ValueError("el modelo de la IA de hurtos es de otra versión o está mal armado")
        self.threshold = float(self.detector.threshold)

    def step(self, ts: float, tracks: list[Track], bags: list, size: tuple[int, int],
             base: dict[int, dict] | None = None) -> list:
        """Un cuadro ya analizado: las personas del seguimiento con su postura (`kpts`), los bolsos (recuadros o
        Detection) y `base`, las medidas de cada persona (RuleEngine.current_features, para el filtro). Devuelve los
        avisos como RuleEvent de tipo `hurto_ia`."""
        from core.analyzer import theft_events

        size = (int(size[0]), int(size[1]))
        return theft_events(self.detector, ts, size, tracks, bags, self.zones, base or {})

    def shown(self) -> dict[int, tuple[float, bool]]:
        """Persona -> (puntaje de su última ventana, ¿pasa el umbral?): lo que se escribe sobre cada recuadro."""
        return {tid: (float(score), score >= self.threshold) for tid, score in self.detector.scores.items()}


def new_theft_watch(spec: dict | None, zones: list[Zone]) -> TheftWatch | None:
    """Un `TheftWatch` con el modelo de la cámara, o None (sin modelo, o uno que esta versión no puede usar)."""
    if spec is None:
        return None
    try:
        return TheftWatch(spec, zones)
    except Exception as exc:  # noqa: BLE001 - un modelo raro no puede cortar la ventana: sigue sin la IA de hurtos
        log.warning("La IA de movimientos de hurto no se pudo armar (%s): se sigue sin ella", exc)
        return None


class Tuner:
    """Analiza una cámara en esta PC (en vivo, o los `videos` dados uno tras otro) con las zonas y reglas de la
    cámara, sin mandar nada al servidor."""

    def __init__(self, cfg: dict, detector: Detector, videos: list[str] | None = None,
                 keep_seconds: float = LIVE_KEEP_SECONDS, pose=None):
        """`pose`: estimador de postura (core/pose.py) para los gestos; sin él, si la cámara tiene gestos se arma el
        de las máquinas (`build_pose`, compartido) al empezar a analizar."""
        self.cfg = cfg
        self.id: int = cfg["id"]
        self.name = f"{cfg['client_name']} · {cfg['name']}" if cfg.get("client_name") else cfg["name"]
        # el detector compartido de la app; `detector` es ese con la sensibilidad y «Personas lejos» de la cámara
        # (`set_analysis` / `set_detection`), sin cambiarlo para las demás cámaras que se miran a la vez
        self.base_detector = self.detector = detector
        self.detection = DetectionConfig()
        # el modelo entrenado con las etiquetas del cliente (si hay): dice qué alertas descartaría, como en las máquinas
        self.scorer = Scorer.maybe(cfg.get("filter_model"))
        self.videos = [str(v) for v in videos or []]
        self.live = not self.videos
        self.process_fps = float(cfg.get("process_fps") or 5.0)
        self.keep_seconds = keep_seconds
        self._lock = threading.RLock()
        self.samples: list[Sample] = []
        self.marks: list[dict] = []  # por cuadro: zonas y segundos de cada persona (RuleEngine.progress)
        # por cuadro: cuántas personas entraron a cada zona hasta ese momento y cuántas hay (MetricsCollector.live_counts,
        # las mismas reglas que las métricas de las máquinas); cada video cuenta de cero
        self.counts: list[dict] = []
        self._mc: MetricsCollector | None = None
        self.base = 0  # número del primer cuadro guardado (en vivo se descartan los viejos)
        self.alerts: list[TestAlert] = []
        # lo medido de los gestos de cada persona ((video, número) → GestureEngine.summary), para explicar
        self.gsum: dict[tuple[int, int], dict] = {}
        # la IA de movimientos de hurto (`analysis["ia_hurto"]`, si el personal activó un modelo): el detector en línea
        # del video que se analiza y el puntaje más alto de cada persona ((video, número) → puntaje), para explicar
        self.ia_spec: dict | None = None
        self.theft: TheftWatch | None = None
        self.ia_best: dict[tuple[int, int], float] = {}
        self.engine: RuleEngine | None = None
        self._engine_seg = -1
        self.pose = pose
        self._pose_built = pose is not None
        self.pose_missing = False  # la cámara tiene gestos pero en esta PC no está el modelo de postura
        self.set_analysis(cfg.get("analysis") or {})
        self.tracker = CentroidTracker()
        # YOLO completo: qué más ve el modelo en la misma pasada (solo para mostrar; no toca las reglas)
        self.extra = dict(COCO_ES) if hasattr(detector, "detect_also") else None
        self._objects: Counter[str] = Counter()  # en cuántos cuadros se vio cada objeto
        self._dark = self._night = 0
        # iniciando | conectando | en_linea | sin_conexion | analizando | listo | error | detenido
        self.state = "iniciando"
        self.error: str | None = None
        self.seg_names = [Path(v).name for v in self.videos]
        self.seg_seconds = [0.0] * len(self.videos)
        self.seg_errors: dict[int, str] = {}
        self.seg_now = 0
        self.progress = 0.0
        self.truncated = False
        self.speed: float | None = None  # videos: segundos de video por segundo de análisis
        self.cuts = 0
        self.tz = None
        # para `stats`
        self._n = 0
        self._n_people = 0
        self._det_n = 0
        self._det_conf = 0.0
        self._det_h = 0.0
        self._det_small = 0
        self._spans: dict[tuple[int, int], list[float]] = {}
        self._visible_now: set[tuple[int, int]] = set()
        self._times: list[float] = []
        self._pose_people = self._pose_small = 0  # personas vistas / más chicas que lo que pide la postura
        self._pose_tried = self._pose_ok = 0  # personas a las que se les miró la postura / con hombros y caderas a la vista
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self.run, name=f"ajuste-{self.id}", daemon=True)

    # ---------- zonas y reglas ----------
    def set_analysis(self, analysis: dict) -> None:
        """Zonas y reglas nuevas: se recalcula todo lo ya analizado (sin volver a detectar). ValueError si están mal."""
        zones, rule_cfg = parse_config(analysis)
        metrics_only = bool(analysis.get("metrics_only"))
        if metrics_only:  # como en las máquinas (Analyzer sin clips): solo métricas no mira gestos
            rule_cfg.gestures = GestureConfig()
        # la IA de movimientos de hurto: la de las máquinas (nunca en solo métricas; un modelo de otra versión, no)
        ia_spec = theft_spec(analysis, metrics_only)
        # la detección no vuelve a detectar lo ya analizado: rige para lo que se analice desde ahora
        detection = DetectionConfig.from_dict(analysis.get("detection"))
        self.detection, self.detector = detection, tuned(self.base_detector, detection)
        # qué decide el conteo por zona: si no cambia (se prueban otros tiempos o gestos), en vivo el conteo sigue
        # como venía (desde que se abrió) en lugar de rehacerse solo con los últimos 10 minutos que se recuerdan
        # (con la espera: tildar «Medir el tiempo de espera acá» en una zona también recuenta, si no la zona no la mediría)
        count_key = (tuple((z.name, z.type, tuple(map(tuple, z.polygon)), z.direction, z.wait) for z in zones),
                     repr(rule_cfg.schedule))
        with self._lock:
            keep = (self.live and count_key == getattr(self, "_count_key", None)
                    and len(self.counts) == len(self.samples) and self._mc is not None)
            self._count_key = count_key
            self.analysis = analysis
            self.zones, self.rule_cfg = zones, rule_cfg
            self.metrics_only = metrics_only
            self.ia_spec = ia_spec
            if (rule_cfg.gestures.enabled or ia_spec is not None) and not metrics_only:
                # con gestos (o la IA de hurtos, que mira la postura en ventanas de 3 s) la máquina analiza a 5 cuadros
                # por segundo o más (camera_fps del servidor): probar igual
                self.process_fps = max(MIN_GESTURE_FPS, self.process_fps)
            self._reevaluate(recount=not keep)

    def set_times(self, times: dict[str, float], gestures=_KEEP) -> bool:
        """Probar otros tiempos («tomó algo», caja, restringida) sin guardarlos; con `gestures`, también otros
        gestos (como `set_gestures`, en la misma vuelta). Devuelve `needs_pose()`."""
        analysis = {**self.analysis, "rules": {**(self.analysis.get("rules") or {}), **times}}
        self.set_analysis(analysis if gestures is _KEEP else with_gestures(analysis, gestures))
        return self.needs_pose()

    def set_detection(self, cfg: DetectionConfig | dict | None) -> bool:
        """Probar otra sensibilidad o «Personas lejos» sin guardarla: rige para los cuadros que se analicen desde ahora
        (en vivo, enseguida; con clips hay que volver a analizarlos). No se vuelven a pasar las reglas: lo ya analizado
        se detectó con la anterior. En vivo, «Cómo anda la cámara» vuelve a medir el detector de cero, para que se vea
        el efecto (cuántas personas, de qué tamaño, con qué confianza). Devuelve si cambió."""
        det = cfg if isinstance(cfg, DetectionConfig) else DetectionConfig.from_dict(cfg)
        with self._lock:
            if det == self.detection:
                return False
            self.analysis = with_detection(self.analysis, det)
            self.detection, self.detector = det, tuned(self.base_detector, det)
            if self.live:
                self._det_n, self._det_conf, self._det_h, self._det_small = 0, 0.0, 0.0, 0
        log.info("[%s] detección de personas para probar: %s", self.name, det.describe())
        return True

    def set_gestures(self, gestures: dict | None) -> bool:
        """Probar otros gestos o segundos (lo de `GestureConfig.to_dict`; None o sin "enabled" = apagados) sin
        guardarlos: se recalcula todo con la postura ya mirada, sin volver a estimarla. Devuelve `needs_pose()`."""
        self.set_analysis(with_gestures(self.analysis, gestures))
        return self.needs_pose()

    @property
    def gestures_on(self) -> bool:
        return bool(self.rule_cfg.gestures.enabled)

    @property
    def ia_on(self) -> bool:
        """La cámara tiene la IA de movimientos de hurto (un modelo activo que esta versión puede usar)."""
        return self.ia_spec is not None

    def needs_pose(self) -> bool:
        """Hay gestos (o la IA de hurtos) pero lo ya analizado se hizo sin mirar la postura: para verlos hay que volver
        a analizar (en vivo, se ven desde ahora)."""
        with self._lock:
            return ((self.gestures_on or self.ia_on) and not self.pose_missing
                    and any(not s.posed for s in self.samples))

    def _new_theft(self) -> TheftWatch | None:
        """El detector de la IA de hurtos para un video que empieza (cada video, de cero, como el motor de reglas)."""
        return new_theft_watch(self.ia_spec, self.zones)

    def _reevaluate(self, recount: bool = True) -> None:
        """Vuelve a pasar las reglas por todo lo guardado (sin detectar de nuevo). Con `recount`, también el conteo por
        zona (cambiaron las zonas); si no, el conteo queda como estaba."""
        engine, seg, mc, theft = None, None, None, None
        alerts: list[TestAlert] = []
        marks: list[dict] = []
        counts: list[dict] = []
        gsum: dict[tuple[int, int], dict] = {}
        ia_best: dict[tuple[int, int], float] = {}
        for i, s in enumerate(self.samples):
            if s.seg != seg:  # cada video empieza de cero
                engine, seg = RuleEngine(self.zones, self.rule_cfg), s.seg
                mc = self._collector() if recount else None
                # la IA de hurtos también: con la postura guardada en cada cuadro, sin volver a mirarla
                theft = self._new_theft()
            alerts += self._feed(engine, s, self.base + i, marks, gsum, theft, ia_best)
            if recount:
                counts.append(self._count(mc, s))
        self.engine = engine or RuleEngine(self.zones, self.rule_cfg)
        self.theft = theft if engine is not None else self._new_theft()
        if recount:
            self._mc = mc or self._collector()
            self.counts = counts
        self._engine_seg = seg if seg is not None else -1
        self.alerts, self.marks, self.gsum, self.ia_best = alerts, marks, gsum, ia_best

    def _collector(self) -> MetricsCollector:
        """Quién cuenta las personas que entran a cada zona (en vivo, con el horario del local para descontar al
        personal que llega con el local cerrado, como las máquinas; en un video no se sabe la hora)."""
        return MetricsCollector(self.zones, self.rule_cfg.schedule if self.live else None)

    @staticmethod
    def _count(mc: MetricsCollector, s: Sample) -> dict:
        wall = s.wall or VIDEO_ORIGIN + timedelta(seconds=s.ts)
        mc.update(s.tracks, s.ts, s.size, wall)
        out = mc.live_counts()
        if s.wall is None:
            out["since"] = None
        return out

    def _feed(self, engine: RuleEngine, s: Sample, index: int, marks: list[dict],
              gsum: dict[tuple[int, int], dict], theft: TheftWatch | None = None,
              ia_best: dict[tuple[int, int], float] | None = None) -> list[TestAlert]:
        gest = engine.gestures
        bags = [(label, box) for label, _conf, box in s.objects if label in BAG_LABELS] \
            if gest is not None or theft is not None else []
        evs = engine.update(s.tracks, s.ts, s.size, s.wall,
                            [Detection(box, label) for label, box in bags] if gest is not None else [])
        prog = engine.progress(s.ts)
        if theft is not None:  # la IA de hurtos: sus avisos son alertas de prueba más, como las reglas y los gestos
            evs = list(evs) + theft.step(s.ts, s.tracks, [box for _label, box in bags], s.size,
                                         engine.current_features(s.tracks, s.ts, s.size, s.wall))
            for tid, shown in theft.shown().items():  # el puntaje de cada persona, para escribirlo sobre ella
                prog.setdefault(tid, {"zones": [], "took": False, "gesture": None})["ia"] = shown
                if ia_best is not None:  # (y el más alto de cada una, para explicar)
                    ia_best[(s.seg, tid)] = max(shown[0], ia_best.get((s.seg, tid), 0.0))
        if gest is not None:
            for t in s.tracks:
                if t.label == VEHICLE_LABEL:
                    continue
                sm = gest.summary(t.id)
                if sm is not None:
                    gsum[(s.seg, t.id)] = sm  # se pisa en cada cuadro: queda lo de la última vez que se la vio
        marks.append(prog)  # con lo que pide cada gesto según el motor (la bolsa sin bolso a la vista, el doble)
        out = []
        for e in evs:
            score = self.scorer.score(e.type, e.features) if self.scorer else None
            out.append(TestAlert(index, s.seg, s.ts, s.wall, e.type, e.description, e.track_id, score,
                                 self.scorer.is_alert(score) if self.scorer else True, dict(e.features), e.bbox))
        return out

    # ---------- análisis ----------
    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        try:
            if self.live:
                self._run_live()
            else:
                self._run_videos()
        except Exception as exc:  # noqa: BLE001 - se muestra en la ventana
            log.exception("[%s] error en el ajuste", self.name)
            self.state, self.error = "error", str(exc)

    def _run_videos(self) -> None:
        t0, total = time.monotonic(), 0.0
        for seg, path in enumerate(self.videos):
            if self._stop.is_set() or self.truncated:
                break
            self.seg_now, self.state = seg, "analizando"
            self.tracker = CentroidTracker()
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                self.seg_errors[seg] = "no se pudo abrir (si es .dav o .264, exportalo del grabador como MP4 o AVI)"
                log.warning("[%s] %s: %s", self.name, self.seg_names[seg], self.seg_errors[seg])
                continue
            try:
                frames, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0, cap.get(cv2.CAP_PROP_FPS) or 0
                dur = frames / fps if frames > 0 and 1 <= fps <= 120 else None
                sar = pixel_aspect(cap)
                clock, last = MediaClock(cap), -1e9
                while not self._stop.is_set():
                    if not cap.grab():
                        break
                    ts = clock.tick()
                    if ts - last < 1.0 / self.process_fps - 1e-3:
                        continue
                    if total + ts > MAX_VIDEO_SECONDS:
                        self.truncated = True
                        break
                    ok, frame = cap.retrieve()
                    if not ok or frame is None:
                        continue
                    if sar:
                        frame = square_pixels(frame, sar)
                    last = ts
                    self._analyze(frame, ts, None, seg)
                    self.seg_seconds[seg] = ts
                    if dur:
                        self.progress = (seg + min(ts / dur, 1.0)) / len(self.videos)
            finally:
                cap.release()
            total += self.seg_seconds[seg]
        self.speed = total / max(time.monotonic() - t0, 1e-6)
        self.progress = 1.0
        self.state = "detenido" if self._stop.is_set() else "listo"

    def _run_live(self) -> None:
        cam = build_camera(self.cfg["connection"])
        self.tz = cam.tz
        t0, last_ts, backoff = time.monotonic(), -1.0, 2.0
        while not self._stop.is_set():
            self.state = "conectando"
            try:
                url = cam.live_url()  # los links de Hik-Connect vencen: uno nuevo en cada conexión
            except Exception as exc:  # noqa: BLE001 - sin link no hay video; se reintenta
                self.state, self.error = "sin_conexion", str(exc)
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 60.0)
                continue
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                self.state, self.error = "sin_conexion", cam.open_failure(url) or "No se pudo abrir el video"
                cap.release()
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 60.0)
                continue
            self.state, self.error, backoff = "en_linea", None, 2.0
            cam.opened()  # (Hik-Connect: este formato anda)
            # Hik-Connect (o un link HLS) manda el video en ráfagas: se usa el reloj del video, como en las máquinas
            clock = MediaClock(cap) if bursty(cam.brand, url) else None
            sar = pixel_aspect(cap)  # «1080p lite»: con píxeles cuadrados, como en las máquinas
            offset = max(last_ts + 0.2, time.monotonic() - t0)
            last, fails = -1e9, 0
            try:
                while not self._stop.is_set():
                    if not cap.grab():
                        fails += 1
                        if fails >= 3:
                            break
                        continue
                    fails = 0
                    ts = offset + clock.tick() if clock else time.monotonic() - t0
                    if ts - last < 1.0 / self.process_fps - 1e-3:
                        continue
                    ok, frame = cap.retrieve()
                    if not ok or frame is None:
                        continue
                    if sar:
                        frame = square_pixels(frame, sar)
                    last = last_ts = ts
                    self._analyze(frame, ts, datetime.now(cam.tz), 0)
            finally:
                cap.release()
            if not self._stop.is_set():
                self.cuts += 1
                self.state = "conectando"
                self._stop.wait(2.0)
        self.state = "detenido"

    def _pose_for(self):
        """El estimador de postura (se arma la primera vez que hace falta: tarda un poco)."""
        if not self._pose_built:
            self._pose_built = True
            self.pose = build_pose()
            self.pose_missing = self.pose is None
        return self.pose

    def _detect(self, frame: np.ndarray, gest: bool) -> tuple[list, list]:
        return view_detect(self.detector, frame, gest) if self.extra else (self.detector.detect(frame), [])

    def _analyze(self, frame: np.ndarray, ts: float, wall: datetime | None, seg: int) -> None:
        gest = self.gestures_on
        ia = self.ia_on  # la IA de hurtos también mira la postura y los bolsos, aunque la cámara no tenga gestos
        dets, objs = self._detect(frame, gest or ia)
        tracks = self.tracker.update(dets, ts, frame)
        # el mínimo es relativo al alto del cuadro: lo mismo que en la máquina aunque se pruebe con el stream principal
        min_px = self.rule_cfg.gestures.min_height(frame.shape[0])
        posed = (gest or ia) and self._pose_for() is not None
        if posed:  # postura de cada persona (lo caro): se guarda con el cuadro y no se vuelve a mirar
            look(self.pose, frame, tracks, min_px)
        else:
            for t in tracks:
                t.kpts = None
        h, w = frame.shape[:2]
        small = frame if w <= WIDTH else cv2.resize(frame, (WIDTH, int(h * WIDTH / w)), interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        sample = Sample(seg, ts, wall, jpg.tobytes() if ok else b"", (w, h), [replace(t, hist=None) for t in tracks],
                        [(o.label, o.conf, o.bbox) for o in objs], posed)
        dark, night = light(small)
        people = [t for t in tracks if t.visible and t.label != VEHICLE_LABEL]
        big = [t for t in people if t.bbox[3] - t.bbox[1] >= min_px]
        well = sum(1 for t in big if t.kpts is not None and body_of(t.kpts, t.bbox) is not None) if posed else 0
        with self._lock:
            self._dark += dark
            self._night += night
            self._objects.update({o[0] for o in sample.objects if o[1] >= OBJECT_CONF})
            self._measure(dets, sample, h)
            self._pose_people += len(people)
            self._pose_small += len(people) - len(big)
            if posed:
                self._pose_tried += len(big)
                self._pose_ok += well
            if seg != self._engine_seg or self.engine is None:
                self.engine, self._engine_seg = RuleEngine(self.zones, self.rule_cfg), seg
                self._mc = self._collector()
                self.theft = self._new_theft()
            self.samples.append(sample)
            self.alerts += self._feed(self.engine, sample, self.base + len(self.samples) - 1, self.marks, self.gsum,
                                      self.theft, self.ia_best)
            self.counts.append(self._count(self._mc, sample))
            if self.live:
                self._trim()

    def _trim(self) -> None:
        last = self.samples[-1].ts
        if last - self.samples[0].ts <= self.keep_seconds + 30:  # se recorta de a tandas
            return
        cut = next(i for i, s in enumerate(self.samples) if last - s.ts <= self.keep_seconds)
        del self.samples[:cut]
        del self.marks[:cut]
        del self.counts[:cut]
        self.base += cut
        self.alerts = [a for a in self.alerts if a.index >= self.base]
        if self.gsum or self.ia_best:
            keep = {(s.seg, t.id) for s in self.samples for t in s.tracks}
            self.gsum = {k: v for k, v in self.gsum.items() if k in keep}
            self.ia_best = {k: v for k, v in self.ia_best.items() if k in keep}

    def _measure(self, dets, sample: Sample, h: int) -> None:
        self._n += 1
        for d in dets:
            if d.label == VEHICLE_LABEL:
                continue
            self._det_n += 1
            self._det_conf += d.conf
            hh = (d.bbox[3] - d.bbox[1]) / h if h else 0.0
            self._det_h += hh
            self._det_small += hh < SMALL_PERSON
        visible = [t for t in sample.tracks if t.visible and t.label != VEHICLE_LABEL]
        self._n_people += bool(visible)
        for t in visible:
            span = self._spans.setdefault((sample.seg, t.id), [t.first_ts, sample.ts])
            span[1] = sample.ts
        self._visible_now = {(sample.seg, t.id) for t in visible}
        now = time.monotonic()
        self._times = [x for x in self._times if now - x < 10] + [now]

    # ---------- lo que muestra la ventana ----------
    def bounds(self) -> tuple[int, int]:
        """Cuadros que se pueden mostrar: [primero, último + 1)."""
        with self._lock:
            return self.base, self.base + len(self.samples)

    def sample_at(self, index: int) -> Sample | None:
        with self._lock:
            i = index - self.base
            return self.samples[i] if 0 <= i < len(self.samples) else None

    def image(self, index: int | None = None) -> np.ndarray | None:
        """El cuadro tal cual (sin dibujos): para dibujar zonas o dejar la imagen de la cámara."""
        with self._lock:
            if not self.samples:
                return None
            s = self.samples[-1] if index is None else self.sample_at(index)
        return cv2.imdecode(np.frombuffer(s.jpg, np.uint8), cv2.IMREAD_COLOR) if s else None

    def render(self, index: int | None = None, objects: bool = True) -> np.ndarray | None:
        """El cuadro `index` (o el último) con zonas, personas, segundos por zona y el cartel de la alerta; con
        `objects`, también lo demás que ve YOLO."""
        with self._lock:
            if not self.samples:
                return None
            i = len(self.samples) - 1 if index is None else min(max(index - self.base, 0), len(self.samples) - 1)
            s, mark, zones, cfg = self.samples[i], self.marks[i], self.zones, self.rule_cfg
            counts = self.counts[i] if i < len(self.counts) else None
            hot = [a for a in self.alerts if a.seg == s.seg and 0 <= s.ts - a.ts <= BANNER_SECONDS]
        img = cv2.imdecode(np.frombuffer(s.jpg, np.uint8), cv2.IMREAD_COLOR)
        return None if img is None else draw(img, s, mark, zones, cfg, hot, self.clock_text(s), objects, counts)

    def counts_at(self, index: int | None = None) -> dict | None:
        """Cuántas personas entraron a cada zona hasta el cuadro `index` (o el último) y cuántas había en ese momento
        (`MetricsCollector.live_counts`). En vivo se cuenta lo que se recuerda (los últimos 10 minutos); en un video,
        desde su comienzo. None si todavía no hay cuadros."""
        with self._lock:
            if not self.counts:
                return None
            i = len(self.counts) - 1 if index is None else min(max(index - self.base, 0), len(self.counts) - 1)
            return self.counts[i]

    def export_clip(self, path: Path, index: int | None = None, before: float = CLIP_BEFORE,
                    after: float = CLIP_AFTER, at: tuple[int, float] | None = None) -> tuple[Path, float, float]:
        """«⚑ Marcar hurto» en vivo: guarda como video (`path`, MP4) el pedazo alrededor del cuadro `index` (o el
        último), desde `before` segundos antes hasta `after` después (lo que haya), con los cuadros tal cual los vio el
        análisis (sin dibujos, píxeles cuadrados, 640 de ancho) a los cuadros por segundo del análisis: así ese clip se
        vuelve a analizar igual que en vivo. Devuelve (archivo, segundo del clip en que está el momento marcado,
        duración del clip). `at` = (video, segundo) del momento: mejor que el índice, porque en vivo, mientras se
        confirma, se siguen descartando los cuadros más viejos. ValueError si no hay video, si ese momento ya se
        descartó o si no se puede escribir."""
        from .grabacion import open_writer
        with self._lock:
            if not self.samples:
                raise ValueError("Todavía no hay video para guardar.")
            if at is not None:
                seg, ts = at
                same = [s for s in self.samples if s.seg == seg]
                if not same or ts < same[0].ts - 1.0:
                    raise ValueError("ese momento ya se descartó (en vivo se recuerdan los últimos 10 minutos)")
                center = min(same, key=lambda s: abs(s.ts - ts))
            else:
                i = len(self.samples) - 1 if index is None else min(max(index - self.base, 0), len(self.samples) - 1)
                center = self.samples[i]
            chosen = [s for s in self.samples
                      if s.seg == center.seg and center.ts - before <= s.ts <= center.ts + after and s.jpg]
            fps = self.process_fps
        if not chosen:
            raise ValueError("Todavía no hay video para guardar.")
        first = cv2.imdecode(np.frombuffer(chosen[0].jpg, np.uint8), cv2.IMREAD_COLOR)
        if first is None:
            raise ValueError("No se pudo leer el video guardado en memoria.")
        size = (first.shape[1], first.shape[0])
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        writer, _codec = open_writer(path, fps, size)
        if writer is None:
            raise ValueError("Esta PC no puede escribir videos MP4 (OpenCV sin códec).")
        start, end = chosen[0].ts, chosen[-1].ts
        n = int(round((end - start) * fps)) + 1
        j, shown, img = 0, -1, first
        try:
            # a ritmo parejo: cada cuadro se repite hasta que toca el siguiente (en vivo pueden llegar disparejos)
            for f in range(n):
                t = start + f / fps
                while j + 1 < len(chosen) and chosen[j + 1].ts <= t + 1e-6:
                    j += 1
                if j != shown:
                    shown = j
                    dec = cv2.imdecode(np.frombuffer(chosen[j].jpg, np.uint8), cv2.IMREAD_COLOR)
                    if dec is not None:
                        img = dec if (dec.shape[1], dec.shape[0]) == size else cv2.resize(dec, size)
                writer.write(img)
        finally:
            writer.release()
        return path, center.ts - start, (n - 1) / fps

    def clock_text(self, s: Sample | TestAlert) -> str:
        if s.wall is not None:
            return s.wall.strftime("%H:%M:%S")
        t = int(s.ts)
        name = self.seg_names[s.seg] if s.seg < len(self.seg_names) else ""
        return f"{name} {t // 60:02d}:{t % 60:02d}".strip()

    def segments(self) -> list[dict]:
        """Videos: cuántas alertas dio cada uno (un video de hurto sin alertas = la cámara no lo vería)."""
        with self._lock:
            alerts = list(self.alerts)
        return [{"name": n, "seconds": self.seg_seconds[i], "error": self.seg_errors.get(i),
                 "done": i < self.seg_now or self.state in ("listo", "detenido", "error"),
                 "alerts": [a for a in alerts if a.seg == i]} for i, n in enumerate(self.seg_names)]

    def stats(self) -> dict:
        with self._lock:
            n, det = self._n, self._det_n
            span = self.samples[-1].ts - self.samples[0].ts if len(self.samples) > 1 else 0.0
            alerts = len(self.alerts)
            dropped = sum(1 for a in self.alerts if not a.kept)
            persons = len(self._spans)
            short = sum(1 for k, (a, b) in self._spans.items() if b - a < SHORT_TRACK_SECONDS and k not in self._visible_now)
            objects = [(k, 100.0 * v / n) for k, v in self._objects.most_common(6)] if n else []
            dark_pct, night_pct = (100.0 * self._dark / n, 100.0 * self._night / n) if n else (0.0, 0.0)
            size = self.samples[-1].size if self.samples else None  # el tamaño con que se detecta (píxeles cuadrados)
            pose = {"people": self._pose_people, "tried": self._pose_tried,
                    "small_pct": 100.0 * self._pose_small / self._pose_people if self._pose_people else None,
                    "ok_pct": 100.0 * self._pose_ok / self._pose_tried if self._pose_tried else None,
                    "min_pct": 100.0 * self.rule_cfg.gestures.min_person_px / REF_HEIGHT}
        return {"frames": n, "seconds": n / self.process_fps, "people_pct": 100.0 * self._n_people / n if n else 0.0,
                "persons": persons, "short": short, "detections": det, "objects": objects,
                "dark_pct": dark_pct, "night_pct": night_pct,
                "height_pct": 100.0 * self._det_h / det if det else None,
                "small_pct": 100.0 * self._det_small / det if det else None,
                "conf": self._det_conf / det if det else None,
                "fps": len(self._times) / 10.0 if self.live else None, "cuts": self.cuts, "alerts": alerts,
                "dropped": dropped,
                "per_hour": alerts / (span / 3600.0) if self.live and span >= 300 else None,
                "door": self.door(), "gestures": self.gestures_on, "ia": self.ia_on, "pose": pose,
                "sensitivity": self.detection.sensitivity, "far": self.detection.far, "frame_size": size}

    def footmap(self) -> FootMap:
        """Mapa de pisadas de lo analizado (cada video por separado): dónde camina y dónde se para la gente, para
        dibujar las zonas sobre datos (core/pisadas.py). En vivo se recalcula como mucho cada 2 s."""
        with self._lock:
            key = (len(self.samples), self.base)
            old = getattr(self, "_fm_key", None)
            if key == old or (old and time.monotonic() - self._fm_at < 2.0):
                return self._fm
            samples = list(self.samples)
        fm = FootMap.empty()
        for seg in sorted({s.seg for s in samples}):
            fm = fm.merge(FootMap.build([(s.ts, s.size, s.tracks) for s in samples if s.seg == seg]))
        with self._lock:
            self._fm_key, self._fm, self._fm_at = key, fm, time.monotonic()
        return fm

    def door(self) -> dict | None:
        """Puerta (zona Entrada): cuántos entraron, salieron y cuántos se tomaron como personal en lo analizado, para
        compararlo con un conteo a mano. Se recalcula con las zonas de ahora sobre los recuadros guardados (cada
        video por separado). None si no hay zona Entrada."""
        if not any(z.type == "entrance" for z in self.zones):
            return None
        with self._lock:
            key = (len(self.samples), self.base, id(self.zones), id(self.rule_cfg))
            old = getattr(self, "_door_key", None)
            if key == old or (old and key[2:] == old[2:] and time.monotonic() - self._door_at < 2.0):
                return self._door  # (en vivo crece a cada cuadro: como mucho cada 2 s)
            samples = list(self.samples)
        origin = VIDEO_ORIGIN  # videos sin hora real: el horario del local no se aplica
        out = {"entries": 0, "exits": 0, "staff": 0, "staff_entries": 0, "door": 0, "people": 0}
        for seg in sorted({s.seg for s in samples}):
            frames = [(s.tracks, s.ts, s.size, s.wall or origin + timedelta(seconds=s.ts)) for s in samples if s.seg == seg]
            got = door_counts(frames, self.zones, self.rule_cfg.schedule if self.live else None)
            out = {k: max(out[k], got[k]) if k == "door" else out[k] + got[k] for k in out}
        with self._lock:
            self._door_key, self._door, self._door_at = key, out, time.monotonic()
        return out

    def explain(self, seg: int, top: int = 3) -> list[str]:
        """Por qué un video dio (o no) alertas con la configuración de ahora: qué midió de cada persona (segundos a la
        vista, en cada zona y si «tomó algo»; con gestos, la postura, las tomas y la mano en el bolsillo, la ropa o la
        bolsa contra lo que pide, y por qué se descartó) contra lo que pide cada regla. Para «🎞 Clips»."""
        with self._lock:
            rows = [(s, m) for s, m in zip(self.samples, self.marks) if s.seg == seg]
            alerts = [a for a in self.alerts if a.seg == seg]
            zones, cfg = list(self.zones), self.rule_cfg
            gsum = {tid: v for (sg, tid), v in self.gsum.items() if sg == seg}
            ia_best = {tid: v for (sg, tid), v in self.ia_best.items() if sg == seg}
            ia = self.ia_spec
        seen: dict[int, list[float]] = {}  # id -> [primer, último] momento a la vista
        in_zone: dict[int, dict[str, float]] = {}  # id -> zona -> segundos seguidos (máximo)
        took: set[int] = set()
        edge: Counter[int] = Counter()  # cuadros con los pies cortados por el borde de abajo
        frames: Counter[int] = Counter()
        for s, m in rows:
            for t in s.tracks:
                if not t.visible or t.label == VEHICLE_LABEL:
                    continue
                span = seen.setdefault(t.id, [s.ts, s.ts])
                span[1] = s.ts
                frames[t.id] += 1
                edge[t.id] += t.bbox[3] >= 0.97 * s.size[1]
                info = m.get(t.id) or {}
                for name, _typ, secs in info.get("zones") or []:
                    z = in_zone.setdefault(t.id, {})
                    z[name] = max(z.get(name, 0.0), secs)
                if info.get("took"):
                    took.add(t.id)
        alerted = {a.track_id for a in alerts}
        # primero los que dieron alerta; después, los que más se vieron
        people = sorted((tid for tid, (a, b) in seen.items() if b - a >= SHORT_TRACK_SECONDS),
                        key=lambda tid: (tid not in alerted, seen[tid][0] - seen[tid][1]))[:top]
        types = {z.name: z.type for z in zones}
        rules = set(cfg.enabled_rules)
        g = cfg.gestures
        out: list[str] = []
        if not people:
            return ["No se vio a nadie más de 2 s: no hay nada que medir."]
        if not zones:
            out.append("La cámara no tiene zonas: ninguna regla puede saltar. Dibujalas con «✎ Marcar zonas»."
                       + (" Los gestos sí miran sin zonas." if g.enabled else "")
                       + (" La IA de movimientos de hurto también." if ia is not None else ""))
        for tid in people:
            secs = seen[tid][1] - seen[tid][0]
            text = f"Persona #{tid}: {secs:.0f} s a la vista"
            if in_zone.get(tid):
                parts = []
                for name, v in sorted(in_zone[tid].items(), key=lambda kv: -kv[1]):
                    parts.append(f"{v:.0f} s seguidos en «{name}»")
                text += "; " + ", ".join(parts)
                if tid in took:
                    text += "; «tomó algo»"
            elif zones:
                text += " y nunca pisó una zona (cuenta el punto de los pies)"
                if edge[tid] >= 0.5 * frames[tid]:
                    text += ". Tiene los pies cortados por el borde de abajo: la zona tiene que llegar hasta el borde"
            if g.enabled:
                text += ". Gestos: " + gesture_text(gsum.get(tid), frames[tid], g)
            if ia is not None:
                text += ". IA de movimientos: " + ia_text(ia_best.get(tid), ia)
            out.append(text + ".")
        if not alerts and zones:
            if took and SALIDA_SIN_PAGO in rules:
                if not any(t == "exit" for t in types.values()):
                    out.append("«Tomó algo» solo da alerta («salida sin pago») si después pisa una zona Salida sin pasar "
                               "por Caja, y esta cámara no tiene Salida.")
                else:
                    out.append("«Tomó algo» da alerta si después pisa la Salida sin pasar por Caja: no llegó a la Salida "
                               "en este video.")
        # lo que mira la postura y no dio alerta en este video: si la postura no se pudo mirar, se dice una sola vez
        quiet = [who for who, on in (("los gestos", g.enabled and not any(a.type in g.enabled for a in alerts)),
                                     ("la IA de movimientos de hurto",
                                      ia is not None and not any(a.type == HURTO_IA for a in alerts))) if on]
        if quiet and self.pose_missing:
            what = ("ni los gestos ni la IA de movimientos de hurto se probaron" if len(quiet) == 2 else
                    f"{quiet[0]} no se {'probaron' if quiet[0] == 'los gestos' else 'probó'}")
            out.append(f"En esta PC no está el modelo de postura: {what} (actualizá la app).")
        elif quiet and any(not s.posed for s, _m in rows):
            out.append(f"Este video se analizó sin mirar la postura: para ver {' y '.join(quiet)}, volvé a analizarlo "
                       "(«▶ Analizar todos de nuevo»).")
        elif quiet:
            if "la IA de movimientos de hurto" in quiet and not ia_best:
                out.append("A nadie se le vio la postura en suficientes cuadros (chicas, tapadas, de espaldas o "
                           "cortadas por el borde): la IA de movimientos de hurto no pudo mirar este video.")
            if "los gestos" in quiet and not any((gsum.get(tid) or {}).get("frames") for tid in people):
                out.append("No se le vio la postura a nadie (chicas, tapadas, de espaldas o cortadas por el borde): los "
                           "gestos no pueden ver nada en este video. Mirá «Cómo anda la cámara».")
            elif "los gestos" in quiet and g.require_take and not any((gsum.get(tid) or {}).get("takes")
                                                                        for tid in people):
                out.append("Los gestos piden ver antes que tome algo (parado, estira el brazo hacia el estante y lo trae "
                           "al cuerpo) y no se vio ninguna toma. Si la cámara ve la góndola de costado o muy de arriba, "
                           "probá sin «Pedir que tome algo antes» (da más falsas alarmas).")
        return out

    def hints(self) -> list[str]:
        """Avisos para decidir si la cámara está lista. Dicen qué mirar; no reemplazan mirar las alertas."""
        st = self.stats()
        types = {z.type for z in self.zones}
        rules = set(self.rule_cfg.enabled_rules)
        g = self.rule_cfg.gestures
        out: list[str] = []
        if any(z.type == "entrance" and not z.direction for z in self.zones):
            out.append("La zona Entrada no tiene sentido: cuenta igual a los que entran y a los que salen. Marcá hacia "
                       "dónde caminan los que entran en «✎ Marcar zonas» → «Sentido».")
        if self.metrics_only or (not rules and not g.enabled and not self.ia_on):
            why = gestures_blocked(self.analysis) or "el tipo de cámara no usa reglas"
            out.append(f"Esta cámara no da alertas: es «solo métricas» ({why}).")
        elif not self.zones:
            out.append("No hay zonas: no se puede disparar ninguna regla. Dibujá góndolas, caja y salida con «✎ Marcar zonas»."
                       + (" Los gestos sí miran sin zonas (con las góndolas marcadas dan menos falsas alarmas)."
                          if g.enabled else "")
                       + (" La IA de movimientos de hurto también mira sin zonas." if self.ia_on else ""))
        else:
            if SALIDA_SIN_PAGO in rules and "shelf" not in types:
                out.append("Sin zona «Góndola» no hay «tomó algo» (ni «salida sin pago»).")
            if SALIDA_SIN_PAGO in rules:
                if "exit" not in types:
                    out.append("Sin zona «Salida», «salida sin pago» no se puede disparar.")
                elif "shelf" in types and "checkout" not in types:
                    out.append("Sin zona «Caja», cualquiera que pase por una góndola y salga dispara «salida sin pago» "
                               "(aunque haya pagado). Si la caja se ve en otra cámara, mejor elegir otro tipo de cámara.")
        det = self.detection
        small = ""
        if st["detections"] >= 20:
            if st["height_pct"] < SMALL_PERSON * 100:
                small = f"Las personas se ven chicas (en promedio {st['height_pct']:.0f} % del alto de la imagen)"
            elif (st["small_pct"] or 0.0) >= SMALL_SHARE:  # (el promedio solo cuenta a las que ve: las del fondo no)
                small = (f"El {st['small_pct']:.0f} % de las personas que ve ocupan menos del "
                         f"{SMALL_PERSON * 100:.0f} % del alto de la imagen")
            if small:
                out.append(small + ": el detector pierde a las del fondo. " + self._far_advice())
            if st["conf"] < 0.5:
                if det.sensitivity <= DEFAULT_SENSITIVITY:
                    # (el pedido del 6 nov: un depósito grande, «confianza 0.49», y la gente del fondo no aparecía)
                    out.append(f"El detector duda (confianza media {st['conf']:.2f} de 1): poca luz, contraluz, reflejos o "
                               "cámara muy de arriba. Probala también de noche."
                               + ("" if small else " Si no ve a las personas del fondo, " + self._far_advice(lower=True)))
                else:  # con más sensibilidad la confianza media baja sola: acepta personas más dudosas
                    out.append(f"Con la sensibilidad en {det.sensitivity} el detector acepta personas desde "
                               f"{det.conf:.2f} de confianza: fijate que los recuadros nuevos sean personas y no cajas, "
                               "maniquíes o carteles.")
        res = self._resolution_hint(st.get("frame_size"), small=bool(small)) if st["frames"] >= 20 else ""
        if res:
            out.append(res)
        if g.enabled:
            out += self._gesture_hints(st, types)
        elif self.ia_on:
            out += self._ia_hints(st)
        if st["frames"] >= 20 and st["night_pct"] > 50:
            out.append("La imagen está en blanco y negro (infrarrojo, de noche): el detector ve bastante menos"
                       + (" y la postura peor (los gestos fallan más)" if g.enabled else "")
                       + ". Probala también de día antes de aprobar.")
        elif st["frames"] >= 20 and st["dark_pct"] > 50:
            out.append("La imagen está oscura: el detector ve menos. Si se puede, más luz o subir el brillo en el grabador.")
        if st["persons"] >= 5 and st["short"] / st["persons"] > 0.3:
            out.append(f"{st['short']} de {st['persons']} personas se vieron menos de 2 s: detecciones falsas (reflejos, "
                       "maniquíes, carteles) o alguien que cambia de número. Mirá el video: si una persona cambia de "
                       "número, las reglas de tiempo fallan."
                       + (f" Con la sensibilidad en {det.sensitivity} es más probable que sean falsas: si son cosas y no "
                          "personas, bajala un poco." if det.sensitivity > DEFAULT_SENSITIVITY else ""))
        if self.live:
            if st["seconds"] >= 30 and st["fps"] is not None and st["fps"] < 0.7 * self.process_fps:
                out.append(f"Se analizan {st['fps']:.1f} cuadros por segundo de {self.process_fps:.0f}: esta PC no da "
                           "abasto o el video llega cortado"
                           + (f" (con «Personas lejos» el detector hace {FAR_PASSES} veces el trabajo)." if det.far
                              else "."))
            if self.cuts >= 2:
                out.append(f"El video se cortó {self.cuts} veces.")
            if st["seconds"] >= 60 and st["persons"] == 0:
                out.append("Todavía no pasó nadie: dejala correr hasta que haya gente, o probá con un video.")
        if not out and st["detections"] >= 20:
            out.append("No se ven problemas. Antes de aprobar, revisá que cada alerta de prueba tenga sentido.")
        return out

    def _far_advice(self, lower: bool = False) -> str:
        """Qué probar cuando las personas se ven chicas, según la detección que ya tiene la cámara (`lower`: para
        seguir una frase)."""
        text = self._far_text()
        return text[0].lower() + text[1:] if lower else text

    def _far_text(self) -> str:
        det = self.detection
        far = (f"«Personas lejos» (mira la imagen por partes: el detector trabaja unas {FAR_PASSES} veces más; rinde "
               "más con la calidad principal o con la secundaria del grabador en 1280x720)")
        if not det.far:
            if det.sensitivity >= MAX_SENSITIVITY:
                return f"La sensibilidad ya está en {MAX_SENSITIVITY}: probá prender {far}."
            return f"Probá subir la «Sensibilidad» (ahora {det.sensitivity}) o prender {far}."
        if det.sensitivity < MAX_SENSITIVITY:
            return (f"Ya está «Personas lejos»: probá subir la «Sensibilidad» (ahora {det.sensitivity}), la calidad "
                    "principal del grabador o una cámara más cerca.")
        return (f"Ya está todo al máximo («Personas lejos» y sensibilidad {MAX_SENSITIVITY}): hace falta la calidad "
                "principal del grabador (o la secundaria en 1280x720) o una cámara más cerca.")

    def _stream(self) -> str | None:
        """Qué calidad del grabador mira la máquina («sub» o «main»); None si la cámara va por un link propio (no se
        sabe qué calidad es)."""
        conn = self.cfg.get("connection") or {}
        if conn.get("source_url"):
            return None
        return conn.get("detect_stream") or self.cfg.get("detect_stream") or "sub"

    def _resolution_hint(self, size, small: bool = False) -> str:
        """Si lo que se ve acá con la detección no es lo que va a ver la máquina, por el tamaño de la imagen: un clip
        en alta (de «Bajar grabación» o exportado del grabador) contra la calidad secundaria que mira la máquina (en
        general 640x360), o un vivo chico con «Personas lejos» (cada parte queda de pocos píxeles)."""
        if not size:
            return ""
        w, h = size
        det = self.detection
        if not self.live and w >= 1280 and self._stream() == "sub" and (det.far or small
                                                                         or det.sensitivity > DEFAULT_SENSITIVITY):
            what = "con «Personas lejos»" if det.far else "con la gente del fondo"
            return (f"Este clip es de {w}x{h} y la máquina mira la calidad secundaria: lo que ves acá {what} puede no "
                    "pasar allá. Confirmalo en «● En vivo» o pasá la cámara a la calidad principal (Editar → Calidad "
                    "del video).")
        if self.live and det.far and w <= 640:
            return (f"La imagen llega en {w}x{h}: «Personas lejos» ayuda poco (cada parte queda de pocos píxeles). "
                    "Subí la calidad secundaria a 1280x720 en el grabador o usá la principal (Editar → Calidad del "
                    "video).")
        return ""

    def _ia_hints(self, st: dict) -> list[str]:
        """Avisos de la IA de movimientos de hurto en una cámara sin gestos (con gestos ya los dan `_gesture_hints`): si
        la postura se puede ver bien, porque la IA aprende con los esqueletos."""
        pose, g = st["pose"], self.rule_cfg.gestures
        if self.pose_missing:
            return [("En esta PC no está el modelo de postura (models/postura.onnx): la IA de movimientos de hurto no "
                     "se puede probar acá. Actualizá la app.")]
        out = []
        if pose["people"] >= 20 and pose["small_pct"] is not None and pose["small_pct"] > 50:
            out.append(f"El {pose['small_pct']:.0f} % de las veces las personas ocupan menos del "
                       f"{100.0 * g.min_person_px / REF_HEIGHT:.0f} % del alto de la imagen: a esas no se les mira la "
                       "postura y la IA de movimientos de hurto no las ve. Mejor una cámara más cerca o con más zoom.")
        if pose["tried"] >= 20 and pose["ok_pct"] is not None and pose["ok_pct"] < POSE_OK_MIN:
            out.append(f"La postura (hombros y caderas) se ve bien solo en el {pose['ok_pct']:.0f} % de las veces: la "
                       "IA de movimientos de hurto no va a andar bien en esta cámara (gente tapada, de espaldas, "
                       "cortada por el borde o cámara muy de arriba).")
        return out

    def _gesture_hints(self, st: dict, types: set[str]) -> list[str]:
        """Avisos de los gestos: si la postura se puede ver bien en esta cámara."""
        g, pose, out = self.rule_cfg.gestures, st["pose"], []
        if self.pose_missing:
            return ["En esta PC no está el modelo de postura (models/postura.onnx): los gestos no se pueden probar acá. "
                    "Actualizá la app."]
        if pose["people"] >= 20 and pose["small_pct"] is not None and pose["small_pct"] > 50:
            frac = 100.0 * g.min_person_px / REF_HEIGHT
            out.append(f"El {pose['small_pct']:.0f} % de las veces las personas ocupan menos del {frac:.0f} % del alto "
                       "de la imagen: a esas no se les mira la postura y los gestos no las ven. Mejor una cámara más "
                       "cerca o con más zoom.")
        if pose["tried"] >= 20 and pose["ok_pct"] is not None and pose["ok_pct"] < POSE_OK_MIN:
            out.append(f"La postura (hombros y caderas) se ve bien solo en el {pose['ok_pct']:.0f} % de las veces: los "
                       "gestos no van a andar bien en esta cámara (gente tapada por góndolas o mostrador, de espaldas, "
                       "cortada por el borde o cámara muy de arriba).")
        if g.require_take and self.zones and "shelf" not in types:
            out.append("Sin zona «Góndola», cualquier brazo estirado cuenta como «tomar algo» para los gestos: marcá las "
                       "góndolas para que den menos falsas alarmas.")
        return out


# píxeles no cuadrados («1080p lite»): core/imagen.py, lo mismo que usan las máquinas y las alarmas
from core.imagen import LITE_SIZES, pixel_aspect, square_pixels  # noqa: E402,F401


OBJECT_COLOR = (230, 80, 230)  # violeta: lo que ve YOLO además de las personas
POSE_COLOR = (0, 220, 255)  # esqueleto (amarillo; naranja el de la persona de la alerta), como en las fotos de alertas
GESTURE_COLOR = (255, 255, 0)  # celeste: la mano en el bolsillo, la ropa o la bolsa (rojo al llegar)


def gesture_text(sm: dict | None, frames: int, g: GestureConfig) -> str:
    """Lo medido de los gestos de una persona (`GestureEngine.summary`) en castellano: si se le vio la postura, las
    tomas, el máximo de segundos con la mano en el bolsillo, la ropa o la bolsa contra lo que pide, y por qué se
    descartó. `frames`: cuadros en que se la vio."""
    if not sm or not sm.get("frames"):
        return "no se le vio la postura (chica, tapada, de espaldas o cortada por el borde)"
    ok = sm["pose_ok_ratio"] * sm["frames"] / max(1, frames, sm["frames"])
    parts = [f"postura bien vista en el {100 * ok:.0f} % de los cuadros"]
    takes = sm.get("takes") or 0
    parts.append(f"{takes} toma{'s' if takes != 1 else ''} (brazo al estante y vuelta)" if takes else
                 "no se la vio tomar algo")
    need = {"bolsillo": g.pocket_seconds, "ropa": g.clothes_seconds, "bolsa": g.bag_seconds}
    dwell = []
    for gesture in g.enabled:
        region = REGION_OF[gesture]
        v = (sm.get("max_dwell") or {}).get(region, 0.0)
        if v > 0:
            dwell.append(f"{GESTURE_NAMES[gesture].lower()} hasta {v:.1f} s (pide {need[region]:g} s"
                         + (", el doble sin bolso a la vista" if region == "bolsa" else "") + ")")
    if not dwell:
        names = [REGION_PLACE[REGION_OF[x]] for x in g.enabled]
        dwell = ["nunca llevó la mano " + (" ni ".join(names) if len(names) <= 2 else
                                          ", ".join(names[:-1]) + " ni " + names[-1])]
    parts.append(", ".join(dwell))
    sup = sm.get("suppressed") or {}
    if sup:
        why = {**SUPPRESS_TEXT, "la toma fue hace mucho": f"la toma fue hace más de {g.take_window:g} s"}
        parts.append("descartado: " + ", ".join(why.get(k, k) for k in sorted(sup, key=lambda k: -sup[k])))
    return "; ".join(parts)


def ia_text(best: float | None, spec: dict) -> str:
    """Lo que midió la IA de movimientos de hurto de una persona, contra lo que pide para avisar."""
    if best is None:
        return "no se le vio la postura en suficientes cuadros para mirarla"
    need = float(spec.get("threshold", 0.5))
    times = int(spec.get("consecutive", 2))
    return (f"puntaje más alto {best:.2f} (avisa desde {need:.2f}"
            + (f", {times} veces seguidas" if times > 1 else "") + ")").replace(".", ",")


def ascii_text(text: str) -> str:
    """OpenCV no dibuja tildes: «ómnibus» → «omnibus»."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def wait_clock(secs: float | None) -> str:
    """Un tiempo de espera como reloj: «1:20»; de una hora o más, «1 h 12 min» (como la app, la web y el PDF)."""
    if secs is None:
        return "-"
    s = max(0, round(secs))
    if s < 3600:
        return f"{s // 60}:{s % 60:02d}"
    h, m = divmod(round(s / 60), 60)
    return f"{h} h {m} min"


def count_text(z: Zone, counts: dict | None) -> str | None:
    """Lo contado en una zona, corto para dibujarlo debajo de su nombre (sin tildes: OpenCV no las dibuja): la puerta
    con sentido, entraron y salieron; la de personal, cuántos empleados y qué parte del tiempo hubo alguien; las demás,
    cuántas personas entraron y cuántas hay ahora. Una zona que mide la espera suma un renglón (`\n`): cuántos clientes
    esperan ahora y el promedio y el máximo de las esperas (`MetricsCollector.live_counts`: «wait»)."""
    if not counts:
        return None
    c = (counts.get("zones") or {}).get(z.name)
    if c is None:
        return None
    if z.type == "staff":
        pct = c.get("covered_pct")
        return f"personal {c.get('staff', 0)}" + (f"  cubierto {pct:.0f}%" if pct is not None else "")
    if z.type == "table":  # la mesa: si está ocupada (y hace cuánto), y el promedio en la mesa y hasta la atención
        tb = c.get("table") or {}
        text = f"OCUPADA {wait_clock(tb.get('for_s'))}" if tb.get("busy") else "libre"
        if tb.get("avg_s") is not None:
            text += f"\nveces {tb.get('n', 0)}  prom {wait_clock(tb['avg_s'])}"
            if tb.get("attend_avg_s") is not None:
                text += f"  atencion {wait_clock(tb['attend_avg_s'])}"
        return text
    if z.type == "entrance" and z.direction:
        text = f"entraron {c.get('entries', 0)}  salieron {c.get('exits', 0)}"
    else:
        text = f"{'pasaron' if z.type == 'entrance' else 'entraron'} {c['visits']}"
        text += f"  ahora {c['now']}" if c.get("now") else ""
    w = c.get("wait") if z.wait else None
    if w is not None:
        text += f"\nesperando {w.get('now', 0)}"
        if w.get("avg_s") is not None:
            text += f"  prom {wait_clock(w['avg_s'])}  max {wait_clock(w.get('max_s'))}"
    return text


def draw_counts(out: np.ndarray, zones: list[Zone], counts: dict | None) -> None:
    """Debajo del nombre de cada zona, cuántas personas entraron (y cuántas hay ahora): se ve contar en la imagen."""
    h, w = out.shape[:2]
    thick = max(1, w // 640)
    for z in zones:
        text = count_text(z, counts)
        if not text:
            continue
        color = ZONE_COLORS.get(z.type, (200, 200, 200))
        lines = text.split("\n")  # (la espera va en un renglón aparte)
        size = 0.5 * thick
        step = 18 * thick
        x0 = int(min(p[0] for p in z.polygon) * w) + 4
        y = int(min(p[1] for p in z.polygon) * h) + 16 * thick + 20 * thick  # debajo del nombre
        y = min(max(y, 14), h - 26 - step * (len(lines) - 1))
        for line in lines:
            tw, th = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, size, thick)[0]
            x = max(2, min(x0, w - tw - 6))
            cv2.rectangle(out, (x - 3, y - th - 4), (x + tw + 3, y + 4), (0, 0, 0), -1)
            cv2.putText(out, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, thick, cv2.LINE_AA)
            y += step


IA_COLOR = (255, 255, 255)  # el puntaje de la IA de hurtos sobre el recuadro (en rojo si pasa el umbral)


def draw_ia(out: np.ndarray, tracks: list[Track], mark: dict) -> None:
    """Sobre el recuadro de cada persona, al lado de su número, el puntaje de la IA de movimientos de hurto («IA
    0,72», en rojo si pasa el umbral). `mark[id]["ia"]` = (puntaje, ¿pasa?); `tracks` en la escala de `out`."""
    w = out.shape[1]
    thick = max(1, w // 640)
    size = 0.5 * thick
    for t in tracks:
        ia = (mark.get(t.id) or {}).get("ia")
        if ia is None or not t.visible or t.label == VEHICLE_LABEL:
            continue
        score, over = ia
        text = ia_score_text(score)
        x1, y1 = int(t.bbox[0]), int(t.bbox[1])
        num = cv2.getTextSize(f"#{t.id}", cv2.FONT_HERSHEY_SIMPLEX, size, thick)[0][0]  # (el número va a la izquierda)
        tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, thick)[0][0]
        x, y = max(2, min(x1 + num + 8, w - tw - 4)), max(12, y1 - 6)
        cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 255) if over else IA_COLOR, thick,
                    cv2.LINE_AA)


def draw(img: np.ndarray, s: Sample, mark: dict, zones: list[Zone], cfg: RuleConfig, hot: list[TestAlert],
         clock: str, objects: bool = True, counts: dict | None = None) -> np.ndarray:
    """El cuadro con zonas, personas (número, puntaje de la IA de hurtos, segundos en cada zona, gestos), lo demás que
    ve YOLO, el cartel de la alerta de prueba y, con `counts` (`MetricsCollector.live_counts`), cuántas personas
    entraron a cada zona."""
    h, w = img.shape[:2]
    k = w / s.size[0] if s.size[0] else 1.0
    # el esqueleto se dibuja aparte, escalado como los recuadros (los puntos están en la escala del cuadro original)
    tracks = [replace(t, bbox=tuple(v * k for v in t.bbox), kpts=None) for t in s.tracks]
    ev = SimpleNamespace(track_id=hot[-1].track_id, type=hot[-1].type) if hot else None
    out = annotate(img, zones, tracks, ev)  # zonas, recuadros, #número y el cartel de la alerta
    draw_counts(out, zones, counts)
    for t in s.tracks:
        if t.kpts is None or not t.visible:
            continue
        hands = ((mark.get(t.id) or {}).get("gesture") or {}).get("hands") or {}
        hot_wrists = tuple(9 if side == "izq" else 10 for side, (region, _d, _n) in hands.items()
                           if GESTURE_OF[region] in cfg.gestures.enabled)  # en rojo, la mano que cuenta para un gesto
        draw_pose(out, t.kpts, (0, 140, 255) if ev is not None and t.id == ev.track_id else POSE_COLOR, scale=k,
                  hot_wrists=hot_wrists)
    for label, conf, box in (s.objects if objects else []):
        x1, y1, x2, y2 = (int(v * k) for v in box)
        cv2.rectangle(out, (x1, y1), (x2, y2), OBJECT_COLOR, 1)
        text = f"{ascii_text(label)} {conf * 100:.0f}%"
        cv2.putText(out, text, (x1 + 2, max(10, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, (x1 + 2, max(10, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, OBJECT_COLOR, 1, cv2.LINE_AA)
    draw_ia(out, tracks, mark)
    limits = {"checkout": cfg.checkout_min_seconds, "restricted": cfg.restricted_min_seconds}
    bar = 22
    people = 0
    for t in tracks:
        if not t.visible or t.label == VEHICLE_LABEL:
            continue
        people += 1
        info = mark.get(t.id) or {}
        lines = []
        for _name, ztype, dwell in info.get("zones", []):
            lim = limits.get(ztype)
            lines.append((f"{ZONE_TEXT.get(ztype, ztype)} {dwell:.0f}" + (f"/{lim:.0f}s" if lim else "s"),
                          (0, 0, 255) if lim and dwell >= lim else (0, 255, 255)))
        g = info.get("gesture") or {}
        for _side, (region, dwell, need) in sorted((g.get("hands") or {}).items()):
            if GESTURE_OF[region] in cfg.gestures.enabled:
                lines.append((f"{REGION_TEXT[region]} {dwell:.1f}/{need:.1f}s",
                              (0, 0, 255) if dwell + 1e-6 >= need else GESTURE_COLOR))
        if g.get("reaching"):
            lines.append(("ESTIRA EL BRAZO", GESTURE_COLOR))
        if g.get("took_ago") is not None:  # tomó algo hace poco (brazo al estante y vuelta): cuenta para los gestos
            lines.append((f"TOMO ALGO ({g['took_ago']:.0f}s)", (0, 140, 255)))
        elif info.get("took") and SALIDA_SIN_PAGO in cfg.enabled_rules:
            lines.append(("TOMO ALGO", (0, 140, 255)))
        x1, _y1, _x2, y2 = (int(v) for v in t.bbox)
        y = min(y2 + 14, h - bar - 14 * len(lines))
        for text, color in lines:
            tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
            x = max(2, min(x1 + 2, w - tw - 4))  # que no se salga de la imagen
            cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            y += 14
    cv2.rectangle(out, (0, h - bar), (w, h), (0, 0, 0), -1)
    text = f"{clock}   personas: {people}"
    if hot:
        text += f"   ALERTA DE PRUEBA: {ALERT_TEXT.get(hot[-1].type, hot[-1].type)} #{hot[-1].track_id}"
    text = ascii_text(text)
    size = 0.5
    tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, 1)[0][0]
    if tw > w - 12:  # que entre en la barra (los carteles de los gestos son largos)
        size *= (w - 12) / tw
    cv2.putText(out, text, (6, h - 7), cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), 1, cv2.LINE_AA)
    return out
