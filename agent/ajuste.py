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

from core.analyzer import annotate
from core.detectors import Detection, Detector
from core.escena import COCO_ES, OBJECT_CONF, light
from core.gestos import (GESTURE_NAMES, GESTURE_OF, GESTURE_TYPES, GestureConfig, MANO_BOLSA, MANO_BOLSILLO, MANO_ROPA,
                         REF_HEIGHT, REGION_OF, body_of)
from core.metrics import door_counts
from core.pisadas import FootMap
from core.pose import BAG_CLASSES, build_pose, look
from core.pose import draw as draw_pose
from core.rules import BAG_LABELS, SALIDA_SIN_PAGO, VEHICLE_LABEL, RuleConfig, RuleEngine, Zone, parse_config
from core.scorer import Scorer
from core.tracker import CentroidTracker, Track

from .cameras import build_camera
from .worker import MediaClock

log = logging.getLogger("vigia.ajuste")

WIDTH = 640  # los cuadros se guardan achicados (JPEG, ~30 KB): 10 min a 5 fps ≈ 90 MB
JPEG_QUALITY = 60
MIN_GESTURE_FPS = 5.0  # con gestos, la máquina analiza a 5 cuadros por segundo o más
LIVE_KEEP_SECONDS = 600.0  # en vivo se recuerdan (y se recalculan) los últimos 10 minutos
MAX_VIDEO_SECONDS = 30 * 60  # tope entre todos los videos de una prueba
SHORT_TRACK_SECONDS = 2.0
SMALL_PERSON = 0.10  # persona de menos del 10 % del alto de la imagen: el detector falla más
BANNER_SECONDS = 2.0  # cuánto se ve el cartel de una alerta de prueba
ZONE_TEXT = {"shelf": "gondola", "checkout": "caja", "exit": "salida", "restricted": "restringida",
             "entrance": "entrada", "pump": "surtidor", "shop": "tienda", "staff": "personal",
             "aisle": "pasillo"}
ALERT_TEXT = {"salida_sin_pago": "SALIDA SIN PAGO", "zona_restringida": "ZONA RESTRINGIDA",
              "fuera_de_horario": "FUERA DE HORARIO", MANO_BOLSILLO: "POSIBLE OCULTAMIENTO (BOLSILLO)",
              MANO_ROPA: "POSIBLE OCULTAMIENTO (ROPA)", MANO_BOLSA: "POSIBLE OCULTAMIENTO (BOLSA)"}
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


def gestures_blocked(analysis: dict) -> str | None:
    """Por qué la cámara no puede mirar gestos aunque se elijan (como el servidor, `camera_analysis`), o None."""
    if (analysis.get("profile") or "general") == "metricas":
        return "la cámara es del tipo «Solo métricas»"
    plans = analysis.get("plans")
    if plans is not None and "hurto" not in plans:
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
        self.detector = detector
        # el modelo entrenado con las etiquetas del cliente (si hay): dice qué alertas descartaría, como en las máquinas
        self.scorer = Scorer.maybe(cfg.get("filter_model"))
        self.videos = [str(v) for v in videos or []]
        self.live = not self.videos
        self.process_fps = float(cfg.get("process_fps") or 5.0)
        self.keep_seconds = keep_seconds
        self._lock = threading.RLock()
        self.samples: list[Sample] = []
        self.marks: list[dict] = []  # por cuadro: zonas y segundos de cada persona (RuleEngine.progress)
        self.base = 0  # número del primer cuadro guardado (en vivo se descartan los viejos)
        self.alerts: list[TestAlert] = []
        # lo medido de los gestos de cada persona ((video, número) → GestureEngine.summary), para explicar
        self.gsum: dict[tuple[int, int], dict] = {}
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
        with self._lock:
            self.analysis = analysis
            self.zones, self.rule_cfg = zones, rule_cfg
            self.metrics_only = metrics_only
            if rule_cfg.gestures.enabled and not metrics_only:
                # con gestos la máquina analiza a 5 cuadros por segundo o más (camera_fps del servidor): probar igual
                self.process_fps = max(MIN_GESTURE_FPS, self.process_fps)
            self._reevaluate()

    def set_times(self, times: dict[str, float], gestures=_KEEP) -> bool:
        """Probar otros tiempos («tomó algo», caja, restringida) sin guardarlos; con `gestures`, también otros
        gestos (como `set_gestures`, en la misma vuelta). Devuelve `needs_pose()`."""
        analysis = {**self.analysis, "rules": {**(self.analysis.get("rules") or {}), **times}}
        self.set_analysis(analysis if gestures is _KEEP else with_gestures(analysis, gestures))
        return self.needs_pose()

    def set_gestures(self, gestures: dict | None) -> bool:
        """Probar otros gestos o segundos (lo de `GestureConfig.to_dict`; None o sin "enabled" = apagados) sin
        guardarlos: se recalcula todo con la postura ya mirada, sin volver a estimarla. Devuelve `needs_pose()`."""
        self.set_analysis(with_gestures(self.analysis, gestures))
        return self.needs_pose()

    @property
    def gestures_on(self) -> bool:
        return bool(self.rule_cfg.gestures.enabled)

    def needs_pose(self) -> bool:
        """Hay gestos pero lo ya analizado se hizo sin mirar la postura: para verlos hay que volver a analizar (en vivo,
        se ven desde ahora)."""
        with self._lock:
            return self.gestures_on and not self.pose_missing and any(not s.posed for s in self.samples)

    def _reevaluate(self) -> None:
        engine, seg = None, None
        alerts: list[TestAlert] = []
        marks: list[dict] = []
        gsum: dict[tuple[int, int], dict] = {}
        for i, s in enumerate(self.samples):
            if s.seg != seg:  # cada video empieza de cero
                engine, seg = RuleEngine(self.zones, self.rule_cfg), s.seg
            alerts += self._feed(engine, s, self.base + i, marks, gsum)
        self.engine = engine or RuleEngine(self.zones, self.rule_cfg)
        self._engine_seg = seg if seg is not None else -1
        self.alerts, self.marks, self.gsum = alerts, marks, gsum

    def _feed(self, engine: RuleEngine, s: Sample, index: int, marks: list[dict],
              gsum: dict[tuple[int, int], dict]) -> list[TestAlert]:
        gest = engine.gestures
        bags = [(label, box) for label, _conf, box in s.objects if label in BAG_LABELS] if gest is not None else []
        evs = engine.update(s.tracks, s.ts, s.size, s.wall, [Detection(box, label) for label, box in bags])
        prog = engine.progress(s.ts)
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
            # Hik-Connect manda el video en ráfagas: se usa el reloj del video, como en las máquinas
            clock = MediaClock(cap) if cam.brand == "hikconnect" else None
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
        """Personas y lo demás que ve YOLO, en una sola pasada. Con gestos, los bolsos salen como en las máquinas
        (core/pose.detect_with_bags: la clase de más puntaje entre mochila, cartera y valija, desde 0,3) y lo demás
        aparte, desde 0,45: así «mano en la bolsa» ve acá los mismos bolsos que en la máquina."""
        if not self.extra:
            return self.detector.detect(frame), []
        if gest and hasattr(self.detector, "detect_groups"):
            rest = {c: n for c, n in self.extra.items() if c not in BAG_CLASSES}
            dets, (objs, bags) = self.detector.detect_groups(frame, [(rest, OBJECT_CONF), (BAG_CLASSES, BAG_CONF)])
            return dets, objs + bags
        dets, objs = self.detector.detect_also(frame, self.extra, BAG_CONF if gest else OBJECT_CONF)
        if gest:  # un detector sin grupos: los bolsos con la confianza de las máquinas; lo demás, como siempre
            objs = [o for o in objs if o.conf >= OBJECT_CONF or o.label in BAG_LABELS]
        return dets, objs

    def _analyze(self, frame: np.ndarray, ts: float, wall: datetime | None, seg: int) -> None:
        gest = self.gestures_on
        dets, objs = self._detect(frame, gest)
        tracks = self.tracker.update(dets, ts, frame)
        # el mínimo es relativo al alto del cuadro: lo mismo que en la máquina aunque se pruebe con el stream principal
        min_px = self.rule_cfg.gestures.min_height(frame.shape[0])
        posed = gest and self._pose_for() is not None
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
            self.samples.append(sample)
            self.alerts += self._feed(self.engine, sample, self.base + len(self.samples) - 1, self.marks, self.gsum)
            if self.live:
                self._trim()

    def _trim(self) -> None:
        last = self.samples[-1].ts
        if last - self.samples[0].ts <= self.keep_seconds + 30:  # se recorta de a tandas
            return
        cut = next(i for i, s in enumerate(self.samples) if last - s.ts <= self.keep_seconds)
        del self.samples[:cut]
        del self.marks[:cut]
        self.base += cut
        self.alerts = [a for a in self.alerts if a.index >= self.base]
        if self.gsum:
            keep = {(s.seg, t.id) for s in self.samples for t in s.tracks}
            self.gsum = {k: v for k, v in self.gsum.items() if k in keep}

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
            hot = [a for a in self.alerts if a.seg == s.seg and 0 <= s.ts - a.ts <= BANNER_SECONDS]
        img = cv2.imdecode(np.frombuffer(s.jpg, np.uint8), cv2.IMREAD_COLOR)
        return None if img is None else draw(img, s, mark, zones, cfg, hot, self.clock_text(s), objects)

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
                "door": self.door(), "gestures": self.gestures_on, "pose": pose}

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
        origin = datetime(2000, 1, 1, 12)  # videos sin hora real: el horario del local no se aplica
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
        bolsa contra lo que pide, y por qué se descartó) contra lo que pide cada regla. Para «Probar clip»."""
        with self._lock:
            rows = [(s, m) for s, m in zip(self.samples, self.marks) if s.seg == seg]
            alerts = [a for a in self.alerts if a.seg == seg]
            zones, cfg = list(self.zones), self.rule_cfg
            gsum = {tid: v for (sg, tid), v in self.gsum.items() if sg == seg}
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
            out.append("La cámara no tiene zonas: ninguna regla puede saltar. Dibujalas en «Zonas…»."
                       + (" Los gestos sí miran sin zonas." if g.enabled else ""))
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
            out.append(text + ".")
        if not alerts and zones:
            if took and SALIDA_SIN_PAGO in rules:
                if not any(t == "exit" for t in types.values()):
                    out.append("«Tomó algo» solo da alerta («salida sin pago») si después pisa una zona Salida sin pasar "
                               "por Caja, y esta cámara no tiene Salida.")
                else:
                    out.append("«Tomó algo» da alerta si después pisa la Salida sin pasar por Caja: no llegó a la Salida "
                               "en este video.")
        if g.enabled and not any(a.type in g.enabled for a in alerts):
            if self.pose_missing:
                out.append("En esta PC no está el modelo de postura: los gestos no se probaron (actualizá la app).")
            elif any(not s.posed for s, _m in rows):
                out.append("Este video se analizó sin mirar la postura: para ver los gestos, volvé a analizarlo "
                           "(«▶ Videos guardados»).")
            elif not any((gsum.get(tid) or {}).get("frames") for tid in people):
                out.append("No se le vio la postura a nadie (chicas, tapadas, de espaldas o cortadas por el borde): los "
                           "gestos no pueden ver nada en este video. Mirá «Cómo anda la cámara».")
            elif g.require_take and not any((gsum.get(tid) or {}).get("takes") for tid in people):
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
                       "dónde caminan los que entran en «Zonas…» → «Sentido».")
        if self.metrics_only or (not rules and not g.enabled):
            out.append("Esta cámara no da alertas: es «solo métricas» (el cliente no tiene el plan de hurto o el tipo de "
                       "cámara no usa reglas).")
        elif not self.zones:
            out.append("No hay zonas: no se puede disparar ninguna regla. Dibujá góndolas, caja y salida con «Zonas…»."
                       + (" Los gestos sí miran sin zonas (con las góndolas marcadas dan menos falsas alarmas)."
                          if g.enabled else ""))
        else:
            if SALIDA_SIN_PAGO in rules and "shelf" not in types:
                out.append("Sin zona «Góndola» no hay «tomó algo» (ni «salida sin pago»).")
            if SALIDA_SIN_PAGO in rules:
                if "exit" not in types:
                    out.append("Sin zona «Salida», «salida sin pago» no se puede disparar.")
                elif "shelf" in types and "checkout" not in types:
                    out.append("Sin zona «Caja», cualquiera que pase por una góndola y salga dispara «salida sin pago» "
                               "(aunque haya pagado). Si la caja se ve en otra cámara, mejor elegir otro tipo de cámara.")
        if st["detections"] >= 20:
            if st["height_pct"] < SMALL_PERSON * 100:
                out.append(f"Las personas se ven chicas (en promedio {st['height_pct']:.0f} % del alto de la imagen): el "
                           "detector falla más. Mejor una cámara más cerca o el stream principal.")
            if st["conf"] < 0.5:
                out.append(f"El detector duda (confianza media {st['conf']:.2f} de 1): poca luz, contraluz, reflejos o "
                           "cámara muy de arriba. Probala también de noche.")
        if g.enabled:
            out += self._gesture_hints(st, types)
        if st["frames"] >= 20 and st["night_pct"] > 50:
            out.append("La imagen está en blanco y negro (infrarrojo, de noche): el detector ve bastante menos"
                       + (" y la postura peor (los gestos fallan más)" if g.enabled else "")
                       + ". Probala también de día antes de aprobar.")
        elif st["frames"] >= 20 and st["dark_pct"] > 50:
            out.append("La imagen está oscura: el detector ve menos. Si se puede, más luz o subir el brillo en el grabador.")
        if st["persons"] >= 5 and st["short"] / st["persons"] > 0.3:
            out.append(f"{st['short']} de {st['persons']} personas se vieron menos de 2 s: detecciones falsas (reflejos, "
                       "maniquíes, carteles) o alguien que cambia de número. Mirá el video: si una persona cambia de "
                       "número, las reglas de tiempo fallan.")
        if self.live:
            if st["seconds"] >= 30 and st["fps"] is not None and st["fps"] < 0.7 * self.process_fps:
                out.append(f"Se analizan {st['fps']:.1f} cuadros por segundo de {self.process_fps:.0f}: esta PC no da "
                           "abasto o el video llega cortado.")
            if self.cuts >= 2:
                out.append(f"El video se cortó {self.cuts} veces.")
            if st["seconds"] >= 60 and st["persons"] == 0:
                out.append("Todavía no pasó nadie: dejala correr hasta que haya gente, o probá con un video.")
        if not out and st["detections"] >= 20:
            out.append("No se ven problemas. Antes de aprobar, revisá que cada alerta de prueba tenga sentido.")
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


def ascii_text(text: str) -> str:
    """OpenCV no dibuja tildes: «ómnibus» → «omnibus»."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def draw(img: np.ndarray, s: Sample, mark: dict, zones: list[Zone], cfg: RuleConfig, hot: list[TestAlert],
         clock: str, objects: bool = True) -> np.ndarray:
    h, w = img.shape[:2]
    k = w / s.size[0] if s.size[0] else 1.0
    # el esqueleto se dibuja aparte, escalado como los recuadros (los puntos están en la escala del cuadro original)
    tracks = [replace(t, bbox=tuple(v * k for v in t.bbox), kpts=None) for t in s.tracks]
    ev = SimpleNamespace(track_id=hot[-1].track_id, type=hot[-1].type) if hot else None
    out = annotate(img, zones, tracks, ev)  # zonas, recuadros, #número y el cartel de la alerta
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
