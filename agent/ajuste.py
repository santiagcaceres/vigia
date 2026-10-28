"""«Ajustar cámara» en la app de escritorio: se analiza UNA cámara en esta PC para ponerla a punto antes de mandarla
a las máquinas de análisis. Sirve en vivo o con videos grabados (por ejemplo, hurtos reales de esa cámara).

- Se ve lo que ve el análisis: los cuadros que se analizan (a los fps de la cámara), las zonas, cada persona con su
  número y, por zona, cuánto lleva contra el tiempo de la regla («gondola 32/45s»), y si ya «tomó algo».
- Las alertas son **de prueba**: quedan en la ventana; no van a los empleados ni al cliente ni sirven para entrenar.
- Cambiar zonas o tiempos recalcula al instante todo lo analizado (los últimos 10 min en vivo, o los videos
  enteros): detectar y seguir personas (lo caro) no depende de las zonas ni de los tiempos, así que se guarda lo
  que salió de cada cuadro y solo se vuelven a pasar las reglas.
- `stats` y `hints`: números y avisos para decidir si la cámara sirve (tamaño de las personas, confianza del
  detector, apariciones breves, cuadros por segundo, cortes, zonas que faltan).
- YOLO completo: en la misma pasada del modelo se ven también los objetos (mochilas, carteras, botellas…). Se
  dibujan y se cuentan, pero no entran al seguimiento ni a las reglas: las personas son las mismas que en las
  máquinas (`OnnxYoloDetector.detect_also`).
"""

from __future__ import annotations

import logging
import threading
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from core.analyzer import annotate
from core.detectors import Detector
from core.escena import COCO_ES, OBJECT_CONF, light
from core.rules import MERODEO, SALIDA_SIN_PAGO, VEHICLE_LABEL, RuleConfig, RuleEngine, Zone, parse_config
from core.scorer import Scorer
from core.tracker import CentroidTracker, Track

from .cameras import build_camera
from .worker import MediaClock

log = logging.getLogger("vigia.ajuste")

WIDTH = 640  # los cuadros se guardan achicados (JPEG, ~30 KB): 10 min a 5 fps ≈ 90 MB
JPEG_QUALITY = 60
LIVE_KEEP_SECONDS = 600.0  # en vivo se recuerdan (y se recalculan) los últimos 10 minutos
MAX_VIDEO_SECONDS = 30 * 60  # tope entre todos los videos de una prueba
SHORT_TRACK_SECONDS = 2.0
SMALL_PERSON = 0.10  # persona de menos del 10 % del alto de la imagen: el detector falla más
BANNER_SECONDS = 2.0  # cuánto se ve el cartel de una alerta de prueba
ZONE_TEXT = {"shelf": "gondola", "checkout": "caja", "exit": "salida", "restricted": "restringida",
             "entrance": "entrada", "pump": "surtidor", "shop": "tienda"}
ALERT_TEXT = {"salida_sin_pago": "SALIDA SIN PAGO", "merodeo": "MERODEO", "zona_restringida": "ZONA RESTRINGIDA",
              "fuera_de_horario": "FUERA DE HORARIO"}


@dataclass
class Sample:
    """Un cuadro analizado: lo necesario para volver a pasar las reglas y para mostrarlo."""
    seg: int  # qué video (en vivo: 0)
    ts: float  # segundos desde el inicio del video (o desde que se abrió la cámara)
    wall: datetime | None  # hora real (solo en vivo)
    jpg: bytes
    size: tuple[int, int]  # tamaño del cuadro original: los recuadros están en esa escala
    tracks: list[Track]  # copia de las personas de ese cuadro (sin el histograma de color)
    objects: list[tuple[str, float, tuple]] = field(default_factory=list)  # (qué, confianza, recuadro): solo se muestran


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
                 keep_seconds: float = LIVE_KEEP_SECONDS):
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
        self.engine: RuleEngine | None = None
        self._engine_seg = -1
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
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self.run, name=f"ajuste-{self.id}", daemon=True)

    # ---------- zonas y reglas ----------
    def set_analysis(self, analysis: dict) -> None:
        """Zonas y reglas nuevas: se recalcula todo lo ya analizado (sin volver a detectar). ValueError si están mal."""
        zones, rule_cfg = parse_config(analysis)
        with self._lock:
            self.analysis = analysis
            self.zones, self.rule_cfg = zones, rule_cfg
            self.metrics_only = bool(analysis.get("metrics_only"))
            self._reevaluate()

    def set_times(self, times: dict[str, float]) -> None:
        """Probar otros tiempos (merodeo, «tomó algo», caja, restringida) sin guardarlos."""
        self.set_analysis({**self.analysis, "rules": {**(self.analysis.get("rules") or {}), **times}})

    def _reevaluate(self) -> None:
        engine, seg = None, None
        alerts: list[TestAlert] = []
        marks: list[dict] = []
        for i, s in enumerate(self.samples):
            if s.seg != seg:  # cada video empieza de cero
                engine, seg = RuleEngine(self.zones, self.rule_cfg), s.seg
            alerts += self._feed(engine, s, self.base + i, marks)
        self.engine = engine or RuleEngine(self.zones, self.rule_cfg)
        self._engine_seg = seg if seg is not None else -1
        self.alerts, self.marks = alerts, marks

    def _feed(self, engine: RuleEngine, s: Sample, index: int, marks: list[dict]) -> list[TestAlert]:
        evs = engine.update(s.tracks, s.ts, s.size, s.wall)
        marks.append(engine.progress(s.ts))
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
                    last = last_ts = ts
                    self._analyze(frame, ts, datetime.now(cam.tz), 0)
            finally:
                cap.release()
            if not self._stop.is_set():
                self.cuts += 1
                self.state = "conectando"
                self._stop.wait(2.0)
        self.state = "detenido"

    def _analyze(self, frame: np.ndarray, ts: float, wall: datetime | None, seg: int) -> None:
        if self.extra:
            dets, objs = self.detector.detect_also(frame, self.extra, OBJECT_CONF)
        else:
            dets, objs = self.detector.detect(frame), []
        tracks = self.tracker.update(dets, ts, frame)
        h, w = frame.shape[:2]
        small = frame if w <= WIDTH else cv2.resize(frame, (WIDTH, int(h * WIDTH / w)), interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        sample = Sample(seg, ts, wall, jpg.tobytes() if ok else b"", (w, h), [replace(t, hist=None) for t in tracks],
                        [(o.label, o.conf, o.bbox) for o in objs])
        dark, night = light(small)
        with self._lock:
            self._dark += dark
            self._night += night
            self._objects.update({o[0] for o in sample.objects})
            self._measure(dets, sample, h)
            if seg != self._engine_seg or self.engine is None:
                self.engine, self._engine_seg = RuleEngine(self.zones, self.rule_cfg), seg
            self.samples.append(sample)
            self.alerts += self._feed(self.engine, sample, self.base + len(self.samples) - 1, self.marks)
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
        return {"frames": n, "seconds": n / self.process_fps, "people_pct": 100.0 * self._n_people / n if n else 0.0,
                "persons": persons, "short": short, "detections": det, "objects": objects,
                "dark_pct": dark_pct, "night_pct": night_pct,
                "height_pct": 100.0 * self._det_h / det if det else None,
                "small_pct": 100.0 * self._det_small / det if det else None,
                "conf": self._det_conf / det if det else None,
                "fps": len(self._times) / 10.0 if self.live else None, "cuts": self.cuts, "alerts": alerts,
                "dropped": dropped,
                "per_hour": alerts / (span / 3600.0) if self.live and span >= 300 else None}

    def hints(self) -> list[str]:
        """Avisos para decidir si la cámara está lista. Dicen qué mirar; no reemplazan mirar las alertas."""
        st = self.stats()
        types = {z.type for z in self.zones}
        rules = set(self.rule_cfg.enabled_rules)
        out: list[str] = []
        if self.metrics_only or not rules:
            out.append("Esta cámara no da alertas: es «solo métricas» (el cliente no tiene el plan de hurto o el tipo de "
                       "cámara no usa reglas).")
        elif not self.zones:
            out.append("No hay zonas: no se puede disparar ninguna regla. Dibujá góndolas, caja y salida con «Zonas…».")
        else:
            if MERODEO in rules and "shelf" not in types:
                out.append("Sin zona «Góndola» no hay «merodeo» ni «tomó algo».")
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
        if st["frames"] >= 20 and st["night_pct"] > 50:
            out.append("La imagen está en blanco y negro (infrarrojo, de noche): el detector ve bastante menos. Probala "
                       "también de día antes de aprobar.")
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


# resoluciones «lite» de los grabadores (Hikvision «1080P Lite», Dahua «1080N», etc.): la mitad del ancho, con píxeles
# el doble de anchos. Muchos videos exportados no lo declaran (el de Góndolas ventana de Minas, 27 oct).
LITE_SIZES = {(960, 1080), (640, 720), (1280, 1440)}


def pixel_aspect(cap: cv2.VideoCapture) -> float | None:
    """Forma de los píxeles del video (ancho/alto) si no son cuadrados. Los grabadores graban «1080p lite» como
    960x1080 con píxeles el doble de anchos: OpenCV lo ignora y las personas quedan flacas (el detector falla más y
    la imagen no se parece a la del vivo). Si el archivo no lo declara, se reconoce por el tamaño."""
    try:
        num, den = cap.get(cv2.CAP_PROP_SAR_NUM), cap.get(cv2.CAP_PROP_SAR_DEN)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    except Exception:  # noqa: BLE001 - backends sin ese dato
        return None
    if num > 0 and den > 0 and abs(num / den - 1) > 0.05:
        return num / den
    if size in LITE_SIZES:
        return 2.0
    return None


def square_pixels(frame: np.ndarray, sar: float) -> np.ndarray:
    """Estira el cuadro para que se vea como en el grabador (960x1080 con píxeles 2:1 → 1920x1080)."""
    h, w = frame.shape[:2]
    if sar > 1:
        return cv2.resize(frame, (int(round(w * sar)), h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(frame, (w, int(round(h / sar))), interpolation=cv2.INTER_LINEAR)


OBJECT_COLOR = (230, 80, 230)  # violeta: lo que ve YOLO además de las personas


def ascii_text(text: str) -> str:
    """OpenCV no dibuja tildes: «ómnibus» → «omnibus»."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def draw(img: np.ndarray, s: Sample, mark: dict, zones: list[Zone], cfg: RuleConfig, hot: list[TestAlert],
         clock: str, objects: bool = True) -> np.ndarray:
    h, w = img.shape[:2]
    k = w / s.size[0] if s.size[0] else 1.0
    tracks = [replace(t, bbox=tuple(v * k for v in t.bbox)) for t in s.tracks]
    ev = SimpleNamespace(track_id=hot[-1].track_id, type=hot[-1].type) if hot else None
    out = annotate(img, zones, tracks, ev)  # zonas, recuadros, #número y el cartel de la alerta
    for label, conf, box in (s.objects if objects else []):
        x1, y1, x2, y2 = (int(v * k) for v in box)
        cv2.rectangle(out, (x1, y1), (x2, y2), OBJECT_COLOR, 1)
        text = f"{ascii_text(label)} {conf * 100:.0f}%"
        cv2.putText(out, text, (x1 + 2, max(10, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, (x1 + 2, max(10, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, OBJECT_COLOR, 1, cv2.LINE_AA)
    limits = {"shelf": cfg.loiter_seconds if MERODEO in cfg.enabled_rules else None,
              "checkout": cfg.checkout_min_seconds, "restricted": cfg.restricted_min_seconds}
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
        if info.get("took") and SALIDA_SIN_PAGO in cfg.enabled_rules:
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
    cv2.putText(out, text, (6, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out
