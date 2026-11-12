"""El reproductor de la pestaña «Alertas» (pedido del dueño, 11 nov: «cuando miro las alertas, que se muestre con la IA, y
en el reproductor una barra para ver por dónde va y dónde sería el hurto, y opciones de iniciar y pausar el video; sacá
la opción de ver en el reproductor»).

- Al elegir una alerta se ve su foto; el clip se baja solo y se analiza **en esta PC** con la misma IA que «Entrenar IA
  hurtos» (agent/ia_clips.py: YOLO con la sensibilidad de la cámara, seguimiento, postura y bolsos), de a poco y en
  otro hilo: los recuadros, los esqueletos y las zonas de la cámara aparecen apenas esa parte está analizada. No guarda
  ni manda nada (como «Ver con la IA de esta PC»).
- La barra (`Timeline`): por dónde va (el círculo), hasta dónde llegó la IA (la parte más oscura) y el momento de la
  alerta (en rojo). Clic o arrastrar = ir ahí. «▶ Reproducir» / «❚❚ Pausa», «−5 s» / «+5 s»; con el teclado, espacio y
  las flechas izquierda y derecha. Al abrir el clip queda unos segundos antes de la alerta: con ▶ se ve el momento.
- Dónde está la alerta en el clip (`alert_offset`): el segundo que manda la máquina (`clip_offset_s`, desde que las
  máquinas se actualizan); en las alarmas e importados, `video_ts` (ya es el segundo del clip); si no, se estima con
  cuánto llevaba la persona a la vista (`track_age_s`: el clip arranca cuando apareció, con 3 s de margen) y, cuando la
  IA analizó ese rato, se ajusta buscando el recuadro de la persona de la alerta (`refine_offset`).
"""

from __future__ import annotations

import bisect
import logging
import queue
import threading
import time
from collections import OrderedDict
from collections.abc import Callable

import cv2
import numpy as np

from core.pose import draw as draw_pose

from .entrenar_ia import FrameReader, ascii_text, fit, fmt_t

log = logging.getLogger("vigia.alerta_video")

# cómo arma el clip la máquina (core/analyzer.py y los valores del servidor): desde que apareció la persona, con 3 s de
# margen, nunca menos de 6 s antes de la alerta ni más de 3 minutos en total
CLIP_PRE, CLIP_LEAD, CLIP_MAX, CLIP_POST = 6.0, 3.0, 180.0, 6.0
LEAD_IN = 4.0  # al abrir el clip, cuántos segundos antes de la alerta queda
ALERT_SHOW = 2.0  # el cartel «ALERTA» se ve hasta 2 s antes y después del momento
STEP_S = 5.0  # «−5 s» / «+5 s» y las flechas
MAX_DECODE = (960, 540)  # el cuadro se decodifica a lo sumo a este tamaño (después se ajusta al lugar)
REFINE_WINDOW = 8.0  # se busca a la persona de la alerta hasta 8 s antes y después de lo estimado
REFINE_IOU = 0.35
POLL_MS = 40
CACHE = 6  # lo que vio la IA en las últimas alertas (volver a una no la vuelve a analizar)
GREEN, RED, VIOLET, WHITE = (0, 200, 0), (40, 40, 230), (230, 80, 230), (255, 255, 255)
ZONE_BGR = {"shelf": (0, 200, 255), "aisle": (255, 255, 120), "table": (40, 110, 200), "checkout": (0, 200, 0), "exit": (255, 150, 0),
            "restricted": (48, 48, 255), "entrance": (255, 0, 200), "pump": (0, 120, 255), "shop": (200, 200, 0),
            "staff": (138, 138, 138)}
SOURCES_IN_CLIP = ("alarma", "importado", "ajuste", "prueba")  # su `video_ts` ya es el segundo del clip


def _num(x) -> float | None:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def alert_offset(e: dict, duration: float | None) -> tuple[float, bool]:
    """(segundo del clip donde está la alerta, ¿es exacto?). Exacto: lo mandó la máquina, o es una alarma o un video
    importado. Si no, una estimación que `refine_offset` corrige cuando la IA de esta PC analizó ese rato."""
    top = max(0.0, (duration or 0.0) - 0.1) if duration else None

    def clamp(v: float) -> float:
        v = max(0.0, v)
        return min(v, top) if top is not None else v
    off = _num(e.get("clip_offset_s"))
    if off is not None:
        return clamp(off), True
    vts = _num(e.get("video_ts"))
    if e.get("source") in SOURCES_IN_CLIP and vts is not None:
        return clamp(vts), True
    age = _num((e.get("features") or {}).get("track_age_s"))
    back = max(CLIP_PRE, (age or 0.0) + CLIP_LEAD)
    return clamp(min(back, CLIP_MAX - CLIP_POST)), False


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _scaled_bbox(bbox, frame_size, rec: dict):
    """El recuadro de la alerta (en el cuadro que analizó la máquina) en las medidas del cuadro de este registro."""
    if not bbox or len(bbox) != 4 or not frame_size or len(frame_size) != 2:
        return None
    fw, fh = (_num(v) or 0.0 for v in frame_size)
    rw, rh = float(rec.get("w") or 0), float(rec.get("h") or 0)
    if fw <= 0 or fh <= 0 or rw <= 0 or rh <= 0:
        return None
    kx, ky = rw / fw, rh / fh
    return [bbox[0] * kx, bbox[1] * ky, bbox[2] * kx, bbox[3] * ky]


def match_person(rec: dict | None, bbox, frame_size) -> tuple[int | None, float]:
    """La persona del registro que más se parece al recuadro de la alerta: (número, cuánto se pisan)."""
    box = _scaled_bbox(bbox, frame_size, rec or {})
    best, best_iou = None, 0.0
    if box is None:
        return None, 0.0
    for p in (rec or {}).get("people") or []:
        v = iou(box, p["box"])
        if v > best_iou:
            best, best_iou = p.get("id"), v
    return best, best_iou


def refine_offset(frames: list[dict], guess: float, bbox, frame_size, window: float = REFINE_WINDOW) -> float | None:
    """Corrige la estimación con lo que vio la IA: el momento, cerca de lo estimado, en que alguien está donde estaba la
    persona de la alerta. Con empate (la persona quieta), el más cercano a lo estimado. None si no aparece."""
    best_t, best_key = None, None
    for rec in frames:
        t = float(rec.get("t", 0.0))
        if abs(t - guess) > window:
            continue
        _pid, v = match_person(rec, bbox, frame_size)
        if v < REFINE_IOU:
            continue
        key = (round(v, 1), -abs(t - guess))
        if best_key is None or key > best_key:
            best_t, best_key = t, key
    return best_t


def clip_duration(path: str) -> float | None:
    cap = cv2.VideoCapture(path)
    try:
        n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0, cap.get(cv2.CAP_PROP_FPS) or 0
        return n / fps if n > 0 and 1 <= fps <= 120 else None
    finally:
        cap.release()


def draw_overlay(img: np.ndarray, rec: dict | None, zones: list[dict], alert: str | None, alert_pid: int | None,
                 alert_box=None) -> np.ndarray:
    """Las zonas de la cámara, y lo que vio la IA en ese cuadro: personas (número y esqueleto), bolsos y, cerca del
    momento de la alerta, la persona de la alerta en rojo con el cartel arriba."""
    h, w = img.shape[:2]
    for z in zones or []:
        pts = np.array([[int(x * w), int(y * h)] for x, y in z.get("polygon") or []], np.int32)
        if len(pts) < 3:
            continue
        color = ZONE_BGR.get(z.get("type"), (200, 200, 200))
        cv2.polylines(img, [pts], True, color, 1, cv2.LINE_AA)
        x, y = pts[:, 0].min(), pts[:, 1].min()
        cv2.putText(img, ascii_text(str(z.get("name") or "")), (int(x) + 3, int(y) + 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, color, 1, cv2.LINE_AA)
    if rec:
        k = w / float(rec.get("w") or w)
        for b in rec.get("bags") or []:
            x1, y1, x2, y2 = (int(v * k) for v in b[:4])
            cv2.rectangle(img, (x1, y1), (x2, y2), VIOLET, 1)
        for p in rec.get("people") or []:
            pid = p.get("id")
            hot = alert is not None and pid is not None and pid == alert_pid
            color = RED if hot else GREEN
            x1, y1, x2, y2 = (int(v * k) for v in p["box"][:4])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3 if hot else 2)
            if p.get("kp"):
                try:
                    draw_pose(img, np.asarray(p["kp"], dtype=np.float32), color if hot else (0, 220, 255), scale=k)
                except (ValueError, IndexError):
                    pass
            cv2.putText(img, f"#{pid}", (x1 + 2, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3,
                        cv2.LINE_AA)
            cv2.putText(img, f"#{pid}", (x1 + 2, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1,
                        cv2.LINE_AA)
    if alert is not None:
        if alert_pid is None and alert_box is not None:  # (la IA todavía no llegó o no la encontró: el recuadro de la
            x1, y1, x2, y2 = (int(v) for v in alert_box)  # máquina)
            cv2.rectangle(img, (x1, y1), (x2, y2), RED, 2)
        cv2.rectangle(img, (0, 0), (w, 26), (0, 0, 170), -1)
        cv2.putText(img, ascii_text(f"ALERTA: {alert}"), (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1,
                    cv2.LINE_AA)
    return img


class Timeline:
    """La barra del reproductor, dibujada en un Canvas: el largo del clip, hasta dónde llegó la IA, el momento de la
    alerta en rojo y por dónde va. Clic o arrastrar llama a `on_seek(segundo)`."""

    PAD, H = 10, 30

    def __init__(self, tk, parent, on_seek: Callable[[float], None], bg: str):
        # (ancho pedido chico: toma lo que sobre; con el de fábrica, unos 380 px, empujaba el tiempo afuera)
        self.c = tk.Canvas(parent, width=120, height=self.H, bg=bg, highlightthickness=0, cursor="hand2")
        self.on_seek = on_seek
        self.duration = 0.0
        self.pos = 0.0
        self.alert: float | None = None
        self.exact = True
        self.ai_upto = 0.0
        self.c.bind("<Configure>", lambda _e: self.draw())
        self.c.bind("<Button-1>", self._click)
        self.c.bind("<B1-Motion>", self._click)

    def x_of(self, t: float) -> float:
        w = max(self.c.winfo_width(), 2 * self.PAD + 1)
        span = w - 2 * self.PAD
        return self.PAD + (span * t / self.duration if self.duration > 0 else 0.0)

    def t_of(self, x: float) -> float:
        w = max(self.c.winfo_width(), 2 * self.PAD + 1)
        span = w - 2 * self.PAD
        return max(0.0, min(self.duration, (x - self.PAD) / span * self.duration)) if self.duration > 0 else 0.0

    def _click(self, ev) -> None:
        if self.duration > 0:
            self.on_seek(self.t_of(ev.x))

    def draw(self) -> None:
        c = self.c
        c.delete("all")
        w = max(c.winfo_width(), 2 * self.PAD + 1)
        mid = self.H // 2 + 3
        c.create_rectangle(self.PAD, mid - 3, w - self.PAD, mid + 3, fill="#d8dee9", outline="")
        if self.duration <= 0:
            return
        if self.ai_upto > 0:
            c.create_rectangle(self.PAD, mid - 3, self.x_of(min(self.ai_upto, self.duration)), mid + 3,
                               fill="#a9b8d6", outline="")
        if self.alert is not None:
            x = self.x_of(self.alert)
            band = max(3.0, self.x_of(min(self.duration, self.alert + 1.5)) - x)
            c.create_rectangle(x - band, mid - 5, x + band, mid + 5, fill="#e5484d", outline="")
            c.create_polygon(x - 6, 1, x + 6, 1, x, 9, fill="#c0392b", outline="")
            c.create_text(x + 9, 5, text="alerta" if self.exact else "alerta (aprox.)", anchor="w",
                          fill="#c0392b", font=("Segoe UI", 8))
        x = self.x_of(self.pos)
        c.create_oval(x - 7, mid - 7, x + 7, mid + 7, fill="#011b5d", outline="white", width=2)


class AlertPlayer:
    """El reproductor: imagen (con lo que vio la IA), controles y la barra. `show_photo` mientras baja el clip;
    `open(path)` cuando está; `toggle`, `seek`, `step`; `clear` al cambiar de alerta. `app`: la Oficina."""

    def __init__(self, app, parent):
        self.app = app
        self.tk, self.ttk = app.tk, app.ttk
        tk, ttk = self.tk, self.ttk
        self.root = app.v.root
        # se pueden cambiar en las pruebas (una IA doble)
        from .ia_clips import analyze
        self.analyze_fn: Callable = analyze
        self.ai_setup: Callable | None = None
        self.frame = ttk.Frame(parent)
        self.view = tk.Label(self.frame, bg="#111", fg="#c8ced8", height=20)
        self.view.pack(fill="both", expand=True, pady=(6, 4))
        self.view.bind("<Configure>", lambda _e: self._show())
        ctl = ttk.Frame(self.frame)
        ctl.pack(fill="x")
        self.play_btn = ttk.Button(ctl, text="▶ Reproducir", width=13, takefocus=False, command=self.toggle)
        self.play_btn.pack(side="left")
        ttk.Button(ctl, text="−5 s", style="Chico.TButton", width=5, takefocus=False,
                   command=lambda: self.step(-STEP_S)).pack(side="left", padx=(6, 0))
        ttk.Button(ctl, text="+5 s", style="Chico.TButton", width=5, takefocus=False,
                   command=lambda: self.step(STEP_S)).pack(side="left", padx=(2, 6))
        from .estilo import FONDO
        self.bar = Timeline(tk, ctl, self.seek, FONDO)
        self.bar.c.pack(side="left", fill="x", expand=True)
        self.time_lbl = ttk.Label(ctl, text="", width=13, anchor="e")
        self.time_lbl.pack(side="left", padx=(6, 0))
        self.status = ttk.Label(self.frame, text="", style="Suave.TLabel")
        self.status.pack(anchor="w", pady=(2, 0))
        self._cache: OrderedDict = OrderedDict()
        self._q: queue.Queue = queue.Queue()
        self._job = None
        self._gen = 0
        self._reset()

    # ---------- estado ----------
    def _reset(self) -> None:
        self.event: dict | None = None
        self.path: str | None = None
        self.reader: FrameReader | None = None
        self.duration = 0.0
        self.pos = 0.0
        self.playing = False
        self.autoplay = False
        self.offset: float | None = None
        self.exact = True
        self.refined = False
        self.records: list[dict] = []
        self.times: list[float] = []
        self.ai_done = False
        self.ai_error: str | None = None
        self.base: np.ndarray | None = None  # el cuadro del video que se ve (o la foto de la alerta)
        self.base_t: float | None = None  # de qué momento es (None = la foto)
        self.zones: list[dict] = []
        self.alert_text: str | None = None
        self.detection = None
        self._stop = threading.Event()
        self._last = time.monotonic()
        self._photo = None
        self.shown: np.ndarray | None = None

    def clear(self, text: str = "") -> None:
        """Otra alerta (o ninguna): suelta el video y frena la IA."""
        self._gen += 1
        self._stop.set()
        if self.reader is not None:
            self.reader.close()
        self._reset()
        self.view.configure(image="", text=text)
        self.play_btn.configure(text="▶ Reproducir")
        self.bar.duration, self.bar.alert, self.bar.ai_upto, self.bar.pos = 0.0, None, 0.0, 0.0
        self.bar.draw()
        self.time_lbl.configure(text="")
        self.status.configure(text="")

    def select(self, event: dict, cam: dict | None, alert_text: str) -> None:
        """Se eligió una alerta: todavía sin clip (lo baja la Oficina y llama a `open`)."""
        self.clear()
        self.event = event
        self.zones = list((cam or {}).get("zones") or [])
        self.alert_text = alert_text
        if cam is not None:
            from .oficina import camera_detection
            self.detection = camera_detection(cam)
        self.status.configure(text="Bajando el clip…" if event.get("clip_url") else "El clip todavía no llegó.")

    def show_photo(self, img: np.ndarray) -> None:
        """La foto de la alerta (hasta que se ve el clip)."""
        if self.base_t is None:
            self.base = img
            self._show()

    def failed(self, why: str) -> None:
        self.autoplay = False
        self.status.configure(text=why)

    # ---------- el clip ----------
    def open(self, path: str) -> None:
        """El clip ya está en esta PC: la barra, el momento de la alerta, el video y la IA."""
        if self.event is None:
            return
        self.path = path
        self.duration = clip_duration(path) or 0.0
        cached = self._cache.get(self.event.get("id"))
        if cached and not self.duration:
            self.duration = cached["duration"]
        self.offset, self.exact = alert_offset(self.event, self.duration or None)
        self.pos = max(0.0, self.offset - LEAD_IN)
        self.reader = FrameReader(path, MAX_DECODE)
        self.reader.want(self.pos)
        self.bar.duration, self.bar.alert, self.bar.exact = self.duration, self.offset, self.exact
        if cached:
            self.records, self.times = cached["records"], [r["t"] for r in cached["records"]]
            self.ai_done = True
            self._refine()
        else:
            self._start_ai()
        self._update_status()
        if self.autoplay:
            self.autoplay = False
            self.play()
        self._ensure_poll()

    def _start_ai(self) -> None:
        gen, stop, path, det = self._gen, self._stop, self.path, self.detection

        def work() -> None:
            try:
                if self.ai_setup is not None:
                    detector, pose = self.ai_setup()
                else:
                    agent = self.app.agent_of() if getattr(self.app, "agent_of", None) else None
                    if agent is None:
                        raise RuntimeError("la app todavía está arrancando la IA")
                    from .worker import shared_pose
                    detector, pose = agent._detector("yolo"), shared_pose()
                res = self.analyze_fn(path, detector, pose, fps=5.0, detection=det, stop=stop,
                                      on_frame=lambda rec: self._q.put(("rec", gen, rec)))
                self._q.put(("done", gen, res))
            except Exception as exc:  # noqa: BLE001 - sin la IA se ve el video igual
                if not stop.is_set():
                    log.warning("Alertas: no se pudo analizar el clip con la IA: %s", exc)
                    self._q.put(("error", gen, str(exc)))
        threading.Thread(target=work, name="alerta-ia", daemon=True).start()

    # ---------- controles ----------
    def toggle(self) -> None:
        if self.playing:
            return self.pause()
        if self.reader is None:  # todavía bajando: arranca solo cuando llegue
            self.autoplay = self.event is not None and bool(self.event.get("clip_url"))
            return
        self.play()

    def play(self) -> None:
        if self.reader is None or self.duration <= 0:
            return
        if self.pos >= self.duration - 0.05:
            self.pos = 0.0
        self.playing = True
        self._last = time.monotonic()
        self.play_btn.configure(text="❚❚ Pausa")
        self._ensure_poll()

    def pause(self) -> None:
        self.playing = False
        self.play_btn.configure(text="▶ Reproducir")

    def seek(self, t: float) -> None:
        if self.reader is None or self.duration <= 0:
            return
        self.pos = max(0.0, min(float(t), self.duration))
        self._last = time.monotonic()
        self.reader.want(self.pos)
        self._update_bar()
        self._ensure_poll()

    def step(self, delta: float) -> None:
        self.seek(self.pos + delta)

    # ---------- el ciclo ----------
    def _ensure_poll(self) -> None:
        if self._job is None:
            try:
                self._job = self.root.after(POLL_MS, self._poll)
            except Exception:  # noqa: BLE001 - la ventana se cerró
                self._job = None

    def _poll(self) -> None:
        self._job = None
        changed = self._drain()
        if self.playing and self.reader is not None:
            now = time.monotonic()
            self.pos = min(self.duration, self.pos + (now - self._last))
            self._last = now
            self.reader.want(self.pos)
            if self.pos >= self.duration - 1e-3:
                self.pause()
        got = self.reader.take() if self.reader is not None else None
        if got is not None:
            self.base_t, self.base = got
            changed = True
        if changed:
            self._show()
        self._update_bar()
        busy = self.playing or (self.reader is not None and self.reader.waiting) or (
            self.reader is not None and not self.ai_done and self.ai_error is None)
        if busy:
            self._ensure_poll()

    def _drain(self) -> bool:
        changed = False
        while True:
            try:
                kind, gen, data = self._q.get_nowait()
            except queue.Empty:
                break
            if gen != self._gen:
                continue  # (de una alerta anterior)
            if kind == "rec":
                self.records.append(data)
                self.times.append(float(data.get("t", 0.0)))
                changed = True
            elif kind == "done":
                self.ai_done = True
                if isinstance(data, dict) and data.get("frames"):
                    self.records = list(data["frames"])
                    self.times = [float(r.get("t", 0.0)) for r in self.records]
                if self.duration <= 0 and isinstance(data, dict) and _num(data.get("duration_s")):
                    # (el archivo no decía cuánto dura: lo que midió la IA)
                    self.duration = float(data["duration_s"])
                    self.offset, self.exact = alert_offset(self.event or {}, self.duration)
                    self.bar.duration, self.bar.alert, self.bar.exact = self.duration, self.offset, self.exact
                if self.event is not None:
                    self._cache[self.event.get("id")] = {"records": self.records, "duration": self.duration}
                    while len(self._cache) > CACHE:
                        self._cache.popitem(last=False)
                self._update_status()
                changed = True
            elif kind == "error":
                self.ai_error = data
                self._update_status()
        if changed:
            self._refine()
        return changed

    def _refine(self) -> None:
        """Una alerta sin el segundo exacto: cuando la IA ya analizó ese rato, se corrige con el recuadro."""
        if self.exact or self.refined or self.event is None or self.offset is None:
            return
        upto = self.times[-1] if self.times else 0.0
        if not self.ai_done and upto < self.offset + REFINE_WINDOW:
            return
        self.refined = True
        t = refine_offset(self.records, self.offset, self.event.get("bbox"), self.event.get("frame_size"))
        if t is not None:
            self.offset, self.exact = t, True
            self.bar.alert, self.bar.exact = t, True
            self._update_bar()
            self._update_status()

    def _rec_at(self, t: float | None) -> dict | None:
        if t is None or not self.times:
            return None
        i = bisect.bisect_left(self.times, t)
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(self.times) and (best is None or abs(self.times[j] - t) < abs(self.times[best] - t)):
                best = j
        return self.records[best] if best is not None and abs(self.times[best] - t) <= 0.35 else None

    def _show(self) -> None:
        """Dibuja el cuadro (o la foto) con lo que vio la IA, al tamaño del lugar."""
        if self.base is None:
            return
        img = self.base.copy()
        if self.base_t is not None:  # (la foto ya viene dibujada por la máquina)
            rec = self._rec_at(self.base_t)
            near = self.offset is not None and abs(self.base_t - self.offset) <= ALERT_SHOW
            pid, box = None, None
            if near and self.event is not None:
                if rec is not None:
                    pid, v = match_person(rec, self.event.get("bbox"), self.event.get("frame_size"))
                    pid = pid if v >= 0.2 else None
                fake = {"w": img.shape[1], "h": img.shape[0]}
                box = _scaled_bbox(self.event.get("bbox"), self.event.get("frame_size"), fake) if self.exact else None
            draw_overlay(img, rec, self.zones, self.alert_text if near else None, pid, box)
        self.shown = img  # (lo que se ve, antes de ajustarlo al lugar: lo miran las pruebas)
        w, h = max(self.view.winfo_width(), 320), max(self.view.winfo_height(), 180)
        ih, iw = img.shape[:2]
        _k, dw, dh = fit(iw, ih, w - 4, h - 4)
        img = cv2.resize(img, (dw, dh), interpolation=cv2.INTER_AREA if dw < iw else cv2.INTER_LINEAR)
        ok, ppm = cv2.imencode(".ppm", img)
        if ok:
            self._photo = self.tk.PhotoImage(data=ppm.tobytes())
            self.view.configure(image=self._photo, text="")

    def _update_bar(self) -> None:
        self.bar.pos = self.pos
        self.bar.ai_upto = self.duration if self.ai_done else (self.times[-1] if self.times else 0.0)
        self.bar.draw()
        if self.duration > 0:
            self.time_lbl.configure(text=f"{fmt_t(self.pos)} / {fmt_t(self.duration)}")

    def _update_status(self) -> None:
        if self.ai_error:
            text = f"Sin la IA: {self.ai_error}. Se ve el video solo."
        elif self.ai_done:
            text = "Con la IA de esta PC (personas, postura, bolsos y zonas; no guarda nada)."
        else:
            text = "Analizando el clip con la IA de esta PC… (lo ya analizado se ve con recuadros)"
        if self.offset is not None:
            text += f"  ·  La alerta, en el {fmt_t(self.offset)}" + ("" if self.exact else " (aproximado)") + "."
        self.status.configure(text=text)
