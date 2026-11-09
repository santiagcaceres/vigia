"""Pestaña «Entrenar IA hurtos» de la app (pedido del dueño, 9 nov: «una nueva pestaña… donde subiremos clips y
pondremos si hubo hurto y dónde; aplicarle la IA de YOLO y la postura a estos videos y ahí entrenaremos a la IA»).

Todo lo de entrenar está acá (la web queda para los clientes), para administradores y operadores:
- **Clips** (izquierda): los de la nube (`GET /api/training/clips`), con cliente, cámara, duración, estado, quién lo
  subió y si el video está en esta PC. «⬆ Subir clips…»: se eligen videos, el cliente y la cámara, y se analizan EN
  ESTA PC, de a uno, en otro hilo (agent/ia_clips.py: YOLO con la sensibilidad de la cámara, seguimiento, postura de
  todas las personas y bolsos) → core.hurto_ia.person_windows → `POST /api/training/clips` y el .json.gz de los
  cuadros directo al almacenamiento (`data_upload`). **El video no sale de la PC.**
- **Clip elegido** (derecha): se reproduce con los recuadros, el número de cada persona y su esqueleto (si el video
  no está en esta PC, los mismos datos sobre fondo gris). Marcar un hurto = «Inicio del hurto» y «Fin del hurto» en
  el momento de la barra, clic en la persona que roba y «Cómo» → la marca va a la lista. «Este clip es normal»,
  «Guardar marcas» (`PUT`) y «Probar con las reglas y mandar al filtro…» (la ventana de la cámara en «🎞 Clips»).
- **Entrenar** (abajo a la izquierda), con el alcance de arriba (todos los clientes o uno):
  «IA de movimientos» (core/hurto_ia.py: aprende con los esqueletos de los clips marcados y da alertas propias,
  «Posible hurto (IA de movimientos)») y «Filtro de alertas» (server/training.py: aprende a descartar falsas alarmas
  con las alertas etiquetadas). Entrenar, ver qué tan bien anduvo en palabras y activar o apagar cada modelo.

Las llamadas a la red van en hilos (`app.bg.run`); analizar un clip y leer el video para mostrarlo también (nunca se
traba la ventana). Entra en una pantalla de 1366x768.
"""

from __future__ import annotations

import bisect
import copy
import json
import logging
import queue
import threading
import time
import unicodedata
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from core.deteccion import DetectionConfig
from core.pose import draw as draw_pose

from .ia_clips import FPS, ClipError, LocalClips, analyze, clip_uuid, pack, unpack
from .marca import NOMBRE

log = logging.getLogger("vigia.entrenar_ia")

ALL_CLIENTS = "Todos los clientes"
NO_CLIENT = "Sin cliente"
NO_CAMERA = "Sin cámara"
VERDICT_NAMES = {"hurto": "Hurto", "normal": "Normal", "sin_marcar": "Sin marcar"}
# «Cómo» lo escondió (contrato: bolsillo | ropa | bolsa | otro | None)
KIND_NAMES = {"bolsillo": "Al bolsillo", "ropa": "Dentro de la ropa", "bolsa": "En una bolsa o cartera",
              "otro": "Otro", None: "No sé"}
PLAYER_W, PLAYER_H = 528, 297  # el reproductor (16:9): entra con todo lo demás en 1366x768
STEP_S = 1.0  # «−1 s» / «+1 s»
PICK_SLACK_PX = 8  # un clic hasta 8 píxeles afuera del recuadro igual elige a esa persona
# lo mínimo para entrenar (core/hurto_ia.NotEnoughData): el servidor lo vuelve a mirar y dice el motivo exacto
IA_MIN_MARKS, IA_MIN_NORMAL_CLIPS, IA_MIN_NORMAL_MIN = 5, 3, 10
IA_GOOD_MARKS = 30  # «empieza a servir con 30 a 50 hurtos marcados»
TARGET_RECALL = 0.8
POLL_MS = 50
MAX_BODY = 4_200_000  # Vercel no acepta pedidos de más de 4,5 MB
IDLE_HINT = "El video queda en esta PC: a la nube va solo lo que vio la IA (recuadros y esqueletos)."
GREEN, ORANGE, RED, VIOLET = (0, 200, 0), (0, 140, 255), (0, 0, 255), (230, 80, 230)


# ------------------------------------------------------------------ textos (sin ventana: se prueban sueltos)
def ascii_text(text: str) -> str:
    """OpenCV no dibuja tildes: «Marca 1: al bolsillo» sin acentos."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def fmt_t(sec: float | None, tenths: bool = False) -> str:
    """«0:12» (o «0:12.4» con décimas): un momento del clip."""
    if sec is None:
        return "—"
    sec = max(0.0, float(sec))
    if tenths:
        m, s = divmod(round(sec, 1), 60)
        return f"{int(m)}:{s:04.1f}"
    s = int(round(sec))
    return f"{s // 60}:{s % 60:02d}"


def _num(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _items(res) -> list[dict]:
    """Una lista del servidor (tal cual, o dentro de «items» / «clips» / «models»)."""
    if isinstance(res, list):
        return [x for x in res if isinstance(x, dict)]
    if isinstance(res, dict):
        for key in ("items", "clips", "models"):
            if isinstance(res.get(key), list):
                return [x for x in res[key] if isinstance(x, dict)]
    return []


def plural(n: int, one: str, many: str | None = None) -> str:
    return f"{n} {one if n == 1 else (many or one + 's')}"


def verdict_text(clip: dict) -> str:
    """«Hurto: 2 marcas» / «Normal» / «Sin marcar» (y «· subir de nuevo» si sus rasgos son de otra versión)."""
    v = clip.get("verdict") or "sin_marcar"
    text = f"Hurto: {plural(len(clip.get('marks') or []), 'marca')}" if v == "hurto" else VERDICT_NAMES.get(v, v)
    return text + (" · subir de nuevo" if clip.get("feature_ok") is False else "")


def uploader_of(clip: dict) -> str:
    return str(clip.get("created_by_name") or clip.get("created_by_username") or clip.get("user_name")
               or clip.get("username") or "")


def mark_text(m: dict) -> str:
    who = f"#{m['track_id']}" if m.get("track_id") is not None else "cualquier persona"
    return f"{fmt_t(m.get('start'), True)} a {fmt_t(m.get('end'), True)} · {who} · {KIND_NAMES.get(m.get('kind'), 'No sé')}"


def ia_counts(clips: list[dict]) -> dict:
    """Lo que hay para la IA de movimientos en esos clips (los del alcance elegido)."""
    theft = [c for c in clips if c.get("verdict") == "hurto"]
    normal = [c for c in clips if c.get("verdict") == "normal"]
    return {"clips": len(clips), "theft_clips": len(theft),
            "marks": sum(len(c.get("marks") or []) for c in theft),
            "normal_clips": len(normal), "normal_minutes": sum(_num(c.get("duration_s")) for c in normal) / 60,
            "unmarked": sum(1 for c in clips if (c.get("verdict") or "sin_marcar") == "sin_marcar")}


COUNT_KEYS = ("clips", "theft_clips", "marks", "normal_clips", "normal_minutes", "unmarked", "old_version")
MIN_KEYS = {"min_marks": IA_MIN_MARKS, "min_normal_clips": IA_MIN_NORMAL_CLIPS,
            "min_normal_minutes": IA_MIN_NORMAL_MIN, "recommended_marks": IA_GOOD_MARKS}


def summary_counts(summary, clips: list[dict]) -> dict:
    """Las cuentas de `/api/training/ia/summary` (y los mínimos para entrenar); lo que no venga, contado de la lista
    de clips y con los mínimos de siempre."""
    out = {**ia_counts(clips), "old_version": 0, **MIN_KEYS}
    if isinstance(summary, dict):
        for key in COUNT_KEYS + tuple(MIN_KEYS):
            v = summary.get(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out[key] = v
    return out


def ia_counts_text(c: dict) -> str:
    text = (f"{plural(int(c['clips']), 'clip')}: {int(c['theft_clips'])} con hurto "
            f"({plural(int(c['marks']), 'marca')}) · {int(c['normal_clips'])} normales "
            f"({_num(c['normal_minutes']):.0f} min) · {int(c['unmarked'])} sin marcar.")
    if c.get("old_version"):
        text += (f" {plural(int(c['old_version']), 'clip')} de una versión vieja de la app no cuenta"
                 f"{'n' if int(c['old_version']) != 1 else ''} (volvé a subirlo{'s' if int(c['old_version']) != 1 else ''}).")
    return text


def ia_advice(c: dict) -> tuple[str, bool]:
    """Qué decir según cuánto hay: (texto, alcanza para entrenar). El servidor vuelve a mirarlo al entrenar y, si
    falta algo, dice qué (core/hurto_ia.NotEnoughData)."""
    marks, normal, mins = int(c["marks"]), int(c["normal_clips"]), _num(c["normal_minutes"])
    need_m, need_n = int(c.get("min_marks", IA_MIN_MARKS)), int(c.get("min_normal_clips", IA_MIN_NORMAL_CLIPS))
    need_min, good = _num(c.get("min_normal_minutes"), IA_MIN_NORMAL_MIN), int(c.get("recommended_marks", IA_GOOD_MARKS))
    enough = marks >= need_m and (normal >= need_n or mins >= need_min) and int(c["theft_clips"]) >= 2
    if not enough:
        return (f"Faltan datos: al menos {need_m} hurtos marcados (en 2 clips o más) y {need_n} clips normales (o "
                f"{need_min:.0f} min). Sirve de verdad con {good} a 50 hurtos."), False
    if marks < good:
        return (f"Alcanza para una primera prueba, pero {plural(marks, 'hurto marcado', 'hurtos marcados')} es poco: "
                f"sirve de verdad con {good} a 50 y varias horas de video normal."), True
    return ("Hay bastantes ejemplos. Después de entrenar mirá cuántos hurtos ve y cuántas falsas alarmas da antes de "
            "activarla."), True


def ia_result_text(met: dict | None) -> str:
    """«De 12 hurtos marcados vio 9 · 1,5 falsas alarmas por hora de video normal»."""
    met = met or {}
    marks, seen = int(_num(met.get("marks"))), int(_num(met.get("marks_seen")))
    text = f"De {marks} {'hurto marcado' if marks == 1 else 'hurtos marcados'} vio {seen}"
    fph = met.get("false_per_hour")
    if fph is not None:
        text += f" · {_num(fph):.1f}".replace(".", ",") + " falsas alarmas por hora de video normal"
    return text


def ia_model_text(met: dict | None) -> str:
    """Una fila de la lista de modelos: «vio 9 de 12 · 1,5 falsas/hora»."""
    met = met or {}
    text = f"vio {int(_num(met.get('marks_seen')))} de {int(_num(met.get('marks')))}"
    if met.get("false_per_hour") is not None:
        text += f" · {_num(met['false_per_hour']):.1f}".replace(".", ",") + " falsas/hora"
    return text


def filter_model_text(m: dict) -> str:
    """«40 ej. · deja pasar 90 % · saca 55 %»."""
    met = m.get("metrics") or {}
    return (f"{m.get('n_samples') or 0} ej. · deja pasar {_num(met.get('recall')) * 100:.0f} % · saca "
            f"{_num(met.get('falsas_alarmas_eliminadas')) * 100:.0f} %")


def filter_counts(summary: dict | None, client_id: int | None) -> dict:
    """Alertas etiquetadas para el filtro: del cliente o de todos (`/api/training/summary`)."""
    rows = [r for r in (summary or {}).get("clients") or [] if client_id is None or r.get("client_id") == client_id]
    labels: dict[str, int] = {}
    tuning = 0
    for r in rows:
        tuning += int(r.get("from_tuning") or 0)
        for k, n in (r.get("labels") or {}).items():
            labels[k] = labels.get(k, 0) + int(n or 0)
    pos = labels.get("hurto", 0) + labels.get("sospechoso", 0)
    neg = labels.get("normal", 0) + labels.get("falso_positivo", 0)
    return {"pos": pos, "neg": neg, "tuning": tuning, "unlabeled": labels.get("sin_etiqueta", 0),
            "min_total": int((summary or {}).get("min_total") or 30),
            "min_per_class": int((summary or {}).get("min_per_class") or 8)}


def filter_text(c: dict) -> tuple[str, str, bool]:
    """(cuántas alertas etiquetadas hay, qué falta o qué mirar, alcanza)."""
    text = (f"{c['pos'] + c['neg']} alertas etiquetadas: {c['pos']} hurto o sospechoso, {c['neg']} normal o falsa "
            f"alarma ({c['tuning']} desde la ventana de la cámara) · {c['unlabeled']} sin revisar.")
    enough = c["pos"] + c["neg"] >= c["min_total"] and min(c["pos"], c["neg"]) >= c["min_per_class"]
    if enough:
        advice = "Alcanza. Después de entrenar mirá cuántos hurtos deja pasar: tienen que ser casi todos."
    else:
        advice = (f"Hacen falta {c['min_total']} en total y {c['min_per_class']} de cada lado: se suman etiquetando "
                  "en «Alertas» o desde la ventana de la cámara.")
    return text, advice, enough


def filter_result_text(met: dict | None) -> str:
    met = met or {}
    return (f"Deja pasar el {_num(met.get('recall')) * 100:.0f} % de los hurtos y saca el "
            f"{_num(met.get('falsas_alarmas_eliminadas')) * 100:.0f} % de las falsas alarmas")


def scope_name(m: dict) -> str:
    return m.get("client_name") or (ALL_CLIENTS if m.get("client_id") is None else f"Cliente {m.get('client_id')}")


def model_date(m: dict) -> str:
    """«09/11 12:00» (en hora local)."""
    from .oficina import fmt_date
    return fmt_date(m.get("created_at")) if m.get("created_at") else ""


# ------------------------------------------------------------------ dibujo (lo mismo que la ventana de la cámara)
def nearest(times: list[float], t: float) -> int:
    """El registro de cuadro que corresponde al momento `t` (el último que no pasa de `t`)."""
    if not times:
        return -1
    return max(0, bisect.bisect_right(times, t + 1e-6) - 1)


def fit(w: int, h: int, box_w: int = PLAYER_W, box_h: int = PLAYER_H) -> tuple[float, int, int]:
    """Escala y tamaño para que un cuadro de w x h entre en el reproductor."""
    if w <= 0 or h <= 0:
        return 1.0, box_w, box_h
    k = min(box_w / w, box_h / h)
    return k, max(1, int(round(w * k))), max(1, int(round(h * k)))


def render(rec: dict | None, base: np.ndarray | None, selected: int | None, active: list[dict],
           footer: str, size: tuple[int, int] | None = None) -> tuple[np.ndarray, float]:
    """El cuadro del reproductor: el video (o fondo gris si no está en esta PC) con los bolsos, el recuadro, el número
    y el esqueleto de cada persona; la elegida en naranja y la de una marca que está pasando, en rojo, con el cartel
    arriba. Devuelve (imagen, escala de los píxeles del registro a los de la imagen)."""
    w = int((rec or {}).get("w") or (size or (0, 0))[0] or 640)
    h = int((rec or {}).get("h") or (size or (0, 0))[1] or 360)
    k, dw, dh = fit(w, h)
    if base is not None and base.shape[:2] == (dh, dw):
        img = base.copy()
    elif base is not None:
        img = cv2.resize(base, (dw, dh), interpolation=cv2.INTER_AREA)
    else:
        img = np.full((dh, dw, 3), (78, 72, 66), np.uint8)
    marked = {m.get("track_id") for m in active}
    anyone = None in marked
    for b in (rec or {}).get("bags") or []:
        x1, y1, x2, y2 = (int(v * k) for v in b[:4])
        cv2.rectangle(img, (x1, y1), (x2, y2), VIOLET, 1)
    for p in (rec or {}).get("people") or []:
        pid = p.get("id")
        color = ORANGE if pid == selected else (RED if pid in marked else GREEN)
        thick = 3 if pid == selected or pid in marked else 2
        x1, y1, x2, y2 = (int(v * k) for v in p["box"][:4])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        kp = p.get("kp")
        if kp:
            try:
                draw_pose(img, np.asarray(kp, dtype=np.float32), (0, 220, 255) if color == GREEN else color, scale=k)
            except (ValueError, IndexError):
                pass
        cv2.putText(img, f"#{pid}", (x1 + 2, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3,
                    cv2.LINE_AA)
        cv2.putText(img, f"#{pid}", (x1 + 2, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    if active:
        text = " | ".join(f"MARCA: {'cualquiera' if m.get('track_id') is None else '#' + str(m['track_id'])} "
                          f"{KIND_NAMES.get(m.get('kind'), 'No se')}" for m in active[:2])
        cv2.rectangle(img, (0, 0), (dw, 24), (0, 0, 160), -1)
        cv2.putText(img, ascii_text(text + (" (todas las personas)" if anyone else "")), (8, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(img, (0, dh - 22), (dw, dh), (0, 0, 0), -1)
    cv2.putText(img, ascii_text(footer), (6, dh - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img, k


def person_at(rec: dict | None, x: float, y: float, slack: float = 0.0) -> int | None:
    """La persona sobre la que se hizo clic (x, y en píxeles del registro): la de recuadro más chico que lo contiene;
    si ninguno, la más cercana a menos de `slack` píxeles."""
    best, best_area = None, None
    near, near_d = None, None
    for p in (rec or {}).get("people") or []:
        x1, y1, x2, y2 = p["box"][:4]
        if x1 <= x <= x2 and y1 <= y <= y2:
            area = (x2 - x1) * (y2 - y1)
            if best_area is None or area < best_area:
                best, best_area = p.get("id"), area
        elif slack > 0:
            d = max(x1 - x, 0, x - x2) + max(y1 - y, 0, y - y2)
            if d <= slack and (near_d is None or d < near_d):
                near, near_d = p.get("id"), d
    return best if best is not None else near


class FrameReader:
    """Lee el video en otro hilo (abrir, buscar y decodificar puede tardar): se le pide un momento (`want`) y deja el
    cuadro listo, con píxeles cuadrados y al tamaño del reproductor (`take`). Para reproducir lee para adelante sin
    volver a buscar."""

    def __init__(self, path: str):
        self.path = path
        self.error: str | None = None
        self._cv = threading.Condition()
        self._want: float | None = None
        self._result: tuple[float, np.ndarray] | None = None
        self._stop = False
        self.pending = False  # se pidió un cuadro que todavía no llegó
        self.thread = threading.Thread(target=self._run, name="ia-video", daemon=True)
        self.thread.start()

    def want(self, t: float) -> None:
        with self._cv:
            self._want = max(0.0, float(t))
            self.pending = True
            self._cv.notify()

    def take(self) -> tuple[float, np.ndarray] | None:
        with self._cv:
            res, self._result = self._result, None
            return res

    def close(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify()

    def _run(self) -> None:
        from core.imagen import pixel_aspect, square_pixels
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            self.error = "no se pudo abrir el video"
            self.pending = False
            return
        try:
            sar = pixel_aspect(cap)
            vfps = cap.get(cv2.CAP_PROP_FPS) or 0
            half = 0.5 / vfps if 1 <= vfps <= 120 else 0.04
            pos: float | None = None
            while True:
                with self._cv:
                    while self._want is None and not self._stop:
                        self._cv.wait(0.5)
                    if self._stop:
                        return
                    t, self._want = self._want, None
                if pos is None or t < pos - half or t > pos + 1.5:
                    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
                    pos = None
                frame = None
                for _ in range(400):  # (como mucho unos segundos de video para llegar)
                    if not cap.grab():
                        break
                    pos = (cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
                    if pos + 1e-3 >= t - half:
                        ok, f = cap.retrieve()
                        frame = f if ok else None
                        break
                if frame is None:
                    with self._cv:
                        self.pending = self._want is not None
                    continue
                if sar:
                    frame = square_pixels(frame, sar)
                h, w = frame.shape[:2]
                _k, dw, dh = fit(w, h)
                small = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA)
                with self._cv:
                    self._result = (t, small)
                    self.pending = self._want is not None
        except Exception as exc:  # noqa: BLE001 - se muestra sin video
            log.warning("No se pudo leer el video %s: %s", self.path, exc)
            self.error = str(exc)
        finally:
            self.pending = False
            cap.release()

    @property
    def waiting(self) -> bool:
        """Hay un cuadro pedido o listo para mostrar."""
        with self._cv:
            return self.thread.is_alive() and (self.pending or self._result is not None)


# ------------------------------------------------------------------ la pestaña
class EntrenarIA:
    """La pestaña. `app`: la Oficina (api, bg, clientes y cámaras, la ventana de la cámara)."""

    def __init__(self, app, parent):
        self.app = app
        self.tk, self.ttk = app.tk, app.ttk
        self.root = app.v.root
        self.frame = parent
        # se pueden cambiar en las pruebas (un analizador y una IA dobles, otra carpeta)
        self.analyze_fn: Callable = analyze
        self.windows_fn: Callable | None = None  # None = core.hurto_ia.person_windows
        self.ai_setup: Callable | None = None  # None = el detector y la postura compartidos de la app
        self.data_dir: Path | None = None
        self._local_obj: LocalClips | None = None
        self.clips: list[dict] = []
        self._req = 0
        self._q: queue.Queue = queue.Queue()
        self._poll_job = None
        # subida
        self.uploading = False
        self._stop = threading.Event()
        self.upload_log: list[tuple[str, str]] = []  # (nombre, qué pasó) de la última tanda
        # clip elegido
        self.clip: dict | None = None
        self.frames: list[dict] = []
        self.times: list[float] = []
        self.video: str | None = None
        self.reader: FrameReader | None = None
        self.base: np.ndarray | None = None  # el cuadro del video que se muestra (al tamaño del reproductor)
        self.base_t = 0.0  # de qué momento es (los recuadros que se dibujan son de ese momento)
        self._shown_rec: dict | None = None
        self._shown_t = 0.0
        self.pos = 0.0
        self.playing = False
        self._last_tick = time.monotonic()
        self._setting_scale = False
        self._photo = None
        self._k, self._ox, self._oy = 1.0, 0, 0
        self.sel_track: int | None = None
        self.m_start: float | None = None
        self.m_end: float | None = None
        self.marks: list[dict] = []
        self.dirty = False
        self._frames_error: str | None = None  # los cuadros del clip no se pudieron bajar
        self._loaded_scope: int | None = None
        self.ia_models: list[dict] = []
        self.filter_models: list[dict] = []
        self.ia_summary: dict | None = None
        self._build()

    # ---------- utilidades ----------
    @property
    def local(self) -> LocalClips:
        """Lo de esta PC (`<datos>/entrenamiento/`). Se resuelve al usarlo: al armar la pestaña el analizador de la app
        todavía no existe."""
        base = self.data_dir
        if base is None:
            agent = self.app.agent_of() if getattr(self.app, "agent_of", None) else None
            base = Path(agent.data_dir) if agent is not None else Path.home() / "AnaliticaCritica"
        folder = Path(base) / "entrenamiento"
        if self._local_obj is None or self._local_obj.folder != folder:
            self._local_obj = LocalClips(folder)
        return self._local_obj

    def scope_id(self) -> int | None:
        name = self.scope.get()
        return next((c["id"] for c in self.app.clients if c["name"] == name), None)

    def _cam(self, cid: int | None) -> dict | None:
        return next((c for c in self.app.cameras if c["id"] == cid), None) if cid is not None else None

    def _client_name(self, cid: int | None) -> str:
        return next((c["name"] for c in self.app.clients if c["id"] == cid), "") if cid is not None else ""

    def _say(self, kind: str, text: str) -> None:
        getattr(self.app.mb, f"show{kind}")(NOMBRE, text)

    def _fail(self, exc: Exception) -> None:
        net = getattr(self.app, "_net_fail", None)
        if net is not None and getattr(exc, "status", 0) == 401:
            return net(exc)
        self._say("error", str(exc))

    def _ensure_poll(self) -> None:
        if self._poll_job is None:
            try:
                self._poll_job = self.root.after(POLL_MS, self._poll)
            except Exception:  # noqa: BLE001 - la ventana se cerró
                self._poll_job = None

    # ---------- armado ----------
    def _build(self) -> None:
        tk, ttk, f = self.tk, self.ttk, self.frame
        left = ttk.Frame(f, padding=(2, 0, 12, 0))
        left.pack(side="left", fill="y")
        right = ttk.Frame(f, padding=(2, 0, 2, 0))
        right.pack(side="left", fill="both", expand=True)

        # --- clips
        lh = ttk.Frame(left)
        lh.pack(fill="x")
        ttk.Label(lh, text="Clips para entrenar", style="Subtitulo.TLabel").pack(side="left")
        ttk.Button(lh, text="⬆ Subir clips…", style="Primario.TButton", command=self.upload_dialog).pack(side="right")
        sh = ttk.Frame(left)
        sh.pack(fill="x", pady=(4, 4))
        ttk.Label(sh, text="Ver y entrenar").pack(side="left")
        self.scope = tk.StringVar(value=ALL_CLIENTS)
        self.scope_cb = ttk.Combobox(sh, textvariable=self.scope, values=[ALL_CLIENTS], state="readonly", width=22)
        self.scope_cb.pack(side="left", padx=6)
        self.scope_cb.bind("<<ComboboxSelected>>", lambda e: self.load())
        self.count_lbl = ttk.Label(sh, text="", style="Suave.TLabel")
        self.count_lbl.pack(side="left")
        self.tree = ttk.Treeview(left, columns=("donde", "dura", "estado", "subio", "aca"), show="tree headings",
                                 height=5, selectmode="browse")
        for col, txt, w, anchor in (("#0", "Clip", 104, "w"), ("donde", "Cliente · cámara", 100, "w"),
                                    ("dura", "Dura", 44, "center"), ("estado", "Estado", 112, "w"),
                                    ("subio", "Subió", 56, "w"), ("aca", "En PC", 60, "center")):
            self.tree.heading(col, text=txt, anchor="w" if anchor == "w" else "center")
            self.tree.column(col, width=w, minwidth=36, stretch=col in ("#0", "donde"), anchor=anchor)
        self.tree.pack(fill="x")
        self.tree.bind("<<TreeviewSelect>>", lambda e: self.show_clip())
        # estado de la subida (lugar fijo: la ventana no salta al empezar o terminar)
        up = ttk.Frame(left, height=38)
        up.pack(fill="x", pady=(4, 4))
        up.pack_propagate(False)
        row = ttk.Frame(up)
        row.pack(fill="x")
        self.up_lbl = ttk.Label(row, text=IDLE_HINT, style="Suave.TLabel", wraplength=372, justify="left")
        self.up_lbl.pack(side="left", fill="x", expand=True)
        self.stop_btn = ttk.Button(row, text="■ Frenar", style="Chico.TButton", width=8, command=self.stop_upload)
        self.up_bar = ttk.Progressbar(up, mode="determinate", maximum=100)

        # --- entrenar (las dos IA, con el alcance de arriba); pestañas más bajas que las de la ventana
        st = ttk.Style(self.root)
        st.configure("Entrenar.TNotebook", tabmargins=(0, 2, 0, 0))
        st.configure("Entrenar.TNotebook.Tab", padding=(12, 4))
        nb = ttk.Notebook(left, style="Entrenar.TNotebook")
        nb.pack(fill="both", expand=True)
        self.train_nb = nb
        self.ia = self._train_panel(nb, "IA de movimientos", ia=True)
        self.flt = self._train_panel(nb, "Filtro de alertas", ia=False)

        # --- clip elegido
        rh = ttk.Frame(right)
        rh.pack(fill="x")
        self.title = ttk.Label(rh, text="Elegí un clip de la lista", style="Titulo.TLabel")
        self.title.pack(side="left")
        more = ttk.Menubutton(rh, text="Más", width=6)
        menu = tk.Menu(more, tearoff=False)
        menu.add_command(label="Ubicar el video en esta PC…", command=self.locate_video)
        menu.add_command(label="Abrir la carpeta del video", command=self.open_folder)
        menu.add_separator()
        menu.add_command(label="Borrar el clip…", command=self.delete_clip)
        more["menu"] = menu
        more.pack(side="right")
        self.info = ttk.Label(right, text="Subí clips de hurtos y de ratos normales; después marcá en cada uno quién "
                                          "roba y en qué momento.", style="Suave.TLabel", wraplength=780,
                              justify="left")
        self.info.pack(anchor="w", pady=(0, 6), fill="x")
        body = ttk.Frame(right)
        body.pack(fill="both", expand=True)
        pl = ttk.Frame(body)
        pl.pack(side="left", anchor="n")
        self.canvas = tk.Canvas(pl, width=PLAYER_W, height=PLAYER_H, bg="#111111", highlightthickness=0,
                                cursor="hand2")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)
        ctl = ttk.Frame(pl)
        ctl.pack(fill="x", pady=(6, 0))
        self.play_btn = ttk.Button(ctl, text="▶", width=3, command=self.toggle_play)
        self.play_btn.pack(side="left")
        ttk.Button(ctl, text="−1 s", style="Chico.TButton", width=5, command=lambda: self.step(-STEP_S)).pack(
            side="left", padx=(4, 0))
        ttk.Button(ctl, text="+1 s", style="Chico.TButton", width=5, command=lambda: self.step(STEP_S)).pack(
            side="left", padx=(2, 6))
        self.scale = ttk.Scale(ctl, from_=0, to=1, orient="horizontal", command=self.on_scale)
        self.scale.pack(side="left", fill="x", expand=True)
        self.time_lbl = ttk.Label(ctl, text="0:00.0 / 0:00", width=13, anchor="e")
        self.time_lbl.pack(side="left", padx=(6, 0))
        ttk.Label(pl, text="Clic sobre una persona para elegirla (se pone naranja). En rojo, la persona de una marca "
                           "mientras dura.", style="Suave.TLabel", wraplength=PLAYER_W, justify="left").pack(
            anchor="w", pady=(4, 0))
        ttk.Button(pl, text="Probar con las reglas y mandar al filtro…", command=self.try_rules).pack(
            anchor="w", pady=(8, 0))
        ttk.Label(pl, text="Abre la ventana de la cámara con este clip: si las reglas o los gestos dan alerta, se "
                           "etiquetan y se mandan al filtro de alertas.", style="Suave.TLabel", wraplength=PLAYER_W,
                  justify="left").pack(anchor="w", pady=(2, 0))

        mk = ttk.Frame(body, padding=(12, 0, 0, 0))
        mk.pack(side="left", fill="both", expand=True, anchor="n")
        ttk.Label(mk, text="Marcar el hurto", style="Subtitulo.TLabel").pack(anchor="w")
        r1 = ttk.Frame(mk)
        r1.pack(fill="x", pady=(4, 0))
        ttk.Button(r1, text="Inicio del hurto", width=14, command=self.set_start).pack(side="left")
        self.start_lbl = ttk.Label(r1, text="—", width=7)
        self.start_lbl.pack(side="left", padx=6)
        r2 = ttk.Frame(mk)
        r2.pack(fill="x", pady=(4, 0))
        ttk.Button(r2, text="Fin del hurto", width=14, command=self.set_end).pack(side="left")
        self.end_lbl = ttk.Label(r2, text="—", width=7)
        self.end_lbl.pack(side="left", padx=6)
        self.who_lbl = ttk.Label(mk, text="", style="Suave.TLabel", wraplength=250, justify="left")
        self.who_lbl.pack(anchor="w", pady=(6, 0))
        r3 = ttk.Frame(mk)
        r3.pack(fill="x", pady=(4, 0))
        ttk.Label(r3, text="Cómo").pack(side="left")
        self.kind = tk.StringVar(value=KIND_NAMES[None])
        ttk.Combobox(r3, textvariable=self.kind, values=list(KIND_NAMES.values()), state="readonly",
                     width=19).pack(side="left", padx=6)
        ttk.Button(mk, text="＋ Agregar la marca", style="Primario.TButton", command=self.add_mark).pack(
            anchor="w", pady=(6, 4))
        self.mtree = ttk.Treeview(mk, columns=("desde", "hasta", "persona", "como"), show="headings", height=3,
                                  selectmode="browse")
        for col, txt, w in (("desde", "Desde", 60), ("hasta", "Hasta", 58), ("persona", "Persona", 68),
                            ("como", "Cómo", 86)):
            self.mtree.heading(col, text=txt, anchor="w")
            self.mtree.column(col, width=w, stretch=col == "como")
        self.mtree.pack(fill="x")
        self.mtree.bind("<Double-1>", lambda e: self.goto_mark())
        r4 = ttk.Frame(mk)
        r4.pack(fill="x", pady=(4, 0))
        ttk.Button(r4, text="Quitar", style="Chico.TButton", width=7, command=self.remove_mark).pack(side="left")
        ttk.Button(r4, text="Ir a la marca", style="Chico.TButton", width=12, command=self.goto_mark).pack(
            side="left", padx=4)
        ttk.Button(mk, text="Este clip es normal (no hay hurto)", command=self.set_normal).pack(anchor="w",
                                                                                             pady=(8, 0))
        r5 = ttk.Frame(mk)
        r5.pack(fill="x", pady=(6, 0))
        self.save_btn = ttk.Button(r5, text="Guardar marcas", style="Primario.TButton", command=self.save_marks)
        self.save_btn.pack(side="left")
        self.save_lbl = ttk.Label(r5, text="", style="Suave.TLabel")
        self.save_lbl.pack(side="left", padx=6)
        self._show_who()
        self._draw_empty()

    def _train_panel(self, nb, title: str, ia: bool) -> dict:
        """Una de las dos IA: cuántos ejemplos hay, «Entrenar» (al lado, qué falta o cómo anduvo) y sus modelos con
        «Activar» / «Apagar»."""
        ttk = self.ttk
        p = ttk.Frame(nb, padding=(6, 4, 6, 2))
        nb.add(p, text=f"  {title}  ")
        info = ttk.Label(p, text="Cargando…", wraplength=440, justify="left")
        info.pack(anchor="w", fill="x")
        row = ttk.Frame(p)
        row.pack(fill="x", pady=(4, 4))
        btn = ttk.Button(row, text="Entrenar", style="Primario.TButton", width=8,
                         command=self.train_ia if ia else self.train_filter)
        btn.pack(side="left", anchor="n")
        result = ttk.Label(row, text="", wraplength=356, justify="left")
        result.pack(side="left", padx=(8, 0), fill="x", expand=True)
        cols = (("fecha", "Fecha", 94), ("para", "Para", 98), ("res", "Cómo anduvo", 196), ("uso", "En uso", 64))
        tree = self.ttk.Treeview(p, columns=[c[0] for c in cols], show="headings", height=2, selectmode="browse")
        for col, txt, w in cols:
            tree.heading(col, text=txt, anchor="w")
            tree.column(col, width=w, minwidth=40, stretch=col == "res")
        tree.pack(fill="x")
        bb = ttk.Frame(p)
        bb.pack(fill="x", pady=(4, 0))
        ttk.Button(bb, text="Activar", style="Chico.TButton", width=8,
                   command=lambda: self.set_model(ia, True)).pack(side="left")
        ttk.Button(bb, text="Apagar", style="Chico.TButton", width=8,
                   command=lambda: self.set_model(ia, False)).pack(side="left", padx=4)
        ttk.Label(bb, text="Activado, lo usan las máquinas." if not ia else "Activada, la usan las máquinas.",
                  style="Suave.TLabel").pack(side="left", padx=(6, 0))
        return {"info": info, "btn": btn, "result": result, "tree": tree, "shown": False}

    # ---------- carga ----------
    def clients_changed(self) -> None:
        """Llegaron los clientes: el selector de alcance los ofrece (sin cambiar lo elegido)."""
        names = [ALL_CLIENTS] + [c["name"] for c in self.app.clients]
        self.scope_cb.configure(values=names)
        if self.scope.get() not in names:
            self.scope.set(ALL_CLIENTS)

    def on_show(self) -> None:
        """Se abrió la pestaña: lo último de la nube."""
        self.clients_changed()
        self.load()

    def focus_client(self, cid: int | None) -> None:
        """Desde «Clientes»: la pestaña con ese cliente elegido."""
        self.clients_changed()
        self.scope.set(self._client_name(cid) or ALL_CLIENTS)
        try:
            if self.app.notebook.select() != str(self.frame):
                return self.app.notebook.select(self.frame)  # (al cambiar de pestaña se carga sola)
        except Exception:  # noqa: BLE001
            pass
        self.load()

    def load(self) -> None:
        cid = self.scope_id()
        if cid != self._loaded_scope:  # otro alcance: el resultado de antes ya no corresponde
            self.ia["shown"] = self.flt["shown"] = False
        self._loaded_scope = cid
        self._req += 1
        req = self._req

        def done(res) -> None:
            if req != self._req:
                return  # se eligió otro alcance mientras tanto
            self.clips = _items(res)
            self._fill_clips()
            self._show_ia_counts()
        self.app.bg.run(lambda: self.app.api.get("/api/training/clips", client_id=cid), done, self._fail)
        self.load_ia()
        self.load_filter()

    def _fill_clips(self) -> None:
        prev = self.tree.selection()
        self.tree.delete(*self.tree.get_children())
        local = self.local
        one = self.scope_id() is not None  # con un cliente elegido, la columna dice solo la cámara
        self.tree.heading("donde", text="Cámara" if one else "Cliente · cámara")
        for c in self.clips:
            cam = c.get("camera_name") or (self._cam(c.get("camera_id")) or {}).get("name") or ""
            cli = "" if one else (c.get("client_name") or self._client_name(c.get("client_id")))
            where = " · ".join(x for x in (cli, cam) if x) or "—"
            name = str(c.get("name") or "")
            self.tree.insert("", "end", iid=str(c["id"]), text=name, values=(
                where, fmt_t(_num(c.get("duration_s"))), verdict_text(c), uploader_of(c),
                "✔" if local.path_of(c.get("uuid")) else ""))
        n_here = sum(1 for c in self.clips if local.path_of(c.get("uuid")))
        self.count_lbl.configure(text=f"{len(self.clips)} · {n_here} en esta PC")
        if prev and self.tree.exists(prev[0]):
            self.tree.selection_set(prev)
        elif self.clip is not None and self.tree.exists(str(self.clip["id"])):
            self.tree.selection_set(str(self.clip["id"]))

    def _row_of(self, clip: dict) -> None:
        """Pone al día la fila de un clip (después de guardar sus marcas)."""
        iid = str(clip["id"])
        if self.tree.exists(iid):
            self.tree.set(iid, "estado", verdict_text(clip))

    def load_ia(self) -> None:
        cid = self.scope_id()

        def done(summary) -> None:
            if cid != self.scope_id():
                return
            self.ia_summary = summary if isinstance(summary, dict) else None
            # (del alcance: con un cliente, los suyos y los de todos los clientes; con «Todos», todos)
            self.ia_models = _items((summary or {}).get("models") if isinstance(summary, dict) else None)
            self._show_ia_counts()
            self._fill_models(True)

        def fail(exc: Exception) -> None:
            if getattr(exc, "status", 0) == 404:
                self.ia["info"].configure(text="El servidor todavía no tiene la IA de movimientos: falta actualizarlo.")
                return
            self.ia["info"].configure(text=f"No se pudo traer lo de la IA de movimientos: {exc}")
        self.app.bg.run(lambda: self.app.api.get("/api/training/ia/summary", client_id=cid), done, fail)

    def _show_ia_counts(self) -> None:
        c = summary_counts(self.ia_summary, self.clips)
        advice, enough = ia_advice(c)
        self.ia["info"].configure(text=ia_counts_text(c))
        if not self.ia["shown"]:  # (hasta que se entrene: qué falta, o que alcanza)
            self.ia["result"].configure(text=advice, foreground="" if enough else "#b45309")

    def load_filter(self) -> None:
        cid = self.scope_id()

        def done(res) -> None:
            if cid != self.scope_id():
                return
            summary, models = res
            text, advice, enough = filter_text(filter_counts(summary, cid))
            self.flt["info"].configure(text=text)
            if not self.flt["shown"]:
                self.flt["result"].configure(text=advice, foreground="" if enough else "#b45309")
            # (como en la IA: con un cliente, los suyos y los de todos; con «Todos», todos)
            self.filter_models = [m for m in _items(models) if cid is None or m.get("client_id") in (cid, None)]
            self._fill_models(False)
        self.app.bg.run(lambda: (self.app.api.get("/api/training/summary"), self.app.api.get("/api/models")), done,
                        lambda e: self.flt["info"].configure(text=f"No se pudo traer lo del filtro: {e}"))

    def _fill_models(self, ia: bool) -> None:
        panel = self.ia if ia else self.flt
        tree = panel["tree"]
        tree.delete(*tree.get_children())
        for m in self.ia_models if ia else self.filter_models:
            met = m.get("metrics") or {}
            if ia:
                use = "sí" if m.get("active") else ("vieja" if m.get("usable") is False else "")
                vals = (model_date(m), scope_name(m), ia_model_text(met), use)
            else:
                vals = (model_date(m), scope_name(m), filter_model_text(m), "sí" if m.get("active") else "")
            tree.insert("", "end", iid=str(m["id"]), values=vals)

    # ---------- entrenar ----------
    def train_ia(self) -> None:
        cid = self.scope_id()
        panel = self.ia
        panel["btn"].configure(text="Entrenando…", state="disabled")
        panel["result"].configure(text="Entrenando con los clips marcados… (unos segundos)", foreground="")

        def done(m) -> None:
            panel["btn"].configure(text="Entrenar", state="normal")
            met = (m or {}).get("metrics") or {}
            panel["shown"] = True
            panel["result"].configure(text=ia_result_text(met) + ". Todavía no está en uso: elegila y «Activar».",
                                      foreground="")
            self.load_ia()

        def fail(exc: Exception) -> None:
            panel["btn"].configure(text="Entrenar", state="normal")
            panel["shown"] = True
            if getattr(exc, "status", 0) == 422:  # pocos datos: el motivo, en palabras (core/hurto_ia.NotEnoughData)
                return panel["result"].configure(text=str(exc), foreground="#b45309")
            panel["result"].configure(text="")
            self._fail(exc)
        self.app.bg.run(lambda: self.app.api.post("/api/training/ia/run", {"client_id": cid,
                                                                             "target_recall": TARGET_RECALL}),
                        done, fail)

    def train_filter(self) -> None:
        cid = self.scope_id()
        panel = self.flt
        panel["btn"].configure(text="Entrenando…", state="disabled")

        def done(m) -> None:
            panel["btn"].configure(text="Entrenar", state="normal")
            panel["shown"] = True
            panel["result"].configure(text=f"Con {(m or {}).get('n_samples', 0)} ejemplos: "
                                           + filter_result_text((m or {}).get("metrics"))
                                           + ". Todavía no está en uso: elegilo y «Activar».", foreground="")
            self.load_filter()

        def fail(exc: Exception) -> None:
            panel["btn"].configure(text="Entrenar", state="normal")
            panel["shown"] = True
            if getattr(exc, "status", 0) == 422:
                return panel["result"].configure(text=str(exc), foreground="#b45309")
            self._fail(exc)
        self.app.bg.run(lambda: self.app.api.post("/api/training/run", {"client_id": cid}), done, fail)

    def set_model(self, ia: bool, on: bool) -> None:
        panel = self.ia if ia else self.flt
        sel = panel["tree"].selection()
        if not sel:
            return self._say("info", "Elegí un modelo de la lista.")
        models = self.ia_models if ia else self.filter_models
        m = next((x for x in models if str(x["id"]) == sel[0]), None)
        if m is None:
            return
        if on and ia:
            met = m.get("metrics") or {}
            who = "todos los clientes" if m.get("client_id") is None else scope_name(m)
            if not self.app.mb.askyesno(NOMBRE, f"¿Activar la IA de movimientos para {who}?\n\n"
                                                f"{ia_result_text(met)}.\n\nLas máquinas van a dar alertas «Posible "
                                                "hurto (IA de movimientos)» en las cámaras que miden hurtos (la toman "
                                                "en menos de un minuto). Con pocos ejemplos puede dar falsas alarmas: "
                                                "revisalas en «Alertas» y etiquetalas."):
                return
        base = "/api/training/ia/models" if ia else "/api/models"
        action = "activate" if on else "deactivate"
        self.app.bg.run(lambda: self.app.api.post(f"{base}/{m['id']}/{action}"),
                        lambda _r: self.load_ia() if ia else self.load_filter(), self._fail)

    # ---------- subir ----------
    def upload_dialog(self) -> None:
        """«⬆ Subir clips…»: los videos, de qué cliente y cámara son y si son normales; después se analizan acá."""
        if self.uploading:
            return self._say("info", "Ya se están analizando clips: esperá a que termine (o «■ Frenar»).")
        from tkinter import filedialog

        from .biblioteca import VIDEO_TYPES, default_folder
        paths = filedialog.askopenfilenames(parent=self.root, title="Clips para entrenar la IA (hurtos y ratos "
                                            "normales)", initialdir=str(default_folder()),
                                            filetypes=[VIDEO_TYPES, ("Todos los archivos", "*.*")])
        if not paths:
            return
        answer = self.ask_upload(list(paths))
        if answer is not None:
            self.start_upload(list(paths), *answer)

    def ask_upload(self, paths: list[str]):
        """De qué cliente y cámara son los clips y qué hay en ellos. Devuelve (client_id, cámara, veredicto) o None."""
        tk, ttk = self.tk, self.ttk
        top, frm = self.app._dialog("Subir clips para entrenar", 520)
        top.grab_set()
        out: list = [None]
        names = ", ".join(Path(p).name for p in paths[:3]) + (f" y {len(paths) - 3} más" if len(paths) > 3 else "")
        ttk.Label(frm, text=f"{plural(len(paths), 'video')}: {names}", wraplength=480, justify="left",
                  font=("Segoe UI", 10, "bold")).grid(row=0, column=0, columnspan=2, sticky="w")
        cname = tk.StringVar(value=self._client_name(self.scope_id()) or (
            self.app.clients[0]["name"] if self.app.clients else NO_CLIENT))
        camname = tk.StringVar(value=NO_CAMERA)
        ttk.Label(frm, text="Cliente").grid(row=1, column=0, sticky="w", pady=(10, 3), padx=(0, 10))
        ccb = ttk.Combobox(frm, textvariable=cname, values=[c["name"] for c in self.app.clients] + [NO_CLIENT],
                           state="readonly", width=32)
        ccb.grid(row=1, column=1, sticky="w", pady=(10, 3))
        ttk.Label(frm, text="Cámara").grid(row=2, column=0, sticky="w", pady=3, padx=(0, 10))
        kcb = ttk.Combobox(frm, textvariable=camname, state="readonly", width=32)
        kcb.grid(row=2, column=1, sticky="w", pady=3)

        def cams() -> list[dict]:
            cid = next((c["id"] for c in self.app.clients if c["name"] == cname.get()), None)
            return [k for k in self.app.cameras if cid is not None and k.get("client_id") == cid]

        def refill(_e=None) -> None:
            names_ = [k["name"] for k in cams()]
            kcb.configure(values=[NO_CAMERA] + names_)
            if camname.get() not in names_:
                camname.set(names_[0] if len(names_) == 1 else NO_CAMERA)
        ccb.bind("<<ComboboxSelected>>", refill)
        refill()
        ttk.Label(frm, text="Con la cámara se usa su sensibilidad para ver personas, y después se puede probar el clip "
                            "con sus reglas.", style="Suave.TLabel", wraplength=480, justify="left").grid(
            row=3, column=0, columnspan=2, sticky="w")
        verdict = tk.StringVar(value="sin_marcar")
        ttk.Label(frm, text="¿Qué hay en estos clips?").grid(row=4, column=0, columnspan=2, sticky="w", pady=(10, 2))
        ttk.Radiobutton(frm, text="Un hurto, o todavía no sé (lo marco después en la lista)", variable=verdict,
                        value="sin_marcar").grid(row=5, column=0, columnspan=2, sticky="w")
        ttk.Radiobutton(frm, text="Nada raro: un rato normal, sin hurto", variable=verdict, value="normal").grid(
            row=6, column=0, columnspan=2, sticky="w")
        ttk.Label(frm, text="Se analizan en esta PC, de a uno, con YOLO y la postura (sin placa de video, un clip de 1 "
                            "minuto tarda más o menos 1 minuto). El video no se sube: a la nube va solo lo que vio la "
                            "IA.", style="Suave.TLabel", wraplength=480, justify="left").grid(
            row=7, column=0, columnspan=2, sticky="w", pady=(10, 0))

        def ok() -> None:
            cid = next((c["id"] for c in self.app.clients if c["name"] == cname.get()), None)
            cam = next((k for k in cams() if k["name"] == camname.get()), None)
            out[0] = (cid, cam, verdict.get())
            top.destroy()
        bb = ttk.Frame(frm)
        bb.grid(row=8, column=0, columnspan=2, sticky="e", pady=(12, 0))
        ttk.Button(bb, text="Cancelar", command=top.destroy).pack(side="right")
        ttk.Button(bb, text="Analizar y subir", style="Primario.TButton", command=ok).pack(side="right", padx=6)
        self._ask_top = top  # (las pruebas lo miden y lo contestan)
        top.wait_window()
        return out[0]

    def start_upload(self, paths: list[str], client_id: int | None, cam: dict | None, verdict: str) -> None:
        """Analiza los videos en otro hilo, de a uno, y sube cada uno al terminarlo."""
        self.uploading = True
        self._stop = threading.Event()
        self.upload_log = []
        known = {c.get("uuid"): c for c in self.clips if c.get("uuid")}
        self.up_lbl.configure(text="Preparando la IA de esta PC…")
        self.stop_btn.pack(side="right")
        self.up_bar.pack(fill="x", pady=(2, 0))
        self.up_bar["value"] = 0
        threading.Thread(target=self._upload_work, args=(list(paths), client_id, cam, verdict, known, self._stop),
                         name="ia-subir", daemon=True).start()
        self._ensure_poll()

    def stop_upload(self) -> None:
        self._stop.set()
        self.up_lbl.configure(text="Frenando…")

    def _ai(self):
        if self.ai_setup is not None:
            return self.ai_setup()
        agent = self.app.agent_of() if getattr(self.app, "agent_of", None) else None
        if agent is None:
            raise ClipError("la app todavía está arrancando el analizador: probá en unos segundos")
        from .worker import shared_pose
        return agent._detector("yolo"), shared_pose()

    def _windows(self, frames: list[dict], cam: dict | None) -> tuple[list[dict], int]:
        """Las ventanas de rasgos por persona (core/hurto_ia.py), con las zonas de la cámara si tiene. Los rasgos van
        redondeados (el pedido no pasa los 4,5 MB de Vercel)."""
        try:
            from core import hurto_ia
        except ImportError:
            hurto_ia = None
            if self.windows_fn is None:
                raise
        fn = self.windows_fn or hurto_ia.person_windows
        # las zonas como las guarda el servidor (góndola, caja, salida…: algunos rasgos son «frente a la góndola»)
        wins = fn(frames, list((cam or {}).get("zones") or []) or None)
        out = [{"track_id": int(w["track_id"]), "t0": round(float(w["t0"]), 2), "t1": round(float(w["t1"]), 2),
                "f": [round(float(x), 3) for x in w["f"]]} for w in wins]
        return out, int(getattr(hurto_ia, "FEATURE_VERSION", 1))

    def _upload_work(self, paths, client_id, cam, verdict, known, stop) -> None:
        put = self._q.put
        try:
            detector, pose = self._ai()
        except Exception as exc:  # noqa: BLE001 - se dice en la ventana
            put(("finished", f"No se pudo preparar la IA de esta PC: {exc}"))
            return
        detection = DetectionConfig.from_dict((cam or {}).get("detection"))
        cam_id = (cam or {}).get("id")
        n = len(paths)
        try:  # todos los de la nube (no solo los del cliente elegido): uno ya subido no se vuelve a analizar
            known = {c.get("uuid"): c for c in _items(self.app.api.get("/api/training/clips")) if c.get("uuid")}
        except Exception as exc:  # noqa: BLE001 - sin la lista, con lo que ya se tenía
            log.warning("No se pudo traer la lista de clips antes de subir: %s", exc)
        for i, path in enumerate(paths):
            if stop.is_set():
                break
            name = Path(path).name
            put(("start", i, n, name))
            try:
                uuid = clip_uuid(path)
                if uuid in known:  # ya subido (desde esta u otra PC): queda vinculado al video
                    self.local.link(uuid, path)
                    put(("linked", known[uuid], name))
                    continue
                res = self.analyze_fn(path, detector, pose, fps=FPS, detection=detection, stop=stop,
                                      progress=lambda frac, _s, i=i, name=name: put(("progress", i, n, name, frac)))
                put(("step", i, n, name, "subiendo"))
                windows, version = self._windows(res["frames"], cam)
                body = {"uuid": uuid, "name": name[:120], "client_id": client_id, "camera_id": cam_id,
                        "duration_s": res["duration_s"], "fps": res["fps"], "width": res["width"],
                        "height": res["height"], "verdict": verdict, "marks": [], "windows": windows,
                        "feature_version": version, "people": res["people"]}
                if len(json.dumps(body, separators=(",", ":"))) > MAX_BODY:
                    raise ClipError(f"es muy largo para tanta gente ({len(windows)} ventanas de personas): cortalo en "
                                    "partes más cortas")
                row = self.app.api.post("/api/training/clips", body) or {}
                clip = row["clip"] if isinstance(row.get("clip"), dict) else row
                clip = {**body, **{k: v for k, v in clip.items() if k != "windows"}}
                clip.pop("windows", None)
                data = pack({"uuid": uuid, "name": name, "fps": res["fps"], "width": res["width"],
                             "height": res["height"], "duration_s": res["duration_s"],
                             "video_fps": res.get("video_fps")}, res["frames"])
                self.local.link(uuid, path)
                self.local.save_data(uuid, data)
                note = ""
                url = row.get("data_upload")
                if isinstance(url, dict):
                    url = url.get("url") or url.get("upload_url")
                if url:
                    try:
                        self.app.api.upload(url, data, "application/gzip")
                    except Exception as exc:  # noqa: BLE001 - el clip quedó: solo se ve dibujado en esta PC
                        log.warning("No se subieron los cuadros de %s: %s", name, exc)
                        note = "los cuadros para dibujarlo no se subieron (se ve solo en esta PC)"
                if not windows:
                    note = ("la IA no vio bien la postura de nadie (personas chicas, de espaldas o de noche): no sirve "
                            "para la IA de movimientos") + (f"; {note}" if note else "")
                put(("done", clip, name, note))
            except ClipError as exc:
                if stop.is_set():
                    break
                put(("error", name, str(exc)))
            except Exception as exc:  # noqa: BLE001 - un clip que falla no frena los demás
                log.exception("No se pudo subir el clip %s", name)
                put(("error", name, str(exc)))
        put(("finished", "Frenado." if stop.is_set() else None))

    def _poll(self) -> None:
        """Cada 50 ms mientras haga falta: lo que avisa la subida, la reproducción y los cuadros que llegan."""
        self._poll_job = None
        try:
            while True:
                msg = self._q.get_nowait()
                self._on_msg(msg)
        except queue.Empty:
            pass
        if self.playing and self.frames:
            now = time.monotonic()
            self.pos = min(self.duration, self.pos + (now - self._last_tick))
            self._last_tick = now
            if self.pos >= self.duration - 1e-3:
                self.playing = False
                self.play_btn.configure(text="▶")
            self._goto(self.pos, scale=True)
        if self.reader is not None:
            got = self.reader.take()
            if got is not None:
                self.base_t, self.base = got
                self._redraw()
        if self.uploading or self.playing or (self.reader is not None and self.reader.waiting):
            self._ensure_poll()

    def _on_msg(self, msg: tuple) -> None:
        kind = msg[0]
        if kind == "start":
            _k, i, n, name = msg
            self.up_lbl.configure(text=f"Analizando {i + 1} de {n} · {name}")
            self.up_bar["value"] = 100 * i / max(n, 1)
        elif kind == "progress":
            _k, i, n, name, frac = msg
            self.up_lbl.configure(text=f"Analizando {i + 1} de {n} · {name} · {frac * 100:.0f} %")
            self.up_bar["value"] = 100 * (i + frac) / max(n, 1)
        elif kind == "step":
            _k, i, n, name, what = msg
            self.up_lbl.configure(text=f"{'Subiendo' if what == 'subiendo' else what} {i + 1} de {n} · {name}")
        elif kind in ("done", "linked"):
            clip = msg[1]
            self.upload_log.append((msg[2], "ya estaba subido: quedó vinculado a este video" if kind == "linked"
                                    else ("subido" + (f" ({msg[3]})" if msg[3] else ""))))
            self.clips = [c for c in self.clips if c.get("id") != clip.get("id")]
            if clip.get("id") is not None and (self.scope_id() is None or clip.get("client_id") == self.scope_id()):
                self.clips.insert(0, clip)
            self._fill_clips()
            self._show_ia_counts()
            if kind == "done" and clip.get("id") is not None and self.tree.exists(str(clip["id"])):
                self.tree.selection_set(str(clip["id"]))
                self.tree.see(str(clip["id"]))
        elif kind == "error":
            self.upload_log.append((msg[1], f"no se pudo: {msg[2]}"))
        elif kind == "finished":
            self._upload_finished(msg[1])

    def _upload_finished(self, problem: str | None) -> None:
        self.uploading = False
        self.stop_btn.pack_forget()
        self.up_bar.pack_forget()
        ok = [n for n, what in self.upload_log if what.startswith("subido") or what.startswith("ya estaba")]
        bad = [(n, what) for n, what in self.upload_log if not (what.startswith("subido") or what.startswith("ya "))]
        notes = [(n, what) for n, what in self.upload_log if what.startswith("subido (")]
        if not self.upload_log:
            if problem == "Frenado.":
                self.up_lbl.configure(text="Frenado: no se subió nada.")
            else:
                self.up_lbl.configure(text=IDLE_HINT)
                if problem:
                    self._say("error", problem)
            return
        text = f"✔ {plural(len(ok), 'clip subido', 'clips subidos')}" + (
            f" · {len(bad)} con problemas" if bad else "") + (" · frenado" if problem == "Frenado." else "")
        if ok:
            text += ". Elegí cada uno y marcá el hurto (o «Este clip es normal»)."
        self.up_lbl.configure(text=text)
        if bad or notes:
            lines = [f"• {n}: {w}" for n, w in bad + notes]
            self._say("warning", "Algunos clips:\n" + "\n".join(lines))
        self.load_ia()

    # ---------- clip elegido ----------
    @property
    def duration(self) -> float:
        d = _num((self.clip or {}).get("duration_s"))
        return max(d, self.times[-1] if self.times else 0.0)

    def _sel(self) -> dict | None:
        sel = self.tree.selection()
        return next((c for c in self.clips if str(c.get("id")) == (sel[0] if sel else "")), None)

    def show_clip(self) -> None:
        clip = self._sel()
        if clip is None or (self.clip is not None and clip.get("id") == self.clip.get("id")):
            return
        if self.dirty and self.clip is not None:
            if self.app.mb.askyesno(NOMBRE, f"Las marcas de «{self.clip.get('name')}» no se guardaron. ¿Guardarlas "
                                            "antes de cambiar de clip?"):
                self.save_marks()
        self._open(clip)

    def _open(self, clip: dict) -> None:
        self._close_video()
        self.clip = clip
        self.marks = copy.deepcopy(clip.get("marks") or [])
        self.dirty = False
        self.sel_track, self.m_start, self.m_end = None, None, None
        self.pos, self.playing, self.base, self.base_t, self._shown_rec = 0.0, False, None, 0.0, None
        self.play_btn.configure(text="▶")
        self.frames, self.times = [], []
        uuid = clip.get("uuid") or ""
        self.video = self.local.path_of(uuid)
        self.title.configure(text=self._short(str(clip.get("name") or "Clip")))
        self._fill_marks()
        self._show_who()
        self.save_lbl.configure(text="")
        data = self.local.load_data(uuid) if uuid else None
        self._frames_error = None
        if data is not None:
            self._set_frames(data.get("frames") or [])
        if self.video:
            self.reader = FrameReader(self.video)
            self.reader.want(0.0)
            self._ensure_poll()
        self._show_info()
        self._set_scale()
        self._redraw()
        if data is None:
            self._fetch_frames(clip)

    @staticmethod
    def _short(name: str, n: int = 46) -> str:
        return name if len(name) <= n else name[: n - 1].rstrip() + "…"

    def _show_info(self) -> None:
        c = self.clip or {}
        cam = c.get("camera_name") or (self._cam(c.get("camera_id")) or {}).get("name")
        cli = c.get("client_name") or self._client_name(c.get("client_id"))
        parts = [" · ".join(x for x in (cli, cam) if x) or "sin cliente ni cámara",
                 fmt_t(_num(c.get("duration_s"))), plural(int(_num(c.get("people"))), "persona")]
        who = uploader_of(c)
        me = ((getattr(self.app.api, "user", None) or {}).get("username") or "")
        if who:
            parts.append(f"subió {who}")
        if self.video:
            parts.append("el video está en esta PC")
        elif who and who != me:
            parts.append(f"el video está en la PC de {who}: acá se ven solo los datos")
        else:
            parts.append("el video no está en esta PC: se ven solo los datos")
        if self._frames_error:
            parts.append("sin los cuadros (ver la imagen)")
        elif not self.frames and not self.video:
            parts.append("bajando los datos…")
        self.info.configure(text=" · ".join(p for p in parts if p))

    def _fetch_frames(self, clip: dict) -> None:
        """Los cuadros del clip desde la nube (si no están en esta PC): quedan guardados acá para la próxima."""
        cid, uuid = clip.get("id"), clip.get("uuid") or ""

        def work():
            res = self.app.api.get(f"/api/training/clips/{cid}/data")
            if isinstance(res, dict):
                url = res.get("url") or res.get("download_url")
                if not url:
                    raise ValueError("este clip no tiene los datos de los cuadros en la nube")
                res = self.app.api.fetch(url)
            data = bytes(res)
            body = unpack(data)
            if uuid:
                self.local.save_data(uuid, data)
            return body

        def done(body) -> None:
            if self.clip is None or self.clip.get("id") != cid:
                return
            self._set_frames(body.get("frames") or [])
            self._show_info()
            self._set_scale()
            self._redraw()

        def fail(exc: Exception) -> None:
            if self.clip is not None and self.clip.get("id") == cid:
                self._frames_error = str(exc)
                self._show_info()
                self._redraw()
        self.app.bg.run(work, done, fail)

    def _set_frames(self, frames: list[dict]) -> None:
        self.frames = [f for f in frames if isinstance(f, dict) and "t" in f]
        self.times = [float(f["t"]) for f in self.frames]

    def _set_scale(self) -> None:
        self._setting_scale = True
        try:
            self.scale.configure(to=max(self.duration, 0.1))
            self.scale.set(self.pos)
        finally:
            self._setting_scale = False
        self._show_time()

    def _show_time(self) -> None:
        self.time_lbl.configure(text=f"{fmt_t(self.pos, True)} / {fmt_t(self.duration)}")

    def _close_video(self) -> None:
        if self.reader is not None:
            self.reader.close()
        self.reader = None

    def close(self) -> None:
        """La app se cierra: soltar el video."""
        self._close_video()
        self._stop.set()

    def _rec(self, t: float | None = None) -> dict | None:
        i = nearest(self.times, self.pos if t is None else t)
        return self.frames[i] if i >= 0 else None

    def _active(self, t: float) -> list[dict]:
        return [m for m in self.marks if _num(m.get("start")) - 1e-6 <= t <= _num(m.get("end")) + 1e-6]

    def _draw_empty(self) -> None:
        self.canvas.delete("all")
        self.canvas.create_text(PLAYER_W // 2, PLAYER_H // 2, fill="#c8ced8", width=PLAYER_W - 60,
                                text="Elegí un clip de la lista (o subí uno con «⬆ Subir clips…»).")

    def _redraw(self) -> None:
        if self.clip is None:
            return self._draw_empty()
        # con video, los recuadros son los del momento del cuadro que se ve (el video llega un poco después)
        t = self.base_t if self.base is not None else self.pos
        rec = self._rec(t)
        if rec is None and self.base is None:
            self.canvas.delete("all")
            text = ("Abriendo el video…" if self.video and (self.reader is None or self.reader.error is None) else
                    f"No hay nada para mostrar: el video no está en esta PC y no se pudieron bajar sus cuadros "
                    f"({self._frames_error})." if self._frames_error else "Bajando los datos del clip…")
            self.canvas.create_text(PLAYER_W // 2, PLAYER_H // 2, fill="#c8ced8", width=PLAYER_W - 60, text=text)
            return
        self._shown_rec, self._shown_t = rec, t
        n = len((rec or {}).get("people") or [])
        size = (int(_num((self.clip or {}).get("width"))), int(_num((self.clip or {}).get("height"))))
        img, k = render(rec, self.base, self.sel_track, self._active(t),
                        f"{fmt_t(t, True)}   personas: {n}" + ("" if self.video else "   (sin video: solo datos)"),
                        size)
        h, w = img.shape[:2]
        self._k, self._ox, self._oy = k, (PLAYER_W - w) // 2, (PLAYER_H - h) // 2
        ok, ppm = cv2.imencode(".ppm", img)
        if not ok:
            return
        self._photo = self.tk.PhotoImage(data=ppm.tobytes())
        self.canvas.delete("all")
        self.canvas.create_image(self._ox, self._oy, image=self._photo, anchor="nw")

    def _goto(self, t: float, scale: bool = True) -> None:
        self.pos = max(0.0, min(float(t), self.duration))
        if scale:
            self._setting_scale = True
            try:
                self.scale.set(self.pos)
            finally:
                self._setting_scale = False
        self._show_time()
        if self.reader is not None and self.reader.error is None:
            self.reader.want(self.pos)
            self._ensure_poll()
            if self.playing:
                return  # se dibuja cuando llegue el cuadro (así el video y los recuadros van juntos)
        self._redraw()

    def on_scale(self, value) -> None:
        if self._setting_scale or self.clip is None:
            return
        self._goto(_num(value), scale=False)

    def step(self, delta: float) -> None:
        if self.clip is not None:
            self.playing = False
            self.play_btn.configure(text="▶")
            self._goto(self.pos + delta)

    def toggle_play(self) -> None:
        if self.clip is None or not (self.frames or self.video):
            return
        self.playing = not self.playing
        if self.playing and self.pos >= self.duration - 1e-3:
            self._goto(0.0)
        self._last_tick = time.monotonic()
        self.play_btn.configure(text="⏸" if self.playing else "▶")
        if self.playing:
            self._ensure_poll()

    def on_click(self, ev) -> None:
        """Clic en la imagen: elegir a la persona (para marcarla)."""
        if self.clip is None:
            return
        rec = self._shown_rec
        if not (rec or {}).get("people"):
            return self._say("info", "En este momento la IA no ve a nadie. Llevá la barra hasta donde se vea a la "
                                     "persona y hacé clic sobre su recuadro.")
        x, y = (ev.x - self._ox) / self._k, (ev.y - self._oy) / self._k
        pid = person_at(rec, x, y, PICK_SLACK_PX / self._k)
        if pid is None:
            return self._say("info", "Ahí no hay ninguna persona. Hacé clic sobre el recuadro de la que roba.")
        self.sel_track = pid
        self._show_who()
        self._redraw()

    def _show_who(self) -> None:
        self.who_lbl.configure(text=(f"Persona: #{self.sel_track} (elegida en la imagen)" if self.sel_track is not None
                                     else "Persona: hacé clic sobre la que roba, en la imagen"),
                               style="TLabel" if self.sel_track is not None else "Suave.TLabel")
        self.start_lbl.configure(text=fmt_t(self.m_start, True) if self.m_start is not None else "—")
        self.end_lbl.configure(text=fmt_t(self.m_end, True) if self.m_end is not None else "—")

    # ---------- marcas ----------
    def set_start(self) -> None:
        if self.clip is None:
            return
        self.m_start = round(self.pos, 1)
        if self.m_end is not None and self.m_end <= self.m_start:
            self.m_end = None
        self._show_who()

    def set_end(self) -> None:
        if self.clip is None:
            return
        if self.m_start is not None and self.pos <= self.m_start:
            return self._say("info", "El fin tiene que ser después del inicio: llevá la barra más adelante.")
        self.m_end = round(self.pos, 1)
        self._show_who()

    def add_mark(self) -> None:
        if self.clip is None:
            return self._say("info", "Elegí un clip de la lista.")
        if self.m_start is None or self.m_end is None:
            return self._say("info", "Primero «Inicio del hurto» y «Fin del hurto»: llevá la barra a cada momento y "
                                     "apretá el botón.")
        if self.m_end <= self.m_start:
            return self._say("info", "El fin tiene que ser después del inicio.")
        if self.sel_track is None and not self.app.mb.askyesno(
                NOMBRE, "No elegiste a la persona que roba (clic sobre ella en la imagen). ¿Agregar la marca igual? "
                        "Vale para cualquier persona que esté en ese momento: sirve menos para entrenar."):
            return
        kind = next((k for k, v in KIND_NAMES.items() if v == self.kind.get()), None)
        self.marks.append({"start": float(self.m_start), "end": float(min(self.m_end, self.duration or self.m_end)),
                           "track_id": self.sel_track, "kind": kind})
        self.marks.sort(key=lambda m: m["start"])
        self.m_start = self.m_end = None
        self.sel_track = None
        self._mark_changed()

    def _mark_changed(self) -> None:
        self.dirty = True
        self.save_lbl.configure(text="sin guardar", style="Aviso.TLabel")
        self._fill_marks()
        self._show_who()
        self._redraw()

    def _fill_marks(self) -> None:
        self.mtree.delete(*self.mtree.get_children())
        for i, m in enumerate(self.marks):
            self.mtree.insert("", "end", iid=str(i), values=(
                fmt_t(m.get("start"), True), fmt_t(m.get("end"), True),
                f"#{m['track_id']}" if m.get("track_id") is not None else "cualquiera",
                KIND_NAMES.get(m.get("kind"), "No sé")))

    def _sel_mark(self) -> int | None:
        sel = self.mtree.selection()
        return int(sel[0]) if sel else None

    def remove_mark(self) -> None:
        i = self._sel_mark()
        if i is None:
            return self._say("info", "Elegí una marca de la lista.")
        del self.marks[i]
        self._mark_changed()

    def goto_mark(self) -> None:
        i = self._sel_mark()
        if i is None or i >= len(self.marks):
            return
        self.playing = False
        self.play_btn.configure(text="▶")
        self._goto(_num(self.marks[i].get("start")))

    def _verdict(self) -> str:
        if self.marks:
            return "hurto"
        v = (self.clip or {}).get("verdict") or "sin_marcar"
        return "normal" if v == "normal" else "sin_marcar"

    def save_marks(self, verdict: str | None = None) -> None:
        clip = self.clip
        if clip is None:
            return
        body = {"verdict": verdict or self._verdict(), "marks": copy.deepcopy(self.marks)}
        cid = clip.get("id")
        self.save_lbl.configure(text="guardando…", style="Suave.TLabel")

        def done(res) -> None:
            row = res if isinstance(res, dict) and res.get("id") == cid else None
            for c in {id(x): x for x in self.clips + ([clip] if clip is not None else [])}.values():
                if c.get("id") == cid:
                    c.update(body)
                    if row:
                        c.update({k: v for k, v in row.items() if k != "windows"})
                    self._row_of(c)
            if self.clip is not None and self.clip.get("id") == cid:
                self.dirty = False
                self.save_lbl.configure(text="✔ guardado", style="Suave.TLabel")
            self._show_ia_counts()
            empty = [i for i in (row or {}).get("marks_without_windows") or [] if isinstance(i, int)
                     and 0 <= i < len(body["marks"])]
            if empty and body["verdict"] == "hurto":
                lines = [f"• {mark_text(body['marks'][i])}" for i in empty]
                self._say("warning", f"{'Esta marca no enseña' if len(empty) == 1 else 'Estas marcas no enseñan'} nada "
                                     "a la IA de movimientos:\n" + "\n".join(lines) + "\n\nEn ese rato la IA no vio "
                                     "la postura de esa persona (tapada, de espaldas, muy chica o fuera de la imagen). "
                                     "Marcá el rato en que se la ve bien, o elegí a la persona correcta.")

        def fail(exc: Exception) -> None:
            if self.clip is not None and self.clip.get("id") == cid:
                self.save_lbl.configure(text="sin guardar", style="Aviso.TLabel")
            self._fail(exc)
        self.app.bg.run(lambda: self.app.api.put(f"/api/training/clips/{cid}", body), done, fail)

    def set_normal(self) -> None:
        if self.clip is None:
            return self._say("info", "Elegí un clip de la lista.")
        if self.marks and not self.app.mb.askyesno(NOMBRE, f"El clip tiene {plural(len(self.marks), 'marca')} de hurto: "
                                                           "se borran. ¿Seguro que es un clip normal?"):
            return
        self.marks = []
        self._fill_marks()
        self._redraw()
        self.save_marks("normal")

    # ---------- más ----------
    def try_rules(self) -> None:
        """«Probar con las reglas y mandar al filtro…»: la ventana de la cámara en «🎞 Clips» con este video (marcado
        como hurto en el primer momento marcado, o normal), para ver si las reglas y gestos dan alerta y mandar esas
        alertas al filtro."""
        clip = self.clip
        if clip is None:
            return self._say("info", "Elegí un clip de la lista.")
        if not clip.get("camera_id"):
            return self._say("info", "Este clip no tiene cámara: las reglas son de cada cámara (sus zonas y tiempos). "
                                     "Subilo de nuevo eligiendo la cámara.")
        if not self.video:
            who = uploader_of(clip)
            return self._say("info", "El video no está en esta PC" + (f" (está en la PC de {who})" if who else "")
                             + ": probalo desde ahí, o «Más» → «Ubicar el video en esta PC…» si lo tenés acá.")
        cam = self._cam(clip["camera_id"])
        if cam is None:
            return self._say("info", "No encuentro la cámara de este clip en la lista (¿la borraron?).")
        if self.dirty and self.app.mb.askyesno(NOMBRE, "Las marcas no se guardaron. ¿Guardarlas primero?"):
            self.save_marks()
        lib = self.app.library()
        lib.add(cam["id"], [self.video])
        import os
        key = os.path.normcase(os.path.abspath(self.video))
        video = next((v for v in lib.videos(cam["id"]) if os.path.normcase(os.path.abspath(v.path)) == key), None)
        if video is None:
            return
        verdict = self._verdict()
        video.kind = {"hurto": "hurto", "normal": "normal"}.get(verdict, "")
        video.marks = [round(min(_num(m.get("start")) for m in self.marks), 1)] if verdict == "hurto" else []
        lib.update(cam["id"], video)
        self.app.tune_dialog(cam, clips=[video], mode="clips", pick=False)

    def locate_video(self) -> None:
        """El clip se subió desde otra PC (o se movió el archivo): elegir el video acá para verlo."""
        clip = self.clip
        if clip is None:
            return self._say("info", "Elegí un clip de la lista.")
        from tkinter import filedialog

        from .biblioteca import VIDEO_TYPES, default_folder
        path = filedialog.askopenfilename(parent=self.root, title=f"¿Dónde está «{clip.get('name')}» en esta PC?",
                                          initialdir=str(default_folder()),
                                          filetypes=[VIDEO_TYPES, ("Todos los archivos", "*.*")])
        if path:
            self.link_video(path)

    def link_video(self, path: str) -> None:
        clip = self.clip
        if clip is None:
            return

        def done(uuid: str) -> None:
            if uuid != clip.get("uuid"):
                return self._say("warning", "Ese video no es el de este clip (es otro archivo, o está recortado o "
                                            "convertido).")
            self.local.link(uuid, path)
            self._fill_clips()
            if self.clip is clip:
                self._open(clip)
        self.app.bg.run(lambda: clip_uuid(path), done, self._fail)

    def open_folder(self) -> None:
        if not self.video:
            return self._say("info", "El video de este clip no está en esta PC.")
        from .oficina import open_file
        open_file(str(Path(self.video).parent))

    def delete_clip(self) -> None:
        clip = self.clip
        if clip is None:
            return self._say("info", "Elegí un clip de la lista.")
        if not self.app.mb.askyesno(NOMBRE, f"¿Borrar «{clip.get('name')}» del entrenamiento? Se borran sus marcas y "
                                            "lo que vio la IA. El video de esta PC no se toca."):
            return

        def done(_r) -> None:
            if clip.get("uuid"):
                self.local.forget(clip["uuid"])
            self.clips = [c for c in self.clips if c.get("id") != clip.get("id")]
            if self.clip is clip:
                self._close_video()
                self.clip, self.frames, self.times, self.marks, self.dirty = None, [], [], [], False
                self.title.configure(text="Elegí un clip de la lista")
                self._fill_marks()
                self._draw_empty()
            self._fill_clips()
            self._show_ia_counts()
        self.app.bg.run(lambda: self.app.api.delete(f"/api/training/clips/{clip['id']}"), done, self._fail)
