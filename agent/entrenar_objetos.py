"""Pestaña «Entrenar objetos» de la app (18 nov; el dueño: «hacé otra pestaña… primero añadir el nuevo objeto y ponerle
para qué plan de métricas sería… hacelo completo con una guía de qué hacer y para qué en cada paso… ¿podemos subir los
videos y fotos desde la app y que los analicen las máquinas virtuales?»).

Le enseña a la IA un objeto propio (pallets, cajas, carros…) en 5 pasos, cada uno con qué hacer y para qué, y una
frase arriba que dice qué sigue (`guide_text`):

1. **El objeto y su plan**: nombre, las clases (los estados que tiene que distinguir: «Pallet vacío», «Terminado con
   film»), con una palabra en inglés para que la IA proponga recuadros antes de tener un modelo propio (YOLOE), el plan
   de métricas donde se va a medir (el nuevo «Conteo de objetos» u otro) y de qué cliente es.
2. **Juntar fotos**: subir fotos, sacar fotos de un video (en esta PC, que sube solo las fotos, o en una máquina
   virtual, que sube el video entero) o una foto de una cámara ahora. Las fotos se achican a 1280 px.
3. **Marcar**: un recuadro alrededor de cada objeto con su clase (teclas 1 a 8), «No hay ninguno» (también enseña) o
   descartar; lo que propone la IA se acepta con A. Las dudosas para la IA van primero.
4. **Entrenar**: lo hace una máquina con Ultralytics (mejor con placa de video); se ve cómo va y se puede cancelar.
5. **Probar y activar**: el resultado en palabras, probar una versión con una foto o una cámara en esta PC, activarla
   (pre-marca las fotos nuevas y es la que van a usar las cámaras).

Todo lo de red va en hilos (`app.bg.run` y un hilo para las subidas); la ventana nunca se traba. Entra en 1366x768.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from core import objetos as ob

from .marca import NOMBRE

log = logging.getLogger("vigia.entrenar_objetos")

ALL_CLIENTS = "Todos los clientes"
COLORS = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#e377c2", "#8c564b")
PROPOSED = "#ff9800"
FILTERS = (("dudosas", "Sin marcar (las dudosas primero)", "sin_marcar"), ("nuevas", "Sin marcar (las nuevas primero)",
           "sin_marcar"), ("marcadas", "Marcadas", "marcada"), ("descartadas", "Descartadas", "descartada"),
           ("todas", "Todas", None))
PAGE = 300
CACHE = 40
POLL_MS = 60
REFRESH_BUSY_MS = 5000
HANDLE = 6  # píxeles de las esquinas para agrandar un recuadro
PHOTO_TYPES = ("Fotos", "*.jpg *.jpeg *.png *.bmp *.webp *.JPG *.JPEG *.PNG")
VIDEO_TYPES = ("Videos", "*.mp4 *.avi *.mkv *.mov *.m4v *.ts *.webm *.MP4 *.AVI *.MKV *.MOV")
VM_VIDEO_MB = 1000  # más que esto, a la máquina no: tarda mucho en subir (las fotos se sacan acá)
STEP_TITLES = ("1 · El objeto", "2 · Juntar fotos", "3 · Marcar", "4 · Entrenar", "5 · Probar y activar")

# lo que se ve en cada paso: qué hacer y para qué (pedido del dueño: «una guía de qué hacer y para qué en cada paso»)
STEP_GUIDE = (
    ("Qué hacer: poné el nombre del objeto y sus clases, es decir los estados que la IA tiene que distinguir (por "
     "ejemplo «Pallet vacío», «En armado» y «Terminado con film»). Si te sirve contarlos a todos juntos, una sola "
     "clase alcanza. Elegí el plan de métricas donde se va a medir y, si es de un solo cliente, cuál.",
     "Para qué: la IA aprende exactamente estas clases, ni más ni menos. Con estados distintos se puede contar cuántos "
     "hay de cada uno y cuántos se terminan por hora. La palabra en inglés (opcional) deja que la IA proponga "
     "recuadros desde la primera foto, antes de entrenar: después solo se corrigen."),
    ("Qué hacer: subí fotos, sacá fotos de un video (una cada 2 segundos, salteando las repetidas) o una foto de una "
     "cámara ahora. Que sean de las cámaras donde se va a usar y de momentos distintos: día y noche, lleno y vacío, "
     "con gente y autoelevadores adelante.",
     "Para qué: la IA aprende de ejemplos. Con fotos parecidas entre sí aprende poco; con fotos variadas aprende a "
     "reconocer el objeto en cualquier situación. Para empezar alcanzan 100 a 200; para que ande bien, unas 300 a 1000."),
    ("Qué hacer: en cada foto, arrastrá un recuadro alrededor de CADA objeto y elegí su clase (teclas 1 a 8). Si no "
     "hay ninguno, «No hay ninguno». Si la foto está mal (movida, negra, repetida), descartala. Lo que propone la IA "
     "(en naranja) se acepta con A o haciendo clic en cada recuadro.",
     "Para qué: la IA copia lo que marcás. Un objeto sin marcar le enseña que eso NO es el objeto: marcá todos, con "
     "recuadros justos. Las fotos sin ninguno también enseñan (a no ver objetos donde no hay). Las que la IA duda van "
     "primero: son las que más le enseñan."),
    ("Qué hacer: cuando haya al menos 30 fotos marcadas y 20 recuadros de una clase, «Entrenar en la nube». Lo hace "
     "una máquina virtual (con placa de video tarda minutos; sin placa, horas). Podés seguir marcando mientras tanto.",
     "Para qué: con las fotos marcadas se arma un modelo propio (YOLO afinado con tus fotos). Una de cada cinco fotos "
     "queda aparte y la IA no la ve al entrenar: con esas se mide qué tan bien anda de verdad."),
    ("Qué hacer: mirá el resultado de la versión, probala con una foto o una cámara (corre en esta PC) y activala. Si "
     "falla seguido, marcá más fotos (sobre todo de lo que confunde) y volvé a entrenar: cada vuelta es una versión "
     "nueva y se puede volver a cualquiera.",
     "Para qué: la versión activa pre-marca las fotos nuevas (marcar va cada vez más rápido) y es la que van a usar las "
     "cámaras del plan para contar el objeto en cada sector. Probarla antes evita usar un modelo que se equivoca."),
)
# para qué sirve cada plan, en el formulario del objeto
PLAN_HELP = {
    "objetos": "Plan nuevo para contar cosas que no son personas ni vehículos (pallets, cajas, carros, bultos). El "
               "cliente lo contrata como los demás planes y cada cámara elige si lo mide. Recomendado.",
    "comercio": "Si el objeto completa lo que ya se mide de la gente en un comercio (por ejemplo carros o canastos).",
    "estacion": "Para algo de la playa de una estación de servicio (bidones, conos, mangueras).",
    "empleados": "Para algo del personal (chaleco, casco, uniforme).",
    "gastronomia": "Para algo de las mesas (platos, vasos, bandejas).",
    "hurto": "Para un objeto que ayude a revisar posibles hurtos (por ejemplo un producto caro o una caja abierta).",
}


# ------------------------------------------------------------------ textos (sin ventana: se prueban sueltos)
def plural(n: int, one: str, many: str | None = None) -> str:
    return f"{n} {one if n == 1 else (many or one + 's')}"


def active_job(model: dict | None, kind: str) -> dict | None:
    for j in (model or {}).get("jobs") or []:
        if j["kind"] == kind and j["status"] in ("subiendo", "en_cola", "procesando"):
            return j
    return None


def guide_text(model: dict | None) -> tuple[str, int]:
    """La frase de arriba: qué sigue, y en qué paso (0 a 4)."""
    if not model:
        return "Paso 1: creá el objeto que querés que la IA aprenda («＋ Nuevo objeto»): por ejemplo «Pallets».", 0
    c = model.get("counts") or {}
    total, marked, todo = c.get("total", 0), c.get("marcada", 0), c.get("sin_marcar", 0)
    check = model.get("check") or {}
    versions = model.get("version_list") or []
    train = active_job(model, "entrenar")
    if train:
        if train["status"] == "en_cola":
            return ("Paso 4: el entrenamiento está en cola, esperando una máquina que entrene. Mientras tanto podés "
                    "seguir marcando fotos."), 3
        return f"Paso 4: entrenando ({train.get('message') or 'empezando'}). Podés seguir marcando mientras tanto.", 3
    if total < ob.MIN_PHOTOS:
        return (f"Paso 2: juntá fotos del objeto: hay {total} y para entrenar hacen falta al menos {ob.MIN_PHOTOS} "
                f"marcadas (para que ande bien, unas {ob.GOOD_PHOTOS}).", 1)
    if not check.get("ok"):
        if todo:
            return f"Paso 3: marcá las fotos ({plural(todo, 'foto')} sin marcar). " + " ".join(
                check.get("problems") or []), 2
        return "Paso 2: hacen falta más fotos. " + " ".join(check.get("problems") or []), 1
    if not versions:
        return f"Paso 4: ya se puede entrenar con {plural(marked, 'foto marcada', 'fotos marcadas')}.", 3
    last = versions[0]
    if not any(v.get("active") for v in versions):
        return f"Paso 5: probá la versión {last['number']} ({last.get('grade')}) y activala si anda bien.", 4
    act = next(v for v in versions if v.get("active"))
    if last["number"] > act["number"]:
        return (f"Paso 5: hay una versión nueva ({last['number']}, {last.get('grade')}): probala y, si anda mejor que "
                f"la {act['number']}, activala.", 4)
    if todo:
        return (f"La versión {act['number']} está activa ({act.get('grade')}). Para mejorarla: marcá las "
                f"{plural(todo, 'foto')} sin marcar (las dudosas primero) y volvé a entrenar.", 2)
    return (f"La versión {act['number']} está activa ({act.get('grade')}). Para mejorarla: más fotos variadas, "
            f"marcarlas y volver a entrenar.", 1)


def counts_text(model: dict) -> str:
    c = model.get("counts") or {}
    return (f"{plural(c.get('total', 0), 'foto')} · {c.get('sin_marcar', 0)} sin marcar · {c.get('marcada', 0)} "
            f"marcadas · {c.get('descartada', 0)} descartadas")


def check_lines(model: dict) -> list[tuple[str, bool]]:
    """Lo que hace falta para entrenar, renglón por renglón (texto, ¿está bien?)."""
    c = model.get("counts") or {}
    boxes = model.get("boxes") or {}
    marked = c.get("marcada", 0)
    out = [(f"{plural(marked, 'foto marcada', 'fotos marcadas')} (mínimo {ob.MIN_PHOTOS}, para que ande bien "
            f"{ob.GOOD_PHOTOS})", marked >= ob.MIN_PHOTOS)]
    for cl in model.get("classes") or []:
        n = boxes.get(cl["key"], 0)
        out.append((f"«{cl['name']}»: {plural(n, 'recuadro')} (mínimo {ob.MIN_BOXES}, para que ande bien "
                    f"{ob.GOOD_BOXES})", n >= ob.MIN_BOXES))
    out.append((f"{c.get('test_marked', 0)} de las marcadas quedan para probar (1 de cada {ob.TEST_SHARE})",
                c.get("test_marked", 0) >= ob.MIN_TEST_PHOTOS))
    return out


def job_text(j: dict) -> str:
    st = j.get("status_name") or j.get("status")
    if j["status"] == "procesando":
        st = f"{j.get('message') or 'Trabajando'} ({int(round((j.get('progress') or 0) * 100))} %)"
    elif j["status"] == "error":
        st = f"Error: {j.get('error') or '—'}"
    elif j["status"] == "listo":
        r = j.get("result") or {}
        if j["kind"] == "video":
            st = f"Listo: {plural(r.get('images') or 0, 'foto')}" + (
                f" ({r['proposed']} pre-marcadas)" if r.get("proposed") else "")
        elif j["kind"] == "premarcar":
            st = f"Listo: {plural(r.get('proposed') or 0, 'foto pre-marcada', 'fotos pre-marcadas')}"
        elif j["kind"] == "entrenar":
            st = f"Listo: versión {r.get('number')}" + (f" en {r['minutes']} min" if r.get("minutes") else "")
    return st


def fit(w: int, h: int, box_w: int, box_h: int) -> tuple[float, int, int]:
    """Escala y corrimiento para que una foto de w x h entre centrada en box_w x box_h."""
    k = min(box_w / max(1, w), box_h / max(1, h))
    return k, int((box_w - w * k) / 2), int((box_h - h * k) / 2)


def box_at(boxes: list[dict], x: float, y: float) -> int | None:
    """El recuadro (el más chico) que contiene el punto (en fracciones de la foto)."""
    best, area = None, 2.0
    for i, b in enumerate(boxes):
        if b["x1"] <= x <= b["x2"] and b["y1"] <= y <= b["y2"]:
            a = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
            if a < area:
                best, area = i, a
    return best


def draw_detections(img: np.ndarray, boxes: list[dict], classes: list[dict]) -> np.ndarray:
    """La foto con lo que encontró el modelo (para «Probar»)."""
    out = img.copy()
    h, w = out.shape[:2]
    idx = {c["key"]: i for i, c in enumerate(classes)}
    for b in boxes:
        i = idx.get(b.get("k"), 0)
        hexc = COLORS[i % len(COLORS)].lstrip("#")
        bgr = (int(hexc[4:6], 16), int(hexc[2:4], 16), int(hexc[0:2], 16))
        p1 = (int(b["x1"] * w), int(b["y1"] * h))
        p2 = (int(b["x2"] * w), int(b["y2"] * h))
        cv2.rectangle(out, p1, p2, bgr, max(2, w // 400))
        cv2.putText(out, f"{i + 1} {int(round(b.get('p', 0) * 100))}%", (p1[0] + 3, max(14, p1[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.45, w / 1600), bgr, max(1, w // 800), cv2.LINE_AA)
    return out


def detections_summary(boxes: list[dict], classes: list[dict]) -> str:
    if not boxes:
        return "No encontró ninguno."
    parts = []
    for i, c in enumerate(classes):
        n = sum(1 for b in boxes if b.get("k") == c["key"])
        if n:
            parts.append(f"{n} «{c['name']}» (color {i + 1})")
    return "Encontró " + ", ".join(parts) + "."


# ------------------------------------------------------------------ la pestaña
class EntrenarObjetos:
    """La pestaña. `app`: la Oficina (api, bg, mb, clientes y cámaras, `_dialog`, `agent_of`)."""

    def __init__(self, app, parent):
        self.app = app
        self.tk, self.ttk = app.tk, app.ttk
        self.root = app.v.root
        self.frame = parent
        self.models: list[dict] = []
        self.model: dict | None = None
        self.data_dir: Path | None = None  # (las pruebas lo cambian)
        self._q: queue.Queue = queue.Queue()
        self._poll_job = None
        self._refresh_job = None
        self.busy = False  # subiendo fotos o sacándolas de un video en esta PC
        self._stop = threading.Event()
        # paso 3: marcar
        self.images: list[dict] = []
        self.cur: int | None = None
        self.boxes: list[dict] = []
        self.proposed: list[dict] = []
        self.sel: int | None = None
        self.dirty = False
        self.cls: str | None = None
        self._pix: OrderedDict[int, np.ndarray] = OrderedDict()
        self._fetching: set[int] = set()
        self._photo = None
        self._geo = (1.0, 0, 0, 1, 1)  # k, ox, oy, ancho y alto de la foto en pantalla
        self._drag: dict | None = None
        self._img_req = 0
        self._build()

    # ---------- utilidades ----------
    def _say(self, kind: str, text: str) -> None:
        getattr(self.app.mb, f"show{kind}")(NOMBRE, text)

    def _fail(self, exc: Exception) -> None:
        net = getattr(self.app, "_net_fail", None)
        if net is not None and getattr(exc, "status", 0) == 401:
            return net(exc)
        self._say("error", str(exc))

    def _alive(self) -> bool:
        try:
            return bool(self.frame.winfo_exists())
        except Exception:  # noqa: BLE001
            return False

    def visible(self) -> bool:
        try:
            nb = self.app.notebook
            return nb.select() == str(self.frame)
        except Exception:  # noqa: BLE001
            return True

    def step(self) -> int:
        try:
            return self.steps.index(self.steps.select())
        except Exception:  # noqa: BLE001
            return 0

    def go(self, i: int) -> None:
        self.steps.select(i)

    @property
    def classes(self) -> list[dict]:
        return (self.model or {}).get("classes") or []

    def color_of(self, key: str | None) -> str:
        keys = [c["key"] for c in self.classes]
        return COLORS[keys.index(key) % len(COLORS)] if key in keys else PROPOSED

    def _local_dir(self) -> Path:
        base = self.data_dir
        if base is None:
            agent = self.app.agent_of() if getattr(self.app, "agent_of", None) else None
            base = Path(agent.data_dir) if agent is not None else Path.home() / "AnaliticaCritica"
        d = Path(base) / "objetos"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _ensure_poll(self) -> None:
        if self._poll_job is None:
            try:
                self._poll_job = self.root.after(POLL_MS, self._poll)
            except Exception:  # noqa: BLE001
                self._poll_job = None

    def _poll(self) -> None:
        self._poll_job = None
        while True:
            try:
                fn, arg = self._q.get_nowait()
            except queue.Empty:
                break
            try:
                fn(arg)
            except Exception:  # noqa: BLE001
                log.exception("Entrenar objetos: error en la ventana")
        if self.busy:
            self._ensure_poll()

    def _later(self, fn, arg=None) -> None:
        """Desde otro hilo: que `fn(arg)` corra en el de la ventana."""
        self._q.put((fn, arg))

    # ---------- armado ----------
    def _build(self) -> None:
        tk, ttk, f = self.tk, self.ttk, self.frame
        top = ttk.Frame(f)
        top.pack(fill="x", pady=(0, 4))
        ttk.Label(top, text="Objeto:", style="Subtitulo.TLabel").pack(side="left")
        self.model_var = tk.StringVar()
        self.model_cb = ttk.Combobox(top, textvariable=self.model_var, state="readonly", width=26)
        self.model_cb.pack(side="left", padx=6)
        self.model_cb.bind("<<ComboboxSelected>>", lambda _e: self.select_model(self.model_var.get()))
        ttk.Button(top, text="＋ Nuevo objeto", style="Primario.TButton", command=self.new_object).pack(side="left")
        ttk.Button(top, text="✎ Editar objeto", command=self.edit_object).pack(side="left", padx=6)
        self.guide = ttk.Label(f, text="", style="Aviso.TLabel", wraplength=1300, justify="left")
        self.guide.pack(fill="x", pady=(0, 4))
        self.steps = ttk.Notebook(f)
        self.steps.pack(fill="both", expand=True)
        self.panels = []
        for i, title in enumerate(STEP_TITLES):
            p = ttk.Frame(self.steps, padding=8)
            self.steps.add(p, text=f"  {title}  ")
            if i != 2:  # (en «Marcar» la guía va en una columna a la derecha: la foto necesita el alto)
                what, why = STEP_GUIDE[i]
                box = ttk.Frame(p)
                box.pack(fill="x", pady=(0, 6))
                ttk.Label(box, text=what, wraplength=1260, justify="left").pack(anchor="w")
                ttk.Label(box, text=why, style="Suave.TLabel", wraplength=1260, justify="left").pack(anchor="w",
                                                                                                    pady=(2, 0))
            self.panels.append(p)
        self.steps.bind("<<NotebookTabChanged>>", lambda _e: self._step_changed())
        self._build_object(self.panels[0])
        self._build_photos(self.panels[1])
        self._build_label(self.panels[2])
        self._build_train(self.panels[3])
        self._build_versions(self.panels[4])
        self.root.bind_all("<Key>", self._key, add="+")
        self._show_model()

    # --- paso 1
    def _build_object(self, p) -> None:
        ttk = self.ttk
        self.obj_info = self.tk.Text(p, height=14, wrap="word", relief="flat", borderwidth=0, padx=4, pady=4,
                                     font=("Segoe UI", 10), background="#ffffff")
        self.obj_info.pack(fill="both", expand=True)
        self.obj_info.tag_configure("t", font=("Segoe UI", 12, "bold"))
        self.obj_info.tag_configure("s", foreground="#555555")
        bb = ttk.Frame(p)
        bb.pack(fill="x", pady=(6, 0))
        ttk.Button(bb, text="✎ Editar objeto", command=self.edit_object).pack(side="left")
        if getattr(self.app, "is_admin", False):
            ttk.Button(bb, text="Borrar objeto…", command=self.delete_object).pack(side="left", padx=6)
        ttk.Button(bb, text="Siguiente: juntar fotos →", style="Primario.TButton",
                   command=lambda: self.go(1)).pack(side="right")

    # --- paso 2
    def _build_photos(self, p) -> None:
        tk, ttk = self.tk, self.ttk
        bb = ttk.Frame(p)
        bb.pack(fill="x")
        ttk.Button(bb, text="⬆ Subir fotos…", style="Primario.TButton", command=self.upload_photos).pack(side="left")
        ttk.Button(bb, text="🎞 Sacar fotos de un video…", command=self.video_dialog).pack(side="left", padx=6)
        ttk.Button(bb, text="📷 Foto de una cámara…", command=self.camera_dialog).pack(side="left")
        self.stop_btn = ttk.Button(bb, text="■ Frenar", command=self._stop.set)
        self.photo_counts = ttk.Label(p, text="", style="Subtitulo.TLabel")
        self.photo_counts.pack(anchor="w", pady=(8, 2))
        self.work_line = ttk.Label(p, text="", style="Suave.TLabel", wraplength=1260, justify="left")
        self.work_line.pack(anchor="w")
        jb = ttk.Frame(p)  # (abajo y antes que la lista: la lista toma lo que sobra y no lo tapa)
        jb.pack(side="bottom", fill="x", pady=(4, 0))
        ttk.Button(jb, text="Cancelar el trabajo elegido", command=self.cancel_job).pack(side="left")
        self.next2_btn = ttk.Button(jb, text="Siguiente: marcar →", style="Primario.TButton",
                                    command=lambda: self.go(2))
        self.next2_btn.pack(side="right")
        ttk.Label(p, text="Trabajos en las máquinas virtuales (videos, pre-marcar y entrenar):").pack(anchor="w",
                                                                                                  pady=(8, 2))
        tf = ttk.Frame(p)
        tf.pack(fill="both", expand=True)
        self.jobs = ttk.Treeview(tf, columns=("que", "archivo", "estado", "maquina", "cuando"), show="headings",
                                 height=7, selectmode="browse")
        for col, title, w in (("que", "Qué", 170), ("archivo", "Archivo", 200), ("estado", "Cómo va", 520),
                              ("maquina", "Máquina", 130), ("cuando", "Pedido", 120)):
            self.jobs.heading(col, text=title, anchor="w")
            self.jobs.column(col, width=w, stretch=col == "estado")
        sb = ttk.Scrollbar(tf, orient="vertical", command=self.jobs.yview)
        self.jobs.configure(yscrollcommand=sb.set)
        self.jobs.pack(side="left", fill="both", expand=True)
        sb.pack(side="left", fill="y")

    # --- paso 3
    def _build_label(self, p) -> None:
        tk, ttk = self.tk, self.ttk
        body = ttk.Frame(p)
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body, width=270)
        left.pack(side="left", fill="y", padx=(0, 8))
        side = ttk.Frame(body, width=290)
        side.pack(side="right", fill="y", padx=(10, 0))
        what, why = STEP_GUIDE[2]
        ttk.Label(side, text=what, wraplength=285, justify="left").pack(anchor="w")
        ttk.Label(side, text=why, style="Suave.TLabel", wraplength=285, justify="left").pack(anchor="w", pady=(6, 0))
        ttk.Label(side, text="Mouse: arrastrar = recuadro nuevo · clic = elegirlo (se mueve, o se agranda desde las "
                             "esquinas) · clic derecho = borrarlo.\nTeclas: 1 a 8 = clase · Enter = guardar y seguir · "
                             "N = no hay ninguno · D = descartar · A = aceptar lo de la IA · ← → = moverse.",
                  wraplength=285, justify="left").pack(anchor="w", pady=(8, 0))
        self.filter_var = tk.StringVar(value=FILTERS[0][1])
        fcb = ttk.Combobox(left, textvariable=self.filter_var, values=[x[1] for x in FILTERS], state="readonly",
                           width=30)
        fcb.pack(fill="x")
        fcb.bind("<<ComboboxSelected>>", lambda _e: self.load_images())
        self.list_title = ttk.Label(left, text="", style="Suave.TLabel")
        self.list_title.pack(anchor="w", pady=(4, 2))
        lf = ttk.Frame(left)
        lf.pack(fill="both", expand=True)
        self.img_list = ttk.Treeview(lf, columns=("n", "estado", "rec"), show="headings", selectmode="browse",
                                     height=14)
        for col, title, w in (("n", "Foto", 118), ("estado", "Estado", 92), ("rec", "Rec", 42)):
            self.img_list.heading(col, text=title, anchor="w")
            self.img_list.column(col, width=w, stretch=col == "n")
        sb = ttk.Scrollbar(lf, orient="vertical", command=self.img_list.yview)
        self.img_list.configure(yscrollcommand=sb.set)
        self.img_list.pack(side="left", fill="both", expand=True)
        sb.pack(side="left", fill="y")
        self.img_list.bind("<<TreeviewSelect>>", lambda _e: self._list_pick())
        ttk.Button(left, text="🤖 Pre-marcar con la IA", command=self.prelabel).pack(fill="x", pady=(6, 0))
        center = ttk.Frame(body)
        center.pack(side="left", fill="both", expand=True)
        # (los controles van abajo y se arman primero: la foto toma lo que sobra y nunca los tapa)
        ab2 = ttk.Frame(center)
        ab2.pack(side="bottom", fill="x", pady=(4, 0))
        ab = ttk.Frame(center)
        ab.pack(side="bottom", fill="x", pady=(4, 0))
        self.label_line = ttk.Label(center, text="", style="Suave.TLabel", wraplength=760, justify="left")
        self.label_line.pack(side="bottom", anchor="w", pady=(2, 0))
        self.cls_bar = ttk.Frame(center)
        self.cls_bar.pack(side="bottom", fill="x", pady=(4, 0))
        self.cls_var = tk.StringVar()
        self.canvas = tk.Canvas(center, width=700, height=330, background="#202020", highlightthickness=0,
                                cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._motion)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.canvas.bind("<Button-3>", self._right_click)
        self.canvas.bind("<Configure>", lambda _e: self._redraw())
        self.btn_ok = ttk.Button(ab, text="✔ Listo, siguiente (Enter)", style="Primario.TButton",
                                 command=self.save_and_next, takefocus=False)
        self.btn_ok.pack(side="left")
        for text, cmd in (("No hay ninguno (N)", self.none_here), ("Aceptar lo de la IA (A)", self.accept_proposed)):
            ttk.Button(ab, text=text, command=cmd, takefocus=False).pack(side="left", padx=(6, 0))
        for text, cmd in (("Borrar recuadro (Supr)", self.delete_box), ("Descartar foto (D)", self.discard)):
            ttk.Button(ab2, text=text, command=cmd, takefocus=False).pack(side="left", padx=(0, 6))
        ttk.Button(ab2, text="Siguiente ▶", command=lambda: self.move(1), takefocus=False).pack(side="right")
        ttk.Button(ab2, text="◀ Anterior", command=lambda: self.move(-1), takefocus=False).pack(side="right", padx=6)

    # --- paso 4
    def _build_train(self, p) -> None:
        tk, ttk = self.tk, self.ttk
        self.check_box = tk.Text(p, height=9, wrap="word", relief="flat", borderwidth=0, padx=4, pady=4,
                                 font=("Segoe UI", 10), background="#ffffff")
        self.check_box.pack(fill="x")
        self.check_box.tag_configure("ok", foreground="#1a7f43")
        self.check_box.tag_configure("no", foreground="#c25e00")
        self.check_box.tag_configure("t", font=("Segoe UI", 10, "bold"))
        self.machines_line = ttk.Label(p, text="", wraplength=1260, justify="left")
        self.machines_line.pack(anchor="w", pady=(6, 0))
        bb = ttk.Frame(p)
        bb.pack(fill="x", pady=(8, 0))
        self.train_btn = ttk.Button(bb, text="▶ Entrenar en la nube", style="Primario.TButton", command=self.train)
        self.train_btn.pack(side="left")
        self.fresh_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(bb, text="Empezar de cero (no seguir desde la versión anterior)",
                        variable=self.fresh_var).pack(side="left", padx=10)
        self.cancel_train_btn = ttk.Button(bb, text="Cancelar el entrenamiento", command=self.cancel_training)
        self.train_state = ttk.Label(p, text="", style="Subtitulo.TLabel", wraplength=1260, justify="left")
        self.train_state.pack(anchor="w", pady=(10, 0))
        self.train_bar = ttk.Progressbar(p, length=600, maximum=100)
        self.train_bar.pack(anchor="w", pady=(4, 0))

    # --- paso 5
    def _build_versions(self, p) -> None:
        tk, ttk = self.tk, self.ttk
        vf = ttk.Frame(p)
        vf.pack(fill="x")
        self.ver_list = ttk.Treeview(vf, columns=("v", "fecha", "fotos", "nota", "activa", "donde"), show="headings",
                                     height=4, selectmode="browse")
        for col, title, w in (("v", "Versión", 70), ("fecha", "Fecha", 150), ("fotos", "Fotos", 70),
                              ("nota", "Resultado", 160), ("activa", "Estado", 90), ("donde", "Dónde se entrenó", 260)):
            self.ver_list.heading(col, text=title, anchor="w")
            self.ver_list.column(col, width=w, stretch=col == "donde")
        self.ver_list.pack(fill="x")
        self.ver_list.bind("<<TreeviewSelect>>", lambda _e: self._show_version())
        ttk.Label(p, text="Medir en las cámaras: cuando una versión anda bien, las cámaras del plan del objeto la van a "
                          "usar para contarlo en cada sector (lo terminamos con el primer modelo entrenado con video "
                          "real del cliente).", style="Suave.TLabel", wraplength=1260, justify="left").pack(
            side="bottom", anchor="w", pady=(6, 0))
        bb = ttk.Frame(p)
        bb.pack(fill="x", pady=(6, 0))
        ttk.Button(bb, text="✔ Activar esta versión", style="Primario.TButton",
                   command=lambda: self.set_version(True)).pack(side="left")
        ttk.Button(bb, text="Apagar", command=lambda: self.set_version(False)).pack(side="left", padx=6)
        ttk.Button(bb, text="🔍 Probar con una foto…", command=self.test_photo).pack(side="left", padx=(12, 0))
        ttk.Button(bb, text="🔍 Probar con una cámara…", command=self.test_camera).pack(side="left", padx=6)
        self.ver_text = tk.Text(p, height=6, wrap="word", relief="flat", borderwidth=0, padx=4, pady=4,
                                font=("Segoe UI", 10), background="#ffffff")
        self.ver_text.pack(fill="both", expand=True, pady=(6, 0))

    # ---------- carga ----------
    def on_show(self) -> None:
        self.load_models()

    def clients_changed(self) -> None:
        if self.model:
            self._show_model()

    def load_models(self, select_id: int | None = None) -> None:
        def done(res) -> None:
            self.models = res or []
            names = [m["name"] for m in self.models]
            self.model_cb.configure(values=names)
            want = select_id if select_id is not None else (self.model or {}).get("id")
            m = next((x for x in self.models if x["id"] == want), None) or (self.models[0] if self.models else None)
            if m is None:
                self.model = None
                self.model_var.set("")
                self._show_model()
                return
            self.model_var.set(m["name"])
            self.load_model(m["id"])
        self.app.bg.run(lambda: self.app.api.get("/api/objects"), done, self._fail)

    def select_model(self, name: str) -> None:
        m = next((x for x in self.models if x["name"] == name), None)
        if m is not None:
            self._flush_label()
            self.images, self.cur = [], None
            self.load_model(m["id"])

    def load_model(self, mid: int, images: bool = True) -> None:
        def done(res) -> None:
            first = (self.model or {}).get("id") != res["id"]
            self.model = res
            self._show_model()
            if images and (first or not self.images):
                self.load_images()
            self._schedule_refresh()
        self.app.bg.run(lambda: self.app.api.get(f"/api/objects/{mid}"), done, self._fail)

    def _schedule_refresh(self) -> None:
        """Mientras una máquina trabaja (o falta subir), se mira cada 5 s cómo va."""
        if self._refresh_job is not None:
            return
        m = self.model or {}
        if not any(j["status"] in ("subiendo", "en_cola", "procesando") for j in m.get("jobs") or []):
            return

        def tick() -> None:
            self._refresh_job = None
            if self._alive() and self.model and self.visible():
                self.load_model(self.model["id"], images=False)  # (al llegar, vuelve a programar si sigue trabajando)
            elif self._alive() and self.model:
                self._schedule_refresh()
        self._refresh_job = self.root.after(REFRESH_BUSY_MS, tick)

    # ---------- mostrar ----------
    def _show_model(self) -> None:
        m = self.model
        text, _ = guide_text(m)
        self.guide.configure(text="➜ " + text)
        for i, title in enumerate(STEP_TITLES):
            extra = ""
            if m:
                c = m.get("counts") or {}
                extra = {0: " ✔", 1: f" ({c.get('total', 0)})", 2: f" ({c.get('marcada', 0)}/{c.get('total', 0)})",
                         3: " ✔" if m.get("version_list") else "", 4: " ✔" if m.get("active_version_id") else ""}[i]
            self.steps.tab(i, text=f"  {title}{extra}  ")
        self._show_object()
        self._show_photos()
        self._show_classes()
        self._show_train()
        self._show_versions()

    def _write(self, widget, parts: list[tuple[str, str | None]]) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        for text, tag in parts:
            widget.insert("end", text, tag) if tag else widget.insert("end", text)
        widget.configure(state="disabled")

    def _show_object(self) -> None:
        m = self.model
        if not m:
            self._write(self.obj_info, [("Todavía no hay ningún objeto.\n\n", "t"), (
                "Tocá «＋ Nuevo objeto» y poné, por ejemplo:\n"
                "  · Nombre: Pallets\n  · Clases: Pallet vacío · En armado · Terminado con film\n"
                "  · Palabra en inglés (para que la IA proponga recuadros): pallet\n"
                "  · Plan: Conteo de objetos (el plan nuevo, para contar cosas que no son personas)\n", None)])
            return
        parts = [(m["name"] + "\n", "t"),
                 (f"Plan de métricas: {m.get('plan_name')}\n", None),
                 (PLAN_HELP.get(m.get("plan"), "") + "\n\n", "s"),
                 ("Cliente: " + (m.get("client_name") or ALL_CLIENTS) + "\n\n", None),
                 ("Clases (lo que la IA tiene que distinguir):\n", None)]
        for i, c in enumerate(self.classes):
            n = (m.get("boxes") or {}).get(c["key"], 0)
            parts.append((f"  {i + 1}. {c['name']}" + (f"   · palabra para la IA: «{c['prompt']}»" if c.get("prompt")
                                                         else "") + f"   · {plural(n, 'recuadro')} marcados\n", None))
        if m.get("notes"):
            parts.append(("\nNotas: " + m["notes"] + "\n", "s"))
        self._write(self.obj_info, parts)

    def _show_photos(self) -> None:
        m = self.model
        self.photo_counts.configure(text=counts_text(m) if m else "Primero creá el objeto (paso 1).")
        self.jobs.delete(*self.jobs.get_children())
        for j in (m or {}).get("jobs") or []:
            self.jobs.insert("", "end", iid=str(j["id"]), values=(
                j.get("kind_name"), j.get("filename") or "—", job_text(j), j.get("machine") or "—",
                (j.get("created_at") or "")[:16].replace("T", " ")))

    def _show_train(self) -> None:
        m = self.model
        if not m:
            self._write(self.check_box, [("Primero creá el objeto (paso 1).", None)])
            self.machines_line.configure(text="")
            self.train_state.configure(text="")
            return
        parts = [("Para entrenar:\n", "t")]
        for text, ok in check_lines(m):
            parts.append((("✔ " if ok else "✘ ") + text + "\n", "ok" if ok else "no"))
        for w in (m.get("check") or {}).get("warnings") or []:
            parts.append(("⚠ " + w + "\n", "no"))
        for t in (m.get("check") or {}).get("tips") or []:
            parts.append(("· " + t + "\n", None))
        self._write(self.check_box, parts)
        mach = [x for x in m.get("machines") or [] if x.get("train")]
        if mach:
            self.machines_line.configure(text="Máquinas que pueden entrenar ahora: " + ", ".join(
                f"{x['name']} ({x.get('device') or ('placa de video' if x.get('gpu') else 'procesador')})"
                for x in mach) + ("" if any(x.get("gpu") for x in mach) else
                                  ". Sin placa de video tarda horas: conviene la de Tigo con placa."))
        else:
            self.machines_line.configure(text=(
                "Ahora no hay ninguna máquina que pueda entrenar: hace falta una máquina de análisis instalada con "
                "ENTRENAR=si (ver deploy/maquina-analisis/README.md; lo ideal, la de Tigo con placa de video). Podés "
                "pedir el entrenamiento igual: queda en cola."))
        j = active_job(m, "entrenar")
        last = next((x for x in m.get("jobs") or [] if x["kind"] == "entrenar"), None)
        if j:
            self.train_btn.configure(state="disabled")
            self.cancel_train_btn.pack(side="left", padx=10)
            where = f" en {j['machine']}" if j.get("machine") else ""
            self.train_state.configure(text=f"Entrenando{where}: {job_text(j)}" if j["status"] == "procesando" else
                                       "En cola: esperando una máquina que entrene.")
            self.train_bar.configure(value=int(round((j.get("progress") or 0) * 100)))
        else:
            self.train_btn.configure(state="normal" if (m.get("check") or {}).get("ok") else "disabled")
            self.cancel_train_btn.pack_forget()
            self.train_bar.configure(value=0)
            if last and last["status"] == "error":
                self.train_state.configure(text=f"El último entrenamiento falló: {last.get('error')}")
            elif last and last["status"] == "listo":
                self.train_state.configure(text=f"Último entrenamiento: {job_text(last)}. Miralo en «5 · Probar y "
                                                f"activar».")
            else:
                self.train_state.configure(text="")

    def _show_versions(self) -> None:
        sel = self.ver_list.selection()
        self.ver_list.delete(*self.ver_list.get_children())
        vs = (self.model or {}).get("version_list") or []
        for v in vs:
            self.ver_list.insert("", "end", iid=str(v["id"]), values=(
                v["number"], (v.get("created_at") or "")[:16].replace("T", " "), v.get("n_images"), v.get("grade"),
                "✔ Activa" if v.get("active") else "—", v.get("trained_on") or "—"))
        keep = sel[0] if sel and self.ver_list.exists(sel[0]) else (str(vs[0]["id"]) if vs else None)
        if keep:
            self.ver_list.selection_set(keep)
        self._show_version()

    def _version(self) -> dict | None:
        sel = self.ver_list.selection()
        return next((v for v in (self.model or {}).get("version_list") or [] if str(v["id"]) == (sel[0] if sel else
                                                                                                  None)), None)

    def _show_version(self) -> None:
        v = self._version()
        if not v:
            self._write(self.ver_text, [("Todavía no hay versiones: se crean al entrenar (paso 4).", None)])
            return
        lines = list(v.get("text") or [])
        lines.append(f"Entrenada con {plural(v.get('n_images') or 0, 'foto')} y "
                     f"{plural(v.get('n_boxes') or 0, 'recuadro')}.")
        if [c["key"] for c in v.get("classes") or []] != [c["key"] for c in self.classes]:
            lines.append("Ojo: el objeto cambió de clases después de esta versión: entrenar de nuevo para que las "
                         "aprenda.")
        self._write(self.ver_text, [("\n".join(lines), None)])

    def _step_changed(self) -> None:
        if self.step() == 2 and self.model and not self.images:
            self.load_images()

    # ---------- paso 1: el objeto ----------
    def new_object(self) -> None:
        self.object_dialog(None)

    def edit_object(self) -> None:
        if not self.model:
            return self.new_object()
        self.object_dialog(self.model)

    def object_dialog(self, m: dict | None) -> None:
        tk, ttk = self.tk, self.ttk
        top, frm = self.app._dialog("Editar objeto" if m else "Nuevo objeto", 640)
        name = tk.StringVar(value=(m or {}).get("name", ""))
        ttk.Label(frm, text="Nombre del objeto").grid(row=0, column=0, sticky="w", padx=(0, 10))
        ttk.Entry(frm, textvariable=name, width=36).grid(row=0, column=1, sticky="w")
        ttk.Label(frm, text="Por ejemplo «Pallets», «Cajas», «Carros».", style="Suave.TLabel").grid(
            row=1, column=1, sticky="w")
        ttk.Label(frm, text="Clases (los estados que la IA tiene que distinguir) y, opcional, la palabra en inglés "
                            "para que la IA proponga recuadros antes de entrenar:", wraplength=600,
                  justify="left").grid(row=2, column=0, columnspan=2, sticky="w", pady=(10, 2))
        rows_f = ttk.Frame(frm)
        rows_f.grid(row=3, column=0, columnspan=2, sticky="w")
        ttk.Label(rows_f, text="Clase", style="Suave.TLabel").grid(row=0, column=1, sticky="w")
        ttk.Label(rows_f, text="Palabra en inglés", style="Suave.TLabel").grid(row=0, column=2, sticky="w")
        rows: list[dict] = []

        def add_row(c: dict | None = None) -> None:
            if len(rows) >= ob.MAX_CLASSES:
                return
            r = {"key": (c or {}).get("key"), "name": tk.StringVar(value=(c or {}).get("name", "")),
                 "prompt": tk.StringVar(value=(c or {}).get("prompt", ""))}
            i = len(rows) + 1
            r["w"] = [ttk.Label(rows_f, text=f"{i}."),
                      ttk.Entry(rows_f, textvariable=r["name"], width=28),
                      ttk.Entry(rows_f, textvariable=r["prompt"], width=22)]
            for col, w in enumerate(r["w"]):
                w.grid(row=i, column=col, sticky="w", padx=(0, 6), pady=1)
            rows.append(r)

        for c in (m or {}).get("classes") or [None]:
            add_row(c)
        ttk.Button(frm, text="＋ Otra clase", command=add_row).grid(row=4, column=0, columnspan=2, sticky="w",
                                                                   pady=(2, 0))
        ttk.Label(frm, text="Consejo: hasta 3 o 4 clases que una persona distinga de un vistazo. Si dos estados se "
                            "parecen mucho, mejor una sola clase. La palabra en inglés va sin tildes: pallet, box, "
                            "wrapped pallet, shopping cart.", style="Suave.TLabel", wraplength=600,
                  justify="left").grid(row=5, column=0, columnspan=2, sticky="w")
        plans = getattr(self.app, "plan_names", None) or {}
        if not plans:
            from .oficina import PLAN_NAMES as plans  # noqa: N811
        plan_keys = ["objetos"] + [k for k in plans if k != "objetos"]
        plan_var = tk.StringVar(value=plans.get((m or {}).get("plan") or "objetos", plans.get("objetos")))
        ttk.Label(frm, text="Plan de métricas").grid(row=6, column=0, sticky="w", pady=(10, 0), padx=(0, 10))
        pcb = ttk.Combobox(frm, textvariable=plan_var, values=[plans[k] + (" (plan nuevo)" if k == "objetos" else "")
                                                               for k in plan_keys], state="readonly", width=34)
        pcb.grid(row=6, column=1, sticky="w", pady=(10, 0))
        if (m or {}).get("plan", "objetos") == "objetos":
            pcb.current(0)
        plan_help = ttk.Label(frm, text="", style="Suave.TLabel", wraplength=600, justify="left")
        plan_help.grid(row=7, column=0, columnspan=2, sticky="w")

        def plan_key() -> str:
            i = pcb.current()
            return plan_keys[i] if i >= 0 else "objetos"

        def show_help(_e=None) -> None:
            plan_help.configure(text=PLAN_HELP.get(plan_key(), ""))
        pcb.bind("<<ComboboxSelected>>", show_help)
        show_help()
        clients = [ALL_CLIENTS] + [c["name"] for c in self.app.clients]
        cli = tk.StringVar(value=next((c["name"] for c in self.app.clients if c["id"] == (m or {}).get("client_id")),
                                      ALL_CLIENTS))
        ttk.Label(frm, text="Cliente").grid(row=8, column=0, sticky="w", pady=(10, 0), padx=(0, 10))
        ttk.Combobox(frm, textvariable=cli, values=clients, state="readonly", width=34).grid(row=8, column=1,
                                                                                           sticky="w", pady=(10, 0))
        ttk.Label(frm, text="«Todos los clientes» si el mismo objeto sirve en varios (por ejemplo pallets en todos los "
                            "depósitos): así aprende de las fotos de todos.", style="Suave.TLabel", wraplength=600,
                  justify="left").grid(row=9, column=0, columnspan=2, sticky="w")
        notes = tk.StringVar(value=(m or {}).get("notes") or "")
        ttk.Label(frm, text="Notas").grid(row=10, column=0, sticky="w", pady=(10, 0), padx=(0, 10))
        ttk.Entry(frm, textvariable=notes, width=50).grid(row=10, column=1, sticky="w", pady=(10, 0))

        def save() -> None:
            classes = [{**({"key": r["key"]} if r["key"] else {}), "name": r["name"].get().strip(),
                        "prompt": r["prompt"].get().strip()} for r in rows if r["name"].get().strip() or r["key"]]
            cid = next((c["id"] for c in self.app.clients if c["name"] == cli.get()), None)
            body = {"name": name.get().strip(), "classes": classes, "plan": plan_key(), "client_id": cid,
                    "notes": notes.get().strip() or None}

            def done(res) -> None:
                top.destroy()
                self.model = res
                self.load_models(select_id=res["id"])
                if m is None:
                    self.go(1)  # creado: a juntar fotos
            path = f"/api/objects/{m['id']}" if m else "/api/objects"
            fn = (lambda: self.app.api.put(path, body)) if m else (lambda: self.app.api.post(path, body))
            self.app.bg.run(fn, done, self._fail)
        bb = ttk.Frame(frm)
        bb.grid(row=11, column=0, columnspan=2, sticky="e", pady=(14, 0))
        ttk.Button(bb, text="Cancelar", command=top.destroy).pack(side="right")
        ttk.Button(bb, text="Guardar", style="Primario.TButton", command=save).pack(side="right", padx=6)
        top._keep = (name, plan_var, cli, notes, rows)  # (si no, Python borra las variables y el formulario queda vacío)
        self._dialog_top = top  # (las pruebas lo contestan)
        self._dialog_rows = rows

    def delete_object(self) -> None:
        m = self.model
        if not m:
            return
        c = m.get("counts") or {}
        if not self.app.mb.askyesno(NOMBRE, f"¿Borrar el objeto «{m['name']}» con sus {c.get('total', 0)} fotos, "
                                            f"sus marcas y sus versiones? No se puede deshacer.", default="no"):
            return

        def done(_r) -> None:
            self.model, self.images, self.cur = None, [], None
            self.load_models()
        self.app.bg.run(lambda: self.app.api.delete(f"/api/objects/{m['id']}"), done, self._fail)

    # ---------- paso 2: fotos ----------
    def _need_model(self) -> bool:
        if not self.model:
            self._say("info", "Primero creá el objeto («＋ Nuevo objeto»).")
            return False
        if self.busy:
            self._say("info", "Ya se están subiendo fotos: esperá a que termine (o «■ Frenar»).")
            return False
        return True

    def upload_photos(self) -> None:
        if not self._need_model():
            return
        from tkinter import filedialog
        paths = filedialog.askopenfilenames(parent=self.root, title=f"Fotos de «{self.model['name']}»",
                                            filetypes=[PHOTO_TYPES, ("Todos los archivos", "*.*")])
        if paths:
            self.start_work(lambda stop: self._photos_work(list(paths), stop), f"Subiendo {plural(len(paths), 'foto')}")

    def start_work(self, fn, what: str) -> None:
        """Corre `fn(stop)` en otro hilo (subir fotos o sacarlas de un video) y muestra cómo va."""
        self.busy = True
        self._stop.clear()
        self.stop_btn.pack(side="left", padx=12)
        self.work_line.configure(text=what + "…")
        mid = self.model["id"]

        def run() -> None:
            try:
                msg = fn(self._stop)
            except Exception as exc:  # noqa: BLE001
                log.exception("Entrenar objetos: falló la subida")
                msg = f"No se pudo terminar: {exc}"
            self._later(self._work_done, (mid, msg))
        threading.Thread(target=run, daemon=True, name="objetos-subir").start()
        self._ensure_poll()

    def _work_done(self, arg) -> None:
        mid, msg = arg
        self.busy = False
        self.stop_btn.pack_forget()
        self.work_line.configure(text=msg or "")
        if self.model and self.model["id"] == mid:
            self.images, self.cur = [], None
            self.load_model(mid)

    def _progress(self, text: str) -> None:
        self._later(lambda t: self.work_line.configure(text=t), text)

    def send_photos(self, mid: int, shots: list[tuple[bytes, int, int, str, float | None]], source: str,
                    camera_id: int | None = None) -> tuple[int, int]:
        """Registra y sube fotos ya achicadas (JPEG, ancho, alto, nombre, segundo del video). (nuevas, repetidas)."""
        api = self.app.api
        body = {"images": [{"uuid": ob.image_uid(j), "name": n[:200], "width": w, "height": h, "source": source,
                            "camera_id": camera_id, "video_ts": ts} for j, w, h, n, ts in shots]}
        by_uid = {ob.image_uid(j): j for j, *_ in shots}
        items = api.post(f"/api/objects/{mid}/images", body)["items"]
        ids = []
        for it in items:
            if it.get("upload_url"):
                api.upload(it["upload_url"], by_uid[it["uuid"]], "image/jpeg")
                ids.append(it["id"])
        if ids:
            api.post(f"/api/objects/{mid}/images/uploaded", {"ids": ids})
        return len(ids), sum(1 for it in items if it.get("existed"))

    def _photos_work(self, paths: list[str], stop) -> str:
        mid = self.model["id"]
        new = old = bad = 0
        batch: list = []
        for n, path in enumerate(paths):
            if stop.is_set():
                break
            img = ob.read_photo(path)
            if img is None:
                bad += 1
                continue
            img = ob.fit_photo(img)
            batch.append((ob.encode_photo(img), img.shape[1], img.shape[0], Path(path).name, None))
            if len(batch) >= 20:
                a, b = self.send_photos(mid, batch, "foto")
                new, old, batch = new + a, old + b, []
            self._progress(f"Subiendo fotos: {n + 1} de {len(paths)}…")
        if batch and not stop.is_set():
            a, b = self.send_photos(mid, batch, "foto")
            new, old = new + a, old + b
        return self._upload_summary(new, old, bad, stop.is_set())

    @staticmethod
    def _upload_summary(new: int, old: int, bad: int, stopped: bool) -> str:
        parts = [f"✔ {plural(new, 'foto nueva', 'fotos nuevas')}"]
        if old:
            parts.append(f"{plural(old, 'ya estaba', 'ya estaban')}")
        if bad:
            parts.append(f"{plural(bad, 'archivo')} no {'es una imagen' if bad == 1 else 'son imágenes'}")
        return " · ".join(parts) + (" (frenado)" if stopped else "") + ". Siguiente: «3 · Marcar»."

    def video_dialog(self) -> None:
        if not self._need_model():
            return
        from tkinter import filedialog
        path = filedialog.askopenfilename(parent=self.root, title=f"Video para sacar fotos de «{self.model['name']}»",
                                          filetypes=[VIDEO_TYPES, ("Todos los archivos", "*.*")])
        if not path:
            return
        tk, ttk = self.tk, self.ttk
        size_mb = Path(path).stat().st_size / 1e6
        top, frm = self.app._dialog("Sacar fotos de un video", 560)
        ttk.Label(frm, text=f"{Path(path).name} ({size_mb:.0f} MB)", font=("Segoe UI", 10, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w")
        every = tk.DoubleVar(value=ob.VIDEO_EVERY_S)
        most = tk.IntVar(value=ob.VIDEO_MAX_IMAGES)
        ttk.Label(frm, text="Una foto cada (segundos)").grid(row=1, column=0, sticky="w", pady=(10, 2), padx=(0, 10))
        ttk.Spinbox(frm, from_=0.5, to=60, increment=0.5, textvariable=every, width=6).grid(row=1, column=1,
                                                                                          sticky="w", pady=(10, 2))
        ttk.Label(frm, text="Hasta (fotos)").grid(row=2, column=0, sticky="w", pady=2, padx=(0, 10))
        ttk.Spinbox(frm, from_=10, to=2000, increment=10, textvariable=most, width=6).grid(row=2, column=1,
                                                                                         sticky="w", pady=2)
        ttk.Label(frm, text="Las fotos casi iguales a la anterior se saltean (la escena quieta no enseña nada).",
                  style="Suave.TLabel", wraplength=520, justify="left").grid(row=3, column=0, columnspan=2,
                                                                             sticky="w")
        where = tk.StringVar(value="pc" if size_mb > VM_VIDEO_MB else "vm")
        ttk.Label(frm, text="¿Dónde se sacan las fotos?").grid(row=4, column=0, columnspan=2, sticky="w", pady=(10, 2))
        vm = ttk.Radiobutton(frm, text="En una máquina virtual: se sube el video entero y la máquina saca las fotos "
                                       "(y, si puede, la IA las pre-marca). Esta PC queda libre.", variable=where,
                             value="vm")
        vm.grid(row=5, column=0, columnspan=2, sticky="w")
        ttk.Radiobutton(frm, text="En esta PC: se suben solo las fotos (más rápido con videos largos o internet "
                                  "lenta).", variable=where, value="pc").grid(row=6, column=0, columnspan=2,
                                                                              sticky="w")
        if size_mb > VM_VIDEO_MB:
            vm.configure(state="disabled")
            ttk.Label(frm, text=f"El video pesa más de {VM_VIDEO_MB} MB: tardaría mucho en subir; se sacan las fotos "
                                f"acá.", style="Aviso.TLabel", wraplength=520).grid(row=7, column=0, columnspan=2,
                                                                                    sticky="w")
        out: list = [None]

        def ok() -> None:
            try:
                out[0] = (where.get(), max(0.5, float(every.get())), max(1, int(most.get())))
            except (ValueError, tk.TclError):
                return
            top.destroy()
        bb = ttk.Frame(frm)
        bb.grid(row=8, column=0, columnspan=2, sticky="e", pady=(12, 0))
        ttk.Button(bb, text="Cancelar", command=top.destroy).pack(side="right")
        ttk.Button(bb, text="Sacar fotos", style="Primario.TButton", command=ok).pack(side="right", padx=6)
        self._dialog_top = top
        top.grab_set()
        top.wait_window()
        if out[0] is None:
            return
        place, every_s, max_images = out[0]
        if place == "pc":
            self.start_work(lambda stop: self._video_work(path, every_s, max_images, stop),
                            f"Sacando fotos de {Path(path).name}")
        else:
            self.start_work(lambda stop: self._video_vm(path, every_s, max_images, stop),
                            f"Subiendo {Path(path).name} a la nube")

    def _video_work(self, path: str, every_s: float, max_images: int, stop) -> str:
        mid = self.model["id"]
        new = old = 0
        batch: list = []
        name = Path(path).name
        seen = 0
        for ts, frame in ob.video_frames(path, every_s, max_images, stop=stop.is_set,
                                         progress=lambda f: self._progress(
                                             f"Sacando fotos de {name}: {int(f * 100)} % del video, "
                                             f"{seen} fotos…")):
            img = ob.fit_photo(frame)
            batch.append((ob.encode_photo(img), img.shape[1], img.shape[0], f"{name} {int(ts // 60)}:{int(ts % 60):02d}",
                          ts))
            seen += 1
            if len(batch) >= 20:
                a, b = self.send_photos(mid, batch, "video")
                new, old, batch = new + a, old + b, []
        if batch and not stop.is_set():
            a, b = self.send_photos(mid, batch, "video")
            new, old = new + a, old + b
        return self._upload_summary(new, old, 0, stop.is_set())

    def _video_vm(self, path: str, every_s: float, max_images: int, stop) -> str:
        api = self.app.api
        mid = self.model["id"]
        size = Path(path).stat().st_size
        job = api.post(f"/api/objects/{mid}/videos", {"filename": Path(path).name, "size": size,
                                                      "every_s": every_s, "max_images": max_images})
        self._progress(f"Subiendo {Path(path).name} ({size / 1e6:.0f} MB) a la nube… (la ventana sigue andando)")
        try:
            with open(path, "rb") as fh:  # (de a pedazos: un video de 1 GB no entra entero en memoria)
                api.upload(job["upload_url"], fh, "video/mp4")
        except Exception as exc:
            try:
                api.post(f"/api/objects/jobs/{job['id']}/cancel")
            except Exception:  # noqa: BLE001
                pass
            raise RuntimeError(f"{exc}. Si el video es muy grande para la nube, sacá las fotos en esta PC.") from exc
        if stop.is_set():
            api.post(f"/api/objects/jobs/{job['id']}/cancel")
            return "Frenado: el video no se mandó a la máquina."
        res = api.post(f"/api/objects/jobs/{job['id']}/uploaded")
        return ("✔ Video subido: una máquina virtual saca las fotos (se ve en la lista de abajo). "
                + (res.get("note") or ""))

    def camera_dialog(self) -> None:
        if not self._need_model():
            return
        tk, ttk = self.tk, self.ttk
        top, frm = self.app._dialog("Foto de una cámara", 520)
        cams = sorted(self.app.cameras, key=lambda k: (k.get("client_name") or "", k.get("name") or ""))
        labels = [f"{k.get('client_name') or self._client(k.get('client_id'))} · {k['name']}" for k in cams]
        var = tk.StringVar(value=labels[0] if labels else "")
        ttk.Label(frm, text="Cámara").grid(row=0, column=0, sticky="w", padx=(0, 10))
        ttk.Combobox(frm, textvariable=var, values=labels, state="readonly", width=46).grid(row=0, column=1,
                                                                                          sticky="w")
        ttk.Label(frm, text="Se abre el video de la cámara unos segundos desde esta PC y se guarda una foto de ahora. "
                            "Sacá varias en momentos distintos del día.", style="Suave.TLabel", wraplength=480,
                  justify="left").grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))
        out: list = [None]

        def ok() -> None:
            if var.get() in labels:
                out[0] = cams[labels.index(var.get())]
            top.destroy()
        bb = ttk.Frame(frm)
        bb.grid(row=2, column=0, columnspan=2, sticky="e", pady=(12, 0))
        ttk.Button(bb, text="Cancelar", command=top.destroy).pack(side="right")
        ttk.Button(bb, text="Sacar la foto", style="Primario.TButton", command=ok).pack(side="right", padx=6)
        self._dialog_top = top
        top.grab_set()
        top.wait_window()
        cam = out[0]
        if cam is None:
            return
        self.start_work(lambda stop: self._camera_work(cam), f"Sacando una foto de {cam['name']}")

    def _client(self, cid) -> str:
        return next((c["name"] for c in self.app.clients if c["id"] == cid), "—")

    def camera_frame(self, cid: int) -> np.ndarray:
        """Un cuadro de la cámara: el del vivo si se está mirando, si no se abre el video unos segundos."""
        agent = self.app.agent_of() if getattr(self.app, "agent_of", None) else None
        if agent is None:
            raise RuntimeError("la app todavía está arrancando: probá de nuevo en unos segundos")
        frame = agent.live_frame(cid)
        if frame is None:
            from .nombres import grab_frame
            views = agent.client.view([cid])
            if not views:
                raise RuntimeError("la cámara no está disponible (¿está deshabilitada?)")
            frame = grab_frame(views[0]["connection"])
        return frame

    def _camera_work(self, cam: dict) -> str:
        frame = ob.fit_photo(self.camera_frame(cam["id"]))
        when = time.strftime("%d/%m %H:%M")
        a, b = self.send_photos(self.model["id"], [(ob.encode_photo(frame), frame.shape[1], frame.shape[0],
                                                     f"{cam['name']} {when}", None)], "camara", cam["id"])
        return f"✔ Foto de {cam['name']} guardada." if a else "Esa foto ya estaba."

    def _selected_job(self) -> dict | None:
        sel = self.jobs.selection()
        return next((j for j in (self.model or {}).get("jobs") or [] if sel and str(j["id"]) == sel[0]), None)

    def cancel_job(self, job: dict | None = None) -> None:
        j = job or self._selected_job()
        if not j:
            return self._say("info", "Elegí un trabajo de la lista.")
        if j["status"] not in ("subiendo", "en_cola", "procesando"):
            return self._say("info", "Ese trabajo ya terminó.")
        if not self.app.mb.askyesno(NOMBRE, f"¿Cancelar «{j['kind_name']}»?", default="no"):
            return
        self.app.bg.run(lambda: self.app.api.post(f"/api/objects/jobs/{j['id']}/cancel"),
                        lambda _r: self.load_model(self.model["id"], images=False), self._fail)

    # ---------- paso 3: marcar ----------
    def _filter(self) -> tuple[str, str | None]:
        val = self.filter_var.get()
        f = next((x for x in FILTERS if x[1] == val), FILTERS[0])
        return f[0], f[2]

    def load_images(self) -> None:
        if not self.model:
            return
        self._flush_label()
        mid = self.model["id"]
        order, status = self._filter()
        self._img_req += 1
        req = self._img_req

        def done(res) -> None:
            if req != self._img_req or not self.model or self.model["id"] != mid:
                return
            self.images = res.get("items") or []
            self.list_title.configure(text=f"{plural(res.get('total', 0), 'foto')}"
                                           + (f" (se muestran {len(self.images)})" if res.get("total", 0) > len(
                                               self.images) else ""))
            self._fill_list()
            self.open_image(0 if self.images else None)
        self.app.bg.run(lambda: self.app.api.get(f"/api/objects/{mid}/images", status=status,
                                                 order="dudosas" if order == "dudosas" else "nuevas", limit=PAGE),
                        done, self._fail)

    def _row(self, i: int, im: dict) -> tuple:
        name = im.get("source_name") or f"Foto {im['id']}"
        st = ob.STATUS_NAMES.get(im["status"], im["status"])
        if im["status"] == "sin_marcar" and im.get("proposed"):
            st = "IA propone"
        return (name, st, len(im.get("boxes") or []) if im["status"] == "marcada" else "")

    def _fill_list(self) -> None:
        self.img_list.delete(*self.img_list.get_children())
        for i, im in enumerate(self.images):
            self.img_list.insert("", "end", iid=str(i), values=self._row(i, im))

    def _list_pick(self) -> None:
        sel = self.img_list.selection()
        if sel and int(sel[0]) != self.cur:
            self._flush_label()
            self.open_image(int(sel[0]))

    def _show_classes(self) -> None:
        keys = [c["key"] for c in self.classes]
        if self.cls not in keys:
            self.cls = keys[0] if keys else None
        self.cls_var.set(self.cls or "")
        shown = [(c["key"], c["name"]) for c in self.classes]
        if shown == getattr(self, "_cls_shown", None):
            return  # (se refresca cada 5 s mientras trabaja una máquina: no rehacer los botones)
        self._cls_shown = shown
        for w in self.cls_bar.winfo_children():
            w.destroy()
        self.ttk.Label(self.cls_bar, text="Clase:").pack(side="left")
        for i, c in enumerate(self.classes):
            b = self.tk.Radiobutton(self.cls_bar, text=f"{i + 1} {c['name']}", variable=self.cls_var, value=c["key"],
                                    indicatoron=False, foreground=COLORS[i % len(COLORS)], selectcolor="#ffe08a",
                                    background="#ffffff", activebackground="#fff4cc", font=("Segoe UI", 10, "bold"),
                                    borderwidth=2, padx=8, pady=2, takefocus=0,
                                    command=lambda k=c["key"]: self.set_class(k))
            b.pack(side="left", padx=(6, 0))

    def set_class(self, key: str) -> None:
        """Elige la clase para los recuadros nuevos y, si hay uno elegido, se la cambia."""
        self.cls = key
        self.cls_var.set(key)
        if self.sel is not None and self.sel < len(self.boxes) and self.boxes[self.sel]["k"] != key:
            self.boxes[self.sel]["k"] = key
            self.dirty = True
            self._redraw()

    def open_image(self, i: int | None) -> None:
        self.cur, self.sel, self._drag = i, None, None
        if i is None or i >= len(self.images):
            self.cur = None
            self.boxes, self.proposed, self.dirty = [], [], False
            self._redraw()
            return
        im = self.images[i]
        self.boxes = [dict(b) for b in im.get("boxes") or []]
        self.proposed = [dict(b) for b in im.get("proposed") or []] if im["status"] != "marcada" else []
        self.dirty = False
        if not (self.img_list.selection() and self.img_list.selection()[0] == str(i)):
            self.img_list.selection_set(str(i))
            self.img_list.see(str(i))
        self._prefetch(i)
        self._redraw()

    def _prefetch(self, i: int) -> None:
        want = [im["id"] for im in self.images[i:i + 6] if im["id"] not in self._pix and im["id"] not in self._fetching]
        if not want:
            return
        self._fetching.update(want)
        api = self.app.api

        def work():
            urls = api.post("/api/objects/images/urls", {"ids": want})["urls"]
            got = {}
            for iid in want:
                url = urls.get(str(iid))
                if not url:
                    continue
                data = api.fetch(url)
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    got[iid] = img
            return got

        def done(got) -> None:
            self._fetching.difference_update(want)
            for iid, img in got.items():
                self._pix[iid] = img
                self._pix.move_to_end(iid)
            while len(self._pix) > CACHE:
                self._pix.popitem(last=False)
            self._redraw()

        def fail(exc) -> None:
            self._fetching.difference_update(want)
            log.warning("Entrenar objetos: no se pudieron bajar fotos (%s)", exc)
            self.label_line.configure(text=f"No se pudo bajar la foto: {exc}")
        self.app.bg.run(work, done, fail)

    def current(self) -> dict | None:
        return self.images[self.cur] if self.cur is not None and self.cur < len(self.images) else None

    def _redraw(self) -> None:
        c = self.canvas
        c.delete("all")
        im = self.current()
        cw, ch = max(50, c.winfo_width()), max(50, c.winfo_height())
        if im is None:
            msg = ("Todavía no hay fotos: subilas en «2 · Juntar fotos»." if self.model else
                   "Primero creá el objeto («＋ Nuevo objeto»).")
            if self.model and self.images == [] and (self.model.get("counts") or {}).get("total"):
                msg = "No hay fotos con este filtro. Cambiá «Sin marcar» por «Todas» o «Marcadas» para verlas."
            c.create_text(cw // 2, ch // 2, text=msg, fill="#dddddd", font=("Segoe UI", 12), width=cw - 40)
            self.label_line.configure(text="")
            return
        w, h = im.get("width") or 1280, im.get("height") or 720
        k, ox, oy = fit(w, h, cw, ch)
        dw, dh = int(w * k), int(h * k)
        self._geo = (k, ox, oy, max(1, dw), max(1, dh))
        img = self._pix.get(im["id"])
        if img is not None:
            small = cv2.resize(img, (max(1, dw), max(1, dh)), interpolation=cv2.INTER_AREA)
            ok, ppm = cv2.imencode(".ppm", small)
            self._photo = self.tk.PhotoImage(data=ppm.tobytes()) if ok else None
            if self._photo is not None:
                c.create_image(ox, oy, image=self._photo, anchor="nw")
        else:
            c.create_rectangle(ox, oy, ox + dw, oy + dh, fill="#444444", outline="")
            c.create_text(cw // 2, ch // 2, text="Bajando la foto…", fill="#dddddd", font=("Segoe UI", 11))
        names = ob.class_names(self.classes)
        for b in self.proposed:
            x1, y1, x2, y2 = self._to_canvas(b)
            c.create_rectangle(x1, y1, x2, y2, outline=PROPOSED, dash=(5, 3), width=2)
            c.create_text(x1 + 3, y1 + 2, anchor="nw", fill=PROPOSED, font=("Segoe UI", 9, "bold"),
                          text=f"¿{names.get(b.get('k'), 'objeto')}? {int(round(b.get('p', 0) * 100))}%")
        for i, b in enumerate(self.boxes):
            x1, y1, x2, y2 = self._to_canvas(b)
            col = self.color_of(b["k"])
            c.create_rectangle(x1, y1, x2, y2, outline=col, width=3 if i == self.sel else 2)
            num = [x["key"] for x in self.classes].index(b["k"]) + 1 if b["k"] in names else "?"
            c.create_rectangle(x1, y1, x1 + 18, y1 + 16, fill=col, outline="")
            c.create_text(x1 + 9, y1 + 8, text=str(num), fill="white", font=("Segoe UI", 9, "bold"))
            if i == self.sel:
                for hx, hy in ((x1, y1), (x2, y1), (x1, y2), (x2, y2)):
                    c.create_rectangle(hx - HANDLE, hy - HANDLE, hx + HANDLE, hy + HANDLE, fill="white", outline=col)
        if self._drag and self._drag.get("mode") == "new" and self._drag.get("rect"):
            x1, y1, x2, y2 = self._drag["rect"]
            c.create_rectangle(x1, y1, x2, y2, outline=self.color_of(self.cls), width=2, dash=(3, 2))
        self._show_label_line()

    def _show_label_line(self) -> None:
        im = self.current()
        if im is None:
            return
        n = len(self.boxes)
        state = ob.STATUS_NAMES.get(im["status"], im["status"])
        src = im.get("source_name") or f"Foto {im['id']}"
        extra = ""
        if self.proposed:
            extra = (f" · {im.get('proposed_by') or 'la IA'} propone {plural(len(self.proposed), 'recuadro')} en "
                     f"naranja: A acepta todos, clic acepta uno")
        uso = "de prueba (la IA no la ve al entrenar)" if im.get("split") == "prueba" else "para entrenar"
        self.label_line.configure(text=(
            f"{src} · {state} · {plural(n, 'recuadro')} · foto {uso}{extra}{' · sin guardar' if self.dirty else ''}"))

    # coordenadas
    def _to_canvas(self, b: dict) -> tuple[float, float, float, float]:
        _, ox, oy, dw, dh = self._geo
        return ox + b["x1"] * dw, oy + b["y1"] * dh, ox + b["x2"] * dw, oy + b["y2"] * dh

    def _to_img(self, x: float, y: float) -> tuple[float, float]:
        _, ox, oy, dw, dh = self._geo
        return min(1.0, max(0.0, (x - ox) / dw)), min(1.0, max(0.0, (y - oy) / dh))

    def _handle_at(self, x: float, y: float) -> int | None:
        if self.sel is None or self.sel >= len(self.boxes):
            return None
        x1, y1, x2, y2 = self._to_canvas(self.boxes[self.sel])
        for i, (hx, hy) in enumerate(((x1, y1), (x2, y1), (x1, y2), (x2, y2))):
            if abs(x - hx) <= HANDLE + 2 and abs(y - hy) <= HANDLE + 2:
                return i
        return None

    def _press(self, e) -> None:
        if self.current() is None:
            return
        self.canvas.focus_set()
        corner = self._handle_at(e.x, e.y)
        if corner is not None:
            self._drag = {"mode": "resize", "corner": corner}
            return
        px, py = self._to_img(e.x, e.y)
        hit = box_at(self.boxes, px, py)
        if hit is not None:
            self.sel = hit
            b = self.boxes[hit]
            self._drag = {"mode": "move", "start": (px, py), "orig": dict(b)}
            self.cls = b["k"]
            self.cls_var.set(b["k"])
            self._redraw()
            return
        prop = box_at(self.proposed, px, py)
        if prop is not None:  # clic en lo que propone la IA: se acepta ese
            self.accept_proposed(prop)
            return
        self.sel = None
        self._drag = {"mode": "new", "start": (e.x, e.y), "rect": None}

    def _motion(self, e) -> None:
        d = self._drag
        if not d:
            return
        if d["mode"] == "new":
            sx, sy = d["start"]
            d["rect"] = (min(sx, e.x), min(sy, e.y), max(sx, e.x), max(sy, e.y))
        elif d["mode"] == "move" and self.sel is not None:
            px, py = self._to_img(e.x, e.y)
            o = d["orig"]
            dx = min(max(px - d["start"][0], -o["x1"]), 1 - o["x2"])
            dy = min(max(py - d["start"][1], -o["y1"]), 1 - o["y2"])
            self.boxes[self.sel].update(x1=o["x1"] + dx, x2=o["x2"] + dx, y1=o["y1"] + dy, y2=o["y2"] + dy)
            self.dirty = True
        elif d["mode"] == "resize" and self.sel is not None:
            px, py = self._to_img(e.x, e.y)
            b = self.boxes[self.sel]
            kx = "x1" if d["corner"] in (0, 2) else "x2"
            ky = "y1" if d["corner"] in (0, 1) else "y2"
            b[kx], b[ky] = px, py
            self.dirty = True
        self._redraw()

    def _release(self, e) -> None:
        d, self._drag = self._drag, None
        if not d:
            return
        if d["mode"] == "new" and d.get("rect") and self.cls:
            x1, y1, x2, y2 = d["rect"]
            a, b_ = self._to_img(x1, y1)
            c_, d_ = self._to_img(x2, y2)
            box = ob.clean_boxes([{"k": self.cls, "x1": a, "y1": b_, "x2": c_, "y2": d_}], [self.cls])
            if box:
                self.boxes.append(box[0])
                self.sel = len(self.boxes) - 1
                self.dirty = True
        elif d["mode"] in ("move", "resize") and self.sel is not None:
            fixed = ob.clean_boxes([self.boxes[self.sel]], [x["key"] for x in self.classes] + [None])
            if fixed:
                self.boxes[self.sel] = fixed[0]
            else:
                self.boxes.pop(self.sel)
                self.sel = None
        self._redraw()

    def _right_click(self, e) -> None:
        px, py = self._to_img(e.x, e.y)
        hit = box_at(self.boxes, px, py)
        if hit is not None:
            self.boxes.pop(hit)
            self.sel, self.dirty = None, True
            self._redraw()

    def _key(self, e) -> None:
        """Teclas de «Marcar» (solo con ese paso a la vista y sin estar escribiendo en un campo)."""
        if not self._alive() or not self.visible() or self.step() != 2 or self.current() is None:
            return
        try:
            w = self.root.focus_get()
            if w is not None and w.winfo_class() in ("TEntry", "Entry", "Text", "TCombobox", "TSpinbox"):
                return
            if w is not None and w.winfo_toplevel() is not self.root.winfo_toplevel():
                return
        except Exception:  # noqa: BLE001
            pass
        k = e.keysym
        if k in [str(i) for i in range(1, 9)]:
            i = int(k) - 1
            if i < len(self.classes):
                self.set_class(self.classes[i]["key"])
        elif k in ("Return", "KP_Enter"):
            self.save_and_next()
        elif k in ("Delete", "BackSpace"):
            self.delete_box()
        elif k.lower() == "n":
            self.none_here()
        elif k.lower() == "a":
            self.accept_proposed()
        elif k.lower() == "d":
            self.discard()
        elif k == "Right":
            self.move(1)
        elif k == "Left":
            self.move(-1)

    # acciones
    def accept_proposed(self, only: int | None = None) -> None:
        if not self.proposed:
            return
        keys = [c["key"] for c in self.classes]
        take = [self.proposed[only]] if only is not None else list(self.proposed)
        for b in take:
            k = b.get("k") if b.get("k") in keys else self.cls
            if k:
                self.boxes.append({"k": k, **{n: b[n] for n in ("x1", "y1", "x2", "y2")}})
        self.proposed = [b for i, b in enumerate(self.proposed) if only is not None and i != only]
        self.dirty = True
        self.sel = len(self.boxes) - 1 if self.boxes else None
        self._redraw()

    def delete_box(self) -> None:
        if self.sel is not None and self.sel < len(self.boxes):
            self.boxes.pop(self.sel)
            self.sel, self.dirty = None, True
            self._redraw()

    def _save(self, body: dict, advance: bool = True) -> None:
        im = self.current()
        if im is None:
            return
        idx = self.cur
        iid = im["id"]
        _, status = self._filter()

        def done(res) -> None:
            if idx < len(self.images) and self.images[idx]["id"] == iid:
                self.images[idx] = res
                if self.img_list.exists(str(idx)):
                    self.img_list.item(str(idx), values=self._row(idx, res))
            if self.model:
                self.load_model(self.model["id"], images=False)  # (cuenta lo marcado)
        self.app.bg.run(lambda: self.app.api.put(f"/api/objects/images/{iid}", body), done, self._fail)
        # mientras tanto, se sigue con la próxima (no se espera al servidor)
        im.update(status=body.get("status") or "marcada", boxes=body.get("boxes", im.get("boxes")), proposed=[])
        self.dirty = False
        if self.img_list.exists(str(idx)):
            self.img_list.item(str(idx), values=self._row(idx, im))
        if advance:
            self.move(1, flush=False)

    def save_and_next(self) -> None:
        if self.current() is None:
            return
        if not self.boxes and self.proposed:
            return self._say("info", "La IA propone recuadros (en naranja): aceptalos con A, o «No hay ninguno» si "
                                     "no hay.")
        if not self.boxes:
            return self.none_here()
        self._save({"boxes": self.boxes})

    def none_here(self) -> None:
        if self.current() is not None:
            self.boxes, self.proposed = [], []
            self._save({"boxes": []})

    def discard(self) -> None:
        if self.current() is not None:
            self._save({"status": "descartada"})

    def _flush_label(self) -> None:
        """Al dejar una foto con cambios sin guardar (otra foto, otro filtro u otro objeto), se guardan."""
        if self.dirty and self.current() is not None and self.boxes:
            self._save({"boxes": self.boxes}, advance=False)
        self.dirty = False

    def move(self, step: int, flush: bool = True) -> None:
        if self.cur is None:
            return
        if flush:
            self._flush_label()
        j = self.cur + step
        if 0 <= j < len(self.images):
            self.open_image(j)
        elif j >= len(self.images):
            self.open_image(self.cur)
            self.label_line.configure(text="✔ Era la última de la lista. Cambiá el filtro para ver otras, o pasá a "
                                           "«4 · Entrenar».")

    def prelabel(self) -> None:
        if not self.model:
            return
        m = self.model
        if not m.get("active_version_id") and not any(c.get("prompt") for c in self.classes):
            return self._say("info", "Para que la IA proponga recuadros hace falta una versión activa (paso 5) o la "
                                     "palabra en inglés de alguna clase («✎ Editar objeto»).")

        def done(res) -> None:
            self._say("info", "La IA va a proponer recuadros en las fotos sin marcar (lo hace una máquina virtual: "
                              "en unos minutos aparecen en naranja). " + (res.get("note") or ""))
            self.load_model(m["id"], images=False)
        self.app.bg.run(lambda: self.app.api.post(f"/api/objects/{m['id']}/prelabel"), done, self._fail)

    # ---------- paso 4: entrenar ----------
    def train(self) -> None:
        m = self.model
        if not m:
            return

        def done(res) -> None:
            text = "Listo: el entrenamiento quedó pedido. Se ve cómo va acá mismo (se actualiza solo)."
            if res.get("warnings"):
                text += "\n\nOjo: " + " ".join(res["warnings"])
            if res.get("note"):
                text += "\n\n" + res["note"]
            self._say("info", text)
            self.load_model(m["id"], images=False)
        self.app.bg.run(lambda: self.app.api.post(f"/api/objects/{m['id']}/train", {"fresh": self.fresh_var.get()}),
                        done, self._fail)

    def cancel_training(self) -> None:
        j = active_job(self.model, "entrenar")
        if j:
            self.cancel_job(j)

    # ---------- paso 5: versiones ----------
    def set_version(self, on: bool) -> None:
        v = self._version()
        if not v:
            return self._say("info", "Todavía no hay versiones: se crean al entrenar (paso 4).")
        path = f"/api/objects/versions/{v['id']}/{'activate' if on else 'deactivate'}"
        self.app.bg.run(lambda: self.app.api.post(path), lambda _r: self.load_model(self.model["id"], images=False),
                        self._fail)

    def version_detector(self, v: dict):
        """El modelo de una versión para correrlo en esta PC (se baja una vez y queda en la carpeta de datos)."""
        path = self._local_dir() / f"objeto{v['model_id']}-v{v['number']}-{v['id']}.onnx"
        info = self.app.api.get(f"/api/objects/versions/{v['id']}/model")
        if not path.exists():
            data = self.app.api.fetch(info["url"])
            tmp = path.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.replace(path)
        return ob.ObjectDetector(path, [c["key"] for c in info["classes"]], conf=info.get("conf") or ob.DEFAULT_CONF), \
            info["classes"]

    def test_photo(self) -> None:
        v = self._version()
        if not v:
            return self._say("info", "Todavía no hay versiones: se crean al entrenar (paso 4).")
        from tkinter import filedialog
        path = filedialog.askopenfilename(parent=self.root, title="Foto para probar la versión",
                                          filetypes=[PHOTO_TYPES, ("Todos los archivos", "*.*")])
        if path:
            self._test(v, lambda: ob.read_photo(path), Path(path).name)

    def test_camera(self) -> None:
        v = self._version()
        if not v:
            return self._say("info", "Todavía no hay versiones: se crean al entrenar (paso 4).")
        tk, ttk = self.tk, self.ttk
        cams = sorted(self.app.cameras, key=lambda k: (k.get("client_name") or "", k.get("name") or ""))
        labels = [f"{k.get('client_name') or self._client(k.get('client_id'))} · {k['name']}" for k in cams]
        if not labels:
            return self._say("info", "No hay cámaras cargadas.")
        top, frm = self.app._dialog("Probar con una cámara", 520)
        var = tk.StringVar(value=labels[0])
        ttk.Combobox(frm, textvariable=var, values=labels, state="readonly", width=50).grid(row=0, column=0,
                                                                                          columnspan=2, sticky="w")
        out: list = [None]

        def ok() -> None:
            out[0] = cams[labels.index(var.get())]
            top.destroy()
        bb = ttk.Frame(frm)
        bb.grid(row=1, column=0, columnspan=2, sticky="e", pady=(12, 0))
        ttk.Button(bb, text="Cancelar", command=top.destroy).pack(side="right")
        ttk.Button(bb, text="Probar", style="Primario.TButton", command=ok).pack(side="right", padx=6)
        top.grab_set()
        top.wait_window()
        if out[0] is not None:
            cam = out[0]
            self._test(v, lambda: self.camera_frame(cam["id"]), cam["name"])

    def _test(self, v: dict, get_frame, name: str) -> None:
        def work():
            img = get_frame()
            if img is None:
                raise RuntimeError("no se pudo leer la imagen")
            det, classes = self.version_detector(v)
            boxes = det.detect(img)
            return img, boxes, classes, det.device

        def done(res) -> None:
            img, boxes, classes, device = res
            self.show_test(v, name, img, boxes, classes, device)
        self.app.bg.run(work, done, self._fail)

    def show_test(self, v: dict, name: str, img: np.ndarray, boxes: list[dict], classes: list[dict],
                  device: str) -> None:
        from .oficina import ppm_photo
        top = self.tk.Toplevel(self.root)
        top.title(f"Versión {v['number']} · {name}")
        top.transient(self.root)
        frm = self.ttk.Frame(top, padding=10)
        frm.pack(fill="both", expand=True)
        photo = ppm_photo(self.tk, draw_detections(img, boxes, classes), 1000, 540)
        lbl = self.ttk.Label(frm, image=photo)
        lbl.image = photo  # (si no, Tk la borra)
        lbl.pack()
        legend = " · ".join(f"{i + 1} = {c['name']}" for i, c in enumerate(classes))
        self.ttk.Label(frm, text=f"{detections_summary(boxes, classes)}  ({legend}; corrió en esta PC, en {device})",
                       wraplength=1080, justify="left").pack(anchor="w", pady=(6, 0))
        self.ttk.Button(frm, text="Cerrar", command=top.destroy).pack(anchor="e", pady=(6, 0))
        self._test_top = top
