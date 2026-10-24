"""Ventana de la PC central: lista de clientes y cámaras a la izquierda, mosaico en vivo a la derecha.

Es la vista "de escritorio" para la oficina. A la izquierda están todos los clientes con sus
cámaras y el estado de cada una; se marcan las que se quieren mirar (una, varias, un cliente
entero) y el mosaico de la derecha muestra solo esas, a la velocidad del stream, con las personas
recuadradas, su número, los segundos en escena y la zona. La selección se guarda en disco.

Las cámaras en modo alarmas aparecen como "Escuchando alarmas" y solo muestran video mientras
alguien las mira desde el panel web.

Es también la "cara" del programa en Windows (el .exe no abre consola): pantalla de inicio de
sesión con usuario y clave de administrador, cabecera con el estado, y abajo el registro de lo que
va pasando. Cerrar la ventana cierra el programa (pide confirmación).

Se activa con `--ventana` (en el .exe viene activada; `--sin-ventana` la apaga, por ejemplo en un
servidor sin pantalla). Usa Tkinter (viene con Python) y muestra las imágenes como PPM, así no hace
falta Pillow.
"""

from __future__ import annotations

import json
import logging
import math
import queue
import socket
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("vigia.ventana")

TITULO = "Vigia - PC central"
TILE_W, TILE_H = 640, 360
ESTADOS = {"en_linea": "En línea", "sin_conexion": "Sin conexión", "reconectando": "Reconectando",
           "error": "Error", "iniciando": "Conectando", "detenida": "Detenida", "escuchando": "Escuchando alarmas"}


def tile(frame: np.ndarray | None, name: str, state: str) -> np.ndarray:
    """Una celda del mosaico: el cuadro (o un fondo oscuro con el estado) y el nombre arriba."""
    if frame is None:
        img = np.full((TILE_H, TILE_W, 3), (30, 30, 30), np.uint8)
        cv2.putText(img, state, (16, TILE_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (170, 170, 170), 1, cv2.LINE_AA)
    else:
        h, w = frame.shape[:2]
        scale = min(TILE_W / w, TILE_H / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        img = np.zeros((TILE_H, TILE_W, 3), np.uint8)
        y, x = (TILE_H - nh) // 2, (TILE_W - nw) // 2
        img[y:y + nh, x:x + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    cv2.rectangle(img, (0, 0), (TILE_W, 22), (0, 0, 0), -1)
    cv2.putText(img, name, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def mosaic(tiles: list[tuple[str, np.ndarray | None, str]]) -> np.ndarray:
    """Arma la grilla: 1 cámara = 1 celda grande; 2 = lado a lado; 3-4 = 2x2; 5-9 = 3x3..."""
    if not tiles:
        return tile(None, "Vigia", "Elegi a la izquierda las camaras para ver")
    n = len(tiles)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    grid = np.zeros((rows * TILE_H, cols * TILE_W, 3), np.uint8)
    for i, (name, frame, state) in enumerate(tiles):
        r, c = divmod(i, cols)
        grid[r * TILE_H:(r + 1) * TILE_H, c * TILE_W:(c + 1) * TILE_W] = tile(frame, name, state)
    return grid


def fit(img: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0) if max_w > 0 and max_h > 0 else 1.0
    if scale < 1.0:
        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return img


class LogHandler(logging.Handler):
    """Manda los mensajes del registro a la ventana (se muestran abajo, en lugar de la consola)."""

    def __init__(self):
        super().__init__(logging.INFO)
        self.q: queue.Queue[str] = queue.Queue(maxsize=2000)
        self.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        if record.name == "httpx":
            return  # los pedidos HTTP de rutina no le sirven al usuario
        try:
            self.q.put_nowait(self.format(record))
        except queue.Full:
            pass


def default_pc_name() -> str:
    try:
        return socket.gethostname() or "PC central"
    except OSError:
        return "PC central"


class Ventana:
    """Ventana Tkinter. `login` pide usuario y clave; `refresh` dibuja el mosaico; `pump` procesa
    los eventos. `selected_ids` dice qué cámaras hay que dibujar."""

    def __init__(self, state_file: Path | None = None):
        self.state_file = state_file
        self.failed = False
        self.open = False
        self.main_built = False
        self.on_logout: Callable[[], None] | None = None
        self.hooks: list[Callable[[], None]] = []  # se llaman en cada pump (resultados de red de la oficina)
        self.analyze = False
        self._selected: set[int] | None = self._load_selection()  # None = todavía no hay cámaras
        self._tree_ids: dict[str, int] = {}  # id del árbol -> id de cámara
        self._cam_rows: dict[int, str] = {}
        self._client_rows: dict[str, str] = {}
        self._known: list[tuple[int, str, str, str]] = []
        self._photo = None
        self.root = None
        self.log_handler = LogHandler()
        self._init_tk()

    # ---------------- Tkinter ----------------
    def _init_tk(self) -> None:
        try:
            import tkinter as tk
            from tkinter import messagebox, ttk
        except Exception as exc:  # Python sin Tk
            self.failed = True
            log.warning("No se puede abrir la ventana (%s). El agente sigue sin ella.", exc)
            return
        try:
            self.root = tk.Tk()
        except Exception as exc:  # sin pantalla (servidor)
            self.failed = True
            log.warning("No se puede abrir la ventana (%s). El agente sigue sin ella.", str(exc).splitlines()[0])
            return
        self.tk, self.ttk, self.messagebox = tk, ttk, messagebox
        self.root.title(TITULO)
        self.root.geometry("1280x760")
        self.root.minsize(900, 560)
        self.root.protocol("WM_DELETE_WINDOW", self._ask_close)
        try:
            ttk.Style().theme_use("vista" if "vista" in ttk.Style().theme_names() else "clam")
        except Exception:
            pass
        self.open = True
        logging.getLogger().addHandler(self.log_handler)

    def _ask_close(self) -> None:
        if not self.main_built or not getattr(self, "analyze", False) or self.messagebox.askyesno(
                TITULO, "¿Cerrar Vigía?\n\nMientras esté cerrado, esta PC deja de analizar las cámaras."):
            self.close()

    # ---------------- inicio de sesión ----------------
    def login(self, do_login: Callable[[str, str, str, bool], object], pc_name: str | None = None,
              error: str | None = None, analyze: bool = True) -> object | None:
        """Pantalla de inicio de sesión. `do_login(usuario, clave, nombre_pc, analiza)` devuelve algo
        verdadero o lanza ValueError con el mensaje a mostrar. Devuelve ese resultado, o None si cerraron."""
        if not self.open:
            return None
        tk, ttk = self.tk, self.ttk
        frame = ttk.Frame(self.root, padding=30)
        frame.place(relx=0.5, rely=0.45, anchor="center")
        ttk.Label(frame, text="Vigía", font=("Segoe UI", 26, "bold")).grid(row=0, column=0, columnspan=2, pady=(0, 2))
        ttk.Label(frame, text="Gestión de clientes, cámaras y alertas", foreground="#666").grid(row=1, column=0, columnspan=2, pady=(0, 18))
        user, pwd = tk.StringVar(), tk.StringVar()
        for i, (label, var, show) in enumerate((("Usuario", user, ""), ("Clave", pwd, "•")), start=2):
            ttk.Label(frame, text=label).grid(row=i, column=0, sticky="w", pady=5, padx=(0, 12))
            e = ttk.Entry(frame, textvariable=var, show=show, width=32)
            e.grid(row=i, column=1, pady=5)
            if i == 2:
                e.focus_set()
        msg = ttk.Label(frame, text=error or "Entrá con el usuario y la clave que te dio el administrador.",
                        foreground="#c0392b" if error else "#666", wraplength=380, justify="left")
        msg.grid(row=6, column=0, columnspan=2, pady=(10, 6), sticky="w")
        result: dict[str, object] = {"res": None}
        done = tk.BooleanVar(value=False)

        def submit(_e=None) -> None:
            if not user.get().strip() or not pwd.get():
                msg.configure(text="Completá usuario y clave.", foreground="#c0392b")
                return
            msg.configure(text="Conectando…", foreground="#666")
            self.root.update()
            try:
                result["res"] = do_login(user.get().strip(), pwd.get())
                done.set(True)
            except ValueError as exc:
                msg.configure(text=str(exc), foreground="#c0392b")

        ttk.Button(frame, text="Entrar", command=submit).grid(row=7, column=0, columnspan=2, pady=(8, 0), sticky="ew")
        self.root.bind("<Return>", submit)
        self.root.protocol("WM_DELETE_WINDOW", lambda: (self.close(), done.set(True)))
        try:
            self.root.wait_variable(done)
        except Exception:
            pass
        if self.open:
            self.root.unbind("<Return>")
            self.root.protocol("WM_DELETE_WINDOW", self._ask_close)
            frame.destroy()
        return result["res"]

    # ---------------- pantalla principal ----------------
    def build_main(self, title: str, analyze: bool = True) -> None:
        """Cabecera + pestañas. Con `analyze`, la primera pestaña es «En vivo» (mosaico de la IA)."""
        if not self.open or self.main_built:
            return
        tk, ttk = self.tk, self.ttk
        self.analyze = analyze
        top = ttk.Frame(self.root, padding=(10, 6))
        top.pack(side="top", fill="x")
        self.header = top  # la Oficina suma acá el botón de actualizar
        ttk.Label(top, text="Vigía", font=("Segoe UI", 14, "bold")).pack(side="left")
        self.status = ttk.Label(top, text=f"  ·  {title}", foreground="#666")
        self.status.pack(side="left")
        ttk.Button(top, text="Cerrar sesión", command=self._logout).pack(side="right")

        bottom = ttk.Frame(self.root, padding=(6, 0, 6, 6))
        bottom.pack(side="bottom", fill="x")
        ttk.Label(bottom, text="Registro", foreground="#666").pack(anchor="w")
        self.logbox = tk.Text(bottom, height=5, bg="#1b1f24", fg="#d0d6de", font=("Consolas", 9), relief="flat",
                              state="disabled", wrap="none")
        self.logbox.pack(fill="x")

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(side="top", fill="both", expand=True, padx=6)
        if analyze:
            body = ttk.Frame(self.notebook)
            self.notebook.add(body, text="  En vivo  ")
            left = ttk.Frame(body, padding=6)
            left.pack(side="left", fill="y")
            ttk.Label(left, text="Clientes y cámaras", font=("Segoe UI", 11, "bold")).pack(anchor="w")
            ttk.Label(left, text="Elegí qué ver (Ctrl o Shift para varias).\nUn cliente = todas sus cámaras.",
                      foreground="#666").pack(anchor="w", pady=(0, 6))
            self.tree = ttk.Treeview(left, columns=("estado",), show="tree headings", selectmode="extended", height=18)
            self.tree.heading("#0", text="Cámara")
            self.tree.heading("estado", text="Estado")
            self.tree.column("#0", width=210)
            self.tree.column("estado", width=150)
            self.tree.pack(fill="y", expand=True)
            self.tree.bind("<<TreeviewSelect>>", self._on_select)
            btns = ttk.Frame(left)
            btns.pack(fill="x", pady=6)
            ttk.Button(btns, text="Todas", command=self._select_all).pack(side="left")
            ttk.Button(btns, text="Ninguna", command=self._select_none).pack(side="left", padx=4)
            self.info = ttk.Label(left, text="", foreground="#666", wraplength=340, justify="left")
            self.info.pack(anchor="w", pady=(4, 0))
            right = tk.Frame(body, bg="#111")
            right.pack(side="right", fill="both", expand=True)
            self.canvas = tk.Label(right, bg="#111")
            self.canvas.pack(fill="both", expand=True)
            self.live_tab = body
        self.main_built = True

    def set_status(self, text: str, ok: bool = True) -> None:
        if self.main_built and self.open:
            self.status.configure(text=f"  ·  {text}", foreground="#2f8f46" if ok else "#c0392b")

    def _logout(self) -> None:
        extra = "\nLas cámaras dejan de analizarse hasta que vuelvas a entrar." if getattr(self, "analyze", False) else ""
        if self.messagebox.askyesno(TITULO, "¿Cerrar sesión y entrar con otro usuario?" + extra):
            if self.on_logout:
                self.on_logout()

    def _drain_log(self) -> None:
        if not self.main_built:
            return
        lines = []
        while True:
            try:
                lines.append(self.log_handler.q.get_nowait())
            except queue.Empty:
                break
        if not lines:
            return
        self.logbox.configure(state="normal")
        self.logbox.insert("end", "\n".join(lines) + "\n")
        n = int(self.logbox.index("end-1c").split(".")[0])
        if n > 300:
            self.logbox.delete("1.0", f"{n - 300}.0")
        self.logbox.see("end")
        self.logbox.configure(state="disabled")

    def _load_selection(self) -> set[int] | None:
        try:
            if self.state_file and self.state_file.exists():
                data = json.loads(self.state_file.read_text(encoding="utf-8"))
                return {int(x) for x in data.get("camaras", [])}
        except Exception:
            pass
        return None

    def _save_selection(self) -> None:
        if not self.state_file or self._selected is None:
            return
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(json.dumps({"camaras": sorted(self._selected)}), encoding="utf-8")
        except Exception as exc:
            log.debug("No se pudo guardar la selección de la ventana: %s", exc)

    # ---------------- selección ----------------
    def selected_ids(self) -> set[int]:
        return set(self._selected or ())

    def _on_select(self, _event=None) -> None:
        chosen: set[int] = set()
        for item in self.tree.selection():
            if item in self._tree_ids:
                chosen.add(self._tree_ids[item])
            else:  # un cliente: todas sus cámaras
                chosen.update(self._tree_ids[c] for c in self.tree.get_children(item) if c in self._tree_ids)
        self._selected = chosen
        self._save_selection()

    def _select_all(self) -> None:
        self.tree.selection_set(list(self._cam_rows.values()))

    def _select_none(self) -> None:
        self.tree.selection_set(())
        self._selected = set()
        self._save_selection()

    def set_cameras(self, cams: list[tuple[int, str, str, str]]) -> None:
        """cams: (id, cliente, nombre, estado). Rearma el árbol si cambió la lista; actualiza estados."""
        if not self.open or not self.main_built or not self.analyze:
            return
        key = [(c[0], c[1], c[2]) for c in cams]
        if key != [(c[0], c[1], c[2]) for c in self._known]:
            self.tree.delete(*self.tree.get_children())
            self._tree_ids, self._cam_rows, self._client_rows = {}, {}, {}
            for cid, client, name, state in cams:
                if client not in self._client_rows:
                    self._client_rows[client] = self.tree.insert("", "end", text=client or "Sin cliente", open=True)
                row = self.tree.insert(self._client_rows[client], "end", text=name, values=(state,))
                self._tree_ids[row] = cid
                self._cam_rows[cid] = row
            if self._selected is None:  # primera vez: todas (si son pocas) o el primer cliente
                ids = [c[0] for c in cams]
                first = cams[0][1] if cams else None
                self._selected = set(ids) if len(ids) <= 4 else {c[0] for c in cams if c[1] == first}
            rows = [self._cam_rows[cid] for cid in self._selected if cid in self._cam_rows]
            self.tree.selection_set(rows)
        else:
            for cid, _client, _name, state in cams:
                row = self._cam_rows.get(cid)
                if row is not None and self.tree.set(row, "estado") != state:
                    self.tree.set(row, "estado", state)
        self._known = cams

    # ---------------- dibujo ----------------
    def refresh(self, tiles: list[tuple[str, np.ndarray | None, str]], note: str = "") -> None:
        if not self.open or not self.main_built or not self.analyze:
            return
        if self.notebook.select() != str(self.live_tab):
            return  # no se dibuja el mosaico si no se está mirando esa pestaña
        try:
            img = mosaic(tiles)
            w, h = max(self.canvas.winfo_width(), 1), max(self.canvas.winfo_height(), 1)
            if w > 1 and h > 1:
                img = fit(img, w, h)
            ok, ppm = cv2.imencode(".ppm", img)
            if ok:
                self._photo = self.tk.PhotoImage(data=ppm.tobytes())
                self.canvas.configure(image=self._photo)
            if note:
                self.info.configure(text=note)
        except Exception as exc:
            if self.open:
                log.debug("Ventana: no se pudo dibujar (%s)", exc)

    def pump(self, ms: int = 40) -> bool:
        """Procesa los eventos de la ventana durante `ms` milisegundos. False si se cerró."""
        if not self.open:
            return False
        try:
            self._drain_log()
            for hook in self.hooks:
                hook()
            self.root.update_idletasks()
            self.root.update()
            self.root.after(ms)  # sin callback: espera `ms` (como un sleep que no congela la ventana)
            return self.open
        except Exception:
            self.open = False
            return False

    def close(self) -> None:
        logging.getLogger().removeHandler(self.log_handler)
        if self.open and self.root is not None:
            try:
                self.root.destroy()
            except Exception:
                pass
        self.open = False
