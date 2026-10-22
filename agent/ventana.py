"""Ventana de la PC central: lista de clientes y cámaras a la izquierda, mosaico en vivo a la derecha.

Es la vista "de escritorio" para la oficina. A la izquierda están todos los clientes con sus
cámaras y el estado de cada una; se marcan las que se quieren mirar (una, varias, un cliente
entero) y el mosaico de la derecha muestra solo esas, a la velocidad del stream, con las personas
recuadradas, su número, los segundos en escena y la zona. La selección se guarda en disco.

Las cámaras en modo alarmas aparecen como "Escuchando alarmas" y solo muestran video mientras
alguien las mira desde el panel web.

Se activa con `--ventana` (en el .exe viene activada; `--sin-ventana` la apaga, por ejemplo en un
servidor sin pantalla). Cerrar la ventana no detiene el agente. Usa Tkinter (viene con Python) y
muestra las imágenes como PPM, así no hace falta Pillow.
"""

from __future__ import annotations

import json
import logging
import math
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
        return tile(None, "Vigia", "Elegi a la izquierda que camaras ver")
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


class Ventana:
    """Ventana Tkinter. `refresh` recibe todas las cámaras con su estado y el último cuadro de las
    elegidas; `pump` procesa los eventos. `selected_ids` dice qué cámaras hay que dibujar."""

    def __init__(self, state_file: Path | None = None):
        self.state_file = state_file
        self.failed = False
        self.open = False
        self._selected: set[int] | None = self._load_selection()  # None = todavía no hay cámaras
        self._tree_ids: dict[str, int] = {}  # id del árbol -> id de cámara
        self._cam_rows: dict[int, str] = {}
        self._client_rows: dict[str, str] = {}
        self._known: list[tuple[int, str, str, str]] = []
        self._photo = None
        self.root = None
        self._init_tk()

    # ---------------- Tkinter ----------------
    def _init_tk(self) -> None:
        try:
            import tkinter as tk
            from tkinter import ttk
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
        self.tk, self.ttk = tk, ttk
        self.root.title(TITULO)
        self.root.geometry("1280x720")
        self.root.minsize(900, 500)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.open = True

        left = ttk.Frame(self.root, padding=6)
        left.pack(side="left", fill="y")
        ttk.Label(left, text="Clientes y cámaras", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Label(left, text="Marcá las que querés ver (Ctrl o Shift para varias).\nUn cliente = todas sus cámaras.",
                  foreground="#666").pack(anchor="w", pady=(0, 6))
        self.tree = ttk.Treeview(left, columns=("estado",), show="tree headings", selectmode="extended", height=25)
        self.tree.heading("#0", text="Cámara")
        self.tree.heading("estado", text="Estado")
        self.tree.column("#0", width=230)
        self.tree.column("estado", width=160)
        self.tree.pack(fill="y", expand=True)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        btns = ttk.Frame(left)
        btns.pack(fill="x", pady=6)
        ttk.Button(btns, text="Todas", command=self._select_all).pack(side="left")
        ttk.Button(btns, text="Ninguna", command=self._select_none).pack(side="left", padx=4)
        self.info = ttk.Label(left, text="", foreground="#666", wraplength=340, justify="left")
        self.info.pack(anchor="w", pady=(4, 0))

        right = tk.Frame(self.root, bg="#111")
        right.pack(side="right", fill="both", expand=True)
        self.canvas = tk.Label(right, bg="#111")
        self.canvas.pack(fill="both", expand=True)

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
        if not self.open:
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
        if not self.open:
            return
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
            self.root.update_idletasks()
            self.root.update()
            self.root.after(ms)  # sin callback: espera `ms` (como un sleep que no congela la ventana)
            return self.open
        except Exception:
            self.open = False
            return False

    def close(self) -> None:
        if self.open and self.root is not None:
            try:
                self.root.destroy()
            except Exception:
                pass
        self.open = False
