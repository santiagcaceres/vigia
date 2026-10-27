"""Gestión de la empresa desde la app de escritorio: clientes, cámaras, zonas, alertas, usuarios y PCs.

La web queda para los clientes (su resumen y sus alertas). Todo lo de la empresa se hace acá, contra
la misma API del servidor, con la sesión de un usuario (admin o empleado). Las llamadas a la red
corren en hilos aparte (`Bg`) y el resultado se aplica en el hilo de la ventana, así nada se congela.

Pestañas (las arma `Oficina.build` dentro del `ttk.Notebook` de la ventana; «En vivo», para mirar cualquier cámara
sin analizarla, la arma `Ventana`):
- Clientes: alta/edición/borrado de clientes y cámaras, «Nombres e imágenes…» (renombrar viendo la imagen, con lo que
  ve la IA: `NamesWindow`), qué cámaras se analizan («✔ Analizar», en las máquinas
  de análisis), probar conexión, zonas y reglas, métricas del día. «Ajustar…» pone a punto una cámara analizándola
  en esta PC (en vivo o con videos grabados) antes de aprobarla (`TuneWindow`, motor en agent/ajuste.py).
- Alertas: lista con filtros, foto, reproducción del clip dentro de la app, etiquetas (teclas 1-4) y
  "Avisar al cliente".
- Administración (solo admin): usuarios y PCs centrales.
"""

from __future__ import annotations

import logging
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np

from core.schedule import DAY_NAMES, DAYS, open_hours_per_week

from . import actualizar
from .version import VERSION

log = logging.getLogger("vigia.oficina")

PLAN_NAMES = {"hurto": "Detección de hurtos", "comercio": "Métricas de comercio", "estacion": "Estación de servicio"}
ROLE_NAMES = {"admin": "Administrador", "entrenador": "Empleado", "cliente": "Cliente"}
# qué cámaras analizan las máquinas (server/pool.py): las elegidas en «Clientes» o todas
ANALYSIS_MODES = {"pcs": "Solo las elegidas en «Clientes»",
                  "servidores": "Todas las cámaras"}
# por cliente: qué cámaras se analizan («general» sigue el ajuste de Administración). Lo que se marque en cada
# cámara («✔ Analizar» / «✘ No analizar») manda sobre esto.
LOCATION_NAMES = {"general": "Como el ajuste general", "pcs": "Solo las cámaras marcadas «Analizar»",
                  "servidores": "Todas sus cámaras"}
AUTO_REFRESH_MS = 30000  # estado de las cámaras en «Clientes» (solo con la pestaña a la vista)
MODE_NAMES = {"vivo": "En vivo", "alarmas_equipo": "Alarmas del propio grabador", "alarmas": "Alarmas por Hik-Partner Pro"}
BRAND_NAMES = {"hikconnect": "Hik-Connect", "ip": "IP", "hikvision": "Hikvision", "dahua": "Dahua"}
# tipo de cámara: decide qué reglas se aplican (el servidor tiene la lista, CAMERA_PROFILES)
PROFILE_NAMES = {"general": "General (todas las reglas)", "gondola": "Góndola / pasillo", "caja": "Caja",
                 "entrada": "Entrada / puerta", "deposito": "Depósito / trastienda", "exterior": "Calle / exterior",
                 "surtidor": "Surtidor (estación)", "metricas": "Solo métricas (conteo y tiempos, sin hurto)"}
# qué hace la cámara con el local cerrado (según el horario del cliente)
OFF_HOURS_NAMES = {"apagada": "No mirar (ahorra todo: video, internet y análisis)",
                   "alarmas": "Escuchar solo las alarmas del grabador (por IP y puertos)",
                   "vivo": "Seguir analizando igual (24 h, gasta todo)"}
# formulario de cámara: Hikvision por la nube, o cualquier marca por IP y puertos (la marca se detecta sola)
CONN_NAMES = {"hikconnect": "Hikvision por Hik-Connect (recomendado, sin puertos)",
              "ip": "Por IP y puertos (Hikvision o Dahua: la marca se detecta sola)"}
TYPE_NAMES = {"salida_sin_pago": "Salida sin pago", "merodeo": "Merodeo", "zona_restringida": "Zona restringida",
              "fuera_de_horario": "Fuera de horario", "alarma_dvr": "Alarma del grabador"}
LABEL_NAMES = {"hurto": "Hurto", "sospechoso": "Sospechoso", "normal": "Normal", "falso_positivo": "Falso positivo"}
LABEL_KEYS = {"1": "hurto", "2": "sospechoso", "3": "normal", "4": "falso_positivo"}
ZONE_NAMES = {"shelf": "Góndola", "checkout": "Caja", "exit": "Salida", "restricted": "Restringida",
              "entrance": "Entrada (conteo)", "pump": "Surtidor", "shop": "Tienda"}
ZONE_COLORS = {"shelf": "#ffc800", "checkout": "#00c800", "exit": "#0096ff", "restricted": "#ff3030",
               "entrance": "#c800ff", "pump": "#ff7800", "shop": "#00c8c8"}
ALARM_TYPES = {"VMD": "Movimiento", "fielddetection": "Intrusión", "linedetection": "Cruce de línea",
               "regionEntrance": "Entrada a zona", "regionExiting": "Salida de zona"}
TUNING_NAMES = {"sin_ajustar": "—", "en_ajuste": "En ajuste", "aprobada": "✔ Aprobada"}
STATE_NAMES = {"escuchando": "Escuchando alarmas", "en_linea": "En línea", "sin_conexion": "Sin conexión",
               "reconectando": "Reconectando", "error": "Error", "iniciando": "Iniciando", "detenida": "Detenida",
               "fuera_de_horario": "Local cerrado (no se mira)"}


def fmt_counts(d: dict | None) -> str:
    """«hoy · 7 días · sin revisar» de alertas."""
    if not d:
        return "0 · 0 · 0"
    return f"{d.get('today', 0)} · {d.get('week', 0)} · {d.get('unreviewed', 0)}"


def fmt_date(iso: str | None) -> str:
    if not iso:
        return "—"
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.astimezone().strftime("%d/%m %H:%M")
    except ValueError:
        return iso


def fmt_dur(s: float) -> str:
    s = round(s or 0)
    if s >= 3600:
        return f"{s // 3600} h {round((s % 3600) / 60)} min"
    return f"{round(s / 60)} min" if s >= 90 else f"{s} s"


# ------------------------------------------------------------------ API
def make_password(n: int = 10) -> str:
    """Clave fácil de dictar: sin letras que se confunden (l/1, O/0)."""
    import secrets
    abc = "abcdefghjkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(secrets.choice(abc) for _ in range(n))


def slug_user(name: str) -> str:
    """Usuario sugerido a partir del nombre del cliente: «Súper Minas» → «superminas»"""
    import unicodedata
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    return "".join(ch for ch in s if ch.isalnum())[:20]


class ApiError(Exception):
    def __init__(self, msg: str, status: int = 0):
        super().__init__(msg)
        self.status = status


class Api:
    """Cliente de la API con la sesión de un usuario (cabecera Authorization: Bearer)."""

    def __init__(self, server: str, token: str | None = None, verify: bool | str = True):
        self.server = server.rstrip("/")
        self.http = httpx.Client(base_url=self.server, timeout=40.0, verify=verify, follow_redirects=True)
        self.token = token
        self.user: dict | None = None

    def set_token(self, token: str | None) -> None:
        self.token = token
        if token:
            self.http.headers["Authorization"] = f"Bearer {token}"
        else:
            self.http.headers.pop("Authorization", None)

    def _check(self, r: httpx.Response) -> Any:
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail")
                if isinstance(detail, list):  # error de validación de FastAPI
                    detail = "; ".join(f"{'.'.join(str(x) for x in d.get('loc', [])[1:])}: {d.get('msg')}" for d in detail)
            except ValueError:
                detail = r.text[:200]
            raise ApiError(str(detail or f"Error {r.status_code}"), r.status_code)
        if r.status_code == 204 or not r.content:
            return None
        ctype = r.headers.get("content-type", "")
        return r.json() if "json" in ctype else r.content

    def call(self, method: str, path: str, **kw) -> Any:
        try:
            return self._check(self.http.request(method, path, **kw))
        except httpx.HTTPError as exc:
            raise ApiError(f"Sin conexión con el servidor ({type(exc).__name__})") from exc

    def get(self, path: str, **params) -> Any:
        return self.call("GET", path, params={k: v for k, v in params.items() if v is not None})

    def post(self, path: str, body: Any = None) -> Any:
        return self.call("POST", path, json=body if body is not None else {})

    def put(self, path: str, body: Any) -> Any:
        return self.call("PUT", path, json=body)

    def patch(self, path: str, body: Any) -> Any:
        return self.call("PATCH", path, json=body)

    def delete(self, path: str) -> Any:
        return self.call("DELETE", path)

    def download(self, path: str) -> bytes:
        """Baja un archivo de la API (sigue la redirección a la URL firmada del almacenamiento)."""
        r = self.http.get(path)
        if r.status_code >= 400:
            raise ApiError(f"No se pudo bajar el archivo ({r.status_code})", r.status_code)
        return r.content

    def login(self, username: str, password: str) -> dict:
        """POST /api/login → guarda el token de sesión. Devuelve el usuario."""
        res = self.call("POST", "/api/login", json={"username": username, "password": password})
        self.set_token(res["token"])
        self.user = res["user"]
        return res

    def me(self) -> dict:
        self.user = self.get("/api/me")
        return self.user


class Bg:
    """Corre funciones en hilos y entrega el resultado en el hilo de la ventana (`drain`)."""

    def __init__(self):
        self.q: queue.Queue = queue.Queue()

    def run(self, fn: Callable[[], Any], done: Callable[[Any], None] | None = None,
            fail: Callable[[Exception], None] | None = None) -> None:
        def work() -> None:
            try:
                res = fn()
            except Exception as exc:  # noqa: BLE001 - se informa en la ventana
                self.q.put((fail, exc))
            else:
                self.q.put((done, res))
        threading.Thread(target=work, daemon=True).start()

    def drain(self) -> None:
        while True:
            try:
                cb, arg = self.q.get_nowait()
            except queue.Empty:
                return
            if cb is not None:
                try:
                    cb(arg)
                except Exception:
                    log.exception("Error en la ventana")


def ppm_photo(tk, img: np.ndarray, max_w: int, max_h: int):
    """ndarray BGR → PhotoImage (vía PPM, sin Pillow), achicado para entrar en max_w x max_h."""
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    ok, ppm = cv2.imencode(".ppm", img)
    return tk.PhotoImage(data=ppm.tobytes()) if ok else None


def open_file(path: str) -> None:
    if sys.platform.startswith("win"):
        os.startfile(path)  # type: ignore[attr-defined]
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", path])


# ------------------------------------------------------------------ app
class Oficina:
    def __init__(self, ventana, api: Api, on_logout: Callable[[], None] | None = None,
                 local_probe: Callable[[int], dict | None] | None = None,
                 on_config_saved: Callable[[], None] | None = None,
                 on_restart: Callable[[], None] | None = None,
                 agent: Callable[[], Any] | None = None):
        self.local_probe = local_probe
        self.agent_of = agent  # el analizador de esta PC (para «Ajustar»: analizar una cámara acá)
        self.on_restart = on_restart
        self.latest: dict | None = None  # última versión publicada (para el botón de actualizar)
        self.on_config_saved = on_config_saved
        self.v = ventana
        self.tk, self.ttk, self.mb = ventana.tk, ventana.ttk, ventana.messagebox
        self.api = api
        self.bg = Bg()
        self.on_logout = on_logout
        self.clients: list[dict] = []
        self.cameras: list[dict] = []
        self.agents: list[dict] = []
        self.is_admin = (api.user or {}).get("role") == "admin"

    # ---------- utilidades ----------
    def err(self, exc: Exception) -> None:
        self.mb.showerror("Vigía", str(exc))

    def client_name(self, cid: int | None) -> str:
        return next((c["name"] for c in self.clients if c["id"] == cid), "—")

    def build(self, notebook) -> None:
        ttk = self.ttk
        self.notebook = notebook
        self.tab_clients = ttk.Frame(notebook, padding=8)
        self.tab_alerts = ttk.Frame(notebook, padding=8)
        notebook.add(self.tab_clients, text="  Clientes  ")
        notebook.add(self.tab_alerts, text="  Alertas  ")
        self._build_clients()
        self._build_alerts()
        if self.is_admin:
            self.tab_admin = ttk.Frame(notebook, padding=8)
            notebook.add(self.tab_admin, text="  Administración  ")
            self._build_admin()
        self.refresh_all()
        self.v.root.after(AUTO_REFRESH_MS, self._auto_refresh)
        header = getattr(self.v, "header", None)
        if header is not None:
            self.upd_btn = ttk.Button(header, text="Buscar actualización", command=self.update_dialog)
            self.upd_btn.pack(side="right", padx=6)
            ttk.Label(header, text=f"versión {VERSION}", foreground="#888").pack(side="right", padx=6)
            self.v.root.after(4000, self._check_update)

    # ---------- actualización de la app ----------
    def _check_update(self) -> None:
        """Al abrir: si hay versión nueva, el botón lo dice (no interrumpe nada)."""
        def done(info):
            self.latest = info
            if actualizar.is_newer(info.get("version")):
                self.upd_btn.configure(text=f"★ Actualizar a {info['version']}")
                log.info("Hay una versión nueva de la app: %s (esta es %s). Botón «Actualizar» arriba a la derecha.",
                         info["version"], VERSION)
        self.bg.run(lambda: self.api.get("/api/app/update"), done, lambda e: log.debug("Sin datos de versión: %s", e))

    def update_dialog(self) -> None:
        """«Buscar actualización»: el botón dice «Buscando…» y siempre termina en un mensaje (también si el servidor
        no contesta en 30 s), y la respuesta queda en el registro."""
        if getattr(self, "_checking_update", False):
            return
        self._checking_update = True
        self.upd_btn.configure(text="Buscando…", state="disabled")
        log.info("Buscando actualización…")

        def finish(newer: str | None = None) -> None:
            self._checking_update = False
            self.upd_btn.configure(state="normal", text=f"★ Actualizar a {newer}" if newer else "Buscar actualización")

        def fail(exc: Exception) -> None:
            finish()
            log.warning("No se pudo buscar la actualización: %s", exc)
            self.mb.showerror("Vigía", f"No se pudo buscar la actualización: {exc}")

        def timeout() -> None:
            if self._checking_update:
                finish()
                log.warning("Buscar actualización: el servidor no contestó en 30 s")
                self.mb.showwarning("Vigía", "El servidor no contestó en 30 segundos. Probá de nuevo en un rato, o bajá "
                                             "VigiaAgente-windows.zip de GitHub (Releases) y reemplazá el .exe a mano.")
        self.v.root.after(30000, timeout)

        def done(info):
            self.latest = info
            new = info.get("version")
            newer = actualizar.is_newer(new)
            finish(new if newer else None)
            log.info("Última versión publicada: %s (esta es %s)", new or "?", VERSION)
            if not newer:
                return self.mb.showinfo("Vigía", f"Ya tenés la última versión ({VERSION}).")
            if not actualizar.can_self_update():
                return self.mb.showinfo("Vigía", f"Hay una versión nueva ({new}). Esta copia corre con Python: "
                                                 "actualizá con «git pull» y volvé a abrir.")
            size = f" ({info['size'] / 1e6:.0f} MB)" if info.get("size") else ""
            extra = ("\n\nLas cámaras dejan de analizarse unos segundos mientras se reabre."
                     if getattr(self.v, "analyze", False) else "")
            if self.mb.askyesno("Actualizar Vigía", f"Versión instalada: {VERSION}\nVersión nueva: {new}{size}\n\n"
                                "Se descarga, se reemplaza el programa y se vuelve a abrir solo, con la misma sesión."
                                + extra + "\n\n¿Actualizar ahora?"):
                self._install_update(new)
        self.bg.run(lambda: self.api.get("/api/app/update"), done, fail)

    def _install_update(self, new: str) -> None:
        tk, ttk = self.tk, self.ttk
        top, frm = self._dialog("Actualizando Vigía", width=420)
        top.protocol("WM_DELETE_WINDOW", lambda: None)  # no se cierra a mitad de camino
        msg = ttk.Label(frm, text=f"Descargando la versión {new}…")
        msg.grid(row=0, column=0, columnspan=2, sticky="w")
        bar = ttk.Progressbar(frm, length=380, mode="determinate", maximum=100)
        bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=10)

        def progress(arg):
            got, total = arg
            if total:
                bar["value"] = 100 * got / total
                msg.configure(text=f"Descargando la versión {new}… {got / 1e6:.0f} de {total / 1e6:.0f} MB")

        def work():
            url = self.api.get("/api/app/update", download=1)["url"]
            return actualizar.install(url, progress=lambda g, tot: self.bg.q.put((progress, (g, tot))))

        def done(_):
            msg.configure(text="Listo. Abriendo la versión nueva…")
            bar["value"] = 100
            self.v.root.after(1200, self.on_restart or top.destroy)

        def fail(exc):
            top.destroy()
            self.mb.showerror("Vigía", f"No se pudo actualizar: {exc}\n\nEl programa sigue igual. También podés bajar "
                                       "VigiaAgente-windows.zip de GitHub (Releases) y reemplazar el .exe a mano.")
        self.bg.run(work, done, fail)

    def _auto_refresh(self) -> None:
        """Estado de cámaras cada 30 s (sin tocar lo que se está editando). Solo con la pestaña «Clientes y
        cámaras» a la vista y la ventana abierta: si nadie la mira, no se gastan pedidos al servidor."""
        if not self.v.open:
            return
        try:
            visible = (self.v.root.state() != "iconic"
                       and self.notebook.select() == str(self.tab_clients))
        except Exception:  # noqa: BLE001
            visible = True
        if not visible:
            self.v.root.after(AUTO_REFRESH_MS, self._auto_refresh)
            return

        def done(res):
            self.cameras, self.agents, self.alert_counts = res
            self._fill_clients(keep_info=True)
        self.bg.run(lambda: (self.api.get("/api/cameras"), self.api.get("/api/agents"),
                             self.api.get("/api/stats/alerts")), done, lambda e: None)
        self.v.root.after(AUTO_REFRESH_MS, self._auto_refresh)

    def config_saved(self) -> None:
        """Se guardó una cámara, zonas o reglas: refrescar la lista y avisarle a la PC central."""
        self.refresh_all()
        if self.on_config_saved:
            self.on_config_saved()

    def refresh_all(self) -> None:
        def load():
            return (self.api.get("/api/clients"), self.api.get("/api/cameras"),
                    self.api.get("/api/agents"), self.api.get("/api/stats/alerts"))

        def done(res):
            self.clients, self.cameras, self.agents, self.alert_counts = res
            self._fill_clients()
            self._fill_alert_filters()
            if self.is_admin:  # después de los clientes: muestran sus nombres (antes salía «—» si llegaban primero)
                self.load_users()
                self.load_machines()
        self.bg.run(load, done, self._net_fail)
        self.load_alerts()

    def _net_fail(self, exc: Exception) -> None:
        if isinstance(exc, ApiError) and exc.status == 401:
            self.mb.showwarning("Vigía", "La sesión venció. Entrá de nuevo.")
            if self.on_logout:
                self.on_logout()
            return
        log.warning("%s", exc)

    # ================================================================ clientes y cámaras
    def _build_clients(self) -> None:
        tk, ttk, f = self.tk, self.ttk, self.tab_clients
        left = ttk.Frame(f)
        left.pack(side="left", fill="y")
        ttk.Label(left, text="Clientes", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.cl_tree = ttk.Treeview(left, columns=("planes", "alertas"), show="tree headings", height=20,
                                    selectmode="browse")
        self.cl_tree.heading("#0", text="Cliente")
        self.cl_tree.heading("planes", text="Planes")
        self.cl_tree.heading("alertas", text="Alertas")
        self.cl_tree.column("#0", width=170)
        self.cl_tree.column("planes", width=110)
        self.cl_tree.column("alertas", width=100, anchor="center")
        self.cl_tree.pack(fill="y", expand=True)
        ttk.Label(left, text="Alertas: hoy · últimos 7 días · sin revisar", foreground="#666").pack(anchor="w")
        self.cl_tree.bind("<<TreeviewSelect>>", lambda e: self._fill_cameras())
        b = ttk.Frame(left)
        b.pack(fill="x", pady=6)
        ttk.Button(b, text="Nuevo cliente", command=lambda: self.client_dialog()).pack(side="left")
        ttk.Button(b, text="Editar", command=lambda: self.client_dialog(self._sel_client())).pack(side="left", padx=4)
        if self.is_admin:
            ttk.Button(b, text="Eliminar", command=self.delete_client).pack(side="left")
        ttk.Button(b, text="Actualizar", command=self.refresh_all).pack(side="right")

        right = ttk.Frame(f, padding=(12, 0, 0, 0))
        right.pack(side="left", fill="both", expand=True)
        self.cam_title = ttk.Label(right, text="Elegí un cliente", font=("Segoe UI", 11, "bold"))
        self.cam_title.pack(anchor="w")
        self.cam_tree = ttk.Treeview(right, columns=("nombre", "estado", "modo", "conexion", "ajuste", "analiza",
                                                     "alertas", "zonas"), show="headings", height=12, selectmode="extended")
        for col, txt, w in (("nombre", "Cámara", 120), ("estado", "Estado", 115), ("modo", "Modo", 95),
                            ("conexion", "Grabador", 125), ("ajuste", "Ajuste", 80), ("analiza", "Analizar", 105),
                            ("alertas", "Alertas", 80), ("zonas", "Zonas", 45)):
            self.cam_tree.heading(col, text=txt)
            self.cam_tree.column(col, width=w)
        self.cam_tree.pack(fill="x")
        self.cam_tree.bind("<Double-1>", lambda e: self.camera_dialog(self._sel_camera()))
        b2 = ttk.Frame(right)
        b2.pack(fill="x", pady=6)
        ttk.Button(b2, text="Agregar cámara", command=lambda: self.camera_dialog()).pack(side="left")
        ttk.Button(b2, text="Editar", command=lambda: self.camera_dialog(self._sel_camera())).pack(side="left", padx=4)
        ttk.Button(b2, text="Ajustar…", command=lambda: self.tune_dialog(self._sel_camera())).pack(side="left")
        ttk.Button(b2, text="Zonas y reglas", command=lambda: self.zones_dialog(self._sel_camera())).pack(side="left",
                                                                                                         padx=(4, 0))
        ttk.Button(b2, text="Probar conexión", command=lambda: self.probe(self._sel_camera())).pack(side="left", padx=4)
        if self.is_admin:
            ttk.Button(b2, text="Eliminar", command=self.delete_camera).pack(side="left")
        # qué se analiza (en las máquinas de análisis): se eligen varias con Ctrl o Shift
        b3 = ttk.Frame(right)
        b3.pack(fill="x")
        ttk.Button(b3, text="✔ Analizar", command=lambda: self.set_analyze(True)).pack(side="left")
        ttk.Button(b3, text="✘ No analizar", command=lambda: self.set_analyze(False)).pack(side="left", padx=4)
        ttk.Button(b3, text="Elegir todas", command=lambda: self.cam_tree.selection_set(
            self.cam_tree.get_children())).pack(side="left")
        ttk.Button(b3, text="Nombres e imágenes…", command=self.names_dialog).pack(side="left", padx=4)
        ttk.Button(b3, text="Exportar métricas…", command=self.export_dialog).pack(side="right")
        ttk.Button(b3, text="Métricas", command=self.metrics_dialog).pack(side="right", padx=4)
        self.cam_info = tk.Text(right, height=10, wrap="word", relief="flat", bg="#f6f7f9", font=("Segoe UI", 10))
        self.cam_info.pack(fill="both", expand=True, pady=(6, 0))
        self._set_info("Elegí una cámara para ver el detalle. Doble clic para editarla.")

    def _set_info(self, text: str) -> None:
        self.cam_info.configure(state="normal")
        self.cam_info.delete("1.0", "end")
        self.cam_info.insert("1.0", text)
        self.cam_info.configure(state="disabled")

    def _sel_client(self) -> dict | None:
        sel = self.cl_tree.selection()
        return next((c for c in self.clients if str(c["id"]) == (sel[0] if sel else "")), None)

    def _sel_camera(self) -> dict | None:
        sel = self.cam_tree.selection()
        return next((c for c in self.cameras if str(c["id"]) == (sel[0] if sel else "")), None)

    def _fill_clients(self, keep_info: bool = False) -> None:
        prev = self.cl_tree.selection()
        self.cl_tree.delete(*self.cl_tree.get_children())
        counts = (getattr(self, "alert_counts", None) or {}).get("clients") or {}
        for c in self.clients:
            n = sum(1 for x in self.cameras if x["client_id"] == c["id"])
            self.cl_tree.insert("", "end", iid=str(c["id"]), text=f"{c['name']}  ({n})",
                                values=(", ".join(PLAN_NAMES.get(p, p).split()[0] for p in c.get("plans") or []),
                                        fmt_counts(counts.get(str(c["id"])))))
        if prev and self.cl_tree.exists(prev[0]):
            self.cl_tree.selection_set(prev)
        elif self.clients:
            self.cl_tree.selection_set(str(self.clients[0]["id"]))
        self._fill_cameras(keep_info)

    def _cam_state(self, c: dict) -> str:
        st = c.get("status")
        if not c.get("enabled", True):
            return "Deshabilitada"
        if not st:  # nadie la analiza (en «En vivo» se puede mirar igual)
            if not c.get("analyzed"):
                return "Sin analizar"
            return "Esperando máquina" if not c.get("agent_id") else "Sin datos de la máquina"
        if not st.get("agent_online"):
            return "PC desconectada"
        txt = STATE_NAMES.get(st.get("state"), st.get("state") or "—")
        if st.get("fps") and st.get("state") == "en_linea":
            txt += f" · {st['fps']:.0f} fps"
            if st.get("idle_pct"):
                txt += f" · {st['idle_pct']}% quieta"
        return txt + (" · solo métricas" if st.get("metrics_only") else "")

    def _fill_cameras(self, keep_info: bool = False) -> None:
        c = self._sel_client()
        prev_cam = self.cam_tree.selection()
        self.cam_tree.delete(*self.cam_tree.get_children())
        if not c:
            self.cam_title.configure(text="Elegí un cliente")
            return
        web = f"   ·   acceso web: {c['web_username']}" if c.get("web_username") else "   ·   sin acceso web (Editar para crearlo)"
        self.cam_title.configure(text=f"Cámaras de {c['name']}{web}")
        counts = (getattr(self, "alert_counts", None) or {}).get("cameras") or {}
        for cam in [x for x in self.cameras if x["client_id"] == c["id"]]:
            conn = (f"Hik-Connect · serie {cam.get('hpp_serial') or '—'} · canal {cam.get('channel')}"
                    if cam.get("brand") == "hikconnect" else
                    f"{BRAND_NAMES.get(cam.get('brand'), '')} · {cam.get('host', '')}:{cam.get('rtsp_port')}/{cam.get('http_port')} · canal {cam.get('channel')}")
            self.cam_tree.insert("", "end", iid=str(cam["id"]), values=(
                cam["name"], self._cam_state(cam),
                f"{MODE_NAMES.get(cam.get('mode'), cam.get('mode'))} · {PROFILE_NAMES.get(cam.get('profile') or 'general', '').split(' (')[0]}", conn,
                TUNING_NAMES.get(cam.get("tuning") or "sin_ajustar", "—"),
                ("Sí · " + (cam.get("analyzer_name") or "esperando máquina")) if cam.get("analyzed") else "No",
                fmt_counts(counts.get(str(cam["id"]))), len(cam.get("zones") or [])))
        self.cam_tree.bind("<<TreeviewSelect>>", lambda e: self._show_camera())
        if prev_cam and self.cam_tree.exists(prev_cam[0]):
            self.cam_tree.selection_set(prev_cam)  # sigue elegida (el detalle se actualiza solo si no hay otra cosa)
        elif not keep_info:
            self._set_info("Elegí una cámara para ver el detalle. Doble clic para editarla.")

    def _show_camera(self) -> None:
        cam = self._sel_camera()
        if not cam:
            return
        if getattr(self, "_info_locked", None) == cam["id"]:
            return  # se está mostrando el resultado de una prueba o las métricas: no pisarlo
        self._info_locked = None
        st = cam.get("status") or {}
        zones = ", ".join(f"{z['name']} ({ZONE_NAMES.get(z['type'], z['type'])})" for z in cam.get("zones") or []) or \
            "ninguna: sin zonas no se disparan reglas ni hay métricas por zona"
        self._set_info(
            f"{cam['name']}\n"
            f"Estado: {self._cam_state(cam)}{(' · ' + st['error']) if st.get('error') else ''}\n"
            f"Modo: {MODE_NAMES.get(cam.get('mode'), cam.get('mode'))}\n"
            f"Analizar: {('sí · ' + (cam.get('analyzer_name') or 'esperando una máquina en línea')) if cam.get('analyzed') else 'no (se puede mirar en «En vivo»)'}\n"
            f"Ajuste: {self._tuning_text(cam)}\n"
            f"Grabador: {BRAND_NAMES.get(cam.get('brand'), '')} {cam.get('host')} · RTSP {cam.get('rtsp_port')} · "
            f"HTTP {cam.get('http_port')} · canal {cam.get('channel')} · usuario {cam.get('username')}\n"
            f"Zonas: {zones}\n\n"
            "Enlace para probar en VLC (con la clave del grabador):\n"
            f"rtsp://{cam.get('username')}:CLAVE@{cam.get('host')}:{cam.get('rtsp_port')}/"
            + (f"cam/realmonitor?channel={cam.get('channel')}&subtype=1" if cam.get("brand") == "dahua"
               else f"Streaming/Channels/{cam.get('channel')}02"))

    @staticmethod
    def _tuning_text(cam: dict) -> str:
        t = cam.get("tuning") or "sin_ajustar"
        if t == "sin_ajustar":
            return "sin ajustar («Ajustar…» la analiza en esta PC, en vivo o con videos, antes de aprobarla)"
        txt = {"en_ajuste": "en ajuste", "aprobada": "aprobada"}.get(t, t)
        if cam.get("tuned_by"):
            txt += f" por {cam['tuned_by']} el {fmt_date(cam.get('tuned_at'))}"
        return txt + (f" · {cam['tuning_notes']}" if cam.get("tuning_notes") else "")

    # ---------- diálogos de cliente ----------
    def _dialog(self, title: str, width: int = 520):
        top = self.tk.Toplevel(self.v.root)
        top.title(title)
        top.transient(self.v.root)
        top.resizable(False, False)
        frm = self.ttk.Frame(top, padding=14)
        frm.pack(fill="both", expand=True)
        frm.columnconfigure(1, weight=1, minsize=max(120, width - 200))

        def center() -> None:  # centrado sobre la ventana principal
            try:
                top.update_idletasks()
                r = self.v.root
                x = r.winfo_rootx() + max(0, (r.winfo_width() - top.winfo_width()) // 2)
                y = r.winfo_rooty() + max(0, (r.winfo_height() - top.winfo_height()) // 3)
                top.geometry(f"+{x}+{y}")
            except Exception:
                pass
        top.after(20, center)
        return top, frm

    def _row(self, frm, r: int, label: str, widget) -> None:
        self.ttk.Label(frm, text=label).grid(row=r, column=0, sticky="w", pady=3, padx=(0, 10))
        widget.grid(row=r, column=1, sticky="ew", pady=3)

    def client_dialog(self, c: dict | None = None) -> None:
        tk, ttk = self.tk, self.ttk
        c = c or {}
        top, frm = self._dialog("Editar cliente" if c.get("id") else "Nuevo cliente")
        v = {k: tk.StringVar(value=c.get(k) or d) for k, d in (("name", ""), ("timezone", "America/Montevideo"),
             ("contact_whatsapp", ""), ("contact_email", ""), ("telegram_chat_id", ""))}
        labels = {"name": "Nombre", "timezone": "Zona horaria", "contact_whatsapp": "WhatsApp (opcional)",
                  "contact_email": "Email (opcional)", "telegram_chat_id": "Chat de Telegram (opcional)"}
        for i, (k, lab) in enumerate(labels.items()):
            self._row(frm, i, lab, ttk.Entry(frm, textvariable=v[k]))
        plans = {p: tk.BooleanVar(value=p in (c.get("plans") or ["hurto"])) for p in PLAN_NAMES}
        pf = ttk.LabelFrame(frm, text="Planes contratados", padding=6)
        pf.grid(row=10, column=0, columnspan=2, sticky="ew", pady=8)
        for p, t in PLAN_NAMES.items():
            ttk.Checkbutton(pf, text=t, variable=plans[p]).pack(anchor="w")
        ttk.Label(pf, text="Sin «Detección de hurtos» las cámaras quedan en «solo métricas»: 2-3 cuadros por "
                           "segundo y sin clips (mucho más barato).", foreground="#666", wraplength=440,
                  justify="left").pack(anchor="w", pady=(2, 0))

        # horario del local: con el local cerrado la PC central no abre el video (ahorra internet y análisis)
        of = ttk.LabelFrame(frm, text="Horario del local (fuera de horario no se gasta video ni análisis)", padding=6)
        of.grid(row=11, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        cur_hours = c.get("hours") or {}
        hours = {d: tk.StringVar(value=cur_hours.get(d) or "") for d in DAYS}
        for i, d in enumerate(DAYS):
            r, col = i % 4, (i // 4) * 2
            ttk.Label(of, text=DAY_NAMES[d]).grid(row=r, column=col, sticky="w", padx=(0 if col == 0 else 14, 6), pady=1)
            ttk.Entry(of, textvariable=hours[d], width=22).grid(row=r, column=col + 1, sticky="w", pady=1)

        def copy_monday() -> None:
            for d in DAYS[1:5]:
                hours[d].set(hours["lun"].get())
        ttk.Button(of, text="Martes a viernes = lunes", command=copy_monday).grid(row=3, column=2, columnspan=2,
                                                                                   sticky="w", padx=(14, 0))
        hours_note = tk.StringVar()
        ttk.Label(of, textvariable=hours_note, foreground="#666", justify="left", wraplength=460).grid(
            row=4, column=0, columnspan=4, sticky="w", pady=(4, 0))

        def hours_changed(*_):
            try:
                week = open_hours_per_week({d: var.get() for d, var in hours.items()})
            except ValueError as exc:
                return hours_note.set(f"⚠ {exc}")
            base = ("Ej. 09:00-17:00 · cortado: 09:00-13:00, 16:00-20:00 · vacío = cerrado. Se empieza "
                    "15 min antes de abrir y se sigue 15 min después de cerrar.")
            if week >= 24 * 7:
                hours_note.set(base + "\nSin horario: las cámaras se miran las 24 h.")
            else:
                hours_note.set(base + f"\nAbierto {week:.0f} h por semana: se mira el {week / (24 * 7):.0%} "
                                      "del tiempo (ahorro de internet y análisis en el resto).")
        for var in hours.values():
            var.trace_add("write", hours_changed)
        hours_changed()

        # acceso a la web: con esto el dueño del comercio entra y ve sus hurtos, alertas y métricas
        had_user = bool(c.get("web_username"))
        # segunda columna (web y Hik-Connect): así la ventana entra en la pantalla de una notebook
        right = ttk.Frame(frm)
        right.grid(row=0, column=2, rowspan=15, sticky="n", padx=(16, 0))
        right.columnconfigure(0, weight=1, minsize=420)
        wf = ttk.LabelFrame(right, text="Acceso a la web del cliente", padding=6)
        wf.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        wf.columnconfigure(1, weight=1)
        web_user = tk.StringVar(value=c.get("web_username") or "")
        web_pass = tk.StringVar()
        ttk.Label(wf, text="Usuario").grid(row=0, column=0, sticky="w", padx=(0, 10), pady=2)
        ttk.Entry(wf, textvariable=web_user).grid(row=0, column=1, columnspan=2, sticky="ew", pady=2)
        ttk.Label(wf, text="Clave").grid(row=1, column=0, sticky="w", padx=(0, 10), pady=2)
        ttk.Entry(wf, textvariable=web_pass).grid(row=1, column=1, sticky="ew", pady=2)
        ttk.Button(wf, text="Generar", command=lambda: web_pass.set(make_password())).grid(row=1, column=2, padx=(6, 0))
        ttk.Label(wf, text=("Dejá la clave vacía para no cambiarla." if had_user else
                            "Mínimo 8 caracteres. Se la pasás al cliente por privado."),
                  foreground="#666").grid(row=2, column=0, columnspan=3, sticky="w")
        hf = ttk.LabelFrame(right, text="Hik-Connect: video de Hikvision sin abrir puertos (opcional)", padding=6)
        hf.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        hf.columnconfigure(1, weight=1)
        hc_key = tk.StringVar(value=c.get("hc_app_key") or "")
        hc_secret = tk.StringVar()
        hc_url = tk.StringVar(value=c.get("hc_api_url") or "")
        for i, (lab, var, show) in enumerate((("AppKey", hc_key, ""), ("SecretKey", hc_secret, "•"),
                                              ("Servidor (opcional)", hc_url, ""))):
            ttk.Label(hf, text=lab).grid(row=i, column=0, sticky="w", padx=(0, 10), pady=2)
            ttk.Entry(hf, textvariable=var, show=show).grid(row=i, column=1, sticky="ew", pady=2)
        ttk.Label(hf, text=("Portal de Hik-Connect → Team Management → API Integration."
                            + (" SecretKey vacío = no cambia." if c.get("hc_has_secret") else "")
                            + "\nServidor vacío = se busca solo (o pegá la dirección del portal, ej. isa.hikcentralconnect.com)."
                            + "\nImportante: apagá el cifrado de video de los grabadores (con cifrado, Hik-Connect no entrega el video)."),
                  foreground="#666", justify="left", wraplength=400).grid(row=4, column=0, columnspan=2, sticky="w")
        if c.get("id") and c.get("hc_app_key"):
            ttk.Button(hf, text="Probar: ver cámaras de Hik-Connect",
                       command=lambda: self.hikconnect_cameras(c, None, parent=top)).grid(row=5, column=0, columnspan=2, sticky="w", pady=(4, 0))
        loc = tk.StringVar(value=LOCATION_NAMES.get(c.get("analysis_location") or "general"))
        if self.is_admin:
            lf = ttk.LabelFrame(right, text="Qué cámaras se analizan (en las máquinas)", padding=6)
            lf.grid(row=2, column=0, sticky="ew", pady=(0, 8))
            ttk.Combobox(lf, textvariable=loc, values=list(LOCATION_NAMES.values()), state="readonly",
                         width=40).pack(anchor="w")
            ttk.Label(lf, text="Lo normal es marcar cámara por cámara con «✔ Analizar». «Todas sus cámaras» pasa "
                               "el cliente entero (salvo las marcadas «No analizar»).", foreground="#666", wraplength=400,
                      justify="left").pack(anchor="w", pady=(2, 0))

        if not c.get("id") and not web_user.get():
            # sugerir el usuario a partir del nombre (se puede cambiar)
            def suggest(*_):
                if not getattr(suggest, "touched", False):
                    web_user.set(slug_user(v["name"].get()))
            v["name"].trace_add("write", suggest)
            web_user.trace_add("write", lambda *_: setattr(suggest, "touched", web_user.get() != slug_user(v["name"].get())))

        def save() -> None:
            body = {k: (var.get().strip() or None) for k, var in v.items()}
            body["timezone"] = body["timezone"] or "America/Montevideo"
            body["webhook_url"] = c.get("webhook_url")
            body["plans"] = [p for p, var in plans.items() if var.get()]
            body["hours"] = {d: var.get().strip() for d, var in hours.items()}
            if self.is_admin:
                body["analysis_location"] = next((k for k, t in LOCATION_NAMES.items() if t == loc.get()), "general")
            try:
                open_hours_per_week(body["hours"])
            except ValueError as exc:
                return self.mb.showwarning("Vigía", f"Horario: {exc}", parent=top)
            if not body["name"]:
                return self.mb.showwarning("Vigía", "Falta el nombre", parent=top)
            user, pwd = web_user.get().strip(), web_pass.get().strip()
            if user and not had_user and not pwd:
                return self.mb.showwarning("Vigía", "Falta la clave para la web (o tocá «Generar»).", parent=top)
            if pwd and len(pwd) < 8:
                return self.mb.showwarning("Vigía", "La clave necesita al menos 8 caracteres.", parent=top)
            if had_user and not user:
                return self.mb.showwarning("Vigía", "El usuario para la web no puede quedar vacío.", parent=top)
            body["web_username"], body["web_password"] = user or None, pwd or None
            body["hc_app_key"], body["hc_api_url"] = hc_key.get().strip() or None, hc_url.get().strip() or None
            if hc_secret.get().strip():
                body["hc_secret"] = hc_secret.get().strip()
            elif body["hc_app_key"] and not c.get("hc_has_secret"):
                return self.mb.showwarning("Vigía", "Falta el SecretKey de Hik-Connect.", parent=top)

            def done(res):
                top.destroy()
                self.refresh_all()
                if pwd:  # se creó el acceso o se cambió la clave: mostrar los datos para pasárselos
                    self.web_access_dialog(res.get("name") or body["name"], user, pwd)
            self.bg.run(lambda: (self.api.put(f"/api/clients/{c['id']}", body) if c.get("id")
                                 else self.api.post("/api/clients", body)), done, lambda e: self.mb.showerror("Vigía", str(e), parent=top))
        bb = ttk.Frame(frm)
        bb.grid(row=15, column=0, columnspan=3, sticky="e")
        ttk.Button(bb, text="Cancelar", command=top.destroy).pack(side="right")
        ttk.Button(bb, text="Guardar", command=save).pack(side="right", padx=6)

    def export_dialog(self) -> None:
        """Métricas a un archivo CSV que abre Excel: por cliente (o todos), entre dos fechas, por día u hora."""
        tk, ttk = self.tk, self.ttk
        c = self._sel_client()
        top, frm = self._dialog("Exportar métricas", width=460)
        todos = "Todos los clientes"
        who = tk.StringVar(value=c["name"] if c else todos)
        today = datetime.now().date()
        d1 = tk.StringVar(value=(today - timedelta(days=6)).strftime("%d/%m/%Y"))
        d2 = tk.StringVar(value=today.strftime("%d/%m/%Y"))
        grp = tk.StringVar(value="Por día")
        self._row(frm, 0, "Cliente", ttk.Combobox(frm, textvariable=who, state="readonly",
                                                 values=[todos] + [x["name"] for x in self.clients]))
        self._row(frm, 1, "Desde (dd/mm/aaaa)", ttk.Entry(frm, textvariable=d1))
        self._row(frm, 2, "Hasta (dd/mm/aaaa)", ttk.Entry(frm, textvariable=d2))
        self._row(frm, 3, "Agrupar", ttk.Combobox(frm, textvariable=grp, state="readonly", values=["Por día", "Por hora"]))
        ttk.Label(frm, text="Se abre con Excel. Una fila por cámara y por día (u hora), en la hora del local: personas, "
                            "tiempo promedio, ingresos, vehículos, surtidor y zonas.", foreground="#666",
                  wraplength=420, justify="left").grid(row=4, column=0, columnspan=2, sticky="w", pady=(6, 0))

        def go() -> None:
            try:
                a = datetime.strptime(d1.get().strip(), "%d/%m/%Y").date()
                b = datetime.strptime(d2.get().strip(), "%d/%m/%Y").date()
            except ValueError:
                return self.mb.showwarning("Vigía", "Fechas con formato dd/mm/aaaa (por ejemplo 01/11/2026).", parent=top)
            cl = next((x for x in self.clients if x["name"] == who.get()), None)
            from tkinter import filedialog

            name = (cl["name"] if cl else "todos").replace(" ", "_")
            path = filedialog.asksaveasfilename(parent=top, defaultextension=".csv", filetypes=[("CSV para Excel", "*.csv")],
                                                initialfile=f"metricas_{name}_{a:%Y-%m-%d}_a_{b:%Y-%m-%d}.csv")
            if not path:
                return

            def work():
                return self.api.get("/api/metrics/export", client_id=cl["id"] if cl else None, desde=a.isoformat(),
                                    hasta=b.isoformat(), agrupar="hora" if grp.get() == "Por hora" else "dia")

            def done(data):
                Path(path).write_bytes(data if isinstance(data, bytes) else str(data).encode("utf-8"))
                top.destroy()
                if self.mb.askyesno("Vigía", f"Listo: {path}\n\n¿Abrirlo ahora?"):
                    try:
                        if sys.platform.startswith("win"):
                            os.startfile(path)  # type: ignore[attr-defined]
                        else:
                            subprocess.Popen(["xdg-open", path])
                    except Exception:  # noqa: BLE001
                        pass
            self.bg.run(work, done, lambda e: self.mb.showerror("Vigía", str(e), parent=top))
        bb = ttk.Frame(frm)
        bb.grid(row=5, column=0, columnspan=2, sticky="e", pady=(8, 0))
        ttk.Button(bb, text="Cancelar", command=top.destroy).pack(side="right")
        ttk.Button(bb, text="Exportar", command=go).pack(side="right", padx=6)

    def hikconnect_cameras(self, client: dict, pick: Callable[[list[dict]], None] | None, parent=None,
                           multi: bool = False) -> None:
        """Lista las cámaras que ve la API de Hik-Connect del cliente. Con `pick`, se eligen (una o varias)."""
        tk, ttk = self.tk, self.ttk
        if not client or not client.get("hc_app_key"):
            return self.mb.showwarning("Vigía", "Primero cargá la clave de Hik-Connect en el cliente "
                                                "(Editar cliente → recuadro Hik-Connect).", parent=parent)

        def done(res):
            cams = sorted(res.get("cameras") or [], key=lambda x: (x["serial"], x["channel"] or 0, x["name"]))
            top, frm = self._dialog(f"Hik-Connect · {client['name']}", width=620)
            ttk.Label(frm, text=f"{len(cams)} cámaras · servidor {res.get('server') or '—'}"
                               + ("\nElegí una o varias (Ctrl o Shift) y tocá «Usar»." if pick and multi else
                                  "\nElegí una y tocá «Usar»." if pick else "")).grid(row=0, column=0, sticky="w")
            lb = tk.Listbox(frm, height=min(18, max(4, len(cams))), width=80,
                            selectmode="extended" if multi else "browse")
            for x in cams:
                lb.insert("end", f"{x['name']}  ·  serie {x['serial'] or '—'}  ·  canal {x['channel'] or '?'}"
                                 f"{'  ·  ' + x['device_name'] if x.get('device_name') else ''}")
            lb.grid(row=1, column=0, sticky="ew", pady=6)
            bb = ttk.Frame(frm)
            bb.grid(row=2, column=0, sticky="e")
            ttk.Button(bb, text="Cerrar", command=top.destroy).pack(side="right")
            if pick:
                def use():
                    sel = [cams[i] for i in lb.curselection()]
                    if sel:
                        pick(sel)
                        top.destroy()
                ttk.Button(bb, text="Usar", command=use).pack(side="right", padx=6)
                if multi:
                    ttk.Button(bb, text="Todas", command=lambda: lb.select_set(0, "end")).pack(side="right")
        self.bg.run(lambda: self.api.get(f"/api/clients/{client['id']}/hikconnect/cameras"), done,
                    lambda e: self.mb.showerror("Vigía", f"Hik-Connect: {e}", parent=parent))

    def web_access_dialog(self, name: str, user: str, pwd: str) -> None:
        """Datos de acceso a la web para pasarle al cliente, con botón para copiarlos."""
        ttk = self.ttk
        url = self.api.server
        text = (f"Hola, estos son tus datos para ver tu comercio en Vigía:\n\nPágina: {url}\n"
                f"Usuario: {user}\nClave: {pwd}\n\nAhí vas a ver tus alertas, los hurtos revisados y el resumen del día.")
        top, frm = self._dialog(f"Acceso web de {name}", width=480)
        ttk.Label(frm, text="Pasale estos datos al cliente por privado (WhatsApp o en persona):").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))
        box = self.tk.Text(frm, height=8, width=58, wrap="word", font=("Segoe UI", 10))
        box.insert("1.0", text)
        box.grid(row=1, column=0, columnspan=2, sticky="ew")

        def copy():
            self.v.root.clipboard_clear()
            self.v.root.clipboard_append(text)
            btn.configure(text="Copiado ✓")
        bb = ttk.Frame(frm)
        bb.grid(row=2, column=0, columnspan=2, sticky="e", pady=(8, 0))
        ttk.Button(bb, text="Cerrar", command=top.destroy).pack(side="right")
        btn = ttk.Button(bb, text="Copiar", command=copy)
        btn.pack(side="right", padx=6)

    def delete_client(self) -> None:
        c = self._sel_client()
        if not c:
            return
        from tkinter import simpledialog
        typed = simpledialog.askstring("Eliminar cliente", f"Esto borra «{c['name']}» con TODAS sus cámaras, alertas y clips, "
                                       "métricas y usuarios. No se puede deshacer.\n\nEscribí el nombre para confirmar:",
                                       parent=self.v.root)
        if typed is None:
            return
        if typed.strip() != c["name"]:
            return self.mb.showwarning("Vigía", "El nombre no coincide: no se borró nada.")
        self.bg.run(lambda: self.api.delete(f"/api/clients/{c['id']}"), lambda _: self.refresh_all(), self.err)

    # ---------- diálogos de cámara ----------
    def camera_dialog(self, cam: dict | None = None) -> None:
        """Alta o edición de cámaras. Hikvision entra por Hik-Connect (sin IP ni puertos; se eligen una o
        varias cámaras de la lista de la API). Dahua, y Hikvision sin Hik-Connect, por IP y puertos."""
        tk, ttk = self.tk, self.ttk
        client = self._sel_client()
        if cam is None and client is None:
            return self.mb.showinfo("Vigía", "Elegí primero un cliente a la izquierda.")
        cam = cam or {}
        top, frm = self._dialog("Editar cámara" if cam.get("id") else f"Nueva cámara · {client['name']}", 600)
        owner = next((x for x in self.clients if x["id"] == (cam.get("client_id") or (client or {}).get("id"))), client)
        default_brand = cam.get("brand") or ("hikconnect" if (owner or {}).get("hc_app_key") else "hikvision")
        v = {
            "name": tk.StringVar(value=cam.get("name", "")),
            "brand": tk.StringVar(value=CONN_NAMES["hikconnect" if default_brand == "hikconnect" else "ip"]),
            "mode": tk.StringVar(value=MODE_NAMES.get(cam.get("mode", "vivo"))),
            "host": tk.StringVar(value=cam.get("host", "")),
            "rtsp_port": tk.StringVar(value=str(cam.get("rtsp_port", 554))),
            "http_port": tk.StringVar(value=str(cam.get("http_port", 80))),
            "channel": tk.StringVar(value=str(cam.get("channel", 1))),
            "username": tk.StringVar(value=cam.get("username", "admin")),
            "password": tk.StringVar(value=""),
            "profile": tk.StringVar(value=PROFILE_NAMES.get(cam.get("profile") or "general")),
            "fps": tk.StringVar(value=str(int(max(5, cam.get("process_fps") or 5)))),
            "off_hours": tk.StringVar(value=OFF_HOURS_NAMES.get(cam.get("off_hours") or "apagada")),
            "analyze": tk.BooleanVar(value=bool(cam.get("analyzed"))),
        }
        enabled = tk.BooleanVar(value=cam.get("enabled", True))
        nvr = tk.BooleanVar(value=cam.get("nvr_clips", True))
        # cámaras elegidas de Hik-Connect: [{id, name, serial, channel}]
        hc_sel: list[dict] = ([{"id": cam.get("hc_resource_id"), "name": cam.get("name"), "serial": cam.get("hpp_serial"),
                                "channel": cam.get("channel")}] if cam.get("brand") == "hikconnect" else [])
        hc_text = tk.StringVar()

        def show_sel() -> None:
            if not hc_sel:
                hc_text.set("Ninguna elegida")
            elif len(hc_sel) == 1:
                x = hc_sel[0]
                hc_text.set(f"{x.get('name') or '—'} · serie {x.get('serial') or '—'} · canal {x.get('channel') or '?'}")
            else:
                hc_text.set(f"{len(hc_sel)} cámaras (se crea una por cada una)")
        show_sel()

        def picked(items: list[dict]) -> None:
            hc_sel[:] = items[:1] if cam.get("id") else items
            if len(hc_sel) == 1 and (not v["name"].get().strip() or v["name"].get() == cam.get("name")):
                v["name"].set(hc_sel[0]["name"])
            show_sel()

        rows: dict[str, list] = {"hc": [], "ip": []}

        def row(kind: str | None, r: int, label: str, widget) -> None:
            lab = ttk.Label(frm, text=label)
            lab.grid(row=r, column=0, sticky="w", pady=3, padx=(0, 10))
            widget.grid(row=r, column=1, sticky="ew", pady=3)
            if kind:
                rows[kind] += [lab, widget]

        row(None, 0, "Conexión", ttk.Combobox(frm, textvariable=v["brand"], values=list(CONN_NAMES.values()), state="readonly"))
        row(None, 1, "Nombre", ttk.Entry(frm, textvariable=v["name"]))
        hc_box = ttk.Frame(frm)
        ttk.Label(hc_box, textvariable=hc_text).pack(side="left", fill="x", expand=True)
        ttk.Button(hc_box, text="Elegir de Hik-Connect…",
                   command=lambda: self.hikconnect_cameras(owner, picked, parent=top, multi=not cam.get("id"))
                   ).pack(side="left", padx=(6, 0))
        row("hc", 2, "Cámara(s)", hc_box)
        hc_hint = ttk.Label(frm, foreground="#666", justify="left",
                            text="Sin IP ni puertos: el video llega por la nube de Hik-Connect (en vivo, en el horario del local).\n"
                                 "El grabador tiene que tener APAGADO el cifrado de video.\n"
                                 "Al agregar se pueden elegir varias cámaras a la vez (Ctrl o Shift).")
        hc_hint.grid(row=3, column=0, columnspan=2, sticky="w", pady=(2, 6))
        rows["hc"].append(hc_hint)
        row("ip", 4, "Modo", ttk.Combobox(frm, textvariable=v["mode"], values=list(MODE_NAMES.values()), state="readonly"))
        row("ip", 5, "IP o DDNS del grabador", ttk.Entry(frm, textvariable=v["host"]))
        row("ip", 6, "Puerto RTSP", ttk.Entry(frm, textvariable=v["rtsp_port"]))
        row("ip", 7, "Puerto HTTP", ttk.Entry(frm, textvariable=v["http_port"]))
        row("ip", 8, "Canal", ttk.Entry(frm, textvariable=v["channel"]))
        row("ip", 9, "Usuario del grabador", ttk.Entry(frm, textvariable=v["username"]))
        row("ip", 10, "Clave del grabador" + (" (vacía = sin cambios)" if cam.get("has_password") else ""),
            ttk.Entry(frm, textvariable=v["password"], show="•"))
        ip_hint = ttk.Label(frm, foreground="#666", justify="left",
                            text="Misma red que la PC que la analiza: IP de adentro (192.168.x.x) y puertos 554/80.\n"
                                 "Otra red: IP pública o DDNS y los puertos externos del router (ej. 38554/38080).")
        ip_hint.grid(row=11, column=0, columnspan=2, sticky="w", pady=(4, 6))
        cfg = cam.get("alarm_cfg") or {}
        al = {k: tk.BooleanVar(value=k in (cfg.get("event_types") or ["VMD", "fielddetection", "linedetection"])) for k in ALARM_TYPES}
        af = ttk.LabelFrame(frm, text="Alarmas del grabador que se analizan (modos de alarma)", padding=6)
        af.grid(row=12, column=0, columnspan=2, sticky="ew", pady=4)
        for i, (k, t) in enumerate(ALARM_TYPES.items()):
            ttk.Checkbutton(af, text=t, variable=al[k]).grid(row=i // 3, column=i % 3, sticky="w", padx=4)
        nvr_cb = ttk.Checkbutton(frm, text="Bajar el clip de la grabación del grabador (si tiene disco)", variable=nvr)
        nvr_cb.grid(row=13, column=0, columnspan=2, sticky="w")
        rows["ip"] += [ip_hint, af, nvr_cb]
        row(None, 14, "Tipo de cámara (qué mira)", ttk.Combobox(frm, textvariable=v["profile"],
                                                                  values=list(PROFILE_NAMES.values()), state="readonly"))
        off_cb = ttk.Combobox(frm, textvariable=v["off_hours"], values=list(OFF_HOURS_NAMES.values()), state="readonly")
        row(None, 15, "Con el local cerrado", off_cb)
        hours_txt = ("Horario del local: se carga en el cliente (Editar cliente)." if not (owner or {}).get("hours") else
                     "Horario del local: " + " · ".join(f"{d[:2].capitalize()} {(owner['hours'].get(d) or 'cerrado')}"
                                                       for d in DAYS))
        ttk.Label(frm, text=hours_txt, foreground="#666", wraplength=560, justify="left").grid(
            row=16, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(frm, text="Analizar en las máquinas de análisis (si no, solo se mira en «En vivo»)",
                        variable=v["analyze"]).grid(row=19, column=1, sticky="w")
        row(None, 17, "Cuadros por segundo (5 a 15)", ttk.Spinbox(frm, from_=5, to=15, textvariable=v["fps"]))
        if "hurto" not in ((owner or {}).get("plans") or ["hurto"]):
            ttk.Label(frm, text="Este cliente es solo de métricas: la cámara usa 2-3 cuadros por segundo y no guarda "
                                "clips (lo de arriba no se aplica).", foreground="#666", wraplength=560,
                      justify="left").grid(row=20, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(frm, text="Habilitada", variable=enabled).grid(row=21, column=0, columnspan=2, sticky="w")

        inv = lambda d, val: next((k for k, t in d.items() if t == val), None)  # noqa: E731

        def layout(*_) -> None:
            hc = inv(CONN_NAMES, v["brand"].get()) == "hikconnect"
            for w in rows["hc"]:
                w.grid() if hc else w.grid_remove()
            for w in rows["ip"]:
                w.grid_remove() if hc else w.grid()
            # por Hik-Connect todavía no llegan las alarmas del grabador: con el local cerrado, mirar o no mirar
            opts = [t for k, t in OFF_HOURS_NAMES.items() if not (hc and k == "alarmas")]
            off_cb.configure(values=opts)
            if v["off_hours"].get() not in opts:
                v["off_hours"].set(OFF_HOURS_NAMES["apagada"])
        v["brand"].trace_add("write", layout)
        layout()

        def save() -> None:
            brand = inv(CONN_NAMES, v["brand"].get()) or "ip"
            if brand == "ip" and cam.get("brand") in ("hikvision", "dahua"):
                brand = cam["brand"]  # cámara vieja con la marca ya conocida: se respeta
            try:
                common = {
                    "client_id": cam.get("client_id") or client["id"], "brand": brand,
                    "profile": inv(PROFILE_NAMES, v["profile"].get()) or "general",
                    "process_fps": min(15.0, max(5.0, float(v["fps"].get()))), "enabled": enabled.get(),
                    "off_hours": inv(OFF_HOURS_NAMES, v["off_hours"].get()) or "apagada",
                    "detect_stream": cam.get("detect_stream", "sub"), "detector": cam.get("detector", "yolo"),
                    "playback_time": cam.get("playback_time", "local"),
                    "alarm_cfg": {**cfg, "event_types": [k for k, var in al.items() if var.get()]},
                }
                if brand != "hikconnect":
                    common.update(mode=inv(MODE_NAMES, v["mode"].get()) or "vivo", host=v["host"].get().strip(),
                                  rtsp_port=int(v["rtsp_port"].get()), http_port=int(v["http_port"].get()),
                                  channel=int(v["channel"].get()), username=v["username"].get().strip(),
                                  password=v["password"].get() or None, nvr_clips=nvr.get(),
                                  hpp_serial=cam.get("hpp_serial"))
            except ValueError:
                return self.mb.showwarning("Vigía", "Puertos, canal y cuadros por segundo tienen que ser números.", parent=top)
            if brand == "hikconnect":
                if not hc_sel or not hc_sel[0].get("serial"):
                    return self.mb.showwarning("Vigía", "Elegí la cámara con «Elegir de Hik-Connect…».", parent=top)
                bodies = [{**common, "mode": "vivo", "host": "", "username": "", "password": None, "nvr_clips": False,
                           "name": (v["name"].get().strip() if len(hc_sel) == 1 else "") or x.get("name") or "Cámara",
                           "hpp_serial": x["serial"], "channel": int(x.get("channel") or 1),
                           "hc_resource_id": x.get("id")} for x in hc_sel]
            else:
                if not v["name"].get().strip() or not common["host"]:
                    return self.mb.showwarning("Vigía", "Faltan el nombre o la IP del grabador.", parent=top)
                bodies = [{**common, "name": v["name"].get().strip()}]

            analyze = v["analyze"].get()

            def send():
                if cam.get("id"):
                    added, repeated = [self.api.put(f"/api/cameras/{cam['id']}", bodies[0])], []
                else:
                    added, repeated = [], []
                    for b in bodies:
                        try:
                            added.append(self.api.post("/api/cameras", b))
                        except ApiError as exc:
                            if exc.status != 409 or len(bodies) == 1:
                                raise
                            repeated.append(b["name"])  # ya estaba cargada: no se repite
                changed = [x["id"] for x in added if bool(x.get("analyzed")) != analyze]
                if changed:  # analizar o no: el mismo camino que los botones de la lista
                    self.api.post("/api/cameras/analyze", {"camera_ids": changed, "analyze": analyze})
                return added, repeated

            def done(res):
                added, repeated = res
                top.destroy()
                self.config_saved()
                if repeated:
                    self.mb.showinfo("Vigía", f"Se agregaron {len(added)} cámara(s). {len(repeated)} ya estaban cargadas "
                                              f"y no se repitieron: {', '.join(repeated[:6])}{'…' if len(repeated) > 6 else ''}.")
                elif len(added) > 1:
                    self.mb.showinfo("Vigía", f"Se agregaron {len(added)} cámaras.")
            self.bg.run(send, done, lambda e: self.mb.showerror("Vigía", str(e), parent=top))
        bb = ttk.Frame(frm)
        bb.grid(row=22, column=0, columnspan=2, sticky="e", pady=(8, 0))
        ttk.Button(bb, text="Cancelar", command=top.destroy).pack(side="right")
        ttk.Button(bb, text="Guardar", command=save).pack(side="right", padx=6)

    def set_analyze(self, analyze: bool) -> None:
        """Elegir qué cámaras se analizan: las marcadas van a las máquinas de análisis; las demás solo se miran
        en «En vivo»."""
        ids = [int(i) for i in self.cam_tree.selection()]
        if not ids:
            return self.mb.showinfo("Vigía", "Elegí una o varias cámaras (Ctrl o Shift para varias).")
        if not analyze and not self.mb.askyesno(
                "Vigía", f"¿Dejar de analizar {len(ids)} cámara(s)?\n\nNo van a generar alertas ni métricas. "
                         "Se pueden seguir mirando en «En vivo»."):
            return
        raw = [c["name"] for c in self.cameras if c["id"] in ids and c.get("tuning") != "aprobada" and not c.get("analyzed")]
        if analyze and raw and not self.mb.askyesno(
                "Vigía", f"{len(raw)} cámara(s) sin aprobar en «Ajustar…» ({', '.join(raw[:4])}"
                         f"{'…' if len(raw) > 4 else ''}).\n\nSin ajustar pueden dar muchas alertas falsas. "
                         "¿Analizarlas igual?"):
            return
        self.bg.run(lambda: self.api.post("/api/cameras/analyze", {"camera_ids": ids, "analyze": analyze}),
                    lambda _: self.config_saved(), self.err)

    def delete_camera(self) -> None:
        """Borra las cámaras elegidas (con Ctrl o Shift se eligen varias; «Elegir todas» toma las del cliente)."""
        ids = [int(i) for i in self.cam_tree.selection()]
        cams = [c for c in self.cameras if c["id"] in ids]
        if not cams:
            return self.mb.showinfo("Vigía", "Elegí una o varias cámaras (Ctrl o Shift para varias).")
        names = ", ".join(c["name"] for c in cams[:5]) + (f" y {len(cams) - 5} más" if len(cams) > 5 else "")
        what = f"la cámara «{cams[0]['name']}»" if len(cams) == 1 else f"{len(cams)} cámaras ({names})"
        if not self.mb.askyesno("Vigía", f"¿Eliminar {what}?\n\nSe borran también sus métricas. No se puede deshacer."):
            return

        def send(force: bool):
            return self.api.post("/api/cameras/delete", {"camera_ids": [c["id"] for c in cams], "con_alertas": force})

        def fail(exc: Exception) -> None:
            if isinstance(exc, ApiError) and exc.status == 409:
                if self.mb.askyesno("Vigía", f"{exc}\n\n¿Borrar también sus alertas, fotos y clips? No se puede deshacer."):
                    self.bg.run(lambda: send(True), lambda _: self.config_saved(), self.err)
                return
            self.err(exc)
        self.bg.run(lambda: send(False), lambda _: self.config_saved(), fail)

    def probe(self, cam: dict | None) -> None:
        if not cam:
            return self.mb.showinfo("Vigía", "Elegí una cámara.")
        self._info_locked = cam["id"]
        local = self.local_probe is not None and cam.get("agent_id") is not None
        self._set_info(f"Probando {cam['name']}" + (" desde esta PC (página del grabador y video, hasta 20 s)…"
                                                    if local else " desde el servidor (hasta 10 s)…"))

        def run():
            if self.local_probe is not None:
                res = self.local_probe(cam["id"])
                if res is not None:
                    return ("local", res)
            return ("server", self.api.post(f"/api/cameras/{cam['id']}/probe"))

        def done(out):
            where, r = out
            if where == "local":
                http, video = r["http"], r["video"]
                chans = ", ".join(f"{c['channel']}: {c.get('name') or ''}" for c in http.get("channels") or [])
                lines = [f"{cam['name']} · prueba desde esta PC", ""]
                if http.get("error"):
                    lines.append(f"Página del grabador (HTTP): NO responde · {http['error']}")
                else:
                    lines.append(f"Página del grabador (HTTP): OK · {http.get('model') or ''} · serie {http.get('serialNumber') or '—'}")
                    if chans:
                        lines.append(f"Canales: {chans}")
                if video.get("ok"):
                    lines.append(f"Video (RTSP): OK · {video.get('width')}x{video.get('height')}")
                else:
                    lines.append(f"Video (RTSP): NO anda · {video.get('error')}")
                lines += ["", f"Enlace: {r.get('rtsp_url')}"]
                if not video.get("ok") or http.get("error"):
                    lines += ["", "Revisá: misma red → IP de adentro y puertos 554/80; otra red → DDNS/IP pública y "
                                  "puertos externos (38554/38080); usuario y clave del GRABADOR."]
                self._set_info("\n".join(lines))
                return
            if r.get("hikconnect"):
                self._set_info(f"{cam['name']} · Hik-Connect (prueba desde el servidor)\n\n"
                               + (f"OK: {r.get('detail')}" if r.get("ok") else f"NO anda: {r.get('error')}")
                               + "\n\nSi falla: revisá AppKey/SecretKey en el cliente, que el grabador esté en el equipo "
                                 "de Hik-Connect y en línea, la serie y el canal.")
                return
            if r.get("ok"):
                chans = ", ".join(f"{c['channel']}: {c.get('name') or ''}" for c in r.get("channels") or [])
                self._set_info(f"{cam['name']}: el grabador responde (prueba desde el servidor).\nModelo: {r.get('model') or '—'} · "
                               f"Serie: {r.get('serialNumber') or '—'}\nCanales: {chans or '—'}")
            else:
                msg = f"{cam['name']}: el servidor no llega al grabador.\n{r.get('error')}"
                if not r.get("public"):
                    msg += ("\n\nEs normal: la IP es de una red interna (192.168…) y el servidor está en internet. "
                            "Lo que importa es que la PC central llegue: mirá el estado en la lista o probá desde la PC central.")
                self._set_info(msg + f"\n\nEnlace para VLC: {r.get('rtsp_url')}")
        self.bg.run(run, done, self.err)

    # ---------- zonas y ajuste ----------
    def zones_dialog(self, cam: dict | None) -> None:
        if not cam:
            return self.mb.showinfo("Vigía", "Elegí una cámara.")
        ZonesEditor(self, cam)

    def tune_dialog(self, cam: dict | None) -> None:
        if not cam:
            return self.mb.showinfo("Vigía", "Elegí una cámara.")
        if len(self.cam_tree.selection()) > 1:
            return self.mb.showinfo("Vigía", "«Ajustar» es de a una cámara: elegí solo una.")
        if self.agent_of is None or self.agent_of() is None:
            return self.mb.showinfo("Vigía", "La app todavía está arrancando el analizador. Probá en unos segundos.")
        TuneWindow(self, cam)

    def names_dialog(self) -> None:
        c = self._sel_client()
        if not c:
            return self.mb.showinfo("Vigía", "Elegí un cliente.")
        if not any(x["client_id"] == c["id"] for x in self.cameras):
            return self.mb.showinfo("Vigía", f"{c['name']} no tiene cámaras.")
        if self.agent_of is None or self.agent_of() is None:
            return self.mb.showinfo("Vigía", "La app todavía está arrancando. Probá en unos segundos.")
        NamesWindow(self, c)

    # ---------- métricas ----------
    def metrics_dialog(self) -> None:
        c = self._sel_client()
        if not c:
            return

        self._info_locked = (self._sel_camera() or {}).get("id")

        def done(res):
            today, week = res
            t, w = today["total"], week["total"]
            zones = sorted((today["total"].get("zones") or {}).items(), key=lambda kv: -kv[1]["visits"])
            lines = [f"Métricas de {c['name']}", "", "HOY"]
            lines += [f"  Personas vistas: {t['people']}" + (f" · ingresos: {t['entries']}" if t.get("entries") else ""),
                      f"  Tiempo promedio en cámara: {fmt_dur(t['people_dwell_s'] / t['people']) if t['people'] else '—'}",
                      f"  Máximo al mismo tiempo: {t['people_max']}"]
            if "estacion" in (c.get("plans") or []):
                lines += [f"  Vehículos: {t['vehicles']} · espera hasta el pistero: "
                          f"{fmt_dur(t['pump_wait_s'] / t['pump_wait_n']) if t['pump_wait_n'] else '—'} · "
                          f"sin atención: {t['pump_unattended']} · bajaron a la tienda: {t['shop_visits']}"]
            if zones:
                lines += ["", "  A dónde van:"] + [f"    {n}: {z['visits']} visitas, promedio {fmt_dur(z['dwell_s'] / max(1, z['visits']))}"
                                                   for n, z in zones]
            lines += ["", "ÚLTIMOS 7 DÍAS", f"  Personas: {w['people']} · vehículos: {w['vehicles']}"]
            lines += [f"    {d['date']}: {d['people']} personas" + (f", {d['vehicles']} vehículos" if d.get("vehicles") else "")
                      for d in week["by_day"]]
            if not t.get("frames") and not w.get("frames"):
                lines += ["", "Todavía no hay datos: se generan con cámaras en modo «En vivo 24 h» y se envían cada 5 minutos."]
            self._set_info("\n".join(lines))
        self.bg.run(lambda: (self.api.get("/api/metrics", client_id=c["id"], days=1),
                             self.api.get("/api/metrics", client_id=c["id"], days=7)), done, self.err)

    # ================================================================ alertas
    def _build_alerts(self) -> None:
        tk, ttk, f = self.tk, self.ttk, self.tab_alerts
        top = ttk.Frame(f)
        top.pack(fill="x")
        self.al_state = tk.StringVar(value="Sin revisar")
        self.al_client = tk.StringVar(value="Todos")
        ttk.Label(top, text="Mostrar").pack(side="left")
        ttk.Combobox(top, textvariable=self.al_state, values=["Sin revisar", "Todas", "Hurto", "Sospechoso", "Normal",
                                                               "Falso positivo"], state="readonly", width=14).pack(side="left", padx=6)
        ttk.Label(top, text="Cliente").pack(side="left")
        self.al_client_cb = ttk.Combobox(top, textvariable=self.al_client, values=["Todos"], state="readonly", width=24)
        self.al_client_cb.pack(side="left", padx=6)
        ttk.Button(top, text="Actualizar", command=self.load_alerts).pack(side="left")
        self.al_count = ttk.Label(top, text="", foreground="#666")
        self.al_count.pack(side="left", padx=10)
        for var in (self.al_state, self.al_client):
            var.trace_add("write", lambda *_: self.load_alerts())

        body = ttk.Frame(f)
        body.pack(fill="both", expand=True, pady=(6, 0))
        self.al_tree = ttk.Treeview(body, columns=("fecha", "cliente", "camara", "tipo", "etiqueta"), show="headings",
                                    selectmode="browse")
        for col, txt, w in (("fecha", "Fecha", 95), ("cliente", "Cliente", 130), ("camara", "Cámara", 120),
                            ("tipo", "Tipo", 120), ("etiqueta", "Etiqueta", 100)):
            self.al_tree.heading(col, text=txt)
            self.al_tree.column(col, width=w)
        self.al_tree.pack(side="left", fill="y")
        self.al_tree.bind("<<TreeviewSelect>>", lambda e: self.show_alert())
        right = ttk.Frame(body, padding=(10, 0, 0, 0))
        right.pack(side="left", fill="both", expand=True)
        self.al_title = ttk.Label(right, text="Elegí una alerta", font=("Segoe UI", 11, "bold"))
        self.al_title.pack(anchor="w")
        self.al_desc = ttk.Label(right, text="", foreground="#444", wraplength=640, justify="left")
        self.al_desc.pack(anchor="w")
        self.al_img = tk.Label(right, bg="#111", height=20)
        self.al_img.pack(fill="both", expand=True, pady=6)
        b = ttk.Frame(right)
        b.pack(fill="x")
        ttk.Button(b, text="▶ Reproducir clip", command=self.play_clip).pack(side="left")
        ttk.Button(b, text="Abrir en el reproductor", command=self.open_clip).pack(side="left", padx=4)
        ttk.Button(b, text="Avisar al cliente…", command=self.report_dialog).pack(side="right")
        lb = ttk.Frame(right)
        lb.pack(fill="x", pady=(6, 0))
        ttk.Label(lb, text="Etiqueta:").pack(side="left")
        for key, lab in LABEL_KEYS.items():
            ttk.Button(lb, text=f"{key} · {LABEL_NAMES[lab]}", command=lambda l=lab: self.set_label(l)).pack(side="left", padx=3)
        ttk.Label(lb, text="(teclas 1-4; pasa sola a la siguiente)", foreground="#666").pack(side="left", padx=6)
        self.v.root.bind("<Key>", self._on_key, add="+")
        self.alerts: list[dict] = []
        self._play = {"cap": None, "job": None, "photo": None, "path": None}

    def _fill_alert_filters(self) -> None:
        self.al_client_cb.configure(values=["Todos"] + [c["name"] for c in self.clients])

    def load_alerts(self) -> None:
        label = {"Sin revisar": "sin_etiqueta", "Todas": None}.get(self.al_state.get())
        if label is None and self.al_state.get() != "Todas":
            label = next((k for k, t in LABEL_NAMES.items() if t == self.al_state.get()), None)
        cid = next((c["id"] for c in self.clients if c["name"] == self.al_client.get()), None)

        def done(res):
            self.alerts = res["items"]
            self.al_tree.delete(*self.al_tree.get_children())
            for e in self.alerts:
                self.al_tree.insert("", "end", iid=str(e["id"]), values=(
                    fmt_date(e["occurred_at"]), e.get("client_name") or "", e.get("camera_name") or "",
                    TYPE_NAMES.get(e["type"], e["type"]), LABEL_NAMES.get(e.get("label"), "Sin revisar")))
            self.al_count.configure(text=f"{res['total']} en total")
        self.bg.run(lambda: self.api.get("/api/events", label=label, client_id=cid, limit=200), done, self._net_fail)

    def _sel_alert(self) -> dict | None:
        sel = self.al_tree.selection()
        return next((e for e in self.alerts if str(e["id"]) == (sel[0] if sel else "")), None)

    def show_alert(self) -> None:
        self.stop_clip()
        e = self._sel_alert()
        if not e:
            return
        self.al_title.configure(text=f"{TYPE_NAMES.get(e['type'], e['type'])} · {e.get('client_name')} · {e.get('camera_name')}")
        self.al_desc.configure(text=f"{fmt_date(e['occurred_at'])} · {e.get('description') or ''}"
                               + ("" if e.get("clip_url") else "  (el clip todavía no llegó)"))
        if e.get("snapshot_url"):
            def done(data):
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if img is not None and self._sel_alert() is e:
                    self._show_frame(img)
            self.bg.run(lambda: self.api.download(e["snapshot_url"]), done, lambda exc: None)

    def _show_frame(self, img: np.ndarray) -> None:
        w, h = max(self.al_img.winfo_width(), 320), max(self.al_img.winfo_height(), 240)
        photo = ppm_photo(self.tk, img, w, h)
        if photo:
            self._play["photo"] = photo
            self.al_img.configure(image=photo)

    def _clip_file(self, e: dict, then: Callable[[str], None]) -> None:
        if not e.get("clip_url"):
            return self.mb.showinfo("Vigía", "Esta alerta todavía no tiene clip.")
        path = Path(tempfile.gettempdir()) / f"vigia_clip_{e['id']}.mp4"
        if path.exists() and path.stat().st_size > 0:
            return then(str(path))
        self.al_desc.configure(text="Bajando el clip…")

        def fetch():
            path.write_bytes(self.api.download(e["clip_url"]))
            return str(path)
        self.bg.run(fetch, then, self.err)

    def play_clip(self) -> None:
        e = self._sel_alert()
        if not e:
            return

        def start(path: str) -> None:
            self.stop_clip()
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                return self.mb.showwarning("Vigía", "No se pudo abrir el clip acá. Probá «Abrir en el reproductor».")
            fps = cap.get(cv2.CAP_PROP_FPS) or 10
            self._play.update(cap=cap, path=path)
            self.al_desc.configure(text=f"{fmt_date(e['occurred_at'])} · {e.get('description') or ''}")
            delay = max(20, int(1000 / min(fps, 25)))

            def step() -> None:
                c = self._play["cap"]
                if c is None:
                    return
                ok, frame = c.read()
                if not ok:
                    self.stop_clip()
                    return
                self._show_frame(frame)
                self._play["job"] = self.v.root.after(delay, step)
            step()
        self._clip_file(e, start)

    def stop_clip(self) -> None:
        if self._play.get("job"):
            try:
                self.v.root.after_cancel(self._play["job"])
            except Exception:
                pass
        if self._play.get("cap") is not None:
            self._play["cap"].release()
        self._play.update(cap=None, job=None)

    def open_clip(self) -> None:
        e = self._sel_alert()
        if e:
            self._clip_file(e, open_file)

    def _on_key(self, ev) -> None:
        if ev.char in LABEL_KEYS and self.v.notebook.select() == str(self.tab_alerts):
            if not isinstance(ev.widget, (self.tk.Entry, self.ttk.Entry, self.tk.Text)):
                self.set_label(LABEL_KEYS[ev.char])

    def set_label(self, label: str) -> None:
        e = self._sel_alert()
        if not e:
            return
        idx = self.al_tree.index(str(e["id"]))

        def done(res):
            e["label"] = res.get("label")
            if self.al_state.get() == "Sin revisar":
                self.al_tree.delete(str(e["id"]))
                self.alerts = [x for x in self.alerts if x["id"] != e["id"]]
            else:
                self.al_tree.set(str(e["id"]), "etiqueta", LABEL_NAMES.get(label, label))
            kids = self.al_tree.get_children()
            if kids:  # pasa a la siguiente
                nxt = kids[min(idx if self.al_state.get() == "Sin revisar" else idx + 1, len(kids) - 1)]
                self.al_tree.selection_set(nxt)
                self.al_tree.see(nxt)
        self.bg.run(lambda: self.api.patch(f"/api/events/{e['id']}", {"label": label}), done, self.err)

    def report_dialog(self) -> None:
        e = self._sel_alert()
        if not e:
            return
        if not e.get("clip_url"):
            return self.mb.showinfo("Vigía", "Esta alerta todavía no tiene clip para mandar.")
        tk, ttk = self.tk, self.ttk
        top, frm = self._dialog(f"Avisar a {e.get('client_name')}", 560)
        ttk.Label(frm, text="Revisá el clip antes de avisar. El mensaje habla de «posible hurto», nunca acusa.",
                  foreground="#666", wraplength=520).grid(row=0, column=0, columnspan=2, sticky="w")
        txt = tk.Text(frm, height=6, width=60, wrap="word")
        txt.insert("1.0", f"Detectamos una situación sospechosa (posible hurto) en la cámara «{e.get('camera_name')}» "
                          f"el {fmt_date(e['occurred_at'])}. Te compartimos el clip para que lo revises.")
        txt.grid(row=1, column=0, columnspan=2, pady=8)
        out = ttk.Label(frm, text="", wraplength=520, justify="left")
        out.grid(row=2, column=0, columnspan=2, sticky="w")

        def send():
            msg = txt.get("1.0", "end").strip()

            def done(rep):
                link = rep.get("link") or ""
                self.v.root.clipboard_clear()
                self.v.root.clipboard_append(link)
                chans = ", ".join(rep.get("channels") or []) or "portal"
                out.configure(text=f"Enviado por: {chans}.\nLink privado (copiado al portapapeles, vence en 14 días):\n{link}")
            self.bg.run(lambda: self.api.post(f"/api/events/{e['id']}/report", {"message": msg}), done,
                        lambda exc: self.mb.showerror("Vigía", str(exc), parent=top))
        bb = ttk.Frame(frm)
        bb.grid(row=3, column=0, columnspan=2, sticky="e", pady=(8, 0))
        ttk.Button(bb, text="Cerrar", command=top.destroy).pack(side="right")
        ttk.Button(bb, text="Enviar aviso", command=send).pack(side="right", padx=6)

    # ================================================================ administración
    def _build_admin(self) -> None:
        ttk, f = self.ttk, self.tab_admin
        ttk.Label(f, text="Usuarios", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Label(f, text="Administradores y empleados entran a esta app con su usuario y clave y abren las cámaras "
                          "desde «En vivo» (no hay que asignar nada). Los clientes entran a la web y ven solo su local.",
                  foreground="#666", wraplength=900, justify="left").pack(anchor="w")
        self.us_tree = ttk.Treeview(f, columns=("rol", "cliente", "activo"), show="tree headings", height=5,
                                    selectmode="browse")
        self.us_tree.heading("#0", text="Usuario")
        for col, txt, w in (("rol", "Rol", 120), ("cliente", "Cliente", 160),
                            ("activo", "Activo", 60)):
            self.us_tree.heading(col, text=txt)
            self.us_tree.column(col, width=w)
        self.us_tree.pack(fill="x")
        self.us_tree.bind("<Double-1>", lambda e: self.user_dialog(self._sel_user()))
        b = ttk.Frame(f)
        b.pack(fill="x", pady=6)
        ttk.Button(b, text="Nuevo usuario", command=lambda: self.user_dialog()).pack(side="left")
        ttk.Button(b, text="Editar", command=lambda: self.user_dialog(self._sel_user())).pack(side="left", padx=4)
        ttk.Button(b, text="Eliminar", command=self.delete_user).pack(side="left")
        self.users: list[dict] = []
        self._build_machines(f)

    # ---------- máquinas de análisis ----------
    def _build_machines(self, f) -> None:
        tk, ttk = self.tk, self.ttk
        ttk.Separator(f).pack(fill="x", pady=8)
        ttk.Label(f, text="Dónde se analizan las cámaras", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        top = ttk.Frame(f)
        top.pack(fill="x", pady=(2, 4))
        self.an_mode = tk.StringVar()
        self.an_cap = tk.StringVar(value="30")
        self.an_spare = tk.StringVar(value="0")
        ttk.Combobox(top, textvariable=self.an_mode, values=list(ANALYSIS_MODES.values()), state="readonly",
                     width=34).pack(side="left")
        ttk.Label(top, text="  Cámaras por máquina").pack(side="left")
        ttk.Spinbox(top, from_=5, to=500, textvariable=self.an_cap, width=5).pack(side="left", padx=4)
        ttk.Label(top, text="Máquinas de reserva").pack(side="left")
        ttk.Spinbox(top, from_=0, to=5, textvariable=self.an_spare, width=3).pack(side="left", padx=4)
        ttk.Button(top, text="Guardar", command=self.save_analysis).pack(side="left", padx=6)
        self.an_info = tk.StringVar(value="Cargando…")
        ttk.Label(f, textvariable=self.an_info, foreground="#444", wraplength=900, justify="left").pack(anchor="w")
        self.mq_tree = ttk.Treeview(f, columns=("estado", "camaras", "carga", "instancia"), show="tree headings",
                                    height=8, selectmode="browse")
        self.mq_tree.heading("#0", text="Máquina")
        self.mq_tree.column("#0", width=300)
        for col, txt, w in (("estado", "Estado", 150), ("camaras", "Cámaras", 80), ("carga", "Carga / capacidad", 130),
                            ("instancia", "Instancia en la nube", 190)):
            self.mq_tree.heading(col, text=txt)
            self.mq_tree.column(col, width=w)
        b = ttk.Frame(f)  # los botones van antes que la lista: en pantallas bajas no quedan tapados
        b.pack(side="bottom", fill="x", pady=6)
        self.mq_tree.pack(fill="both", expand=True)
        ttk.Button(b, text="Clave para máquinas nuevas", command=self.pool_key_dialog).pack(side="left")
        ttk.Button(b, text="Quitar máquina o PC", command=self.remove_machine).pack(side="left", padx=4)
        ttk.Button(b, text="Actualizar", command=self.load_machines).pack(side="left")
        self.analysis: dict = {}

    def load_machines(self) -> None:
        def done(res):
            a, cameras, agents = res
            self.analysis = a
            self.an_mode.set(ANALYSIS_MODES.get(a.get("mode"), ANALYSIS_MODES["pcs"]))
            self.an_cap.set(str(int(a.get("capacity") or 30)))
            self.an_spare.set(str(a.get("spare") or 0))
            p = a.get("plan") or {}
            if a.get("mode") == "servidores":
                info = (f"Ahora: {a.get('cameras_active', 0)} de {a.get('cameras', 0)} cámaras para analizar "
                        f"(las demás con el local cerrado) · carga {p.get('units', 0)} · hacen falta "
                        f"{p.get('desired', 0)} máquina(s) · {p.get('online', 0)} en línea.")
                if not a.get("has_pool_key"):
                    info += "  Falta crear la clave para máquinas (botón de abajo)."
            else:
                info = ("Las máquinas de análisis analizan las cámaras marcadas «✔ Analizar» en Clientes. Las demás solo "
                        "se miran en «En vivo». Cómo instalar una máquina: deploy/maquina-analisis/README.md.")
                if a.get("cameras"):
                    p = a.get("plan") or {}
                    info += (f"  Ahora {a['cameras']} cámara(s) van a máquinas · carga {p.get('units', 0)} · hacen falta "
                             f"{p.get('desired', 0)} máquina(s) · {p.get('online', 0)} en línea.")
            self.an_info.set(info)
            self._fill_machine_tree(a, cameras, agents)
        self.bg.run(lambda: (self.api.get("/api/analysis"), self.api.get("/api/cameras"), self.api.get("/api/agents")),
                    done, self._net_fail)

    def _fill_machine_tree(self, a: dict, cameras: list[dict], agents: list[dict]) -> None:
        """Qué analiza cada máquina (y cada PC): máquina → cliente → cámara. Solo datos, sin imágenes."""
        tree = self.mq_tree
        opened = {i for i in tree.get_children() if tree.item(i, "open")}
        tree.delete(*tree.get_children())
        by_agent = {x["id"]: x for x in agents}
        groups: list[tuple[str, str, str, str, str, set]] = []
        used: set[int] = set()
        for m in a.get("machines") or []:
            name = m["names"][0] if len(m["names"]) == 1 else (
                m["names"][0].rsplit("-", 1)[0] + f" ({len(m['names'])} procesos)")
            estado = "Vaciando para apagar" if m["online"] and m["draining"] else (
                "En línea" if m["online"] else "Apagada")
            groups.append((m["key"], name, estado, f"{m['load']} / {m['capacity']:.0f}", m.get("instance_id") or "—",
                           set(m["agent_ids"])))
            used |= set(m["agent_ids"])
        pcs = {c["agent_id"] for c in cameras if c.get("agent_id") and c["agent_id"] not in used}
        pcs |= {x["id"] for x in agents if (x.get("kind") or "pc") == "pc" and x["id"] not in used}
        for aid in sorted(pcs, key=lambda i: ((by_agent.get(i) or {}).get("name") or "", i)):
            ag = by_agent.get(aid) or {}
            groups.append((f"pc-{aid}", ag.get("name") or f"PC {aid}", "En línea" if ag.get("online") else "Desconectada",
                           "—", "PC", {aid}))
        if any(not c.get("agent_id") for c in cameras):
            groups.append(("ninguna", "Sin máquina ni PC", "No se analizan", "—", "—", {None}))
        for key, name, estado, carga, inst, ids in groups:
            cams = [c for c in cameras if c.get("agent_id") in ids]
            node = tree.insert("", "end", iid=key, text=name, values=(estado, len(cams), carga, inst),
                               open=key in opened)
            by_client: dict[int, list[dict]] = {}
            for c in cams:
                by_client.setdefault(c["client_id"], []).append(c)
            for cl_id, lst in sorted(by_client.items(), key=lambda kv: self.client_name(kv[0])):
                cnode = tree.insert(node, "end", iid=f"{key}/c{cl_id}", text=self.client_name(cl_id),
                                    values=("", len(lst), "", ""))
                for c in sorted(lst, key=lambda x: x["name"]):
                    tree.insert(cnode, "end", iid=f"{key}/k{c['id']}", text=c["name"],
                                values=(self._cam_state(c), "", "", ""))

    def save_analysis(self) -> None:
        mode = next((k for k, t in ANALYSIS_MODES.items() if t == self.an_mode.get()), "pcs")
        try:
            body = {"mode": mode, "capacity": float(self.an_cap.get()), "spare": int(self.an_spare.get())}
        except ValueError:
            return self.mb.showwarning("Vigía", "Cámaras por máquina y reserva tienen que ser números.")
        if mode != (self.analysis.get("mode") or "pcs"):
            msg = ("Las cámaras pasan a analizarse en las máquinas de análisis. Mientras no haya ninguna en línea, "
                   "siguen en la PC de cada usuario.") if mode == "servidores" else (
                   "Las cámaras vuelven a analizarse en la PC de quien las revisa.")
            if not self.mb.askyesno("Vigía", msg + "\n\n¿Seguir?"):
                return
        self.bg.run(lambda: self.api.put("/api/analysis", body), lambda _: (self.load_machines(), self.config_saved()),
                    self.err)

    def pool_key_dialog(self) -> None:
        if self.analysis.get("has_pool_key") and not self.mb.askyesno(
                "Vigía", "Ya hay una clave. Si creás otra, las máquinas nuevas tienen que usar la nueva "
                         "(las que ya están dadas de alta siguen andando).\n\n¿Crear otra?"):
            return

        def done(res):
            ttk = self.ttk
            key = res["key"]
            top, frm = self._dialog("Clave para máquinas de análisis", width=560)
            ttk.Label(frm, text="Copiala ahora: no se vuelve a mostrar. Va en la instalación de cada máquina "
                                "(VIGIA_POOL_KEY) y en el encendedor.", wraplength=520, justify="left").grid(
                row=0, column=0, columnspan=2, sticky="w")
            box = self.tk.Text(frm, height=3, width=64, wrap="char")
            box.insert("1.0", key)
            box.grid(row=1, column=0, columnspan=2, sticky="ew", pady=6)

            def copy():
                self.v.root.clipboard_clear()
                self.v.root.clipboard_append(key)
                btn.configure(text="Copiada ✓")
            btn = ttk.Button(frm, text="Copiar", command=copy)
            btn.grid(row=2, column=1, sticky="e")
            self.load_machines()
        self.bg.run(lambda: self.api.post("/api/analysis/pool-key"), done, self.err)

    def remove_machine(self) -> None:
        sel = self.mq_tree.selection()
        key = sel[0] if sel else ""
        while key and self.mq_tree.parent(key):
            key = self.mq_tree.parent(key)
        if key.startswith("pc-"):
            name = self.mq_tree.item(key, "text")
            if not self.mb.askyesno("Vigía", f"¿Quitar la PC «{name}»? Deja de analizar y las cámaras que tenga abiertas "
                                             "se cierran. Si alguien vuelve a entrar en esa PC con su usuario, aparece de "
                                             "nuevo (sin cámaras)."):
                return
            return self.bg.run(lambda: self.api.delete(f"/api/agents/{key[3:]}"), lambda _: self.load_machines(), self.err)
        m = next((x for x in (self.analysis.get("machines") or []) if x["key"] == key), None)
        if not m:
            return self.mb.showinfo("Vigía", "Elegí una máquina o una PC de la lista.")
        if not self.mb.askyesno("Vigía", f"¿Quitar «{', '.join(m['names'])}»? Sus cámaras pasan a las otras máquinas. "
                                         "Si la máquina sigue prendida, se vuelve a dar de alta sola."):
            return
        self.bg.run(lambda: [self.api.delete(f"/api/agents/{aid}") for aid in m["agent_ids"]],
                    lambda _: self.load_machines(), self.err)

    def load_users(self) -> None:
        def done(users):
            self.users = users
            self.us_tree.delete(*self.us_tree.get_children())
            for u in users:
                self.us_tree.insert("", "end", iid=str(u["id"]), text=u["username"], values=(
                    ROLE_NAMES.get(u["role"], u["role"]), self.client_name(u.get("client_id")) if u.get("client_id") else "—",
                    "Sí" if u.get("active") else "No"))
        self.bg.run(lambda: self.api.get("/api/users"), done, self._net_fail)

    def _sel_user(self) -> dict | None:
        sel = self.us_tree.selection()
        return next((u for u in self.users if str(u["id"]) == (sel[0] if sel else "")), None)

    def delete_user(self) -> None:
        u = self._sel_user()
        if not u:
            return self.mb.showinfo("Vigía", "Elegí un usuario de la lista.")
        if not self.mb.askyesno("Vigía", f"¿Eliminar el usuario «{u['username']}»?\n\nYa no va a poder entrar. Lo que hizo "
                                         "(etiquetas, avisos) queda guardado. No se puede deshacer."):
            return
        self.bg.run(lambda: self.api.delete(f"/api/users/{u['id']}"), lambda _: self.load_users(), self.err)

    def user_dialog(self, u: dict | None = None) -> None:
        tk, ttk = self.tk, self.ttk
        u = u or {}
        top, frm = self._dialog("Editar usuario" if u.get("id") else "Nuevo usuario")
        name = tk.StringVar(value=u.get("username", ""))
        pwd = tk.StringVar()
        role = tk.StringVar(value=ROLE_NAMES.get(u.get("role", "entrenador")))
        client = tk.StringVar(value=self.client_name(u.get("client_id")) if u.get("client_id") else "")
        active = tk.BooleanVar(value=u.get("active", True))
        self._row(frm, 0, "Usuario", ttk.Entry(frm, textvariable=name))
        self._row(frm, 1, "Clave nueva (mín. 8)" if u.get("id") else "Clave (mín. 8)", ttk.Entry(frm, textvariable=pwd, show="•"))
        self._row(frm, 2, "Rol", ttk.Combobox(frm, textvariable=role, values=list(ROLE_NAMES.values()), state="readonly"))
        self._row(frm, 3, "Cliente (solo rol Cliente)", ttk.Combobox(frm, textvariable=client, values=[c["name"] for c in self.clients], state="readonly"))
        ttk.Checkbutton(frm, text="Activo", variable=active).grid(row=4, column=1, sticky="w")
        def save():
            body = {"username": name.get().strip(), "password": pwd.get() or None,
                    "role": next((k for k, t in ROLE_NAMES.items() if t == role.get()), "entrenador"),
                    "client_id": next((c["id"] for c in self.clients if c["name"] == client.get()), None),
                    "active": active.get()}
            if body["role"] != "cliente":
                body["client_id"] = None

            def send():
                return self.api.put(f"/api/users/{u['id']}", body) if u.get("id") else self.api.post("/api/users", body)

            def done(_):
                top.destroy()
                self.config_saved()
            self.bg.run(send, done, lambda e: self.mb.showerror("Vigía", str(e), parent=top))
        bb = ttk.Frame(frm)
        bb.grid(row=5, column=0, columnspan=2, sticky="e", pady=(8, 0))
        ttk.Button(bb, text="Cancelar", command=top.destroy).pack(side="right")
        ttk.Button(bb, text="Guardar", command=save).pack(side="right", padx=6)


class ZonesEditor:
    """Editor de zonas: clics sobre la imagen de la cámara para marcar vértices, doble clic para cerrar."""

    W, H = 720, 405

    def __init__(self, app: Oficina, cam: dict, image: np.ndarray | None = None,
                 on_saved: Callable[[dict], None] | None = None, confirm: str | None = None):
        """`image`: dibujar sobre esa imagen (la del ajuste) en vez de bajar la última de la cámara. `on_saved`
        recibe lo guardado (zonas y reglas). `confirm`: pregunta antes de guardar (cámara que ya se analiza)."""
        self.app, self.cam = app, cam
        self.on_saved = on_saved
        self.confirm = confirm
        tk, ttk = app.tk, app.ttk
        self.zones = [dict(z) for z in cam.get("zones") or []]
        self.rules = dict(cam.get("rules") or {})
        self.draft: list[tuple[float, float]] = []
        self.top, frm = app._dialog(f"Zonas y reglas · {cam['name']}", 300)
        self.top.resizable(True, True)
        left = ttk.Frame(frm)
        left.grid(row=0, column=0, sticky="n")
        self.canvas = tk.Canvas(left, width=self.W, height=self.H, bg="#111", highlightthickness=0, cursor="crosshair")
        self.canvas.pack()
        self.info = ttk.Label(left, text="Cargando la imagen de la cámara…", foreground="#666")
        self.info.pack(anchor="w", pady=4)
        ttk.Label(left, text="Clic: agregar punto · Doble clic: cerrar la zona · Las zonas van sobre el PISO (se mira dónde pisa cada persona).",
                  foreground="#666").pack(anchor="w")
        right = ttk.Frame(frm, padding=(12, 0, 0, 0))
        right.grid(row=0, column=1, sticky="n")
        self.zname = tk.StringVar()
        self.ztype = tk.StringVar(value=ZONE_NAMES["shelf"])
        ttk.Label(right, text="Nueva zona", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        ttk.Label(right, text="Nombre").pack(anchor="w")
        ttk.Entry(right, textvariable=self.zname, width=28).pack(anchor="w")
        ttk.Label(right, text="Tipo").pack(anchor="w", pady=(4, 0))
        ttk.Combobox(right, textvariable=self.ztype, values=list(ZONE_NAMES.values()), state="readonly", width=26).pack(anchor="w")
        bz = ttk.Frame(right)
        bz.pack(anchor="w", pady=4)
        ttk.Button(bz, text="Cerrar zona", command=self.close_zone).pack(side="left")
        ttk.Button(bz, text="Deshacer punto", command=self.undo).pack(side="left", padx=4)
        ttk.Label(right, text="Zonas", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(8, 0))
        self.zlist = tk.Listbox(right, height=8, width=34)
        self.zlist.pack(anchor="w")
        ttk.Button(right, text="Borrar zona elegida", command=self.delete_zone).pack(anchor="w", pady=4)
        ttk.Label(right, text="Reglas (segundos)", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(8, 0))
        self.rv = {}
        for key, lab, default in (("loiter_seconds", "Merodeo: tiempo en góndola", 45),
                                  ("shelf_dwell_seconds", "En góndola para «tomó algo»", 3),
                                  ("checkout_min_seconds", "En caja para «pagó»", 1),
                                  ("restricted_min_seconds", "En zona restringida", 1)):
            row = ttk.Frame(right)
            row.pack(anchor="w", fill="x")
            ttk.Label(row, text=lab, width=28).pack(side="left")
            var = tk.StringVar(value=str(self.rules.get(key, default)))
            ttk.Entry(row, textvariable=var, width=6).pack(side="left")
            self.rv[key] = var
        bb = ttk.Frame(right)
        bb.pack(anchor="e", pady=(14, 0), fill="x")
        ttk.Button(bb, text="Cancelar", command=self.top.destroy).pack(side="right")
        ttk.Button(bb, text="Guardar", command=self.save).pack(side="right", padx=6)
        self.photo = None
        self.canvas.bind("<Button-1>", self.click)
        self.canvas.bind("<Double-Button-1>", lambda e: self.close_zone())
        self.refresh_list()
        self.draw()
        if image is not None:
            self.photo = ppm_photo(tk, cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA), self.W, self.H)
            self.info.configure(text="Imagen del ajuste (lo que se está analizando)")
            self.draw()
        else:
            self.load_image()

    def load_image(self) -> None:
        if not self.cam.get("has_preview"):
            self.info.configure(text="Todavía no hay imagen: la PC central la manda al conectarse a la cámara. "
                                     "Podés dibujar igual sobre el fondo negro.")
            return

        def done(data):
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return
            img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
            self.photo = ppm_photo(self.app.tk, img, self.W, self.H)
            self.info.configure(text=f"Imagen de {fmt_date(self.cam.get('preview_at'))}")
            self.draw()
        self.app.bg.run(lambda: self.app.api.download(f"/api/cameras/{self.cam['id']}/preview"), done,
                        lambda e: self.info.configure(text=f"No se pudo bajar la imagen: {e}"))

    def click(self, ev) -> None:
        self.draft.append((min(max(ev.x / self.W, 0.0), 1.0), min(max(ev.y / self.H, 0.0), 1.0)))
        self.draw()

    def undo(self) -> None:
        if self.draft:
            self.draft.pop()
            self.draw()

    def close_zone(self) -> None:
        pts = list(dict.fromkeys(self.draft))
        if len(pts) < 3:
            return self.app.mb.showinfo("Vigía", "Una zona necesita al menos 3 puntos.", parent=self.top)
        ztype = next((k for k, t in ZONE_NAMES.items() if t == self.ztype.get()), "shelf")
        name = self.zname.get().strip() or f"{ZONE_NAMES[ztype].split()[0].lower()}_{len(self.zones) + 1}"
        if any(z["name"] == name for z in self.zones):
            return self.app.mb.showinfo("Vigía", "Ya hay una zona con ese nombre.", parent=self.top)
        self.zones.append({"name": name, "type": ztype, "polygon": [list(p) for p in pts]})
        self.draft = []
        self.zname.set("")
        self.refresh_list()
        self.draw()

    def delete_zone(self) -> None:
        sel = self.zlist.curselection()
        if sel:
            del self.zones[sel[0]]
            self.refresh_list()
            self.draw()

    def refresh_list(self) -> None:
        self.zlist.delete(0, "end")
        for z in self.zones:
            self.zlist.insert("end", f"{z['name']}  ·  {ZONE_NAMES.get(z['type'], z['type'])}")

    def draw(self) -> None:
        c = self.canvas
        c.delete("all")
        if self.photo is not None:
            c.create_image(0, 0, image=self.photo, anchor="nw")
        for z in self.zones:
            col = ZONE_COLORS.get(z["type"], "#fff")
            pts = [v for x, y in z["polygon"] for v in (x * self.W, y * self.H)]
            c.create_polygon(*pts, outline=col, fill=col, stipple="gray25", width=2)
            c.create_text(pts[0] + 4, pts[1] + 4, text=z["name"], fill=col, anchor="nw", font=("Segoe UI", 10, "bold"))
        if self.draft:
            col = ZONE_COLORS.get(next((k for k, t in ZONE_NAMES.items() if t == self.ztype.get()), "shelf"), "#fff")
            pts = [v for x, y in self.draft for v in (x * self.W, y * self.H)]
            if len(self.draft) > 1:
                c.create_line(*pts, fill=col, width=2)
            for x, y in self.draft:
                c.create_oval(x * self.W - 4, y * self.H - 4, x * self.W + 4, y * self.H + 4, fill=col, outline="")

    def save(self) -> None:
        try:
            rules = {**self.rules, **{k: float(v.get()) for k, v in self.rv.items()}}
        except ValueError:
            return self.app.mb.showwarning("Vigía", "Las reglas tienen que ser números.", parent=self.top)
        if self.confirm and not self.app.mb.askyesno("Vigía", self.confirm, parent=self.top):
            return
        body = {"zones": self.zones, "rules": rules}

        def done(_):
            self.top.destroy()
            self.app.config_saved()
            if self.on_saved:
                self.on_saved(body)
            self.app.v.log_handler.q.put_nowait(f"{datetime.now():%H:%M:%S}  Zonas de «{self.cam['name']}» guardadas: "
                                                 "la PC central las aplica en segundos")
        self.app.bg.run(lambda: self.app.api.put(f"/api/cameras/{self.cam['id']}/analysis", body), done,
                        lambda e: self.app.mb.showerror("Vigía", str(e), parent=self.top))


class TuneWindow:
    """«Ajustar cámara»: la analiza en esta PC (en vivo o con videos grabados, por ejemplo hurtos de esa cámara) con
    alertas de prueba que no salen de la app. Los tiempos se prueban y todo lo analizado se recalcula al instante;
    «Aprobar y analizar» la manda a las máquinas de análisis. El motor está en agent/ajuste.py (`Tuner`)."""

    W, H = 640, 360  # el tamaño en que se guardan los cuadros: entra en una pantalla de 1366x768
    TIMES = (("loiter_seconds", "Merodeo: tiempo en góndola", 45),
             ("shelf_dwell_seconds", "En góndola para «tomó algo»", 3),
             ("checkout_min_seconds", "En caja para «pagó»", 1),
             ("restricted_min_seconds", "En zona restringida", 1))

    def __init__(self, app: Oficina, cam: dict):
        self.app, self.cam = app, cam
        tk, ttk = app.tk, app.ttk
        self.agent = app.agent_of() if app.agent_of else None
        self.tuner = None
        self.info: dict | None = None  # lo que da /api/agent/tune
        self.detector = None
        self.analysis: dict = {}  # zonas y reglas con las que se analiza (con los tiempos que se están probando)
        self.index: int | None = None  # cuadro que se muestra (None = el último)
        self.playing = False
        self._play_f = 0.0
        self._last_tick = time.monotonic()
        self._shown: tuple | None = None
        self._alerts_key: tuple | None = None
        self._alert_rows: list = []
        self._stats_at = 0.0
        self._setting_scale = False
        self.dirty = False  # tiempos probados sin guardar
        self.closed = False
        self.preview_sent = False
        self.top, frm = app._dialog(f"Ajustar cámara · {cam['name']}")
        self.top.resizable(True, True)
        self.top.protocol("WM_DELETE_WINDOW", self.close)

        left = ttk.Frame(frm)
        left.grid(row=0, column=0, sticky="nw")
        self.canvas = tk.Canvas(left, width=self.W, height=self.H, bg="#111", highlightthickness=0)
        self.canvas.pack()
        self.photo = None
        nav = ttk.Frame(left)
        nav.pack(fill="x", pady=(4, 0))
        self.play_btn = ttk.Button(nav, text="▶", width=3, command=self.toggle_play)
        self.play_btn.pack(side="left")
        self.scale = ttk.Scale(nav, from_=0, to=1, orient="horizontal", command=self.on_scale)
        self.scale.pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(nav, text="Último ▸▸", command=self.follow).pack(side="left")
        self.pos = ttk.Label(left, text="", foreground="#666")
        self.pos.pack(anchor="w")
        self.state_lbl = ttk.Label(left, text="Preparando el detector…", font=("Segoe UI", 10, "bold"))
        self.state_lbl.pack(anchor="w", pady=(2, 0))
        self.live_warn = ttk.Label(left, text="", foreground="#b45309", wraplength=self.W, justify="left")
        self.live_warn.pack(anchor="w")
        ttk.Label(left, foreground="#666", wraplength=self.W, justify="left", text=(
            "Se analiza en esta PC igual que en las máquinas. Debajo de cada persona: segundos en cada zona contra el "
            "tiempo de la regla (en rojo: llegó) y «TOMO ALGO» si estuvo en una góndola y todavía no pasó por caja. "
            "En vivo se recuerdan los últimos 10 minutos y se usa una transmisión del grabador mientras esta ventana "
            "esté abierta.")).pack(anchor="w", pady=(2, 0))
        self.rules_lbl = ttk.Label(left, text="", foreground="#666", wraplength=self.W, justify="left")
        self.rules_lbl.pack(anchor="w", pady=(4, 0))
        nf = ttk.Frame(left)
        nf.pack(fill="x", pady=(8, 0))
        ttk.Label(nf, text="Notas del ajuste").pack(side="left")
        self.notes = tk.StringVar(value=cam.get("tuning_notes") or "")
        ttk.Entry(nf, textvariable=self.notes).pack(side="left", fill="x", expand=True, padx=4)
        bb = ttk.Frame(left)
        bb.pack(fill="x", pady=(8, 0))
        ttk.Button(bb, text="Cerrar", command=self.close).pack(side="right")
        ttk.Button(bb, text="✔ Aprobar y analizar", command=self.approve).pack(side="right", padx=4)
        self.save_btn = ttk.Button(bb, text="Guardar tiempos", command=self.save_times)
        self.save_btn.pack(side="right")

        right = ttk.Frame(frm, padding=(12, 0, 0, 0))
        right.grid(row=0, column=1, sticky="nsew")
        src = ttk.LabelFrame(right, text="Qué se analiza", padding=6)
        src.pack(fill="x")
        ttk.Button(src, text="En vivo", command=self.start_live).pack(side="left")
        ttk.Button(src, text="Probar con videos…", command=self.pick_videos).pack(side="left", padx=4)
        self.bar = ttk.Progressbar(src, length=110, mode="determinate", maximum=100)
        self.bar.pack(side="left", padx=4)
        tf = ttk.LabelFrame(right, text="Tiempos de las reglas (segundos)", padding=6)
        tf.pack(fill="x", pady=(6, 0))
        self.tv: dict[str, Any] = {}
        for key, lab, default in self.TIMES:
            row = ttk.Frame(tf)
            row.pack(fill="x")
            ttk.Label(row, text=lab, width=28).pack(side="left")
            var = tk.StringVar(value=str(default))
            ttk.Entry(row, textvariable=var, width=6).pack(side="left")
            self.tv[key] = var
        bt = ttk.Frame(tf)
        bt.pack(fill="x", pady=(4, 0))
        ttk.Button(bt, text="Probar estos tiempos", command=self.apply_times).pack(side="left")
        ttk.Button(bt, text="Zonas…", command=self.edit_zones).pack(side="left", padx=4)
        af = ttk.LabelFrame(right, text="Alertas de prueba (no le llegan a nadie) · clic = ir al momento", padding=6)
        af.pack(fill="both", expand=True, pady=(6, 0))
        self.alist = tk.Listbox(af, height=6, width=50)
        self.alist.pack(fill="both", expand=True)
        self.alist.bind("<<ListboxSelect>>", self.goto_alert)
        sf = ttk.LabelFrame(right, text="Cómo anda la cámara", padding=6)
        sf.pack(fill="x", pady=(6, 0))
        self.stats_txt = tk.Text(sf, height=10, width=50, wrap="word", relief="flat", bg="#f6f7f9", font=("Segoe UI", 9))
        self.stats_txt.pack(fill="x")
        self.load()
        self.top.after(150, self.tick)

    # ---------- preparar ----------
    def load(self) -> None:
        """Configuración de la cámara y detector (en otro hilo: armar el detector tarda la primera vez); después
        arranca en vivo."""
        def done(res):
            if self.closed:
                return
            self.info, self.detector = res
            self.analysis = self.info["camera"].get("analysis") or {}
            dev = getattr(self.detector, "device", None)
            self.detector_text = (f"Detector de personas: YOLO en {dev}." if dev else
                                  "Detector: movimiento (solo para pruebas; cambialo a YOLO en «Editar»).")
            if self.info["camera"].get("filter_model"):
                self.detector_text += " Con el modelo entrenado del cliente: cada alerta muestra su puntaje."
            if self.info.get("analyzed"):
                self.live_warn.configure(text="⚠ Esta cámara ya se analiza en las máquinas: las zonas y tiempos que "
                                              "guardes empiezan a regir ahí enseguida (alertas reales). Probar tiempos "
                                              "sin guardar no cambia nada.")
            self._fill_times()
            if self.tuner is None:
                self.start_live()

        def fail(exc):
            self.state_lbl.configure(text=f"No se pudo preparar el ajuste: {exc}")
        self.app.bg.run(lambda: self.agent.tune_setup(self.cam["id"]), done, fail)

    def _fill_times(self) -> None:
        rules = self.analysis.get("rules") or {}
        for key, _lab, default in self.TIMES:
            v = float(rules.get(key, default))
            self.tv[key].set(f"{v:g}")
        enabled = rules.get("enabled_rules")
        if self.analysis.get("metrics_only"):
            txt = "Esta cámara es «solo métricas»: no da alertas."
        else:
            names = [TYPE_NAMES.get(r, r) for r in (enabled if enabled is not None else
                                                     ("salida_sin_pago", "merodeo", "zona_restringida", "fuera_de_horario"))]
            prof = PROFILE_NAMES.get(self.analysis.get("profile") or "general", "").split(" (")[0]
            txt = f"Reglas de esta cámara (tipo {prof}): {', '.join(names).lower() or 'ninguna'}."
        self.rules_lbl.configure(text=txt + " El tipo de cámara se cambia en «Editar» o en «Nombres e imágenes». "
                                 + getattr(self, "detector_text", ""))

    def _times(self) -> dict[str, float] | None:
        try:
            times = {k: float(v.get().replace(",", ".")) for k, v in self.tv.items()}
        except ValueError:
            self.app.mb.showwarning("Vigía", "Los tiempos tienen que ser números.", parent=self.top)
            return None
        if any(v < 0 for v in times.values()):
            self.app.mb.showwarning("Vigía", "Los tiempos no pueden ser negativos.", parent=self.top)
            return None
        return times

    # ---------- qué se analiza ----------
    def _start(self, videos: list[str] | None = None) -> None:
        if self.info is None:
            return
        from .ajuste import Tuner
        if self.tuner is not None:
            self.tuner.stop()
        cfg = {**self.info["camera"], "analysis": self.analysis}
        try:
            self.tuner = Tuner(cfg, self.detector, videos)
        except ValueError as exc:
            self.tuner = None
            return self.app.mb.showerror("Vigía", f"Las zonas o reglas de la cámara tienen un error: {exc}", parent=self.top)
        self.index, self.playing, self._shown, self._alerts_key = None, False, None, None
        self.play_btn.configure(text="▶")
        self.bar["value"] = 0
        self.tuner.start()

    def start_live(self) -> None:
        self._start()

    def pick_videos(self) -> None:
        from tkinter import filedialog
        paths = filedialog.askopenfilenames(parent=self.top, title="Videos de esta cámara (se analizan uno tras otro)",
                                            filetypes=[("Videos", "*.mp4 *.avi *.mkv *.mov *.ts *.dav *.264 *.h264"),
                                                       ("Todos los archivos", "*.*")])
        if paths:
            self._start(list(paths))

    # ---------- tiempos y zonas ----------
    def apply_times(self) -> None:
        times = self._times()
        if times is None:
            return
        self.analysis = {**self.analysis, "rules": {**(self.analysis.get("rules") or {}), **times}}
        self.dirty = True
        self.save_btn.configure(text="Guardar tiempos *")
        if self.tuner is not None:
            t = self.tuner
            self.app.bg.run(lambda: t.set_times(times), lambda _: self._redraw(),
                            lambda e: self.app.mb.showerror("Vigía", str(e), parent=self.top))

    def _confirm_live(self) -> str | None:
        if not (self.info or {}).get("analyzed"):
            return None
        return (f"«{self.cam['name']}» ya se analiza en las máquinas de análisis: lo que guardes empieza a regir ahí "
                "enseguida y puede dar alertas reales a los empleados.\n\n¿Guardar igual?")

    def save_times(self, then: Callable[[], None] | None = None) -> None:
        times = self._times()
        if times is None or self.info is None:
            return
        warn = self._confirm_live()
        if then is None and warn and not self.app.mb.askyesno("Vigía", warn, parent=self.top):
            return
        raw = self.info["raw"]
        rules = {**(raw.get("rules") or {}), **times}
        cid = self.cam["id"]

        def work():
            self.app.api.put(f"/api/cameras/{cid}/analysis", {"zones": raw.get("zones") or [], "rules": rules})
            if (self.info or {}).get("tuning", "sin_ajustar") == "sin_ajustar":
                self.app.api.post(f"/api/cameras/{cid}/tuning", {"status": "en_ajuste"})
                self.info["tuning"] = "en_ajuste"

        def done(_):
            raw["rules"] = rules
            self.dirty = False
            self.save_btn.configure(text="Guardar tiempos")
            self.app.config_saved()
            if then:
                then()
        self.app.bg.run(work, done, lambda e: self.app.mb.showerror("Vigía", str(e), parent=self.top))

    def edit_zones(self) -> None:
        if self.info is None:
            return
        times = self._times() or {}
        img = self.tuner.image(self.index) if self.tuner is not None else None
        cam = {**self.cam, "zones": self.info["raw"].get("zones") or [],
               "rules": {**(self.info["raw"].get("rules") or {}), **times}}
        ZonesEditor(self.app, cam, image=img, on_saved=self.zones_saved, confirm=self._confirm_live())

    def zones_saved(self, _body: dict) -> None:
        """Se guardaron zonas (y tiempos) en el editor: traer la configuración nueva y recalcular."""
        def done(info):
            if self.closed:
                return
            self.info = {**info}
            self.analysis = info["camera"].get("analysis") or {}
            self.dirty = False
            self.save_btn.configure(text="Guardar tiempos")
            self._fill_times()
            if self.tuner is not None:
                t, analysis = self.tuner, self.analysis
                self.app.bg.run(lambda: t.set_analysis(analysis), lambda _: self._redraw(),
                                lambda e: self.app.mb.showerror("Vigía", str(e), parent=self.top))
        self.app.bg.run(lambda: self.agent.client.tune(self.cam["id"]), done,
                        lambda e: self.app.mb.showerror("Vigía", str(e), parent=self.top))

    # ---------- aprobar y cerrar ----------
    def approve(self) -> None:
        if self.info is None:
            return
        msg = (f"¿Aprobar «{self.cam['name']}» y mandarla a analizar en las máquinas de análisis?\n\nDesde ahí da alertas "
               "de verdad (les llegan a los empleados) y métricas.")
        if self.dirty:
            msg += "\n\nLos tiempos que estás probando se guardan."
        if not self.app.mb.askyesno("Vigía", msg, parent=self.top):
            return
        cid, notes = self.cam["id"], self.notes.get().strip()

        def send():
            def work():
                return self.app.api.post(f"/api/cameras/{cid}/tuning",
                                         {"status": "aprobada", "notes": notes, "analyze": True})

            def done(_):
                self.app.config_saved()
                self.app.mb.showinfo("Vigía", f"«{self.cam['name']}» quedó aprobada. Las máquinas de análisis la toman en "
                                              "menos de un minuto (si hay alguna en línea: mirá «Analizar» en Clientes).",
                                     parent=self.top)
                self.close(force=True)
            self.app.bg.run(work, done, lambda e: self.app.mb.showerror("Vigía", str(e), parent=self.top))
        if self.dirty:
            self.save_times(then=send)
        else:
            send()

    def close(self, force: bool = False) -> None:
        if not force and self.dirty and not self.app.mb.askyesno(
                "Vigía", "Hay tiempos probados sin guardar. ¿Cerrar igual?", parent=self.top):
            return
        self.closed = True
        if self.tuner is not None:
            self.tuner.stop()
        try:
            self.top.destroy()
        except Exception:  # noqa: BLE001
            pass

    # ---------- recorrer lo analizado ----------
    def toggle_play(self) -> None:
        t = self.tuner
        if t is None:
            return
        if self.playing:
            self.playing = False
            self.play_btn.configure(text="▶")
            return
        first, end = t.bounds()
        if end <= first:
            return
        if self.index is None or self.index >= end - 1:
            self.index = first
        self._play_f = float(self.index)
        self.playing = True
        self.play_btn.configure(text="⏸")

    def on_scale(self, value) -> None:
        if self._setting_scale:
            return
        self.index = int(float(value))
        self.playing = False
        self.play_btn.configure(text="▶")

    def follow(self) -> None:
        self.index, self.playing = None, False
        self.play_btn.configure(text="▶")

    def goto_alert(self, _ev=None) -> None:
        sel = self.alist.curselection()
        if sel and sel[0] < len(self._alert_rows):
            self.index, self.playing = self._alert_rows[sel[0]].index, False
            self.play_btn.configure(text="▶")

    def _redraw(self) -> None:
        self._shown, self._alerts_key = None, None

    def tick(self) -> None:
        if self.closed:
            return
        try:
            self._tick()
        except Exception:  # noqa: BLE001 - la ventana sigue aunque falle un dibujo
            log.exception("Ajuste: error en la ventana")
        self.top.after(100, self.tick)

    def _tick(self) -> None:
        t = self.tuner
        now = time.monotonic()
        dt, self._last_tick = now - self._last_tick, now
        if t is None:
            return
        first, end = t.bounds()
        if end > first:
            if self.playing:
                self._play_f = min(max(self._play_f, first) + dt * t.process_fps, end - 1)
                self.index = int(self._play_f)
                if self.index >= end - 1:
                    self.playing = False
                    self.play_btn.configure(text="▶")
            idx = end - 1 if self.index is None else min(max(self.index, first), end - 1)
            key = (id(t), idx, id(t.alerts), id(t.marks))
            if key != self._shown:
                img = t.render(idx)
                if img is not None:
                    self.photo = ppm_photo(self.app.tk, img, self.W, self.H)
                    self.canvas.delete("all")
                    self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
                self._shown = key
                s = t.sample_at(idx)
                if s is not None:
                    self.pos.configure(text=f"{t.clock_text(s)} · cuadro {idx - first + 1} de {end - first}"
                                            + ("" if self.index is not None else " · mostrando el último"))
            self._setting_scale = True
            self.scale.configure(from_=first, to=max(first, end - 1))
            self.scale.set(idx)
            self._setting_scale = False
            if t.live and not self.preview_sent:  # la cámara queda con imagen (para dibujar zonas y en la lista)
                self.preview_sent = True
                jpg = t.sample_at(end - 1).jpg
                self.app.bg.run(lambda: self.agent.client.preview(self.cam["id"], jpg), None, lambda e: None)
        self._show_state(t)
        self._show_alerts(t)
        if now - self._stats_at > 1.0:
            self._stats_at = now
            self._show_stats(t)

    def _show_state(self, t) -> None:
        if t.live:
            st = t.stats()
            txt = {"iniciando": "Conectando…", "conectando": "Conectando con la cámara…",
                   "en_linea": f"En vivo · se analizan {st['fps'] or 0:.1f} cuadros por segundo en esta PC",
                   "sin_conexion": f"Sin conexión: {t.error}", "error": f"Error: {t.error}",
                   "detenido": "Detenido"}.get(t.state, t.state)
        else:
            n = len(t.videos)
            if t.state == "analizando":
                txt = f"Analizando video {t.seg_now + 1} de {n}: {t.seg_names[t.seg_now]} · {t.progress * 100:.0f} %"
            elif t.state == "listo":
                txt = (f"Listo: {n} video(s) analizados ({t.speed or 0:.1f} veces más rápido que el video). Recorrelos con "
                       "la barra o ▶.") + (" Se cortó a los 30 minutos de video." if t.truncated else "")
            elif t.state == "error":
                txt = f"Error: {t.error}"
            else:
                txt = "Preparando…"
            self.bar["value"] = t.progress * 100
        self.state_lbl.configure(text=txt)

    def _show_alerts(self, t) -> None:
        alerts = t.alerts
        key = (id(t), id(alerts), len(alerts))
        if key == self._alerts_key:
            return
        self._alerts_key = key
        self._alert_rows = list(alerts)
        self.alist.delete(0, "end")
        for a in self._alert_rows:
            model = "" if a.score is None else f" · modelo {a.score:.2f}" + ("" if a.kept else " (la descartaría)")
            self.alist.insert("end", f"{t.clock_text(a)}   {TYPE_NAMES.get(a.type, a.type)}{model} · {a.description}")
        if not self._alert_rows:
            self.alist.insert("end", "Todavía ninguna." if t.state not in ("listo",) else "Ninguna con estas zonas y tiempos.")
            self._alert_rows = []

    def _show_stats(self, t) -> None:
        st = t.stats()
        lines = []
        if st["frames"]:
            lines.append(f"Analizado: {fmt_dur(st['seconds'])} · personas distintas: {st['persons']} · con gente en el "
                         f"{st['people_pct']:.0f} % de los cuadros")
            if st["detections"]:
                lines.append(f"Tamaño medio de las personas: {st['height_pct']:.0f} % del alto de la imagen · confianza "
                             f"del detector: {st['conf']:.2f} de 1")
            per_hour = f" (≈ {st['per_hour']:.1f} por hora)" if st.get("per_hour") is not None else ""
            dropped = f" · el modelo entrenado descartaría {st['dropped']}" if st.get("dropped") else ""
            lines.append(f"Alertas de prueba: {st['alerts']}{per_hour}{dropped}")
        if not t.live:
            for seg in t.segments():
                if seg["error"]:
                    res = seg["error"]
                elif not seg["alerts"]:
                    res = "sin alertas" if seg["done"] else "analizando…"
                else:
                    res = ", ".join(f"{TYPE_NAMES.get(a.type, a.type).lower()} {t.clock_text(a).split()[-1]}"
                                    for a in seg["alerts"][:4]) + ("…" if len(seg["alerts"]) > 4 else "")
                lines.append(f"• {seg['name']}: {res}")
        hints = t.hints()
        if hints:
            lines += [""] + [f"→ {h}" for h in hints]
        self.stats_txt.configure(state="normal")
        self.stats_txt.delete("1.0", "end")
        self.stats_txt.insert("1.0", "\n".join(lines) or "Esperando los primeros cuadros…")
        self.stats_txt.configure(state="disabled")


class NamesWindow:
    """«Nombres e imágenes»: todas las cámaras de un cliente con su imagen, para reconocerlas y ponerles un nombre que
    diga qué muestran (las de Hik-Connect vienen como «camera 3@E02380030»). YOLO dice qué se ve en cada una y sugiere
    tipo y nombre (core/escena.py): es una pista, no una clasificación. «Traer imágenes nuevas» abre de a una las
    cámaras que no analiza una máquina, toma un cuadro y lo deja como imagen de la cámara; las que analiza una máquina
    usan la imagen que ella sube (no se le pide otra transmisión al grabador)."""

    TW, TH = 288, 162
    COLS = 4

    def __init__(self, app: Oficina, client: dict):
        self.app, self.client = app, client
        tk, ttk = app.tk, app.ttk
        self.agent = app.agent_of() if app.agent_of else None
        self.cams = sorted((c for c in app.cameras if c["client_id"] == client["id"]),
                           key=lambda c: (c.get("hpp_serial") or c.get("host") or "", c.get("channel") or 0, c["name"]))
        self.closed = False
        self.busy = False
        self.cards: dict[int, dict] = {}
        self.top, frm = app._dialog(f"Nombres e imágenes · {client['name']}")
        self.top.resizable(True, True)
        self.top.protocol("WM_DELETE_WINDOW", self.close)
        frm.columnconfigure(0, weight=1)
        frm.rowconfigure(2, weight=1)
        ttk.Label(frm, wraplength=self.COLS * (self.TW + 24), justify="left", text=(
            "Mirá la imagen y ponele a cada cámara un nombre que diga qué muestra (por ejemplo «Caja 1», «Heladeras», "
            "«Puerta»). Debajo, lo que ve la IA y un tipo sugerido: es una pista, puede equivocarse. El nombre se cambia "
            "en Vigía (no en el grabador) y las alertas viejas pasan a mostrar el nombre nuevo.")).grid(
            row=0, column=0, columnspan=2, sticky="w")
        bar = ttk.Frame(frm)
        bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 4))
        self.fetch_btn = ttk.Button(bar, text="Traer imágenes nuevas", command=self.fetch_all)
        self.fetch_btn.pack(side="left")
        self.status = ttk.Label(bar, text="", foreground="#666")
        self.status.pack(side="left", padx=8)
        ttk.Button(bar, text="Cerrar", command=self.close).pack(side="right")
        ttk.Button(bar, text="Guardar nombres", command=self.save).pack(side="right", padx=4)
        outer = ttk.Frame(frm)
        outer.grid(row=2, column=0, columnspan=2, sticky="nsew")
        self.canvas = tk.Canvas(outer, width=self.COLS * (self.TW + 24), height=540, highlightthickness=0)
        sb = ttk.Scrollbar(outer, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.inner = ttk.Frame(self.canvas)
        self.canvas.create_window(0, 0, window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.top.bind_all(seq, self._wheel, add="+")
        for i, cam in enumerate(self.cams):
            self._card(i, cam)
        self._run([(c, "preview") for c in self.cams if c.get("has_preview")])
        missing = sum(1 for c in self.cams if not c.get("has_preview"))
        if missing:
            self.status.configure(text=f"{missing} cámara(s) sin imagen: «Traer imágenes nuevas» las abre de a una.")

    def _wheel(self, ev) -> None:
        if self.closed or not str(ev.widget).startswith(str(self.top)):
            return
        step = -1 if getattr(ev, "num", 0) == 4 or getattr(ev, "delta", 0) > 0 else 1
        self.canvas.yview_scroll(step * 2, "units")

    def _card(self, i: int, cam: dict) -> None:
        tk, ttk = self.app.tk, self.app.ttk
        f = ttk.Frame(self.inner, padding=6, relief="groove")
        f.grid(row=i // self.COLS, column=i % self.COLS, padx=4, pady=4, sticky="n")
        cv = tk.Canvas(f, width=self.TW, height=self.TH, bg="#111", highlightthickness=0)
        cv.pack()
        cv.create_text(self.TW // 2, self.TH // 2, text="Sin imagen", fill="#888")
        where = (f"Grabador {cam.get('hpp_serial') or '—'} · canal {cam.get('channel')}" if cam.get("brand") == "hikconnect"
                 else f"{cam.get('host') or '—'} · canal {cam.get('channel')}")
        ttk.Label(f, text=where, foreground="#666").pack(anchor="w", pady=(4, 0))
        name = tk.StringVar(value=cam["name"])
        ttk.Entry(f, textvariable=name, width=38).pack(fill="x")
        prof = tk.StringVar(value=PROFILE_NAMES.get(cam.get("profile") or "general", PROFILE_NAMES["general"]))
        ttk.Combobox(f, textvariable=prof, values=list(PROFILE_NAMES.values()), state="readonly",
                     width=36).pack(fill="x", pady=(2, 0))
        ia = ttk.Label(f, text="", foreground="#666", wraplength=self.TW, justify="left")
        ia.pack(anchor="w", pady=(2, 0))
        sug_row = ttk.Frame(f)
        sug_row.pack(fill="x")
        sug = ttk.Label(sug_row, text="", foreground="#0a7d3a", wraplength=self.TW - 60, justify="left")
        sug.pack(side="left")
        btn = ttk.Button(sug_row, text="Usar", width=6, command=lambda cid=cam["id"]: self.use(cid))
        self.cards[cam["id"]] = {"canvas": cv, "name": name, "prof": prof, "ia": ia, "sug": sug, "btn": btn,
                                 "photo": None, "suggestion": None, "orig": (cam["name"], prof.get())}

    # ---------- imágenes e IA (en otro hilo, de a una cámara) ----------
    def _post(self, fn, arg=None) -> None:
        self.app.bg.q.put((fn, arg))

    def _run(self, items: list) -> None:
        if not items or self.agent is None:
            self.busy = False
            self.fetch_btn.configure(state="normal")
            return
        self.busy = True
        self.fetch_btn.configure(state="disabled")
        threading.Thread(target=self._work, args=(items,), daemon=True).start()

    def _work(self, items: list) -> None:
        from core.escena import describe, scene_detector

        from .nombres import grab_frame
        try:
            det = scene_detector(self.agent.model_path)
        except Exception as exc:  # noqa: BLE001 - sin modelo: se muestran las imágenes sin lo que ve la IA
            log.warning("Nombres e imágenes: sin detector (%s)", exc)
            det = None
        plans = tuple(self.client.get("plans") or ())
        ok = 0
        for n, (cam, how) in enumerate(items, 1):
            if self.closed:
                return
            try:
                if how == "preview":
                    data = self.app.api.download(f"/api/cameras/{cam['id']}/preview")
                    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                    if img is None:
                        raise RuntimeError("la imagen guardada no se pudo leer")
                else:
                    self._post(lambda t: self.status.configure(text=t),
                               f"Imagen {n} de {len(items)}: abriendo «{cam['name']}»…")
                    img = grab_frame(how)
                    okj, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if okj:
                        self.agent.client.preview(cam["id"], jpg.tobytes())
                info = describe(img, det, plans) if det is not None else None
                self._post(self._show, (cam["id"], img, info))
                ok += 1
            except Exception as exc:  # noqa: BLE001 - se muestra en la tarjeta de esa cámara
                self._post(self._fail, (cam["id"], str(exc)))
        self._post(self._finished, (ok, len(items)))

    def _show(self, arg) -> None:
        cid, img, info = arg
        card = self.cards.get(cid)
        if self.closed or card is None:
            return
        from core.escena import summary
        card["photo"] = ppm_photo(self.app.tk, img, self.TW, self.TH)
        cv = card["canvas"]
        cv.delete("all")
        cv.create_image(self.TW // 2, self.TH // 2, image=card["photo"])
        if info is None:
            return
        card["ia"].configure(text="La IA ve: " + summary(info))
        sg = info.get("suggestion")
        card["suggestion"] = sg
        if sg:
            card["sug"].configure(text=f"Sugerencia: «{sg['name']}», tipo {PROFILE_NAMES[sg['profile']].split(' (')[0]} "
                                       f"({sg['why']})")
            card["btn"].pack(side="right")

    def _fail(self, arg) -> None:
        cid, msg = arg
        card = self.cards.get(cid)
        if self.closed or card is None:
            return
        card["ia"].configure(text=f"Sin imagen: {msg[:160]}")

    def _finished(self, arg) -> None:
        ok, total = arg
        self.busy = False
        if self.closed:
            return
        self.fetch_btn.configure(state="normal")
        self.status.configure(text=f"Listo: {ok} de {total} imágenes." + ("" if ok == total else
                                                                           " Las que faltan dicen por qué."))

    def fetch_all(self) -> None:
        """Imagen nueva de cada cámara: las que analiza una máquina, la última que subió; las demás, se abren de a una."""
        if self.busy or self.agent is None:
            return
        self.busy = True
        self.fetch_btn.configure(state="disabled")
        self.status.configure(text="Preparando…")
        cams = list(self.cams)

        def plan():
            views = {v["id"]: v for v in self.agent.client.view([c["id"] for c in cams])}
            items = []
            for c in cams:
                v = views.get(c["id"])
                if v is None:  # deshabilitada: la imagen que tenga
                    if c.get("has_preview"):
                        items.append((c, "preview"))
                elif v.get("analyzed") and c.get("has_preview"):
                    items.append((c, "preview"))  # la sube la máquina cada 5 minutos
                else:
                    items.append((c, v["connection"]))
            return items

        def fail(exc):
            self.busy = False
            self.fetch_btn.configure(state="normal")
            self.status.configure(text=f"No se pudo: {exc}")
        self.app.bg.run(plan, lambda items: (setattr(self, "busy", False), self._run(items)), fail)

    def use(self, cid: int) -> None:
        card = self.cards[cid]
        sg = card.get("suggestion")
        if sg:
            card["name"].set(sg["name"])
            card["prof"].set(PROFILE_NAMES[sg["profile"]])

    # ---------- guardar y cerrar ----------
    def _changes(self) -> list[dict] | None:
        rows = []
        for cid, card in self.cards.items():
            name, prof = card["name"].get().strip(), card["prof"].get()
            if not name:
                self.app.mb.showwarning("Vigía", "Hay una cámara sin nombre.", parent=self.top)
                return None
            if (name, prof) != card["orig"]:
                key = next((k for k, t in PROFILE_NAMES.items() if t == prof), None)
                rows.append({"id": cid, "name": name, "profile": key if prof != card["orig"][1] else None})
        return rows

    def save(self) -> None:
        rows = self._changes()
        if rows is None:
            return
        if not rows:
            return self.app.mb.showinfo("Vigía", "No cambiaste ningún nombre ni tipo.", parent=self.top)

        def done(res):
            for r in rows:
                card = self.cards[r["id"]]
                card["orig"] = (card["name"].get().strip(), card["prof"].get())
            self.app.config_saved()
            self.app.mb.showinfo("Vigía", f"Guardado: {res['renamed']} nombre(s) y {res['retyped']} tipo(s).",
                                 parent=self.top)
        self.app.bg.run(lambda: self.app.api.post("/api/cameras/names", {"cameras": rows}), done,
                        lambda e: self.app.mb.showerror("Vigía", str(e), parent=self.top))

    def close(self) -> None:
        rows = self._changes()
        if rows and not self.app.mb.askyesno("Vigía", f"Hay {len(rows)} cambio(s) sin guardar. ¿Cerrar igual?",
                                             parent=self.top):
            return
        self.closed = True
        try:
            self.top.destroy()
        except Exception:  # noqa: BLE001
            pass
