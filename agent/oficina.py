"""Gestión de la empresa desde la app de escritorio: clientes, cámaras, zonas, alertas, usuarios y PCs.

La web queda para los clientes (su resumen y sus alertas). Todo lo de la empresa se hace acá, contra
la misma API del servidor, con la sesión de un usuario (admin o empleado). Las llamadas a la red
corren en hilos aparte (`Bg`) y el resultado se aplica en el hilo de la ventana, así nada se congela.

Pestañas (las arma `Oficina.build` dentro del `ttk.Notebook` de la ventana):
- Clientes y cámaras: alta/edición/borrado de clientes y cámaras, probar conexión, zonas y reglas,
  métricas del día.
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
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np

log = logging.getLogger("vigia.oficina")

PLAN_NAMES = {"hurto": "Detección de hurtos", "comercio": "Métricas de comercio", "estacion": "Estación de servicio"}
ROLE_NAMES = {"admin": "Administrador", "entrenador": "Empleado", "cliente": "Cliente"}
MODE_NAMES = {"vivo": "En vivo 24 h", "alarmas_equipo": "Alarmas del propio grabador", "alarmas": "Alarmas por Hik-Partner Pro"}
BRAND_NAMES = {"hikvision": "Hikvision", "dahua": "Dahua"}
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
STATE_NAMES = {"escuchando": "Escuchando alarmas", "en_linea": "En línea", "sin_conexion": "Sin conexión",
               "reconectando": "Reconectando", "error": "Error", "iniciando": "Iniciando", "detenida": "Detenida"}


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
                 on_config_saved: Callable[[], None] | None = None):
        self.local_probe = local_probe
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
        self.tab_clients = ttk.Frame(notebook, padding=8)
        self.tab_alerts = ttk.Frame(notebook, padding=8)
        notebook.add(self.tab_clients, text="  Clientes y cámaras  ")
        notebook.add(self.tab_alerts, text="  Alertas  ")
        self._build_clients()
        self._build_alerts()
        if self.is_admin:
            self.tab_admin = ttk.Frame(notebook, padding=8)
            notebook.add(self.tab_admin, text="  Administración  ")
            self._build_admin()
        self.refresh_all()
        self.v.root.after(15000, self._auto_refresh)

    def _auto_refresh(self) -> None:
        """Estado de cámaras y PCs cada 15 s (sin tocar lo que se está editando)."""
        if not self.v.open:
            return

        def done(res):
            self.cameras, self.agents = res
            self._fill_clients(keep_info=True)
            if self.is_admin:
                self._fill_admin_pcs()
        self.bg.run(lambda: (self.api.get("/api/cameras"), self.api.get("/api/agents")), done, lambda e: None)
        self.v.root.after(15000, self._auto_refresh)

    def config_saved(self) -> None:
        """Se guardó una cámara, zonas o reglas: refrescar la lista y avisarle a la PC central."""
        self.refresh_all()
        if self.on_config_saved:
            self.on_config_saved()

    def refresh_all(self) -> None:
        def load():
            return (self.api.get("/api/clients"), self.api.get("/api/cameras"),
                    self.api.get("/api/agents"))

        def done(res):
            self.clients, self.cameras, self.agents = res
            self._fill_clients()
            self._fill_alert_filters()
            if self.is_admin:
                self._fill_admin_pcs()
        self.bg.run(load, done, self._net_fail)
        self.load_alerts()
        if self.is_admin:
            self.load_users()

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
        self.cl_tree = ttk.Treeview(left, columns=("planes",), show="tree headings", height=20, selectmode="browse")
        self.cl_tree.heading("#0", text="Cliente")
        self.cl_tree.heading("planes", text="Planes")
        self.cl_tree.column("#0", width=200)
        self.cl_tree.column("planes", width=170)
        self.cl_tree.pack(fill="y", expand=True)
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
        self.cam_tree = ttk.Treeview(right, columns=("nombre", "estado", "modo", "conexion", "pc", "zonas"),
                                     show="headings", height=12, selectmode="browse")
        for col, txt, w in (("nombre", "Cámara", 160), ("estado", "Estado", 130), ("modo", "Modo", 160),
                            ("conexion", "Grabador", 230), ("pc", "PC central", 140), ("zonas", "Zonas", 55)):
            self.cam_tree.heading(col, text=txt)
            self.cam_tree.column(col, width=w)
        self.cam_tree.pack(fill="x")
        self.cam_tree.bind("<Double-1>", lambda e: self.camera_dialog(self._sel_camera()))
        b2 = ttk.Frame(right)
        b2.pack(fill="x", pady=6)
        ttk.Button(b2, text="Agregar cámara", command=lambda: self.camera_dialog()).pack(side="left")
        ttk.Button(b2, text="Editar", command=lambda: self.camera_dialog(self._sel_camera())).pack(side="left", padx=4)
        ttk.Button(b2, text="Zonas y reglas", command=lambda: self.zones_dialog(self._sel_camera())).pack(side="left")
        ttk.Button(b2, text="Probar conexión", command=lambda: self.probe(self._sel_camera())).pack(side="left", padx=4)
        if self.is_admin:
            ttk.Button(b2, text="Eliminar", command=self.delete_camera).pack(side="left")
        ttk.Button(b2, text="Métricas del cliente", command=self.metrics_dialog).pack(side="right")
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
        for c in self.clients:
            n = sum(1 for x in self.cameras if x["client_id"] == c["id"])
            self.cl_tree.insert("", "end", iid=str(c["id"]), text=f"{c['name']}  ({n})",
                                values=(", ".join(PLAN_NAMES.get(p, p).split()[0] for p in c.get("plans") or []),))
        if prev and self.cl_tree.exists(prev[0]):
            self.cl_tree.selection_set(prev)
        elif self.clients:
            self.cl_tree.selection_set(str(self.clients[0]["id"]))
        self._fill_cameras(keep_info)

    def _cam_state(self, c: dict) -> str:
        st = c.get("status")
        if not c.get("enabled", True):
            return "Deshabilitada"
        if not st:
            return "Sin datos de la PC"
        if not st.get("agent_online"):
            return "PC desconectada"
        txt = STATE_NAMES.get(st.get("state"), st.get("state") or "—")
        return txt + (f" · {st['fps']:.0f} fps" if st.get("fps") and st.get("state") == "en_linea" else "")

    def _fill_cameras(self, keep_info: bool = False) -> None:
        c = self._sel_client()
        prev_cam = self.cam_tree.selection()
        self.cam_tree.delete(*self.cam_tree.get_children())
        if not c:
            self.cam_title.configure(text="Elegí un cliente")
            return
        self.cam_title.configure(text=f"Cámaras de {c['name']}")
        agents = {a["id"]: a["name"] for a in self.agents}
        for cam in [x for x in self.cameras if x["client_id"] == c["id"]]:
            conn = f"{BRAND_NAMES.get(cam.get('brand'), '')} · {cam.get('host', '')}:{cam.get('rtsp_port')}/{cam.get('http_port')} · canal {cam.get('channel')}"
            self.cam_tree.insert("", "end", iid=str(cam["id"]), values=(
                cam["name"], self._cam_state(cam), MODE_NAMES.get(cam.get("mode"), cam.get("mode")), conn,
                agents.get(cam.get("agent_id"), "sin PC"), len(cam.get("zones") or [])))
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
            f"Grabador: {BRAND_NAMES.get(cam.get('brand'), '')} {cam.get('host')} · RTSP {cam.get('rtsp_port')} · "
            f"HTTP {cam.get('http_port')} · canal {cam.get('channel')} · usuario {cam.get('username')}\n"
            f"Zonas: {zones}\n\n"
            "Enlace para probar en VLC (con la clave del grabador):\n"
            f"rtsp://{cam.get('username')}:CLAVE@{cam.get('host')}:{cam.get('rtsp_port')}/"
            + (f"cam/realmonitor?channel={cam.get('channel')}&subtype=1" if cam.get("brand") == "dahua"
               else f"Streaming/Channels/{cam.get('channel')}02"))

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

        def save() -> None:
            body = {k: (var.get().strip() or None) for k, var in v.items()}
            body["timezone"] = body["timezone"] or "America/Montevideo"
            body["webhook_url"] = c.get("webhook_url")
            body["plans"] = [p for p, var in plans.items() if var.get()]
            if not body["name"]:
                return self.mb.showwarning("Vigía", "Falta el nombre", parent=top)

            def done(_):
                top.destroy()
                self.refresh_all()
            self.bg.run(lambda: (self.api.put(f"/api/clients/{c['id']}", body) if c.get("id")
                                 else self.api.post("/api/clients", body)), done, lambda e: self.mb.showerror("Vigía", str(e), parent=top))
        bb = ttk.Frame(frm)
        bb.grid(row=11, column=0, columnspan=2, sticky="e")
        ttk.Button(bb, text="Cancelar", command=top.destroy).pack(side="right")
        ttk.Button(bb, text="Guardar", command=save).pack(side="right", padx=6)

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
        tk, ttk = self.tk, self.ttk
        client = self._sel_client()
        if cam is None and client is None:
            return self.mb.showinfo("Vigía", "Elegí primero un cliente a la izquierda.")
        cam = cam or {}
        top, frm = self._dialog("Editar cámara" if cam.get("id") else f"Nueva cámara · {client['name']}", 560)
        centrals = [a for a in self.agents if a.get("central")]
        agent_names = [a["name"] for a in centrals]
        cur_agent = next((a["name"] for a in centrals if a["id"] == cam.get("agent_id")),
                         agent_names[0] if agent_names else "")
        v = {
            "name": tk.StringVar(value=cam.get("name", "")),
            "brand": tk.StringVar(value=BRAND_NAMES.get(cam.get("brand", "hikvision"))),
            "mode": tk.StringVar(value=MODE_NAMES.get(cam.get("mode", "vivo"))),
            "host": tk.StringVar(value=cam.get("host", "")),
            "rtsp_port": tk.StringVar(value=str(cam.get("rtsp_port", 554))),
            "http_port": tk.StringVar(value=str(cam.get("http_port", 80))),
            "channel": tk.StringVar(value=str(cam.get("channel", 1))),
            "username": tk.StringVar(value=cam.get("username", "admin")),
            "password": tk.StringVar(value=""),
            "agent": tk.StringVar(value=cur_agent),
            "fps": tk.StringVar(value=str(cam.get("process_fps", 5))),
        }
        enabled = tk.BooleanVar(value=cam.get("enabled", True))
        nvr = tk.BooleanVar(value=cam.get("nvr_clips", True))
        r = 0
        for key, lab, w in (("name", "Nombre", ttk.Entry(frm, textvariable=v["name"])),
                            ("brand", "Marca", ttk.Combobox(frm, textvariable=v["brand"], values=list(BRAND_NAMES.values()), state="readonly")),
                            ("mode", "Modo", ttk.Combobox(frm, textvariable=v["mode"], values=list(MODE_NAMES.values()), state="readonly")),
                            ("host", "IP o DDNS del grabador", ttk.Entry(frm, textvariable=v["host"])),
                            ("rtsp_port", "Puerto RTSP", ttk.Entry(frm, textvariable=v["rtsp_port"])),
                            ("http_port", "Puerto HTTP", ttk.Entry(frm, textvariable=v["http_port"])),
                            ("channel", "Canal", ttk.Entry(frm, textvariable=v["channel"])),
                            ("username", "Usuario del grabador", ttk.Entry(frm, textvariable=v["username"])),
                            ("password", "Clave del grabador" + (" (vacía = sin cambios)" if cam.get("has_password") else ""),
                             ttk.Entry(frm, textvariable=v["password"], show="•")),
                            ("agent", "PC central que la analiza", ttk.Combobox(frm, textvariable=v["agent"], values=agent_names, state="readonly")),
                            ("fps", "Cuadros por segundo", ttk.Entry(frm, textvariable=v["fps"]))):
            self._row(frm, r, lab, w)
            r += 1
        ttk.Label(frm, text="Misma red que la PC central: IP de adentro (192.168.x.x) y puertos 554/80.\n"
                            "Otra red: IP pública o DDNS y los puertos externos del router (ej. 38554/38080).",
                  foreground="#666", justify="left").grid(row=r, column=0, columnspan=2, sticky="w", pady=(4, 6))
        r += 1
        cfg = cam.get("alarm_cfg") or {}
        al = {k: tk.BooleanVar(value=k in (cfg.get("event_types") or ["VMD", "fielddetection", "linedetection"])) for k in ALARM_TYPES}
        af = ttk.LabelFrame(frm, text="Alarmas del grabador que se analizan (modos de alarma)", padding=6)
        af.grid(row=r, column=0, columnspan=2, sticky="ew", pady=4)
        for i, (k, t) in enumerate(ALARM_TYPES.items()):
            ttk.Checkbutton(af, text=t, variable=al[k]).grid(row=i // 3, column=i % 3, sticky="w", padx=4)
        r += 1
        ttk.Checkbutton(frm, text="Bajar el clip de la grabación del grabador (si tiene disco)", variable=nvr).grid(row=r, column=0, columnspan=2, sticky="w")
        r += 1
        ttk.Checkbutton(frm, text="Habilitada", variable=enabled).grid(row=r, column=0, columnspan=2, sticky="w")
        r += 1

        def save() -> None:
            inv = lambda d, val: next((k for k, t in d.items() if t == val), None)  # noqa: E731
            try:
                body = {
                    "client_id": cam.get("client_id") or client["id"], "name": v["name"].get().strip(),
                    "brand": inv(BRAND_NAMES, v["brand"].get()) or "hikvision",
                    "mode": inv(MODE_NAMES, v["mode"].get()) or "vivo", "host": v["host"].get().strip(),
                    "rtsp_port": int(v["rtsp_port"].get()), "http_port": int(v["http_port"].get()),
                    "channel": int(v["channel"].get()), "username": v["username"].get().strip(),
                    "password": v["password"].get() or None,
                    "agent_id": next((a["id"] for a in centrals if a["name"] == v["agent"].get()), None),
                    "process_fps": float(v["fps"].get()), "enabled": enabled.get(), "nvr_clips": nvr.get(),
                    "detect_stream": cam.get("detect_stream", "sub"), "detector": cam.get("detector", "yolo"),
                    "playback_time": cam.get("playback_time", "local"), "hpp_serial": cam.get("hpp_serial"),
                    "alarm_cfg": {**cfg, "event_types": [k for k, var in al.items() if var.get()]},
                }
            except ValueError:
                return self.mb.showwarning("Vigía", "Puertos, canal y cuadros por segundo tienen que ser números.", parent=top)
            if not body["name"] or not body["host"]:
                return self.mb.showwarning("Vigía", "Faltan el nombre o la IP del grabador.", parent=top)
            if body["agent_id"] is None:
                self.mb.showwarning("Vigía", "No hay ninguna PC central elegida: la cámara no se va a analizar "
                                             "hasta que le asignes una.", parent=top)

            def done(_):
                top.destroy()
                self.config_saved()
            self.bg.run(lambda: (self.api.put(f"/api/cameras/{cam['id']}", body) if cam.get("id")
                                 else self.api.post("/api/cameras", body)), done,
                        lambda e: self.mb.showerror("Vigía", str(e), parent=top))
        bb = ttk.Frame(frm)
        bb.grid(row=r, column=0, columnspan=2, sticky="e", pady=(8, 0))
        ttk.Button(bb, text="Cancelar", command=top.destroy).pack(side="right")
        ttk.Button(bb, text="Guardar", command=save).pack(side="right", padx=6)

    def delete_camera(self) -> None:
        cam = self._sel_camera()
        if cam and self.mb.askyesno("Vigía", f"¿Eliminar la cámara «{cam['name']}»?\nSe borran también sus métricas. "
                                              "Si tiene alertas, no se puede: deshabilitala."):
            self.bg.run(lambda: self.api.delete(f"/api/cameras/{cam['id']}"), lambda _: self.refresh_all(), self.err)

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

    # ---------- zonas ----------
    def zones_dialog(self, cam: dict | None) -> None:
        if not cam:
            return self.mb.showinfo("Vigía", "Elegí una cámara.")
        ZonesEditor(self, cam)

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
        left = ttk.Frame(f)
        left.pack(side="left", fill="both", expand=True)
        ttk.Label(left, text="Usuarios", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Label(left, text="Administradores y empleados entran a esta app. Los clientes entran a la web y ven solo su local.",
                  foreground="#666").pack(anchor="w")
        self.us_tree = ttk.Treeview(left, columns=("rol", "cliente", "activo"), show="tree headings", height=14,
                                    selectmode="browse")
        self.us_tree.heading("#0", text="Usuario")
        for col, txt, w in (("rol", "Rol", 120), ("cliente", "Cliente", 160), ("activo", "Activo", 60)):
            self.us_tree.heading(col, text=txt)
            self.us_tree.column(col, width=w)
        self.us_tree.pack(fill="both", expand=True)
        b = ttk.Frame(left)
        b.pack(fill="x", pady=6)
        ttk.Button(b, text="Nuevo usuario", command=lambda: self.user_dialog()).pack(side="left")
        ttk.Button(b, text="Editar", command=lambda: self.user_dialog(self._sel_user())).pack(side="left", padx=4)

        right = ttk.Frame(f, padding=(14, 0, 0, 0))
        right.pack(side="left", fill="both", expand=True)
        ttk.Label(right, text="PCs centrales", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Label(right, text="Se vinculan solas al entrar con «Esta PC analiza las cámaras».", foreground="#666").pack(anchor="w")
        self.pc_tree = ttk.Treeview(right, columns=("estado", "visto"), show="tree headings", height=14, selectmode="browse")
        self.pc_tree.heading("#0", text="PC")
        self.pc_tree.heading("estado", text="Estado")
        self.pc_tree.heading("visto", text="Última conexión")
        self.pc_tree.pack(fill="both", expand=True)
        b2 = ttk.Frame(right)
        b2.pack(fill="x", pady=6)
        ttk.Button(b2, text="Eliminar PC", command=self.delete_pc).pack(side="left")
        self.users: list[dict] = []

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

            def done(_):
                top.destroy()
                self.load_users()
            self.bg.run(lambda: (self.api.put(f"/api/users/{u['id']}", body) if u.get("id")
                                 else self.api.post("/api/users", body)), done,
                        lambda e: self.mb.showerror("Vigía", str(e), parent=top))
        bb = ttk.Frame(frm)
        bb.grid(row=5, column=0, columnspan=2, sticky="e", pady=(8, 0))
        ttk.Button(bb, text="Cancelar", command=top.destroy).pack(side="right")
        ttk.Button(bb, text="Guardar", command=save).pack(side="right", padx=6)

    def _fill_admin_pcs(self) -> None:
        self.pc_tree.delete(*self.pc_tree.get_children())
        for a in self.agents:
            if a.get("central"):
                self.pc_tree.insert("", "end", iid=str(a["id"]), text=a["name"],
                                    values=("En línea" if a.get("online") else "Desconectada", fmt_date(a.get("last_seen"))))

    def delete_pc(self) -> None:
        sel = self.pc_tree.selection()
        a = next((x for x in self.agents if str(x["id"]) == (sel[0] if sel else "")), None)
        if a and self.mb.askyesno("Vigía", f"¿Eliminar la PC «{a['name']}»? Sus cámaras quedan sin PC asignada."):
            self.bg.run(lambda: self.api.delete(f"/api/agents/{a['id']}"), lambda _: self.refresh_all(), self.err)


class ZonesEditor:
    """Editor de zonas: clics sobre la imagen de la cámara para marcar vértices, doble clic para cerrar."""

    W, H = 720, 405

    def __init__(self, app: Oficina, cam: dict):
        self.app, self.cam = app, cam
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
        body = {"zones": self.zones, "rules": rules}

        def done(_):
            self.top.destroy()
            self.app.config_saved()
            self.app.v.log_handler.q.put_nowait(f"{datetime.now():%H:%M:%S}  Zonas de «{self.cam['name']}» guardadas: "
                                                 "la PC central las aplica en segundos")
        self.app.bg.run(lambda: self.app.api.put(f"/api/cameras/{self.cam['id']}/analysis", body), done,
                        lambda e: self.app.mb.showerror("Vigía", str(e), parent=self.top))
