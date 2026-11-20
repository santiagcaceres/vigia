"""Pestaña «Inicio» de la app: el resumen de todo y el aviso cuando una cámara deja de andar.

Pedido del dueño (20 nov): «que la pantalla de inicio sea como un resumen, con las alertas por un lado, clientes en otra,
y otro que salen las cámaras que se desconectan o pasa algo; si una cámara deja de estar en verde, que mande un aviso».
Elegido con él: en la app de escritorio (la primera pestaña al abrir) y el aviso **solo en la app** (un cartel rojo, un
sonido y la barra de tareas que parpadea si la ventana no está al frente); nada por Telegram ni por mail. Solo avisa si
alguien tiene la app abierta: minimizada o en otra pestaña también.

- **Qué se mira:** las cámaras que se analizan. Las demás no tienen a nadie mirándolas (no hay «verde» que perder).
- **Verde:** analizando, escuchando alarmas o con el local cerrado (no se mira a propósito).
- **Problema** (rojo): sin conexión, reconectando, error, detenida, la máquina que la analiza no responde, en línea pero
  sin imagen nueva hace 2 min, o marcada para analizar y ninguna máquina la tomó.
- **Atención** (amarillo, sin cartel): la máquina no da abasto y saltea cuadros (los clips salen «como fotos»).
- **Cuándo avisa:** si el problema dura 2 minutos (5 si la cámara recién arranca): Hik-Connect y los routers cortan y
  vuelven solos todo el tiempo. Y avisa que «volvió» cuando anda de nuevo 1 minuto seguido. Lo que ya estaba mal al
  abrir la app no avisa (se ve en la lista). Si se cae una máquina, un solo aviso por todas sus cámaras.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import estilo
from .oficina import STATE_NAMES, TYPE_NAMES, fmt_date

log = logging.getLogger("vigia.inicio")

OK, CERRADO, ATENCION, PROBLEMA, NO_SE_MIRA = "ok", "cerrado", "atencion", "problema", "no_se_mira"
GRACE_S = 120  # una caída tiene que durar esto para avisar (las reconexiones de un momento no cuentan)
START_GRACE_S = 300  # una cámara que recién se puso a analizar tiene más tiempo para arrancar
RECOVER_S = 60  # anda de nuevo este rato seguido = volvió (una cámara que corta y vuelve cada rato sigue «caída»)
STALE_FRAME_S = 120  # «en línea» pero sin cuadros nuevos hace esto = trabada
SLOW_SKIP_PCT = 25  # la máquina saltea 1 de cada 4 cuadros o más: no da abasto
MAX_LOG = 300
MANY = 3  # más que esto a la vez: un solo cartel que las cuenta
HOME_TAB = "  Inicio  "
ALERTS_LIMIT = 60


@dataclass
class Health:
    level: str
    text: str  # «Sin conexión», en pocas palabras (la lista)
    detail: str = ""  # el motivo entero (lo que dijo la máquina)
    age_s: float | None = None  # hace cuánto está así, si se sabe
    group: str | None = None  # «maquina:7»: el problema es la máquina, no la cámara (un solo aviso para todas)
    grace_s: float = GRACE_S


@dataclass
class Notice:
    kind: str  # "cayo" | "volvio"
    text: str
    camera_ids: list[int]


def _num(v) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def seconds_since(iso: str | None, now: float | None = None) -> float | None:
    """Segundos desde una fecha del servidor (UTC, «2026-11-20T10:32:00Z»)."""
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return max(0.0, (now if now is not None else time.time()) - dt.timestamp())


def duration(s: float | None) -> str:
    """«40 s», «12 min», «3 h 05 min», «2 días»."""
    if s is None:
        return "—"
    s = max(0, int(s))
    if s < 60:
        return f"{s} s"
    if s < 3600:
        return f"{s // 60} min"
    if s < 86400:
        return f"{s // 3600} h {s % 3600 // 60:02d} min"
    d = s // 86400
    return f"{d} día{'s' if d != 1 else ''}"


def camera_health(cam: dict, agents: dict[int, dict] | None = None, now: float | None = None) -> Health:
    """Cómo está una cámara, con el mismo criterio que la columna «Estado» de «Clientes» (`Oficina._cam_state`)."""
    agents = agents or {}
    st = cam.get("status") or None
    if not cam.get("enabled", True):
        return Health(NO_SE_MIRA, "Deshabilitada")
    if not cam.get("analyzed") and not (st and cam.get("agent_id")):
        return Health(NO_SE_MIRA, "No se analiza")
    if not st:
        if not cam.get("agent_id"):
            return Health(PROBLEMA, "Esperando una máquina de análisis",
                          "Está marcada para analizar pero ninguna máquina de análisis la tomó: puede que no haya "
                          "ninguna en línea.", grace_s=START_GRACE_S)
        return Health(PROBLEMA, "Empezando a analizar…", "La máquina todavía no dijo cómo le va.",
                      grace_s=START_GRACE_S)
    if not st.get("agent_online"):
        a = agents.get(cam.get("agent_id")) or {}
        name = cam.get("analyzer_name") or a.get("name") or "de análisis"
        age = seconds_since(a.get("last_seen"), now)
        return Health(PROBLEMA, f"La máquina «{name}» no responde",
                      "La máquina que la analiza dejó de comunicarse con el servidor"
                      + (f" hace {duration(age)}" if age is not None else "") + ": sus cámaras no se analizan.",
                      age_s=age, group=f"maquina:{cam.get('agent_id')}")
    state = st.get("state")
    age = _num(st.get("state_age_s"))
    if state == "fuera_de_horario":
        return Health(CERRADO, "Local cerrado (no se mira)")
    if state == "escuchando":
        return Health(OK, "Escuchando alarmas")
    if state == "en_linea":
        stale = _num(st.get("last_frame_age_s"))
        if stale is not None and stale >= STALE_FRAME_S:
            return Health(PROBLEMA, f"Sin imagen hace {duration(stale)}",
                          "El video está abierto pero no llegan cuadros nuevos (el grabador o la red se trabaron).",
                          age_s=stale)
        skipped = _num(st.get("skipped_pct"))
        if skipped is not None and skipped >= SLOW_SKIP_PCT:
            return Health(ATENCION, f"No da abasto (saltea {skipped:.0f} %)",
                          "La máquina analiza menos cuadros de los que debería: los clips salen entrecortados y los gestos "
                          "aciertan menos. Hace falta otra máquina o analizar menos cámaras.")
        fps = _num(st.get("fps"))
        return Health(OK, "Analizando" + (f" · {fps:.0f} cuadros/s" if fps else ""))
    if state == "iniciando":
        return Health(PROBLEMA, "Conectando…", st.get("error") or "", age_s=age, grace_s=START_GRACE_S)
    return Health(PROBLEMA, STATE_NAMES.get(state, state or "Sin estado"), st.get("error") or "", age_s=age)


def camera_label(cam: dict, client_names: dict[int, str]) -> str:
    client = client_names.get(cam.get("client_id"))
    return f"{client} · {cam.get('name')}" if client else str(cam.get("name"))


class HealthWatch:
    """Lleva la cuenta de cada cámara entre una consulta y la siguiente y dice cuándo avisar (`update`). Sin ventana:
    se prueba sola."""

    def __init__(self) -> None:
        self.track: dict[int, dict] = {}
        self.first = True  # la primera vez solo se anota (lo que ya estaba mal al abrir la app no avisa)
        self.log: list[tuple[float, str, str]] = []  # (cuándo, "cayo"/"volvio", texto)
        self.health: dict[int, Health] = {}
        self.labels: dict[int, str] = {}

    def update(self, cameras: list[dict], agents: list[dict] | None, client_names: dict[int, str],
               now: float | None = None) -> list[Notice]:
        now = time.time() if now is None else now
        by_id = {a.get("id"): a for a in agents or []}
        fell: list[tuple[int, Health]] = []
        back: list[tuple[int, Health]] = []
        seen: set[int] = set()
        self.health, self.labels = {}, {}
        for cam in cameras:
            cid = cam.get("id")
            if type(cid) is not int:
                continue
            h = camera_health(cam, by_id, now)
            if h.level == NO_SE_MIRA:
                self.track.pop(cid, None)  # (dejó de analizarse: no es una caída)
                continue
            seen.add(cid)
            self.health[cid], self.labels[cid] = h, camera_label(cam, client_names)
            t = self.track.get(cid)
            if t is None:
                t = self.track[cid] = {"bad_since": None, "ok_since": None, "notified": False, "pre": self.first}
            if h.level == PROBLEMA:
                t["ok_since"] = None
                if t["bad_since"] is None:
                    t["bad_since"] = now - h.age_s if h.age_s is not None else now
                if not t["notified"] and not t["pre"] and now - t["bad_since"] >= h.grace_s:
                    t["notified"] = True
                    fell.append((cid, h))
            elif t["bad_since"] is None:
                t["pre"] = False  # anda: lo que le pase de ahora en más es nuevo
            else:
                if t["ok_since"] is None:
                    t["ok_since"] = now
                if now - t["ok_since"] >= RECOVER_S:  # volvió de verdad (un rato seguido, no un momento)
                    if t["notified"]:
                        back.append((cid, h))
                    t.update(bad_since=None, ok_since=None, notified=False, pre=False)
        for cid in [c for c in self.track if c not in seen]:  # borrada o ya no se analiza
            del self.track[cid]
        self.first = False
        return self._notices(fell, back, now)

    def since(self, cid: int) -> float | None:
        t = self.track.get(cid)
        return t["bad_since"] if t else None

    def was_bad_at_start(self, cid: int) -> bool:
        t = self.track.get(cid)
        return bool(t and t["pre"])

    def _notices(self, fell, back, now) -> list[Notice]:
        out: list[Notice] = []
        groups: dict[str, list[tuple[int, Health]]] = {}
        single: list[tuple[int, Health]] = []
        for cid, h in fell:
            (groups.setdefault(h.group, []) if h.group else single).append((cid, h))
            self._log(now, "cayo", f"{self.labels[cid]}: {h.text}")
        for items in groups.values():
            n = len(items)
            out.append(Notice("cayo", f"{items[0][1].text}: {n} cámara{'s' if n != 1 else ''} sin analizar",
                              [c for c, _ in items]))
        if len(single) > MANY:
            names = ", ".join(self.labels[c] for c, _ in single[:MANY]) + "…"
            out.append(Notice("cayo", f"{len(single)} cámaras dejaron de andar: {names}", [c for c, _ in single]))
        else:
            out.extend(Notice("cayo", f"{self.labels[c]}: {h.text}", [c]) for c, h in single)
        for cid, _h in back:
            self._log(now, "volvio", f"{self.labels[cid]}: volvió a andar")
        if len(back) > MANY:
            out.append(Notice("volvio", f"{len(back)} cámaras volvieron a andar", [c for c, _ in back]))
        else:
            out.extend(Notice("volvio", f"{self.labels[c]} volvió a andar", [c]) for c, _ in back)
        return out

    def _log(self, at: float, kind: str, text: str) -> None:
        self.log.append((at, kind, text))
        del self.log[:-MAX_LOG]


def hint(h: Health, is_admin: bool) -> str:
    """Qué hacer, en una línea."""
    d = (h.detail or "").lower()
    if h.group:
        return ("Pestaña «Máquinas»: reiniciala (al arrancar se pone al día)." if is_admin
                else "Avisale a un administrador: hay que reiniciar la máquina.")
    if h.text.startswith("Esperando una máquina"):
        return ("Prendé una máquina en la pestaña «Máquinas»." if is_admin
                else "Avisale a un administrador: no hay máquinas de análisis en línea.")
    if h.level == ATENCION:
        return "Sacá cámaras de análisis o sumá otra máquina."
    if "no se encuentra" in d:
        return "El nombre (DDNS) del local no existe o venció: revisalo con el técnico."
    if "no contesta" in d or "rechaza" in d:
        return "Revisá con el local el router, la IP fija del grabador y el DDNS («Probar conexión» dice el motivo)."
    if "usuario o clave" in d or "bloque" in d:
        return "Usuario o clave del grabador rechazados (o bloqueado por intentos): revisalos en «Editar»."
    return "«Probar conexión» dice el motivo exacto."


def flash_window(root) -> None:
    """Hace parpadear el botón de la app en la barra de tareas de Windows hasta que se la mire (en otro sistema, nada)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import wintypes

        class FLASHWINFO(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.UINT), ("hwnd", wintypes.HWND), ("dwFlags", wintypes.DWORD),
                        ("uCount", wintypes.UINT), ("dwTimeout", wintypes.DWORD)]
        info = FLASHWINFO(ctypes.sizeof(FLASHWINFO), int(root.wm_frame(), 16), 0x3 | 0xC, 0, 0)  # ALL | TIMERNOFG
        ctypes.windll.user32.FlashWindowEx(ctypes.byref(info))
    except Exception as exc:  # noqa: BLE001 - es un adorno: si falla, queda el cartel y el sonido
        log.debug("No se pudo hacer parpadear la ventana: %s", exc)


class Inicio:
    """La pestaña. `app` es la `Oficina` (sus listas de clientes, cámaras y máquinas, y cómo ir a cada cosa)."""

    def __init__(self, app, parent) -> None:
        self.app, self.parent = app, parent
        self.tk, self.ttk = app.tk, app.ttk
        self.watch = HealthWatch()
        self.clock = time.time  # (las pruebas lo adelantan)
        self.alerts: list[dict] = []
        self.alerts_total = 0
        self.last_ok: float | None = None
        self.net_down_since: float | None = None
        self.prefs = self._load_prefs()
        self._build()

    # ---------- preferencias (en la carpeta de datos de la app) ----------
    def _prefs_path(self) -> Path:
        agent = self.app.agent_of() if getattr(self.app, "agent_of", None) else None
        base = Path(agent.data_dir) if agent is not None and getattr(agent, "data_dir", None) else \
            Path.home() / "AnaliticaCritica"
        return base / "inicio.json"

    def _load_prefs(self) -> dict:
        try:
            return json.loads(self._prefs_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save_prefs(self) -> None:
        self.prefs["avisos"] = bool(self.notify_var.get())
        try:
            p = self._prefs_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(self.prefs), encoding="utf-8")
        except OSError as exc:
            log.debug("No se pudo guardar inicio.json: %s", exc)

    # ---------- armado ----------
    def _build(self) -> None:
        tk, ttk, p = self.tk, self.ttk, self.parent
        fam = estilo.familia(self.app.v.root)
        top = ttk.Frame(p)
        top.pack(side="top", fill="x")
        ttk.Label(top, text="Resumen", style="Titulo.TLabel").pack(side="left")
        self.updated = ttk.Label(top, text="", style="Suave.TLabel")
        self.updated.pack(side="left", padx=12)
        self.notify_var = tk.BooleanVar(value=self.prefs.get("avisos", True))
        ttk.Checkbutton(top, text="Avisarme con un cartel y un sonido cuando una cámara deja de andar",
                        variable=self.notify_var, command=self._save_prefs).pack(side="right")

        tiles = ttk.Frame(p)
        tiles.pack(side="top", fill="x", pady=(8, 8))
        self.tiles: dict[str, tuple] = {}
        spec = [("unreviewed", "Alertas sin revisar", self._go_alerts), ("today", "Alertas hoy", self._go_alerts),
                ("ok", "Cámaras andando", self._go_problems), ("bad", "Cámaras con problemas", self._go_problems)]
        if self.app.is_admin:
            spec.append(("machines", "Máquinas en línea", self._go_machines))
        for key, caption, go in spec:
            f = tk.Frame(tiles, bg=estilo.BLANCO, highlightthickness=1, highlightbackground=estilo.BORDE, padx=14,
                         pady=6, cursor="hand2")
            num = tk.Label(f, text="—", font=(fam, 20, "bold"), bg=estilo.BLANCO, fg=estilo.MARCA, cursor="hand2")
            cap = tk.Label(f, text=caption, bg=estilo.BLANCO, fg=estilo.SUAVE, cursor="hand2")
            num.pack(anchor="w")
            cap.pack(anchor="w")
            f.pack(side="left", fill="x", expand=True, padx=(0, 10))
            for w in (f, num, cap):
                w.bind("<Button-1>", lambda _e, g=go: g())
            self.tiles[key] = (f, num, cap)

        body = ttk.Frame(p)
        body.pack(side="top", fill="both", expand=True)
        for i, w in enumerate((5, 4, 6)):  # (sin «uniform»: obligaría a todas a medir lo de la más ancha)
            body.columnconfigure(i, weight=w)
        body.rowconfigure(0, weight=1)

        # 1 · alertas sin revisar
        c1 = ttk.Frame(body)
        c1.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.al_head = ttk.Label(c1, text="Alertas sin revisar", style="Subtitulo.TLabel")
        self.al_head.pack(side="top", anchor="w")
        b1 = ttk.Frame(c1)
        b1.pack(side="bottom", fill="x", pady=(6, 0))
        ttk.Button(b1, text="Abrir en «Alertas»", command=self._open_alert).pack(side="left")
        ttk.Label(b1, text="o doble clic", style="Suave.TLabel").pack(side="left", padx=6)
        self.al_tree = self._tree(c1, (("hora", "Cuándo", 84), ("cliente", "Cliente", 96), ("camara", "Cámara", 96),
                                       ("que", "Qué", 120)))
        self.al_tree.bind("<Double-1>", lambda _e: self._open_alert())
        self.al_tree.bind("<Return>", lambda _e: self._open_alert())

        # 2 · clientes
        c2 = ttk.Frame(body)
        c2.grid(row=0, column=1, sticky="nsew", padx=(0, 8))
        ttk.Label(c2, text="Clientes", style="Subtitulo.TLabel").pack(side="top", anchor="w")
        b2 = ttk.Frame(c2)
        b2.pack(side="bottom", fill="x", pady=(6, 0))
        ttk.Button(b2, text="Ver el cliente", command=self._open_client).pack(side="left")
        ttk.Label(b2, text="o doble clic", style="Suave.TLabel").pack(side="left", padx=6)
        self.cl_tree = self._tree(c2, (("cliente", "Cliente", 112), ("camaras", "Cámaras", 96),
                                       ("hoy", "Hoy", 40), ("revisar", "Sin revisar", 88)))
        self.cl_tree.tag_configure("mal", foreground=estilo.ERROR)
        self.cl_tree.tag_configure("nada", foreground=estilo.SUAVE)
        self.cl_tree.bind("<Double-1>", lambda _e: self._open_client())
        self.cl_tree.bind("<Return>", lambda _e: self._open_client())

        # 3 · cámaras con problemas + lo que pasó
        c3 = ttk.Frame(body)
        c3.grid(row=0, column=2, sticky="nsew")
        self.pr_head = ttk.Label(c3, text="Cámaras con problemas", style="Subtitulo.TLabel")
        self.pr_head.pack(side="top", anchor="w")
        lg = ttk.Frame(c3)
        lg.pack(side="bottom", fill="x", pady=(8, 0))
        ttk.Label(lg, text="Lo que pasó (desde que abriste la app)", style="Subtitulo.TLabel").pack(anchor="w")
        self.log_tree = self._tree(lg, (("hora", "Hora", 56), ("que", "Qué pasó", 340)), height=4, fill="x")
        self.log_tree.tag_configure("cayo", foreground=estilo.ERROR)
        self.log_tree.tag_configure("volvio", foreground=estilo.OK)
        b3 = ttk.Frame(c3)
        b3.pack(side="bottom", fill="x", pady=(6, 0))
        ttk.Button(b3, text="Ir a la cámara", command=self._open_camera).pack(side="left")
        ttk.Button(b3, text="Probar conexión", command=self._probe).pack(side="left", padx=6)
        self.pr_detail = ttk.Label(c3, text="", style="Suave.TLabel", wraplength=420, justify="left")
        self.pr_detail.pack(side="bottom", fill="x", pady=(6, 0))
        self.pr_tree = self._tree(c3, (("camara", "Cámara", 160), ("que", "Qué pasa", 180), ("desde", "Hace", 70)), height=6)
        self.pr_tree.tag_configure(PROBLEMA, foreground=estilo.ERROR)
        self.pr_tree.tag_configure(ATENCION, foreground=estilo.AVISO)
        self.pr_tree.bind("<<TreeviewSelect>>", lambda _e: self._show_problem())
        self.pr_tree.bind("<Double-1>", lambda _e: self._open_camera())
        self.pr_tree.bind("<Return>", lambda _e: self._open_camera())
        self.pr_empty = ttk.Label(c3, text="", style="Suave.TLabel")

    def _tree(self, parent, cols, height: int = 8, fill: str = "both"):
        ttk = self.ttk
        frame = ttk.Frame(parent)
        frame.pack(side="top", fill=fill, expand=fill == "both")
        tree = ttk.Treeview(frame, columns=[c[0] for c in cols], show="headings", height=height,
                            selectmode="browse")
        for key, title, width in cols:
            tree.heading(key, text=title, anchor="w")
            tree.column(key, width=width, minwidth=40, stretch=True, anchor="w")
        sb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        tree.pack(side="left", fill="both", expand=True)
        return tree

    # ---------- datos ----------
    def client_names(self) -> dict[int, str]:
        return {c["id"]: c["name"] for c in self.app.clients}

    def health_tick(self) -> None:
        """Después de cada consulta del estado de las cámaras (cada 30 s, también minimizada o en otra pestaña): anota
        y avisa lo que se cayó o volvió."""
        now = self.clock()
        self.last_ok, self.net_down_since = now, None
        notices = self.watch.update(self.app.cameras, self.app.agents, self.client_names(), now=now)
        for n in notices:
            if n.kind == "cayo":
                log.warning("Cámara con problemas: %s", n.text)
            else:
                log.info("%s", n.text)
        if notices and self.notify_var.get():
            self._notify(notices)
        self._tab_title()

    def net_failed(self, exc: Exception) -> None:
        if self.net_down_since is None:
            self.net_down_since = self.clock()
        log.debug("No se pudo consultar el estado de las cámaras: %s", exc)
        self._updated_text()

    def _notify(self, notices: list[Notice]) -> None:
        toasts = getattr(self.app, "toasts", None)
        root = self.app.v.root
        bad = False
        for n in notices:
            if toasts is not None:
                if n.kind == "cayo":
                    bad = True
                    first = n.camera_ids[0] if len(n.camera_ids) == 1 else None
                    toasts.show(n.text, kind="error", on_click=lambda c=first: self.app.goto_home(c))
                else:
                    toasts.show(n.text)
        if bad:
            try:
                root.bell()
            except Exception:  # noqa: BLE001
                pass
            flash_window(root)

    def _tab_title(self) -> None:
        n = sum(1 for h in self.watch.health.values() if h.level == PROBLEMA)
        try:
            self.app.notebook.tab(self.parent, text=f"  Inicio · ⚠ {n}  " if n else HOME_TAB)
        except Exception:  # noqa: BLE001
            pass

    def on_show(self) -> None:
        self.load_alerts()
        self.refresh()

    def load_alerts(self) -> None:
        """Las alertas sin revisar de todos los clientes (las más nuevas) y los números de arriba."""
        def done(res):
            events, counts = res
            self.alerts, self.alerts_total = events.get("items") or [], events.get("total") or 0
            if counts is not None:
                self.app.alert_counts = counts
            self._fill_alerts()
            self._fill_tiles()
            self._fill_clients()
        self.app.bg.run(lambda: (self.app.api.get("/api/events", label="sin_etiqueta", limit=ALERTS_LIMIT),
                                 self.app.api.get("/api/stats/alerts")),
                        done, lambda e: log.debug("No se pudieron traer las alertas de «Inicio»: %s", e))

    def refresh(self) -> None:
        self._fill_tiles()
        self._fill_clients()
        self._fill_problems()
        self._fill_log()
        self._updated_text()

    def _updated_text(self) -> None:
        if self.net_down_since is not None:
            txt = (f"⚠ Sin conexión con el servidor desde las {time.strftime('%H:%M', time.localtime(self.net_down_since))}"
                   ": lo de abajo puede estar viejo")
        elif self.last_ok:
            txt = f"Al día a las {time.strftime('%H:%M:%S', time.localtime(self.last_ok))} · se pone al día solo cada 30 s"
        else:
            txt = "Se pone al día solo cada 30 s"
        self.updated.configure(text=txt)

    def _counts(self) -> dict:
        return (getattr(self.app, "alert_counts", None) or {}).get("clients") or {}

    def _fill_tiles(self) -> None:
        counts = self._counts()
        unreviewed = sum(d.get("unreviewed", 0) for d in counts.values())
        today = sum(d.get("today", 0) for d in counts.values())
        hs = list(self.watch.health.values())
        ok = sum(1 for h in hs if h.level in (OK, CERRADO, ATENCION))
        bad = sum(1 for h in hs if h.level == PROBLEMA)
        vals = {"unreviewed": (str(unreviewed), estilo.AVISO if unreviewed else estilo.MARCA),
                "today": (str(today), estilo.MARCA),
                "ok": (f"{ok} de {len(hs)}" if hs else "—", estilo.OK if hs and not bad else estilo.MARCA),
                "bad": (str(bad), estilo.ERROR if bad else estilo.OK)}
        if "machines" in self.tiles:
            ms = [a for a in self.app.agents if a.get("kind") == "servidor"]
            on = sum(1 for a in ms if a.get("online"))
            vals["machines"] = (f"{on} de {len(ms)}" if ms else "0", estilo.OK if ms and on == len(ms) else
                                estilo.ERROR if ms and not on else estilo.AVISO if ms else estilo.SUAVE)
        for key, (_f, num, _cap) in self.tiles.items():
            text, color = vals.get(key, ("—", estilo.MARCA))
            num.configure(text=text, fg=color)

    def _fill_alerts(self) -> None:
        prev = self.al_tree.selection()
        self.al_tree.delete(*self.al_tree.get_children())
        for e in self.alerts:
            self.al_tree.insert("", "end", iid=str(e["id"]), values=(
                fmt_date(e.get("occurred_at")), e.get("client_name") or "", e.get("camera_name") or "",
                TYPE_NAMES.get(e.get("type"), e.get("type") or "")))
        n = self.alerts_total
        self.al_head.configure(text=f"Alertas sin revisar ({n})" if n else "Alertas sin revisar: ninguna ✔")
        keep = [i for i in prev if self.al_tree.exists(i)]
        if keep:
            self.al_tree.selection_set(keep)

    def _fill_clients(self) -> None:
        prev = self.cl_tree.selection()
        counts = self._counts()
        rows = []
        for c in self.app.clients:
            cams = [x for x in self.app.cameras if x.get("client_id") == c["id"]]
            hs = [self.watch.health.get(x["id"]) for x in cams]
            hs = [h for h in hs if h is not None]
            bad = sum(1 for h in hs if h.level == PROBLEMA)
            good = len(hs) - bad
            cnt = counts.get(str(c["id"])) or {}
            if hs:
                cams_txt = f"⚠ {good} de {len(hs)} bien" if bad else f"✔ {len(hs)} bien"
            else:
                cams_txt = "no analiza" if cams else "sin cámaras"
            tag = "mal" if bad else "" if hs else "nada"
            rows.append(((-bad, -(cnt.get("unreviewed") or 0), c["name"].lower()), c["id"], (
                c["name"], cams_txt, cnt.get("today", 0) or "", cnt.get("unreviewed", 0) or ""), tag))
        self.cl_tree.delete(*self.cl_tree.get_children())
        for _k, cid, values, tag in sorted(rows):
            self.cl_tree.insert("", "end", iid=str(cid), values=values, tags=(tag,) if tag else ())
        keep = [i for i in prev if self.cl_tree.exists(i)]
        if keep:
            self.cl_tree.selection_set(keep)

    def _since_text(self, cid: int, now: float) -> str:
        since = self.watch.since(cid)
        if since is None:
            return ""
        txt = duration(now - since)
        return f"{txt} o más" if self.watch.was_bad_at_start(cid) and (self.watch.health[cid].age_s is None) else txt

    def _fill_problems(self) -> None:
        prev = self.pr_tree.selection()
        now = self.clock()
        rows = [(cid, h) for cid, h in self.watch.health.items() if h.level in (PROBLEMA, ATENCION)]
        rows.sort(key=lambda r: (r[1].level != PROBLEMA, self.watch.since(r[0]) or now, self.watch.labels[r[0]]))
        self.pr_tree.delete(*self.pr_tree.get_children())
        for cid, h in rows:
            self.pr_tree.insert("", "end", iid=str(cid), values=(self.watch.labels[cid], h.text,
                                                                 self._since_text(cid, now)), tags=(h.level,))
        n_bad = sum(1 for _c, h in rows if h.level == PROBLEMA)
        total = len(self.watch.health)
        if not total:
            head = "Cámaras con problemas: todavía no se analiza ninguna"
        elif not rows:
            head = f"Cámaras con problemas: ninguna ✔ (las {total} que se analizan andan bien)"
        else:
            head = f"Cámaras con problemas ({n_bad})" if n_bad else "Cámaras con problemas: ninguna ✔"
        self.pr_head.configure(text=head)
        keep = [i for i in prev if self.pr_tree.exists(i)]
        if keep:
            self.pr_tree.selection_set(keep)
        elif rows:
            self.pr_tree.selection_set(str(rows[0][0]))
        else:
            self.pr_detail.configure(text="Cuando una cámara que se analiza deje de andar aparece acá"
                                     + (" y te aviso con un cartel rojo." if self.notify_var.get() else "."))
        self._show_problem()

    def _show_problem(self) -> None:
        cid = self._sel(self.pr_tree)
        h = self.watch.health.get(cid) if cid is not None else None
        if h is None:
            return
        txt = h.detail or h.text
        self.pr_detail.configure(text=f"{txt}\nQué hacer: {hint(h, self.app.is_admin)}")

    def _fill_log(self) -> None:
        self.log_tree.delete(*self.log_tree.get_children())
        today = time.strftime("%Y-%m-%d")
        for at, kind, text in reversed(self.watch.log):
            if time.strftime("%Y-%m-%d", time.localtime(at)) != today:
                break
            self.log_tree.insert("", "end", values=(time.strftime("%H:%M", time.localtime(at)),
                                                    ("✘ " if kind == "cayo" else "✔ ") + text), tags=(kind,))

    # ---------- ir a cada cosa ----------
    @staticmethod
    def _sel(tree) -> int | None:
        sel = tree.selection()
        try:
            return int(sel[0]) if sel else None
        except ValueError:
            return None

    def _open_alert(self) -> None:
        eid = self._sel(self.al_tree)
        self.app.goto_alert(eid)

    def _open_client(self) -> None:
        cid = self._sel(self.cl_tree)
        if cid is not None:
            self.app.goto_client(cid)

    def _open_camera(self) -> None:
        cid = self._sel(self.pr_tree)
        if cid is not None:
            self.app.goto_camera(cid)

    def _probe(self) -> None:
        cid = self._sel(self.pr_tree)
        if cid is None:
            return self.app.mb.showinfo("Inicio", "Elegí una cámara de la lista de problemas.")
        cam = next((c for c in self.app.cameras if c.get("id") == cid), None)
        if cam is not None and self.app.goto_camera(cid):
            self.app.probe(cam)  # (el resultado sale en el detalle de la cámara, en «Clientes»)

    def select_problem(self, cid: int | None) -> None:
        """Del cartel rojo: deja elegida esa cámara en la lista de problemas."""
        self.refresh()
        if cid is not None and self.pr_tree.exists(str(cid)):
            self.pr_tree.selection_set(str(cid))
            self.pr_tree.see(str(cid))
            self._show_problem()

    def _go_alerts(self) -> None:
        self.app.goto_alert(None)

    def _go_problems(self) -> None:
        rows = self.pr_tree.get_children()
        if rows:
            self.pr_tree.selection_set(rows[0])
            self.pr_tree.focus_set()

    def _go_machines(self) -> None:
        tab = getattr(self.app, "tab_mq", None)
        if tab is not None:
            self.app.notebook.select(tab)


def install(app, notebook) -> Inicio:
    """Pone la pestaña «Inicio» primera (antes de «En vivo») y la deja elegida."""
    tab = app.ttk.Frame(notebook, padding=8)
    if notebook.tabs():
        notebook.insert(0, tab, text=HOME_TAB)
    else:
        notebook.add(tab, text=HOME_TAB)
    app.tab_home = tab
    return Inicio(app, tab)
