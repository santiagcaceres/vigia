"""Pestaña «Inicio» (20 nov, el dueño: «que la pantalla de inicio sea como un resumen, con las alertas por un lado,
clientes en otra, y otro que salen las cámaras que se desconectan o pasa algo; si una cámara deja de estar en verde, que
mande un aviso»). El aviso es solo en la app (elegido con el dueño): cartel rojo, sonido y la barra de tareas."""

import os
import queue
import time as _time
import types

import pytest

import agent.inicio as ini
import agent.oficina as of

OK_ST = {"agent_online": True, "state": "en_linea", "fps": 5.0}
CLIENTS = {3: "Minas", 4: "Kiosco"}


def _cam(i, client_id=3, **kw):
    return {"id": i, "client_id": client_id, "name": f"Cámara {i}", "enabled": True, "analyzed": True, "agent_id": 7,
            "analyzer_name": "maquina-1", "status": dict(OK_ST), **kw}


def _down(cam, age=10, error="no contesta en el puerto 21554", state="sin_conexion"):
    return {**cam, "status": {"agent_online": True, "state": state, "error": error, "state_age_s": age}}


# ---------------------------------------------------------------- qué es «verde» y qué es un problema
def test_que_cuenta_como_problema():
    h = ini.camera_health
    assert h(_cam(1)).level == ini.OK and h(_cam(1)).text == "Analizando · 5 cuadros/s"
    # las que no se analizan no se miran (no hay verde que perder)
    assert h(_cam(1, analyzed=False, agent_id=None, status=None)).level == ini.NO_SE_MIRA
    assert h(_cam(1, enabled=False)).level == ini.NO_SE_MIRA
    # local cerrado y alarmas: bien
    assert h(_cam(1, status={**OK_ST, "state": "fuera_de_horario"})).level == ini.CERRADO
    assert h(_cam(1, status={**OK_ST, "state": "escuchando"})).level == ini.OK
    # sin conexión: el motivo entero y desde cuándo
    d = h(_down(_cam(1), age=600))
    assert (d.level, d.text, d.age_s) == (ini.PROBLEMA, "Sin conexión", 600)
    assert "no contesta" in d.detail and "router" in ini.hint(d, True)
    # marcada para analizar y ninguna máquina la tomó: más tiempo antes de avisar (puede estar arrancando)
    w = h(_cam(1, agent_id=None, status=None))
    assert w.level == ini.PROBLEMA and w.text.startswith("Esperando una máquina") and w.grace_s == ini.START_GRACE_S
    assert "«Máquinas»" in ini.hint(w, True) and "administrador" in ini.hint(w, False)
    # la máquina no responde: el problema es de la máquina (un solo aviso para todas sus cámaras)
    m = h(_cam(1, status={**OK_ST, "agent_online": False}),
          {7: {"id": 7, "name": "maquina-1", "last_seen": "2026-11-20T10:00:00Z"}}, now=1795176000.0 + 300)
    assert m.text == "La máquina «maquina-1» no responde" and m.group == "maquina:7"
    # en línea pero trabada: sin cuadros nuevos hace 3 min
    s = h(_cam(1, status={**OK_ST, "last_frame_age_s": 190}))
    assert s.level == ini.PROBLEMA and s.text == "Sin imagen hace 3 min"
    # no da abasto: amarillo (no es caída: no avisa con cartel)
    a = h(_cam(1, status={**OK_ST, "skipped_pct": 40}))
    assert a.level == ini.ATENCION and "saltea 40 %" in a.text


def test_duracion_en_palabras():
    assert [ini.duration(s) for s in (5, 125, 3600 * 3 + 300, 86400 * 2)] == ["5 s", "2 min", "3 h 05 min", "2 días"]


# ---------------------------------------------------------------- cuándo avisa
def test_avisa_si_la_caida_dura_y_avisa_cuando_vuelve():
    w = ini.HealthWatch()
    t = 1000.0
    cam = _cam(1, name="Entrada")
    assert w.update([cam], [], CLIENTS, now=t) == []  # al abrir: solo anota
    # se cae: recién a los 2 minutos avisa (las reconexiones de un momento no cuentan)
    assert w.update([_down(cam, age=10)], [], CLIENTS, now=t + 30) == []
    assert w.update([_down(cam, age=70)], [], CLIENTS, now=t + 90) == []
    n = w.update([_down(cam, age=130)], [], CLIENTS, now=t + 150)
    assert [(x.kind, x.text, x.camera_ids) for x in n] == [("cayo", "Minas · Entrada: Sin conexión", [1])]
    assert w.update([_down(cam, age=160)], [], CLIENTS, now=t + 180) == []  # (una sola vez)
    # vuelve: avisa cuando anduvo 1 minuto seguido
    assert w.update([cam], [], CLIENTS, now=t + 210) == []
    assert w.update([cam], [], CLIENTS, now=t + 240) == []
    n = w.update([cam], [], CLIENTS, now=t + 270)
    assert [(x.kind, x.text) for x in n] == [("volvio", "Minas · Entrada volvió a andar")]
    assert [k for _at, k, _t in w.log] == ["cayo", "volvio"]


def test_lo_que_ya_estaba_mal_al_abrir_no_avisa_y_un_corte_corto_tampoco():
    w = ini.HealthWatch()
    t = 1000.0
    rota, cam = _down(_cam(1), age=5000), _cam(2)
    w.update([rota, cam], [], CLIENTS, now=t)
    assert w.update([rota, cam], [], CLIENTS, now=t + 600) == []  # (está en la lista, pero no salta un cartel)
    assert w.was_bad_at_start(1) and w.since(1) == pytest.approx(t - 5000)
    # un corte de 30 s y vuelve: nada
    assert w.update([rota, _down(cam, age=5)], [], CLIENTS, now=t + 630) == []
    for k in range(1, 6):
        assert w.update([rota, cam], [], CLIENTS, now=t + 630 + 30 * k) == []
    assert w.since(2) is None
    # la que estaba mal al abrir anduvo 30 s y se cortó: no volvió de verdad, sigue sin cartel
    w.update([_cam(1), cam], [], CLIENTS, now=t + 900)
    assert w.update([_down(_cam(1), age=5), cam], [], CLIENTS, now=t + 930) == []
    assert w.update([_down(_cam(1), age=125), cam], [], CLIENTS, now=t + 1050) == []
    # anduvo un minuto seguido y se volvió a caer: eso sí es nuevo
    for k in range(3):
        w.update([_cam(1), cam], [], CLIENTS, now=t + 1080 + 30 * k)
    w.update([_down(_cam(1), age=5), cam], [], CLIENTS, now=t + 1200)
    assert [x.camera_ids for x in w.update([_down(_cam(1), age=125), cam], [], CLIENTS, now=t + 1320)] == [[1]]


def test_una_camara_que_corta_y_vuelve_todo_el_rato_cuenta_como_caida():
    w = ini.HealthWatch()
    t = 1000.0
    cam = _cam(1)
    w.update([cam], [], CLIENTS, now=t)
    out = []
    for k in range(1, 8):  # mal, bien, mal, bien…: nunca anda un minuto seguido
        out += w.update([_down(cam, age=5) if k % 2 else cam], [], CLIENTS, now=t + 30 * k)
    assert [x.kind for x in out] == ["cayo"]


def test_si_se_cae_una_maquina_es_un_solo_aviso():
    w = ini.HealthWatch()
    t = 1795176000.0
    cams = [_cam(i) for i in (1, 2, 3)]
    agents = [{"id": 7, "name": "maquina-1", "last_seen": "2026-11-20T10:00:00Z"}]
    w.update(cams, agents, CLIENTS, now=t)
    off = [{**c, "status": {**OK_ST, "agent_online": False}} for c in cams]
    n = w.update(off, agents, CLIENTS, now=t + 200)  # (el servidor la da por caída a los 3 min: ya pasó el margen)
    assert [x.text for x in n] == ["La máquina «maquina-1» no responde: 3 cámaras sin analizar"]


def test_muchas_a_la_vez_un_solo_cartel_y_las_que_dejan_de_analizarse_no_son_caidas():
    w = ini.HealthWatch()
    t = 1000.0
    cams = [_cam(i) for i in range(1, 6)]
    w.update(cams, [], CLIENTS, now=t)
    n = w.update([_down(c, age=200) for c in cams], [], CLIENTS, now=t + 30)
    assert len(n) == 1 and n[0].text.startswith("5 cámaras dejaron de andar: Minas · Cámara 1") and \
        n[0].camera_ids == [1, 2, 3, 4, 5]
    # dejaron de analizarse («■ Dejar de analizar»): se olvidan, sin «volvió»
    gone = [{**c, "analyzed": False, "agent_id": None, "status": None} for c in cams]
    assert w.update(gone, [], CLIENTS, now=t + 60) == [] and w.track == {}


# ---------------------------------------------------------------- con ventana
def _tk():
    tk = pytest.importorskip("tkinter")
    if os.name != "nt" and not os.environ.get("DISPLAY"):
        pytest.skip("sin pantalla (correr con xvfb-run)")
    return tk


def _walk(w):
    yield w
    for c in w.winfo_children():
        yield from _walk(c)


def test_la_pestana_inicio_va_primera_muestra_todo_y_avisa(tmp_path, monkeypatch):
    tk = _tk()
    from tkinter import ttk

    from agent import estilo
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"sin Tk: {exc}")
    try:
        root.geometry("1366x698+0+0")
        estilo.aplicar(root)
        nb = ttk.Notebook(root)
        nb.pack(fill="both", expand=True, padx=8, pady=6)
        nb.add(ttk.Frame(nb), text="  En vivo  ")
        shown, went, bells = [], [], []
        root.bell = lambda: bells.append(1)
        # hoy al mediodía: el registro de «Lo que pasó» muestra solo lo de hoy (con una fecha fija, dejaba de andar)
        clock = types.SimpleNamespace(t=_time.mktime(_time.localtime()[:3] + (12, 0, 0, 0, 0, -1)))
        events = {"items": [{"id": 51, "occurred_at": "2026-11-20T13:05:00Z", "client_name": "Minas",
                             "camera_name": "Pasillo", "type": "mano_bolsa"}], "total": 9}
        counts = {"clients": {"3": {"today": 4, "week": 9, "unreviewed": 9}}, "cameras": {}}
        app = types.SimpleNamespace(
            tk=tk, ttk=ttk, notebook=nb, is_admin=True, agent_of=lambda: types.SimpleNamespace(data_dir=str(tmp_path)),
            v=types.SimpleNamespace(root=root), clients=[{"id": 3, "name": "Minas"}, {"id": 4, "name": "Kiosco"}],
            cameras=[_cam(1, name="Entrada"), _down(_cam(2, name="Caja"), age=4000),
                     _cam(3, name="Depósito", status={**OK_ST, "skipped_pct": 50}),
                     _cam(5, client_id=4, name="Mostrador", analyzed=False, agent_id=None, status=None)],
            agents=[{"id": 7, "kind": "servidor", "online": True, "name": "maquina-1"}], alert_counts={},
            toasts=types.SimpleNamespace(show=lambda text, kind="ok", on_click=None: shown.append((kind, text, on_click))),
            api=types.SimpleNamespace(get=lambda path, **kw: events if path == "/api/events" else counts),
            bg=types.SimpleNamespace(run=lambda fn, done=None, fail=None: done(fn())),
            goto_home=lambda cid=None: went.append(("inicio", cid)), goto_alert=lambda eid: went.append(("alerta", eid)),
            goto_client=lambda cid: went.append(("cliente", cid)) or True,
            goto_camera=lambda cid: went.append(("camara", cid)) or True, probe=lambda cam: went.append(("probar", cam["id"])),
            mb=types.SimpleNamespace(showinfo=lambda *a, **k: None))
        home = ini.install(app, nb)
        home.clock = lambda: clock.t
        nb.select(app.tab_home)
        assert nb.tabs()[0] == str(app.tab_home) and nb.tab(app.tab_home, "text") == ini.HOME_TAB
        home.health_tick()
        home.on_show()
        root.update()
        # entra en una pantalla de 1366x768
        assert app.tab_home.winfo_reqwidth() <= app.tab_home.winfo_width() + 2
        assert app.tab_home.winfo_reqheight() <= app.tab_home.winfo_height() + 2
        # los números de arriba
        nums = {k: v[1].cget("text") for k, v in home.tiles.items()}
        assert nums == {"unreviewed": "9", "today": "4", "ok": "2 de 3", "bad": "1", "machines": "1 de 1"}
        # las alertas sin revisar de todos los clientes
        assert home.al_tree.item("51")["values"][1:] == ["Minas", "Pasillo", of.TYPE_NAMES["mano_bolsa"]]
        assert home.al_head.cget("text") == "Alertas sin revisar (9)"
        # clientes: Minas tiene una cámara caída (en rojo, primero); Kiosco no analiza
        assert home.cl_tree.get_children() == ("3", "4")
        assert home.cl_tree.item("3")["values"][:2] == ["Minas", "⚠ 2 de 3 bien"]
        assert home.cl_tree.item("3")["tags"] == ["mal"]
        assert home.cl_tree.item("4")["values"][1] == "no analiza"
        # problemas: la caída primero (en rojo, con hace cuánto), después la que no da abasto (amarillo)
        assert home.pr_tree.get_children() == ("2", "3")
        assert home.pr_tree.item("2")["values"] == ["Minas · Caja", "Sin conexión", "1 h 06 min"]
        assert home.pr_tree.item("3")["tags"] == [ini.ATENCION]
        assert "no contesta en el puerto 21554" in home.pr_detail.cget("text")
        assert "Qué hacer:" in home.pr_detail.cget("text")
        assert nb.tab(app.tab_home, "text") == "  Inicio · ⚠ 1  "
        assert shown == [] and bells == []  # (lo que ya estaba mal al abrir no salta)
        # se cae la Entrada: a los 2 minutos, cartel rojo + sonido; el clic lleva a Inicio con esa cámara
        cams = list(app.cameras)
        app.cameras = [_down(cams[0], age=5)] + cams[1:]
        clock.t += 30
        home.health_tick()
        assert shown == []
        app.cameras = [_down(cams[0], age=125)] + cams[1:]
        clock.t += 120
        home.health_tick()
        assert [(k, t) for k, t, _ in shown] == [("error", "Minas · Entrada: Sin conexión")] and bells == [1]
        shown[0][2]()
        assert went[-1] == ("inicio", 1)
        home.select_problem(1)
        assert home.pr_tree.selection() == ("1",)
        assert nb.tab(app.tab_home, "text") == "  Inicio · ⚠ 2  "
        assert home.log_tree.item(home.log_tree.get_children()[0])["values"][1] == "✘ Minas · Entrada: Sin conexión"
        # con los avisos apagados (se recuerda), no salta nada
        home.notify_var.set(False)
        home._save_prefs()
        assert (tmp_path / "inicio.json").read_text() == '{"avisos": false}'
        app.cameras = cams
        for _ in range(3):
            clock.t += 30
            home.health_tick()
        assert len(shown) == 1
        # ir a cada cosa
        home.al_tree.selection_set("51")
        home._open_alert()
        home.cl_tree.selection_set("4")
        home._open_client()
        home.refresh()
        home.pr_tree.selection_set("3")
        home._probe()
        assert went[-4:] == [("alerta", 51), ("cliente", 4), ("camara", 3), ("probar", 3)]
        texts = [str(w.cget("text")) for w in _walk(app.tab_home) if w.winfo_class() in ("TButton", "TCheckbutton")]
        assert {"Abrir en «Alertas»", "Ver el cliente", "Ir a la cámara", "Probar conexión"} <= set(texts)
        if os.environ.get("VIGIA_CAPTURAS"):
            root.update()
            os.system(f"import -window root {os.environ['VIGIA_CAPTURAS']}/inicio.png")
    finally:
        root.destroy()


def test_la_oficina_lleva_a_la_alerta_y_a_la_camara(monkeypatch):
    """Desde «Inicio»: «Abrir en «Alertas»» deja elegida esa alerta; «Ir a la cámara», el cliente y la cámara."""
    tk = _tk()
    from tkinter import ttk

    from agent import estilo
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"sin Tk: {exc}")
    try:
        root.geometry("1366x698+0+0")
        estilo.aplicar(root)
        nb = ttk.Notebook(root)
        nb.pack(fill="both", expand=True)
        o = of.Oficina.__new__(of.Oficina)
        o.tk, o.ttk, o.notebook, o.is_admin = tk, ttk, nb, True
        o.v = types.SimpleNamespace(root=root, log_handler=types.SimpleNamespace(q=queue.Queue()))
        o.mb = types.SimpleNamespace(showinfo=lambda *a, **k: None, askyesno=lambda *a, **k: False)
        events = {"items": [{"id": i, "occurred_at": "2026-11-20T13:05:00Z", "client_name": "Minas",
                             "camera_name": "Caja", "type": "salida_sin_pago", "label": None} for i in (60, 61, 62)],
                  "total": 3}
        o.bg = types.SimpleNamespace(run=lambda fn, done=None, fail=None: done(fn()) if done else fn())
        o.api = types.SimpleNamespace(get=lambda path, **kw: events)
        o.clients = [{"id": 3, "name": "Minas", "plans": ["hurto"]}, {"id": 4, "name": "Kiosco", "plans": ["hurto"]}]
        o.cameras = [_cam(1, measures=["hurto"], zones=[]), _cam(2, client_id=4, measures=["hurto"], zones=[])]
        o.alert_counts, o._tune_windows, o.agent_of = {}, [], None
        o.tab_clients = ttk.Frame(nb, padding=8)
        o.tab_alerts = ttk.Frame(nb, padding=8)
        nb.add(o.tab_clients, text="  Clientes  ")
        nb.add(o.tab_alerts, text="  Alertas  ")
        o._clip_frame = lambda cam: None
        o.refresh_all = lambda *a, **k: None
        o._show_alert = lambda *a, **k: None
        monkeypatch.setattr(of.Oficina, "_show_camera", lambda self: None, raising=False)
        o._build_clients()
        o._fill_clients()
        try:
            o._build_alerts()
        except Exception as exc:  # noqa: BLE001 - (la pestaña de alertas pide más cosas que esta prueba no arma)
            pytest.skip(f"no se pudo armar «Alertas» sin la app entera: {exc}")
        root.update()
        o.goto_alert(61)
        root.update()
        assert nb.select() == str(o.tab_alerts) and o.al_tree.selection() == ("61",)
        assert o.goto_camera(2)
        root.update()
        assert nb.select() == str(o.tab_clients)
        assert o.cl_tree.selection() == ("4",) and o.cam_tree.selection() == ("2",)
        assert not o.goto_camera(99)
    finally:
        root.destroy()


ADMIN = ("admin", "clave-admin-123")


def test_de_punta_a_punta_con_el_servidor_y_una_maquina(tmp_path, db_url):
    """La app de verdad (ventana, pestañas, consultas cada 30 s) contra el servidor de verdad, con una máquina de
    análisis que informa que una cámara se cayó y después volvió."""
    _tk()
    import time as _time
    from pathlib import Path

    from fastapi.testclient import TestClient

    from agent.oficina import Api, Oficina
    from agent.ventana import Ventana
    from server.main import Settings, create_app

    app = create_app(Settings(data_dir=tmp_path / "data", database_url=db_url, admin_password=ADMIN[1],
                              serve_public=False))
    c = TestClient(app)
    assert c.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]}).status_code == 200
    cid = c.post("/api/clients", json={"name": "Minas", "plans": ["hurto"]}).json()["id"]
    ids = [c.post("/api/cameras", json={"client_id": cid, "name": n, "brand": "ip", "host": "dvr", "password": "x"}
                  ).json()["id"] for n in ("Entrada", "Caja")]
    key = c.post("/api/analysis/pool-key").json()["key"]
    tok = TestClient(app).post("/api/agent/enroll", json={"key": key, "name": "maquina-1", "instance_id": "i-1"}
                               ).json()["token"]
    mq = TestClient(app)
    mq.headers["X-Agent-Token"] = tok
    assert mq.post("/api/agent/status", json={"cameras": [], "fresh": True}).status_code == 200
    assert c.post("/api/cameras/analyze", json={"camera_ids": ids, "analyze": True}).status_code == 200

    def report(caja_state="en_linea", age=600, error=None):
        cams = [{"camera_id": ids[0], "state": "en_linea", "fps": 5.0, "state_age_s": 600},
                {"camera_id": ids[1], "state": caja_state, "fps": 5.0 if caja_state == "en_linea" else 0.0,
                 "state_age_s": age, "error": error}]
        r = mq.post("/api/agent/status", json={"cameras": cams})
        assert r.status_code == 200, r.text
        assert set(r.json().get("cameras") or []) == set(ids)  # (el reparto se las dio a esta máquina)
    report()

    api = Api("http://testserver")
    api.http = c
    api.user = {"username": "admin", "role": "admin"}
    v = Ventana(tmp_path / "ventana.json")
    if v.failed or not v.open:
        pytest.skip("sin Tk")
    shots = os.environ.get("VIGIA_CAPTURAS")
    try:
        v.root.attributes("-zoomed", False)
        v.root.geometry("1366x698+0+0")
        v.build_main("admin (administrador)", analyze=True, on_demand=True)
        agent = types.SimpleNamespace(data_dir=tmp_path / "datos")

        def sync(fn, done=None, fail=None):
            try:
                res = fn()
            except Exception as exc:  # noqa: BLE001
                return fail(exc) if fail else None
            return done(res) if done else None
        off = Oficina(v, api, agent=lambda: agent)
        off.bg = types.SimpleNamespace(run=sync, q=queue.Queue(), drain=lambda: None)
        off.mb = types.SimpleNamespace(showinfo=lambda *a, **k: None, showwarning=lambda *a, **k: None,
                                       showerror=lambda *a, **k: None, askyesno=lambda *a, **k: True)
        off.build(v.notebook)
        home = off.home
        clock = types.SimpleNamespace(t=_time.time())
        home.clock = lambda: clock.t
        v.root.update()
        # «Inicio» va primera y es la que se ve al abrir
        assert v.notebook.tabs()[0] == str(off.tab_home) and v.notebook.select() == str(off.tab_home)
        assert [v.notebook.tab(t, "text").strip() for t in v.notebook.tabs()][:3] == ["Inicio", "En vivo", "Clientes"]
        off._auto_refresh()
        v.root.update()
        assert home.tiles["ok"][1].cget("text") == "2 de 2" and home.tiles["bad"][1].cget("text") == "0"
        assert "ninguna ✔" in home.pr_head.cget("text") and home.cl_tree.item(str(cid))["values"][1] == "✔ 2 bien"
        assert home.tiles["machines"][1].cget("text") == "1 de 1"
        # se cae la Caja: primero nada (puede ser un corte de un momento)…
        report("sin_conexion", age=10, error="maquina: dvr no contesta en el puerto 21554 desde esta PC")
        clock.t += 30
        off._auto_refresh()
        v.root.update()
        assert not off.toasts.items and home.pr_tree.get_children() == (str(ids[1]),)
        # …a los 2 minutos, el cartel rojo (también con la app en otra pestaña)
        v.notebook.select(off.tab_clients)
        report("sin_conexion", age=130, error="maquina: dvr no contesta en el puerto 21554 desde esta PC")
        clock.t += 120
        off._auto_refresh()
        v.root.update()
        reds = [it for it in off.toasts.items if it.get("kind") == "error"]
        assert [it["text"] for it in reds] == ["Minas · Caja: Sin conexión"]
        assert v.notebook.tab(off.tab_home, "text") == "  Inicio · ⚠ 1  "
        if shots:
            Path(shots).mkdir(parents=True, exist_ok=True)
            v.root.update()
            os.system(f"import -window root {Path(shots) / 'inicio-cartel.png'}")
        # el clic en el cartel lleva a «Inicio» con esa cámara elegida y el motivo a la vista
        reds[0]["label"].event_generate("<Button-1>")
        v.root.update()
        assert v.notebook.select() == str(off.tab_home) and home.pr_tree.selection() == (str(ids[1]),)
        assert "no contesta en el puerto 21554" in home.pr_detail.cget("text")
        assert "router" in home.pr_detail.cget("text")
        if shots:
            os.system(f"import -window root {Path(shots) / 'inicio-problema.png'}")
        # «Ir a la cámara»: «Clientes» con la cámara elegida
        home._open_camera()
        v.root.update()
        assert v.notebook.select() == str(off.tab_clients) and off.cam_tree.selection() == (str(ids[1]),)
        # vuelve: cuando anda un minuto seguido, el cartel verde
        report("en_linea", age=5)
        for _ in range(3):
            clock.t += 30
            off._auto_refresh()
            v.root.update()
        assert any(it["text"] == "Minas · Caja volvió a andar" and it.get("kind") == "ok" for it in off.toasts.items)
        v.notebook.select(off.tab_home)
        v.root.update()
        assert [home.log_tree.item(i)["values"][1] for i in home.log_tree.get_children()] == [
            "✔ Minas · Caja: volvió a andar", "✘ Minas · Caja: Sin conexión"]
        assert v.notebook.tab(off.tab_home, "text") == ini.HOME_TAB
    finally:
        v.close()
