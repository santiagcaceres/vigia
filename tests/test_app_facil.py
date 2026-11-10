"""La app más fácil de usar (10 nov, el dueño: «la app de escritorio es poco intuitiva… hacelo lo más fluido para el
usuario: cómo entrenar, cómo poner a analizar; sacar el en vivo y clips de cada cámara»).

- «Clientes»: la cámara tiene dos pasos a la vista («✎ Marcar zonas» y «▶ Empezar a analizar» / «■ Dejar de analizar»);
  una sola columna dice cómo está y el detalle dice qué le falta. Ya no hay «● En vivo» ni «🎞 Clips» por cámara.
- Guardar las zonas de una cámara que no se analiza ofrece empezar a analizarla ahí mismo.
- «En vivo»: sin nada elegido explica qué hacer; «✎ Marcar zonas» de la cámara que se mira.
- «Entrenar IA hurtos»: una frase dice qué sigue (lo demás, en tests/test_entrenar_ia_app.py)."""

import os
import queue
import types

import pytest

import agent.entrenar_ia as ei
import agent.oficina as of


def _oficina(cams, client=None, admin=True, answer=True):
    """Una Oficina sin ventana: lo que necesitan los pasos de la cámara."""
    o = of.Oficina.__new__(of.Oficina)
    o.is_admin = admin
    o.cameras = cams
    o.clients = [client or {"id": 3, "name": "Minas", "plans": ["hurto", "comercio"]}]
    o.said, o.posts = [], []
    o.mb = types.SimpleNamespace(
        showinfo=lambda *a, **k: o.said.append(("info", a[1])), showwarning=lambda *a, **k: o.said.append(("aviso", a[1])),
        showerror=lambda *a, **k: o.said.append(("error", a[1])),
        askyesno=lambda *a, **k: o.said.append(("pregunta", a[1])) or answer)
    o.bg = types.SimpleNamespace(run=lambda fn, done=None, fail=None: done(fn()) if done else fn())
    o.api = types.SimpleNamespace(post=lambda path, body=None: o.posts.append((path, body)) or {"changed": 1})
    o.v = types.SimpleNamespace(root=types.SimpleNamespace(after=lambda ms, fn: fn()))
    o.on_config_saved = lambda: o.posts.append(("config", None))
    o.refresh_all = lambda on_done=None, on_fail=None: on_done() if on_done else None
    o.config_saved = lambda: o.posts.append(("guardado", None))
    o.chosen = list(cams)
    o._sel_cameras = lambda: o.chosen
    o._sel_client = lambda: o.clients[0]
    o._client_of = lambda cam: o.clients[0]
    return o


def _cam(i=1, **kw):
    return {"id": i, "client_id": 3, "name": f"Cámara {i}", "enabled": True, "plans": None, "profile": "general",
            "measures": ["hurto", "comercio"], "zones": [], "analyzed": False, "agent_id": None, "status": None, **kw}


ZONA = [{"name": "Góndola", "type": "shelf", "polygon": [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5]]}]
EN_LINEA = {"agent_online": True, "state": "en_linea", "fps": 5.0}


def test_el_estado_de_la_camara_en_una_frase_y_que_le_falta():
    o = _oficina([])
    # sin analizar: lo dice así (antes eran dos columnas: «—» y «No»)
    c = _cam()
    assert o._cam_state(c) == "No se analiza" and o._zones_text(c) == "Sin marcar"
    assert o._next_step(c, o.clients[0]) == ("Paso 2: «✎ Marcar zonas» (góndolas, caja, salida, entrada). Después, "
                                             "«▶ Empezar a analizar».", False)
    c = _cam(zones=ZONA)
    assert o._zones_text(c) == "✔ 1"
    assert o._next_step(c, o.clients[0])[0].startswith("Paso 3: «▶ Empezar a analizar»")
    # marcada para analizar, sin máquina
    c = _cam(zones=ZONA, analyzed=True)
    assert o._cam_state(c) == "Esperando una máquina de análisis"
    assert "prendé una en la pestaña «Máquinas»" in o._next_step(c, o.clients[0])[0]
    assert "avisale a un administrador" in _oficina([], admin=False)._next_step(c, o.clients[0])[0]
    # en una máquina, andando
    c = _cam(zones=ZONA, analyzed=True, agent_id=7, status={**EN_LINEA, "idle_pct": 40})
    assert o._cam_state(c) == "Analizando · 5 cuadros/s · 40% quieta"
    assert o._next_step(c, o.clients[0]) == ("Lista: se analiza en una máquina y da alertas y métricas.", True)
    # se analiza sin zonas: lo avisa (no es un error: los gestos y el conteo general andan)
    c = _cam(analyzed=True, agent_id=7, status=EN_LINEA)
    step, ready = o._next_step(c, o.clients[0])
    assert not ready and "le faltan las zonas" in step and "«Salida sin pagar»" in step
    c = _cam(analyzed=True, agent_id=7, status=EN_LINEA, measures=["comercio"])
    assert "no cuenta quién entra ni por sector" in o._next_step(c, o.clients[0])[0]
    # un depósito sin zonas vale (toda la imagen es zona restringida)
    c = _cam(analyzed=True, agent_id=7, status=EN_LINEA, profile="deposito")
    assert o._next_step(c, o.clients[0])[1]
    # la máquina se cayó, o no puede abrir el video
    c = _cam(zones=ZONA, analyzed=True, agent_id=7, status={"agent_online": False, "state": "en_linea"})
    assert o._cam_state(c) == "La máquina está desconectada"
    c = _cam(zones=ZONA, analyzed=True, agent_id=7, status={"agent_online": True, "state": "sin_conexion"})
    assert o._cam_state(c) == "Sin conexión" and "«Probar conexión»" in o._next_step(c, o.clients[0])[0]
    assert o._cam_state(_cam(enabled=False)) == "Deshabilitada"
    # recién se dejó de analizar: el último estado que mandó la máquina ya no cuenta
    assert o._cam_state(_cam(zones=ZONA, status=EN_LINEA)) == "No se analiza"


def test_empezar_y_dejar_de_analizar_con_un_boton():
    # con zonas: va directo, sin preguntas
    c = _cam(zones=ZONA)
    o = _oficina([c])
    assert o._analyze_label() == of.ANALYZE_ON == "▶ Empezar a analizar"
    o.set_analyze(None)
    assert o.posts[0] == ("/api/cameras/analyze", {"camera_ids": [1], "analyze": True})
    assert not [m for m in o.said if m[0] == "pregunta"]
    # sin máquina en línea lo dice (no queda «analizando» en silencio)
    assert o.said and o.said[-1][0] == "aviso" and "no hay ninguna máquina de análisis en línea" in o.said[-1][1]
    # sin zonas pregunta, explicando qué se pierde; con «No», nada
    o = _oficina([_cam()], answer=False)
    o.set_analyze(None)
    assert o.posts == [] and "no tiene zonas marcadas" in o.said[-1][1] and "«✎ Marcar zonas»" in o.said[-1][1]
    # la que ya se analiza: el botón cambia y pide confirmar
    c = _cam(zones=ZONA, analyzed=True, agent_id=7, status=EN_LINEA)
    o = _oficina([c])
    assert o._analyze_label() == of.ANALYZE_OFF == "■ Dejar de analizar"
    o.set_analyze(None)
    assert "¿Dejar de analizar «Cámara 1»?" in o.said[0][1]
    assert o.posts[0] == ("/api/cameras/analyze", {"camera_ids": [1], "analyze": False})
    # varias a la vez; si alguna no se analiza, el botón las empieza
    o = _oficina([_cam(1, zones=ZONA), _cam(2, zones=ZONA, analyzed=True, agent_id=7)])
    assert o._analyze_label() == of.ANALYZE_ON
    o.set_analyze(None)
    assert o.posts[0][1] == {"camera_ids": [1, 2], "analyze": True}
    # sin nada elegido, lo dice
    o = _oficina([])
    o.set_analyze(None)
    assert o.posts == [] and "Elegí una cámara" in o.said[-1][1]


def test_al_guardar_las_zonas_ofrece_empezar_a_analizar():
    c = _cam(zones=ZONA)
    o = _oficina([c])
    o.zones_done(c, ZONA)
    assert "¿Empezar a analizarla ahora?" in o.said[0][1]
    assert o.posts[0] == ("/api/cameras/analyze", {"camera_ids": [1], "analyze": True}) and ("guardado", None) in o.posts
    # con «No» queda como estaba
    o = _oficina([c], answer=False)
    o.zones_done(c, ZONA)
    assert o.posts == []
    # la que ya se analiza (o se quedó sin zonas): no hay nada que preguntar
    o = _oficina([c])
    o.zones_done({**c, "analyzed": True}, ZONA)
    o.zones_done(c, [])
    assert o.said == [] and o.posts == []


def test_ya_no_hay_en_vivo_ni_clips_por_camara():
    assert not hasattr(of.Oficina, "clips_dialog") and not hasattr(of.Oficina, "upload_clip")
    o = _oficina([_cam()])
    o.cam_tree = types.SimpleNamespace(selection_set=lambda *a: None, get_children=lambda: ())

    class Menu:
        def __init__(self):
            self.labels = []

        def add_command(self, label=None, command=None):
            self.labels.append(label)

        def add_separator(self):
            pass
    for full in (False, True):
        m = Menu()
        o._fill_camera_menu(m, full=full)
        text = " | ".join(m.labels)
        assert "Clips" not in text and "Subir" not in text and "con la IA" not in text, text
        assert "Ver en la pestaña «En vivo»" in text and "Alertas que da…" in text and "Qué mide…" in text
    assert m.labels[:3] == ["✎ Marcar zonas…", "▶ Empezar a analizar", "Editar…"]
    # (un operador no borra cámaras)
    o.is_admin = False
    m = Menu()
    o._fill_camera_menu(m, full=True)
    assert "Eliminar…" not in m.labels
    # «Ver en la pestaña «En vivo»» pasa a esa pestaña con la cámara elegida
    seen = []
    o.v = types.SimpleNamespace(show_camera=lambda cid: seen.append(cid) or True)
    o.watch_live(o.cameras[0])
    assert seen == [1] and o.said == []
    o.v = types.SimpleNamespace(show_camera=lambda cid: False)  # (todavía no llegó la lista de «En vivo»)
    o.watch_live(o.cameras[0])
    assert "Abrí la pestaña «En vivo»" in o.said[-1][1]


def test_marcar_zonas_desde_en_vivo_usa_lo_que_se_esta_viendo(monkeypatch):
    import numpy as np
    c = _cam(7, zones=ZONA, analyzed=True)
    o = _oficina([c])
    o._tune_windows = []
    opened = []
    monkeypatch.setattr(of, "ZonesEditor", lambda app, cam, **kw: opened.append((cam["id"], kw)))
    frame = np.zeros((90, 160, 3), np.uint8)
    o.agent_of = lambda: types.SimpleNamespace(live_frame=lambda cid: frame if cid == 7 else None)
    o.zones_from_live(7)
    cam_id, kw = opened[0]
    assert cam_id == 7 and kw["image"] is frame and "En vivo" in kw["note"]
    assert "ya se analiza" in kw["confirm"]  # (las zonas nuevas rigen enseguida en la máquina: pregunta al guardar)
    # el video todavía no abrió: el editor busca la imagen por su cuenta (la guardada o una foto del vivo)
    o.agent_of = lambda: types.SimpleNamespace(live_frame=lambda cid: None)
    o.zones_from_live(7)
    assert opened[1][1]["image"] is None and opened[1][1]["note"] is None
    # una cámara que todavía no está en la lista de «Clientes»
    o.zones_from_live(99)
    assert len(opened) == 2 and "Todavía no llegó la lista" in o.said[-1][1]


def test_las_etiquetas_dicen_falsa_alarma():
    assert of.LABEL_NAMES["falso_positivo"] == "Falsa alarma"


def test_entrenar_dice_que_sigue():
    base = {"clips": 0, "theft_clips": 0, "marks": 0, "normal_clips": 0, "normal_minutes": 0.0, "unmarked": 0,
            **ei.MIN_KEYS}
    assert ei.guide_text(base, []).startswith("Empezá por «⬆ Subir clips…»")
    c = {**base, "clips": 4, "unmarked": 2}
    assert ei.guide_text(c, []).startswith("Falta decir qué pasa en 2 clips («Sin marcar»)")
    c = {**base, "clips": 4, "theft_clips": 1, "marks": 1, "normal_clips": 3}
    assert ei.guide_text(c, []).startswith("Faltan datos: al menos 5 hurtos marcados")
    c = {**base, "clips": 9, "theft_clips": 3, "marks": 6, "normal_clips": 3}
    assert ei.guide_text(c, []) == "Ya alcanza para una primera prueba: apretá «Entrenar» (abajo a la izquierda)."
    assert "apagada" in ei.guide_text(c, [{"id": 1, "active": False}])
    assert ei.guide_text(c, [{"id": 1, "active": False}, {"id": 2, "active": True}]).startswith(
        "La IA de movimientos está activada")
    # lo que falta marcar va antes que todo (aunque ya haya una IA activa)
    assert ei.guide_text({**c, "unmarked": 1}, [{"id": 2, "active": True}]).startswith("Falta decir qué pasa en 1 clip ")


# ---------------------------------------------------------------- con ventana
def _tk():
    tk = pytest.importorskip("tkinter")
    if os.name != "nt" and not os.environ.get("DISPLAY"):
        pytest.skip("sin pantalla (correr con xvfb-run)")
    return tk


def test_en_vivo_explica_que_hacer_y_tiene_marcar_zonas(tmp_path):
    _tk()
    from agent import ventana as vm
    v = vm.Ventana(tmp_path / "ventana.json")
    if not v.open:
        pytest.skip("sin Tk")
    try:
        said, asked = [], []
        v.messagebox = types.SimpleNamespace(showinfo=lambda *a, **k: said.append(a[1]), askyesno=lambda *a, **k: True)
        v.build_main("admin (administrador)", analyze=True, on_demand=True)
        v.on_zones = asked.append
        v.set_cameras([(1, "Minas", "Entrada", ""), (2, "Minas", "Caja", "Se analiza")])
        v.root.update()
        # nada elegido: el lugar del video dice qué hacer, con tildes (antes iba dibujado en la imagen, sin ellas)
        v.refresh([], "")
        assert "Elegí a la izquierda las cámaras" in v.canvas.cget("text")
        assert "«▶ Empezar a analizar»" in v.canvas.cget("text")
        # «✎ Marcar zonas» pide una sola cámara
        btn = next(w for w in _walk(v.root) if w.winfo_class() == "TButton" and w.cget("text") == "✎ Marcar zonas")
        btn.invoke()
        assert asked == [] and "una sola cámara" in said[-1]
        # desde «Clientes»: «Ver en la pestaña «En vivo»» la deja elegida y a la vista
        assert v.show_camera(2) and v.selected_ids() == {2}
        assert v.notebook.select() == str(v.live_tab)
        btn.invoke()
        assert asked == [2]
        assert not v.show_camera(99)  # (una que no está en la lista)
        # con una cámara elegida ya no está el texto
        import numpy as np
        v.refresh([("Caja", np.zeros((90, 160, 3), np.uint8), "")], "")
        assert v.canvas.cget("text") == ""
    finally:
        v.close()


def _walk(w):
    yield w
    for c in w.winfo_children():
        yield from _walk(c)


def test_la_pestana_clientes_entra_y_muestra_los_pasos(tmp_path):
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
        client = {"id": 3, "name": "Minas", "plans": ["hurto", "comercio"], "hours": {"lun": "09:00-17:00"}}
        solo = {"id": 4, "name": "Kiosco", "plans": ["comercio"]}
        cams = [_cam(1, name="Entrada", zones=ZONA),
                _cam(2, name="Caja", zones=ZONA, analyzed=True, agent_id=7, analyzer_name="maquina-1", status=EN_LINEA,
                     rules_default=["salida_sin_pago", "zona_restringida"], gestures_default=["mano_bolsillo"]),
                _cam(3, name="Mostrador", client_id=4, measures=["comercio"])]
        o = of.Oficina.__new__(of.Oficina)
        o.tk, o.ttk, o.notebook, o.is_admin = tk, ttk, nb, True
        o.v = types.SimpleNamespace(root=root, log_handler=types.SimpleNamespace(q=queue.Queue()))
        o.mb = types.SimpleNamespace(showinfo=lambda *a, **k: None, askyesno=lambda *a, **k: False)
        o.bg = types.SimpleNamespace(run=lambda fn, done=None, fail=None: None)
        o.api = types.SimpleNamespace()
        o.clients, o.cameras, o.alert_counts = [client, solo], cams, {}
        o._tune_windows, o.agent_of = [], None
        o.tab_clients = ttk.Frame(nb, padding=8)
        nb.add(o.tab_clients, text="  Clientes  ")
        o._clip_frame = lambda cam: None
        o.load_alerts = o.refresh_all = lambda *a, **k: None  # (al cambiar de pestaña se pide lo último)
        o._build_clients()
        o._fill_clients()
        root.update()
        # entra en una pantalla de 1366x768
        assert o.tab_clients.winfo_reqwidth() <= o.tab_clients.winfo_width()
        assert o.tab_clients.winfo_reqheight() <= o.tab_clients.winfo_height()
        texts = [str(w.cget("text")) for w in _walk(o.tab_clients) if w.winfo_class() in ("TButton", "TMenubutton")]
        assert "✎ Marcar zonas" in texts and of.ANALYZE_ON in texts
        assert not any("Clips" in t or t == "● En vivo" for t in texts), texts
        assert any(str(w.cget("text")) == of.CAMERA_STEPS for w in _walk(o.tab_clients) if w.winfo_class() == "TLabel")
        # las columnas: con dos planes se ve «Mide»; con uno solo, no
        assert o.cam_tree.cget("displaycolumns") == ("nombre", "estado", "zonas", "mide", "alertas")
        row = o.cam_tree.item("2")["values"]
        assert row[:4] == ["Caja", "Analizando · 5 cuadros/s", "✔ 1", "Hurto · Comercio"]
        assert o.cam_tree.item("1")["values"][1] == "No se analiza"
        # el botón dice lo que va a hacer con la elegida, y el detalle qué le falta
        o.cam_tree.selection_set("1")
        root.update()
        assert o.an_btn.cget("text") == of.ANALYZE_ON and str(o.an_btn.cget("style")) == "Primario.TButton"
        info = o.cam_info.get("1.0", "end")
        assert info.startswith("Entrada\nPaso 3: «▶ Empezar a analizar»") and "Zonas: Góndola (Góndola)" in info
        o.cam_tree.selection_set("2")
        root.update()
        assert o.an_btn.cget("text") == of.ANALYZE_OFF and str(o.an_btn.cget("style")) == "TButton"
        info = o.cam_info.get("1.0", "end")
        assert info.startswith("Caja\nLista: se analiza en una máquina")
        assert "Estado: Analizando · 5 cuadros/s · en maquina-1" in info
        assert "Alertas que da: salida sin pagar, zona restringida, gestos (bolsillo)" in info
        o.cl_tree.selection_set("4")
        root.update()
        assert o.cam_tree.cget("displaycolumns") == ("nombre", "estado", "zonas", "alertas")
        assert "ninguna se analiza todavía" in o.cam_title.cget("text")
    finally:
        root.destroy()
