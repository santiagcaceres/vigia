"""Varios usuarios con la app abierta a la vez (pedido del dueño, 5 nov: «un botón que diga actualizar clientes y
canales que sirva para coordinar todo entre usuarios al mismo tiempo… o cómo podemos hacer esto de forma
instantánea»). Lo que cambia uno le llega a los demás en la próxima respuesta del estado (la app lo pide cada 10 s),
sin pedidos de más; el botón «⟳ Actualizar clientes y cámaras» es para no esperar."""

import json
import os
import types

import pytest
from fastapi.testclient import TestClient

from tests.test_seguridad import app, login, setup  # noqa: F401 (app es una fixture)


def status(app, tok) -> dict:
    r = TestClient(app).post("/api/agent/status", headers=tok, json={"cameras": []})
    assert r.status_code == 200, r.text
    return r.json()


def test_el_estado_avisa_cambios_de_clientes_camaras_y_alertas(app):
    a, o, tok, cam_a, *_ = setup(app)
    s0 = status(app, tok)
    assert "alerts_rev" in s0
    # otro usuario crea un cliente (antes no subía la versión: los demás no lo veían hasta cambiar de pestaña)
    nuevo = a.post("/api/clients", json={"name": "Almacén nuevo"}).json()["id"]
    s1 = status(app, tok)
    assert s1["rev"] != s0["rev"] and s1["alerts_rev"] == s0["alerts_rev"]
    # le cambia el nombre a una cámara
    assert a.post("/api/cameras/names", json={"cameras": [{"id": cam_a["id"], "name": "Caja 1"}]}).status_code == 200
    s2 = status(app, tok)
    assert s2["rev"] != s1["rev"]
    # lo borra
    assert a.delete(f"/api/clients/{nuevo}").status_code in (200, 204)
    s3 = status(app, tok)
    assert s3["rev"] != s2["rev"]
    # llega una alerta
    meta = {"uuid": "sync1", "camera_id": cam_a["id"], "type": "zona_restringida", "description": "x"}
    eid = TestClient(app).post("/api/agent/events", headers=tok, data={"meta": json.dumps(meta)},
                               files={"snapshot": ("s.jpg", b"\xff\xd8\xff", "image/jpeg")}).json()["id"]
    s4 = status(app, tok)
    assert s4["alerts_rev"] != s3["alerts_rev"] and s4["rev"] == s3["rev"]  # (las máquinas no piden la config)
    # alguien la revisa
    assert a.patch(f"/api/events/{eid}", json={"label": "normal"}).status_code == 200
    s5 = status(app, tok)
    assert s5["alerts_rev"] != s4["alerts_rev"]
    assert status(app, tok)["alerts_rev"] == s5["alerts_rev"]  # sin cambios, igual
    # una PC sin usuario (agente sin ventana) no lo recibe: no le sirve y es una consulta menos
    pc = a.post("/api/agents", json={"name": "PC sin ventana", "client_id": None}).json()
    assert "alerts_rev" not in status(app, {"X-Agent-Token": pc["token"]})


def test_la_app_avisa_a_las_pestanas_solo_cuando_cambia():
    from agent.__main__ import Agent

    ag = Agent.__new__(Agent)
    ag._seen, ag._catalog_at, ag._views_stale, got = {}, 5.0, False, []
    ag.on_remote_change = got.append
    ag._note_changes({"rev": "3", "alerts_rev": "10."})  # lo primero es la base: no avisa
    ag._note_changes({"rev": "3", "alerts_rev": "10."})
    assert got == [] and ag._catalog_at == 5.0 and not ag._views_stale
    ag._note_changes({"rev": 4, "alerts_rev": "10."})
    assert got == [{"config"}] and ag._catalog_at < 0 and ag._views_stale  # «En vivo» y las zonas, ya
    ag._note_changes({"rev": "4", "alerts_rev": "11."})
    assert got[-1] == {"alerts"}
    ag._note_changes({"rev": "4"})  # una respuesta sin alertas (servidor viejo): nada
    assert len(got) == 2
    ag.on_remote_change = lambda kinds: 1 / 0  # una pestaña que falla no frena la app
    ag._note_changes({"rev": "5"})


def test_las_pestanas_se_ponen_al_dia_segun_lo_que_cambio():
    import agent.oficina as of

    calls: list = []
    off = of.Oficina.__new__(of.Oficina)
    off.v = types.SimpleNamespace(open=True)
    off.notebook = types.SimpleNamespace(select=lambda: "alertas")
    off.tab_clients, off.tab_alerts = "clientes", "alertas"
    off.api = types.SimpleNamespace(get=lambda path, **k: calls.append(path) or [])
    off.bg = types.SimpleNamespace(run=lambda fn, done=None, fail=None: done(fn()))
    off._fill_clients = lambda keep_info=False: calls.append(("clientes", keep_info))
    off._fill_alert_filters = lambda: calls.append("filtros")
    off.load_alerts = lambda remote=False: calls.append("alertas" if remote else "alertas a mano")
    off.remote_changed({"alerts"})  # mirando Alertas: solo esa lista
    assert calls == ["alertas"]
    calls.clear()
    off.remote_changed({"config"})  # un cliente o cámara nuevos: la lista de clientes (sin tocar lo elegido)
    assert "/api/clients" in calls and ("clientes", True) in calls and "filtros" in calls and "alertas" not in calls
    calls.clear()
    off.notebook.select = lambda: "clientes"
    off.remote_changed({"alerts"})  # mirando Clientes: cambian los «Por revisar»
    assert "/api/stats/alerts" in calls and "alertas" not in calls
    calls.clear()
    off.v.open = False
    off.remote_changed({"config", "alerts"})
    assert calls == []


# ------------------------------------------------------------------ la lista de alertas (con Tk)
def _tk():
    tk = pytest.importorskip("tkinter")
    if os.name != "nt" and not os.environ.get("DISPLAY"):
        pytest.skip("sin pantalla (correr con xvfb-run)")
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"sin Tk: {exc}")
    return tk, root


def _ev(i: int) -> dict:
    return {"id": i, "occurred_at": "2026-11-05T10:00:00", "client_name": "Almacén", "camera_name": "Caja",
            "type": "zona_restringida", "label": None, "snapshot_url": None, "clip_url": None, "description": ""}


def test_la_lista_de_alertas_se_recarga_sin_cortar_el_clip():
    tk, root = _tk()
    from tkinter import ttk

    import agent.oficina as of
    try:
        off = of.Oficina.__new__(of.Oficina)
        off.tk, off.ttk = tk, ttk
        off.al_tree = ttk.Treeview(root, columns=("fecha", "cliente", "camara", "tipo", "etiqueta"), show="headings")
        off.al_tree.pack()
        off.al_tree.bind("<<TreeviewSelect>>", lambda e: off.show_alert())
        off.al_state, off.al_client = tk.StringVar(value="Sin revisar"), tk.StringVar(value="Todos")
        off.clients = []
        off.al_count, off.al_title, off.al_desc = ttk.Label(root), ttk.Label(root), ttk.Label(root)
        stops: list = []
        off.stop_clip = lambda: stops.append(1)
        items = [_ev(1), _ev(2), _ev(3)]
        off.api = types.SimpleNamespace(get=lambda path, **k: {"items": list(items), "total": len(items)})
        off.bg = types.SimpleNamespace(run=lambda fn, done=None, fail=None: done(fn()))
        off.err = lambda exc: pytest.fail(str(exc))
        off.load_alerts()
        root.update()
        off.al_tree.selection_set("2")
        root.update()
        assert off._al_shown == 2 and len(stops) == 1
        # llega otra alerta: la lista se recarga, sigue elegida la 2 y el clip no se corta
        items.insert(0, _ev(4))
        off.load_alerts()
        root.update()
        assert off.al_tree.selection() == ("2",) and len(stops) == 1
        # otro usuario revisó la 2 (ya no entra en «Sin revisar»): sale de la lista pero queda a la vista, con el aviso
        items[:] = [x for x in items if x["id"] != 2]
        off.load_alerts(remote=True)
        root.update()
        assert off.al_tree.selection() == () and len(stops) == 1 and off._al_shown == 2
        assert "¿la revisó otro usuario?" in str(off.al_desc.cget("text"))
        # elegir otra sí cambia
        off.al_tree.selection_set("3")
        root.update()
        assert off._al_shown == 3 and len(stops) == 2

        # etiquetar mientras la lista se recarga (llegó otra alerta en el medio): antes daba «Item 3 not found»
        def patch(path, body):
            items[:] = [x for x in items if x["id"] != 3]
            off.load_alerts()
            return {"label": body["label"]}
        off.api.patch = patch
        off.set_label("normal")
        root.update()
        assert "3" not in off.al_tree.get_children()
        assert off.al_tree.selection() and off._al_shown == int(off.al_tree.selection()[0])  # pasó a la siguiente
    finally:
        root.destroy()
