"""Cámaras a demanda: en la app de escritorio el admin elige en «En vivo» qué cámara abrir y su PC la analiza
mientras esté abierta. Nadie asigna cámaras."""

import time

import pytest
from fastapi.testclient import TestClient

from server.main import Settings, create_app

ADMIN = ("admin", "clave-admin-123")


@pytest.fixture
def app(tmp_path, db_url):
    return create_app(Settings(data_dir=tmp_path / "data", database_url=db_url, admin_password=ADMIN[1],
                               serve_public=False))


def test_el_admin_abre_y_cierra_camaras_sin_asignar(app):
    with TestClient(app) as a:
        a.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]})
        cl = a.post("/api/clients", json={"name": "Minas"}).json()
        cams = [a.post("/api/cameras", json={"client_id": cl["id"], "name": f"Cámara {i}", "host": "10.0.0.1",
                                             "channel": i}).json() for i in (1, 2, 3)]
        nube = a.post("/api/cameras", json={"client_id": cl["id"], "name": "En la nube", "host": "10.0.0.1",
                                            "channel": 4, "analysis_location": "servidores"}).json()
        pc = a.post("/api/agent/session").json()
        h = {"X-Agent-Token": pc["token"]}

        lista = a.get("/api/agent/catalog", headers=h).json()["cameras"]
        assert [c["name"] for c in lista] == ["Cámara 1", "Cámara 2", "Cámara 3", "En la nube"]
        assert not any(c["here"] for c in lista) and lista[3]["on_servers"]

        rev = a.get("/api/agent/config", headers=h).json()["rev"]
        res = a.put("/api/agent/watch", headers=h, json={"camera_ids": [cams[0]["id"], cams[1]["id"], nube["id"]]}).json()
        assert res == {"camera_ids": [cams[0]["id"], cams[1]["id"]], "on_servers": [nube["id"]]}
        cfg = a.get("/api/agent/config", headers=h).json()
        assert [c["id"] for c in cfg["cameras"]] == [cams[0]["id"], cams[1]["id"]] and cfg["rev"] != rev

        # cerrar una: queda libre
        a.put("/api/agent/watch", headers=h, json={"camera_ids": [cams[1]["id"]]})
        assert [c["id"] for c in a.get("/api/agent/config", headers=h).json()["cameras"]] == [cams[1]["id"]]
        lista = {c["id"]: c for c in a.get("/api/agent/catalog", headers=h).json()["cameras"]}
        assert lista[cams[1]["id"]]["here"] and not lista[cams[0]["id"]]["here"]

        # una PC vieja (sin usuario) o una máquina de análisis no abren cámaras a demanda
        vieja = a.post("/api/agents", json={"name": "PC vieja", "client_id": None}).json()
        assert a.get("/api/agent/catalog", headers={"X-Agent-Token": vieja["token"]}).status_code == 403
        assert a.put("/api/agent/watch", headers={"X-Agent-Token": vieja["token"]},
                     json={"camera_ids": [cams[0]["id"]]}).status_code == 403


class FakeWindow:
    def __init__(self):
        self.sel: set[int] = set()
        self.selected_at = 0.0

    def selected_ids(self):
        return set(self.sel)


class FakeClient:
    def __init__(self):
        self.sent: list[list[int]] = []
        self.asked: list[list[int]] = []

    def catalog(self):
        return [{"id": 1, "client_name": "Minas", "name": "Caja", "here": False, "on_servers": False, "elsewhere": None},
                {"id": 2, "client_name": "Minas", "name": "Puerta", "here": False, "on_servers": True,
                 "elsewhere": "maquina-1"},
                {"id": 3, "client_name": "Minas", "name": "Depósito", "here": False, "on_servers": False,
                 "elsewhere": None}]

    def watch(self, ids):
        self.sent.append(list(ids))
        return {"camera_ids": list(ids), "on_servers": []}

    def view(self, ids):  # la 3 la borraron mientras tanto: no vuelve
        self.asked.append(list(ids))
        conn = {"brand": "hikvision", "host": "127.0.0.1", "source_url": "/no/existe.mp4", "channel": 1}
        return [{"id": i, "client_name": "Minas", "name": n, "analyzed": i == 2, "analyzer": "maquina-1" if i == 2 else None,
                 "connection": conn} for i, n in ((1, "Caja"), (2, "Puerta")) if i in ids]


def test_en_vivo_solo_mira_y_no_analiza(tmp_path):
    """«En vivo» de la app: lo elegido se mira (video directo, o la imagen de la máquina si la analiza una
    máquina); la PC no analiza nada."""
    from agent.__main__ import Agent
    from agent.vista import RemoteView, ViewWorker

    ag = Agent("http://127.0.0.1:9", "token", tmp_path, None)
    ag.window, ag.on_demand, ag.client = FakeWindow(), True, FakeClient()
    ag.window.set_status = lambda *a, **k: None
    ag.beat = lambda cid: {"frame_url": None, "agent_online": True}

    def tick():
        ag._demand_tick(time.monotonic())
        for _ in range(200):
            if not ag._watch_busy and not ag._view_pending:
                break
            time.sleep(0.01)

    # la app no analiza: aunque el servidor le mande cámaras, no abre ninguna para analizar
    ag.apply_config({"cameras": [{"id": 9, "name": "vieja", "connection": {}}]})
    assert ag.workers == {} and ag._cams_cfg == {}
    tick()
    assert ag.client.sent == [[]]  # al entrar le avisa al servidor que esta PC no tiene nada asignado

    ag.window.sel, ag.window.selected_at = {1, 2, 3}, time.monotonic()
    tick()
    assert ag.client.asked == []  # recién elegidas: espera un momento por si sigue eligiendo
    ag.window.selected_at -= 1
    tick()
    tick()
    assert ag.client.asked == [[1, 2, 3]]
    assert isinstance(ag.viewers[1], ViewWorker) and isinstance(ag.viewers[2], RemoteView)
    tick()
    assert ag.client.asked == [[1, 2, 3]]  # la que no existe más no se pide de nuevo
    estados = [c[3] for c in ag.window_cameras()]
    assert estados[1] == "Esperando la imagen de la máquina..." and estados[2] == "No disponible"
    assert [t[0] for t in ag.window_tiles({1, 2})] == ["Minas · Caja", "Minas · Puerta"]
    assert ag.workers == {}  # nada se analiza en la PC

    ag.window.sel, ag.window.selected_at = {2}, time.monotonic() - 1
    tick()
    assert list(ag.viewers) == [2]  # la Caja se cerró
    ag.shutdown()


def test_elegir_que_se_analiza_y_mirar_cualquiera(app):
    with TestClient(app) as a:
        a.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]})
        cl = a.post("/api/clients", json={"name": "Minas"}).json()
        ids = [a.post("/api/cameras", json={"client_id": cl["id"], "name": f"Cámara {i}", "host": "10.0.0.1",
                                            "channel": i}).json()["id"] for i in (1, 2, 3)]
        assert not any(c["analyzed"] for c in a.get("/api/cameras").json())  # nueva = no se analiza
        assert a.post("/api/cameras/analyze", json={"camera_ids": ids[:2], "analyze": True}).json()["changed"] == 2
        cams = {c["id"]: c for c in a.get("/api/cameras").json()}
        assert cams[ids[0]]["analyzed"] and cams[ids[1]]["analyzed"] and not cams[ids[2]]["analyzed"]
        a.post("/api/cameras/analyze", json={"camera_ids": [ids[1]], "analyze": False})
        cams = {c["id"]: c for c in a.get("/api/cameras").json()}
        assert not cams[ids[1]]["analyzed"] and cams[ids[1]]["agent_id"] is None

        pc = a.post("/api/agent/session").json()
        h = {"X-Agent-Token": pc["token"]}
        vista = a.post("/api/agent/view", headers=h, json={"camera_ids": [ids[0], ids[2]]}).json()["cameras"]
        assert {v["id"]: v["analyzed"] for v in vista} == {ids[0]: True, ids[2]: False}
        assert all(v["connection"]["host"] == "10.0.0.1" for v in vista)
        assert a.get("/api/agent/config", headers=h).json()["cameras"] == []  # la PC no analiza nada
        # una PC vieja (sin usuario) no puede mirar
        vieja = a.post("/api/agents", json={"name": "PC vieja", "client_id": None}).json()
        assert a.post("/api/agent/view", headers={"X-Agent-Token": vieja["token"]},
                      json={"camera_ids": [ids[0]]}).status_code == 403


def test_borrar_usuario_y_no_repetir_camaras(app):
    with TestClient(app) as a, TestClient(app) as b:
        a.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]})
        cl = a.post("/api/clients", json={"name": "Minas"}).json()
        hc = {"client_id": cl["id"], "brand": "hikconnect", "hpp_serial": "E02380030", "hc_resource_id": "r1"}
        uno = a.post("/api/cameras", json={**hc, "name": "Caja", "channel": 1})
        assert uno.status_code == 201
        rep = a.post("/api/cameras", json={**hc, "name": "Caja otra vez", "channel": 1})
        assert rep.status_code == 409 and "Caja" in rep.json()["detail"]
        dos = a.post("/api/cameras", json={**hc, "name": "Puerta", "channel": 2}).json()
        assert a.put(f"/api/cameras/{dos['id']}", json={**hc, "name": "Puerta", "channel": 1}).status_code == 409
        assert a.put(f"/api/cameras/{dos['id']}", json={**hc, "name": "Puerta", "channel": 2}).status_code == 200

        ana = a.post("/api/users", json={"username": "ana", "password": "clave-ana-123", "role": "entrenador"}).json()
        b.post("/api/login", json={"username": "ana", "password": "clave-ana-123"})
        pc = b.post("/api/agent/session").json()
        b.put("/api/agent/watch", headers={"X-Agent-Token": pc["token"]}, json={"camera_ids": [dos["id"]]})
        assert next(c for c in a.get("/api/cameras").json() if c["id"] == dos["id"])["agent_id"] == pc["id"]

        me = a.get("/api/me").json()
        assert a.delete(f"/api/users/{me['id']}").status_code == 409  # no se borra a sí mismo
        assert a.delete(f"/api/users/{ana['id']}").status_code == 204
        assert "ana" not in [u["username"] for u in a.get("/api/users").json()]
        assert pc["id"] not in [x["id"] for x in a.get("/api/agents").json()]  # su PC se fue con ella
        assert next(c for c in a.get("/api/cameras").json() if c["id"] == dos["id"])["agent_id"] is None
        assert b.get("/api/me").status_code == 401  # su sesión ya no sirve


def test_una_sola_app_abierta_por_pc(tmp_path):
    from agent.__main__ import single_instance

    primera = single_instance(tmp_path, wait_s=0)
    assert primera is not None
    assert single_instance(tmp_path, wait_s=0) is None  # la segunda copia no arranca
    primera.close()  # se cerró la primera (o se reinició para actualizar)
    segunda = single_instance(tmp_path, wait_s=0)
    assert segunda is not None
    segunda.close()


def test_cerrar_una_camara_trabada_no_congela_la_ventana(tmp_path):
    """Con ventana, cerrar una cámara no espera a que termine su hilo (si estaba abriendo el video, la espera
    dejaba la ventana sin responder: no andaban ni los botones). Al cerrar la app sí se espera."""
    import threading

    from agent.__main__ import Agent

    ag = Agent("http://127.0.0.1:9", "token", tmp_path, None)
    ag.window = FakeWindow()
    suelta = threading.Event()

    class Trabada:
        name, fingerprint = "Minas · Caja", "x"

        def __init__(self):
            self.thread = threading.Thread(target=suelta.wait, args=(10,), daemon=True)
            self.thread.start()

        def stop(self):
            pass

    w = Trabada()
    ag.workers[1] = w
    t0 = time.monotonic()
    ag.apply_config({"cameras": []})
    assert time.monotonic() - t0 < 1 and 1 not in ag.workers and ag._stopping == [w]
    suelta.set()
    w.thread.join(2)
    ag.apply_live([])  # limpia las que ya terminaron
    assert ag._stopping == []
