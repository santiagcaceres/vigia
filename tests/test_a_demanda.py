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

    def catalog(self):
        return [{"id": 1, "client_name": "Minas", "name": "Caja", "here": False, "on_servers": False, "elsewhere": None},
                {"id": 2, "client_name": "Minas", "name": "Puerta", "here": False, "on_servers": True, "elsewhere": None},
                {"id": 3, "client_name": "Minas", "name": "Depósito", "here": False, "on_servers": False,
                 "elsewhere": "PC de ana"}]

    def watch(self, ids):
        self.sent.append(list(ids))
        return {"camera_ids": [i for i in ids if i != 2], "on_servers": [i for i in ids if i == 2]}


def test_la_app_avisa_que_abrio_y_que_cerro(tmp_path):
    from agent.__main__ import Agent

    ag = Agent("http://127.0.0.1:9", "token", tmp_path, None)
    ag.window, ag.on_demand, ag.client = FakeWindow(), True, FakeClient()

    def tick():
        ag._demand_tick(time.monotonic())
        for _ in range(100):
            if not ag._watch_busy:
                break
            time.sleep(0.01)

    # al entrar: avisa que no hay nada abierto (lo que quedó de antes se libera) y no abre nada hasta entonces
    ag.apply_config({"cameras": [{"id": 9, "name": "vieja", "connection": {}}]})
    assert ag.workers == {} and ag._cams_cfg == {}
    tick()
    assert ag.client.sent == [[]] and ag._force_config
    tick()
    assert ag.client.sent == [[]]  # sin cambios no vuelve a avisar

    ag.window.sel, ag.window.selected_at = {1, 2}, time.monotonic()
    tick()
    assert ag.client.sent == [[]]  # recién elegidas: espera un momento por si sigue eligiendo
    ag.window.selected_at -= 1
    tick()
    assert ag.client.sent[-1] == [1, 2]
    assert [c[3] for c in ag.window_cameras()] == ["Abriendo...", "En una máquina", "Abierta en PC de ana"]
    assert [t[2] for t in ag.window_tiles({1, 2})] == ["Abriendo...", "La analiza una maquina de analisis"]

    ag.shutdown()  # al cerrar la app se liberan las cámaras
    assert ag.client.sent[-1] == []


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
