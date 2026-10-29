"""Usuarios que ven y analizan las cámaras que les asigna el admin (sin elegir ni vincular PCs)."""

import pytest
from fastapi.testclient import TestClient

from server.main import Settings, create_app

ADMIN = ("admin", "clave-admin-123")


@pytest.fixture
def app(tmp_path, db_url):
    return create_app(Settings(data_dir=tmp_path / "data", database_url=db_url, admin_password=ADMIN[1],
                               serve_public=False))


def login(app, user, pwd):
    c = TestClient(app)
    assert c.post("/api/login", json={"username": user, "password": pwd}).status_code == 200
    return c


def test_admin_asigna_camaras_y_el_empleado_las_analiza_en_su_pc(app):
    with TestClient(app) as a:
        a.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]})
        cid = a.post("/api/clients", json={"name": "Minas"}).json()["id"]
        cam = lambda n, **kw: a.post("/api/cameras", json={"client_id": cid, "name": n, "brand": "ip",  # noqa: E731
                                                           "host": "dvr", "password": "x", **kw}).json()
        c1, c2, c3 = cam("Caja", profile="caja"), cam("Góndola"), cam("Calle", profile="exterior")
        ana = a.post("/api/users", json={"username": "ana", "password": "clave-ana-123", "role": "entrenador"}).json()
        # asignar desde el usuario (lista de cámaras) y desde la cámara (operator_id)
        r = a.put(f"/api/users/{ana['id']}/cameras", json={"camera_ids": [c1["id"], c2["id"]]}).json()
        assert sorted(r["camera_ids"]) == sorted([c1["id"], c2["id"]])
        r = a.put(f"/api/cameras/{c3['id']}", json={"client_id": cid, "name": "Calle", "brand": "ip", "host": "dvr",
                                                    "profile": "exterior", "operator_id": ana["id"]}).json()
        assert r["operator_name"] == "ana" and r["profile"] == "exterior"
        users = {u["username"]: u for u in a.get("/api/users").json()}
        assert len(users["ana"]["camera_ids"]) == 3
        # la empleada entra: su PC se crea sola y recibe solo sus cámaras, con las reglas de cada tipo
        e = login(app, "ana", "clave-ana-123")
        tok = e.post("/api/agent/session").json()
        assert len(tok["camera_ids"]) == 3
        cfg = TestClient(app).get("/api/agent/config", headers={"X-Agent-Token": tok["token"]}).json()
        by_name = {x["name"]: x for x in cfg["cameras"]}
        assert set(by_name) == {"Caja", "Góndola", "Calle"}
        assert by_name["Caja"]["analysis"]["rules"]["enabled_rules"] == ["zona_restringida", "fuera_de_horario"]
        # general: todas, siempre explícitas (una máquina vieja sin la lista también miraría merodeo, que ya no existe)
        assert by_name["Góndola"]["analysis"]["rules"]["enabled_rules"] == ["salida_sin_pago", "zona_restringida",
                                                                            "fuera_de_horario"]
        assert by_name["Calle"]["analysis"]["vehicles"] is True
        # un empleado editando una cámara no le cambia la asignación
        e.put(f"/api/cameras/{c1['id']}", json={"client_id": cid, "name": "Caja 1", "brand": "ip", "host": "dvr",
                                                "operator_id": None})
        assert next(x for x in a.get("/api/cameras").json() if x["id"] == c1["id"])["operator_name"] == "ana"
        # mandar la cámara tal como vino (operator_id None) no la desasigna; 0 sí
        cam1 = next(x for x in a.get("/api/cameras").json() if x["id"] == c1["id"])
        a.put(f"/api/cameras/{c1['id']}", json={**cam1, "operator_id": None})
        assert next(x for x in a.get("/api/cameras").json() if x["id"] == c1["id"])["operator_name"] == "ana"
        a.put(f"/api/cameras/{c1['id']}", json={**cam1, "operator_id": 0})
        assert next(x for x in a.get("/api/cameras").json() if x["id"] == c1["id"])["operator_name"] is None
        # sacarle cámaras
        a.put(f"/api/users/{ana['id']}/cameras", json={"camera_ids": [c3["id"]]})
        cfg = TestClient(app).get("/api/agent/config", headers={"X-Agent-Token": tok["token"]}).json()
        assert [x["name"] for x in cfg["cameras"]] == ["Calle"]
        # entrar de nuevo rota el token: la sesión anterior de la PC deja de servir
        tok2 = e.post("/api/agent/session").json()
        assert TestClient(app).get("/api/agent/config", headers={"X-Agent-Token": tok["token"]}).status_code == 401
        assert tok2["id"] == tok["id"]
        # un cliente no puede pedir PC
        a.post("/api/users", json={"username": "dueno", "password": "clave-dueno-1", "role": "cliente", "client_id": cid})
        assert login(app, "dueno", "clave-dueno-1").post("/api/agent/session").status_code == 403


def test_deposito_marca_toda_la_imagen_como_restringida(app):
    with TestClient(app) as a:
        a.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]})
        cid = a.post("/api/clients", json={"name": "Local"}).json()["id"]
        cam = a.post("/api/cameras", json={"client_id": cid, "name": "Depósito", "brand": "ip", "host": "dvr",
                                           "profile": "deposito", "operator_id": 1}).json()
        tok = a.post("/api/agent/session").json()["token"]
        an = TestClient(app).get("/api/agent/config", headers={"X-Agent-Token": tok}).json()["cameras"][0]["analysis"]
        assert cam["profile"] == "deposito"
        assert [z["type"] for z in an["zones"]] == ["restricted"]
