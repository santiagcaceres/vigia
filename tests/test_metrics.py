"""Métricas por cliente: colector en la PC central, carga y consulta en el servidor, planes."""

from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from core.metrics import MetricsCollector, empty_bucket, merge_bucket
from core.rules import parse_zones
from core.tracker import Track
from server.main import Settings, create_app

ADMIN = ("admin", "clave-admin-123")
ZONES = parse_zones([
    {"name": "entrada", "type": "entrance", "polygon": [[0, 0], [1, 0], [1, 0.2], [0, 0.2]]},
    {"name": "gondola", "type": "shelf", "polygon": [[0, 0.4], [0.5, 0.4], [0.5, 0.8], [0, 0.8]]},
    {"name": "surtidor 1", "type": "pump", "polygon": [[0.5, 0.4], [1, 0.4], [1, 0.8], [0.5, 0.8]]},
    {"name": "tienda", "type": "shop", "polygon": [[0, 0.85], [1, 0.85], [1, 1], [0, 1]]},
])
W, H = 1000, 1000


def person(tid, x, y, first, ts, label="persona", missed=0):
    return Track(tid, (x - 20, y - 80, x + 20, y), label, first, ts, hits=10, missed=missed)


def test_colector_cuenta_personas_destinos_y_espera_del_pistero():
    m = MetricsCollector(ZONES)
    wall0 = datetime(2026, 10, 22, 10, 0, 0)
    ts = 0.0
    # persona 1: entra por la entrada (0-3 s), va a la góndola (3-13 s) y se va. 2 cuadros por segundo.
    # vehículo 7: llega al surtidor en el segundo 5; la persona 2 (pistero) llega al surtidor en el 11.
    for i in range(40):
        ts = i / 2
        tracks = []
        if ts < 13:
            y = 100 if ts < 3 else 600
            tracks.append(person(1, 200, y, 0.0, ts))
        if 5 <= ts < 18:
            tracks.append(person(7, 750, 600, 5.0, ts, label="vehiculo"))
        if 11 <= ts < 18:
            tracks.append(person(2, 700, 700, 11.0, ts))
        m.update(tracks, ts, (W, H), wall0 + timedelta(seconds=ts))
    out = m.close()
    assert len(out) == 1 and out[0]["bucket_start"] == "2026-10-22T10:00:00"
    d = out[0]["data"]
    assert d["people"] == 2 and d["entries"] == 1 and d["vehicles"] == 1
    assert d["people_max"] == 2 and d["vehicles_max"] == 1
    assert 12 <= d["people_dwell_s"] <= 20  # 13 s de la persona 1 + ~7 s del pistero
    assert d["zones"]["gondola"]["visits"] == 1 and 9 <= d["zones"]["gondola"]["dwell_s"] <= 11
    assert d["zones"]["entrada"]["visits"] == 1
    assert d["pump_vehicles"] == 1 and d["pump_wait_n"] == 1 and 5.5 <= d["pump_wait_s"] <= 6.5
    assert d["pump_unattended"] == 0 and d["shop_visits"] == 0
    # visitas cortas (ruido) no cuentan
    m2 = MetricsCollector(ZONES)
    m2.update([person(1, 200, 600, 0.0, 0.0)], 0.0, (W, H), wall0)
    m2.update([person(1, 200, 600, 0.0, 0.5)], 0.5, (W, H), wall0)
    m2.update([], 1.0, (W, H), wall0)
    assert m2.close()[0]["data"]["people"] == 0


def test_merge_de_cubos():
    a = merge_bucket(empty_bucket(), {"people": 2, "people_max": 3, "zones": {"g": {"type": "shelf", "visits": 1, "dwell_s": 4}}})
    b = {"people": 1, "people_max": 1, "zones": {"g": {"type": "shelf", "visits": 2, "dwell_s": 6}, "c": {"type": "checkout", "visits": 1, "dwell_s": 1}}}
    m = merge_bucket(a, b)
    assert m["people"] == 3 and m["people_max"] == 3
    assert m["zones"]["g"] == {"type": "shelf", "visits": 3, "dwell_s": 10} and m["zones"]["c"]["visits"] == 1


@pytest.fixture
def app(tmp_path, db_url):
    return create_app(Settings(data_dir=tmp_path / "data", database_url=db_url, admin_password=ADMIN[1],
                               serve_public=False))


def test_planes_carga_y_consulta_de_metricas(app):
    with TestClient(app) as c:
        c.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]})
        cid = c.post("/api/clients", json={"name": "Estación Norte", "plans": ["comercio", "estacion"],
                                            "timezone": "America/Montevideo"}).json()["id"]
        assert c.get("/api/clients").json()[0]["plans"] == ["comercio", "estacion"]
        ag = c.post("/api/agents", json={"name": "central", "client_id": None}).json()
        h = {"X-Agent-Token": ag["token"]}
        cam = c.post("/api/cameras", json={"client_id": cid, "agent_id": ag["id"], "name": "Playa", "host": "h",
                                           "password": "x", "mode": "vivo"}).json()
        # el agente recibe que debe detectar vehículos (plan estación)
        cfg = c.get("/api/agent/config", headers=h).json()
        assert cfg["cameras"][0]["analysis"]["vehicles"] is True and "estacion" in cfg["cameras"][0]["analysis"]["plans"]
        today = datetime.now().replace(minute=0, second=0, microsecond=0)
        b1 = {"camera_id": cam["id"], "bucket_start": today.isoformat(),
              "data": {"people": 5, "people_dwell_s": 100, "people_max": 2, "vehicles": 3, "pump_vehicles": 2,
                       "pump_wait_s": 30, "pump_wait_n": 2, "shop_visits": 1,
                       "zones": {"tienda": {"type": "shop", "visits": 1, "dwell_s": 40}}}}
        r = c.post("/api/agent/metrics", json={"buckets": [b1]}, headers=h).json()
        assert r["saved"] == 1
        # reenvío del mismo cubo: reemplaza, no duplica
        b1["data"]["people"] = 6
        c.post("/api/agent/metrics", json={"buckets": [b1]}, headers=h)
        b2 = {**b1, "bucket_start": (today - timedelta(hours=1)).isoformat(), "data": {"people": 4, "vehicles": 1}}
        c.post("/api/agent/metrics", json={"buckets": [b2]}, headers=h)
        m = c.get("/api/metrics", params={"client_id": cid, "days": 1}).json()
        assert m["plans"] == ["comercio", "estacion"] and m["total"]["people"] == 10 and m["total"]["vehicles"] == 4
        assert m["total"]["pump_wait_n"] == 2 and m["total"]["zones"]["tienda"]["visits"] == 1
        assert len(m["by_hour"]) == 2 and m["by_camera"][0]["name"] == "Playa"
        # una cámara de otro agente no se acepta
        ag2 = c.post("/api/agents", json={"name": "otra", "client_id": None}).json()
        assert c.post("/api/agent/metrics", json={"buckets": [b1]}, headers={"X-Agent-Token": ag2["token"]}).json()["saved"] == 0
        # el rol cliente ve solo lo suyo
        c.post("/api/users", json={"username": "dueno", "password": "clave-dueno-1", "role": "cliente", "client_id": cid})
        u = TestClient(app)
        u.post("/api/login", json={"username": "dueno", "password": "clave-dueno-1"})
        assert u.get("/api/metrics", params={"days": 7}).json()["total"]["people"] == 10
        otro = c.post("/api/clients", json={"name": "Otro"}).json()["id"]
        assert u.get("/api/metrics", params={"client_id": otro}).status_code == 403


def test_borrar_cliente_con_todo(app):
    with TestClient(app) as c:
        c.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]})
        cid = c.post("/api/clients", json={"name": "Viejo"}).json()["id"]
        keep = c.post("/api/clients", json={"name": "Queda"}).json()["id"]
        ag = c.post("/api/agents", json={"name": "central", "client_id": None}).json()
        h = {"X-Agent-Token": ag["token"]}
        cam = c.post("/api/cameras", json={"client_id": cid, "agent_id": ag["id"], "name": "A", "host": "h", "password": "x"}).json()
        cam2 = c.post("/api/cameras", json={"client_id": keep, "agent_id": ag["id"], "name": "B", "host": "h", "password": "x"}).json()
        now = datetime.now().replace(minute=0, second=0, microsecond=0).isoformat()
        c.post("/api/agent/metrics", json={"buckets": [{"camera_id": cam["id"], "bucket_start": now, "data": {"people": 3}},
                                                       {"camera_id": cam2["id"], "bucket_start": now, "data": {"people": 1}}]}, headers=h)
        c.post("/api/users", json={"username": "dueno", "password": "clave-dueno-1", "role": "cliente", "client_id": cid})
        # un empleado no puede borrar clientes
        c.post("/api/users", json={"username": "empleado", "password": "clave-empl-1", "role": "entrenador"})
        e = TestClient(app)
        e.post("/api/login", json={"username": "empleado", "password": "clave-empl-1"})
        assert e.delete(f"/api/clients/{cid}").status_code == 403
        assert c.delete(f"/api/clients/{cid}").status_code == 204
        assert [x["name"] for x in c.get("/api/clients").json()] == ["Queda"]
        assert [x["name"] for x in c.get("/api/cameras").json()] == ["B"]
        assert "dueno" not in [u["username"] for u in c.get("/api/users").json()]
        assert c.get("/api/metrics", params={"client_id": keep}).json()["total"]["people"] == 1
        # borrar cámara sin alertas (con métricas) y agente
        assert c.delete(f"/api/cameras/{cam2['id']}").status_code == 204
        assert c.delete(f"/api/agents/{ag['id']}").status_code == 204


def test_zonas_nuevas_se_pueden_guardar(app):
    with TestClient(app) as c:
        c.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]})
        cid = c.post("/api/clients", json={"name": "Estación"}).json()["id"]
        cam = c.post("/api/cameras", json={"client_id": cid, "name": "Playa", "host": "h", "password": "x"}).json()
        sq = [[0, 0], [0.5, 0], [0.5, 0.5]]
        r = c.put(f"/api/cameras/{cam['id']}/analysis", json={"zones": [
            {"name": "entrada", "type": "entrance", "polygon": sq}, {"name": "surtidor", "type": "pump", "polygon": sq},
            {"name": "tienda", "type": "shop", "polygon": sq}], "rules": {"loiter_seconds": 15}})
        assert r.status_code == 200, r.text
        assert [z["type"] for z in r.json()["zones"]] == ["entrance", "pump", "shop"]


def test_cliente_con_usuario_y_clave_para_la_web(app):
    with TestClient(app) as c:
        c.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]})
        # falta la clave: no se crea ni el cliente ni el usuario
        r = c.post("/api/clients", json={"name": "Súper Minas", "web_username": "superminas"})
        assert r.status_code == 422 and c.get("/api/clients").json() == []
        r = c.post("/api/clients", json={"name": "Súper Minas", "web_username": "superminas", "web_password": "corta"})
        assert r.status_code == 422
        r = c.post("/api/clients", json={"name": "Súper Minas", "web_username": "superminas",
                                          "web_password": "clave-minas-1", "plans": ["hurto", "comercio"]})
        assert r.status_code == 201 and r.json()["web_username"] == "superminas"
        cid = r.json()["id"]
        assert c.get("/api/clients").json()[0]["web_username"] == "superminas"
        u = TestClient(app)
        me = u.post("/api/login", json={"username": "superminas", "password": "clave-minas-1"}).json()["user"]
        assert me["role"] == "cliente" and me["client_id"] == cid
        assert [x["name"] for x in u.get("/api/clients").json()] == ["Súper Minas"]
        # un empleado también puede crear clientes con acceso
        c.post("/api/users", json={"username": "empleado", "password": "clave-empl-1", "role": "entrenador"})
        e = TestClient(app)
        e.post("/api/login", json={"username": "empleado", "password": "clave-empl-1"})
        assert e.post("/api/clients", json={"name": "Otro", "web_username": "superminas",
                                            "web_password": "clave-otro-1"}).status_code == 409  # usuario repetido
        assert e.post("/api/clients", json={"name": "Otro", "web_username": "otro",
                                            "web_password": "clave-otro-1"}).status_code == 201
        # editar sin clave: cambia solo los datos; con clave: la cambia y cierra la sesión vieja
        base = {"name": "Súper Minas", "plans": ["hurto"]}
        r = c.put(f"/api/clients/{cid}", json={**base, "web_username": "superminas"})
        assert r.status_code == 200 and r.json()["plans"] == ["hurto"]
        assert u.get("/api/clients").status_code == 200
        c.put(f"/api/clients/{cid}", json={**base, "web_username": "minas", "web_password": "clave-nueva-2"})
        assert u.get("/api/clients").status_code == 401
        assert TestClient(app).post("/api/login", json={"username": "minas", "password": "clave-nueva-2"}).status_code == 200
        # un cliente sin acceso puede recibirlo después al editarlo
        sin = c.post("/api/clients", json={"name": "Sin acceso"}).json()
        assert sin["web_username"] is None
        assert c.put(f"/api/clients/{sin['id']}", json={"name": "Sin acceso", "web_username": "sinacceso"}).status_code == 422
        assert c.put(f"/api/clients/{sin['id']}", json={"name": "Sin acceso", "web_username": "sinacceso",
                                                        "web_password": "clave-sin-1"}).json()["web_username"] == "sinacceso"
        # nombre de cliente repetido al editar
        assert c.put(f"/api/clients/{sin['id']}", json={"name": "Otro"}).status_code == 409
