"""Panel de datos: dónde se analiza cada cliente o cámara, alertas por cliente y cámara, y exportar métricas."""

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from server.main import Settings, create_app

ADMIN = ("admin", "clave-admin-123")


@pytest.fixture
def app(tmp_path, db_url):
    return create_app(Settings(data_dir=tmp_path / "data", database_url=db_url, admin_password=ADMIN[1],
                               serve_public=False))


def admin(app) -> TestClient:
    a = TestClient(app)
    a.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]})
    return a


def cam(a, cl, name, **kw):
    r = a.post("/api/cameras", json={"client_id": cl["id"], "name": name, "brand": "ip", "host": "dvr",
                                     "password": "x", **kw})
    assert r.status_code == 201, r.text
    return r.json()


def test_probar_un_solo_cliente_en_una_maquina(app):
    a = admin(app)
    minas = a.post("/api/clients", json={"name": "Minas"}).json()
    otro = a.post("/api/clients", json={"name": "Otro"}).json()
    pc = a.post("/api/agents", json={"name": "PC vieja"}).json()
    m1 = cam(a, minas, "Minas 1", agent_id=pc["id"])
    m2 = cam(a, minas, "Minas 2", agent_id=pc["id"])
    o1 = cam(a, otro, "Otro 1", agent_id=pc["id"])
    key = a.post("/api/analysis/pool-key").json()["key"]
    tok = TestClient(app).post("/api/agent/enroll", json={"key": key, "name": "oracle", "instance_id": "ocid-1",
                                                          "capacity": 4}).json()["token"]
    maq = TestClient(app, headers={"X-Agent-Token": tok})
    maq.post("/api/agent/status", json={"cameras": []})
    # el ajuste general sigue en «PC de cada usuario»: la máquina no recibe nada
    assert maq.get("/api/agent/config").json()["cameras"] == []
    # solo Minas pasa a las máquinas
    r = a.put(f"/api/clients/{minas['id']}", json={"name": "Minas", "analysis_location": "servidores"})
    assert r.status_code == 200 and r.json()["analysis_location"] == "servidores"
    assert sorted(c["name"] for c in maq.get("/api/agent/config").json()["cameras"]) == ["Minas 1", "Minas 2"]
    by = {c["name"]: c for c in a.get("/api/cameras").json()}
    assert by["Otro 1"]["analyzer_name"] == "PC vieja" and by["Minas 1"]["analyzer_name"] == "oracle"
    # una cámara de Minas vuelve a la PC aunque el cliente esté en máquinas
    a.put(f"/api/cameras/{m2['id']}", json={"client_id": minas["id"], "name": "Minas 2", "brand": "ip", "host": "dvr",
                                            "analysis_location": "pcs"})
    assert [c["name"] for c in maq.get("/api/agent/config").json()["cameras"]] == ["Minas 1"]
    assert {c["name"]: c for c in a.get("/api/cameras").json()}["Minas 2"]["analyzer_name"] == "PC vieja"
    # el plan solo cuenta las cámaras de las máquinas
    an = a.get("/api/analysis").json()
    assert an["cameras"] == 1 and an["clients_on_servers"] == [minas["id"]]
    # Minas vuelve entero a la PC que tenía
    a.put(f"/api/clients/{minas['id']}", json={"name": "Minas", "analysis_location": "general"})
    assert maq.get("/api/agent/config").json()["cameras"] == []
    assert {c["analyzer_name"] for c in a.get("/api/cameras").json()} == {"PC vieja"}
    assert m1 and o1


def test_alertas_por_cliente_y_camara_y_exportar_metricas(app):
    a = admin(app)
    cl = a.post("/api/clients", json={"name": "Kiosco Centro"}).json()
    c1, c2 = cam(a, cl, "Caja"), cam(a, cl, "Puerta")
    from server.db import CameraMetric, Event

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    with app.state.Session() as s:
        def ev(c, when, label=None, alert=True, source="vivo"):
            s.add(Event(uuid=f"u{len(s.new)}-{c}-{when.timestamp()}", client_id=cl["id"], camera_id=c, type="salida_sin_pago",
                        severity="media", description="x", occurred_at=when, label=label, alert=alert, source=source))
        ev(c1["id"], now)
        ev(c1["id"], now - timedelta(days=3), label="normal")
        ev(c1["id"], now - timedelta(days=10))  # fuera de la semana, pero sin revisar
        ev(c2["id"], now - timedelta(days=2))
        ev(c2["id"], now, alert=False)  # filtrada por el modelo: no cuenta
        ev(c2["id"], now, source="importado")  # video subido para entrenar: no cuenta
        day = datetime(2026, 10, 20, 10)
        s.add(CameraMetric(client_id=cl["id"], camera_id=c1["id"], bucket_start=day,
                           data={"people": 10, "people_dwell_s": 1200.0, "people_max": 3, "entries": 7,
                                 "zones": {"Góndola": {"type": "shelf", "visits": 4, "dwell_s": 300.0}}}))
        s.add(CameraMetric(client_id=cl["id"], camera_id=c1["id"], bucket_start=day + timedelta(hours=1),
                           data={"people": 5, "people_dwell_s": 300.0, "people_max": 2}))
        s.commit()
    st = a.get("/api/stats/alerts").json()
    assert st["cameras"][str(c1["id"])] == {"client_id": cl["id"], "today": 1, "week": 2, "unreviewed": 2}
    assert st["cameras"][str(c2["id"])]["week"] == 1
    assert st["clients"][str(cl["id"])] == {"today": 1, "week": 3, "unreviewed": 3}

    r = a.get("/api/metrics/export", params={"client_id": cl["id"], "desde": "2026-10-19", "hasta": "2026-10-21"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    assert "metricas_Kiosco_Centro_2026-10-19_a_2026-10-21.csv" in r.headers["content-disposition"]
    lines = r.content.decode("utf-8-sig").strip().split("\r\n")
    assert lines[0].startswith("Fecha;Cliente;Cámara;Personas;")
    # (egresos vacío: sin puerta con sentido; después empleados y vehículos)
    assert lines[1].split(";")[:10] == ["20/10/2026", "Kiosco Centro", "Caja", "15", "1,7", "3", "7", "", "0", "0"]
    assert "Góndola: 4 · 5,0" in lines[1]
    por_hora = a.get("/api/metrics/export", params={"client_id": cl["id"], "desde": "2026-10-20",
                                                    "hasta": "2026-10-20", "agrupar": "hora"}).content.decode("utf-8-sig")
    assert por_hora.count("\r\n") == 3 and "20/10/2026;10:00;Kiosco Centro;Caja;10;2,0" in por_hora
    assert a.get("/api/metrics/export", params={"desde": "20-10-2026"}).status_code == 422
    # un cliente solo exporta lo suyo
    a.post("/api/users", json={"username": "kiosco", "password": "clave-kiosco-1", "role": "cliente",
                               "client_id": cl["id"]})
    cli = TestClient(app)
    cli.post("/api/login", json={"username": "kiosco", "password": "clave-kiosco-1"})
    assert cli.get("/api/metrics/export", params={"desde": "2026-10-20", "hasta": "2026-10-20"}).status_code == 200
    assert cli.get("/api/stats/alerts").status_code == 403


def test_borrar_varias_camaras_y_sus_alertas(app):
    a = admin(app)
    cl = a.post("/api/clients", json={"name": "Minas"}).json()
    c1, c2, c3 = cam(a, cl, "C1"), cam(a, cl, "C2"), cam(a, cl, "C3")
    from server.db import ClientReport, Event

    with app.state.Session() as s:
        e = Event(uuid="ev-borrar", client_id=cl["id"], camera_id=c2["id"], type="salida_sin_pago", severity="alta",
                  description="x", snapshot_path="c1/foto.jpg")
        s.add(e)
        s.flush()
        s.add(ClientReport(client_id=cl["id"], event_id=e.id, public_token="t" * 40, message="posible hurto",
                            expires_at=datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=14)))
        s.commit()
    # una sin alertas se borra sola
    assert a.delete(f"/api/cameras/{c1['id']}").status_code == 204
    # con alertas pide confirmación y no borra nada
    r = a.post("/api/cameras/delete", json={"camera_ids": [c2["id"], c3["id"]]})
    assert r.status_code == 409 and "1 alerta" in r.json()["detail"]
    assert len(a.get("/api/cameras").json()) == 2
    r = a.post("/api/cameras/delete", json={"camera_ids": [c2["id"], c3["id"]], "con_alertas": True})
    assert r.status_code == 200 and r.json()["deleted"] == 2
    assert a.get("/api/cameras").json() == []
    with app.state.Session() as s:
        assert s.query(Event).count() == 0 and s.query(ClientReport).count() == 0
    # un empleado no puede borrar
    a.post("/api/users", json={"username": "ana", "password": "clave-ana-123", "role": "entrenador"})
    e = TestClient(app)
    e.post("/api/login", json={"username": "ana", "password": "clave-ana-123"})
    c4 = cam(a, cl, "C4")
    assert e.post("/api/cameras/delete", json={"camera_ids": [c4["id"]]}).status_code == 403
