"""Qué mide cada cámara: los planes del cliente (hurto, comercio, estación) se eligen por cámara y las métricas se juntan
por cliente con lo que mide cada una (una cámara de adentro cuenta gente; una de afuera, vehículos)."""

import csv
import io
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from core.metrics import PEOPLE_KEYS, STATION_KEYS, empty_bucket, merge_bucket, restrict_bucket
from server.main import METRICS_FPS, MIN_FPS, RULE_TYPES, Settings, create_app

ADMIN = ("admin", "clave-admin-123")
TODOS = ["hurto", "comercio", "estacion"]


@pytest.fixture
def app(tmp_path, db_url):
    return create_app(Settings(data_dir=tmp_path / "data", database_url=db_url, admin_password=ADMIN[1],
                               serve_public=False))


def login(app, user, pwd):
    c = TestClient(app)
    assert c.post("/api/login", json={"username": user, "password": pwd}).status_code == 200
    return c


def cam_body(cid, name, **kw):
    return {"client_id": cid, "name": name, "brand": "ip", "host": "dvr", "password": "x", **kw}


def test_restrict_bucket_saca_lo_que_la_camara_no_mide():
    data = {"people": 5, "people_dwell_s": 50.0, "people_max": 2, "entries": 4, "exits": 3, "door": 1, "staff": 1,
            "staff_entries": 1, "zones": {"g": {"type": "shelf", "visits": 2, "dwell_s": 9.0}},
            "vehicles": 3, "vehicles_dwell_s": 30.0, "vehicles_max": 2, "pump_vehicles": 2, "pump_wait_s": 20.0,
            "pump_wait_n": 2, "pump_unattended": 1, "shop_visits": 1, "frames": 900}
    solo_gente = restrict_bucket(data, TODOS, ["hurto", "comercio"])
    # (un cubo de antes no trae «waits»: la espera de los clientes, que también es de «comercio»)
    assert all(solo_gente.get(k) == data.get(k) for k in PEOPLE_KEYS) and solo_gente["frames"] == 900
    assert all(solo_gente[k] == empty_bucket()[k] for k in STATION_KEYS)
    solo_estacion = restrict_bucket(data, TODOS, ["estacion"])
    assert solo_estacion["zones"] == {} and solo_estacion["people"] == 0 and solo_estacion["door"] == 0
    assert all(solo_estacion[k] == data[k] for k in STATION_KEYS) and solo_estacion["frames"] == 900
    assert data["people"] == 5 and data["zones"]  # es una copia: el original no cambia
    # si el cliente no tiene ese plan no se recorta nada (se ve lo de siempre)
    assert restrict_bucket(data, ["hurto"], ["hurto"]) == data
    assert restrict_bucket(data, ["hurto", "estacion"], ["hurto"])["people"] == 5
    assert restrict_bucket(data, TODOS, TODOS) == data


def test_crear_y_editar_camara_con_lo_que_mide(app):
    with TestClient(app) as a:
        a.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]})
        cid = a.post("/api/clients", json={"name": "Estación Sur", "plans": TODOS}).json()["id"]
        kiosco = a.post("/api/clients", json={"name": "Kiosco", "plans": ["hurto", "comercio"]}).json()["id"]
        # sin elegir: como el cliente (mide todo lo que contrató)
        r = a.post("/api/cameras", json=cam_body(cid, "General"))
        assert r.status_code == 201 and (r.json()["plans"], r.json()["measures"]) == (None, TODOS)
        # elegida: solo eso
        adentro = a.post("/api/cameras", json=cam_body(cid, "Adentro", plans=["comercio"])).json()
        assert (adentro["plans"], adentro["measures"]) == (["comercio"], ["comercio"])
        # todo lo del cliente (en cualquier orden) se guarda como «como el cliente»
        r = a.post("/api/cameras", json=cam_body(cid, "Todo", plans=["estacion", "hurto", "comercio"])).json()
        assert (r["plans"], r["measures"]) == (None, TODOS)
        # nada para medir, un plan que el cliente no tiene o repetidos: 422 y no se crea
        r = a.post("/api/cameras", json=cam_body(cid, "Vacía", plans=[]))
        assert r.status_code == 422 and "al menos una cosa" in r.json()["detail"]
        r = a.post("/api/cameras", json=cam_body(kiosco, "Calle", plans=["estacion"]))
        assert r.status_code == 422
        assert r.json()["detail"] == ("La cámara no puede medir «Estación de servicio»: el cliente no lo tiene "
                                      "contratado")
        assert a.post("/api/cameras", json=cam_body(cid, "Rep", plans=["comercio", "comercio"])).status_code == 422
        assert a.post("/api/cameras", json=cam_body(cid, "Rara", plans=["otra"])).status_code == 422
        assert sorted(x["name"] for x in a.get("/api/cameras").json()) == ["Adentro", "General", "Todo"]
        # editar sin el campo no lo toca
        r = a.put(f"/api/cameras/{adentro['id']}", json=cam_body(cid, "Adentro 1")).json()
        assert (r["name"], r["plans"]) == ("Adentro 1", ["comercio"])
        # la cámara tal como vino (con «measures», que no se guarda) tampoco
        cam = a.get(f"/api/cameras/{adentro['id']}").json()
        assert (cam["plans"], cam["measures"]) == (["comercio"], ["comercio"])
        assert a.put(f"/api/cameras/{adentro['id']}", json={**cam, "password": None}).json()["plans"] == ["comercio"]
        # con el campo: lista (queda en el orden de siempre), todo = None, null = como el cliente
        r = a.put(f"/api/cameras/{adentro['id']}", json=cam_body(cid, "Adentro 1", plans=["estacion", "comercio"]))
        assert (r.json()["plans"], r.json()["measures"]) == (["comercio", "estacion"], ["comercio", "estacion"])
        r = a.put(f"/api/cameras/{adentro['id']}", json=cam_body(cid, "Adentro 1", plans=["comercio"])).json()
        assert r["plans"] == ["comercio"]
        r = a.put(f"/api/cameras/{adentro['id']}", json=cam_body(cid, "Adentro 1", plans=None)).json()
        assert (r["plans"], r["measures"]) == (None, TODOS)
        a.put(f"/api/cameras/{adentro['id']}", json=cam_body(cid, "Adentro 1", plans=["estacion"]))
        r = a.put(f"/api/cameras/{adentro['id']}", json=cam_body(cid, "Adentro 1", plans=[]))
        assert r.status_code == 422
        assert a.get(f"/api/cameras/{adentro['id']}").json()["plans"] == ["estacion"]  # no cambió
        # pasar la cámara a un cliente que no tiene lo elegido: como ese cliente
        r = a.put(f"/api/cameras/{adentro['id']}", json=cam_body(kiosco, "Adentro 1")).json()
        assert (r["client_id"], r["plans"], r["measures"]) == (kiosco, None, ["hurto", "comercio"])
        # el dueño del comercio ve qué mide cada una de sus cámaras, solo las suyas
        a.post("/api/users", json={"username": "dueno", "password": "clave-dueno-1", "role": "cliente",
                                   "client_id": cid})
        u = login(app, "dueno", "clave-dueno-1")
        mine = u.get("/api/cameras").json()
        assert sorted(x["name"] for x in mine) == ["General", "Todo"]
        assert all(x["measures"] == TODOS for x in mine)
        assert u.get(f"/api/cameras/{adentro['id']}").status_code == 403


def test_la_config_de_cada_camara_sigue_lo_que_mide(app):
    with TestClient(app) as a:
        a.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]})
        cid = a.post("/api/clients", json={"name": "Estación Norte", "plans": TODOS}).json()["id"]
        ag = a.post("/api/agents", json={"name": "central", "client_id": None}).json()
        h = {"X-Agent-Token": ag["token"]}
        mk = lambda n, **kw: a.post("/api/cameras", json=cam_body(cid, n, agent_id=ag["id"], **kw)).json()  # noqa: E731
        adentro, playa, todo = mk("Adentro", plans=["comercio"]), mk("Playa", plans=["estacion"]), mk("Todo")
        cfg = {x["id"]: x for x in a.get("/api/agent/config", headers=h).json()["cameras"]}
        # solo comercio: cuenta gente, sin reglas ni gestos ni vehículos, a pocos cuadros por segundo
        x = cfg[adentro["id"]]
        assert x["metrics_only"] is True and x["analysis"]["metrics_only"] is True
        assert x["analysis"]["rules"]["enabled_rules"] == [] and "gestures" not in x["analysis"]
        assert x["analysis"]["vehicles"] is False and x["process_fps"] == METRICS_FPS
        assert (x["analysis"]["plans"], x["analysis"]["client_plans"]) == (["comercio"], TODOS)
        # solo estación: vehículos, sin hurto
        x = cfg[playa["id"]]
        assert x["analysis"]["vehicles"] is True and x["metrics_only"] is True and "gestures" not in x["analysis"]
        assert (x["analysis"]["plans"], x["analysis"]["client_plans"]) == (["estacion"], TODOS)
        # como el cliente: todo (reglas de hurto, gestos y vehículos)
        x = cfg[todo["id"]]
        assert x["metrics_only"] is False and x["analysis"]["rules"]["enabled_rules"] == list(RULE_TYPES)
        assert x["analysis"]["gestures"]["enabled"] and x["analysis"]["vehicles"] is True
        assert x["process_fps"] >= MIN_FPS and x["analysis"]["plans"] == TODOS
        # «Configurar…» y «Ver con la IA» de la app reciben lo mismo
        a.post("/api/users", json={"username": "ana", "password": "clave-ana-123", "role": "entrenador"})
        e = login(app, "ana", "clave-ana-123")
        tok = {"X-Agent-Token": e.post("/api/agent/session").json()["token"]}
        tune = TestClient(app).post("/api/agent/tune", json={"camera_id": adentro["id"]}, headers=tok)
        assert tune.status_code == 200 and tune.json()["camera"]["analysis"]["plans"] == ["comercio"]
        view = TestClient(app).post("/api/agent/view", json={"camera_ids": [playa["id"]]}, headers=tok).json()
        assert view["cameras"][0]["analysis"]["plans"] == ["estacion"]


def test_elegir_lo_que_miden_varias_camaras_a_la_vez(app):
    with TestClient(app) as a:
        a.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]})
        cid = a.post("/api/clients", json={"name": "Estación", "plans": TODOS}).json()["id"]
        kiosco = a.post("/api/clients", json={"name": "Kiosco", "plans": ["hurto"]}).json()["id"]
        ag = a.post("/api/agents", json={"name": "central", "client_id": None}).json()
        h = {"X-Agent-Token": ag["token"]}
        c1 = a.post("/api/cameras", json=cam_body(cid, "Isla 1", agent_id=ag["id"])).json()
        c2 = a.post("/api/cameras", json=cam_body(cid, "Isla 2", agent_id=ag["id"], plans=["hurto"])).json()
        k1 = a.post("/api/cameras", json=cam_body(kiosco, "Mostrador")).json()
        a.post("/api/users", json={"username": "ana", "password": "clave-ana-123", "role": "entrenador"})
        op = login(app, "ana", "clave-ana-123")  # un operador también puede
        rev = a.get("/api/agent/config", headers=h).json()["rev"]
        r = op.post("/api/cameras/plans", json={"camera_ids": [c1["id"], c2["id"]], "plans": ["estacion"]})
        assert r.status_code == 200 and r.json()["updated"] == 2
        assert [(x["id"], x["plans"], x["measures"]) for x in r.json()["cameras"]] == [
            (c1["id"], ["estacion"], ["estacion"]), (c2["id"], ["estacion"], ["estacion"])]
        cfg = a.get("/api/agent/config", headers=h).json()
        assert int(cfg["rev"]) > int(rev)  # las máquinas piden la configuración ya
        assert all(x["analysis"]["plans"] == ["estacion"] for x in cfg["cameras"])
        # si una no puede (su cliente no tiene estación), no se guarda ninguna y se dice cuál
        r = op.post("/api/cameras/plans", json={"camera_ids": [c1["id"], k1["id"]], "plans": ["comercio", "estacion"]})
        assert r.status_code == 422 and r.json()["detail"] == ("La cámara «Mostrador» no puede medir «Métricas de "
                                                              "comercio»: el cliente no lo tiene contratado")
        assert a.get(f"/api/cameras/{c1['id']}").json()["plans"] == ["estacion"]
        r = op.post("/api/cameras/plans", json={"camera_ids": [c1["id"]], "plans": []})
        assert r.status_code == 422 and "«Isla 1»" in r.json()["detail"]
        # null (o todo lo del cliente) = como el cliente
        r = op.post("/api/cameras/plans", json={"camera_ids": [c1["id"], k1["id"]], "plans": None}).json()
        assert [(x["plans"], x["measures"]) for x in r["cameras"]] == [(None, TODOS), (None, ["hurto"])]
        r = op.post("/api/cameras/plans", json={"camera_ids": [c2["id"]], "plans": TODOS}).json()
        assert r["cameras"][0]["plans"] is None
        assert op.post("/api/cameras/plans", json={"camera_ids": [9999], "plans": None}).status_code == 404
        assert op.post("/api/cameras/plans", json={"camera_ids": [], "plans": None}).status_code == 422
        # el dueño del comercio no puede
        a.post("/api/users", json={"username": "dueno", "password": "clave-dueno-1", "role": "cliente",
                                   "client_id": cid})
        u = login(app, "dueno", "clave-dueno-1")
        assert u.post("/api/cameras/plans", json={"camera_ids": [c1["id"]], "plans": ["hurto"]}).status_code == 403


def test_lo_elegido_en_la_camara_no_se_pierde_cuando_el_cliente_cambia_sus_planes(app):
    """(Revisión 2 nov) Antes se recortaba al guardar el cliente: una cámara de afuera puesta sin hurto quedaba «como el
    cliente» y, al volver a contratar hurto, empezaba a dar alertas sin que nadie lo pidiera."""
    with TestClient(app) as a:
        a.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]})
        cid = a.post("/api/clients", json={"name": "Estación", "plans": TODOS}).json()["id"]
        x = a.post("/api/cameras", json=cam_body(cid, "X", plans=["comercio", "estacion"])).json()
        y = a.post("/api/cameras", json=cam_body(cid, "Y", plans=["estacion"])).json()
        w = a.post("/api/cameras", json=cam_body(cid, "W")).json()
        assert a.put(f"/api/clients/{cid}", json={"name": "Estación", "plans": ["hurto", "comercio"]}).status_code == 200
        cams = {c["name"]: c for c in a.get("/api/cameras", params={"client_id": cid}).json()}
        assert (cams["X"]["plans"], cams["X"]["measures"]) == (["comercio", "estacion"], ["comercio"])
        assert (cams["Y"]["plans"], cams["Y"]["measures"]) == (["estacion"], [])  # no mide nada (no pasa a dar alertas)
        assert cams["W"]["measures"] == ["hurto", "comercio"]
        # vuelve a contratar estación: cada una vuelve a lo que se le eligió (la X sigue sin hurto)
        a.put(f"/api/clients/{cid}", json={"name": "Estación", "plans": TODOS})
        cams = {c["id"]: c for c in a.get("/api/cameras", params={"client_id": cid}).json()}
        assert cams[x["id"]]["measures"] == ["comercio", "estacion"]
        assert cams[y["id"]]["measures"] == ["estacion"] and cams[w["id"]]["measures"] == TODOS


def test_un_cliente_tiene_que_tener_al_menos_un_plan(app):
    with TestClient(app) as a:
        a.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]})
        r = a.post("/api/clients", json={"name": "Vacío", "plans": []})
        assert r.status_code == 422 and "al menos un plan" in r.json()["detail"]
        cid = a.post("/api/clients", json={"name": "Kiosco", "plans": ["comercio"]}).json()["id"]
        assert a.put(f"/api/clients/{cid}", json={"name": "Kiosco", "plans": []}).status_code == 422


def test_una_camara_que_no_mide_hurtos_no_da_alertas_por_alarmas(app):
    with TestClient(app) as a:
        a.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]})
        cid = a.post("/api/clients", json={"name": "Estación", "plans": TODOS, "timezone": "America/Montevideo"}).json()["id"]
        ag = a.post("/api/agents", json={"name": "central", "client_id": None}).json()
        h = {"X-Agent-Token": ag["token"]}
        alarm = {"mode": "alarmas_equipo", "alarm_cfg": {"event_types": ["VMD"], "pre_seconds": 5, "post_seconds": 5}}
        afuera = a.post("/api/cameras", json=cam_body(cid, "Afuera", agent_id=ag["id"], plans=["estacion"], **alarm)).json()
        adentro = a.post("/api/cameras", json=cam_body(cid, "Adentro", agent_id=ag["id"], channel=2, **alarm)).json()
        when = datetime.now(ZoneInfo("America/Montevideo")).strftime("%Y-%m-%dT%H:%M:%S")
        r1 = a.post("/api/agent/alarms", headers=h, json={"camera_id": afuera["id"], "event_type": "VMD",
                                                         "occurred_at": when, "raw": {}})
        r2 = a.post("/api/agent/alarms", headers=h, json={"camera_id": adentro["id"], "event_type": "VMD",
                                                         "occurred_at": when, "raw": {}})
        assert r1.json()["result"] == "no_mide_hurto" and r2.json()["result"] == "nueva"


def test_editar_una_camara_que_solo_tiene_link(app):
    with TestClient(app) as a:
        a.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]})
        cid = a.post("/api/clients", json={"name": "Kiosco", "plans": TODOS}).json()["id"]
        cam = a.post("/api/cameras", json={"client_id": cid, "name": "Link", "brand": "hikvision", "host": "",
                                           "source_url": "rtsp://u:p@dvr.ddns.net:554/Streaming/Channels/401"}).json()
        body = {"client_id": cid, "name": "Link nuevo", "brand": "hikvision", "host": "", "keep_source_url": True,
                "plans": ["comercio"]}
        r = a.put(f"/api/cameras/{cam['id']}", json=body)
        assert r.status_code == 200 and r.json()["name"] == "Link nuevo" and r.json()["measures"] == ["comercio"]
        body.pop("keep_source_url")
        assert a.put(f"/api/cameras/{cam['id']}", json=body).status_code == 422  # sin link ni dirección: no


def export_rows(c, cid, day):
    r = c.get("/api/metrics/export", params={"client_id": cid, "desde": day, "hasta": day})
    assert r.status_code == 200 and "metricas_Estacion_Este_" in r.headers["content-disposition"]
    rows = list(csv.reader(io.StringIO(r.text.lstrip("﻿")), delimiter=";"))
    head = rows[0]
    return {row[head.index("Cámara")]: dict(zip(head, row)) for row in rows[1:] if row}


def test_las_metricas_del_cliente_suman_lo_que_mide_cada_camara(app):
    with TestClient(app) as a:
        a.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]})
        cid = a.post("/api/clients", json={"name": "Estación Este", "plans": ["comercio", "estacion"],
                                            "timezone": "America/Montevideo"}).json()["id"]
        ag = a.post("/api/agents", json={"name": "central", "client_id": None}).json()
        h = {"X-Agent-Token": ag["token"]}
        adentro = a.post("/api/cameras", json=cam_body(cid, "Tienda adentro", agent_id=ag["id"],
                                                       plans=["comercio"])).json()
        playa = a.post("/api/cameras", json=cam_body(cid, "Playa", agent_id=ag["id"], plans=["estacion"])).json()
        hour = datetime.now(ZoneInfo("America/Montevideo")).replace(minute=0, second=0, microsecond=0, tzinfo=None)
        # las dos cámaras ven de todo (la de afuera también ve gente y la de adentro, autos por la vidriera)
        d_adentro = {"people": 10, "people_dwell_s": 600, "people_max": 3, "entries": 8, "exits": 7, "door": 1,
                     "staff": 1, "zones": {"Góndola": {"type": "shelf", "visits": 6, "dwell_s": 120}},
                     "vehicles": 3, "pump_vehicles": 1, "frames": 500}
        d_playa = {"people": 4, "people_dwell_s": 80, "people_max": 2, "entries": 2,
                   "zones": {"Surtidor 1": {"type": "pump", "visits": 4, "dwell_s": 60}},
                   "vehicles": 6, "vehicles_dwell_s": 300, "pump_vehicles": 5, "pump_wait_s": 100, "pump_wait_n": 4,
                   "pump_unattended": 1, "shop_visits": 2, "frames": 400}
        prev = {"camera_id": playa["id"], "bucket_start": (hour - timedelta(hours=1)).isoformat(),
                "data": {"people": 1, "vehicles": 2}}
        r = a.post("/api/agent/metrics", headers=h, json={"buckets": [
            {"camera_id": adentro["id"], "bucket_start": hour.isoformat(), "data": d_adentro},
            {"camera_id": playa["id"], "bucket_start": hour.isoformat(), "data": d_playa}, prev]})
        assert r.json()["saved"] == 3
        m = a.get("/api/metrics", params={"client_id": cid, "days": 1}).json()
        t = m["total"]
        assert (t["people"], t["people_dwell_s"], t["entries"], t["exits"], t["door"]) == (10, 600, 8, 7, 1)
        assert list(t["zones"]) == ["Góndola"]  # el surtidor visto por gente no es de «comercio» en esta cámara
        assert (t["vehicles"], t["pump_vehicles"], t["pump_wait_n"], t["shop_visits"]) == (8, 5, 4, 2)
        assert t["frames"] == 900  # cuadros procesados: de todas
        by_cam = {x["name"]: x for x in m["by_camera"]}
        assert by_cam["Tienda adentro"]["measures"] == ["comercio"] and by_cam["Tienda adentro"]["vehicles"] == 0
        assert by_cam["Playa"]["measures"] == ["estacion"] and by_cam["Playa"]["people"] == 0
        assert sum(x["people"] for x in m["by_hour"]) == 10 and sum(x["vehicles"] for x in m["by_day"]) == 8
        # el Excel también
        rows = export_rows(a, cid, hour.date().isoformat())
        assert (rows["Tienda adentro"]["Ingresos"], rows["Tienda adentro"]["Vehículos"]) == ("8", "0")
        assert (rows["Playa"]["Ingresos"], rows["Playa"]["Vehículos"], rows["Playa"]["Visitas a la tienda"]) == (
            "0", "8", "2")
        assert rows["Playa"]["Tiempo promedio en escena (min)"] == "" and rows["Playa"]["Zonas (visitas · minutos)"] == ""
        # y el PDF sale (usa lo mismo)
        pytest.importorskip("fpdf")
        pdf = a.get("/api/metrics/report.pdf", params={"client_id": cid, "days": 1})
        assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")
        # todas como el cliente: igual que antes de poder elegir (se suma todo)
        a.post("/api/cameras/plans", json={"camera_ids": [adentro["id"], playa["id"]], "plans": None})
        m = a.get("/api/metrics", params={"client_id": cid, "days": 1}).json()
        raw = merge_bucket(merge_bucket(merge_bucket(empty_bucket(), d_adentro), d_playa), prev["data"])
        assert m["total"] == raw and m["total"]["people"] == 15 and m["total"]["vehicles"] == 11
        assert all(x["measures"] == ["comercio", "estacion"] for x in m["by_camera"])
        rows = export_rows(a, cid, hour.date().isoformat())
        assert (rows["Playa"]["Ingresos"], rows["Playa"]["Vehículos"]) == ("2", "8")
        # el rol cliente ve lo mismo agrupado
        a.post("/api/cameras/plans", json={"camera_ids": [playa["id"]], "plans": ["estacion"]})
        a.post("/api/users", json={"username": "dueno", "password": "clave-dueno-1", "role": "cliente",
                                   "client_id": cid})
        u = login(app, "dueno", "clave-dueno-1")
        mu = u.get("/api/metrics", params={"days": 1}).json()
        assert (mu["total"]["people"], mu["total"]["vehicles"]) == (10, 11)  # la de adentro mide todo otra vez
