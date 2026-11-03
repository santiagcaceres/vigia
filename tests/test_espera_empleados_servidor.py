"""Tiempo de espera por zona y plan «Seguimiento de empleados» en el servidor (pedido del dueño, 3 nov): el plan nuevo en
clientes y cámaras, la zona marcada «medir espera» hasta las máquinas, y las métricas en /api/metrics, el Excel y el
PDF, cada cámara con lo que mide."""

import csv
import io
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from core.metrics import empty_bucket, merge_bucket, wait_stats
from core.rules import parse_config
from server.main import (
    METRICS_FPS,
    Settings,
    create_app,
    staff_csv_cells,
    wait_csv_cells,
)

ADMIN = ("admin", "clave-admin-123")
TZ = "America/Montevideo"
SQUARE = [[0.1, 0.5], [0.4, 0.5], [0.4, 0.9], [0.1, 0.9]]
OTHER = [[0.6, 0.5], [0.9, 0.5], [0.9, 0.9], [0.6, 0.9]]
TOP = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.3], [0.1, 0.3]]


@pytest.fixture
def app(tmp_path, db_url):
    return create_app(Settings(data_dir=tmp_path / "data", database_url=db_url, admin_password=ADMIN[1],
                               serve_public=False))


def admin(app) -> TestClient:
    c = TestClient(app)
    assert c.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]}).status_code == 200
    return c


def cam_body(cid, name, **kw):
    return {"client_id": cid, "name": name, "brand": "ip", "host": "dvr", "password": "x", **kw}


def num(v, dec=1):
    return f"{v:.{dec}f}".replace(".", ",")


def text_of(pdf: bytes) -> str:
    pypdf = pytest.importorskip("pypdf")
    return " ".join("".join(p.extract_text() for p in pypdf.PdfReader(io.BytesIO(pdf)).pages).split())


# ------------------------------------------------------------------ el plan nuevo
def test_plan_empleados_en_el_cliente_y_en_cada_camara(app):
    with admin(app) as a:
        r = a.post("/api/clients", json={"name": "Farmacia", "plans": ["empleados", "hurto", "comercio"]})
        assert r.status_code == 201 and sorted(r.json()["plans"]) == ["comercio", "empleados", "hurto"]
        cid = r.json()["id"]
        kiosco = a.post("/api/clients", json={"name": "Kiosco", "plans": ["hurto"]}).json()["id"]
        solo = a.post("/api/clients", json={"name": "Oficina", "plans": ["empleados"]})
        assert solo.status_code == 201 and solo.json()["plans"] == ["empleados"]
        assert a.post("/api/clients", json={"name": "Raro", "plans": ["inventado"]}).status_code == 422
        # la cámara mide lo que contrató el cliente, en el orden de siempre (empleados al final)
        todo = a.post("/api/cameras", json=cam_body(cid, "Mostrador")).json()
        assert (todo["plans"], todo["measures"]) == (None, ["hurto", "comercio", "empleados"])
        puesto = a.post("/api/cameras", json=cam_body(cid, "Puesto", plans=["empleados", "comercio"])).json()
        assert (puesto["plans"], puesto["measures"]) == (["comercio", "empleados"], ["comercio", "empleados"])
        r = a.post("/api/cameras", json=cam_body(kiosco, "Caja", plans=["empleados"]))
        assert r.status_code == 422
        assert r.json()["detail"] == ("La cámara no puede medir «Seguimiento de empleados»: el cliente no lo tiene "
                                      "contratado")
        assert a.post("/api/cameras", json=cam_body(cid, "Otra", plans=["inventado"])).status_code == 422
        # varias a la vez
        r = a.post("/api/cameras/plans", json={"camera_ids": [todo["id"], puesto["id"]], "plans": ["empleados"]})
        assert r.status_code == 200 and [x["measures"] for x in r.json()["cameras"]] == [["empleados"], ["empleados"]]
        # solo empleados: sin hurto la cámara solo mide (sin reglas ni gestos), a pocos cuadros por segundo
        ag = a.post("/api/agents", json={"name": "central", "client_id": None}).json()
        a.put(f"/api/cameras/{todo['id']}", json=cam_body(cid, "Mostrador", agent_id=ag["id"]))
        cfg = {x["id"]: x for x in a.get("/api/agent/config", headers={"X-Agent-Token": ag["token"]}).json()["cameras"]}
        x = cfg[todo["id"]]
        assert x["metrics_only"] is True and x["analysis"]["rules"]["enabled_rules"] == []
        assert "gestures" not in x["analysis"] and x["process_fps"] == METRICS_FPS
        assert x["analysis"]["plans"] == ["empleados"]
        assert x["analysis"]["client_plans"] == ["hurto", "comercio", "empleados"]


# ------------------------------------------------------------------ zona «medir espera»
def test_la_zona_con_espera_se_guarda_y_llega_a_las_maquinas(app):
    with admin(app) as a:
        cid = a.post("/api/clients", json={"name": "Farmacia", "plans": ["comercio", "empleados"]}).json()["id"]
        ag = a.post("/api/agents", json={"name": "central", "client_id": None}).json()
        cam = a.post("/api/cameras", json=cam_body(cid, "Mostrador", agent_id=ag["id"])).json()
        zones = [{"name": "Fila caja", "type": "checkout", "polygon": SQUARE, "wait": True},
                 {"name": "Mostrador", "type": "staff", "polygon": OTHER},
                 {"name": "Fila farmacia", "type": "aisle", "polygon": TOP, "wait": True},
                 {"name": "Góndola", "type": "shelf", "polygon": [[0, 0], [0.05, 0], [0.05, 0.05]], "wait": False}]
        r = a.put(f"/api/cameras/{cam['id']}/analysis", json={"zones": zones, "rules": {}})
        assert r.status_code == 200
        saved = {z["name"]: z for z in a.get(f"/api/cameras/{cam['id']}").json()["zones"]}
        assert saved["Fila caja"]["wait"] is True and saved["Fila farmacia"]["wait"] is True
        assert "wait" not in saved["Mostrador"] and "wait" not in saved["Góndola"]  # apagada: como siempre
        # a las máquinas: tal cual (los tipos nuevos, en extra_zones) y el agente la lee
        h = {"X-Agent-Token": ag["token"]}
        an = a.get("/api/agent/config", headers=h).json()["cameras"][0]["analysis"]
        assert next(z for z in an["zones"] if z["name"] == "Fila caja")["wait"] is True
        assert {z["name"]: z.get("wait") for z in an["extra_zones"]} == {"Mostrador": None, "Fila farmacia": True}
        parsed, _ = parse_config(an)
        assert {z.name for z in parsed if z.wait} == {"Fila caja", "Fila farmacia"}
        # «Configurar…» de la app recibe lo guardado
        a.post("/api/users", json={"username": "ana", "password": "clave-ana-123", "role": "entrenador"})
        e = TestClient(app)
        e.post("/api/login", json={"username": "ana", "password": "clave-ana-123"})
        tok = {"X-Agent-Token": e.post("/api/agent/session").json()["token"]}
        tune = TestClient(app).post("/api/agent/tune", json={"camera_id": cam["id"]}, headers=tok).json()
        assert {z["name"] for z in tune["raw"]["zones"] if z.get("wait")} == {"Fila caja", "Fila farmacia"}
        # en una zona de personal o de surtidor no se mide espera: 422 con el motivo y no se guarda nada
        for ztype in ("staff", "pump"):
            bad = zones[:1] + [{"name": "Mal", "type": ztype, "polygon": OTHER, "wait": True}]
            r = a.put(f"/api/cameras/{cam['id']}/analysis", json={"zones": bad, "rules": {}})
            assert r.status_code == 422 and "tiempo de espera" in r.json()["detail"] and "Mal" in r.json()["detail"]
        assert len(a.get(f"/api/cameras/{cam['id']}").json()["zones"]) == 4


# ------------------------------------------------------------------ métricas
def wait(n, sum_s, max_s, hist, obs_s, queue_s, queue_max, busy_s, unattended_s=0.0, left_alone=0, staffed=1):
    return {"n": n, "sum_s": sum_s, "max_s": max_s, "hist": hist, "obs_s": obs_s, "queue_s": queue_s,
            "queue_max": queue_max, "busy_s": busy_s, "unattended_s": unattended_s, "left_alone": left_alone,
            "staffed": staffed}


# la cámara de la caja (mide todo): a las 10, 3 esperas (30 s, 100 s y 400 s) y a las 11, una de 700 s
CAJA_10 = {"people": 9, "people_dwell_s": 900.0, "frames": 1000,
           "waits": {"Fila caja": wait(3, 530.0, 400.0, [1, 1, 0, 1, 0], 3600.0, 1800.0, 4, 1500.0, 120.0, 1)},
           "posts": {"Mostrador": {"obs_s": 3600.0, "covered_s": 2700.0, "people_s": 3240.0, "max": 2}},
           "staff_s": 5400.0, "staff_max": 2, "staff_dwell": {"Mostrador": 3000.0, "Depósito": 1200.0}}
CAJA_11 = {"people": 2, "people_dwell_s": 800.0, "frames": 500,
           "waits": {"Fila caja": wait(1, 700.0, 700.0, [0, 0, 0, 0, 1], 1800.0, 900.0, 2, 700.0, 300.0, 0)},
           "posts": {"Mostrador": {"obs_s": 1800.0, "covered_s": 1800.0, "people_s": 1800.0, "max": 1}},
           "staff_s": 1800.0, "staff_max": 1, "staff_dwell": {"Mostrador": 1800.0}}
# la de la calle mide solo hurto: lo que haya contado de espera y de personal no se suma al cliente
CALLE_10 = {"people": 30, "people_dwell_s": 300.0, "frames": 1000,
            "waits": {"Vereda": wait(5, 100.0, 30.0, [5, 0, 0, 0, 0], 3600.0, 50.0, 1, 100.0)},
            "posts": {"Garita": {"obs_s": 3600.0, "covered_s": 3600.0, "people_s": 3600.0, "max": 3}},
            "staff_s": 9999.0, "staff_max": 3, "staff_dwell": {"Garita": 500.0}}


def send_metrics(a, plans=("hurto", "comercio", "empleados"), name="Farmacia Centro"):
    cid = a.post("/api/clients", json={"name": name, "plans": list(plans), "timezone": TZ}).json()["id"]
    ag = a.post("/api/agents", json={"name": "central " + name, "client_id": None}).json()
    caja = a.post("/api/cameras", json=cam_body(cid, "Caja", agent_id=ag["id"])).json()
    calle = a.post("/api/cameras", json=cam_body(cid, "Calle", agent_id=ag["id"], plans=["hurto"])).json()
    today = datetime.now(ZoneInfo(TZ)).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
    at = lambda h: (today + timedelta(hours=h)).isoformat()  # las dos horas caen hoy, sea la hora que sea
    r = a.post("/api/agent/metrics", headers={"X-Agent-Token": ag["token"]}, json={"buckets": [
        {"camera_id": caja["id"], "bucket_start": at(10), "data": CAJA_10},
        {"camera_id": caja["id"], "bucket_start": at(11), "data": CAJA_11},
        {"camera_id": calle["id"], "bucket_start": at(10), "data": CALLE_10}]})
    assert r.json()["saved"] == 3
    return cid, today


def test_espera_y_empleados_en_las_metricas_del_cliente(app):
    with admin(app) as a:
        cid, _ = send_metrics(a)
        m = a.get("/api/metrics", params={"client_id": cid, "days": 1}).json()
        t = m["total"]
        assert list(t["waits"]) == ["Fila caja"] and list(t["posts"]) == ["Mostrador"]  # sin la Vereda ni la Garita
        st = wait_stats(t["waits"]["Fila caja"])
        assert (st["n"], st["avg_s"], st["max_s"]) == (4, 307.5, 700.0)
        assert st["over"] == {60: 3, 180: 2, 300: 2, 600: 1}
        assert (st["queue_avg"], st["queue_max"], st["unattended_s"], st["left_alone"]) == (0.5, 4, 420.0, 1)
        assert t["posts"]["Mostrador"] == {"obs_s": 5400.0, "covered_s": 4500.0, "people_s": 5040.0, "max": 2}
        assert (t["staff_s"], t["staff_max"]) == (7200.0, 2)
        assert t["staff_dwell"] == {"Mostrador": 4800.0, "Depósito": 1200.0}
        cams = {x["name"]: x for x in m["by_camera"]}
        assert cams["Caja"]["waits"] == t["waits"] and cams["Caja"]["measures"] == ["hurto", "comercio", "empleados"]
        assert (cams["Calle"]["waits"], cams["Calle"]["posts"], cams["Calle"]["staff_dwell"]) == ({}, {}, {})
        assert (cams["Calle"]["staff_s"], cams["Calle"]["staff_max"]) == (0.0, 0)
        assert cams["Calle"]["frames"] == 1000
        assert sum(h["waits"].get("Fila caja", {}).get("n", 0) for h in m["by_hour"]) == 4
        # la calle mide todo de nuevo (como el cliente): ahora sí se suma
        a.post("/api/cameras/plans", json={"camera_ids": [cams["Calle"]["camera_id"]], "plans": None})
        t = a.get("/api/metrics", params={"client_id": cid, "days": 1}).json()["total"]
        assert set(t["waits"]) == {"Fila caja", "Vereda"} and set(t["posts"]) == {"Mostrador", "Garita"}
        assert t["staff_max"] == 3 and t["staff_dwell"]["Garita"] == 500.0
        # el dueño del comercio ve lo mismo
        a.post("/api/users", json={"username": "dueno", "password": "clave-dueno-1", "role": "cliente",
                                   "client_id": cid})
        u = TestClient(app)
        u.post("/api/login", json={"username": "dueno", "password": "clave-dueno-1"})
        assert u.get("/api/metrics", params={"days": 1}).json()["total"]["waits"] == t["waits"]


def csv_rows(a, cid, day, agrupar="dia"):
    r = a.get("/api/metrics/export", params={"client_id": cid, "desde": day, "hasta": day, "agrupar": agrupar})
    assert r.status_code == 200 and r.text.startswith("﻿")
    rows = list(csv.reader(io.StringIO(r.text.lstrip("﻿")), delimiter=";"))
    head = rows[0]
    return head, [dict(zip(head, row)) for row in rows[1:] if row]


NEW_COLS = ["Esperas", "Espera promedio (min)", "Espera máxima (min)", "Esperaron más de 5 min", "Fila promedio",
            "Esperando sin nadie en el puesto (min)", "Puesto cubierto (%)", "Puesto sin nadie (min)",
            "Empleados a la vez (máx.)"]


def test_el_excel_trae_la_espera_y_los_puestos(app):
    with admin(app) as a:
        cid, today = send_metrics(a)
        head, rows = csv_rows(a, cid, today.date().isoformat())
        assert head[-len(NEW_COLS):] == NEW_COLS and head[-len(NEW_COLS) - 1] == "Zonas (visitas · minutos)"
        by_cam = {r["Cámara"]: r for r in rows}
        assert [by_cam["Caja"][k] for k in NEW_COLS] == ["4", num(307.5 / 60), num(700 / 60), "2", "0,5", "7,0",
                                                         "83", "15,0", "2"]
        assert all(by_cam["Calle"][k] == "" for k in NEW_COLS)  # mide solo hurto
        # por hora: cada hora con lo suyo
        head, rows = csv_rows(a, cid, today.date().isoformat(), "hora")
        assert "Hora" in head and head[-len(NEW_COLS):] == NEW_COLS
        caja = {r["Hora"]: r for r in rows if r["Cámara"] == "Caja"}
        assert [caja["10:00"][k] for k in NEW_COLS] == ["3", num(530 / 3 / 60), num(400 / 60), "1", "0,5", "2,0",
                                                        "75", "15,0", "2"]
        assert [caja["11:00"][k] for k in NEW_COLS] == ["1", num(700 / 60), num(700 / 60), "1", "0,5", "5,0",
                                                        "100", "0,0", "1"]
        # sin el plan, sus columnas van vacías (como en la app, la web y el PDF)
        sin_empleados, _ = send_metrics(a, plans=("hurto", "comercio"), name="Kiosco Norte")
        _, rows = csv_rows(a, sin_empleados, today.date().isoformat())
        caja = next(r for r in rows if r["Cámara"] == "Caja")
        assert caja["Esperas"] == "4" and all(caja[k] == "" for k in NEW_COLS[-3:])
        sin_comercio, _ = send_metrics(a, plans=("hurto", "empleados"), name="Oficina Sur")
        _, rows = csv_rows(a, sin_comercio, today.date().isoformat())
        caja = next(r for r in rows if r["Cámara"] == "Caja")
        assert all(caja[k] == "" for k in NEW_COLS[:6]) and caja["Puesto cubierto (%)"] == "83"


def test_el_excel_junta_varias_zonas_de_una_camara():
    b = merge_bucket(empty_bucket(), {
        "waits": {"Caja 1": wait(2, 200.0, 150.0, [0, 2, 0, 0, 0], 3600.0, 3600.0, 3, 900.0, 600.0, 1),
                  "Caja 2": wait(2, 1000.0, 700.0, [0, 0, 1, 0, 1], 3600.0, 1800.0, 5, 1200.0, 300.0, 0),
                  "Sin nadie": wait(0, 0.0, 0.0, [0] * 5, 0.0, 0.0, 0, 0.0)},  # zona sin mirar: no cuenta
        "posts": {"Caja 1 adentro": {"obs_s": 3600.0, "covered_s": 3600.0, "people_s": 3600.0, "max": 1},
                  "Caja 2 adentro": {"obs_s": 3600.0, "covered_s": 1800.0, "people_s": 1800.0, "max": 2}},
        "staff_max": 3})
    # esperas y segundos sumados, el máximo el mayor, la fila por zona y el rato sin nadie, el mayor de las dos
    assert wait_csv_cells(b, num) == ["4", num(1200 / 4 / 60), num(700 / 60), "1", num(5400 / 7200), num(600 / 60)]
    assert staff_csv_cells(b, num) == ["75", num(1800 / 60), "3"]
    assert wait_csv_cells(empty_bucket(), num) == [""] * 6 and staff_csv_cells(empty_bucket(), num) == [""] * 3
    # sin zona de personal, el rato sin nadie no se sabe
    solo = {"waits": {"Fila": wait(1, 90.0, 90.0, [0, 1, 0, 0, 0], 600.0, 90.0, 1, 90.0, staffed=0)}}
    assert wait_csv_cells(merge_bucket(empty_bucket(), solo), num)[-1] == ""
    # sin puestos pero con empleados a la vista (llegaron con el local cerrado): solo el máximo
    assert staff_csv_cells(merge_bucket(empty_bucket(), {"staff_max": 2}), num) == ["", "", "2"]


# ------------------------------------------------------------------ PDF
def test_el_pdf_trae_tiempo_de_espera_y_empleados(app):
    pytest.importorskip("fpdf")
    with admin(app) as a:
        cid, _ = send_metrics(a)
        r = a.get("/api/metrics/report.pdf", params={"client_id": cid, "days": 1})
        assert r.status_code == 200 and r.content.startswith(b"%PDF")
        text = text_of(r.content)
        assert "Seguimiento de empleados" in text  # en los planes del encabezado
        assert "Tiempo de espera" in text and "Fila caja" in text
        assert "5:08" in text and "11:40" in text  # promedio (307,5 s) y máximo (700 s)
        assert "con clientes esperando y nadie en el puesto: 7 min" in text
        assert "se fueron sin que hubiera nadie: 1" in text
        assert "Empleados a la vista: máximo 2 a la vez, promedio 1,3" in text
        assert "Mostrador" in text and "83 %" in text and "15:00" in text
        assert "Dónde pasan el tiempo: Mostrador 1 h 20 min · Depósito 20:00" in text
        assert "Vereda" not in text and "Garita" not in text  # la calle no lo mide
        assert "si la zona incluye el lugar donde se atiende, incluye la atención" in text
        assert "no se sabe quién es quién" in text
        # sin los planes, sin las secciones
        otro, _ = send_metrics(a, plans=("hurto",), name="Kiosco Oeste")
        text = text_of(a.get("/api/metrics/report.pdf", params={"client_id": otro, "days": 1}).content)
        assert "Tiempo de espera" not in text and "Empleados a la vista" not in text


def test_el_pdf_de_varios_dias_y_sin_zona_de_personal():
    pytest.importorskip("fpdf")
    from server.reporte import clock, metrics_pdf, minutes

    assert (clock(None), clock(4.4), clock(80), clock(3599.6), clock(4320), clock(7170)) == (
        "-", "0:04", "1:20", "1 h 0 min", "1 h 12 min", "2 h 0 min")
    assert (minutes(0), minutes(20), minutes(420), minutes(4320)) == ("0 min", "menos de 1 min", "7 min", "1 h 12 min")
    day1 = merge_bucket(empty_bucket(), {"frames": 10, "staff_max": 1,
                                         "waits": {"Fila": wait(2, 100.0, 70.0, [1, 1, 0, 0, 0], 600.0, 60.0, 2, 90.0,
                                                                staffed=0)}})
    day2 = merge_bucket(empty_bucket(), {"frames": 10,
                                         "waits": {"Fila": wait(1, 400.0, 400.0, [0, 0, 0, 1, 0], 600.0, 400.0, 1,
                                                                400.0, staffed=0)}})
    total = merge_bucket(day1, day2)
    m = {"total": total, "by_day": [{"date": "2026-11-02", **day1}, {"date": "2026-11-03", **day2}],
         "by_hour": [], "by_camera": [{"camera_id": 1, "name": "Caja", **total}]}
    text = text_of(metrics_pdf("Almacén", ["comercio", "empleados"], date(2026, 11, 2), date(2026, 11, 3), m, None))
    assert "Tiempo de espera" in text and "lun 02/11" in text and "mar 03/11" in text  # una fila por día
    assert "marcá la zona «Personal (empleados)»" in text  # sin personal: no se sabe si había quien atienda
    assert "falta marcar la zona «Personal (empleados)»" in text  # ni puestos
    assert "Empleados a la vista: máximo 1 a la vez" in text and "promedio" not in text.split("a la vista")[1][:40]
