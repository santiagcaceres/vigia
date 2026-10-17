"""Alarmas de Hik-Partner Pro: webhook firmado -> alarma -> PC central baja y analiza el video."""

import json
import shutil
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine, inspect

from agent.alarms import AlarmRunner
from agent.outbox import ServerClient
from core import synthetic
from server import hpp
from server.db import Alarm, add_missing_columns, utcnow
from server.main import Settings, create_app

ADMIN = ("admin", "clave-admin-123")
SECRET = "firmaSecreta123"
SERIAL = "E44034229"

VMD_XML = """<?xml version="1.0" encoding="UTF-8"?>
<EventNotificationAlert version="2.0" xmlns="http://www.isapi.org/ver20/XMLSchema">
 <deviceSerial>{serial}</deviceSerial><channelID>{ch}</channelID>
 <triggerTime>{when}</triggerTime><eventType>{etype}</eventType>
 <eventDescription>Motion alarm</eventDescription>{extra}
</EventNotificationAlert>"""


def xml_msg(when="2026-10-17T10:00:00", ch=2, etype="VMD", serial=SERIAL, extra=""):
    return {"formatType": "XML", "accountNumber": "x", "deviceSerial": serial,
            "alarmData": VMD_XML.format(serial=serial, ch=ch, when=when, etype=etype, extra=extra)}


# ---------------------------------------------------------------- funciones puras
def test_firma_del_webhook():
    ts = str(int(time.time() * 1000))
    sig = hpp.sign(SECRET, ts, "lote1")
    assert sig.startswith("sha256=") and len(sig) == 7 + 64
    assert hpp.verify(SECRET, ts, "lote1", sig)
    assert not hpp.verify(SECRET, ts, "lote2", sig)
    assert not hpp.verify("otra", ts, "lote1", sig)
    viejo = str(int(time.time() * 1000) - 3600_000)
    assert not hpp.verify(SECRET, viejo, "lote1", hpp.sign(SECRET, viejo, "lote1"))  # repetido una hora después


def test_interpreta_mensajes_xml_y_json():
    batch, msgs = hpp.extract_messages({"data": {"batchId": "b1", "list": [xml_msg(), {
        "formatType": "JSON", "deviceSerial": "ds2323",
        "alarmData": {"deviceSerial": "ds2323", "eventType": "fielddetection", "channelID": "4",
                      "dateTime": "2026-10-17T10:00:05-03:00",
                      "DetectionRegionList": {"DetectionRegionEntry": {"detectionTarget": "vehicle"}}}}]}})
    assert batch == "b1" and len(msgs) == 2
    a, b = (hpp.parse_message(m) for m in msgs)
    assert (a.device_serial, a.channel, a.event_type, a.target) == (SERIAL, 2, "VMD", None)
    assert a.occurred_utc("America/Montevideo").isoformat() == "2026-10-17T13:00:00"
    assert (b.channel, b.event_type, b.target) == (4, "fielddetection", "vehicle")
    assert b.occurred_utc("Europe/Madrid").isoformat() == "2026-10-17T13:00:05"  # trae su desfase
    assert hpp.parse_message({"alarmData": "basura"}) is None
    assert hpp.extract_messages([1, 2]) == (None, [])
    _, single = hpp.extract_messages(xml_msg())  # un solo mensaje suelto
    assert len(single) == 1


def test_agrega_columnas_nuevas_a_tablas_viejas(tmp_path):
    url = f"sqlite:///{tmp_path / 'vieja.db'}"
    eng = create_engine(url)
    md = MetaData()
    Table("cameras", md, Column("id", Integer, primary_key=True), Column("name", String(120)))
    md.create_all(eng)
    from server.db import Base
    Base.metadata.create_all(eng)  # crea las tablas que faltan, pero no toca "cameras"
    added = add_missing_columns(eng)
    assert "cameras.mode" in added and "cameras.hpp_serial" in added
    cols = {c["name"]: c for c in inspect(eng).get_columns("cameras")}
    assert {"mode", "hpp_serial", "alarm_cfg", "rules"} <= set(cols)
    assert add_missing_columns(eng) == []


# ---------------------------------------------------------------- servidor
@pytest.fixture
def app(tmp_path, db_url):
    st = Settings(data_dir=tmp_path / "data", database_url=db_url, admin_password=ADMIN[1], serve_public=False,
                  hpp_app_key="ak", hpp_secret_key="sk", hpp_sign_secret=SECRET, public_url="https://vigia.test")
    return create_app(st)


@pytest.fixture
def admin(app):
    with TestClient(app) as c:
        r = c.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]})
        c.headers["Authorization"] = "Bearer " + r.json()["token"]  # con PUBLIC_URL https la cookie es "secure"
        yield c


def central_setup(admin, **cam):
    cid = admin.post("/api/clients", json={"name": "Super Pocitos", "timezone": "America/Montevideo"}).json()["id"]
    ag = admin.post("/api/agents", json={"name": "PC central"}).json()
    body = {"client_id": cid, "agent_id": ag["id"], "name": "Góndola bebidas", "host": "100.64.0.7",
            "password": "x", "channel": 2, "detector": "motion", "process_fps": synthetic.FPS,
            "mode": "alarmas", "hpp_serial": SERIAL.lower(),
            "alarm_cfg": {"event_types": ["VMD", "fielddetection"], "pre_seconds": 5, "post_seconds": 10}, **cam}
    r = admin.post("/api/cameras", json=body)
    assert r.status_code == 201, r.text
    c = r.json()
    admin.put(f"/api/cameras/{c['id']}/analysis", json={"zones": synthetic.STORE_ZONES, "rules": {}})
    return cid, ag, c


def send_webhook(app, messages, secret=SECRET, batch="lote-1"):
    ts = str(int(time.time() * 1000))
    return TestClient(app).post("/api/hpp/webhook", content=json.dumps({"batchId": batch, "list": messages}),
                                headers={"X-Hook-Batch-Id": batch, "X-Hook-Timestamp": ts,
                                         "X-Hook-Signature": hpp.sign(secret, ts, batch),
                                         "Content-Type": "application/json"})


def make_ready(app):
    """Simula que ya pasó el tiempo para que el DVR haya grabado el "después" de la alarma."""
    with app.state.Session() as s:
        for a in s.query(Alarm).all():
            a.first_at -= timedelta(minutes=5)
            a.last_at -= timedelta(minutes=5)
        s.commit()


def test_validacion_de_url_y_firma(app):
    r = TestClient(app).get("/api/hpp/webhook", headers={"X-Hook-Batch-Id": "abc", "X-Hook-Timestamp": "123"})
    assert r.status_code == 200 and r.headers["X-Hook-Signature"] == hpp.sign(SECRET, "123", "abc")
    assert send_webhook(app, [xml_msg()], secret="otra").status_code == 401


def test_camara_en_modo_alarmas_requiere_serie(admin):
    cid = admin.post("/api/clients", json={"name": "X"}).json()["id"]
    r = admin.post("/api/cameras", json={"client_id": cid, "name": "a", "host": "h", "mode": "alarmas"})
    assert r.status_code == 422


def test_webhook_agrupa_filtra_y_registra_desconocidas(app, admin):
    cid, ag, cam = central_setup(admin)
    assert cam["hpp_serial"] == SERIAL and cam["mode"] == "alarmas"
    r = send_webhook(app, [
        xml_msg("2026-10-17T10:00:00"), xml_msg("2026-10-17T10:00:06"),  # mismo movimiento: se agrupa
        xml_msg("2026-10-17T10:00:06"),  # reenvío idéntico
        xml_msg("2026-10-17T10:05:00", etype="linedetection"),  # tipo no vigilado en esta cámara
        xml_msg("2026-10-17T10:06:00", etype="fielddetection",
                extra="<DetectionRegionList><DetectionRegionEntry><detectionTarget>vehicle</detectionTarget>"
                      "</DetectionRegionEntry></DetectionRegionList>"),  # el DVR dice que es un vehículo
        xml_msg("2026-10-17T10:07:00", ch=9),  # canal sin cámara cargada
        {"formatType": "JSON", "alarmData": "{}"},
    ])
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["nueva"] == 1 and res["agrupada"] == 2 and res["ignorada"] == 1
    assert res["no_persona_dvr"] == 1 and res["sin_camara"] == 1 and res["sin_datos"] == 1
    data = admin.get("/api/alarms").json()
    by_status = {a["status"]: a for a in data["items"]}
    a = by_status["pendiente"]
    assert a["triggers"] == 3 and a["camera_name"] == "Góndola bebidas" and a["client_name"] == "Super Pocitos"
    assert a["first_at"] == "2026-10-17T13:00:00Z" and a["last_at"] == "2026-10-17T13:00:06Z"
    assert by_status["sin_camara"]["channel"] == 9 and by_status["sin_camara"]["camera_id"] is None
    assert data["counts_7d"] == {"pendiente": 1, "sin_camara": 1}
    # al cargar la cámara del canal 9, la alarma "sin cámara" se puede reprocesar
    sid = by_status["sin_camara"]["id"]
    assert admin.post(f"/api/alarms/{sid}/retry").status_code == 409
    admin.post("/api/cameras", json={"client_id": cid, "agent_id": ag["id"], "name": "Caja", "host": "100.64.0.7",
                                     "channel": 9, "mode": "alarmas", "hpp_serial": SERIAL})
    assert admin.post(f"/api/alarms/{sid}/retry").json()["status"] == "pendiente"


def test_pc_central_baja_analiza_y_descarta(app, admin, tmp_path):
    _, ag, cam = central_setup(admin)
    cfg = TestClient(app).get("/api/agent/config", headers={"X-Agent-Token": ag["token"]}).json()
    assert cfg["cameras"][0]["mode"] == "alarmas" and cfg["cameras"][0]["alarm_cfg"]["post_seconds"] == 10
    now = datetime.now(ZoneInfo("America/Montevideo")).replace(microsecond=0, tzinfo=None)
    send_webhook(app, [xml_msg(now.isoformat()), xml_msg((now + timedelta(seconds=60)).isoformat())])

    videos = {"hurto": synthetic.write_video(synthetic.render("hurto"), tmp_path / "hurto.mp4"),
              "vacio": synthetic.write_video([synthetic._background(0)] * 40, tmp_path / "vacio.mp4")}
    pedidos = []

    def fake_fetch(hik, start, end, out):
        pedidos.append((hik.host, hik.channel, start, end))
        shutil.copy(videos["hurto" if len(pedidos) == 1 else "vacio"], out)
        return True

    sc = ServerClient("http://testserver", ag["token"], http=TestClient(app))
    runner = AlarmRunner(sc, None, tmp_path / "w", fetch=fake_fetch)
    assert runner.run_once() is False  # todavía no pasó el "después": no se entrega
    make_ready(app)
    assert runner.run_once() and runner.run_once()
    assert runner.run_once() is False
    host, ch, start, end = pedidos[0]
    assert (host, ch) == ("100.64.0.7", 2)
    # 5 s antes y 10 s después del disparo (make_ready lo corrió 5 minutos), en hora del local
    assert start.replace(tzinfo=None) == now - timedelta(minutes=5, seconds=5)
    assert end - start == timedelta(seconds=15) and start.utcoffset() == timedelta(hours=-3)

    evs = admin.get("/api/events", params={"source": "alarma"}).json()["items"]
    assert len(evs) == 1
    ev = evs[0]
    assert ev["type"] == "salida_sin_pago" and ev["clip_source"] == "nvr" and ev["clip_url"]
    assert ev["alarm"]["dvr_event"] == "VMD" and ev["alarm"]["stats"]["person_frames"] > 0
    items = {a["status"]: a for a in admin.get("/api/alarms").json()["items"]}
    assert items["con_persona"]["event_id"] == ev["id"]
    assert items["sin_persona"]["stats"]["person_frames"] == 0
    assert runner.status() == {"alarms_with_person": 1, "alarms_discarded": 1, "alarms_failed": 0}
    assert admin.get("/api/stats").json()["total"] == 1


def test_si_no_baja_el_video_reintenta_y_luego_queda_en_error(app, admin, tmp_path):
    _, ag, _ = central_setup(admin)
    send_webhook(app, [xml_msg()])
    make_ready(app)
    sc = ServerClient("http://testserver", ag["token"], http=TestClient(app))
    runner = AlarmRunner(sc, None, tmp_path / "w", fetch=lambda *a: False)
    for _ in range(3):
        assert runner.run_once()
    assert runner.run_once() is False
    a = admin.get("/api/alarms").json()["items"][0]
    assert a["status"] == "error" and "DVR" in a["error"] and a["attempts"] == 3
    assert admin.post(f"/api/alarms/{a['id']}/retry").json()["status"] == "pendiente"


def test_otro_agente_no_recibe_alarmas_ajenas(app, admin):
    central_setup(admin)
    otro = admin.post("/api/agents", json={"name": "Otra PC"}).json()
    send_webhook(app, [xml_msg()])
    make_ready(app)
    r = TestClient(app).get("/api/agent/alarms/next", headers={"X-Agent-Token": otro["token"]})
    assert r.status_code == 204


def test_alarmas_viejas_sin_evento_se_limpian(app, admin):
    central_setup(admin)
    send_webhook(app, [xml_msg(ch=9)])
    with app.state.Session() as s:
        s.query(Alarm).update({"received_at": utcnow() - timedelta(days=20)})
        s.commit()
    st = app.state.settings
    st.cron_secret = "cron"
    r = TestClient(app).get("/api/cron/cleanup", headers={"Authorization": "Bearer cron"})
    assert r.status_code == 200
    assert admin.get("/api/alarms").json()["items"] == []
