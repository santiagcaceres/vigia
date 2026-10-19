"""Alarmas que avisa el propio grabador (modo alarmas_equipo): Dahua y Hikvision -> agente -> servidor."""

import shutil
import time
from datetime import timedelta

import httpx
import pytest
from fastapi.testclient import TestClient

from agent.alarms import AlarmRunner
from agent.device_alarms import DeviceAlarmListener
from agent.device_events import (DeviceEvent, dahua_attach_url, dahua_events, hik_events, parse_dahua_line,
                                 parse_hik_alert, split_hik_stream)
from agent.cameras import build_camera
from agent.outbox import ServerClient
from core import synthetic
from server.db import Alarm
from server.main import Settings, create_app

ADMIN = ("admin", "clave-admin-123")

HIK_ALERT = """<EventNotificationAlert version="2.0" xmlns="http://www.isapi.org/ver20/XMLSchema">
<ipAddress>192.168.1.64</ipAddress><channelID>{ch}</channelID><dateTime>2026-10-19T10:00:00-03:00</dateTime>
<activePostCount>1</activePostCount><eventType>{etype}</eventType><eventState>{state}</eventState>
<eventDescription>x</eventDescription>{extra}</EventNotificationAlert>"""


def hik_block(ch=2, etype="VMD", state="active", extra=""):
    return ("--boundary\r\nContent-Type: application/xml; charset=\"UTF-8\"\r\nContent-Length: 1\r\n\r\n"
            + HIK_ALERT.format(ch=ch, etype=etype, state=state, extra=extra) + "\r\n").encode()


# ---------------------------------------------------------------- parseo
def test_dahua_lineas():
    ev = parse_dahua_line("Code=VideoMotion;action=Start;index=1;data={\"RegionName\":[\"Region1\"];\"x\":1}")
    assert (ev.event_type, ev.channel, ev.target) == ("VMD", 2, None)
    assert parse_dahua_line("Code=VideoMotion;action=Stop;index=1") is None
    assert parse_dahua_line("Heartbeat") is None
    assert parse_dahua_line("Code=AlarmLocal;action=Start;index=0") is None  # no es de video
    ev = parse_dahua_line("Code=CrossLineDetection;action=Start;index=0;data={\"Object\":{\"ObjectType\":\"Human\"}}")
    assert (ev.event_type, ev.channel, ev.target) == ("linedetection", 1, "human")
    assert parse_dahua_line("Code=SmartMotionVehicle;action=Start;index=0").target == "vehicle"
    assert parse_dahua_line("Code=CrossRegionDetection;action=Start").channel is None
    assert "codes=[" in dahua_attach_url(build_camera({"host": "h", "brand": "dahua"}))


def test_hikvision_bloques():
    assert parse_hik_alert(HIK_ALERT.format(ch=1, etype="videoloss", state="inactive", extra="")) is None  # latido
    ev = parse_hik_alert(HIK_ALERT.format(ch=3, etype="fielddetection", state="active",
                                          extra="<DetectionRegionList><DetectionRegionEntry><detectionTarget>human"
                                                "</detectionTarget></DetectionRegionEntry></DetectionRegionList>"))
    assert (ev.event_type, ev.channel, ev.target) == ("fielddetection", 3, "human")
    assert parse_hik_alert(b"<basura") is None
    # el multipart llega en pedazos arbitrarios: se arman los bloques completos
    data = hik_block(1, "videoloss", "inactive") + hik_block(2) + hik_block(2, "linedetection")
    chunks = [data[i:i + 37] for i in range(0, len(data), 37)]
    blocks = list(split_hik_stream(iter(chunks)))
    assert len(blocks) == 3
    evs = [parse_hik_alert(b) for b in blocks]
    assert [e.event_type for e in evs if e] == ["VMD", "linedetection"]


def test_stream_http_de_ambas_marcas():
    def dahua_handler(request):
        assert request.url.path == "/cgi-bin/eventManager.cgi"
        return httpx.Response(200, content=iter([b"Code=VideoMotion;action=Start;index=0\r\n",
                                                 b"Heartbeat\r\nCode=VideoMotion;action=Stop;index=0\r\n",
                                                 b"Code=CrossLineDetection;action=Start;index=0\r\n"]))
    cam = build_camera({"host": "10.0.0.5", "brand": "dahua", "channel": 1})
    evs = list(dahua_events(cam, httpx.Client(transport=httpx.MockTransport(dahua_handler))))
    assert [e.event_type for e in evs] == ["VMD", "linedetection"]

    def hik_handler(request):
        assert request.url.path == "/ISAPI/Event/notification/alertStream"
        return httpx.Response(200, content=iter([hik_block(1, "videoloss", "inactive"), hik_block(2)]))
    cam = build_camera({"host": "10.0.0.6", "brand": "hikvision", "channel": 2})
    evs = list(hik_events(cam, httpx.Client(transport=httpx.MockTransport(hik_handler))))
    assert [(e.event_type, e.channel) for e in evs] == [("VMD", 2)]

    def denied(request):
        return httpx.Response(401)
    with pytest.raises(httpx.HTTPStatusError):
        list(hik_events(cam, httpx.Client(transport=httpx.MockTransport(denied))))


# ---------------------------------------------------------------- servidor + agente
@pytest.fixture
def app(tmp_path, db_url):
    return create_app(Settings(data_dir=tmp_path / "data", database_url=db_url, admin_password=ADMIN[1],
                               serve_public=False))


@pytest.fixture
def admin(app):
    with TestClient(app) as c:
        assert c.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]}).status_code == 200
        yield c


def setup(admin, brand="dahua", **extra):
    cid = admin.post("/api/clients", json={"name": "Almacén " + brand, "timezone": "America/Montevideo"}).json()["id"]
    ag = admin.post("/api/agents", json={"name": "PC central " + brand}).json()
    body = {"client_id": cid, "agent_id": ag["id"], "name": "Góndola", "host": "100.64.0.9", "password": "x",
            "channel": 2, "brand": brand, "detector": "motion", "process_fps": synthetic.FPS,
            "mode": "alarmas_equipo", "alarm_cfg": {"event_types": ["VMD", "linedetection"], "pre_seconds": 5,
                                                    "post_seconds": 10}, **extra}
    r = admin.post("/api/cameras", json=body)
    assert r.status_code == 201, r.text
    cam = r.json()
    admin.put(f"/api/cameras/{cam['id']}/analysis", json={"zones": synthetic.STORE_ZONES, "rules": {}})
    return cid, ag, cam


def agent_cfg(app, token, cam_id):
    cfg = TestClient(app).get("/api/agent/config", headers={"X-Agent-Token": token}).json()
    return next(c for c in cfg["cameras"] if c["id"] == cam_id)


def test_dahua_no_puede_usar_hpp_pero_si_alarmas_del_equipo(admin):
    cid, ag, cam = setup(admin)
    assert cam["mode"] == "alarmas_equipo" and cam["brand"] == "dahua" and cam["hpp_serial"] is None
    r = admin.post("/api/cameras", json={"client_id": cid, "name": "x", "host": "h", "brand": "dahua",
                                         "mode": "alarmas", "hpp_serial": "ABC"})
    assert r.status_code == 422 and "Hikvision" in r.json()["detail"]


def test_listener_filtra_y_registra_en_el_servidor(app, admin):
    _, ag, cam = setup(admin)
    sc = ServerClient("http://testserver", ag["token"], http=TestClient(app))
    lst = DeviceAlarmListener(agent_cfg(app, ag["token"], cam["id"]), sc)
    assert lst.cam.brand == "dahua" and lst.cam.channel == 2
    assert not lst.handle(DeviceEvent("VMD", 1))  # otro canal del mismo grabador
    assert not lst.handle(DeviceEvent("fielddetection", 2))  # tipo no elegido para esta cámara
    assert lst.handle(DeviceEvent("VMD", 2))
    assert not lst.handle(DeviceEvent("VMD", 2))  # repetido enseguida: no se vuelve a mandar
    assert lst.handle(DeviceEvent("linedetection", None))  # cámara que no informa canal
    lst._last.clear()
    assert lst.handle(DeviceEvent("VMD", 2))
    items = admin.get("/api/alarms").json()["items"]
    assert len(items) == 1  # los tres disparos seguidos se agruparon en una alarma
    a = items[0]
    assert a["source"] == "equipo" and a["triggers"] == 3 and a["status"] == "pendiente"
    assert a["event_type"] == "VMD+linedetection" and a["camera_name"] == "Góndola"
    assert lst.status()["events"] == 3 and lst.status()["state"] == "iniciando"
    # un vehículo según el propio grabador se descarta sin crear alarma
    assert lst.handle(DeviceEvent("linedetection", 2, target="vehicle")) is True
    assert len(admin.get("/api/alarms").json()["items"]) == 1


def test_el_servidor_rechaza_alarmas_fuera_de_lugar(app, admin):
    _, ag, cam = setup(admin, brand="hikvision")
    otro = admin.post("/api/agents", json={"name": "Otra PC"}).json()
    body = {"camera_id": cam["id"], "event_type": "VMD", "occurred_at": "2026-10-19T10:00:00-03:00"}
    h = lambda t: {"X-Agent-Token": t}
    assert TestClient(app).post("/api/agent/alarms", json=body, headers=h(otro["token"])).status_code == 404
    assert TestClient(app).post("/api/agent/alarms", json=body, headers=h(ag["token"])).status_code == 201
    assert TestClient(app).post("/api/agent/alarms", json={**body, "event_type": "regionExiting"},
                                headers=h(ag["token"])).json() == {"result": "ignorada"}
    admin.put(f"/api/cameras/{cam['id']}", json={**cam, "client_id": cam["client_id"], "mode": "vivo"})
    assert TestClient(app).post("/api/agent/alarms", json=body, headers=h(ag["token"])).status_code == 409


def test_flujo_completo_grabador_agente_evento(app, admin, tmp_path):
    _, ag, cam = setup(admin)
    sc = ServerClient("http://testserver", ag["token"], http=TestClient(app))

    def fuente(_cam):
        yield DeviceEvent("VMD", 2, raw={"code": "VideoMotion"})
        yield DeviceEvent("VMD", 1)  # de otro canal
        lst.stop()  # el grabador real no corta nunca; acá terminamos la "conexión"

    lst = DeviceAlarmListener(agent_cfg(app, ag["token"], cam["id"]), sc, source=fuente)
    lst.thread.start()
    lst.thread.join(10)
    assert lst.events == 1 and lst.status()["state"] == "detenida"
    with app.state.Session() as s:  # simular que ya pasó el "después"
        for a in s.query(Alarm).all():
            a.first_at -= timedelta(minutes=5)
            a.last_at -= timedelta(minutes=5)
        s.commit()
    video = synthetic.write_video(synthetic.render("hurto"), tmp_path / "hurto.mp4")
    pedidos = []

    def fetch(hik, start, end, out):
        pedidos.append((hik.brand, hik.playback_url(start, end)))
        shutil.copy(video, out)
        return True

    runner = AlarmRunner(sc, None, tmp_path / "w", fetch=fetch)
    assert runner.run_once() and not runner.run_once()
    assert pedidos[0][0] == "dahua" and "/cam/playback?channel=2&starttime=" in pedidos[0][1]
    ev = admin.get("/api/events", params={"source": "alarma"}).json()["items"][0]
    assert ev["type"] == "salida_sin_pago" and ev["alarm"]["dvr_event"] == "VMD"
    a = admin.get("/api/alarms").json()["items"][0]
    assert a["status"] == "con_persona" and a["event_id"] == ev["id"] and a["source"] == "equipo"


def test_listener_reintenta_si_el_grabador_no_responde(app, admin):
    _, ag, cam = setup(admin)
    sc = ServerClient("http://testserver", ag["token"], http=TestClient(app))
    calls = []

    def fuente(_cam):
        calls.append(time.monotonic())
        raise httpx.HTTPStatusError("401", request=httpx.Request("GET", "http://x"), response=httpx.Response(401))
        yield  # pragma: no cover

    lst = DeviceAlarmListener(agent_cfg(app, ag["token"], cam["id"]), sc, source=fuente)
    lst.thread.start()
    time.sleep(0.5)
    assert lst.status()["state"] == "sin_conexion" and "401" in lst.status()["error"]
    lst.stop()
    lst.thread.join(5)
    assert lst.status()["state"] == "detenida" and len(calls) >= 1
