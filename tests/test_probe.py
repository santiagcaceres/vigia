"""Prueba de conexión al grabador desde el servidor (grabador expuesto por el router del cliente)."""

import httpx
import pytest
from fastapi.testclient import TestClient

from server import probe
from server.main import Settings, create_app

ADMIN = ("admin", "clave-admin-123")

HIK_INFO = b"""<?xml version="1.0"?><DeviceInfo xmlns="http://www.isapi.org/ver20/XMLSchema">
<deviceName>DVR Almacen</deviceName><model>DS-7104HGHI-M1</model><serialNumber>E44034229</serialNumber>
<firmwareVersion>V4.30</firmwareVersion><deviceType>DVR</deviceType></DeviceInfo>"""
HIK_CHANNELS = b"""<VideoInputChannelList xmlns="http://www.isapi.org/ver20/XMLSchema">
<VideoInputChannel><id>1</id><name>Caja</name></VideoInputChannel>
<VideoInputChannel><id>2</id><name>Gondola</name></VideoInputChannel></VideoInputChannelList>"""


def hik_handler(request):
    if request.url.path == "/ISAPI/System/deviceInfo":
        return httpx.Response(200, content=HIK_INFO)
    if request.url.path == "/ISAPI/ContentMgmt/InputProxy/channels":
        return httpx.Response(404)
    if request.url.path == "/ISAPI/System/Video/inputs/channels":
        return httpx.Response(200, content=HIK_CHANNELS)
    return httpx.Response(404)


def dahua_handler(request):
    q = str(request.url)
    if "getSystemInfo" in q:
        return httpx.Response(200, text="deviceType=XVR5104HS\nserialNumber=ABC123\n")
    if "getDeviceType" in q:
        return httpx.Response(200, text="type=XVR5104HS\n")
    if "ChannelTitle" in q:
        return httpx.Response(200, text="table.ChannelTitle[0].Name=Entrada\ntable.ChannelTitle[1].Name=Gondola\n")
    return httpx.Response(404)


def test_probe_hikvision_y_dahua():
    r = probe.probe("hikvision", "190.64.1.2", 38080, "vigia", "x", httpx.Client(transport=httpx.MockTransport(hik_handler)))
    assert r["ok"] and r["model"] == "DS-7104HGHI-M1" and r["serialNumber"] == "E44034229"
    assert [c["channel"] for c in r["channels"]] == [1, 2] and r["channels"][1]["name"] == "Gondola"
    r = probe.probe("dahua", "190.64.1.2", 38080, "vigia", "x", httpx.Client(transport=httpx.MockTransport(dahua_handler)))
    assert r["ok"] and r["model"] == "XVR5104HS" and [c["name"] for c in r["channels"]] == ["Entrada", "Gondola"]


def test_probe_errores_explicados():
    def h401(_):
        return httpx.Response(401)
    r = probe.probe("hikvision", "h", 80, "u", "p", httpx.Client(transport=httpx.MockTransport(h401)))
    assert not r["ok"] and "401" in r["error"]

    def h404(_):
        return httpx.Response(404)
    assert "marca" in probe.probe("dahua", "h", 80, "u", "p", httpx.Client(transport=httpx.MockTransport(h404)))["error"]

    def down(request):
        raise httpx.ConnectError("refused", request=request)
    assert "router" in probe.probe("hikvision", "h", 80, "u", "p", httpx.Client(transport=httpx.MockTransport(down)))["error"]

    def slow(request):
        raise httpx.ReadTimeout("t", request=request)
    assert "tiempo" in probe.probe("hikvision", "h", 80, "u", "p", httpx.Client(transport=httpx.MockTransport(slow)))["error"]


def test_url_rtsp_y_host_publico():
    assert probe.rtsp_url("hikvision", "almacen.ddns.net", 38554, 2, "vigia", "p@ss") == \
        "rtsp://vigia:p%40ss@almacen.ddns.net:38554/Streaming/Channels/202"
    assert probe.rtsp_url("dahua", "190.64.1.2", 554, 1, "vigia", "x", "main").endswith("/cam/realmonitor?channel=1&subtype=0")
    assert probe.is_public_host("190.64.1.2") and probe.is_public_host("almacen.ddns.net")
    assert not probe.is_public_host("192.168.1.64") and not probe.is_public_host("100.64.0.7")
    assert not probe.is_public_host("pc-local.tail1234.ts.net")


@pytest.fixture
def app(tmp_path, db_url):
    return create_app(Settings(data_dir=tmp_path / "data", database_url=db_url, admin_password=ADMIN[1],
                               serve_public=False))


def test_endpoint_probar_conexion(app, monkeypatch):
    with TestClient(app) as c:
        c.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]})
        cid = c.post("/api/clients", json={"name": "Almacén"}).json()["id"]
        cam = c.post("/api/cameras", json={"client_id": cid, "name": "Góndola", "host": "190.64.1.2", "rtsp_port": 38554,
                                           "http_port": 38080, "channel": 2, "password": "clave", "mode": "alarmas_equipo"}).json()
        seen = {}

        def fake(brand, host, port, user, password, http=None):
            seen.update(brand=brand, host=host, port=port, user=user, password=password)
            return {"ok": True, "error": None, "model": "DS-7104", "serialNumber": "E1", "channels": [{"channel": 2, "name": "G"}]}
        monkeypatch.setattr(probe, "probe", fake)
        r = c.post(f"/api/cameras/{cam['id']}/probe").json()
        assert r["ok"] and r["public"] and not r["standard_ports"]
        assert seen == {"brand": "hikvision", "host": "190.64.1.2", "port": 38080, "user": "admin", "password": "clave"}
        assert r["rtsp_url"] == "rtsp://admin:CLAVE@190.64.1.2:38554/Streaming/Channels/202"
        # el rol cliente no puede probar
        c.post("/api/users", json={"username": "dueno", "password": "clave-dueno-1", "role": "cliente", "client_id": cid})
        u = TestClient(app)
        u.post("/api/login", json={"username": "dueno", "password": "clave-dueno-1"})
        assert u.post(f"/api/cameras/{cam['id']}/probe").status_code == 403
