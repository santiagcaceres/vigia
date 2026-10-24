"""Cámaras Hikvision por la nube de Hik-Connect (sin puertos): servidor (API simulada) y agente."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from agent import hikconnect as agent_hc
from agent.cameras import build_camera
from server import hikconnect
from server.main import Settings, create_app

ADMIN = ("admin", "clave-admin-123")


class FakeHikConnect:
    """Imita la API: el token solo anda en la región 'isa' y hay dos cámaras del grabador F12345678."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content or b"{}")
        self.calls.append((f"{req.url.host}{req.url.path}", body))
        path = req.url.path
        if path.endswith("/platform/v1/token/get"):
            if req.url.host.startswith("isa.") and body.get("secretKey") == "secreto":
                return httpx.Response(200, json={"errorCode": "0", "data": {
                    "accessToken": "tok", "expireTime": 9_999_999_999_000, "areaDomain": "https://isaapi.example"}})
            return httpx.Response(200, json={"errorCode": "EVZ1001", "message": "cuenta inexistente"})
        assert req.headers.get("Token") == "tok"
        if path.endswith("/areas/cameras/get"):
            return httpx.Response(200, json={"errorCode": "0", "data": {"camera": [
                {"id": "res-1", "name": "Caja", "channelNo": 1, "device": {"devInfo": {"serialNo": "F12345678"}}},
                {"id": "res-2", "name": "Puerta", "channelNo": 2, "device": {"devInfo": {"serialNo": "F12345678"}}}]}})
        if path.endswith("/live/address/get"):
            if body["resourceId"] == "res-enc" and not body.get("code"):
                return httpx.Response(200, json={"errorCode": "EVZ60019",
                                                 "message": "Encryption is enabled and parameter code is empty"})
            return httpx.Response(200, json={"errorCode": "0", "data": {
                "url": f"https://hls.example/{body['resourceId']}/t{body['type']}.m3u8?token=abc"}})
        return httpx.Response(404)


@pytest.fixture
def fake(monkeypatch):
    f = FakeHikConnect()
    real = httpx.Client
    monkeypatch.setattr("server.hikconnect.httpx.Client", lambda **kw: real(transport=httpx.MockTransport(f), **kw))
    hikconnect._tokens.clear()
    return f


@pytest.fixture
def app(tmp_path, db_url):
    return create_app(Settings(data_dir=tmp_path / "data", database_url=db_url, admin_password=ADMIN[1],
                               serve_public=False))


def test_camara_por_hik_connect_de_punta_a_punta(app, fake):
    with TestClient(app) as c:
        c.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]})
        cid = c.post("/api/clients", json={"name": "Minas", "hc_app_key": "clave", "hc_secret": "secreto"}).json()["id"]
        cl = c.get("/api/clients").json()[0]
        assert cl["hc_app_key"] == "clave" and cl["hc_has_secret"] and "hc_secret" not in cl
        # editar sin mandar los campos de Hik-Connect (ej. desde la web) no los borra
        c.put(f"/api/clients/{cid}", json={"name": "Minas"})
        assert c.get("/api/clients").json()[0]["hc_has_secret"]
        # cámaras que ve la API (prueba las regiones hasta la que anda y la recuerda)
        r = c.get(f"/api/clients/{cid}/hikconnect/cameras").json()
        assert [x["id"] for x in r["cameras"]] == ["res-1", "res-2"] and r["cameras"][1]["channel"] == 2
        assert r["server"] == "https://isa.hikcentralconnect.com"
        assert c.get("/api/clients").json()[0]["hc_api_url"] == "https://isa.hikcentralconnect.com"
        ag = c.post("/api/agents", json={"name": "central", "client_id": None}).json()
        h = {"X-Agent-Token": ag["token"]}
        # validaciones: sin serie no; en modo alarmas no; sin IP sí
        base = {"client_id": cid, "agent_id": ag["id"], "name": "Puerta", "brand": "hikconnect", "channel": 2}
        assert c.post("/api/cameras", json=base).status_code == 422
        assert c.post("/api/cameras", json={**base, "hpp_serial": "f12345678", "mode": "alarmas_equipo"}).status_code == 422
        cam = c.post("/api/cameras", json={**base, "hpp_serial": "f12345678"}).json()
        assert cam["brand"] == "hikconnect" and cam["hpp_serial"] == "F12345678" and cam["hc_resource_id"] is None
        largo = c.post("/api/cameras", json={**base, "name": "L", "hpp_serial": "DVR/DVS-34-A1620191217CCWRF12345678WCVU"}).json()
        assert largo["hpp_serial"] == "F12345678"
        c.delete(f"/api/cameras/{largo['id']}")
        conn = c.get("/api/agent/config", headers=h).json()["cameras"][0]["connection"]
        assert conn["brand"] == "hikconnect" and conn["camera_id"] == cam["id"] and conn["serial"] == "F12345678"
        # la PC central pide el link: el servidor busca la cámara por serie y canal y la guarda
        r = c.post(f"/api/agent/cameras/{cam["id"]}/stream-url", json={"kind": "live"}, headers=h)
        assert r.status_code == 200, r.text
        url = r.json()["url"]
        assert url.startswith("https://hls.example/res-2/t1.m3u8")
        assert c.get("/api/cameras").json()[0]["hc_resource_id"] == "res-2"
        sent = [b for u, b in fake.calls if u.endswith("/live/address/get")][-1]
        assert sent["protocol"] == 2 and sent["quality"] == "2" and sent["deviceSerial"] == "F12345678"
        # grabación del disco con hora de inicio y fin
        c.post(f"/api/agent/cameras/{cam['id']}/stream-url", headers=h,
               json={"kind": "playback", "quality": "main", "start": "2026-10-23T10:00:00", "end": "2026-10-23T10:02:00"})
        sent = [b for u, b in fake.calls if u.endswith("/live/address/get")][-1]
        assert sent["type"] == "3" and sent["startTime"] == "2026-10-23 10:00:00" and sent["stopTime"] == "2026-10-23 10:02:00"
        # probar conexión desde el servidor
        p = c.post(f"/api/cameras/{cam['id']}/probe").json()
        assert p["ok"] and p["hikconnect"]
        # otra PC central no puede pedir links de esta cámara
        ag2 = c.post("/api/agents", json={"name": "otra", "client_id": None}).json()
        assert c.post(f"/api/agent/cameras/{cam['id']}/stream-url", json={},
                      headers={"X-Agent-Token": ag2["token"]}).status_code == 403
        # canal que no existe: error claro
        bad = c.post("/api/cameras", json={**base, "name": "X", "hpp_serial": "F12345678", "channel": 9}).json()
        r = c.post(f"/api/agent/cameras/{bad['id']}/stream-url", json={}, headers=h)
        assert r.status_code == 502 and "canal 9" in r.json()["detail"]
        # sacar la clave
        c.put(f"/api/clients/{cid}", json={"name": "Minas", "hc_app_key": ""})
        assert not c.get("/api/clients").json()[0]["hc_has_secret"]
        assert c.get(f"/api/clients/{cid}/hikconnect/cameras").status_code == 422


def test_clave_rechazada_explica_por_region(fake):
    api = hikconnect.HikConnect("clave", "otra-cosa")
    with pytest.raises(hikconnect.HikConnectError) as e:
        api.cameras()
    assert "isa" in str(e.value) and "cuenta inexistente" in str(e.value)
    assert hikconnect.normalize_url("isa.hikcentralconnect.com/#/login") == "https://isa.hikcentralconnect.com"


def test_agente_pide_el_link_al_servidor():
    cam = build_camera({"brand": "hikconnect", "host": "", "camera_id": 7, "serial": "F1", "channel": 2})
    pedidos = []
    agent_hc.set_url_provider(lambda cid, kind, q, s, e: pedidos.append((cid, kind, q)) or f"https://x/{cid}.m3u8?t=1")
    try:
        assert cam.live_url() == "https://x/7.m3u8?t=1" and pedidos == [(7, "live", "sub")]
        assert cam.redacted(cam.live_url("main")) == "https://x/7.m3u8" and pedidos[-1] == (7, "live", "main")
        assert cam.ffmpeg_input_args() == []
    finally:
        agent_hc.set_url_provider(None)
    with pytest.raises(RuntimeError):
        cam.live_url()


def test_serie_larga_y_corta():
    assert hikconnect.short_serial("DVR/DVS-34-A1620191217CCWRE02380030WCVU") == "E02380030"
    assert hikconnect.short_serial("DVR-108G-M1/T0820250823CCWRGE7132986WCVU") == "GE7132986"
    assert hikconnect.short_serial("e02380030") == "E02380030"


def test_cifrado_pide_el_codigo(fake):
    api = hikconnect.HikConnect("clave", "secreto")
    with pytest.raises(hikconnect.HikConnectError) as e:
        api.stream_url("res-enc", "F1")
    assert "cifrado de video" in str(e.value) and "EVZ60019" in str(e.value)
    assert api.stream_url("res-enc", "F1", code="ABCDEF").startswith("https://hls.example/res-enc/")
    assert [b for u, b in fake.calls if u.endswith("/live/address/get")][-1]["code"] == "ABCDEF"


def test_reloj_del_video_en_rafagas():
    import cv2

    from agent.worker import MediaClock

    class Cap:
        def __init__(self, fps, pos):
            self.fps, self.pos = fps, list(pos)

        def get(self, prop):
            if prop == cv2.CAP_PROP_FPS:
                return self.fps
            return self.pos.pop(0) if self.pos else 0.0

    # marcas del video (ms): avanzan aunque los cuadros lleguen todos juntos
    c = MediaClock(Cap(10, [5000, 5100, 5200, 5300]))
    assert [round(c.tick(), 2) for _ in range(4)] == [0.0, 0.1, 0.2, 0.3]
    # sin marcas: cuenta cuadros a los fps declarados
    c = MediaClock(Cap(10, []))
    assert [round(c.tick(), 2) for _ in range(3)] == [0.1, 0.2, 0.3]


def test_minimo_5_fps_y_camara_por_ip(app):
    with TestClient(app) as c:
        c.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]})
        cid = c.post("/api/clients", json={"name": "Local"}).json()["id"]
        base = {"client_id": cid, "name": "Caja", "brand": "ip", "host": "dvr.example", "password": "x"}
        assert c.post("/api/cameras", json={**base, "process_fps": 3}).status_code == 422
        cam = c.post("/api/cameras", json={**base, "process_fps": 5}).json()
        assert cam["brand"] == "ip"
