"""Vista en vivo: latido del panel, lista de cámaras en vivo para la PC central y subida de cuadros."""

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from agent.live import LiveStreamer, hud
from core.rules import parse_zones
from core.tracker import Track
from server.main import Settings, create_app

ADMIN = ("admin", "clave-admin-123")


@pytest.fixture
def app(tmp_path, db_url):
    return create_app(Settings(data_dir=tmp_path / "data", database_url=db_url, admin_password=ADMIN[1],
                               serve_public=False, live_seconds=45))


def test_latido_y_lista_en_vivo(app):
    with TestClient(app) as c:
        c.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]})
        cid = c.post("/api/clients", json={"name": "Almacén"}).json()["id"]
        ag = c.post("/api/agents", json={"name": "central", "client_id": None}).json()
        h = {"X-Agent-Token": ag["token"]}
        cam = c.post("/api/cameras", json={"client_id": cid, "agent_id": ag["id"], "name": "Góndola", "host": "h",
                                           "password": "x", "mode": "alarmas_equipo"}).json()
        otra = c.post("/api/cameras", json={"client_id": cid, "agent_id": ag["id"], "name": "Caja", "host": "h",
                                            "password": "x", "mode": "vivo"}).json()
        # sin latidos, la PC central no tiene nada en vivo
        r = c.post("/api/agent/status", json={"cameras": []}, headers=h).json()
        assert r["live"] == []
        # el panel late: todavía no hay cuadro ni agente transmitiendo
        r = c.post(f"/api/cameras/{cam['id']}/live").json()
        assert r["active"] is False and r["agent_online"] is True and r["frame_url"] is None
        assert c.post("/api/agent/status", json={"cameras": []}, headers=h).json()["live"] == [cam["id"]]
        # la PC central pide la URL de subida y sube un cuadro
        up = c.post(f"/api/agent/cameras/{cam['id']}/live-url", headers=h).json()
        assert up["content_type"] == "image/jpeg" and up["upload_url"]
        assert c.put(up["upload_url"], content=b"\xff\xd8cuadro", headers={"Content-Type": "image/jpeg"}).status_code == 200
        # e informa que está transmitiendo: el panel ve "activo" y recibe la URL del cuadro
        c.post("/api/agent/status", json={"cameras": [], "live": [cam["id"]]}, headers=h)
        r = c.post(f"/api/cameras/{cam['id']}/live").json()
        assert r["active"] is True and r["frame_url"]
        assert c.get(r["frame_url"]).content == b"\xff\xd8cuadro"
        # la ficha de la cámara también lo dice, aunque esté en modo alarmas (no figura en cameras[])
        assert c.get(f"/api/cameras/{cam['id']}").json()["live"] is True
        assert c.get(f"/api/cameras/{otra['id']}").json()["live"] is False
        # el rol cliente puede mirar lo suyo, pero no otra cámara
        c.post("/api/users", json={"username": "dueno", "password": "clave-dueno-1", "role": "cliente", "client_id": cid})
        u = TestClient(app)
        u.post("/api/login", json={"username": "dueno", "password": "clave-dueno-1"})
        assert u.post(f"/api/cameras/{cam['id']}/live").status_code == 200
        c2 = c.post("/api/clients", json={"name": "Otro"}).json()["id"]
        ajena = c.post("/api/cameras", json={"client_id": c2, "name": "X", "host": "h", "password": "x"}).json()
        assert u.post(f"/api/cameras/{ajena['id']}/live").status_code == 403
        # otro agente no puede pedir la URL de subida de esta cámara
        ag2 = c.post("/api/agents", json={"name": "otra pc", "client_id": None}).json()
        assert c.post(f"/api/agent/cameras/{cam['id']}/live-url", headers={"X-Agent-Token": ag2["token"]}).status_code == 404


class FakeClient:
    def __init__(self):
        self.urls = 0
        self.uploads: list[bytes] = []
        self.fail_first = False

    def live_upload_url(self, camera_id):
        self.urls += 1
        return f"/up/{camera_id}"

    def upload_bytes(self, url, data, content_type):
        if self.fail_first:
            self.fail_first = False
            raise RuntimeError("url vencida")
        self.uploads.append(data)


def test_streamer_sube_un_cuadro_por_segundo_y_renueva_la_url():
    zones = parse_zones([{"name": "gondola", "type": "shelf", "polygon": [[0, 0.5], [1, 0.5], [1, 1], [0, 1]]}])
    cl = FakeClient()
    st = LiveStreamer(7, "Almacén · Góndola", cl, zones)
    frame = np.zeros((360, 640, 3), np.uint8)
    tracks = [Track(1, (100, 100, 200, 340), "person", 0.0, 5.0), Track(2, (300, 50, 360, 200), "person", 3.0, 5.0, missed=2)]
    assert st.push(frame, tracks, 5.0) is False  # nadie mira: no se sube nada
    st.extend(45)
    assert st.push(frame, tracks, 5.0) is True
    assert st.push(frame, tracks, 5.2) is False  # máximo un cuadro por segundo
    for _ in range(50):
        if cl.uploads:
            break
        time.sleep(0.05)
    assert len(cl.uploads) == 1 and cl.uploads[0][:2] == b"\xff\xd8" and cl.urls == 1
    # si la URL venció, pide otra y reintenta una vez
    cl.fail_first = True
    st._last = -1e9
    assert st.push(frame, tracks, 7.0) is True
    for _ in range(50):
        if len(cl.uploads) == 2:
            break
        time.sleep(0.05)
    assert len(cl.uploads) == 2 and cl.urls == 2 and st.errors == 0
    # el cuadro con recuadros tiene el mismo tamaño que el original
    assert hud(frame, zones, tracks, 5.0, "Góndola").shape == frame.shape
