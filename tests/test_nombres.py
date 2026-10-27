"""«Nombres e imágenes»: renombrar las cámaras viendo su imagen, con lo que ve la IA (YOLO) como pista."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from core.detectors import Detection
from core.escena import describe, suggest, summary
from server.main import Settings, create_app

ADMIN = ("admin", "clave-admin-123")


@pytest.fixture
def app(tmp_path, db_url):
    return create_app(Settings(data_dir=tmp_path / "data", database_url=db_url, admin_password=ADMIN[1],
                               serve_public=False))


def test_renombrar_varias_camaras_y_cambiar_el_tipo(app):
    with TestClient(app) as a:
        a.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]})
        cl = a.post("/api/clients", json={"name": "Minas"}).json()
        hc = {"client_id": cl["id"], "brand": "hikconnect", "hpp_serial": "E02380030"}
        ids = [a.post("/api/cameras", json={**hc, "name": f"camera {i}@E02380030", "channel": i}).json()["id"]
               for i in (1, 2, 3)]
        pc = a.post("/api/agent/session").json()
        rev = a.get("/api/agent/config", headers={"X-Agent-Token": pc["token"]}).json()["rev"]
        res = a.post("/api/cameras/names", json={"cameras": [
            {"id": ids[0], "name": "  Caja 1 ", "profile": "caja"},
            {"id": ids[1], "name": "Heladeras"},
            {"id": ids[2], "name": "camera 3@E02380030"},  # sin cambios
            {"id": 9999, "name": "No existe"}]}).json()
        assert res == {"renamed": 2, "retyped": 1}
        cams = {c["id"]: c for c in a.get("/api/cameras").json()}
        assert cams[ids[0]]["name"] == "Caja 1" and cams[ids[0]]["profile"] == "caja"
        assert cams[ids[1]]["name"] == "Heladeras" and cams[ids[1]]["profile"] == "general"
        assert a.get("/api/agent/config", headers={"X-Agent-Token": pc["token"]}).json()["rev"] != rev
        assert a.post("/api/cameras/names", json={"cameras": [{"id": ids[0], "name": ""}]}).status_code == 422

        # un cliente no puede renombrar
        a.post("/api/users", json={"username": "minas", "password": "clave-minas-123", "role": "cliente",
                                   "client_id": cl["id"]})
        with TestClient(app) as b:
            b.post("/api/login", json={"username": "minas", "password": "clave-minas-123"})
            assert b.post("/api/cameras/names", json={"cameras": [{"id": ids[0], "name": "x"}]}).status_code == 403


def test_sugerencias_de_la_ia_por_lo_que_se_ve():
    assert suggest({"auto": 2, "persona": 1})["profile"] == "exterior"
    assert suggest({"moto": 1}, plans=("estacion",))["name"] == "Surtidores"
    assert suggest({"banana": 1, "naranja": 3})["name"] == "Verdulería"
    assert suggest({"heladera": 1, "botella": 9})["name"] == "Heladeras"
    assert suggest({"botella": 4})["name"] == "Bebidas"
    assert suggest({"pantalla": 1, "teclado": 1})["profile"] == "caja"
    assert suggest({"silla": 2}) is None and suggest({}) is None


class FakeDetector:
    def detect(self, frame):
        return [Detection((10, 10, 40, 90), "persona", 0.9), Detection((100, 20, 160, 120), "heladera", 0.8),
                Detection((5, 5, 8, 8), "botella", 0.40)]  # botella con poca confianza: no se cuenta


def test_describir_la_imagen_de_una_camara():
    noche = np.full((360, 640, 3), 30, np.uint8)  # oscura y sin color (infrarrojo)
    info = describe(noche, FakeDetector())
    assert info["persons"] == 1 and info["objects"] == {"heladera": 1}
    assert info["night"] and info["dark"] and info["size"] == (640, 360)
    assert info["suggestion"]["name"] == "Heladeras"
    assert summary(info) == "1 persona · heladera · de noche (infrarrojo) · 640×360"
