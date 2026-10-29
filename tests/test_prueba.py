"""Pestaña Prueba (eventos creados desde el navegador) y arranque rápido del esquema."""

import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from server.db import SCHEMA_VERSION, ensure_schema
from server.main import Settings, create_app

ADMIN = ("admin", "clave-admin-123")


def test_el_esquema_se_crea_una_vez_y_despues_no_se_toca(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path / 'v.db'}")
    assert ensure_schema(eng) is True
    assert ensure_schema(eng) is False
    with eng.begin() as c:
        c.execute(text("UPDATE schema_meta SET value = 'vieja'"))
    assert ensure_schema(eng) is True
    with eng.connect() as c:
        assert c.execute(text("SELECT value FROM schema_meta")).scalar() == SCHEMA_VERSION


@pytest.fixture
def app(tmp_path, db_url):
    return create_app(Settings(data_dir=tmp_path / "data", database_url=db_url, admin_password=ADMIN[1],
                               serve_public=False))


def test_evento_manual_con_clip(app):
    with TestClient(app) as c:
        c.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]})
        cid = c.post("/api/clients", json={"name": "Almacén"}).json()["id"]
        cam = c.post("/api/cameras", json={"client_id": cid, "name": "Góndola", "host": "h", "mode": "alarmas_equipo"}).json()
        meta = {"camera_id": cam["id"], "type": "zona_restringida", "description": "Persona 20+ s en gondola", "zone": "gondola",
                "occurred_at": "2026-10-20T10:00:00-03:00", "features": {"shelf_total_s": 22.5, "people_in_scene": 1, "x": 9},
                "ext": "webm"}
        r = c.post("/api/events/manual", data={"meta": json.dumps(meta)},
                   files={"snapshot": ("s.jpg", b"\xff\xd8fake", "image/jpeg")})
        assert r.status_code == 201, r.text
        ev = r.json()
        assert ev["source"] == "prueba" and ev["type"] == "zona_restringida" and ev["clip_url"] is None
        assert ev["features"] == {"shelf_total_s": 22.5, "people_in_scene": 1.0}
        assert ev["upload_url"] and ev["content_type"] == "video/webm"
        # sin subir el clip no se acepta
        assert c.post(f"/api/events/{ev['id']}/clip-done", json={"ext": "webm"}).status_code == 409
        assert c.put(ev["upload_url"], content=b"0" * 2048, headers={"Content-Type": "video/webm"}).status_code == 200
        done = c.post(f"/api/events/{ev['id']}/clip-done", json={"ext": "webm"}).json()
        assert done["clip_source"] == "prueba" and done["clip_url"].endswith("/clip")
        assert c.get(done["clip_url"]).content == b"0" * 2048
        # aparece en la lista filtrando por origen y en las estadísticas
        assert c.get("/api/events", params={"source": "prueba"}).json()["total"] == 1
        assert c.get("/api/stats").json()["total"] == 1
        # tipo desconocido cae en alarma_dvr; cámara inexistente da 404
        r = c.post("/api/events/manual", data={"meta": json.dumps({**meta, "type": "lo_que_sea"})},
                   files={"snapshot": ("s.jpg", b"\xff\xd8", "image/jpeg")})
        assert r.json()["type"] == "alarma_dvr"
        r = c.post("/api/events/manual", data={"meta": json.dumps({**meta, "camera_id": 999})},
                   files={"snapshot": ("s.jpg", b"\xff\xd8", "image/jpeg")})
        assert r.status_code == 404
