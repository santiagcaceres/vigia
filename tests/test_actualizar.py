"""Actualización de la app de escritorio: versiones, descarga desde el servidor y reemplazo del .exe."""

import io
import zipfile

import httpx
import pytest
from fastapi.testclient import TestClient

from agent import actualizar
from server.main import Settings, create_app

ADMIN = ("admin", "clave-admin-123")


def test_comparar_versiones():
    assert actualizar.is_newer("1.8.0", "1.7.0")
    assert actualizar.is_newer("1.10.0", "1.9.3")
    assert not actualizar.is_newer("1.7.0", "1.7.0")
    assert not actualizar.is_newer("1.6.9", "1.7.0")
    assert actualizar.is_newer("1.7.0", "dev")  # una copia de desarrollo siempre puede actualizar
    assert actualizar.is_newer("1.7.0", "dev-12")
    assert not actualizar.is_newer(None, "1.0.0") and not actualizar.is_newer("", "1.0.0")


def _zip_with_exe(size: int = 1_200_000) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("VigiaAgente.exe", b"MZ" + b"\0" * size)
        z.writestr("LEEME.txt", "hola")
    return buf.getvalue()


def test_sacar_y_reemplazar_exe(tmp_path):
    zp = tmp_path / "a.zip"
    zp.write_bytes(_zip_with_exe())
    exe = tmp_path / "VigiaAgente.exe"
    exe.write_bytes(b"viejo")
    new = actualizar.extract_exe(zp, tmp_path / "VigiaAgente.nuevo.exe")
    actualizar.swap_exe(new, exe)
    assert exe.read_bytes().startswith(b"MZ") and (tmp_path / "VigiaAgente.viejo.exe").read_bytes() == b"viejo"
    # un .zip sin el programa, o incompleto, no toca nada
    bad = tmp_path / "b.zip"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("otra.txt", "x")
    with pytest.raises(ValueError):
        actualizar.extract_exe(bad, tmp_path / "x.exe")
    small = tmp_path / "c.zip"
    small.write_bytes(_zip_with_exe(10))
    with pytest.raises(ValueError):
        actualizar.extract_exe(small, tmp_path / "y.exe")


@pytest.fixture
def app(tmp_path, db_url):
    return create_app(Settings(data_dir=tmp_path / "data", database_url=db_url, admin_password=ADMIN[1],
                               serve_public=False, github_token="tok-de-prueba"))


def test_ultima_version_y_link_de_descarga(app, monkeypatch):
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        if req.url.path.endswith("/releases/latest"):
            return httpx.Response(200, json={"tag_name": "agente-v1.8.0", "published_at": "2026-10-23T10:00:00Z",
                                             "body": "notas", "assets": [{"id": 77, "name": "VigiaAgente-windows.zip",
                                                                          "size": 90_000_000}]})
        if req.url.path.endswith("/releases/assets/77"):
            return httpx.Response(302, headers={"location": "https://objects.example/firmado.zip"})
        return httpx.Response(404)

    real = httpx.Client
    monkeypatch.setattr("server.main.httpx.Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    with TestClient(app) as c:
        assert c.get("/api/app/update").status_code == 401
        c.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]})
        info = c.get("/api/app/update").json()
        assert info["version"] == "1.8.0" and info["url"] is None and info["size"] == 90_000_000
        info = c.get("/api/app/update", params={"download": 1}).json()
        assert info["url"] == "https://objects.example/firmado.zip"
        assert seen[0].headers["authorization"] == "Bearer tok-de-prueba"
        r = c.get("/api/app/download", follow_redirects=False)
        assert r.status_code == 302 and r.headers["location"] == "https://objects.example/firmado.zip"
        # un cliente no puede pedirla
        cid = c.post("/api/clients", json={"name": "X", "web_username": "dueno-x", "web_password": "clave-larga-1"}).json()["id"]
        u = TestClient(app)
        u.post("/api/login", json={"username": "dueno-x", "password": "clave-larga-1"})
        assert cid and u.get("/api/app/update").status_code == 403
