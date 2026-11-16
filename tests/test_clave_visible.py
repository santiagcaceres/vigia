"""«Editar cámara» muestra la clave del grabador y el link RTSP guardados (pedido del dueño, 16 nov: «dejá todo visible,
contraseña, links»). Los da el servidor solo al equipo, deja registrado quién los vio, y guardar sin tocarlos no cambia
nada (la cámara que usa un link tal cual lo sigue usando)."""

import pytest
from fastapi.testclient import TestClient

from agent.oficina import vlc_link
from server.db import CredentialAccess
from server.main import Settings, create_app

ADMIN = ("admin", "clave-admin-123")


def test_el_link_para_vlc_se_arma_como_en_la_maquina():
    assert vlc_link("hikvision", "local.ddns.net", "5543", "admin", "Clave1234", "15") == (
        "rtsp://admin:Clave1234@local.ddns.net:5543/Streaming/Channels/1502")
    assert vlc_link("hikvision", "h", 554, "ac", "a@b:c", 2, "main") == "rtsp://ac:a%40b%3Ac@h:554/Streaming/Channels/201"
    assert vlc_link("dahua", "h", 554, "u", "p", 3) == "rtsp://u:p@h:554/cam/realmonitor?channel=3&subtype=1"
    assert vlc_link("hikvision", "", 554, "u", "p", 1) == ""


@pytest.fixture
def app(tmp_path, db_url):
    return create_app(Settings(data_dir=tmp_path / "data", database_url=db_url, admin_password=ADMIN[1],
                               serve_public=False))


def test_el_equipo_ve_la_clave_y_el_link_y_queda_registrado(app):
    a = TestClient(app)
    a.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]})
    cl = a.post("/api/clients", json={"name": "Mayorista", "web_username": "mayorista",
                                      "web_password": "clave-mayorista-1"}).json()
    base = {"client_id": cl["id"], "name": "canal 15", "brand": "hikvision", "host": "local.ddns.net",
            "rtsp_port": 5543, "channel": 15, "username": "admin", "password": "Clave1234"}
    cam = a.post("/api/cameras", json=base)
    assert cam.status_code == 201, cam.text
    cam = cam.json()
    assert "Clave1234" not in str(a.get("/api/cameras").json())  # la lista sigue sin la clave
    r = a.get(f"/api/cameras/{cam['id']}/secret")
    assert r.status_code == 200, r.text
    assert r.json() == {"password": "Clave1234", "source_url": None}
    linked = a.post("/api/cameras", json={**base, "name": "por link", "channel": 1,
                                          "source_url": "rtsp://u:p@otro.host:554/live/ch1"}).json()
    assert a.get(f"/api/cameras/{linked['id']}/secret").json()["source_url"] == "rtsp://u:p@otro.host:554/live/ch1"
    # el dueño del comercio no
    u = TestClient(app)
    u.post("/api/login", json={"username": "mayorista", "password": "clave-mayorista-1"})
    assert u.get(f"/api/cameras/{cam['id']}/secret").status_code == 403
    # quién la vio, qué cámara y para qué
    with app.state.Session() as s:
        rows = s.query(CredentialAccess).filter_by(what="edit").all()
        assert sorted(x.camera_id for x in rows) == sorted([cam["id"], linked["id"]])
        assert all(x.user_id for x in rows)
    # guardar sin tocar el link (keep_source_url, como manda la app) lo deja como estaba
    r = a.put(f"/api/cameras/{linked['id']}", json={**base, "name": "por link", "channel": 1, "password": None,
                                                    "keep_source_url": True})
    assert r.status_code == 200, r.text
    assert a.get(f"/api/cameras/{linked['id']}/secret").json() == {
        "password": "Clave1234", "source_url": "rtsp://u:p@otro.host:554/live/ch1"}


# ---------- el formulario de la app (corre con xvfb-run)
def _dialog(off, root, cam):
    before = set(root.winfo_children())
    off.camera_dialog(cam)
    root.update()
    return next(w for w in root.winfo_children() if w not in before)


def _entries(top):
    from tests.test_sensibilidad_app import _walk
    return [w for w in _walk(top) if w.winfo_class() == "TEntry"]


def test_el_formulario_muestra_la_clave_y_el_link_y_guardar_sin_tocar_no_cambia_nada():
    from tests.test_sensibilidad_app import Msgs, _cam, _find, _oficina, _tk
    tk, root = _tk()
    try:
        cam = _cam(host="local.ddns.net", rtsp_port=5543, channel=15, username="admin")
        off = _oficina(tk, root, [cam], Msgs())
        secrets = {9: {"password": "Clave1234", "source_url": None}}
        off.api.get = lambda path: secrets[int(path.split("/")[3])]
        top = _dialog(off, root, cam)
        vals = [e.get() for e in _entries(top)]
        assert "Clave1234" in vals  # la clave a la vista
        pwd = next(e for e in _entries(top) if e.get() == "Clave1234")
        assert str(pwd.cget("show")) == ""  # sin puntitos
        assert "rtsp://admin:Clave1234@local.ddns.net:5543/Streaming/Channels/1502" in vals  # el de VLC
        _find(top, "TButton", "Guardar").invoke()
        body = off.sent[-1][2]
        assert body["password"] is None  # sin tocar = sin cambios
        # cambiar la clave: va la nueva
        top = _dialog(off, root, cam)
        pwd = next(e for e in _entries(top) if e.get() == "Clave1234")
        pwd.delete(0, "end")
        pwd.insert(0, "OtraClave99")
        root.update()
        assert any("OtraClave99@local.ddns.net" in e.get() for e in _entries(top))  # el de VLC la sigue
        _find(top, "TButton", "Guardar").invoke()
        assert off.sent[-1][2]["password"] == "OtraClave99"
        # una cámara que usa un link tal cual: se muestra y, sin tocarlo, lo sigue usando
        link = "rtsp://u:p@otro.host:554/live/ch1"
        cam2 = _cam(id=11, has_source_url=True, host="otro.host", username="u")
        secrets[11] = {"password": "p", "source_url": link}
        top = _dialog(off, root, cam2)
        assert _find(top, "Text").get("1.0", "end").strip() == link
        _find(top, "TButton", "Guardar").invoke()
        body = off.sent[-1][2]
        assert body["keep_source_url"] is True and body["source_url"] is None and body["password"] is None
    finally:
        root.destroy()
