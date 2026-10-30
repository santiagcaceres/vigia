"""Clientes con el grabador por IP y puertos (pedido del dueño, 30 oct): en «Editar cliente» se elige Hik-Connect o IP
y puertos (o se pega el link RTSP que pasa el técnico); al agregar cámaras alcanza con los canales y la clave queda en
el cliente, cifrada. Si cambian el puerto o la clave, sus cámaras pasan solas a los datos nuevos."""

import pytest
from fastapi.testclient import TestClient

from agent.cameras import parse_rtsp
from agent.oficina import parse_channels
from server.main import Settings, create_app

ADMIN = ("admin", "clave-admin-123")
LINK = "rtsp://admin:Cl%40ve-1@local.ddns.net:554/Streaming/Channels/401"


def test_lee_el_link_rtsp_que_pasa_el_tecnico():
    d = parse_rtsp(LINK)
    assert (d["host"], d["rtsp_port"], d["username"], d["password"], d["channel"], d["brand"], d["stream"]) == (
        "local.ddns.net", 554, "admin", "Cl@ve-1", 4, "hikvision", "main")
    assert parse_rtsp("rtsp://u:p@1.2.3.4:38554/Streaming/Channels/1202")["channel"] == 12
    dahua = parse_rtsp("rtsp://u:p@h:554/cam/realmonitor?channel=3&subtype=1")
    assert (dahua["channel"], dahua["brand"], dahua["stream"]) == (3, "dahua", "sub")
    assert parse_rtsp("rtsp://u:p@h/h264/ch2/main/av_stream")["channel"] == 2
    for bad in ("http://h/x", "local.ddns.net", "", "rtsp://h:puerto/x"):
        with pytest.raises(ValueError):
            parse_rtsp(bad)


def test_canales_para_cargar_varios_de_una():
    assert parse_channels("4") == [4]
    assert parse_channels("1-4, 6; 8 a 9") == [1, 2, 3, 4, 6, 8, 9]
    assert parse_channels("2, 2, 1-2") == [2, 1]
    for bad in ("", "cuatro", "0", "5-3", "1-300"):
        with pytest.raises(ValueError):
            parse_channels(bad)


@pytest.fixture
def app(tmp_path, db_url):
    return create_app(Settings(data_dir=tmp_path / "data", database_url=db_url, admin_password=ADMIN[1],
                               serve_public=False))


def test_el_cliente_guarda_su_grabador_y_las_camaras_usan_su_clave(app):
    a = TestClient(app)
    a.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]})
    d = parse_rtsp(LINK)
    cl = a.post("/api/clients", json={"name": "Cristino", "cam_access": "ip", "ip_host": d["host"],
                                      "ip_rtsp_port": d["rtsp_port"], "ip_username": d["username"],
                                      "ip_password": d["password"], "web_username": "cristino",
                                      "web_password": "clave-cristino-1"})
    assert cl.status_code == 201, cl.text
    cl = cl.json()
    assert (cl["cam_access"], cl["ip_host"], cl["ip_rtsp_port"], cl["ip_http_port"], cl["ip_username"],
            cl["ip_has_password"]) == ("ip", "local.ddns.net", 554, None, "admin", True)
    assert "Cl@ve-1" not in str(a.get("/api/clients").json())  # la clave no sale del servidor
    # el dueño del comercio no ve cómo se llega a su grabador
    u = TestClient(app)
    u.post("/api/login", json={"username": "cristino", "password": "clave-cristino-1"})
    assert "ip_host" not in u.get("/api/clients").json()[0]

    # «Agregar cámaras» con los canales 4 y 5, sin escribir la clave: usan la del cliente
    ids = []
    for ch in (4, 5):
        r = a.post("/api/cameras", json={"client_id": cl["id"], "name": f"Canal {ch}", "brand": "ip",
                                         "host": cl["ip_host"], "rtsp_port": 554, "username": "admin",
                                         "channel": ch, "client_password": True})
        assert r.status_code == 201 and r.json()["has_password"], r.text
        ids.append(r.json()["id"])
    otro = a.post("/api/cameras", json={"client_id": cl["id"], "name": "Otro grabador", "brand": "ip",
                                        "host": "otro.ddns.net", "password": "otra", "channel": 1}).json()
    pc = a.post("/api/agent/session").json()
    h = {"X-Agent-Token": pc["token"]}

    def conn(cid):
        return a.post("/api/agent/view", headers=h, json={"camera_ids": [cid]}).json()["cameras"][0]["connection"]
    assert (conn(ids[0])["password"], conn(ids[0])["channel"], conn(ids[1])["channel"]) == ("Cl@ve-1", 4, 5)

    # el técnico pasó a un usuario «vigia» y a un puerto alto: las cámaras de ese grabador cambian solas
    body = {"name": "Cristino", "cam_access": "ip", "ip_host": "local.ddns.net", "ip_rtsp_port": 38554,
            "ip_http_port": 38080, "ip_username": "vigia", "ip_password": "nueva-clave-9"}
    r = a.put(f"/api/clients/{cl['id']}", json=body).json()
    assert r["cameras_updated"] == 2 and r["ip_rtsp_port"] == 38554 and r["ip_http_port"] == 38080
    c4 = conn(ids[0])
    assert (c4["rtsp_port"], c4["http_port"], c4["username"], c4["password"], c4["channel"]) == (
        38554, 38080, "vigia", "nueva-clave-9", 4)
    o = conn(otro["id"])
    assert (o["host"], o["rtsp_port"], o["password"]) == ("otro.ddns.net", 554, "otra")  # otro grabador: no se toca
    # guardar sin tocar los datos del grabador (y sin clave): no cambia nada
    r = a.put(f"/api/clients/{cl['id']}", json={**body, "ip_password": None}).json()
    assert r["cameras_updated"] == 0 and conn(ids[0])["password"] == "nueva-clave-9"
    # una app vieja (sin estos campos) no borra nada
    r = a.put(f"/api/clients/{cl['id']}", json={"name": "Cristino"}).json()
    assert r["ip_host"] == "local.ddns.net" and r["ip_has_password"] and r["cam_access"] == "ip"


def test_hik_connect_sigue_igual(app):
    a = TestClient(app)
    a.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]})
    cl = a.post("/api/clients", json={"name": "Minas", "hc_app_key": "k", "hc_secret": "s"}).json()
    assert cl["cam_access"] == "hikconnect" and cl["ip_host"] is None  # (sin elegir: se deduce de la AppKey)


def test_camara_por_link_como_vlc(app):
    """Un link por cámara, usado tal cual (como VLC). Al editar sin volver a pegarlo, queda el que estaba."""
    a = TestClient(app)
    a.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]})
    cl = a.post("/api/clients", json={"name": "Cristino"}).json()
    d = parse_rtsp(LINK)
    body = {"client_id": cl["id"], "name": "Canal 4", "brand": d["brand"], "host": d["host"],
            "rtsp_port": d["rtsp_port"], "username": d["username"], "password": d["password"], "channel": d["channel"],
            "source_url": LINK, "detect_stream": "main"}
    cam = a.post("/api/cameras", json=body).json()
    assert cam["has_source_url"] and LINK not in str(a.get("/api/cameras").json())  # el link (con la clave) no vuelve
    pc = a.post("/api/agent/session").json()

    def conn():
        return a.post("/api/agent/view", headers={"X-Agent-Token": pc["token"]},
                      json={"camera_ids": [cam["id"]]}).json()["cameras"][0]["connection"]
    assert conn()["source_url"] == LINK
    from agent.cameras import build_camera
    assert build_camera(conn()).live_url() == LINK  # se abre exactamente ese link, como en VLC
    edit = {**body, "name": "Caja", "source_url": None, "password": None, "keep_source_url": True}
    assert a.put(f"/api/cameras/{cam['id']}", json=edit).json()["has_source_url"]
    assert conn()["source_url"] == LINK and conn()["password"] == "Cl@ve-1"
    # pasarla a «por IP y puertos» (sin link): se borra
    assert not a.put(f"/api/cameras/{cam['id']}", json={**edit, "keep_source_url": False}).json()["has_source_url"]


def test_clave_de_hik_connect_desde_la_camara(app):
    """La clave de Hik-Connect se carga sola (desde el formulario de la cámara) sin tocar el resto del cliente."""
    a = TestClient(app)
    a.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]})
    cl = a.post("/api/clients", json={"name": "Minas", "timezone": "America/Montevideo", "plans": ["comercio"],
                                      "hours": {"lun": "08:00-19:00"}}).json()
    assert a.put(f"/api/clients/{cl['id']}/hikconnect", json={"hc_app_key": "k"}).status_code == 422  # falta el secreto
    r = a.put(f"/api/clients/{cl['id']}/hikconnect", json={"hc_app_key": " k ", "hc_secret": "s"}).json()
    assert r["hc_app_key"] == "k" and r["hc_has_secret"] and "hc_secret" not in r  # el SecretKey no vuelve
    assert (r["timezone"], r["plans"], r["hours"]["lun"]) == ("America/Montevideo", ["comercio"], "08:00-19:00")
    r = a.put(f"/api/clients/{cl['id']}/hikconnect", json={"hc_app_key": "k2"}).json()  # sin secreto: queda el que estaba
    assert r["hc_app_key"] == "k2" and r["hc_has_secret"]
