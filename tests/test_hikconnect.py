"""Cámaras Hikvision por la nube de Hik-Connect (sin puertos): servidor (API simulada) y agente."""

import json
import types

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
        serie_larga = "DVR/DVS-34-A1620191217CCWRF12345678WCVU"
        # la misma cámara con la serie larga es la misma: no se repite
        assert c.post("/api/cameras", json={**base, "name": "L", "hpp_serial": serie_larga}).status_code == 409
        largo = c.post("/api/cameras", json={**base, "name": "L", "channel": 3, "hpp_serial": serie_larga}).json()
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
        # plan B (5 nov): el vivo en FLV o RTMP (4 y 3 en la API); un formato que no existe, 422
        for proto, num in (("flv", 4), ("rtmp", 3)):
            r = c.post(f"/api/agent/cameras/{cam['id']}/stream-url", json={"kind": "live", "protocol": proto}, headers=h)
            assert r.status_code == 200, r.text and r.json()["protocol"] == proto
            assert [b for u, b in fake.calls if u.endswith("/live/address/get")][-1]["protocol"] == num
        assert c.post(f"/api/agent/cameras/{cam['id']}/stream-url", json={"protocol": "otro"}, headers=h).status_code == 422
        # grabación del disco con hora de inicio y fin
        c.post(f"/api/agent/cameras/{cam['id']}/stream-url", headers=h,
               json={"kind": "playback", "quality": "main", "start": "2026-10-23T10:00:00", "end": "2026-10-23T10:02:00"})
        sent = [b for u, b in fake.calls if u.endswith("/live/address/get")][-1]
        # «2» = el disco del grabador («3» es la nube: sin plan de nube da OPEN000502)
        assert sent["type"] == "2" and sent["startTime"] == "2026-10-23 10:00:00" and sent["stopTime"] == "2026-10-23 10:02:00"
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


def test_diagnostico_cuando_no_abre_el_video(monkeypatch):
    import httpx

    from agent import hikconnect

    respuestas = {"limite": (403, '{"code":"EVZ60011","msg":"stream limit"}'), "ok": (200, "#EXTM3U\nseg0.ts\n")}

    from contextlib import contextmanager

    @contextmanager
    def falso_stream(method, url, **_):  # (lee solo el principio: un video en vivo no termina)
        if "caida" in url:
            raise httpx.ConnectTimeout("timeout")
        code, body = respuestas["limite" if "limite" in url else "ok"]
        yield httpx.Response(code, text=body)

    monkeypatch.setattr(hikconnect.httpx, "stream", falso_stream)
    assert "403" in hikconnect.diagnose("https://vtm.example:4443/limite.m3u8?t=1")
    assert "H.264" in hikconnect.diagnose("https://vtm.example:4443/ok.m3u8?t=1")
    assert "vtm.example:4443" in hikconnect.diagnose("https://vtm.example:4443/caida.m3u8?t=1")


@pytest.fixture
def proveedor(monkeypatch):
    """Un servidor falso que da links en el formato pedido y puede rechazar formatos; diagnóstico sin red."""
    agent_hc.forget_formats()
    st = types.SimpleNamespace(pedidos=[], rechaza=set())

    def provider(cid, kind, q, s, e, protocol="hls"):
        st.pedidos.append((kind, protocol))
        if protocol in st.rechaza:
            raise RuntimeError(f"Hik-Connect: error con {protocol}")
        return f"https://x/{cid}.{kind}.{protocol}?t=1"
    monkeypatch.setattr(agent_hc, "diagnose", lambda url, timeout=10.0: "cortó la conexión")
    agent_hc.set_url_provider(provider)
    yield st
    agent_hc.set_url_provider(None)
    agent_hc.forget_formats()


def _cam(cid=7):
    return build_camera({"brand": "hikconnect", "host": "", "camera_id": cid, "serial": "F1", "channel": 2})


def test_si_un_formato_no_abre_la_proxima_conexion_prueba_otro(proveedor):
    """5 nov: el HLS de Hik-Connect cortaba todo y su app andaba: se prueba FLV y se queda con el que anda."""
    cam = _cam()
    url = cam.live_url()
    assert url.endswith(".live.hls?t=1")
    assert cam.open_failure(url).startswith("[HLS no abrió; pruebo FLV]")
    assert cam.live_url().endswith(".flv?t=1")
    cam.opened()  # FLV anduvo
    assert cam.live_url().endswith(".flv?t=1") and cam.protocol == "flv"
    # la cámara rearmada (cambio de configuración, «En vivo», «Nombres e imágenes») arranca por FLV, y cualquier otra
    # también (el corte es del servidor de video, para todas)
    assert _cam().live_url().endswith(".flv?t=1") and _cam(8).live_url().endswith(".flv?t=1")


def test_un_corte_no_saca_a_la_camara_del_formato_que_anda(proveedor):
    cam = _cam()
    cam._proto = 1
    url = cam.live_url()
    cam.opened()  # anda en FLV
    # un corte del grabador: el formato que anduvo se reintenta una vez antes de probar otro…
    assert cam.open_failure(url).startswith("[FLV, reintento]") and cam.protocol == "flv"
    url = cam.live_url()
    assert cam.open_failure(url).startswith("[FLV no abrió; pruebo HLS]")
    url = cam.live_url()
    assert url.endswith(".hls?t=1")
    # …y si el otro tampoco abre, vuelve al que anduvo (cuando el grabador vuelve, casi siempre está en FLV)
    assert cam.open_failure(url).startswith("[HLS no abrió; pruebo FLV]")
    assert cam.live_url().endswith(".flv?t=1")


def test_un_error_del_servidor_no_cambia_el_formato_que_anda(proveedor):
    """Antes, un 502 pasajero con la cámara en FLV la dejaba en HLS (caído). Pero si el error se repite, se prueba el
    otro: si Hik-Connect deja de dar FLV, la cámara no queda caída hasta reiniciar."""
    cam = _cam()
    cam._proto = 1
    cam.live_url()
    cam.opened()
    proveedor.rechaza.add("flv")
    for _ in range(agent_hc.LINK_FAILS_BEFORE_SWITCH - 1):
        with pytest.raises(RuntimeError, match="flv"):
            cam.live_url()
        assert cam.protocol == "flv" and proveedor.pedidos[-1] == ("live", "flv")
    assert cam.live_url().endswith(".live.hls?t=1")  # al tercero seguido, prueba HLS en el mismo pedido
    # un formato que todavía no anduvo y Hik-Connect no da: se saltea en el mismo pedido
    agent_hc.forget_formats()
    otra = _cam(9)
    otra._proto = 1
    proveedor.rechaza = {"flv"}
    assert otra.live_url().endswith(".live.hls?t=1") and proveedor.pedidos[-2:] == [("live", "flv"), ("live", "hls")]
    # todos fallan: queda en el que estaba y se muestra el error de HLS (el de siempre: cifrado, grabador sin conexión…)
    otra._proto = 1
    proveedor.rechaza = {"flv", "hls"}
    with pytest.raises(RuntimeError, match="hls"):
        otra.live_url()
    assert otra.protocol == "flv"


def test_un_servidor_que_no_da_el_formato_hace_pasar_al_siguiente_aunque_haya_andado(proveedor):
    cam = _cam()
    cam._proto = 1
    cam.live_url()
    cam.opened()

    def viejo(cid, kind, q, s, e, protocol="hls"):
        if protocol != "hls":
            raise agent_hc.FormatNotGiven("el servidor todavía no sabe pedir el video en FLV")
        return "https://x/7.hls?t=1"
    agent_hc.set_url_provider(viejo)
    assert cam.live_url() == "https://x/7.hls?t=1" and cam._known_good() is None


def test_la_grabacion_no_mueve_el_formato_del_vivo(proveedor):
    from datetime import datetime

    cam = _cam()
    cam.live_url()
    url = cam.playback_url(datetime(2026, 11, 5, 10), datetime(2026, 11, 5, 10, 30))
    assert ".playback.hls" in url
    assert cam.open_failure(url) == "cortó la conexión" and cam.protocol == "hls"


def test_un_proveedor_viejo_de_cinco_argumentos_sigue_andando():
    agent_hc.forget_formats()
    agent_hc.set_url_provider(lambda cid, kind, q, s, e: "https://x/viejo.m3u8")
    try:
        assert _cam(8).live_url() == "https://x/viejo.m3u8"
    finally:
        agent_hc.set_url_provider(None)


def test_el_servidor_que_esta_en_produccion_y_uno_mas_viejo():
    """El servidor de e5fdc0e pide FLV de verdad pero no lo confirma: se acepta. Uno más viejo ignora el formato y
    manda HLS: no se lo toma por FLV."""
    from agent.outbox import ServerClient

    def responde(req):
        body = json.loads(req.content)
        pide = body.get("protocol", "hls")
        if req.headers.get("x-servidor") == "previo":
            return httpx.Response(200, json={"url": "https://vtm/x.m3u8?t=1"})
        if req.headers.get("x-servidor") == "e5fdc0e":
            return httpx.Response(200, json={"url": f"https://vtm/x.{pide}?t=1"})
        return httpx.Response(200, json={"url": f"https://vtm/x.{pide}?t=1", "protocol": pide})

    def cliente(version):
        http = httpx.Client(base_url="http://t", transport=httpx.MockTransport(responde), headers={"x-servidor": version})
        return ServerClient("http://t", "tok", http=http)
    assert cliente("e5fdc0e").stream_url(7, protocol="flv") == "https://vtm/x.flv?t=1"
    assert cliente("nuevo").stream_url(7, protocol="flv") == "https://vtm/x.flv?t=1"
    with pytest.raises(agent_hc.FormatNotGiven, match="FLV"):
        cliente("previo").stream_url(7, protocol="flv")
    assert cliente("previo").stream_url(7) == "https://vtm/x.m3u8?t=1"


def test_nombres_e_imagenes_prueba_los_formatos(proveedor, monkeypatch):
    import numpy as np

    from agent import nombres

    class Cap:
        def __init__(self, url, *_):
            self.ok = ".flv" in url  # solo abre FLV, como el 5 nov
        def isOpened(self):
            return self.ok
        def read(self):
            return True, np.zeros((4, 4, 3), np.uint8)
        def release(self):
            pass
    monkeypatch.setattr(nombres.cv2, "VideoCapture", Cap)
    frame = nombres.grab_frame({"brand": "hikconnect", "host": "", "camera_id": 5, "serial": "F1", "channel": 1})
    assert frame.shape == (4, 4, 3) and [p for _, p in proveedor.pedidos] == ["hls", "flv"]
    assert _cam(5).protocol == "flv"


def test_el_diagnostico_no_se_queda_bajando_un_flv_en_vivo():
    """Un FLV en vivo no termina nunca: el diagnóstico lee solo el principio."""
    import socket
    import threading

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    stop = threading.Event()

    def flv_sin_fin():
        conn, _ = srv.accept()
        conn.recv(4096)
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: video/x-flv\r\n\r\nFLV\x01\x05")
        try:
            while not stop.is_set():
                conn.sendall(b"\x00" * 1024)
        except OSError:
            pass
        conn.close()
    threading.Thread(target=flv_sin_fin, daemon=True).start()
    try:
        msg = agent_hc.diagnose(f"http://127.0.0.1:{port}/live/X.flv", timeout=3.0)
    finally:
        stop.set()
        srv.close()
    assert "llega (FLV)" in msg
    assert "RTMP" in agent_hc.diagnose("rtmp://x.example/live/X")


def test_el_diagnostico_cuando_el_flv_empieza_y_se_corta():
    """Contesta 200 y la cabecera FLV y después se calla (el grabador sin video): no es «revisá el antivirus»."""
    import socket
    import threading
    import time as _t

    def servidor(cuerpo):
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)

        def atender():
            conn, _ = srv.accept()
            conn.recv(4096)
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: video/x-flv\r\n\r\n")
            cuerpo(conn)
            _t.sleep(3)
            conn.close()
        threading.Thread(target=atender, daemon=True).start()
        return srv

    srv = servidor(lambda c: c.sendall(b"FLV\x01\x05"))
    t0 = _t.monotonic()
    msg = agent_hc.diagnose(f"http://127.0.0.1:{srv.getsockname()[1]}/x.flv", timeout=0.5)
    srv.close()
    assert "empezó a mandar el video (FLV)" in msg and "antivirus" not in msg and _t.monotonic() - t0 < 3

    def goteo(c):  # un byte cada 0,2 s: no tiene que trabar el hilo de la cámara
        try:
            for _ in range(12):
                c.sendall(b"x")
                _t.sleep(0.2)
        except OSError:
            pass  # (el diagnóstico ya cortó: es lo que se quiere)
    srv = servidor(goteo)
    t0 = _t.monotonic()
    agent_hc.diagnose(f"http://127.0.0.1:{srv.getsockname()[1]}/x.flv", timeout=0.5)
    srv.close()
    assert _t.monotonic() - t0 < 2.0


def test_el_diagnostico_cuando_el_servidor_acepta_y_no_responde_o_cierra_enseguida():
    import socket
    import threading
    import time as _t

    def servidor(atender):
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        threading.Thread(target=lambda: atender(srv.accept()[0]), daemon=True).start()
        return srv

    def mudo(conn):
        conn.recv(4096)
        _t.sleep(3)
        conn.close()
    srv = servidor(mudo)
    msg = agent_hc.diagnose(f"http://127.0.0.1:{srv.getsockname()[1]}/x.flv", timeout=0.5)
    srv.close()
    assert "aceptó la conexión pero no respondió" in msg and "antivirus" not in msg

    def cierra(conn):
        conn.recv(4096)
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: video/x-flv\r\nContent-Length: 13\r\n\r\nFLV\x01\x05\x00\x00\x00\x09\x00\x00\x00\x00")
        conn.close()
    srv = servidor(cierra)
    msg = agent_hc.diagnose(f"http://127.0.0.1:{srv.getsockname()[1]}/x.flv", timeout=0.5)
    srv.close()
    assert "lo cortó" in msg and "H.264" not in msg

    def nada(conn):  # 200 y después nada (espera al grabador apagado)
        conn.recv(4096)
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: video/x-flv\r\n\r\n")
        _t.sleep(3)
        conn.close()
    srv = servidor(nada)
    msg = agent_hc.diagnose(f"http://127.0.0.1:{srv.getsockname()[1]}/x.flv", timeout=0.5)
    srv.close()
    assert "no mandó el video" in msg and "ReadTimeout" not in msg

    def cabeceras_goteando(conn):  # cabeceras de a un byte: el tope total no las deja trabar el hilo
        conn.recv(4096)
        try:
            for ch in b"HTTP/1.1 200 OK\r\nX-Algo: " + b"a" * 200:
                conn.sendall(bytes([ch]))
                _t.sleep(0.2)
        except OSError:
            pass
        conn.close()
    srv = servidor(cabeceras_goteando)
    t0 = _t.monotonic()
    msg = agent_hc.diagnose(f"http://127.0.0.1:{srv.getsockname()[1]}/x.flv", timeout=0.5)
    srv.close()
    assert _t.monotonic() - t0 < 4 and "tardó más de" in msg
