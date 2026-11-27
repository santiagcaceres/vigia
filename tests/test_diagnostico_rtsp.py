"""Por qué no abre el video de un grabador por IP (30 oct: con un cliente nuevo la app solo decía «no se pudo abrir el
video»). Un grabador de mentira que contesta RTSP como Hikvision: pide clave (Digest), tiene el canal 4 en calidad
principal y no el subflujo."""

import hashlib
import re
import socket
import threading

import pytest

from agent.cameras import rtsp_diagnose

USER, PASSWORD = "admin", "Cl@ve 1"
REALM, NONCE = "IP Camera(C1234)", "4f6d2a"


def _ok(auth: str, method: str) -> bool:
    p = dict(re.findall(r'(\w+)="?([^",]*)"?', auth[6:]))
    md5 = lambda t: hashlib.md5(t.encode()).hexdigest()  # noqa: E731
    want = md5(f"{md5(f'{USER}:{REALM}:{PASSWORD}')}:{NONCE}:{md5(method + ':' + p.get('uri', ''))}")
    return auth.startswith("Digest") and p.get("username") == USER and p.get("response") == want


@pytest.fixture
def grabador():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    paths = {"/Streaming/Channels/401": 200}  # el canal 4, solo en calidad principal

    def serve():
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            with conn:
                req = conn.recv(4096).decode()
                first, *rest = req.split("\r\n")
                method, url, _ = first.split(" ")
                headers = dict(line.split(": ", 1) for line in rest if ": " in line)
                path = "/" + url.split("/", 3)[3].split("?")[0]
                if "Authorization" not in headers or not _ok(headers["Authorization"], method):
                    conn.sendall((f'RTSP/1.0 401 Unauthorized\r\nCSeq: 1\r\nWWW-Authenticate: Digest realm="{REALM}", '
                                  f'nonce="{NONCE}", stale="FALSE"\r\nWWW-Authenticate: Basic realm="{REALM}"\r\n\r\n').encode())
                else:
                    code = paths.get(path, 404)
                    conn.sendall(f"RTSP/1.0 {code} X\r\nCSeq: 1\r\n\r\n".encode())

    threading.Thread(target=serve, daemon=True).start()
    yield srv.getsockname()[1]
    srv.close()


def url(port, path="/Streaming/Channels/401", password=PASSWORD):
    from urllib.parse import quote
    return f"rtsp://{USER}:{quote(password, safe='')}@127.0.0.1:{port}{path}"


def test_dice_que_falla(grabador):
    assert "entrega el video" in rtsp_diagnose(url(grabador))  # todo bien: es la PC la que no lo lee (¿H.265?)
    assert "usuario o la clave" in rtsp_diagnose(url(grabador, password="otra"))
    assert "no tiene ese canal" in rtsp_diagnose(url(grabador, "/Streaming/Channels/902"))
    assert "subflujo" in rtsp_diagnose(url(grabador, "/Streaming/Channels/402"))  # el 401 sí: falta activar el 402


def test_puerto_cerrado_nombre_inexistente_y_otro_servicio():
    libre = socket.socket()
    libre.bind(("127.0.0.1", 0))
    port = libre.getsockname()[1]
    libre.close()
    assert "no está redirigido" in rtsp_diagnose(f"rtsp://a:b@127.0.0.1:{port}/Streaming/Channels/102")
    assert "no se encuentra" in rtsp_diagnose("rtsp://a:b@noexiste.invalid:554/Streaming/Channels/102")
    web = socket.socket()
    web.bind(("127.0.0.1", 0))
    web.listen(1)

    def http():
        conn, _ = web.accept()
        with conn:
            conn.recv(4096)
            conn.sendall(b"HTTP/1.1 400 Bad Request\r\n\r\n")
    threading.Thread(target=http, daemon=True).start()
    assert "no como video" in rtsp_diagnose(f"rtsp://a:b@127.0.0.1:{web.getsockname()[1]}/Streaming/Channels/102")
    web.close()


def test_acepta_la_conexion_pero_no_contesta():
    """No es lo mismo que un puerto cerrado: el grabador (o algo en el medio) acepta y se queda callado."""
    mudo = socket.socket()
    mudo.bind(("127.0.0.1", 0))
    mudo.listen(4)
    conns = []
    threading.Thread(target=lambda: conns.append(mudo.accept()), daemon=True).start()
    msg = rtsp_diagnose(f"rtsp://a:b@127.0.0.1:{mudo.getsockname()[1]}/Streaming/Channels/102", timeout=1.0)
    assert "acepta la conexión pero no contesta" in msg
    mudo.close()


def _corta_al_pedir(srv):
    """Acepta y, al llegar el pedido, corta con RST (como el puerto 8000 de un Hikvision ante un pedido RTSP)."""
    import struct

    conn, _ = srv.accept()
    conn.recv(4096)
    conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    conn.close()


def test_conecta_pero_corta_no_es_el_puerto_de_video(monkeypatch):
    """Ancap San Luis (3 nov): el técnico pasó el 8000 y la app decía «revisá el DDNS y el router», que andaban."""
    import agent.cameras as cameras

    for hint in (None, "el 8000 es el puerto del programa de Hikvision"):
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        if hint:
            monkeypatch.setitem(cameras.SDK_PORTS, port, hint)
        threading.Thread(target=_corta_al_pedir, args=(srv,), daemon=True).start()
        msg = rtsp_diagnose(f"rtsp://a:b@127.0.0.1:{port}/Streaming/Channels/102", timeout=2.0)
        srv.close()
        assert "el DDNS y el router andan" in msg and "no es el de video" in msg
        assert "revisá el DDNS" not in msg
        assert (hint or "554") in msg


def test_con_link_la_marca_sale_del_link_sin_probar(monkeypatch):
    import agent.cameras as cameras

    monkeypatch.setattr(cameras, "detect_brand", lambda cfg: pytest.fail("no tenía que probar la marca"))
    cam = cameras.build_camera({"brand": "ip", "host": "h", "channel": 3,
                                "source_url": "rtsp://u:p@h:554/cam/realmonitor?channel=3&subtype=0"})
    assert cam.resolve().brand == "dahua" and cam.live_url().endswith("subtype=0")


def test_la_marca_se_detecta_aunque_no_este_la_calidad_secundaria(monkeypatch):
    import agent.cameras as cameras

    monkeypatch.setattr(cameras, "build_and_probe", lambda cfg: {"error": "sin página"})
    pedidos = []

    def test_stream(cam, stream="sub", explain=False):
        pedidos.append((cam.brand, stream))
        return {"ok": cam.brand == "hikvision" and stream == "main"}  # solo anda la principal
    monkeypatch.setattr(cameras, "test_stream", test_stream)
    assert cameras.detect_brand({"host": "solo-principal", "channel": 4}) == "hikvision"
    assert pedidos == [("hikvision", "sub"), ("dahua", "sub"), ("hikvision", "main")]


def test_hik_connect_que_corta_no_es_el_antivirus():
    """5 nov: el servidor de video de Hikvision cortaba sin responder y la app decía «revisá el antivirus de esta PC»."""
    from agent.hikconnect import diagnose

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    def corta():
        conn, _ = srv.accept()
        conn.recv(4096)
        conn.close()  # sin respuesta HTTP: httpx levanta RemoteProtocolError
    threading.Thread(target=corta, daemon=True).start()
    msg = diagnose(f"http://127.0.0.1:{port}/v3/openlive/X_1_2.m3u8", timeout=3.0)
    srv.close()
    assert "cortó la conexión sin mandar el video" in msg and "antivirus" not in msg


# ---- 27 nov, Mayorista Alberto: el grabador bloqueó a la PC por claves mal puestas y la app decía «ese puerto no es
# el de video», cuando otra cámara del mismo grabador abría por ese mismo puerto. Y la app reintentaba cada minuto con
# la clave mal: cada intento cuenta como uno fallido y el bloqueo no se terminaba nunca.

def test_clave_mal_no_se_reintenta_enseguida(grabador):
    from agent.cameras import AUTH_HOLD_S, retry_wait

    msg = rtsp_diagnose(url(grabador, password="otra"))
    assert msg.hold and "usuario o la clave" in msg and "30 minutos" in msg
    assert retry_wait(msg, 4.0) == AUTH_HOLD_S
    bien = rtsp_diagnose(url(grabador, "/Streaming/Channels/902"))  # (otra falla: se reintenta como siempre)
    assert not getattr(bien, "hold", False) and retry_wait(bien, 4.0) == 4.0
    assert retry_wait(None, 8.0) == 8.0 and retry_wait("un texto", 8.0) == 8.0


def _bloqueado(srv, pedidos):
    """Un Hikvision que bloqueó a esta IP: contesta como video y pide la clave, y corta cuando se la mandan."""
    import struct

    while True:
        try:
            conn, _ = srv.accept()
        except OSError:
            return
        req = conn.recv(4096).decode()
        pedidos.append(req)
        if "Authorization" not in req:
            conn.sendall((f'RTSP/1.0 401 Unauthorized\r\nCSeq: 1\r\nWWW-Authenticate: Digest realm="{REALM}", '
                          f'nonce="{NONCE}"\r\n\r\n').encode())
            conn.close()
        else:
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            conn.close()


def test_grabador_que_corta_al_recibir_la_clave_es_un_bloqueo():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(4)
    pedidos = []
    threading.Thread(target=_bloqueado, args=(srv, pedidos), daemon=True).start()
    msg = rtsp_diagnose(url(srv.getsockname()[1], "/Streaming/Channels/102"), timeout=2.0)
    srv.close()
    assert len(pedidos) == 2  # el pedido sin clave (contestó como video) y el con clave (ahí cortó)
    assert msg.hold and "contesta como video" in msg and "bloqueó a esta PC" in msg
    assert "no es el de video" not in msg
    assert len(msg) <= 300  # (el estado de las máquinas se corta a 300 letras)


def test_corta_sin_contestar_nada_puede_ser_un_corte_solo_para_esta_pc():
    """Cierra la conexión sin mandar nada (antes: «contesta pero no como video»): o no es el puerto de video, o el corte
    es solo para esta PC (bloqueada, o dentro de la red del local)."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)

    def cierra():
        conn, _ = srv.accept()
        conn.recv(4096)
        conn.close()
    threading.Thread(target=cierra, daemon=True).start()
    msg = rtsp_diagnose(f"rtsp://a:b@127.0.0.1:{srv.getsockname()[1]}/Streaming/Channels/102", timeout=2.0)
    srv.close()
    assert "corta la conexión al pedir el video" in msg and "no es el de video" in msg
    assert "celular con datos" in msg and "red del local" in msg
    assert not msg.hold  # (no se mandó ninguna clave: reintentar no alarga ningún bloqueo)


def test_la_vista_no_insiste_con_la_clave_mal(grabador, monkeypatch):
    """«En vivo» de la app: con la clave rechazada espera 30 minutos (no 2, 4, 8… segundos) antes de volver a probar."""
    from agent import vista
    from agent.cameras import AUTH_HOLD_S

    esperas = []
    cfg = {"id": 9, "name": "Fiambrería", "connection": {
        "brand": "hikvision", "host": "127.0.0.1", "rtsp_port": grabador, "username": USER, "password": "otra",
        "channel": 1, "stream": "sub"}}
    v = vista.ViewWorker(cfg)
    monkeypatch.setattr(vista.cv2, "VideoCapture", lambda *a, **k: _Cerrado())

    def wait(t):
        esperas.append(t)
        v._stop.set()  # (una vuelta alcanza)
        return True
    monkeypatch.setattr(v._stop, "wait", wait)
    v.run()
    assert esperas == [AUTH_HOLD_S]
    assert v.state == "detenida" and "30 minutos" in (v.error or "")


class _Cerrado:
    def isOpened(self):  # noqa: N802 - la cara de cv2.VideoCapture
        return False

    def release(self):
        pass


def test_al_corregir_la_clave_la_vista_se_vuelve_a_abrir():
    """Guardada la clave nueva en «Editar», la cámara que se mira se reabre ya (no espera los 30 minutos)."""
    import queue
    from types import SimpleNamespace

    from agent.__main__ import Agent
    from agent.vista import ViewWorker

    vieja = {"id": 9, "name": "Fiambrería", "connection": {"brand": "hikvision", "host": "h", "password": "mal"}}
    nueva = {**vieja, "connection": {**vieja["connection"], "password": "bien"}}
    hechos = []

    class Vista(ViewWorker):
        def start(self):
            hechos.append(("abre", self.conn_key))

        def stop(self):
            hechos.append(("cierra", self.conn_key))

    v = Vista(vieja)
    fake = SimpleNamespace(_view_updates=queue.Queue(), viewers={9: v}, _ai_view=False, _views_stale=False,
                           _viewer=lambda cfg, ai: Vista(cfg))
    fake._view_updates.put([vieja])  # lo mismo: no se toca
    Agent._refresh_views(fake)
    assert fake.viewers[9] is v and hechos == []
    fake._view_updates.put([nueva])
    Agent._refresh_views(fake)
    assert fake.viewers[9] is not v and [h[0] for h in hechos] == ["cierra", "abre"]
    assert '"bien"' in fake.viewers[9].conn_key


def test_la_maquina_no_insiste_con_la_clave_mal(grabador, tmp_path, monkeypatch):
    """En una máquina de análisis es peor: el bloqueo es para la IP de la máquina y tira todas las cámaras de ese
    grabador, no solo la de la clave mal."""
    import agent.worker as worker_mod
    from agent.cameras import AUTH_HOLD_S
    from tests.test_video_placa import Counter

    cfg = {"id": 7, "name": "Fiambrería", "process_fps": 5, "analysis": {},
           "connection": {"brand": "hikvision", "host": "127.0.0.1", "rtsp_port": grabador, "username": USER,
                          "password": "Secreta9", "channel": 1}}
    w = worker_mod.CameraWorker(cfg, Counter(), None, worker_mod.Outbox(tmp_path / "outbox", None), None, None,
                                {"clip_pre_seconds": 1, "clip_max_seconds": 30})
    monkeypatch.setattr(worker_mod, "open_video", lambda *a, **k: _Cerrado())
    monkeypatch.setattr(w, "_open_gpu", lambda url: None)
    monkeypatch.setattr(w, "_open_av", lambda url: None)
    esperas = []

    def wait(t):
        esperas.append(t)
        w._stop.set()
        return True
    monkeypatch.setattr(w._stop, "wait", wait)
    w.run()
    assert esperas == [AUTH_HOLD_S]
    assert "30 minutos" in w.error and "Secreta9" not in w.error and "rtsp://" not in w.error  # (ni clave ni link)
