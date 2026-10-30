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
