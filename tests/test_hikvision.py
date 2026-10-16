import logging
import socket
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

from agent.hikvision import HikCamera, download_nvr_clip
from core.clips import redact

PW = "Cl@ve#1/x"


def cam(**kw):
    return HikCamera(host="192.168.1.10", username="admin", password=PW, channel=4, **kw)


def test_urls_en_vivo_con_clave_codificada():
    c = cam()
    enc = quote(PW, safe="")
    assert c.live_url() == f"rtsp://admin:{enc}@192.168.1.10:554/Streaming/Channels/402"
    assert c.live_url("main").endswith("/Streaming/Channels/401")
    assert HikCamera(host="h", channel=12).live_url("main").endswith("/Streaming/Channels/1201")
    assert cam(source_url="rtsp://otra/marca").live_url() == "rtsp://otra/marca"


def test_url_de_grabacion_hora_local_y_utc():
    start = datetime(2026, 10, 16, 14, 30, 5, tzinfo=timezone.utc)  # 11:30:05 en Buenos Aires
    end = start + timedelta(seconds=12)
    local = cam(playback_time="local").playback_url(start, end)
    assert "/Streaming/tracks/401?starttime=20261016T113005Z&endtime=20261016T113017Z" in local
    utc = cam(playback_time="utc").playback_url(start, end)
    assert "starttime=20261016T143005Z&endtime=20261016T143017Z" in utc


def test_redact():
    assert redact("rtsp://admin:Cl%40ve@10.0.0.1:554/x") == "rtsp://***@10.0.0.1:554/x"
    assert redact("sin credenciales rtsp://10.0.0.1/x") == "sin credenciales rtsp://10.0.0.1/x"


def test_fallo_del_nvr_no_filtra_la_clave_en_logs(tmp_path, caplog):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]  # puerto cerrado
    c = HikCamera(host="127.0.0.1", rtsp_port=port, username="admin", password=PW)
    now = datetime.now(timezone.utc)
    with caplog.at_level(logging.DEBUG):
        assert download_nvr_clip(c, now - timedelta(seconds=5), now, tmp_path / "c.mp4", timeout=20) is False
    text = caplog.text
    assert "Streaming/tracks/101" in text
    assert PW not in text and quote(PW, safe="") not in text
