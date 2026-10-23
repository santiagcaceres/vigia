from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from agent.cameras import BRANDS, build_camera
from agent.dahua import DahuaCamera
from agent.hikvision import HikCamera

PW = "Cl@ve#1 con espacio"
TZ = "America/Montevideo"


def cam(**kw) -> DahuaCamera:
    return DahuaCamera(host="192.168.1.20", username="vigia", password=PW, channel=3, timezone=TZ, **kw)


def test_url_en_vivo_principal_y_secundario():
    assert cam().live_url("main").endswith("/cam/realmonitor?channel=3&subtype=0")
    assert cam().live_url("sub").endswith("/cam/realmonitor?channel=3&subtype=1")
    # por defecto se analiza el secundario (menos carga para el grabador)
    assert cam().live_url() == cam().live_url("sub")
    # una URL manual gana sobre todo lo demás
    assert cam(source_url="rtsp://otra/marca").live_url() == "rtsp://otra/marca"


def test_la_clave_viaja_codificada_y_no_en_claro_en_los_logs():
    url = cam().live_url()
    assert PW not in url and "Cl%40ve%231%20con%20espacio" in url
    assert PW not in cam().redacted(url) and "%40" not in cam().redacted(url)


def test_playback_usa_formato_de_fecha_dahua():
    inicio = datetime(2026, 10, 19, 15, 30, 5, tzinfo=ZoneInfo(TZ))
    fin = datetime(2026, 10, 19, 15, 30, 25, tzinfo=ZoneInfo(TZ))
    url = cam().playback_url(inicio, fin)
    assert "/cam/playback?channel=3" in url
    assert "starttime=2026_10_19_15_30_05" in url and "endtime=2026_10_19_15_30_25" in url
    # el mismo instante en UTC cuando el equipo guarda en UTC (Montevideo = UTC-3)
    url_utc = cam(playback_time="utc").playback_url(inicio, fin)
    assert "starttime=2026_10_19_18_30_05" in url_utc


def test_la_fabrica_elige_la_marca():
    base = {"host": "10.0.0.5", "username": "u", "password": "p", "channel": 2}
    assert isinstance(build_camera({**base, "brand": "dahua"}), DahuaCamera)
    assert isinstance(build_camera({**base, "brand": "hikvision"}), HikCamera)
    assert isinstance(build_camera(base), HikCamera)  # por defecto, Hikvision
    assert set(BRANDS) == {"hikvision", "dahua", "hikconnect"}


@pytest.mark.parametrize("brand,marca", [("dahua", DahuaCamera), ("hikvision", HikCamera)])
def test_cada_marca_arma_su_propia_url(brand, marca):
    c = build_camera({"host": "h", "username": "u", "password": "p", "channel": 1, "brand": brand})
    assert isinstance(c, marca) and c.brand == brand
    assert c.live_url().startswith("rtsp://u:p@h:554/")
    assert ("/cam/realmonitor" in c.live_url()) == (brand == "dahua")
