"""Ahorro de insumos: horario del local, filtro de movimiento, cámaras solo de métricas y un solo envío
de métricas por PC."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pytest
from fastapi.testclient import TestClient

from agent.__main__ import effective_mode
from agent.worker import MetricsQueue
from core.analyzer import Analyzer
from core.detectors import Detection
from core.motion import MotionGate
from core.rules import FUERA_DE_HORARIO, RuleEngine, parse_config
from core.schedule import is_open, next_open, normalize_hours, open_hours_per_week, parse_hours
from core.tracker import Track
from server.main import Settings, create_app

TZ = ZoneInfo("America/Montevideo")
ADMIN = ("admin", "clave-admin-123")
SEMANA = {"lun": "09:00-17:00", "mar": "9-17", "mie": "09:00-13:00, 16:00-20:00", "jue": "", "vie": "18:00-02:00",
          "sab": "", "dom": ""}


def at(day: int, hh: int, mm: int = 0) -> datetime:
    """Lunes 26/10/2026 = día 0."""
    return datetime(2026, 10, 26, hh, mm, tzinfo=TZ) + timedelta(days=day)


# ---------------------------------------------------------------- horario
def test_horario_se_escribe_como_lo_escribe_un_comerciante():
    h = {**SEMANA, "mar": "09:00 a 17:00"}
    norm = normalize_hours(h)
    assert norm["lun"] == "09:00-17:00" and norm["mar"] == "09:00-17:00"
    assert norm["mie"] == "09:00-13:00, 16:00-20:00" and norm["vie"] == "18:00-02:00" and norm["jue"] == ""
    assert normalize_hours({"lun": "00:00-24:00"})["lun"] == "00:00-24:00"
    assert normalize_hours({}) is None and normalize_hours({"lun": "", "dom": ""}) is None  # sin horario = 24 h
    for bad in ({"lun": "nueve a cinco"}, {"lun": "25:00-26:00"}, {"lun": "09:00-09:00"}, {"lunes": "09:00-17:00"}):
        with pytest.raises(ValueError):
            parse_hours(bad)


def test_abierto_cerrado_con_margen_cortado_y_pasando_medianoche():
    h = {**SEMANA, "mar": "09:00-17:00"}
    assert not is_open(h, at(0, 8, 40)) and is_open(h, at(0, 8, 46))  # 15 min antes de abrir ya mira
    assert is_open(h, at(0, 17, 14)) and not is_open(h, at(0, 17, 16))  # y 15 min después de cerrar
    assert is_open(h, at(2, 12)) and not is_open(h, at(2, 14)) and is_open(h, at(2, 19))  # horario cortado
    assert not is_open(h, at(3, 12))  # jueves cerrado
    assert is_open(h, at(4, 23)) and is_open(h, at(5, 1, 30)) and not is_open(h, at(5, 3))  # viernes a la noche
    assert is_open(None, at(3, 3)) and is_open({}, at(3, 3))  # sin horario: siempre
    assert next_open(h, at(0, 18)) == at(1, 8, 45)
    assert next_open(h, at(5, 10)) == at(7, 8, 45)  # sábado: vuelve el lunes
    assert next_open(h, at(0, 10)) is None
    assert open_hours_per_week({d: "09:00-17:00" for d in ("lun", "mar", "mie", "jue", "vie", "sab")}) == 48


def test_modo_de_la_camara_segun_el_horario():
    base = {"mode": "vivo", "connection": {"brand": "ip", "timezone": "America/Montevideo"}}
    sch = {"hours": {"lun": "09:00-17:00"}, "margin_min": 15}
    abierto, cerrado = at(0, 10), at(0, 22)
    assert effective_mode({**base}, cerrado) == "vivo"  # sin horario, como siempre
    cam = {**base, "schedule": {**sch, "off_hours": "apagada"}}
    assert effective_mode(cam, abierto) == "vivo" and effective_mode(cam, cerrado) is None
    cam = {**base, "schedule": {**sch, "off_hours": "alarmas"}}
    assert effective_mode(cam, cerrado) == "alarmas_equipo"
    hc = {**cam, "connection": {"brand": "hikconnect", "timezone": "America/Montevideo"}}
    assert effective_mode(hc, cerrado) is None  # por Hik-Connect todavía no llegan alarmas
    cam = {**base, "schedule": {**sch, "off_hours": "vivo"}}
    assert effective_mode(cam, cerrado) == "vivo"


def test_regla_fuera_de_horario_usa_el_horario_del_local():
    zones, cfg = parse_config({"zones": [], "schedule": {"lun": "09:00-17:00"}})
    eng = RuleEngine(zones, cfg)
    t = Track(1, (100, 100, 150, 250), "persona", 0.0, 0.0, hits=5)
    assert eng.update([t], 0.0, (640, 360), at(0, 10)) == []
    t.last_ts = 400.0
    evs = eng.update([t], 400.0, (640, 360), at(0, 22))
    assert [e.type for e in evs] == [FUERA_DE_HORARIO] and "cerrado" in evs[0].description


# ---------------------------------------------------------------- filtro de movimiento
def test_filtro_de_movimiento_saltea_la_escena_quieta_y_no_pierde_a_nadie():
    gate = MotionGate(every=2.0)
    fondo = np.full((360, 640, 3), 90, np.uint8)
    rng = np.random.default_rng(1)
    decisiones = []
    for i in range(25):  # 5 s a 5 fps de escena quieta (con un poco de ruido de imagen)
        f = np.clip(fondo.astype(int) + rng.integers(-4, 5, fondo.shape), 0, 255).astype(np.uint8)
        decisiones.append(gate.should_analyze(f, i * 0.2, busy=False))
    assert decisiones[0] and sum(decisiones) <= 4  # el primero y uno cada 2 s
    persona = fondo.copy()
    persona[150:260, 300:330] = 230  # alguien entra (chico, lejos)
    assert gate.should_analyze(persona, 5.2, busy=False)
    assert gate.should_analyze(fondo, 5.4, busy=True)  # con alguien siguiéndose se analiza siempre
    assert 0.7 < gate.take_saved_fraction() < 0.95


class _Det:
    def __init__(self):
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return [Detection((300, 150, 330, 260))]


def test_analizador_solo_metricas_no_guarda_clips_y_quieto_guarda_1_por_segundo():
    det = _Det()
    a = Analyzer({}, det, clips=False)
    frame = np.zeros((360, 640, 3), np.uint8)
    for i in range(10):
        a.process(frame, i * 0.5)
    assert det.calls == 10 and len(a._buffer) == 0 and a.tracker.active
    b = Analyzer({}, _Det())
    for i in range(10):
        b.idle(frame, i * 0.2)  # 2 s quietos a 5 fps
    assert len(b._buffer) == 2 and not b.busy


# ---------------------------------------------------------------- métricas en un solo envío
class _Client:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def post_metrics(self, batch):
        if self.fail:
            raise RuntimeError("sin internet")
        self.calls.append(list(batch))


def test_metricas_de_todas_las_camaras_en_un_solo_pedido():
    cli = _Client()
    q = MetricsQueue(cli)
    for cam in range(1, 41):
        q.add([{"camera_id": cam, "bucket_start": "2026-10-26T10:00:00", "data": {"people": 1}}])
    q.add([{"camera_id": 1, "bucket_start": "2026-10-26T10:00:00", "data": {"people": 7}}])  # el último manda
    q.flush()
    assert len(cli.calls) == 1 and len(cli.calls[0]) == 40
    assert next(b for b in cli.calls[0] if b["camera_id"] == 1)["data"]["people"] == 7
    # si falla, se guarda y no pisa lo más nuevo que llegó mientras tanto
    bad = MetricsQueue(_Client(fail=True))
    bad.add([{"camera_id": 1, "bucket_start": "h", "data": {"people": 1}}])
    bad.flush()
    assert bad.pending() == 1
    bad.add([{"camera_id": 1, "bucket_start": "h", "data": {"people": 3}}])
    bad.add([{"camera_id": 1, "bucket_start": "h", "data": {"people": 1}}], older=True)
    bad.client = cli
    bad.flush()
    assert cli.calls[-1][0]["data"]["people"] == 3


# ---------------------------------------------------------------- servidor
@pytest.fixture
def app(tmp_path, db_url):
    return create_app(Settings(data_dir=tmp_path / "data", database_url=db_url, admin_password=ADMIN[1],
                               serve_public=False))


def test_servidor_horario_solo_metricas_y_fuera_de_horario(app):
    with TestClient(app) as a:
        a.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]})
        r = a.post("/api/clients", json={"name": "Kiosco", "hours": {"lun": "25-26"}})
        assert r.status_code == 422 and "Lunes" in r.json()["detail"]
        kiosco = a.post("/api/clients", json={"name": "Kiosco", "hours": {"lun": "9-17", "mar": "09:00 a 17:00"}}).json()
        assert kiosco["hours"]["lun"] == "09:00-17:00" and kiosco["hours"]["dom"] == ""
        conteo = a.post("/api/clients", json={"name": "Tienda", "plans": ["comercio"]}).json()
        # editar sin mandar el horario (app vieja) no lo borra
        a.put(f"/api/clients/{kiosco['id']}", json={"name": "Kiosco", "plans": ["hurto"]})
        assert next(c for c in a.get("/api/clients").json() if c["id"] == kiosco["id"])["hours"]["lun"] == "09:00-17:00"
        me = a.get("/api/me").json()
        cam = lambda cid, n, **kw: a.post("/api/cameras", json={"client_id": cid, "name": n, "brand": "ip",  # noqa: E731
                                                                "host": "dvr", "password": "x",
                                                                "operator_id": me["id"], **kw}).json()
        caja = cam(kiosco["id"], "Caja", off_hours="alarmas")
        gond = cam(kiosco["id"], "Góndola")
        puerta = cam(conteo["id"], "Puerta")
        assert caja["off_hours"] == "alarmas" and gond["off_hours"] == "apagada"
        # editar sin off_hours no lo cambia
        a.put(f"/api/cameras/{caja['id']}", json={"client_id": kiosco["id"], "name": "Caja", "brand": "ip", "host": "dvr"})
        assert next(x for x in a.get("/api/cameras").json() if x["id"] == caja["id"])["off_hours"] == "alarmas"
        a.put(f"/api/cameras/{puerta['id']}/analysis", json={"zones": [
            {"name": "Puerta", "type": "entrance", "polygon": [[0, 0], [1, 0], [1, 1]]}], "rules": {}})
        tok = a.post("/api/agent/session").json()["token"]
        ag = TestClient(app, headers={"X-Agent-Token": tok})
        by = {c["name"]: c for c in ag.get("/api/agent/config").json()["cameras"]}
        assert by["Caja"]["schedule"] == {"hours": kiosco["hours"], "margin_min": 15, "off_hours": "alarmas"}
        assert by["Caja"]["analysis"]["schedule"] == kiosco["hours"]
        assert by["Caja"]["process_fps"] == 5 and not by["Caja"]["metrics_only"]
        assert by["Puerta"]["metrics_only"] and by["Puerta"]["process_fps"] == 3  # cuenta ingresos
        assert by["Puerta"]["analysis"]["rules"]["enabled_rules"] == []
        assert by["Puerta"]["schedule"]["hours"] is None
        # con el local cerrado la caja escucha las alarmas del grabador (aunque su modo sea «vivo»)
        r = ag.post("/api/agent/alarms", json={"camera_id": caja["id"], "event_type": "VMD",
                                               "occurred_at": "2026-10-26T22:00:00"})
        assert r.status_code == 201, r.text
        r = ag.post("/api/agent/alarms", json={"camera_id": gond["id"], "event_type": "VMD",
                                               "occurred_at": "2026-10-26T22:00:00"})
        assert r.status_code == 409
        # estado «local cerrado» que manda la PC
        ag.post("/api/agent/status", json={"cameras": [{"camera_id": gond["id"], "state": "fuera_de_horario"}]})
        st = next(x for x in a.get("/api/cameras").json() if x["id"] == gond["id"])["status"]
        assert st["state"] == "fuera_de_horario"
