"""Detección de personas por cámara en el servidor (pedido del dueño, 6 nov: «si están lejos no las cuenta»): la
«Sensibilidad» (1 a 10, 5 = como siempre) y «Personas lejos» se guardan en `cameras.detection`, viajan a las PCs y
máquinas en `analysis["detection"]` solo si difieren de lo de siempre (un agente viejo la ignora), editar desde la web
no la borra y «Personas lejos» pesa 4 veces en el reparto de las máquinas."""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("core.deteccion")  # la arma otra parte (Python puro); sin ella el servidor no arranca

from core.rules import RuleConfig  # noqa: E402
from server import db as server_db  # noqa: E402
from server import pool  # noqa: E402
from server.main import Settings, create_app  # noqa: E402

ADMIN = ("admin", "clave-admin-123")
DEFECTO = {"sensitivity": 5, "far": False}
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def app(tmp_path, db_url):
    return create_app(Settings(data_dir=tmp_path / "data", database_url=db_url, admin_password=ADMIN[1],
                               serve_public=False))


@pytest.fixture
def admin(app):
    with TestClient(app) as c:
        assert c.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]}).status_code == 200
        yield c


def cam_body(cid, name, **kw):
    return {"client_id": cid, "name": name, "brand": "ip", "host": "dvr", "password": "x", **kw}


def web_body(cam: dict) -> dict:
    """Lo que manda la web al editar una cámara (public/app.js): no trae `detection`."""
    return {"client_id": cam["client_id"], "agent_id": cam["agent_id"], "name": cam["name"] + " editada",
            "brand": cam["brand"], "host": cam["host"], "channel": cam["channel"], "username": cam["username"],
            "password": None, "rtsp_port": cam["rtsp_port"], "http_port": cam["http_port"],
            "detect_stream": cam["detect_stream"], "detector": cam["detector"], "process_fps": cam["process_fps"],
            "playback_time": cam["playback_time"], "source_url": None, "nvr_clips": cam["nvr_clips"],
            "enabled": cam["enabled"], "mode": cam["mode"], "hpp_serial": None, "alarm_cfg": cam["alarm_cfg"]}


def setup(admin, plans=None):
    """Un cliente, una PC central (para /api/agent/config) y el token de la PC del admin (para /view y /tune)."""
    cid = admin.post("/api/clients", json={"name": "Depósito Sur", **({"plans": plans} if plans else {})}).json()["id"]
    ag = admin.post("/api/agents", json={"name": "central", "client_id": None}).json()
    tok = admin.post("/api/agent/session").json()["token"]
    return cid, ag, {"X-Agent-Token": ag["token"]}, {"X-Agent-Token": tok}


def config_of(app, h: dict, cam_id: int) -> dict:
    cams = {x["id"]: x for x in TestClient(app).get("/api/agent/config", headers=h).json()["cameras"]}
    return cams[cam_id]


def test_camara_nueva_detecta_como_siempre(app, admin):
    cid, ag, h, _ = setup(admin)
    cam = admin.post("/api/cameras", json=cam_body(cid, "Depósito", agent_id=ag["id"])).json()
    assert cam["detection"] == DEFECTO and cam["detection_default"] is True
    assert admin.get(f"/api/cameras/{cam['id']}").json()["detection"] == DEFECTO
    # lo de siempre no viaja: las máquinas reciben exactamente lo de antes
    assert "detection" not in config_of(app, h, cam["id"])["analysis"]


def test_la_deteccion_viaja_compacta_a_maquinas_ver_con_la_ia_y_configurar(app, admin):
    cid, ag, h, user_h = setup(admin)
    r = admin.post("/api/cameras", json=cam_body(cid, "Fondo", agent_id=ag["id"],
                                                 detection={"sensitivity": 8, "far": True}))
    assert r.status_code == 201, r.text
    cam = r.json()
    assert cam["detection"] == {"sensitivity": 8, "far": True} and cam["detection_default"] is False
    an = config_of(app, h, cam["id"])["analysis"]
    assert an["detection"] == {"sensitivity": 8, "far": True}
    RuleConfig.from_dict(an["rules"])  # nada de la detección en `rules` (un agente viejo es estricto ahí)
    assert "detection" not in an["rules"] and "sensitivity" not in str(an["rules"])
    view = TestClient(app).post("/api/agent/view", json={"camera_ids": [cam["id"]]}, headers=user_h)
    assert view.status_code == 200 and view.json()["cameras"][0]["analysis"]["detection"] == {"sensitivity": 8,
                                                                                               "far": True}
    tune = TestClient(app).post("/api/agent/tune", json={"camera_id": cam["id"]}, headers=user_h)
    assert tune.status_code == 200 and tune.json()["camera"]["analysis"]["detection"] == {"sensitivity": 8,
                                                                                           "far": True}
    assert "detection" not in tune.json()["raw"]  # `raw` sigue igual (zonas, reglas y gestos)
    # solo lo que difiere de lo de siempre
    r = admin.put(f"/api/cameras/{cam['id']}", json={**cam_body(cid, "Fondo", agent_id=ag["id"]),
                                                    "detection": {"sensitivity": 5, "far": True}})
    assert r.status_code == 200, r.text
    assert config_of(app, h, cam["id"])["analysis"]["detection"] == {"far": True}
    admin.put(f"/api/cameras/{cam['id']}", json={**cam_body(cid, "Fondo", agent_id=ag["id"]),
                                                "detection": {"sensitivity": 9, "far": False}})
    assert config_of(app, h, cam["id"])["analysis"]["detection"] == {"sensitivity": 9}


def test_editar_desde_la_web_no_borra_la_deteccion(app, admin):
    cid, ag, h, _ = setup(admin)
    cam = admin.post("/api/cameras", json=cam_body(cid, "Fondo", agent_id=ag["id"],
                                                   detection={"sensitivity": 7, "far": True})).json()
    r = admin.put(f"/api/cameras/{cam['id']}", json=web_body(cam))
    assert r.status_code == 200, r.text
    assert r.json()["name"] == "Fondo editada" and r.json()["detection"] == {"sensitivity": 7, "far": True}
    # `null` también es «no cambia» (lo que manda una app vieja que conoce el campo pero no lo tocó)
    r = admin.put(f"/api/cameras/{cam['id']}", json={**web_body(cam), "detection": None})
    assert r.json()["detection"] == {"sensitivity": 7, "far": True}
    # las zonas desde la web (sin `detection`) tampoco la tocan
    r = admin.put(f"/api/cameras/{cam['id']}/analysis", json={"zones": [], "rules": {}})
    assert r.status_code == 200 and r.json()["detection"] == {"sensitivity": 7, "far": True}
    # volver a lo de siempre: se guarda como NULL
    r = admin.put(f"/api/cameras/{cam['id']}", json={**web_body(cam), "detection": DEFECTO})
    assert r.json()["detection"] == DEFECTO and r.json()["detection_default"] is True
    assert "detection" not in config_of(app, h, cam["id"])["analysis"]
    with app.state.Session() as s:
        assert s.get(server_db.Camera, cam["id"]).detection is None


def test_guardar_la_deteccion_desde_la_ventana_de_la_camara(app, admin):
    cid, ag, h, _ = setup(admin)
    cam = admin.post("/api/cameras", json=cam_body(cid, "Fondo", agent_id=ag["id"])).json()
    rev = int(TestClient(app).get("/api/agent/config", headers=h).json()["rev"])
    r = admin.put(f"/api/cameras/{cam['id']}/analysis",
                  json={"zones": [], "rules": {}, "detection": {"sensitivity": 10, "far": False}})
    assert r.status_code == 200, r.text
    assert r.json()["detection"] == {"sensitivity": 10, "far": False}  # la app ve que el servidor la conoce
    cfg = TestClient(app).get("/api/agent/config", headers=h).json()
    assert int(cfg["rev"]) > rev  # las máquinas piden la configuración ya (y la aplican sin reconectar)
    assert {x["id"]: x for x in cfg["cameras"]}[cam["id"]]["analysis"]["detection"] == {"sensitivity": 10}
    # sin `detection` (una app vieja guardando zonas) no se toca
    r = admin.put(f"/api/cameras/{cam['id']}/analysis", json={"zones": [], "rules": {}, "gestures": None})
    assert r.json()["detection"] == {"sensitivity": 10, "far": False}
    # un operador también puede
    admin.post("/api/users", json={"username": "ana", "password": "clave-ana-123", "role": "entrenador"})
    op = TestClient(app)
    assert op.post("/api/login", json={"username": "ana", "password": "clave-ana-123"}).status_code == 200
    r = op.put(f"/api/cameras/{cam['id']}/analysis", json={"zones": [], "rules": {}, "detection": {"far": True}})
    assert r.status_code == 200 and r.json()["detection"] == {"sensitivity": 5, "far": True}


@pytest.mark.parametrize("det, palabra", [
    ({"sensitivity": 0}, "1 a 10"), ({"sensitivity": 11}, "1 a 10"), ({"sensitivity": "7"}, "1 a 10"),
    ({"sensitivity": True}, "1 a 10"), ({"sensitivity": 7.5}, "1 a 10"), ({"far": "si"}, "true o false"),
    ({"sensitivity": 6, "lejos": True}, "desconocid"), ([5, True], "sensitivity"), ("alta", "sensitivity"),
])
def test_deteccion_invalida_da_422_en_castellano(app, admin, det, palabra):
    cid, ag, _, _ = setup(admin)
    cam = admin.post("/api/cameras", json=cam_body(cid, "Fondo", agent_id=ag["id"],
                                                   detection={"sensitivity": 7})).json()
    for r in (admin.post("/api/cameras", json=cam_body(cid, "Otra", detection=det)),
              admin.put(f"/api/cameras/{cam['id']}", json={**web_body(cam), "detection": det}),
              admin.put(f"/api/cameras/{cam['id']}/analysis", json={"zones": [], "rules": {}, "detection": det})):
        assert r.status_code == 422, r.text
        detail = r.json()["detail"]
        assert isinstance(detail, str) and detail.startswith("detection") and palabra in detail, detail
    # nada quedó a medias: la cámara sigue como estaba y no se creó la otra
    assert admin.get(f"/api/cameras/{cam['id']}").json()["detection"] == {"sensitivity": 7, "far": False}
    assert [c["name"] for c in admin.get(f"/api/cameras?client_id={cid}").json()] == ["Fondo"]


def test_una_camara_de_solo_metricas_tambien_recibe_la_deteccion(app, admin):
    """El caso del pedido: un depósito grande que solo cuenta gente, a 2 cuadros por segundo."""
    cid, ag, h, _ = setup(admin, plans=["comercio"])
    cam = admin.post("/api/cameras", json=cam_body(cid, "Depósito", agent_id=ag["id"],
                                                   detection={"sensitivity": 8, "far": True})).json()
    x = config_of(app, h, cam["id"])
    assert x["metrics_only"] is True and x["analysis"]["rules"]["enabled_rules"] == []
    assert x["analysis"]["detection"] == {"sensitivity": 8, "far": True}
    # también en el modo alarmas y en una cámara de solo métricas por su tipo
    cid2 = admin.post("/api/clients", json={"name": "Kiosco"}).json()["id"]
    m = admin.post("/api/cameras", json=cam_body(cid2, "Fondo", agent_id=ag["id"], profile="metricas",
                                                 detection={"sensitivity": 3})).json()
    assert config_of(app, h, m["id"])["analysis"]["detection"] == {"sensitivity": 3}
    al = admin.post("/api/cameras", json=cam_body(cid2, "Alarmas", agent_id=ag["id"], mode="alarmas_equipo",
                                                  detection={"far": True})).json()
    assert config_of(app, h, al["id"])["analysis"]["detection"] == {"far": True}


def test_personas_lejos_pesa_4_veces_en_el_reparto(app, admin):
    now = datetime(2026, 11, 6, 15, 0)
    w = lambda **kw: pool.camera_weight(**{"fps": 5, "mode": "vivo", "brand": "ip", "hours": None,  # noqa: E731
                                           "tz": "America/Montevideo", "off_hours": "apagada", "now": now, **kw})
    assert pool.FAR_FACTOR == 4
    assert w() == pytest.approx(1.0) and w(far=True) == pytest.approx(4.0)
    assert w(far=True, pose=True) == pytest.approx(4.0 * pool.POSE_FACTOR)
    assert w(far=True, fps=2) == pytest.approx(4 * w(fps=2))
    # en los modos de alarma se pesa lo de escuchar al grabador: no cambia
    assert w(far=True, mode="alarmas_equipo") == w(mode="alarmas_equipo") == pytest.approx(0.1)
    cerrado = {"hours": {"lun": "09:00-10:00"}}  # jueves: cerrado
    assert w(far=True, **cerrado) == 0.0  # local cerrado y «No mirar»: no ocupa nada
    assert w(far=True, off_hours="vivo", **cerrado) == pytest.approx(4.0)  # sigue mirando de noche: sigue pesando

    cid = admin.post("/api/clients", json={"name": "Súper"}).json()["id"]
    caja = admin.post("/api/cameras", json=cam_body(cid, "Caja", profile="caja")).json()  # sin gestos: pesa 1
    mov = admin.post("/api/cameras", json=cam_body(cid, "Portón", profile="caja", detector="motion")).json()
    r = admin.post("/api/cameras/analyze", json={"camera_ids": [caja["id"], mov["id"]], "analyze": True})
    assert r.status_code == 200, r.text
    units = lambda: admin.get("/api/analysis").json()["plan"]["units"]  # noqa: E731
    assert units() == pytest.approx(2.0, abs=0.01)
    r = admin.put(f"/api/cameras/{caja['id']}/analysis", json={"zones": [], "rules": {}, "detection": {"far": True}})
    assert r.status_code == 200, r.text
    assert units() == pytest.approx(5.0, abs=0.01)  # la caja pesa 4 y la de movimiento sigue en 1
    # el detector de movimiento no mira por partes: no pesa más aunque tenga «Personas lejos» guardado
    admin.put(f"/api/cameras/{mov['id']}/analysis", json={"zones": [], "rules": {}, "detection": {"far": True}})
    assert units() == pytest.approx(5.0, abs=0.01)
    # solo la sensibilidad no cambia el peso
    admin.put(f"/api/cameras/{caja['id']}/analysis",
              json={"zones": [], "rules": {}, "detection": {"sensitivity": 10, "far": False}})
    assert units() == pytest.approx(2.0, abs=0.01)


def test_el_cliente_no_ve_la_deteccion(app, admin):
    cid, ag, _, _ = setup(admin)
    cam = admin.post("/api/cameras", json=cam_body(cid, "Fondo", detection={"sensitivity": 8})).json()
    admin.post("/api/users", json={"username": "dueno", "password": "clave-dueno-1", "role": "cliente",
                                   "client_id": cid})
    u = TestClient(app)
    assert u.post("/api/login", json={"username": "dueno", "password": "clave-dueno-1"}).status_code == 200
    mine = {c["id"]: c for c in u.get("/api/cameras").json()}
    assert "detection" not in mine[cam["id"]] and "detection_default" not in mine[cam["id"]]


def test_columna_nueva_y_el_servidor_sigue_sin_opencv():
    # la columna nueva se agrega sola en producción solo si sube la versión del esquema
    assert server_db.SCHEMA_VERSION > "2026-11-03.2"
    assert "detection" in server_db.Camera.__table__.columns
    # en Vercel no hay OpenCV ni onnxruntime: el servidor solo puede importar core.deteccion (Python puro)
    code = ("import sys, server.main\n"
            "bad = [m for m in ('cv2', 'onnxruntime', 'core.detectors') if m in sys.modules]\n"
            "print(','.join(bad))\n"
            "sys.exit(1 if bad else 0)\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
