"""Gestos de ocultamiento en el servidor: por tipo de cámara, guardados aparte de `rules` (un agente viejo rechaza
nombres que no conoce ahí), nunca en cámaras solo de métricas ni en clientes sin plan «hurto», el ingreso de las
alertas de gesto y el peso de la cámara en el reparto de las máquinas."""

import json
import uuid as uuidlib
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from core.gestos import GESTURE_TYPES, GestureConfig
from core.rules import RULE_TYPES, RuleConfig, parse_config
from server import pool
from server.main import CAMERA_PROFILES, Settings, create_app
from server.services import LABELS as TELEGRAM_LABELS

ADMIN = ("admin", "clave-admin-123")
PROFILES = ("general", "gondola", "caja", "entrada", "deposito", "exterior", "surtidor", "metricas")
FEATS = {"gesture_dwell_s": 1.4, "since_take_s": 2.5, "takes": 1, "reaches": 2, "wrist_conf_drop": 0.2,
         "bag_seen": 0, "pose_ok_ratio": 0.9, "track_age_s": 30.0, "torso_px": 80}


@pytest.fixture
def app(tmp_path, db_url):
    return create_app(Settings(data_dir=tmp_path / "data", database_url=db_url, admin_password=ADMIN[1],
                               serve_public=False))


@pytest.fixture
def admin(app):
    with TestClient(app) as c:
        assert c.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]}).status_code == 200
        yield c


def new_camera(admin, cid: int, name: str, profile: str = "general") -> dict:
    r = admin.post("/api/cameras", json={"client_id": cid, "name": name, "brand": "ip", "host": "dvr",
                                         "password": "x", "profile": profile})
    assert r.status_code in (200, 201), r.text
    return r.json()


def tune(app, admin, cam_id: int) -> dict:
    """Lo que recibe la app en «Configurar…» (la misma configuración que la de las máquinas)."""
    tok = admin.post("/api/agent/session").json()["token"]
    r = TestClient(app, headers={"X-Agent-Token": tok}).post("/api/agent/tune", json={"camera_id": cam_id})
    assert r.status_code == 200, r.text
    return r.json()


def check_old_agent_safe(analysis: dict) -> None:
    """Un agente viejo lee `rules` con RuleConfig.from_dict (estricto): ahí no puede haber nada de gestos."""
    rules = analysis["rules"]
    RuleConfig.from_dict(rules)
    assert set(rules.get("enabled_rules", [])) <= set(RULE_TYPES)
    assert "mano_" not in json.dumps(rules) and "gestures" not in rules
    parse_config(analysis)  # y el agente nuevo entiende todo junto


def test_gestos_segun_el_tipo_de_camara(app, admin):
    cid = admin.post("/api/clients", json={"name": "Súper"}).json()["id"]  # plan «hurto» por defecto
    for prof in PROFILES:
        cam = new_camera(admin, cid, f"Cámara {prof}", prof)
        assert cam["gestures"] is None  # nada elegido a mano: rige el tipo
        assert cam["gestures_default"] == list(CAMERA_PROFILES[prof]["gestures"])
        analysis = tune(app, admin, cam["id"])["camera"]["analysis"]
        check_old_agent_safe(analysis)
        if prof in ("general", "gondola"):
            # compacto: los gestos (los tiempos por defecto no viajan; una clave nueva llega solo si se cambió)
            assert analysis["gestures"] == {"enabled": list(GESTURE_TYPES)}
            assert not analysis["metrics_only"]
            _, cfg = parse_config(analysis)
            assert cfg.gestures.enabled == GESTURE_TYPES
        else:
            assert "gestures" not in analysis, prof
    # el panel muestra los nombres de los gestos como tipos de alerta
    assert set(GESTURE_TYPES) <= set(admin.get("/api/meta").json()["rule_types"])


def test_solo_gestos_no_es_solo_metricas(app, admin):
    cid = admin.post("/api/clients", json={"name": "Kiosco"}).json()["id"]
    cam = new_camera(admin, cid, "Góndola", "gondola")
    r = admin.put(f"/api/cameras/{cam['id']}/analysis", json={
        "zones": [], "rules": {"enabled_rules": []}, "gestures": {"enabled": ["mano_bolsillo"]}})
    assert r.status_code == 200, r.text
    assert r.json()["gestures"] == {"enabled": ["mano_bolsillo"]}
    t = tune(app, admin, cam["id"])
    analysis = t["camera"]["analysis"]
    check_old_agent_safe(analysis)
    assert analysis["rules"]["enabled_rules"] == [] and analysis["gestures"]["enabled"] == ["mano_bolsillo"]
    assert not analysis["metrics_only"] and not t["camera"]["metrics_only"]  # si no, no habría alertas ni clips
    assert t["camera"]["process_fps"] >= 5

    # en cambio, todas las reglas apagadas a mano y los gestos nunca tocados: sigue solo de métricas, como antes
    # (no se la pone a mirar posturas sin que nadie lo pida)
    cam2 = new_camera(admin, cid, "Pasillo", "general")
    admin.put(f"/api/cameras/{cam2['id']}/analysis", json={"zones": [], "rules": {"enabled_rules": []}})
    a2 = tune(app, admin, cam2["id"])["camera"]["analysis"]
    assert a2["metrics_only"] and "gestures" not in a2
    assert admin.get(f"/api/cameras/{cam2['id']}").json()["gestures_default"] == []


def test_sin_plan_hurto_ni_solo_metricas_no_hay_gestos(app, admin):
    conteo = admin.post("/api/clients", json={"name": "Tienda", "plans": ["comercio"]}).json()["id"]
    cam = new_camera(admin, conteo, "Góndola", "gondola")
    admin.put(f"/api/cameras/{cam['id']}/analysis",
              json={"zones": [], "rules": {}, "gestures": {"enabled": list(GESTURE_TYPES)}})
    analysis = tune(app, admin, cam["id"])["camera"]["analysis"]
    assert analysis["metrics_only"] and "gestures" not in analysis
    check_old_agent_safe(analysis)

    # tipo «Solo métricas»: nunca, ni elegidos a mano
    cid = admin.post("/api/clients", json={"name": "Súper"}).json()["id"]
    met = new_camera(admin, cid, "Conteo", "metricas")
    admin.put(f"/api/cameras/{met['id']}/analysis",
              json={"zones": [], "rules": {}, "gestures": {"enabled": ["mano_ropa"]}})
    analysis = tune(app, admin, met["id"])["camera"]["analysis"]
    assert analysis["metrics_only"] and "gestures" not in analysis


def test_guardar_gestos_aparte_de_las_reglas(app, admin):
    cid = admin.post("/api/clients", json={"name": "Súper"}).json()["id"]
    cam = new_camera(admin, cid, "Góndola", "gondola")
    url = f"/api/cameras/{cam['id']}/analysis"
    zones = [{"name": "Estante", "type": "shelf", "polygon": [[0, 0.5], [1, 0.5], [1, 1], [0, 1]]}]
    ok = admin.put(url, json={"zones": zones, "rules": {}, "gestures": {"enabled": ["mano_ropa", "mano_bolsillo"],
                                                                        "pocket_seconds": 0.8}})
    assert ok.status_code == 200, ok.text
    saved = {"enabled": ["mano_bolsillo", "mano_ropa"], "pocket_seconds": 0.8}  # solo lo que vino, normalizado
    assert ok.json()["gestures"] == saved

    # la app vieja manda solo zonas y reglas: los gestos no se borran
    assert admin.put(url, json={"zones": zones, "rules": {"loiter_seconds": 30}}).status_code == 200
    assert admin.get(f"/api/cameras/{cam['id']}").json()["gestures"] == saved

    # inválidos: 422 con el motivo, y no se toca nada
    for bad in ({"enabled": ["volar"]}, {"enabled": "mano_bolsillo"}, {"enabled": 3}, {"pocket_seconds": -1},
                {"pocket_seconds": "mucho"}, {"parpadeo": 1}):
        r = admin.put(url, json={"zones": zones, "rules": {}, "gestures": bad})
        assert r.status_code == 422 and r.json()["detail"].startswith("gestures"), (bad, r.text)
    # los gestos no van dentro de `rules` (un agente viejo dejaría de analizar la cámara)
    for bad in ({"enabled_rules": ["merodeo", "mano_bolsillo"]}, {"cooldown_seconds": {"mano_ropa": 5}},
                {"mano_bolsa": True}):
        assert admin.put(url, json={"zones": zones, "rules": bad}).status_code == 422, bad
    assert admin.get(f"/api/cameras/{cam['id']}").json()["gestures"] == saved

    # el agente recibe lo efectivo: los dos elegidos, con el tiempo cambiado y el resto por defecto
    t = tune(app, admin, cam["id"])
    assert t["raw"]["gestures"] == saved and t["raw"]["rules"] == {"loiter_seconds": 30}
    eff = t["camera"]["analysis"]["gestures"]
    assert eff == {"enabled": ["mano_bolsillo", "mano_ropa"], "pocket_seconds": 0.8}
    assert GestureConfig.from_dict(eff).clothes_seconds == GestureConfig().clothes_seconds  # el resto, por defecto
    check_old_agent_safe(t["camera"]["analysis"])

    # apagados a propósito (aunque el tipo los traiga)
    admin.put(url, json={"zones": zones, "rules": {}, "gestures": {"enabled": []}})
    t = tune(app, admin, cam["id"])
    assert t["raw"]["gestures"] == {"enabled": []} and "gestures" not in t["camera"]["analysis"]

    # sin "enabled": los del tipo, con esos tiempos
    admin.put(url, json={"zones": zones, "rules": {}, "gestures": {"bag_seconds": 1.5}})
    eff = tune(app, admin, cam["id"])["camera"]["analysis"]["gestures"]
    assert eff["enabled"] == list(GESTURE_TYPES) and eff["bag_seconds"] == 1.5


def test_alertas_de_gesto_se_guardan_con_sus_medidas(app, admin):
    cid = admin.post("/api/clients", json={"name": "Súper"}).json()["id"]
    ag = admin.post("/api/agents", json={"client_id": cid, "name": "PC Súper"}).json()
    cam = new_camera(admin, cid, "Góndola", "gondola")
    uid = uuidlib.uuid4().hex
    meta = {"uuid": uid, "camera_id": cam["id"], "type": "mano_bolsillo", "track_id": 3,
            "description": "Persona #3: posible ocultamiento, mano derecha al bolsillo 1.4 s, después de tomar algo",
            "occurred_at": "2026-10-29T11:00:00-03:00", "features": FEATS, "alert": True}
    r = TestClient(app).post("/api/agent/events", headers={"X-Agent-Token": ag["token"]},
                             data={"meta": json.dumps(meta)},
                             files={"snapshot": ("s.jpg", b"\xff\xd8fakejpg", "image/jpeg")})
    assert r.status_code == 201, r.text
    ev = admin.get(f"/api/events/{r.json()['id']}").json()
    assert ev["type"] == "mano_bolsillo" and ev["severity"] == "alta"
    for k in ("gesture_dwell_s", "since_take_s", "takes", "reaches", "wrist_conf_drop", "bag_seen", "pose_ok_ratio"):
        assert ev["features"][k] == float(FEATS[k])
    assert "torso_px" not in ev["features"]  # lo que el modelo no usa no se guarda
    lista = admin.get("/api/events?type=mano_bolsillo").json()
    assert [e["id"] for e in lista["items"]] == [ev["id"]]

    # ejemplo para entrenar desde «Configurar…» (alerta de prueba etiquetada)
    meta = {"uuid": uuidlib.uuid4().hex, "camera_id": cam["id"], "type": "mano_bolsa", "label": "falso_positivo",
            "description": "prueba", "features": FEATS, "video": "pasillo.mp4"}
    r = admin.post("/api/training/samples", data={"meta": json.dumps(meta)},
                   files={"snapshot": ("a.jpg", b"\xff\xd8jpeg", "image/jpeg")})
    assert r.status_code == 201, r.text
    ev = admin.get(f"/api/events/{r.json()['id']}").json()
    assert ev["type"] == "mano_bolsa" and ev["severity"] == "media" and ev["features"]["gesture_dwell_s"] == 1.4

    # el aviso al equipo por Telegram dice «posible ocultamiento», no acusa
    for g in GESTURE_TYPES:
        assert TELEGRAM_LABELS[g].startswith("Posible ocultamiento")


def test_el_peso_de_la_camara_sube_con_gestos(app, admin):
    now = datetime(2026, 10, 29, 15, 0)
    w = lambda **kw: pool.camera_weight(**{"fps": 5, "mode": "vivo", "brand": "ip", "hours": None,  # noqa: E731
                                           "tz": "America/Montevideo", "off_hours": "apagada", "now": now, **kw})
    assert w(pose=True) == pytest.approx(w() * pool.POSE_FACTOR) and w() == pytest.approx(1.0)
    cerrado = {"hours": {"lun": "09:00-10:00"}}  # jueves: cerrado
    assert w(pose=True, **cerrado) == 0.0

    cid = admin.post("/api/clients", json={"name": "Súper"}).json()["id"]
    cam = new_camera(admin, cid, "Góndola", "gondola")
    caja = new_camera(admin, cid, "Caja", "caja")
    r = admin.post("/api/cameras/analyze", json={"camera_ids": [cam["id"], caja["id"]], "analyze": True})
    assert r.status_code == 200, r.text
    units = lambda: admin.get("/api/analysis").json()["plan"]["units"]  # noqa: E731
    con = units()
    assert con == pytest.approx(pool.POSE_FACTOR + 1.0, abs=0.01)  # la góndola mira posturas, la caja no
    admin.put(f"/api/cameras/{cam['id']}/analysis", json={"zones": [], "rules": {}, "gestures": {"enabled": []}})
    sin = units()
    assert sin == pytest.approx(2.0, abs=0.01) and sin < con
