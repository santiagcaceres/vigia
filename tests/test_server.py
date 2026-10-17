import json
import uuid as uuidlib
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from agent.imports import ImportJobRunner
from agent.outbox import ServerClient
from core import synthetic
from core.scorer import Scorer
from server.db import ClientReport, Event, utcnow
from server.main import Settings, create_app

ADMIN = ("admin", "clave-admin-123")
MODEL = Path(__file__).resolve().parents[1] / "models" / "personas.onnx"


@pytest.fixture
def app(tmp_path, db_url):
    st = Settings(data_dir=tmp_path / "data", database_url=db_url, admin_password=ADMIN[1], serve_public=False)
    return create_app(st)


@pytest.fixture
def admin(app):
    with TestClient(app) as c:
        assert c.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]}).status_code == 200
        yield c


def login(app, user, pw):
    c = TestClient(app)
    r = c.post("/api/login", json={"username": user, "password": pw})
    assert r.status_code == 200, r.text
    return c


def setup_client(admin, name="Kiosco Centro"):
    cid = admin.post("/api/clients", json={"name": name}).json()["id"]
    ag = admin.post("/api/agents", json={"client_id": cid, "name": f"PC {name}"}).json()
    cam = admin.post("/api/cameras", json={
        "client_id": cid, "name": "Salón", "host": "192.168.1.64", "username": "admin",
        "password": "Cl@ve#1", "channel": 3, "detector": "motion", "process_fps": synthetic.FPS}).json()
    admin.put(f"/api/cameras/{cam['id']}/analysis", json={"zones": synthetic.STORE_ZONES, "rules": {}})
    return cid, ag, cam


def agent_client(app, token) -> ServerClient:
    return ServerClient("http://testserver", token, http=TestClient(app))


def post_event(app, token, cam_id, uid=None, type="salida_sin_pago", alert=True):
    uid = uid or uuidlib.uuid4().hex
    meta = {"uuid": uid, "camera_id": cam_id, "type": type, "description": "prueba", "track_id": 1,
            "occurred_at": "2026-10-16T11:00:00-03:00", "features": {"shelf_total_s": 4.0, "x": 1}, "score": None,
            "alert": alert}
    r = TestClient(app).post("/api/agent/events", headers={"X-Agent-Token": token},
                             data={"meta": json.dumps(meta)},
                             files={"snapshot": ("s.jpg", b"\xff\xd8fakejpg", "image/jpeg")})
    return uid, r


def test_login_y_permisos(app, admin):
    assert TestClient(app).get("/api/events").status_code == 401
    assert TestClient(app).post("/api/login", json={"username": "admin", "password": "mala"}).status_code == 401
    assert admin.get("/api/me").json()["role"] == "admin"
    assert admin.get("/api/health").json()["status"] == "ok"
    assert admin.post("/api/me/password", json={"current": "mala", "new": "otra-clave-1"}).status_code == 422
    assert admin.post("/api/me/password", json={"current": ADMIN[1], "new": "otra-clave-1"}).status_code == 200
    login(app, "admin", "otra-clave-1")


def test_camaras_no_exponen_clave_y_agente_recibe_config(app, admin):
    cid, ag, cam = setup_client(admin)
    assert "password" not in cam and cam["has_password"] is True
    cfg = TestClient(app).get("/api/agent/config", headers={"X-Agent-Token": ag["token"]}).json()
    c = cfg["cameras"][0]
    assert c["connection"]["password"] == "Cl@ve#1" and c["connection"]["channel"] == 3
    assert c["analysis"]["zones"][0]["name"] == "gondola"
    assert cfg["filter_model"] is None and cfg["process_imports"] is True
    assert TestClient(app).get("/api/agent/config", headers={"X-Agent-Token": "x"}).status_code == 401


def test_zonas_invalidas_rechazadas(admin):
    _, _, cam = setup_client(admin)
    bad = {"zones": [{"name": "a", "type": "shelf", "polygon": [[0, 0], [3, 0], [1, 1]]}], "rules": {}}
    assert admin.put(f"/api/cameras/{cam['id']}/analysis", json=bad).status_code == 422


def test_evento_clip_por_url_firmada_e_idempotencia(app, admin, tmp_path):
    _, ag, cam = setup_client(admin)
    uid, r = post_event(app, ag["token"], cam["id"])
    assert r.status_code == 201, r.text
    _, r2 = post_event(app, ag["token"], cam["id"], uid=uid)
    assert r2.json()["duplicate"] is True
    sc = agent_client(app, ag["token"])
    # sin subir el archivo, el servidor no acepta el clip
    assert sc.http.post(f"/api/agent/events/{uid}/clip-done", json={"source": "nvr"}).status_code == 409
    clip = tmp_path / "c.mp4"
    clip.write_bytes(b"0" * 4096)
    sc.put_clip(uid, clip, "nvr")
    ev = admin.get("/api/events").json()["items"][0]
    assert ev["clip_source"] == "nvr" and ev["occurred_at"] == "2026-10-16T14:00:00Z"
    assert set(ev["features"]) == {"shelf_total_s"}  # variables desconocidas se descartan
    r = admin.get(ev["clip_url"], follow_redirects=False)
    assert r.status_code == 307 and "/api/_files/" in r.headers["location"]
    full = admin.get(ev["clip_url"])
    assert full.status_code == 200 and len(full.content) == 4096
    rng = admin.get(ev["clip_url"], headers={"Range": "bytes=0-99"})
    assert rng.status_code == 206 and len(rng.content) == 100
    assert admin.get(ev["snapshot_url"]).content.startswith(b"\xff\xd8")
    # la URL firmada no sirve para otro archivo ni vencida
    loc = r.headers["location"]
    assert admin.get(loc.replace(".mp4", ".jpg")).status_code == 404
    assert TestClient(app).get(ev["clip_url"]).status_code == 401  # sin sesión no hay redirección


def test_agente_no_puede_usar_camaras_de_otro_cliente(app, admin):
    _, ag1, _ = setup_client(admin, "Cliente A")
    _, _, cam2 = setup_client(admin, "Cliente B")
    _, r = post_event(app, ag1["token"], cam2["id"])
    assert r.status_code == 404


def test_cliente_solo_ve_lo_suyo_y_no_etiqueta(app, admin):
    cid_a, ag_a, cam_a = setup_client(admin, "Cliente A")
    cid_b, ag_b, cam_b = setup_client(admin, "Cliente B")
    post_event(app, ag_a["token"], cam_a["id"])
    post_event(app, ag_b["token"], cam_b["id"])
    admin.post("/api/users", json={"username": "duenoA", "password": "clave-dueno", "role": "cliente",
                                   "client_id": cid_a})
    u = login(app, "duenoA", "clave-dueno")
    data = u.get("/api/events").json()
    assert data["total"] == 1 and data["items"][0]["client_id"] == cid_a
    assert u.get("/api/events", params={"client_id": cid_b}).status_code == 403
    other = admin.get("/api/events", params={"client_id": cid_b}).json()["items"][0]
    assert u.get(f"/api/events/{other['id']}").status_code == 404
    assert u.get(other["snapshot_url"]).status_code == 404
    mine = data["items"][0]
    assert u.patch(f"/api/events/{mine['id']}", json={"label": "hurto"}).status_code == 403
    cams = u.get("/api/cameras").json()
    assert len(cams) == 1 and "host" not in cams[0]
    assert u.get("/api/imports").status_code == 403


def test_entrenador_etiqueta(app, admin):
    _, ag, cam = setup_client(admin)
    post_event(app, ag["token"], cam["id"])
    admin.post("/api/users", json={"username": "lucia", "password": "entrena-123", "role": "entrenador"})
    t = login(app, "lucia", "entrena-123")
    ev = t.get("/api/events", params={"label": "sin_etiqueta"}).json()["items"][0]
    r = t.patch(f"/api/events/{ev['id']}", json={"label": "hurto", "note": "se ve el bolsillo"})
    assert r.status_code == 200 and r.json()["label"] == "hurto"
    assert t.get("/api/events", params={"label": "sin_etiqueta"}).json()["total"] == 0
    # el empleado da de alta clientes y configura cámaras, pero no maneja usuarios ni agentes
    cid = t.post("/api/clients", json={"name": "Kiosco Norte"}).json()["id"]
    r = t.post("/api/cameras", json={"client_id": cid, "name": "Caja", "host": "100.64.0.9", "channel": 2})
    assert r.status_code == 201, r.text
    assert t.put(f"/api/cameras/{r.json()['id']}", json={**r.json(), "client_id": cid, "name": "Caja 1"}).status_code == 200
    assert t.put(f"/api/clients/{cid}", json={"name": "Kiosco Norte 2"}).status_code == 200
    assert t.delete(f"/api/cameras/{r.json()['id']}").status_code == 403
    assert t.get("/api/users").status_code == 403
    assert t.post("/api/users", json={"username": "otro", "password": "clave-1234", "role": "admin"}).status_code == 403
    assert t.post("/api/agents", json={"name": "x"}).status_code == 403
    assert t.post("/api/hpp/setup", json={}).status_code == 403


def test_varios_admins_y_gestion_de_empleados(app, admin):
    for name in ("socio", "tercero"):
        r = admin.post("/api/users", json={"username": name, "password": "clave-admin-2", "role": "admin"})
        assert r.status_code == 201
    socio = login(app, "socio", "clave-admin-2")
    emp = socio.post("/api/users", json={"username": "ana", "password": "empleada-1", "role": "entrenador"}).json()
    login(app, "ana", "empleada-1")
    # resetear la clave y desactivar a un empleado
    r = socio.put(f"/api/users/{emp['id']}", json={"username": "ana", "password": "nueva-clave-1", "role": "entrenador"})
    assert r.status_code == 200
    login(app, "ana", "nueva-clave-1")
    socio.put(f"/api/users/{emp['id']}", json={"username": "ana", "role": "entrenador", "active": False})
    assert TestClient(app).post("/api/login", json={"username": "ana", "password": "nueva-clave-1"}).status_code == 401
    assert socio.put(f"/api/users/{emp['id']}", json={"username": "socio", "role": "entrenador"}).status_code == 409
    assert len([u for u in socio.get("/api/users").json() if u["role"] == "admin"]) == 3


def upload_import(admin, cam_id, video: Path, recorded_at="2026-10-10T15:00"):
    r = admin.post("/api/imports", json={"camera_id": cam_id, "filename": "export ivms.mp4",
                                         "size": video.stat().st_size, "recorded_at": recorded_at})
    assert r.status_code == 201, r.text
    imp = r.json()
    assert imp["status"] == "subiendo"
    assert admin.post(f"/api/imports/{imp['id']}/uploaded").status_code == 409  # todavía no subió
    assert admin.put(imp["upload_url"], content=video.read_bytes(),
                     headers={"Content-Type": "video/mp4"}).status_code == 200
    assert admin.post(f"/api/imports/{imp['id']}/uploaded").json()["status"] == "en_cola"
    return imp["id"]


def test_video_importado_lo_analiza_el_agente(app, admin, tmp_path):
    cid, ag, cam = setup_client(admin)
    _, ag_other, _ = setup_client(admin, "Otro cliente")
    video = synthetic.write_video(synthetic.render("hurto") + synthetic.render("cliente_paga"), tmp_path / "v.mp4")
    iid = upload_import(admin, cam["id"], video)
    # el agente de otro cliente no lo ve
    assert agent_client(app, ag_other["token"]).next_job() is None
    runner = ImportJobRunner(agent_client(app, ag["token"]), str(MODEL), tmp_path / "work")
    assert runner.run_once() is True
    assert runner.run_once() is False  # no quedan pendientes
    imp = admin.get(f"/api/imports/{iid}").json()
    assert imp["status"] == "listo" and imp["duration_s"] > 10, imp
    evs = admin.get("/api/events", params={"import_id": iid}).json()["items"]
    assert [e["type"] for e in evs] == ["salida_sin_pago"]
    ev = evs[0]
    assert ev["source"] == "importado" and ev["clip_source"] == "video" and ev["alert"] is True
    assert ev["features"]["has_time"] == 1.0
    assert ev["occurred_at"].startswith("2026-10-10T18:00")  # 15:00 local (UTC-3) + segundos del video
    assert admin.get(ev["clip_url"]).status_code == 200
    admin.post(f"/api/imports/{iid}/incidents", json={"start_s": 2, "end_s": 13, "note": "hurto real"})
    admin.post(f"/api/imports/{iid}/incidents", json={"start_s": 200, "end_s": 210, "note": "fuera del video"})
    e = admin.get(f"/api/imports/{iid}/evaluation").json()
    assert e["incidents"] == 2 and e["detected"] == 1 and e["false_alarms"] == 0
    # volver a analizar borra los eventos anteriores y lo pone en cola
    assert admin.post(f"/api/imports/{iid}/reprocess").json()["status"] == "en_cola"
    assert admin.get("/api/events", params={"import_id": iid}).json()["total"] == 0
    assert runner.run_once() is True
    assert admin.get("/api/events", params={"import_id": iid}).json()["total"] == 1
    assert admin.delete(f"/api/imports/{iid}").status_code == 204
    assert admin.get("/api/events", params={"import_id": iid}).json()["total"] == 0


def test_video_que_no_se_puede_abrir_queda_con_error(app, admin, tmp_path):
    _, ag, cam = setup_client(admin)
    bad = tmp_path / "roto.mp4"
    bad.write_bytes(b"esto no es un video")
    iid = upload_import(admin, cam["id"], bad)
    ImportJobRunner(agent_client(app, ag["token"]), str(MODEL), tmp_path / "w").run_once()
    imp = admin.get(f"/api/imports/{iid}").json()
    assert imp["status"] == "error" and "MP4" in imp["error"]


def test_agente_sin_importaciones_no_recibe_trabajos(app, admin, tmp_path):
    cid, _, cam = setup_client(admin)
    ag2 = admin.post("/api/agents", json={"client_id": cid, "name": "Solo cámaras", "process_imports": False}).json()
    video = synthetic.write_video(synthetic.render("hurto"), tmp_path / "v.mp4")
    upload_import(admin, cam["id"], video)
    assert agent_client(app, ag2["token"]).next_job() is None


def event_with_clip(app, admin, ag, cam, tmp_path, name="c.mp4"):
    uid, r = post_event(app, ag["token"], cam["id"])
    assert r.status_code == 201, r.text
    clip = tmp_path / name
    clip.write_bytes(b"0" * 2048)
    agent_client(app, ag["token"]).put_clip(uid, clip, "nvr")
    return next(e for e in admin.get("/api/events").json()["items"] if e["uuid"] == uid)


def token_of(link: str) -> str:
    return link.split("t=", 1)[1]


def test_aviso_al_cliente_flujo_completo(app, admin, tmp_path):
    _, ag, cam = setup_client(admin)
    ev = event_with_clip(app, admin, ag, cam, tmp_path)
    # crear el aviso (solo portal, sin canales externos configurados)
    r = admin.post(f"/api/events/{ev['id']}/report", json={"message": "Mirá esto", "channels": ["portal"]})
    assert r.status_code == 201, r.text
    rep = r.json()
    assert rep["status"] == "enviado" and rep["delivery"]["portal"]["ok"] is True
    assert "aviso.html?t=" in rep["link"]
    token = token_of(rep["link"])

    pub = TestClient(app)  # sin login: página pública por token
    view = pub.get(f"/api/r/{token}")
    assert view.status_code == 200
    body = view.json()
    assert body["message"] == "Mirá esto" and body["has_clip"] is True and body["client_name"] == "Kiosco Centro"
    # al abrirlo queda marcado como visto
    assert admin.get("/api/reports").json()["items"][0]["viewed_at"] is not None
    # el clip público redirige a la URL firmada
    assert pub.get(f"/api/r/{token}/clip", follow_redirects=False).status_code == 307
    assert pub.get(f"/api/r/{token}/clip").content == b"0" * 2048

    # revocar: el link deja de servir
    assert admin.delete(f"/api/reports/{rep['id']}").status_code == 204
    assert pub.get(f"/api/r/{token}").status_code == 404


def test_aviso_requiere_clip_permisos_y_aislamiento(app, admin, tmp_path):
    cid_a, ag_a, cam_a = setup_client(admin, "Cliente A")
    cid_b, ag_b, cam_b = setup_client(admin, "Cliente B")
    # evento sin clip: no se puede avisar
    uid, _ = post_event(app, ag_a["token"], cam_a["id"])
    ev_sin = next(e for e in admin.get("/api/events").json()["items"] if e["uuid"] == uid)
    assert admin.post(f"/api/events/{ev_sin['id']}/report", json={}).status_code == 409

    ev_a = event_with_clip(app, admin, ag_a, cam_a, tmp_path, "a.mp4")
    ev_b = event_with_clip(app, admin, ag_b, cam_b, tmp_path, "b.mp4")
    admin.post(f"/api/events/{ev_a['id']}/report", json={})
    admin.post(f"/api/events/{ev_b['id']}/report", json={})

    # el entrenador puede enviar; el cliente no
    admin.post("/api/users", json={"username": "lucia", "password": "entrena-123", "role": "entrenador"})
    t = login(app, "lucia", "entrena-123")
    assert t.post(f"/api/events/{ev_a['id']}/report", json={}).status_code == 201

    admin.post("/api/users", json={"username": "duenoA", "password": "clave-dueno", "role": "cliente",
                                   "client_id": cid_a})
    u = login(app, "duenoA", "clave-dueno")
    assert u.post(f"/api/events/{ev_a['id']}/report", json={}).status_code == 403
    # el cliente solo ve sus propios avisos
    items = u.get("/api/reports").json()["items"]
    assert items and all(i["client_id"] == cid_a for i in items)
    assert "delivery" not in items[0]  # no ve por qué canal se envió
    assert u.get("/api/reports", params={"client_id": cid_b}).status_code == 403


def test_agente_central_atiende_camaras_de_varios_clientes(app, admin, tmp_path):
    cid_a, ag_a, cam_a = setup_client(admin, "Cliente A")
    cid_b, ag_b, cam_b = setup_client(admin, "Cliente B")
    central = admin.post("/api/agents", json={"client_id": None, "name": "PC central"}).json()
    assert central["central"] is True and central["client_id"] is None
    # sin cámaras asignadas no ve nada
    cfg = TestClient(app).get("/api/agent/config", headers={"X-Agent-Token": central["token"]}).json()
    assert cfg["central"] is True and cfg["cameras"] == []
    # se le asignan una cámara de A y una de B (un agente central puede tomar cámaras de cualquier cliente)
    for cam, cid in ((cam_a, cid_a), (cam_b, cid_b)):
        body = {"client_id": cid, "agent_id": central["id"], "name": cam["name"], "host": cam["host"],
                "channel": cam["channel"], "detector": "motion", "process_fps": synthetic.FPS}
        assert admin.put(f"/api/cameras/{cam['id']}", json=body).status_code == 200, body
    cfg = TestClient(app).get("/api/agent/config", headers={"X-Agent-Token": central["token"]}).json()
    assert {c["client_id"] for c in cfg["cameras"]} == {cid_a, cid_b}
    assert {c["client_name"] for c in cfg["cameras"]} == {"Cliente A", "Cliente B"}
    assert all("filter_model" in c for c in cfg["cameras"])
    # el agente del local A ya no ve su cámara (quedó asignada al central)
    cfg_a = TestClient(app).get("/api/agent/config", headers={"X-Agent-Token": ag_a["token"]}).json()
    assert cfg_a["cameras"] == []
    # los eventos del central caen en el cliente correcto
    _, r = post_event(app, central["token"], cam_b["id"])
    assert r.status_code == 201
    ev = admin.get("/api/events").json()["items"][0]
    assert ev["client_id"] == cid_b and ev["client_name"] == "Cliente B"
    # un agente de local no puede usar la cámara que ahora es del central
    _, r = post_event(app, ag_b["token"], cam_b["id"])
    assert r.status_code == 404
    # un agente de local no puede asignarse a cámaras de otro cliente
    bad = {"client_id": cid_a, "agent_id": ag_b["id"], "name": "x", "host": "1.2.3.4"}
    assert admin.post("/api/cameras", json=bad).status_code == 422
    # videos importados: el central toma los de sus cámaras asignadas
    video = synthetic.write_video(synthetic.render("hurto"), tmp_path / "v.mp4")
    iid = upload_import(admin, cam_a["id"], video)
    assert agent_client(app, ag_a["token"]).next_job() is None
    job = agent_client(app, central["token"]).next_job()
    assert job and job["import_id"] == iid


def test_link_de_aviso_vence(app, admin, tmp_path):
    _, ag, cam = setup_client(admin)
    ev = event_with_clip(app, admin, ag, cam, tmp_path)
    rep = admin.post(f"/api/events/{ev['id']}/report", json={}).json()
    token = token_of(rep["link"])
    with app.state.Session() as s:
        r = s.query(ClientReport).first()
        r.expires_at = utcnow() - timedelta(minutes=1)
        s.commit()
    assert TestClient(app).get(f"/api/r/{token}").status_code == 410


def test_entrenamiento_activacion_y_modelo_en_agente(app, admin):
    cid, ag, cam = setup_client(admin)
    rng = np.random.default_rng(1)
    with app.state.Session() as s:
        for i in range(60):
            real = i % 2 == 0
            feats = {"shelf_total_s": float(rng.normal(8 if real else 2, 1.5)),
                     "checkout_total_s": float(rng.normal(0 if real else 4, 1.0)),
                     "path_len": float(abs(rng.normal(1, 0.3)))}
            s.add(Event(uuid=uuidlib.uuid4().hex, client_id=cid, camera_id=cam["id"], type="salida_sin_pago",
                        severity="alta", description="x", features=feats,
                        label="hurto" if real else "normal"))
        s.commit()
    r = admin.post("/api/training/run", json={"client_id": cid, "target_recall": 0.9})
    assert r.status_code == 201, r.text
    m = r.json()
    assert m["metrics"]["recall"] >= 0.9 and m["metrics"]["falsas_alarmas_eliminadas"] > 0.5
    admin.post(f"/api/models/{m['id']}/activate")
    cfg = TestClient(app).get("/api/agent/config", headers={"X-Agent-Token": ag["token"]}).json()
    scorer = Scorer.maybe(cfg["filter_model"])
    assert scorer is not None and scorer.id == m["id"]
    hi = scorer.score("salida_sin_pago", {"shelf_total_s": 9, "checkout_total_s": 0, "path_len": 1})
    lo = scorer.score("salida_sin_pago", {"shelf_total_s": 1, "checkout_total_s": 5, "path_len": 1})
    assert hi > scorer.threshold > lo
    assert admin.get("/api/training/export.csv").text.count("\n") == 61


def test_entrenamiento_sin_datos_suficientes(admin):
    setup_client(admin)
    r = admin.post("/api/training/run", json={})
    assert r.status_code == 422 and "Faltan etiquetas" in r.json()["detail"]


def test_limpieza_programada(app, admin):
    from datetime import timedelta

    from server.db import utcnow

    _, ag, cam = setup_client(admin)
    old_unlabeled, _ = post_event(app, ag["token"], cam["id"])
    old_labeled, _ = post_event(app, ag["token"], cam["id"])
    recent, _ = post_event(app, ag["token"], cam["id"])
    with app.state.Session() as s:
        for uid in (old_unlabeled, old_labeled):
            e = s.query(Event).filter_by(uuid=uid).one()
            e.received_at = utcnow() - timedelta(days=90)
            if uid == old_labeled:
                e.label = "hurto"
        s.commit()
    assert TestClient(app).get("/api/cron/cleanup").status_code == 401
    app.state.settings.cron_secret = "s3cr3t"
    r = TestClient(app).get("/api/cron/cleanup", headers={"Authorization": "Bearer s3cr3t"})
    assert r.json()["deleted"] == 1 and r.json()["media_removed"] == 1
    items = {e["uuid"]: e for e in admin.get("/api/events").json()["items"]}
    assert old_unlabeled not in items
    assert items[old_labeled]["label"] == "hurto" and items[old_labeled]["snapshot_url"] is None
    assert items[recent]["snapshot_url"]


def test_falta_configuracion_en_vercel(monkeypatch):
    for k in ("DATABASE_URL", "SECRET_KEY", "SUPABASE_URL", "SUPABASE_SECRET_KEY", "SUPABASE_SERVICE_ROLE_KEY",
              "ADMIN_PASSWORD"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.setenv("DATABASE_URL", "postgresql://x")
    r = TestClient(create_app()).get("/api/health")
    assert r.status_code == 503
    assert "SECRET_KEY" in r.json()["detail"] and "DATABASE_URL" not in r.json()["detail"]
