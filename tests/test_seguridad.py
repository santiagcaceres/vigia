"""Seguridad (pedido del dueño, 3 nov: «que no nos tiren el sistema ni nos roben datos»). Cada prueba es un ataque que
encontró la auditoría y que ahora se frena. Todo contra un servidor local de prueba, nunca contra producción."""

import io
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from server import hikconnect, probe, security
from server.db import LoginFailure, User
from server.main import Settings, create_app, csv_cell

ADMIN = ("admin", "clave-admin-123")


@pytest.fixture
def app(tmp_path, db_url):
    return create_app(Settings(data_dir=tmp_path / "data", database_url=db_url, admin_password=ADMIN[1],
                               serve_public=False))


def login(app, user, pwd) -> TestClient:
    c = TestClient(app)
    r = c.post("/api/login", json={"username": user, "password": pwd})
    assert r.status_code == 200, r.text
    return c


def setup(app):
    """Un admin, dos clientes con una cámara cada uno y un operador con su PC mirando la cámara del cliente A."""
    a = login(app, *ADMIN)
    ca = a.post("/api/clients", json={"name": "Cliente A"}).json()["id"]
    cb = a.post("/api/clients", json={"name": "Cliente B"}).json()["id"]
    cam = lambda cid, n: a.post("/api/cameras", json={"client_id": cid, "name": n, "brand": "ip",  # noqa: E731
                                                      "host": "dvr", "password": "CLAVE-DVR"}).json()
    cam_a, cam_b = cam(ca, "Caja A"), cam(cb, "Caja B")
    oper = a.post("/api/users", json={"username": "oper1", "password": "clave-oper-123", "role": "entrenador"}).json()
    o = login(app, "oper1", "clave-oper-123")
    tok = {"X-Agent-Token": o.post("/api/agent/session").json()["token"]}
    assert o.put("/api/agent/watch", json={"camera_ids": [cam_a["id"]]}, headers=tok).status_code == 200
    return a, o, tok, cam_a, cam_b, oper, (ca, cb)


# ------------------------------------------------------------------ permisos
def test_una_pc_no_escribe_el_estado_de_camaras_ajenas_ni_mete_html(app):
    a, o, tok, cam_a, cam_b, *_ = setup(app)
    evil = '<img src=x onerror="fetch(\'/api/users\')">'
    r = TestClient(app).post("/api/agent/status", headers=tok, json={
        "cameras": [{"camera_id": cam_b["id"], "state": "en_linea", "fps": evil},
                    {"camera_id": cam_a["id"], "state": "en_linea", "fps": 5.0, "raro": {"x": [1, 2]}, "X": 1},
                    {"state": "sin camera_id"}],
        "outbox_pending": evil, "live": [cam_b["id"], cam_a["id"]]})
    assert r.status_code == 200
    cams = {c["id"]: c for c in a.get("/api/cameras").json()}  # (antes un estado sin camera_id daba 500 acá)
    assert cams[cam_b["id"]]["status"] is None  # la de otro cliente no se toca
    st = cams[cam_a["id"]]["status"]
    assert st["fps"] == 5.0 and "raro" not in st and "X" not in st
    pcs = [x for x in a.get("/api/agents").json() if x.get("status")]
    assert all(p["status"].get("outbox_pending") is None for p in pcs)
    # un estado con forma rara no tira abajo nada
    assert TestClient(app).post("/api/agent/status", headers=tok, json={"cameras": {"a": 1}}).status_code == 200
    assert a.get("/api/cameras").status_code == 200


def test_la_pc_de_un_usuario_dado_de_baja_deja_de_servir(app):
    a, o, tok, cam_a, cam_b, oper, _ = setup(app)
    assert TestClient(app).get("/api/agent/config", headers=tok).status_code == 200
    body = {"username": "oper1", "role": "entrenador", "active": False}
    assert a.put(f"/api/users/{oper['id']}", json=body).status_code == 200
    assert TestClient(app).get("/api/agent/config", headers=tok).status_code == 401  # ni claves de cámaras
    assert o.get("/api/me").status_code == 401
    # vuelve a estar activo: el token viejo no revive (tiene que entrar de nuevo)
    a.put(f"/api/users/{oper['id']}", json={**body, "active": True})
    assert TestClient(app).get("/api/agent/config", headers=tok).status_code == 401
    o = login(app, "oper1", "clave-oper-123")
    tok2 = {"X-Agent-Token": o.post("/api/agent/session").json()["token"]}
    assert TestClient(app).get("/api/agent/config", headers=tok2).status_code == 200
    # cambiarle la clave también la corta
    a.put(f"/api/users/{oper['id']}", json={**body, "active": True, "password": "otra-clave-larga-9"})
    assert TestClient(app).get("/api/agent/config", headers=tok2).status_code == 401


def test_el_cliente_no_ve_la_nota_interna(app):
    a, o, tok, cam_a, *_, (ca, cb) = setup(app)
    meta = {"uuid": "ev1", "camera_id": cam_a["id"], "type": "zona_restringida", "description": "x"}
    r = TestClient(app).post("/api/agent/events", headers=tok, data={"meta": json.dumps(meta)},
                             files={"snapshot": ("s.jpg", b"\xff\xd8\xff", "image/jpeg")})
    eid = r.json()["id"]
    a.patch(f"/api/events/{eid}", json={"label": "falso_positivo", "note": "es la encargada, no avisar"})
    a.post("/api/users", json={"username": "dueno-a", "password": "clave-dueno-a1", "role": "cliente",
                               "client_id": ca})
    u = login(app, "dueno-a", "clave-dueno-a1")
    assert u.get(f"/api/events/{eid}").json()["note"] is None
    assert all(e["note"] is None for e in u.get("/api/events").json()["items"])
    assert a.get(f"/api/events/{eid}").json()["note"] == "es la encargada, no avisar"


def test_cerrar_sesion_en_la_app_corta_la_sesion(app):
    token = TestClient(app).post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]}).json()["token"]
    h = {"Authorization": f"Bearer {token}"}
    assert TestClient(app).get("/api/me", headers=h).status_code == 200
    assert TestClient(app).post("/api/logout", headers=h).status_code == 200
    assert TestClient(app).get("/api/me", headers=h).status_code == 401


# ------------------------------------------------------------------ claves
def test_freno_a_quien_prueba_claves(app):
    c = TestClient(app)
    for _ in range(5):
        assert c.post("/api/login", json={"username": "admin", "password": "no-es"}).status_code == 401
    r = c.post("/api/login", json={"username": "admin", "password": ADMIN[1]})
    assert r.status_code == 429 and "minuto" in r.json()["detail"]  # ni con la clave buena, por un rato
    # otro usuario desde la misma IP sí puede (el freno de la IP sola es más alto)
    assert c.post("/api/login", json={"username": "otro", "password": "x"}).status_code == 401
    # pasado el bloqueo entra y se borra lo de esa IP y usuario
    with app.state.Session() as s:
        for row in s.query(LoginFailure).filter(LoginFailure.locked_until.isnot(None)):
            row.locked_until = datetime.utcnow() - timedelta(seconds=1)
        s.commit()
    assert c.post("/api/login", json={"username": "admin", "password": ADMIN[1]}).status_code == 200
    # la PC central (usuario y clave de admin) tiene el mismo freno
    for _ in range(5):
        c.post("/api/agent/login", json={"username": "admin", "password": "mal", "pc_name": "x"})
    assert c.post("/api/agent/login", json={"username": "admin", "password": ADMIN[1],
                                            "pc_name": "x"}).status_code == 429


def test_claves_debiles_y_hash_viejo(app):
    a = login(app, *ADMIN)
    for pw in ("12345678", "aaaaaaaa", "contraseña", "pepito123"):
        r = a.post("/api/users", json={"username": "pepito", "password": pw, "role": "entrenador"})
        assert r.status_code == 422, pw
    assert a.post("/api/users", json={"username": "pepito", "password": "caballo bateria grapa",
                                      "role": "entrenador"}).status_code == 201
    # una clave guardada con el formato de antes (scrypt N=2^14) sigue entrando y se rehace con lo de ahora
    import base64
    import hashlib
    import os
    salt = os.urandom(16)
    old = "scrypt$" + base64.b64encode(salt).decode() + "$" + base64.b64encode(
        hashlib.scrypt(b"clave-vieja-123", salt=salt, n=2 ** 14, r=8, p=1, dklen=32)).decode()
    with app.state.Session() as s:
        s.add(User(username="viejo", password_hash=old, role="entrenador", active=True))
        s.commit()
    login(app, "viejo", "clave-vieja-123")
    with app.state.Session() as s:
        new = s.query(User).filter_by(username="viejo").one().password_hash
    assert new.count("$") == 5 and not security.needs_rehash(new)
    login(app, "viejo", "clave-vieja-123")


# ------------------------------------------------------------------ entradas
def test_hik_connect_solo_a_hik_connect(app):
    a = login(app, *ADMIN)
    cid = a.post("/api/clients", json={"name": "Hik"}).json()["id"]
    r = a.put(f"/api/clients/{cid}/hikconnect", json={"hc_app_key": "APP", "hc_secret": "SECRETO",
                                                      "hc_api_url": "https://atacante.example.com"})
    assert r.status_code == 422
    r = a.put(f"/api/clients/{cid}/hikconnect", json={"hc_app_key": "APP", "hc_secret": "SECRETO",
                                                      "hc_api_url": "https://isa.hikcentralconnect.com/portal"})
    assert r.status_code == 200
    # el caché del token no se comparte entre clientes con el mismo AppKey y otra clave
    one, two = hikconnect.HikConnect("APP", "S1"), hikconnect.HikConnect("APP", "S2")
    assert one._key != two._key


def test_hosts_y_webhooks_validos(app):
    a = login(app, *ADMIN)
    cid = a.post("/api/clients", json={"name": "Host"}).json()["id"]
    body = {"client_id": cid, "name": "C", "brand": "hikvision", "password": "x"}
    for bad in ("otro-host:80/ruta?x#", "user@dvr", "http://dvr"):
        assert a.post("/api/cameras", json={**body, "host": bad}).status_code == 422, bad
    assert a.post("/api/cameras", json={**body, "host": "minas.ddns.net"}).status_code == 201
    for bad in ("http://ejemplo.com/hook", "https://127.0.0.1/x", "https://user:pw@ejemplo.com/x",
                "https://169.254.169.254/latest"):
        r = a.put(f"/api/clients/{cid}", json={"name": "Host", "webhook_url": bad})
        assert r.status_code == 422, bad
    # el servidor no prueba direcciones internas («Probar conexión»)
    assert probe.reachable_public("192.168.1.6") and probe.reachable_public("127.0.0.1")
    assert probe.probe("hikvision", "10.0.0.1", 80, "u", "p")["ok"] is False


def test_el_excel_no_ejecuta_formulas():
    assert csv_cell('=HYPERLINK("https://x")') == '\'=HYPERLINK("https://x")'
    assert csv_cell("@SUM(A1)").startswith("'") and csv_cell("+1").startswith("'")
    assert csv_cell("-1,5") == "-1,5" and csv_cell("12") == "12" and csv_cell("a;b") == "a,b"


def test_datos_raros_de_una_pc_no_rompen_nada(app):
    a, o, tok, cam_a, *_ = setup(app)
    # Los cubos van en hora del local: con la hora UTC, de 21 a 24 (Montevideo) el cubo caía en «mañana».
    now = datetime.now(ZoneInfo("America/Montevideo")).replace(tzinfo=None, minute=0, second=0, microsecond=0)
    r = TestClient(app).post("/api/agent/metrics", headers=tok, json={"buckets": [
        {"camera_id": cam_a["id"], "bucket_start": "9999-12-31T23:00:00", "data": {"people": 5}},
        {"camera_id": cam_a["id"], "bucket_start": now.isoformat(),
         "data": {"people": "abc", "entries": -9, "frames": 10, "zones": {f"z{i}": {"visits": 1} for i in range(500)}}}]})
    assert r.status_code == 200 and r.json()["saved"] == 1  # el del año 9999 se saltea
    m = a.get("/api/metrics", params={"client_id": cam_a["client_id"], "days": 1}).json()["total"]
    assert m["people"] == 0 and m["entries"] == 0 and len(m["zones"]) == 64
    # un evento con datos de cualquier forma se guarda limpio (antes daba 500)
    meta = {"uuid": "raro1", "camera_id": cam_a["id"], "type": "zona_restringida", "zone": "x" * 5000,
            "score": "alto", "bbox": "nada", "frame_size": [1, 2, 3], "features": ["no"]}
    r = TestClient(app).post("/api/agent/events", headers=tok, data={"meta": json.dumps(meta)},
                             files={"snapshot": ("s.jpg", io.BytesIO(b"\xff\xd8\xff"), "image/jpeg")})
    assert r.status_code == 201
    ev = a.get(f"/api/events/{r.json()['id']}").json()
    assert len(ev["zone"]) == 120 and ev["score"] is None and ev["features"] == {}


def test_una_maquina_no_dice_que_aguanta_1000_camaras(app):
    a = login(app, *ADMIN)
    key = a.post("/api/analysis/pool-key", json={}).json()["key"]
    c = TestClient(app)
    r = c.post("/api/agent/enroll", json={"key": key, "name": "falsa", "capacity": 1000})
    assert r.status_code == 200
    with app.state.Session() as s:
        from server.db import Agent
        ag = s.query(Agent).filter_by(name="falsa").one()
    # la capacidad que cuenta el reparto es a lo sumo el doble de «Cámaras por máquina»
    over = a.get("/api/analysis").json()
    machine = next(m for m in over.get("machines", []) if "falsa" in (m.get("names") or [m.get("name")]))
    assert machine["capacity"] <= 60
    # clave de máquinas mal: con freno
    for _ in range(5):
        assert c.post("/api/agent/enroll", json={"key": "x" * 20, "name": "otra"}).status_code == 401
    assert c.post("/api/agent/enroll", json={"key": key, "name": "otra"}).status_code == 429
    assert ag.id


# ------------------------------------------------------------------ web
def test_cabeceras_de_seguridad(tmp_path, db_url):
    app = create_app(Settings(data_dir=tmp_path / "d2", database_url=db_url, admin_password=ADMIN[1]))
    c = TestClient(app)
    r = c.get("/index.html")
    assert "frame-ancestors 'none'" in r.headers["content-security-policy"]
    assert "'unsafe-inline'" not in r.headers["content-security-policy"].split("script-src")[1].split(";")[0]
    assert r.headers["x-frame-options"] == "DENY" and r.headers["x-content-type-options"] == "nosniff"
    assert c.get("/api/health").headers["cache-control"] == "no-store"
    # ningún HTML de la web tiene scripts en línea (la política no los deja correr)
    from pathlib import Path
    for page in Path("public").glob("*.html"):
        html = page.read_text()
        assert "<script>" not in html and "javascript:" not in html, page.name


def test_la_politica_de_contenido_es_la_misma_en_vercel_y_sin_nada_externo():
    """Vercel sirve public/ con la cabecera de vercel.json y el servidor local con la de server/main.py: tienen que ser
    la misma. Sin la pestaña «Prueba» (9 nov) no queda nada externo: ni onnxruntime-web de jsdelivr, ni WebAssembly,
    ni workers, ni «blob:». Siguen las imágenes y los clips firmados de Supabase (lo que ve el cliente)."""
    import json
    from pathlib import Path

    from server.main import CSP
    root = Path(__file__).resolve().parents[1]
    conf = json.loads((root / "vercel.json").read_text(encoding="utf-8"))
    values = [h["value"] for block in conf["headers"] for h in block["headers"]
              if h["key"] == "Content-Security-Policy"]
    assert values == [CSP]
    for gone in ("jsdelivr", "wasm-unsafe-eval", "worker-src", "blob:", "unsafe-eval"):
        assert gone not in CSP, gone
    rules = dict(part.strip().split(" ", 1) for part in CSP.split(";"))
    assert rules["script-src"] == "'self'" and rules["frame-ancestors"] == "'none'"
    assert "https://*.supabase.co" in rules["img-src"] and "https://*.supabase.co" in rules["media-src"]
    # y la web ya no carga nada de afuera (si se suma algo externo, va en las dos políticas)
    for page in (root / "public").glob("*.*"):
        if page.suffix in (".js", ".html", ".css"):
            text = page.read_text(encoding="utf-8")
            assert "jsdelivr" not in text and "onnxruntime" not in text and "new Worker" not in text, page.name
            assert "createObjectURL" not in text, page.name  # (sin «blob:» en la política no se vería)


def test_webhook_de_hpp_sin_la_clave_no_firma(app):
    r = TestClient(app).get("/api/hpp/webhook", headers={"X-Hook-Batch-Id": "a", "X-Hook-Timestamp": "1"})
    assert r.status_code == 404 and "x-hook-signature" not in r.headers


# ------------------------------------------------------------------ secretos (tanda 2)
def test_queda_registro_de_quien_baja_claves_y_hay_tope(app, monkeypatch):
    import server.main as sm
    a, o, tok, cam_a, cam_b, oper, _ = setup(app)
    r = TestClient(app).post("/api/agent/view", headers=tok, json={"camera_ids": [cam_a["id"], cam_b["id"]]})
    assert r.status_code == 200
    from server.db import CredentialAccess
    with app.state.Session() as s:
        rows = s.query(CredentialAccess).all()
    assert {r.camera_id for r in rows} >= {cam_a["id"], cam_b["id"]} and all(r.user_id == oper["id"] for r in rows)
    # quien pide las de muchísimas cámaras en una hora queda frenado
    monkeypatch.setattr(sm, "MAX_CREDENTIAL_CAMERAS", 1)
    r = TestClient(app).post("/api/agent/view", headers=tok, json={"camera_ids": [cam_a["id"], cam_b["id"]]})
    assert r.status_code == 429


def test_mi_clave_corta_el_token_de_la_pc(app):
    a, o, tok, *_ = setup(app)
    assert TestClient(app).get("/api/agent/config", headers=tok).status_code == 200
    r = o.post("/api/me/password", json={"current": "clave-oper-123", "new": "otra-clave-larga-2026"})
    assert r.status_code == 200
    assert TestClient(app).get("/api/agent/config", headers=tok).status_code == 401
    assert o.get("/api/me").status_code == 200  # la sesión con la que se cambió sigue


def test_las_claves_no_quedan_a_la_vista_en_los_mensajes():
    from core.clips import redact
    msg = redact("No se pudo abrir rtsp://admin:Abc@2024@dvr.ddns.net:554/Streaming/Channels/101")
    assert "2024" not in msg and "Abc" not in msg and "dvr.ddns.net:554" in msg


def test_la_limpieza_borra_fotos_viejas_de_entrenamiento(app, tmp_path):
    import os
    a, o, tok, cam_a, *_ = setup(app)
    meta = {"uuid": "viejo1", "camera_id": cam_a["id"], "type": "zona_restringida", "source": "importado",
            "description": "x"}
    from server.db import Event, VideoImport
    with app.state.Session() as s:
        s.add(Event(uuid="ajuste1", client_id=cam_a["client_id"], camera_id=cam_a["id"], source="ajuste",
                    type="zona_restringida", severity="alta", label="hurto", snapshot_path="c1/x.jpg", description="",
                    received_at=datetime.utcnow() - timedelta(days=400), alert=False))
        s.add(VideoImport(client_id=cam_a["client_id"], camera_id=cam_a["id"], filename="v.mp4",
                          storage_path="imports/1/v.mp4", status="listo",
                          created_at=datetime.utcnow() - timedelta(days=400)))
        s.commit()
    os.environ["CRON_SECRET"] = "x" * 20
    app.state.settings.cron_secret = "x" * 20
    r = TestClient(app).get("/api/cron/cleanup", headers={"Authorization": "Bearer " + "x" * 20})
    assert r.status_code == 200, r.text
    with app.state.Session() as s:
        ev = s.query(Event).filter_by(uuid="ajuste1").one()
        assert ev.snapshot_path is None and ev.label == "hurto"  # se va la foto, queda lo que entrena
        assert s.query(VideoImport).one().storage_path == ""
    assert meta
