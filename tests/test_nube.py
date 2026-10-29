"""Máquinas virtuales de Google Cloud (server/nube.py y /api/cloud): el token se pide con un JWT firmado por la cuenta
de servicio, se listan las instancias de todas las zonas, se prenden/apagan/reinician, y la clave queda cifrada en
el servidor (nunca vuelve a la app). Google se simula con httpx.MockTransport."""

import base64
import json

import httpx
import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from fastapi.testclient import TestClient

from server import nube
from server.db import AppSetting
from server.main import Settings, create_app

ADMIN = ("admin", "clave-admin-123")

_PRIVATE = rsa.generate_private_key(public_exponent=65537, key_size=2048)
PEM = _PRIVATE.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                             serialization.NoEncryption()).decode()
KEY = {"type": "service_account", "project_id": "vigia-512822", "private_key_id": "abc123",
       "private_key": PEM, "client_email": "analitica-app@vigia-512822.iam.gserviceaccount.com"}

INSTANCE = {
    "name": "vigia-analisis-1", "status": "RUNNING",
    "zone": "https://www.googleapis.com/compute/v1/projects/vigia-512822/zones/southamerica-east1-c",
    "machineType": "https://www.googleapis.com/compute/v1/projects/vigia-512822/zones/southamerica-east1-c/"
                   "machineTypes/e2-standard-4",
    "scheduling": {"provisioningModel": "STANDARD"}, "creationTimestamp": "2026-10-26T10:00:00.000-07:00",
    "lastStartTimestamp": "2026-10-29T15:44:00.000-07:00",
    "networkInterfaces": [{"networkIP": "10.158.0.2", "accessConfigs": [{"natIP": "34.0.0.1"}]}]}
SPOT = {**INSTANCE, "name": "prueba-t2d", "status": "TERMINATED", "scheduling": {"provisioningModel": "SPOT"},
        "zone": INSTANCE["zone"].replace("-c", "-a"), "lastStopTimestamp": "2026-10-27T20:00:00.000-07:00"}


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


class FakeGoogle:
    """Google de prueba: token, lista agregada (en dos páginas) y acciones."""

    def __init__(self, status_list=200, deny=False):
        self.calls: list[tuple[str, str]] = []
        self.tokens = 0
        self.status_list, self.deny = status_list, deny

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.calls.append((request.method, url))
        if url == nube.TOKEN_URL:
            form = dict(x.split("=", 1) for x in request.content.decode().split("&"))
            jwt = form["assertion"]
            head, claims, sig = jwt.split(".")
            _PRIVATE.public_key().verify(_unb64(sig), f"{head}.{claims}".encode(), padding.PKCS1v15(), hashes.SHA256())
            data = json.loads(_unb64(claims))
            assert data["iss"] == KEY["client_email"] and data["aud"] == nube.TOKEN_URL
            assert data["scope"] == nube.SCOPE and data["exp"] - data["iat"] == 3600
            self.tokens += 1
            return httpx.Response(200, json={"access_token": "tok-1", "expires_in": 3599})
        assert request.headers["Authorization"] == "Bearer tok-1"
        if self.deny:
            return httpx.Response(403, json={"error": {"message": "Required 'compute.instances.list' permission"}})
        if "/aggregated/instances" in url:
            if self.status_list != 200:
                return httpx.Response(self.status_list, json={"error": {"message": "boom"}})
            if "pageToken=p2" in url:
                return httpx.Response(200, json={"items": {"zones/southamerica-east1-a": {"instances": [SPOT]}}})
            return httpx.Response(200, json={"items": {"zones/southamerica-east1-c": {"instances": [INSTANCE]},
                                                       "zones/us-east1-b": {"warning": {"code": "NO_RESULTS"}}},
                                             "nextPageToken": "p2"})
        if request.method == "POST" and url.endswith(("/start", "/stop", "/reset")):
            if "/no-existe/" in url:
                return httpx.Response(404, json={"error": {"message": "not found"}})
            return httpx.Response(200, json={"name": "operation-1", "status": "RUNNING"})
        return httpx.Response(500)


@pytest.fixture
def app(tmp_path, db_url):
    return create_app(Settings(data_dir=tmp_path / "data", database_url=db_url, admin_password=ADMIN[1],
                               serve_public=False))


@pytest.fixture
def admin(app):
    with TestClient(app) as c:
        assert c.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]}).status_code == 200
        yield c


def login(app, user, pw):
    c = TestClient(app)
    assert c.post("/api/login", json={"username": user, "password": pw}).status_code == 200
    return c


@pytest.fixture(autouse=True)
def _clear_tokens():
    nube._tokens.clear()
    yield
    nube._tokens.clear()


def _gc(fake: FakeGoogle) -> nube.GoogleCloud:
    return nube.GoogleCloud(KEY, http=httpx.Client(transport=httpx.MockTransport(fake)))


def test_lista_las_maquinas_de_todas_las_zonas_con_un_token_firmado():
    fake = FakeGoogle()
    gc = _gc(fake)
    vms = gc.instances()
    assert [v["name"] for v in vms] == ["prueba-t2d", "vigia-analisis-1"]
    v = vms[1]
    assert v == {"name": "vigia-analisis-1", "zone": "southamerica-east1-c", "status": "RUNNING",
                 "machine_type": "e2-standard-4", "spot": False, "created": INSTANCE["creationTimestamp"],
                 "last_start": INSTANCE["lastStartTimestamp"], "last_stop": None, "last_suspend": None}
    assert vms[0]["spot"] is True and vms[0]["status"] == "TERMINATED"
    assert "34.0.0.1" not in json.dumps(vms)  # sin IPs: la app no las necesita
    gc.instances()
    assert fake.tokens == 1  # el token se reutiliza mientras no venza


def test_prende_apaga_y_reinicia():
    fake = FakeGoogle()
    gc = _gc(fake)
    assert gc.action("southamerica-east1-c", "vigia-analisis-1", "reset") == {"operation": "operation-1",
                                                                             "status": "RUNNING"}
    assert fake.calls[-1] == ("POST", f"{nube.API}/projects/vigia-512822/zones/southamerica-east1-c/instances/"
                                      "vigia-analisis-1/reset")
    with pytest.raises(nube.CloudError) as e:
        gc.action("southamerica-east1-c", "no-existe", "start")
    assert e.value.status == 404
    for zone, name, action in (("zona/../x", "vm", "start"), ("southamerica-east1-c", "VM Mayus", "stop"),
                               ("southamerica-east1-c", "vm", "delete")):
        with pytest.raises(nube.CloudError) as e:
            gc.action(zone, name, action)
        assert e.value.status == 422


def test_sin_permiso_lo_dice_en_castellano():
    gc = _gc(FakeGoogle(deny=True))
    with pytest.raises(nube.CloudError) as e:
        gc.instances()
    assert e.value.status == 403 and "compute.instances.list, .get, .start" in str(e.value)


def test_clave_invalida():
    with pytest.raises(ValueError, match="JSON"):
        nube.parse_key("hola")
    with pytest.raises(ValueError, match="cuenta de servicio"):
        nube.parse_key(json.dumps({"type": "authorized_user"}))
    with pytest.raises(ValueError, match="client_email"):
        nube.parse_key(json.dumps({**KEY, "client_email": ""}))
    assert nube.parse_key(json.dumps(KEY))["project_id"] == "vigia-512822"


def test_api_de_la_nube(app, admin, monkeypatch):
    fake = FakeGoogle()
    real = nube.GoogleCloud
    monkeypatch.setattr(nube, "GoogleCloud",
                        lambda key, http=None, fresh=False: real(key, http=httpx.Client(transport=httpx.MockTransport(fake)),
                                                                  fresh=fresh))
    # sin conectar: las máquinas del sistema, sin las de Google
    r = admin.get("/api/cloud").json()
    assert r["connected"] is False and r["instances"] == [] and r["machines"] == []
    assert admin.post("/api/cloud/action", json={"zone": "southamerica-east1-c", "name": "vigia-analisis-1",
                                                 "action": "start"}).status_code == 409
    # una clave que no es de cuenta de servicio no se guarda
    r = admin.put("/api/cloud/credentials", json={"key": json.dumps({"type": "authorized_user", "x": 1})})
    assert r.status_code == 422 and "cuenta de servicio" in r.json()["detail"]
    # conectar: se prueba, se guarda cifrada y nunca vuelve
    r = admin.put("/api/cloud/credentials", json={"key": json.dumps(KEY)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["connected"] and body["project"] == "vigia-512822" and body["account"] == KEY["client_email"]
    assert [v["name"] for v in body["instances"]] == ["prueba-t2d", "vigia-analisis-1"]
    assert "PRIVATE KEY" not in r.text
    with app.state.Session() as s:
        stored = s.get(AppSetting, "gcp_key_enc").value
    assert stored and "PRIVATE KEY" not in stored and "analitica-app" not in stored
    # acciones: solo las tres, y solo un administrador
    r = admin.post("/api/cloud/action", json={"zone": "southamerica-east1-c", "name": "vigia-analisis-1",
                                              "action": "reset"})
    assert r.status_code == 200 and r.json()["ok"]
    assert admin.post("/api/cloud/action", json={"zone": "southamerica-east1-c", "name": "vigia-analisis-1",
                                                 "action": "delete"}).status_code == 422
    admin.post("/api/users", json={"username": "ana", "password": "empleada-1", "role": "entrenador"})
    emp = login(app, "ana", "empleada-1")
    assert emp.get("/api/cloud").status_code == 403
    assert emp.post("/api/cloud/action", json={"zone": "southamerica-east1-c", "name": "vigia-analisis-1",
                                               "action": "stop"}).status_code == 403
    # si Google falla, se dice y se sigue mostrando lo del sistema
    fake.status_list = 500
    r = admin.get("/api/cloud").json()
    assert r["connected"] and r["instances"] == [] and "500" in r["error"]
    # desconectar
    r = admin.delete("/api/cloud/credentials").json()
    assert r["connected"] is False


def test_el_resumen_dice_desde_cuando_no_se_sabe_de_una_maquina(app, admin):
    key = admin.post("/api/analysis/pool-key").json()["key"]
    r = TestClient(app).post("/api/agent/enroll", json={"key": key, "instance_id": "vigia-analisis-1",
                                                        "name": "maquina-vigia-analisis-1-1", "capacity": 4})
    assert r.status_code in (200, 201), r.text
    m = admin.get("/api/analysis").json()["machines"]
    assert m[0]["instance_id"] == "vigia-analisis-1" and m[0]["last_seen"]


def test_la_app_dice_en_castellano_como_esta_cada_maquina():
    from datetime import datetime, timedelta, timezone

    from agent.oficina import desde, hace, vm_estado

    now = datetime(2026, 10, 29, 19, 50, tzinfo=timezone.utc).astimezone()
    iso = lambda dt: dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: E731
    vm = {"name": "vigia-analisis-1", "zone": "southamerica-east1-c", "status": "RUNNING", "spot": False,
          "last_start": iso(now - timedelta(hours=3))}
    m = {"online": True, "draining": False, "leaving": False, "last_seen": iso(now - timedelta(seconds=10)),
         "cameras": 5}
    assert vm_estado(vm, m, now)[:2] == ("● Prendida · analizando", "ok")
    # prendida pero el sistema no sabe de ella hace rato (lo que pasó con el dominio viejo): reiniciar
    off = {**m, "online": False, "last_seen": iso(now - timedelta(hours=3, minutes=33))}
    text, tag, hint = vm_estado(vm, off, now)
    assert text.startswith("● Prendida pero sin conexión desde las ") and tag == "bad" and "Reiniciar" in hint
    # recién prendida: todavía arrancando
    assert vm_estado({**vm, "last_start": iso(now - timedelta(minutes=2))}, off, now)[1] == "busy"
    # prendida y nunca dada de alta (una prueba olvidada): gasta créditos
    assert "no analiza" in vm_estado(vm, None, now)[0]
    # apagada (Spot)
    text, tag, hint = vm_estado({**vm, "status": "TERMINATED", "spot": True}, off, now)
    assert text == "○ Apagada" and tag == "off" and "Spot" in hint and "Prender" in hint
    # sin Google Cloud conectado: solo lo del sistema, sin decir «apagada» (no se sabe)
    text, tag, hint = vm_estado(None, off, now)
    assert text.startswith("Sin conexión desde") and "Conectá Google Cloud" in hint
    assert vm_estado(None, m, now)[0] == "● En línea · analizando"
    assert hace(iso(now - timedelta(hours=3)), now) == "hace 3 h" and hace(None, now) == ""
    assert desde(None, now) == "(nunca se conectó)"
    assert desde("2026-10-27T15:44:00.000-07:00", now).startswith("desde el 27/10")


def test_una_clave_nueva_de_la_misma_cuenta_se_prueba_con_google_y_una_rota_no_da_500(app, admin, monkeypatch):
    """Lo que encontró la revisión: el token se guardaba por cuenta, así que una clave rota de la misma cuenta se
    «probaba» con el token viejo, se guardaba y después /api/cloud daba 500."""
    fake = FakeGoogle()
    real = nube.GoogleCloud
    monkeypatch.setattr(nube, "GoogleCloud", lambda key, http=None, fresh=False: real(
        key, http=httpx.Client(transport=httpx.MockTransport(fake)), fresh=fresh))
    assert admin.put("/api/cloud/credentials", json={"key": json.dumps(KEY)}).status_code == 200
    tokens = fake.tokens
    roto = {**KEY, "private_key": "-----BEGIN PRIVATE KEY-----\nno-sirve\n-----END PRIVATE KEY-----\n"}
    r = admin.put("/api/cloud/credentials", json={"key": json.dumps(roto)})
    assert r.status_code == 422 and "no sirve" in r.json()["detail"]
    assert fake.tokens == tokens  # no se pidió token: la firma falló antes, y la clave buena sigue guardada
    assert admin.get("/api/cloud").json()["instances"]
    # aunque una clave rota quedara guardada (editada a mano), nada da 500
    with app.state.Session() as s:
        from server.main import Settings  # noqa: F401  (solo para tener la sesión)
        row = s.get(AppSetting, "gcp_key_enc")
        from server.security import SecretBox
        box = SecretBox.from_env_or_file(app.state.settings.data_dir)
        row.value = box.encrypt(json.dumps(roto))
        s.commit()
    nube._tokens.clear()
    r = admin.get("/api/cloud")
    assert r.status_code == 200 and "no sirve" in r.json()["error"]
    r = admin.post("/api/cloud/action", json={"zone": "southamerica-east1-c", "name": "vigia-analisis-1",
                                              "action": "reset"})
    assert r.status_code == 409
    # desconectar olvida el token también
    admin.delete("/api/cloud/credentials")
    assert not nube._tokens
