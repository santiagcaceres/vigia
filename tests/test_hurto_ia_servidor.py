"""«Entrenar IA hurtos» en el servidor (pedido del dueño, 9 nov): la app sube los clips ya analizados en la PC (el video
no sale de ahí: solo los rasgos por persona y ventana, core/hurto_ia.py), se marcan (cuándo y quién roba, o «normal»),
se entrena la IA de movimientos de hurto y, activada, viaja a las cámaras que miden hurtos en `analysis["ia_hurto"]`
(como los gestos: aparte, un agente viejo la ignora). Todo es del personal (admin y operador): el cliente nunca."""

import gzip
import json
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from fastapi.testclient import TestClient

from core.rules import EVENT_TYPES, RuleConfig
from server import db as server_db
from server import pool
from server.main import Settings, create_app

ADMIN = ("admin", "clave-admin-123")
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def ia():
    return pytest.importorskip("core.hurto_ia")  # la arma otra parte (Python puro con numpy)


@pytest.fixture
def app(tmp_path, db_url):
    return create_app(Settings(data_dir=tmp_path / "data", database_url=db_url, admin_password=ADMIN[1],
                               serve_public=False))


@pytest.fixture
def admin(app):
    with TestClient(app) as c:
        assert c.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]}).status_code == 200
        yield c


def login(app, admin, username: str, role: str, client_id: int | None = None) -> TestClient:
    pw = f"clave-{username}-123"
    r = admin.post("/api/users", json={"username": username, "password": pw, "role": role,
                                       **({"client_id": client_id} if client_id else {})})
    assert r.status_code == 201, r.text
    c = TestClient(app)
    assert c.post("/api/login", json={"username": username, "password": pw}).status_code == 200
    return c


def cam_body(cid, name, **kw):
    return {"client_id": cid, "name": name, "brand": "ip", "host": "dvr", "password": "x", **kw}


def ventanas(ia, dur: float, tracks=(1, 2), hurto: tuple | None = None, seed: int = 0) -> list[dict]:
    """Ventanas de 3 s cada 1 s para cada persona. Las del ladrón que se pisan con el hurto (`hurto` = (persona,
    inicio, fin)) tienen rasgos altos; el resto, ruido alrededor de 0 (datos sintéticos separables)."""
    rng = random.Random(seed)
    n = len(ia.FEATURE_NAMES)
    out = []
    for tid in tracks:
        t = 0.0
        while t + 3 <= dur:
            pos = bool(hurto and tid == hurto[0] and min(t + 3, hurto[2]) - max(t, hurto[1]) >= 1)
            out.append({"track_id": tid, "t0": t, "t1": t + 3,
                        "f": [round((2.5 if pos else 0.0) + rng.gauss(0, 0.3), 4) for _ in range(n)]})
            t += 1.0
    return out


def clip_body(ia, uid: str, dur: float = 60.0, **kw) -> dict:
    body = {"uuid": uid, "name": f"{uid}.mp4", "duration_s": dur, "fps": 5, "width": 640, "height": 360,
            "feature_version": ia.FEATURE_VERSION, "people": 2, "windows": ventanas(ia, dur)}
    body.update(kw)
    return body


def setup(admin):
    """Un cliente con una cámara de góndola atendida por una PC central (para leer /api/agent/config)."""
    cid = admin.post("/api/clients", json={"name": "Almacén Demo"}).json()["id"]
    ag = admin.post("/api/agents", json={"name": "central", "client_id": None}).json()
    cam = admin.post("/api/cameras", json=cam_body(cid, "Góndola", agent_id=ag["id"], profile="gondola")).json()
    return cid, cam, {"X-Agent-Token": ag["token"]}, ag


def config_of(app, h: dict) -> dict:
    return {x["id"]: x for x in TestClient(app).get("/api/agent/config", headers=h).json()["cameras"]}


# ------------------------------------------------------------------ permisos
def test_solo_el_personal_entrena(app, admin, ia):
    cid, cam, _, _ = setup(admin)
    dueno = login(app, admin, "dueno", "cliente", cid)
    anon = TestClient(app)
    pedidos = [("get", "/api/training/clips", None), ("post", "/api/training/clips", clip_body(ia, "abc")),
               ("put", "/api/training/clips/1", {"verdict": "normal"}), ("delete", "/api/training/clips/1", None),
               ("get", "/api/training/clips/1/data", None), ("get", "/api/training/ia/summary", None),
               ("post", "/api/training/ia/run", {"client_id": None}), ("get", "/api/training/ia/models", None),
               ("post", "/api/training/ia/models/1/activate", None),
               ("post", "/api/training/ia/models/1/deactivate", None)]
    for method, url, body in pedidos:
        kw = {"json": body} if body is not None else {}
        assert getattr(dueno, method)(url, **kw).status_code == 403, url
        assert getattr(anon, method)(url, **kw).status_code == 401, url
    # el operador sí (sube, marca, entrena y activa, como el admin)
    op = login(app, admin, "lucia", "entrenador")
    r = op.post("/api/training/clips", json=clip_body(ia, "abc", camera_id=cam["id"]))
    assert r.status_code == 201, r.text
    assert op.put(f"/api/training/clips/{r.json()['id']}", json={"verdict": "normal"}).status_code == 200
    assert op.get("/api/training/ia/summary").json()["normal_clips"] == 1
    assert [c["created_by_name"] for c in admin.get("/api/training/clips").json()] == ["lucia"]


# ------------------------------------------------------------------ subir y marcar
def test_crear_y_actualizar_por_uuid(app, admin, ia):
    cid, cam, _, _ = setup(admin)
    r = admin.post("/api/training/clips", json=clip_body(ia, "a1b2c3", camera_id=cam["id"]))
    assert r.status_code == 201, r.text
    clip = r.json()
    assert clip["created"] is True and clip["verdict"] == "sin_marcar" and clip["marks"] == []
    assert clip["client_id"] == cid and clip["client_name"] == "Almacén Demo"  # el cliente sale de la cámara
    assert clip["camera_name"] == "Góndola" and clip["created_by_name"] == "admin"
    assert clip["n_windows"] == 2 * 58 and clip["feature_ok"] is True and clip["has_data"] is True
    assert clip["data_content_type"] == "application/gzip"
    assert urlsplit(clip["data_upload"]).path.endswith("/training/clips/a1b2c3.json.gz")
    assert "windows" not in clip

    marca = {"start": 20.0, "end": 30.0, "track_id": 1, "kind": "bolsillo"}
    r = admin.put(f"/api/training/clips/{clip['id']}", json={"verdict": "hurto", "marks": [marca]})
    assert r.status_code == 200, r.text
    assert r.json()["verdict"] == "hurto" and r.json()["marks"] == [marca] and r.json()["n_marks"] == 1
    assert r.json()["positive_windows"] > 0 and r.json()["marks_without_windows"] == []

    # el mismo archivo analizado de nuevo (otra PC, otra versión de la app): misma fila, conserva las marcas
    r = admin.post("/api/training/clips", json=clip_body(ia, "a1b2c3", dur=58.0, camera_id=cam["id"], fps=8,
                                                       name="otro nombre.mp4"))
    assert r.status_code == 200, r.text
    again = r.json()
    assert again["created"] is False and again["id"] == clip["id"] and again["fps"] == 8
    assert again["verdict"] == "hurto" and again["marks"] == [marca] and again["name"] == "otro nombre.mp4"
    assert again["n_windows"] == 2 * 56 and again["labels_kept"] is True
    # lo que marcó una persona no se pisa con lo que traiga una subida (se cambia con PUT)
    r = admin.post("/api/training/clips", json=clip_body(ia, "a1b2c3", camera_id=cam["id"], verdict="sin_marcar",
                                                       marks=[]))
    assert r.json()["verdict"] == "hurto" and r.json()["marks"] == [marca] and r.json()["labels_kept"] is True

    lista = admin.get("/api/training/clips").json()
    assert [c["id"] for c in lista] == [clip["id"]] and "windows" not in lista[0]
    assert lista[0]["n_windows"] == 2 * 58 and lista[0]["n_marks"] == 1
    assert admin.get(f"/api/training/clips?client_id={cid}").json()[0]["id"] == clip["id"]
    otro = admin.post("/api/clients", json={"name": "Otro"}).json()["id"]
    assert admin.get(f"/api/training/clips?client_id={otro}").json() == []
    # sin cámara ni cliente (un video de otro lado, para la IA de todos)
    r = admin.post("/api/training/clips", json=clip_body(ia, "suelto"))
    assert r.status_code == 201 and r.json()["client_id"] is None and r.json()["camera_id"] is None
    # un clip todavía sin marcar subido otra vez como «normal»: eso sí (no había nada marcado por una persona)
    r = admin.post("/api/training/clips", json=clip_body(ia, "suelto", verdict="normal"))
    assert r.status_code == 200 and r.json()["verdict"] == "normal" and r.json()["labels_kept"] is False
    # los rasgos se guardan redondeados (la base no se llena de decimales)
    with app.state.Session() as s:
        w = s.query(server_db.TrainingClip).filter_by(uuid="suelto").one().windows[0]
        assert set(w) == {"track_id", "t0", "t1", "f"} and all(len(repr(x)) <= 10 for x in w["f"])


def test_los_cuadros_van_directo_al_almacenamiento(app, admin, ia):
    _, cam, _, _ = setup(admin)
    clip = admin.post("/api/training/clips", json=clip_body(ia, "cuadros", camera_id=cam["id"])).json()
    r = admin.get(f"/api/training/clips/{clip['id']}/data")
    assert r.status_code == 404 and "todavía no llegaron" in r.json()["detail"]
    datos = b"\x1f\x8b" + b"cuadros comprimidos" * 10
    assert admin.put(clip["data_upload"], content=datos,
                     headers={"Content-Type": clip["data_content_type"]}).status_code == 200
    r = admin.get(f"/api/training/clips/{clip['id']}/data")
    assert r.status_code == 200, r.text
    assert r.json()["expires_in"] == 3600 and TestClient(app).get(r.json()["url"]).content == datos
    assert admin.get("/api/training/clips/999/data").status_code == 404
    # borrar el clip borra también sus cuadros
    assert admin.delete(f"/api/training/clips/{clip['id']}").status_code == 204
    assert TestClient(app).get(r.json()["url"]).status_code == 404
    assert admin.get("/api/training/clips").json() == []
    assert admin.delete(f"/api/training/clips/{clip['id']}").status_code == 404


@pytest.mark.parametrize("cambio, palabra", [
    ({"uuid": "con espacios"}, "uuid"), ({"uuid": "x" * 41}, "uuid"), ({"uuid": 123}, "uuid"),
    ({"name": ""}, "name"), ({"duration_s": 0}, "duration_s"), ({"duration_s": "60"}, "duration_s"),
    ({"duration_s": 1900, "windows": []}, "30 minutos"), ({"fps": 0}, "fps"), ({"width": 0}, "width"),
    ({"height": 360.5}, "height"), ({"feature_version": 999}, "versión"), ({"feature_version": "1"}, "versión"),
    ({"people": -1}, "people"), ({"windows": "muchas"}, "windows"),
    ({"windows": [{"track_id": 1, "t0": 0, "t1": 3, "f": [1.0]}]}, "rasgos"),
    ({"windows": [{"track_id": "1", "t0": 0, "t1": 3, "f": []}]}, "track_id"),
    ({"windows": [{"track_id": 1, "t0": 5, "t1": 3, "f": []}]}, "t0 y t1"),
    ({"windows": [{"track_id": 1, "t0": 0, "t1": 99, "f": []}]}, "dentro del clip"),
    ({"verdict": "robo"}, "verdict"), ({"verdict": "hurto"}, "al menos una marca"),
    ({"verdict": "normal", "marks": [{"start": 1, "end": 5, "track_id": 1, "kind": None}]}, "normal no lleva"),
    ({"verdict": "hurto", "marks": [{"start": 9, "end": 5, "track_id": 1, "kind": None}]}, "antes del fin"),
    ({"verdict": "hurto", "marks": [{"start": 50, "end": 70, "track_id": 1, "kind": None}]}, "fuera del clip"),
    ({"verdict": "hurto", "marks": [{"start": 1, "end": 5, "track_id": 1.5, "kind": None}]}, "track_id"),
    ({"verdict": "hurto", "marks": [{"start": 1, "end": 5, "track_id": 1, "kind": "mochila"}]}, "Cómo"),
    ({"verdict": "hurto", "marks": [{"end": 5, "track_id": 1}]}, "inicio y el fin"),
    ({"verdict": "hurto", "marks": "en el medio"}, "marks"),
    ({"camera_id": "1"}, "camera_id"),
])
def test_datos_invalidos_dan_422_en_castellano(app, admin, ia, cambio, palabra):
    _, cam, _, _ = setup(admin)
    body = clip_body(ia, "valido", camera_id=cam["id"])
    if cambio.get("windows") and isinstance(cambio["windows"], list):
        for w in cambio["windows"]:
            w["f"] = w["f"] if w["f"] else [0.0] * len(ia.FEATURE_NAMES)
    body.update(cambio)
    r = admin.post("/api/training/clips", json=body)
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert isinstance(detail, str) and palabra in detail, detail
    assert admin.get("/api/training/clips").json() == []  # no quedó nada a medias


def test_limites_de_tamano(app, admin, ia):
    _, cam, _, _ = setup(admin)
    # más de 20.000 ventanas: el pedido se acerca a los 4,5 MB de Vercel; se pide cortar el clip
    w = {"track_id": 1, "t0": 0.0, "t1": 3.0, "f": [0.0] * len(ia.FEATURE_NAMES)}
    r = admin.post("/api/training/clips", json=clip_body(ia, "largo", dur=1800, camera_id=cam["id"],
                                                         windows=[w] * 20001))
    assert r.status_code == 422 and "20000" in r.json()["detail"] and "cort" in r.json()["detail"]
    # 20.000 ventanas de verdad en JSON pasan los 4,5 MB de Vercel: comprimidas con gzip, no
    full = clip_body(ia, "largo", dur=1800, camera_id=cam["id"],
                     windows=ventanas(ia, 1800, tracks=range(1, 13))[:20000])
    raw = json.dumps(full, separators=(",", ":")).encode()
    packed = gzip.compress(raw)
    assert len(raw) > 4.5e6 > 2 * len(packed)  # (con rasgos al azar; los de verdad se comprimen todavía más)
    r = admin.post("/api/training/clips", content=packed,
                   headers={"Content-Type": "application/json", "Content-Encoding": "gzip"})
    assert r.status_code == 201, r.text[:300]  # 30 minutos justos, sí
    assert r.json()["n_windows"] == 20000
    r = admin.post("/api/training/clips", content=b"esto no es gzip",
                   headers={"Content-Type": "application/json", "Content-Encoding": "gzip"})
    assert r.status_code == 422 and "gzip" in r.json()["detail"]
    bomba = gzip.compress(b'{"uuid": "' + b"0" * (41 * 1024 * 1024) + b'"}')  # 40 KB que se inflan a 41 MB
    r = admin.post("/api/training/clips", content=bomba,
                   headers={"Content-Type": "application/json", "Content-Encoding": "gzip"})
    assert r.status_code == 413
    r = admin.post("/api/training/clips", content=b"[1, 2]", headers={"Content-Type": "application/json"})
    assert r.status_code == 422 and isinstance(r.json()["detail"], str)
    # NaN o infinito en los rasgos (json de Python los acepta): no
    bad = json.dumps(clip_body(ia, "nan", camera_id=cam["id"], windows=[{**w, "f": [float("nan")] * len(w["f"])}]))
    r = admin.post("/api/training/clips", content=bad, headers={"Content-Type": "application/json"})
    assert r.status_code == 422 and "NaN" in r.json()["detail"]
    # cámara de otro cliente, cámara o cliente que no existen
    otro = admin.post("/api/clients", json={"name": "Otro"}).json()["id"]
    r = admin.post("/api/training/clips", json=clip_body(ia, "mezcla", camera_id=cam["id"], client_id=otro))
    assert r.status_code == 422 and "no es de ese cliente" in r.json()["detail"]
    assert admin.post("/api/training/clips", json=clip_body(ia, "x1", camera_id=999)).status_code == 404
    assert admin.post("/api/training/clips", json=clip_body(ia, "x2", client_id=999)).status_code == 404


def test_marcas_y_veredicto(app, admin, ia):
    _, cam, _, _ = setup(admin)
    clip = admin.post("/api/training/clips", json=clip_body(ia, "marcas", camera_id=cam["id"])).json()
    url = f"/api/training/clips/{clip['id']}"
    r = admin.put(url, json={"verdict": "hurto"})
    assert r.status_code == 422 and "al menos una marca" in r.json()["detail"]
    marcas = [{"start": 40, "end": 45.5, "track_id": 2, "kind": "bolsa"},
              {"start": 10, "end": 20, "track_id": None, "kind": None},  # sin persona: vale para cualquiera
              {"start": 30, "end": 35, "track_id": 77, "kind": "otro"}]  # una persona que no tiene ventanas
    r = admin.put(url, json={"verdict": "hurto", "marks": marcas, "name": "Hurto en góndola"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert [m["start"] for m in d["marks"]] == [10, 30, 40]  # ordenadas
    assert d["name"] == "Hurto en góndola" and d["n_marks"] == 3
    assert d["marks_without_windows"] == [1]  # la de la persona 77: avisarle al que marca que no enseña nada
    assert d["positive_windows"] > 0 and d["negative_windows"] > 0
    # solo el nombre: no toca las marcas
    r = admin.put(url, json={"name": "Otro nombre"})
    assert r.json()["n_marks"] == 3 and r.json()["verdict"] == "hurto"
    # una marca fuera del clip
    r = admin.put(url, json={"marks": [{"start": 55, "end": 75, "track_id": 1, "kind": None}]})
    assert r.status_code == 422 and "fuera del clip" in r.json()["detail"]
    # pasarlo a «normal» borra las marcas; con marcas, no se puede
    r = admin.put(url, json={"verdict": "normal", "marks": marcas})
    assert r.status_code == 422 and "normal no lleva" in r.json()["detail"]
    r = admin.put(url, json={"verdict": "normal"})
    assert r.status_code == 200 and r.json()["marks"] == [] and r.json()["verdict"] == "normal"
    assert r.json()["positive_windows"] == 0 and r.json()["negative_windows"] == 2 * 58
    # «sin marcar» con marcas guardadas a medias (borrador): sí; no cuentan para entrenar
    r = admin.put(url, json={"verdict": "sin_marcar", "marks": marcas[:1]})
    assert r.status_code == 200 and r.json()["n_marks"] == 1 and r.json()["positive_windows"] == 0
    assert admin.put("/api/training/clips/999", json={"verdict": "normal"}).status_code == 404


# ------------------------------------------------------------------ entrenar, activar y mandar a las cámaras
def cargar_datos(admin, ia, cam_id: int, hurtos: int = 6, normales: int = 4) -> None:
    for i in range(hurtos):
        h = (1, 20.0 + i, 30.0 + i)
        body = clip_body(ia, f"hurto{i}", camera_id=cam_id, windows=ventanas(ia, 60, hurto=h, seed=i),
                         verdict="hurto", marks=[{"start": h[1], "end": h[2], "track_id": 1, "kind": "bolsillo"}])
        assert admin.post("/api/training/clips", json=body).status_code in (200, 201)  # (200: ya estaba)
    for i in range(normales):
        body = clip_body(ia, f"normal{i}", dur=200, camera_id=cam_id, windows=ventanas(ia, 200, seed=100 + i),
                         verdict="normal")
        assert admin.post("/api/training/clips", json=body).status_code in (200, 201)


def test_entrenar_activar_y_mandar_a_las_camaras_que_miden_hurtos(app, admin, ia):
    cid, cam, h, ag = setup(admin)
    # con pocos datos: 422 con lo que falta, en castellano
    cargar_datos(admin, ia, cam["id"], hurtos=2, normales=1)
    r = admin.post("/api/training/ia/run", json={"client_id": cid})
    assert r.status_code == 422 and isinstance(r.json()["detail"], str), r.text
    cargar_datos(admin, ia, cam["id"])
    # un clip sin marcar no cuenta
    assert admin.post("/api/training/clips", json=clip_body(ia, "pendiente", camera_id=cam["id"])).status_code == 201
    resumen = admin.get(f"/api/training/ia/summary?client_id={cid}").json()
    assert (resumen["theft_clips"], resumen["marks"], resumen["normal_clips"], resumen["unmarked"]) == (6, 6, 4, 1)
    assert resumen["clips"] == 11 and resumen["normal_minutes"] == pytest.approx(4 * 200 / 60, abs=0.1)
    assert resumen["min_marks"] == 5 and resumen["recommended_marks"] >= 30 and resumen["models"] == []

    rev = int(TestClient(app).get("/api/agent/config", headers=h).json()["rev"])
    r = admin.post("/api/training/ia/run", json={"client_id": cid, "target_recall": 0.8})
    assert r.status_code == 201, r.text
    m = r.json()
    assert m["active"] is False and m["scope"] == "cliente" and m["client_name"] == "Almacén Demo"
    assert m["usable"] is True and m["n_clips"] == 10 and m["version"] == ia.FEATURE_VERSION
    for k in ("marks", "marks_seen", "false_per_hour", "clips", "windows", "positives", "threshold", "auc"):
        assert k in m["metrics"], k
    assert m["metrics"]["target_recall"] == 0.8
    # sin activar no viaja
    assert "ia_hurto" not in config_of(app, h)[cam["id"]]["analysis"]

    # una cámara de solo métricas (por su tipo) y una de un cliente sin plan de hurto
    sin_hurto = admin.post("/api/clients", json={"name": "Solo conteo", "plans": ["comercio"]}).json()["id"]
    conteo = admin.post("/api/cameras", json=cam_body(sin_hurto, "Puerta", agent_id=ag["id"])).json()
    metricas = admin.post("/api/cameras", json=cam_body(cid, "Depósito", agent_id=ag["id"],
                                                        profile="metricas")).json()
    caja = admin.post("/api/cameras", json=cam_body(cid, "Caja", agent_id=ag["id"], profile="caja")).json()

    r = admin.post(f"/api/training/ia/models/{m['id']}/activate")
    assert r.status_code == 200 and r.json()["active"] is True
    cfg = TestClient(app).get("/api/agent/config", headers=h).json()
    assert int(cfg["rev"]) > rev  # las máquinas piden la configuración ya
    cams = {x["id"]: x for x in cfg["cameras"]}
    an = cams[cam["id"]]["analysis"]
    assert an["ia_hurto"]["kind"] == "ia_hurto" and an["ia_hurto"]["id"] == m["id"]
    assert an["ia_hurto"]["names"] == list(ia.FEATURE_NAMES) and an["ia_hurto"]["version"] == ia.FEATURE_VERSION
    RuleConfig.from_dict(an["rules"])  # nada de la IA en `rules` (un agente viejo es estricto ahí)
    assert "ia_hurto" not in json.dumps(an["rules"])
    assert "ia_hurto" in cams[caja["id"]]["analysis"]  # sin gestos, pero mide hurtos
    assert "ia_hurto" not in cams[metricas["id"]]["analysis"] and cams[metricas["id"]]["metrics_only"] is True
    assert "ia_hurto" not in cams[conteo["id"]]["analysis"]  # su cliente no tiene «hurto»
    # también en «Ver con la IA» y en la ventana de la cámara de la app
    user_h = {"X-Agent-Token": admin.post("/api/agent/session").json()["token"]}
    view = TestClient(app).post("/api/agent/view", json={"camera_ids": [cam["id"]]}, headers=user_h).json()
    assert view["cameras"][0]["analysis"]["ia_hurto"]["id"] == m["id"]
    tune = TestClient(app).post("/api/agent/tune", json={"camera_id": cam["id"]}, headers=user_h).json()
    assert tune["camera"]["analysis"]["ia_hurto"]["id"] == m["id"]

    # otro modelo del mismo alcance: activarlo apaga el anterior
    m2 = admin.post("/api/training/ia/run", json={"client_id": cid}).json()
    admin.post(f"/api/training/ia/models/{m2['id']}/activate")
    activos = {x["id"]: x["active"] for x in admin.get("/api/training/ia/models").json()}
    assert activos == {m["id"]: False, m2["id"]: True}
    assert config_of(app, h)[cam["id"]]["analysis"]["ia_hurto"]["id"] == m2["id"]
    # uno para todos los clientes: el del cliente manda; el de todos le llega a los demás
    g = admin.post("/api/training/ia/run", json={"client_id": None}).json()
    assert g["scope"] == "global" and g["client_id"] is None and g["n_clips"] == 10
    admin.post(f"/api/training/ia/models/{g['id']}/activate")
    otro = admin.post("/api/clients", json={"name": "Kiosco"}).json()["id"]
    kiosco = admin.post("/api/cameras", json=cam_body(otro, "Mostrador", agent_id=ag["id"])).json()
    cams = config_of(app, h)
    assert cams[cam["id"]]["analysis"]["ia_hurto"]["id"] == m2["id"]
    assert cams[kiosco["id"]]["analysis"]["ia_hurto"]["id"] == g["id"]
    assert {x["id"] for x in admin.get(f"/api/training/ia/models?client_id={otro}").json()} == {g["id"]}
    assert admin.get(f"/api/training/ia/summary?client_id={otro}").json()["models"][0]["id"] == g["id"]
    # apagar el del cliente: le llega el de todos
    r = admin.post(f"/api/training/ia/models/{m2['id']}/deactivate")
    assert r.status_code == 200 and r.json()["active"] is False
    assert config_of(app, h)[cam["id"]]["analysis"]["ia_hurto"]["id"] == g["id"]
    admin.post(f"/api/training/ia/models/{g['id']}/deactivate")
    assert "ia_hurto" not in config_of(app, h)[cam["id"]]["analysis"]
    assert admin.post("/api/training/ia/models/999/activate").status_code == 404
    assert admin.post("/api/training/ia/run", json={"client_id": 999}).status_code == 404


def test_con_demasiadas_ventanas_pide_entrenar_por_partes(app, admin, ia, monkeypatch):
    import server.main as server_main

    cid, cam, _, _ = setup(admin)
    cargar_datos(admin, ia, cam["id"])
    monkeypatch.setattr(server_main, "MAX_IA_TRAIN_WINDOWS", 1000)  # (en producción, 300.000: ~15 s y ~1 GB)
    r = admin.post("/api/training/ia/run", json={"client_id": None})
    assert r.status_code == 422 and "ventanas para entrenar" in r.json()["detail"], r.text
    assert "por cliente" in r.json()["detail"]
    assert admin.get("/api/training/ia/models").json() == []


def test_un_modelo_de_otra_version_no_se_activa(app, admin, ia):
    cid, cam, h, _ = setup(admin)
    with app.state.Session() as s:
        viejo = server_db.TheftModel(client_id=None, spec={"kind": "ia_hurto", "version": -1, "names": ["x"]},
                                     metrics={}, n_clips=3)
        s.add(viejo)
        s.commit()
        mid = viejo.id
    lista = admin.get("/api/training/ia/models").json()
    assert lista[0]["usable"] is False
    r = admin.post(f"/api/training/ia/models/{mid}/activate")
    assert r.status_code == 409 and "otra versión" in r.json()["detail"]
    assert "ia_hurto" not in config_of(app, h)[cam["id"]]["analysis"]


def test_la_ia_pesa_como_los_gestos_en_el_reparto(app, admin, ia):
    assert pool.uses_pose({"gestures": {"enabled": ["mano_bolsillo"]}}) is True
    assert pool.uses_pose({"ia_hurto": {"kind": "ia_hurto"}}) is True
    assert pool.uses_pose({"gestures": {"enabled": ["mano_ropa"]}, "ia_hurto": {"kind": "ia_hurto"}}) is True
    assert pool.uses_pose({"zones": [], "rules": {}}) is False
    now = datetime(2026, 11, 9, 15, 0)
    w = lambda **kw: pool.camera_weight(**{"fps": 5, "mode": "vivo", "brand": "ip", "hours": None,  # noqa: E731
                                           "tz": "America/Montevideo", "off_hours": "apagada", "now": now, **kw})
    assert w(pose=True) == pytest.approx(pool.POSE_FACTOR)

    cid, _, _, _ = setup(admin)
    caja = admin.post("/api/cameras", json=cam_body(cid, "Caja", profile="caja")).json()  # sin gestos: pesa 1
    gondola = admin.post("/api/cameras", json=cam_body(cid, "Pasillo", profile="gondola")).json()  # con gestos: 1,3
    r = admin.post("/api/cameras/analyze", json={"camera_ids": [caja["id"], gondola["id"]], "analyze": True})
    assert r.status_code == 200, r.text
    units = lambda: admin.get("/api/analysis").json()["plan"]["units"]  # noqa: E731
    assert units() == pytest.approx(1.0 + pool.POSE_FACTOR, abs=0.01)
    cargar_datos(admin, ia, gondola["id"])
    m = admin.post("/api/training/ia/run", json={"client_id": None}).json()
    admin.post(f"/api/training/ia/models/{m['id']}/activate")
    # la caja ahora mira posturas (1,3); el pasillo ya las miraba por los gestos: la postura se calcula una vez
    assert units() == pytest.approx(2 * pool.POSE_FACTOR, abs=0.01)
    admin.post(f"/api/training/ia/models/{m['id']}/deactivate")
    assert units() == pytest.approx(1.0 + pool.POSE_FACTOR, abs=0.01)


# ------------------------------------------------------------------ alertas de la IA
def test_las_alertas_de_la_ia_se_guardan_con_sus_rasgos(app, admin, ia):
    if "hurto_ia" not in EVENT_TYPES:
        pytest.skip("core.rules todavía no tiene HURTO_IA")
    _, cam, h, _ = setup(admin)
    rasgos = {n: 0.5 for n in ia.FEATURE_NAMES}
    meta = {"uuid": "evia1", "camera_id": cam["id"], "type": "hurto_ia", "track_id": 4,
            "description": "Persona #4: posible hurto (IA de movimientos)", "occurred_at": "2026-11-09T11:00:00-03:00",
            "features": {**rasgos, "ia_score": 0.91, "track_age_s": 12.0, "otra_cosa": 3.0, "ia_texto": "no"},
            "alert": True}
    r = TestClient(app).post("/api/agent/events", headers=h, data={"meta": json.dumps(meta)},
                             files={"snapshot": ("s.jpg", b"\xff\xd8fakejpg", "image/jpeg")})
    assert r.status_code == 201, r.text
    ev = admin.get(f"/api/events/{r.json()['id']}").json()
    assert ev["type"] == "hurto_ia" and ev["severity"] == "alta"
    assert ev["features"]["ia_score"] == 0.91 and ev["features"]["track_age_s"] == 12.0
    assert all(ev["features"][n] == 0.5 for n in ia.FEATURE_NAMES)
    assert "otra_cosa" not in ev["features"] and "ia_texto" not in ev["features"]
    # y como ejemplo para el filtro desde la ventana de la cámara (alerta de prueba etiquetada)
    meta = {**meta, "uuid": "ajia1", "label": "hurto"}
    r = admin.post("/api/training/samples", data={"meta": json.dumps(meta)},
                   files={"snapshot": ("s.jpg", b"\xff\xd8fakejpg", "image/jpeg")})
    assert r.status_code == 201, r.text


# ------------------------------------------------------------------ borrar clientes, cámaras y usuarios
def test_borrar_cliente_camara_o_usuario(app, admin, ia):
    cid, cam, _, _ = setup(admin)
    otra = admin.post("/api/cameras", json=cam_body(cid, "Fondo")).json()
    lucia = login(app, admin, "lucia", "entrenador")
    a = lucia.post("/api/training/clips", json=clip_body(ia, "c1", camera_id=cam["id"])).json()
    b = admin.post("/api/training/clips", json=clip_body(ia, "c2", camera_id=otra["id"])).json()
    assert admin.put(b["data_upload"], content=b"\x1f\x8bdatos").status_code == 200
    # la cámara se borra: el clip queda, sin cámara (son datos para entrenar; el video está en la PC)
    assert admin.post("/api/cameras/delete", json={"camera_ids": [cam["id"]]}).status_code == 200
    clips = {c["id"]: c for c in admin.get("/api/training/clips").json()}
    assert clips[a["id"]]["camera_id"] is None and clips[a["id"]]["client_id"] == cid
    # el usuario se borra: lo que subió queda, sin su nombre
    uid = next(u["id"] for u in admin.get("/api/users").json() if u["username"] == "lucia")
    assert admin.delete(f"/api/users/{uid}").status_code == 204
    assert {c["id"]: c["created_by_name"] for c in admin.get("/api/training/clips").json()}[a["id"]] is None
    # una cámara que pasa a otro cliente se lleva sus clips
    otro = admin.post("/api/clients", json={"name": "Otro"}).json()["id"]
    r = admin.put(f"/api/cameras/{otra['id']}", json=cam_body(otro, "Fondo"))
    assert r.status_code == 200, r.text
    assert {c["id"]: c["client_id"] for c in admin.get("/api/training/clips").json()}[b["id"]] == otro
    # el cliente se borra: sus clips (con sus cuadros) y sus modelos también
    data = admin.get(f"/api/training/clips/{b['id']}/data").json()["url"]
    with app.state.Session() as s:
        s.add(server_db.TheftModel(client_id=otro, spec={"kind": "ia_hurto"}, metrics={}, n_clips=1))
        s.commit()
    assert admin.delete(f"/api/clients/{otro}").status_code == 204
    assert [c["id"] for c in admin.get("/api/training/clips").json()] == [a["id"]]
    assert TestClient(app).get(data).status_code == 404
    assert admin.get("/api/training/ia/models").json() == []


def test_tablas_nuevas_y_el_servidor_sigue_sin_opencv():
    # las tablas nuevas se crean solas en producción solo si sube la versión del esquema
    assert server_db.SCHEMA_VERSION >= "2026-11-09.1"
    assert {"training_clips", "theft_models"} <= set(server_db.Base.metadata.tables)
    assert "windows" in server_db.TrainingClip.__table__.columns
    # en Vercel no hay OpenCV ni onnxruntime: el servidor (y la IA de hurtos, que entrena ahí) son Python puro
    code = ("import sys, server.main\n"
            "try:\n"
            "    import core.hurto_ia\n"
            "except ImportError:\n"
            "    pass\n"
            "bad = [m for m in ('cv2', 'onnxruntime', 'core.detectors', 'core.pose') if m in sys.modules]\n"
            "print(','.join(bad))\n"
            "sys.exit(1 if bad else 0)\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
