"""«Entrenar objetos» en el servidor (18 nov, el dueño: «otra pestaña… primero añadir el nuevo objeto y ponerle para qué
plan de métricas sería… ¿podemos subir los videos y fotos desde la app y que los analicen las máquinas virtuales?»):
objeto con clases y plan, fotos al Storage, marcas, mínimos para entrenar, y los trabajos que toman las máquinas (sacar
fotos de un video, pre-marcar, entrenar) hasta la versión activa."""

import hashlib
from urllib.parse import urlsplit

import pytest
from fastapi.testclient import TestClient

from core import objetos as ob
from server.main import Settings, create_app

ADMIN = ("admin", "clave-admin-123")


@pytest.fixture
def app(tmp_path, db_url):
    return create_app(Settings(data_dir=tmp_path / "data", database_url=db_url, admin_password=ADMIN[1],
                               serve_public=False))


def login(app, user=ADMIN):
    c = TestClient(app)
    r = c.post("/api/login", json={"username": user[0], "password": user[1]})
    assert r.status_code == 200, r.text
    return c


def machine(app, admin, name="m1", caps=None):
    key = admin.post("/api/analysis/pool-key").json()["key"]
    tok = TestClient(app).post("/api/agent/enroll", json={"key": key, "name": name, "instance_id": f"i-{name}"}).json()
    m = TestClient(app)
    m.headers["X-Agent-Token"] = tok["token"]
    full = {"video": True, "ultra": True, "gpu": True, "device": "T4"}
    body = {"cameras": []} if caps == "vieja" else {"cameras": [], "objects": full if caps is None else caps}
    r = m.post("/api/agent/status", json=body)
    assert r.status_code == 200, r.text
    return m


def put(client, url, data, ctype="image/jpeg"):
    """Subida a la URL firmada (en las pruebas, el almacenamiento local del servidor)."""
    u = urlsplit(url)
    r = client.put(u.path + "?" + u.query, content=data, headers={"Content-Type": ctype})
    assert r.status_code < 300, r.text


def photos(n, start=0):
    out = []
    for i in range(start, start + n):
        data = f"foto {i}".encode() * 10
        out.append((hashlib.sha1(data).hexdigest(), data))
    return out


def upload(admin, mid, items):
    r = admin.post(f"/api/objects/{mid}/images", json={"images": [
        {"uuid": u, "name": f"f{k}.jpg", "width": 1280, "height": 720} for k, (u, _) in enumerate(items)]})
    assert r.status_code == 200, r.text
    got = r.json()["items"]
    data = dict(items)
    for it in got:
        if it["upload_url"]:
            put(admin, it["upload_url"], data[it["uuid"]])
    r = admin.post(f"/api/objects/{mid}/images/uploaded", json={"ids": [it["id"] for it in got]})
    assert r.status_code == 200
    return got


BOX = {"k": "c1", "x1": 0.1, "y1": 0.2, "x2": 0.4, "y2": 0.6}


def test_clases_recuadros_y_minimos():
    cl = ob.clean_classes([{"name": "Pallet vacío", "prompt": "empty pallet"}, {"name": " Terminado  con film "}])
    assert [c["key"] for c in cl] == ["c1", "c2"] and cl[1]["name"] == "Terminado con film"
    # renombrar conserva la clave; una nueva toma la siguiente libre (nunca se reusa)
    cl2 = ob.clean_classes([{"key": "c2", "name": "Terminado"}, {"name": "En armado"}], cl)
    assert [c["key"] for c in cl2] == ["c2", "c3"]
    for bad in ([], [{"name": "a"}, {"name": "A"}], [{"name": "x", "prompt": "palé"}], [{"key": "c9", "name": "x"}]):
        with pytest.raises(ob.ObjectDataError):
            ob.clean_classes(bad, cl)
    boxes = ob.clean_boxes([{"k": "c1", "x1": 0.5, "y1": 0.6, "x2": 0.1, "y2": 1.4}, {**BOX, "x2": 0.101}], ["c1"])
    assert boxes == [{"k": "c1", "x1": 0.1, "y1": 0.6, "x2": 0.5, "y2": 1.0}]  # ordenado, recortado; el clic suelto no
    with pytest.raises(ob.ObjectDataError):
        ob.clean_boxes([{**BOX, "k": "c7"}], ["c1"])
    assert ob.clean_boxes([{**BOX, "k": None, "p": 0.52}], ["c1"], with_conf=True)[0]["p"] == 0.52
    assert ob.doubt_of([{"p": 0.5}]) == 1.0 and ob.doubt_of([{"p": 0.95}]) < 0.2 and ob.doubt_of([]) == 0
    assert ob.yolo_lines([BOX], ["c1"]) == ["0 0.250000 0.400000 0.300000 0.400000"]
    # la división es fija por foto y deja más o menos 1 de cada 5 para probar
    uids = [hashlib.sha1(str(i).encode()).hexdigest() for i in range(1000)]
    test = sum(ob.split_of(u) == "prueba" for u in uids)
    assert 150 < test < 250 and ob.split_of(uids[0]) == ob.split_of(uids[0])
    chk = ob.train_check(10, {"c1": 3}, cl)
    assert not chk["ok"] and len(chk["problems"]) == 2
    chk = ob.train_check(40, {"c1": 25, "c2": 4}, cl, test_marked=8)
    assert chk["ok"] and any("Terminado con film" in w for w in chk["warnings"]) and chk["tips"]
    assert ob.grade(0.9)[0] == "Muy bien" and ob.grade(0.3)[0] == "Todavía no sirve"
    txt = ob.metrics_text({"map50": 0.78, "test_images": 12, "per_class": {"c1": {"precision": 0.9, "recall": 0.7}}},
                          cl)
    assert txt[0].startswith("Resultado: Bien") and "encuentra 7 de cada 10" in txt[2] and "9 están bien" in txt[2]


def test_objeto_fotos_marcas_entrenar_y_version(app):
    admin = login(app)
    # el plan nuevo existe y un cliente lo puede contratar
    cl = admin.post("/api/clients", json={"name": "Depósito", "plans": ["objetos"]})
    assert cl.status_code == 201, cl.text
    r = admin.post("/api/objects", json={"name": "Pallets", "plan": "nada", "classes": [{"name": "Pallet"}]})
    assert r.status_code == 422
    r = admin.post("/api/objects", json={"name": "Pallets", "plan": "objetos", "client_id": cl.json()["id"],
                                         "classes": [{"name": "Pallet", "prompt": "pallet"}, {"name": "Con film"}]})
    assert r.status_code == 201, r.text
    m = r.json()
    mid = m["id"]
    assert m["plan_name"] == "Conteo de objetos" and m["client_name"] == "Depósito"
    assert admin.post("/api/objects", json={"name": "pallets", "classes": [{"name": "x"}]}).status_code == 409
    # sin máquina: entrenar no se deja (faltan fotos) y pre-marcar sin fotos tampoco
    assert admin.post(f"/api/objects/{mid}/train", json={}).status_code == 422
    assert admin.post(f"/api/objects/{mid}/prelabel").status_code == 422
    items = photos(40)
    got = upload(admin, mid, items)
    assert len(got) == 40 and not any(g["existed"] for g in got)
    assert upload(admin, mid, items[:2])[0]["existed"]  # la misma foto otra vez no se duplica
    lst = admin.get(f"/api/objects/{mid}/images?status=sin_marcar").json()
    assert lst["total"] == 40
    urls = admin.post("/api/objects/images/urls", json={"ids": [lst["items"][0]["id"]]}).json()["urls"]
    u = urlsplit(next(iter(urls.values())))
    assert admin.get(u.path + "?" + u.query).content in dict(items).values()
    # marcar: 30 con un pallet y 5 «no hay ninguno»; una mala, descartada
    ids = [i["id"] for i in lst["items"]]
    for iid in ids[:30]:
        r = admin.put(f"/api/objects/images/{iid}", json={"boxes": [BOX]})
        assert r.status_code == 200 and r.json()["status"] == "marcada"
    for iid in ids[30:35]:
        assert admin.put(f"/api/objects/images/{iid}", json={"boxes": []}).json()["status"] == "marcada"
    assert admin.put(f"/api/objects/images/{ids[35]}", json={"status": "descartada"}).json()["status"] == "descartada"
    assert admin.put(f"/api/objects/images/{ids[36]}", json={"boxes": [{**BOX, "k": "c9"}]}).status_code == 422
    d = admin.get(f"/api/objects/{mid}").json()
    assert d["counts"]["marcada"] == 35 and d["counts"]["descartada"] == 1 and d["boxes"] == {"c1": 30, "c2": 0}
    assert d["check"]["ok"] and any("Con film" in w for w in d["check"]["warnings"])
    # sacar una clase con recuadros no se deja; renombrar sí
    r = admin.put(f"/api/objects/{mid}", json={"name": "Pallets", "classes": [{"key": "c2", "name": "Con film"}]})
    assert r.status_code == 409 and "Pallet" in r.json()["detail"]
    r = admin.put(f"/api/objects/{mid}", json={"name": "Pallets", "classes": [
        {"key": "c1", "name": "Pallet de madera", "prompt": "pallet"}, {"key": "c2", "name": "Con film"}]})
    assert r.status_code == 200 and r.json()["classes"][0]["name"] == "Pallet de madera"
    # entrenar: queda en cola y avisa que no hay máquina que entrene
    r = admin.post(f"/api/objects/{mid}/train", json={})
    assert r.status_code == 201, r.text
    job = r.json()
    assert job["status"] == "en_cola" and "ENTRENAR=si" in job["note"] and job["params"]["epochs"] == 120
    assert admin.post(f"/api/objects/{mid}/train", json={}).status_code == 409  # uno a la vez
    # una máquina sin Ultralytics no lo toma; una con, sí
    plain = machine(app, admin, "sin-ultra", {"video": True})
    assert plain.get("/api/agent/objects/next").status_code == 204
    mq = machine(app, admin, "tigo")
    nxt = mq.get("/api/agent/objects/next")
    assert nxt.status_code == 200, nxt.text
    p = nxt.json()
    assert p["kind"] == "entrenar" and p["job_id"] == job["id"] and p["model"]["name"] == "Pallets"
    assert mq.get("/api/agent/objects/next").status_code == 204  # ya no hay otro
    assert mq.post(f"/api/agent/objects/jobs/{job['id']}/progress",
                   json={"progress": 0.5, "message": "Vuelta 60 de 120"}).json() == {"cancel": False}
    ds = mq.get(f"/api/agent/objects/jobs/{job['id']}/dataset").json()
    assert len(ds["items"]) == 35 and {i["split"] for i in ds["items"]} <= {"prueba", "entrenar"}
    assert sum(len(i["boxes"]) for i in ds["items"]) == 30 and [c["key"] for c in ds["classes"]] == ["c1", "c2"]
    # avisar que terminó sin subir el modelo: no
    assert mq.post(f"/api/agent/objects/jobs/{job['id']}/done", json={"result": {}}).status_code == 409
    f = mq.post(f"/api/agent/objects/jobs/{job['id']}/files").json()
    put(mq, f["onnx_url"], b"onnx", "application/octet-stream")
    put(mq, f["pt_url"], b"pt", "application/octet-stream")
    r = mq.post(f"/api/agent/objects/jobs/{job['id']}/done", json={"result": {
        "metrics": {"map50": 0.81, "precision": 0.8, "recall": 0.75, "test_images": 7, "bogus": 3,
                    "per_class": {"c1": {"precision": 0.8, "recall": 0.75, "map50": 0.81}, "zz": {"precision": 1}}},
        "n_images": 35, "n_boxes": 30, "device": "tigo (Tesla T4)", "epochs_run": 74}})
    assert r.status_code == 200, r.text
    d = admin.get(f"/api/objects/{mid}").json()
    v = d["version_list"][0]
    assert v["number"] == 1 and v["grade"] == "Bien" and not v["active"] and "zz" not in v["metrics"]["per_class"]
    assert d["jobs"][0]["status"] == "listo" and d["jobs"][0]["machine"] == "tigo"
    assert any("encuentra 8 de cada 10" in t for t in v["text"])  # 0,75 → 8
    # activarla: ahora pre-marcar usa la versión (y el .pt para seguir entrenando)
    assert admin.post(f"/api/objects/versions/{v['id']}/activate").json()["active"]
    assert admin.get(f"/api/objects/versions/{v['id']}/model").json()["number"] == 1
    r = admin.post(f"/api/objects/{mid}/prelabel")
    assert r.status_code == 201 and r.json()["params"]["method"] == "version"
    p = mq.get("/api/agent/objects/next").json()
    assert p["kind"] == "premarcar" and p["prelabel"]["number"] == 1 and p["prelabel"]["pt_url"]
    todo = mq.get(f"/api/agent/objects/jobs/{p['job_id']}/todo").json()["items"]
    assert len(todo) == 4  # 40 − 35 marcadas − 1 descartada
    r = mq.post(f"/api/agent/objects/jobs/{p['job_id']}/proposals", json={"by": "Versión 1", "items": [
        {"id": todo[0]["id"], "boxes": [{**BOX, "p": 0.5}]}, {"id": ids[0], "boxes": [{**BOX, "p": 0.9}]}]})
    assert r.json()["ok"] == 1  # la marcada por una persona no se toca
    assert mq.post(f"/api/agent/objects/jobs/{p['job_id']}/done", json={"result": {"proposed": 1}}).status_code == 200
    first = admin.get(f"/api/objects/{mid}/images?status=sin_marcar&order=dudosas").json()["items"][0]
    assert first["id"] == todo[0]["id"] and first["doubt"] == 1.0 and first["proposed_by"] == "Versión 1"
    # el próximo entrenamiento sigue desde la versión 1 (mismas clases)
    r = admin.post(f"/api/objects/{mid}/train", json={})
    assert r.json()["params"]["base"] == "version"
    p = mq.get("/api/agent/objects/next").json()
    assert p["base"]["number"] == 1 and p["base"]["pt_url"]
    # cancelar: la máquina se entera en el próximo aviso
    assert admin.post(f"/api/objects/jobs/{p['job_id']}/cancel").json()["status"] == "cancelado"
    assert mq.post(f"/api/agent/objects/jobs/{p['job_id']}/progress", json={"progress": 0.1}).json()["cancel"]
    # un operador hace todo menos borrar el objeto
    admin.post("/api/users", json={"username": "lucia", "password": "clave-lucia-123", "role": "entrenador"})
    op = login(app, ("lucia", "clave-lucia-123"))
    assert op.get("/api/objects").status_code == 200
    assert op.delete(f"/api/objects/{mid}").status_code == 403
    assert admin.delete(f"/api/objects/{mid}").status_code == 204
    assert admin.get("/api/objects").json() == []


def test_video_a_la_maquina(app):
    admin = login(app)
    mid = admin.post("/api/objects", json={"name": "Cajas", "classes": [{"name": "Caja", "prompt": "box"}]}).json()["id"]
    assert admin.post(f"/api/objects/{mid}/videos", json={"filename": "x.exe"}).status_code == 422
    r = admin.post(f"/api/objects/{mid}/videos", json={"filename": "depósito.mp4", "size": 1000, "every_s": 3})
    assert r.status_code == 201, r.text
    j = r.json()
    assert j["status"] == "subiendo"
    assert admin.post(f"/api/objects/jobs/{j['id']}/uploaded").status_code == 409  # todavía no llegó
    put(admin, j["upload_url"], b"video", "video/mp4")
    r = admin.post(f"/api/objects/jobs/{j['id']}/uploaded")
    assert r.json()["status"] == "en_cola" and "ninguna máquina" in r.json()["note"]
    # una máquina vieja (sin «objects» en el estado) no lo toma; una nueva sin Ultralytics sí
    old = machine(app, admin, "vieja", "vieja")
    assert old.get("/api/agent/objects/next").status_code == 204
    mq = machine(app, admin, "gcp", {"video": True})
    p = mq.get("/api/agent/objects/next").json()
    assert p["kind"] == "video" and p["params"]["every_s"] == 3 and p["prelabel"]["method"] == "yoloe"
    u = urlsplit(p["download_url"])
    assert mq.get(u.path + "?" + u.query).content == b"video"
    shots = photos(3, 100)
    r = mq.post(f"/api/agent/objects/jobs/{p['job_id']}/images", json={"images": [
        {"uuid": uid, "width": 1280, "height": 720, "video_ts": 3.0 * k} for k, (uid, _) in enumerate(shots)]})
    got = r.json()["items"]
    for it in got:
        put(mq, it["upload_url"], dict(shots)[it["uuid"]])
    assert mq.post(f"/api/agent/objects/jobs/{p['job_id']}/images/uploaded",
                   json={"ids": [g["id"] for g in got]}).json()["ok"] == 3
    assert mq.post(f"/api/agent/objects/jobs/{p['job_id']}/done",
                   json={"result": {"images": 3, "skipped": 40, "duration_s": 130}}).status_code == 200
    imgs = admin.get(f"/api/objects/{mid}/images").json()["items"]
    assert len(imgs) == 3 and all(i["source"] == "video" and i["source_name"] == "depósito.mp4" for i in imgs)
    jobs = admin.get(f"/api/objects/{mid}/jobs").json()
    assert jobs[0]["result"]["images"] == 3 and jobs[0]["machine"] == "gcp"
    # el video ya no se guarda (quedan las fotos)
    with app.state.Session() as s:
        from server.db import ObjectJob
        assert s.get(ObjectJob, p["job_id"]).storage_path is None


def test_lo_que_encontro_la_revision(app):
    """Editar el objeto mientras entrena, sacar y agregar clases, propuestas de una clase que ya no está, y un video que
    falla (revisión del 18 nov)."""
    admin = login(app)
    mid = admin.post("/api/objects", json={"name": "Pallets", "classes": [
        {"name": "Pallet"}, {"name": "Con film"}]}).json()["id"]
    items = photos(31)
    got = upload(admin, mid, items)
    for g in got:
        admin.put(f"/api/objects/images/{g['id']}", json={"boxes": [BOX]})
    job = admin.post(f"/api/objects/{mid}/train", json={}).json()
    mq = machine(app, admin, "tigo")
    assert mq.get("/api/agent/objects/next").json()["job_id"] == job["id"]
    # el dataset viene por partes y guarda las clases con que se entrena
    part = mq.get(f"/api/agent/objects/jobs/{job['id']}/dataset?limit=10").json()
    assert len(part["items"]) == 10 and part["next"] == 10 and part["total"] == 31
    rest = mq.get(f"/api/agent/objects/jobs/{job['id']}/dataset?offset=30&limit=10").json()
    assert len(rest["items"]) == 1 and rest["next"] is None
    # mientras entrena sacan «Con film» y agregan «Vacío»: la clave c2 no se reusa
    r = admin.put(f"/api/objects/{mid}", json={"name": "Pallets", "classes": [
        {"key": "c1", "name": "Pallet"}, {"name": "Vacío"}]})
    assert [c["key"] for c in r.json()["classes"]] == ["c1", "c3"]
    r = admin.put(f"/api/objects/{mid}", json={"name": "Pallets", "classes": [
        {"key": "c1", "name": "Pallet"}, {"key": "c3", "name": "Vacío"}, {"name": "Roto"}]})
    assert [c["key"] for c in r.json()["classes"]] == ["c1", "c3", "c4"]
    f = mq.post(f"/api/agent/objects/jobs/{job['id']}/files").json()
    put(mq, f["onnx_url"], b"onnx", "application/octet-stream")
    mq.post(f"/api/agent/objects/jobs/{job['id']}/done", json={"result": {
        "metrics": {"map50": 0.8, "per_class": {"c2": {"recall": 0.5}}}}})
    v = admin.get(f"/api/objects/{mid}").json()["version_list"][0]
    assert [c["key"] for c in v["classes"]] == ["c1", "c2"] and "c2" in v["metrics"]["per_class"]
    # pre-marcar con esa versión: un recuadro de «c2» (ya no existe) queda como «objeto», no tira el lote
    with app.state.Session() as s:
        from server.db import ObjectVersion
        s.get(ObjectVersion, v["id"]).pt_path = "x.pt"
        s.commit()
    admin.post(f"/api/objects/versions/{v['id']}/activate")
    admin.put(f"/api/objects/images/{got[0]['id']}", json={"status": "sin_marcar"})
    p = admin.post(f"/api/objects/{mid}/prelabel").json()
    assert mq.get("/api/agent/objects/next").json()["job_id"] == p["id"]
    r = mq.post(f"/api/agent/objects/jobs/{p['id']}/proposals", json={"by": "Versión 1", "items": [
        {"id": got[0]["id"], "boxes": [{**BOX, "k": "c2", "p": 0.7}, {**BOX, "p": 0.9}]}]})
    assert r.status_code == 200 and r.json()["ok"] == 1
    img = next(i for i in admin.get(f"/api/objects/{mid}/images?status=sin_marcar").json()["items"]
               if i["id"] == got[0]["id"])
    assert [b["k"] for b in img["proposed"]] == [None, "c1"]
    mq.post(f"/api/agent/objects/jobs/{p['id']}/done", json={"result": {}})
    # un video que falla: se borra de la nube
    j = admin.post(f"/api/objects/{mid}/videos", json={"filename": "v.mp4"}).json()
    put(admin, j["upload_url"], b"video", "video/mp4")
    admin.post(f"/api/objects/jobs/{j['id']}/uploaded")
    assert mq.get("/api/agent/objects/next").json()["job_id"] == j["id"]
    with app.state.Session() as s:
        from server.db import ObjectJob
        path = s.get(ObjectJob, j["id"]).storage_path
    assert app.state.storage.exists(path)
    mq.post(f"/api/agent/objects/jobs/{j['id']}/fail", json={"error": "no abre"})
    assert not app.state.storage.exists(path)
    # borrar un cliente cancela sus videos en cola (y los borra)
    cl = admin.post("/api/clients", json={"name": "Otro", "plans": ["objetos"]}).json()
    cam = admin.post("/api/cameras", json={"client_id": cl["id"], "name": "Depósito", "brand": "hikvision",
                                           "host": "local.ddns.net", "password": "x"}).json()
    j = admin.post(f"/api/objects/{mid}/videos", json={"filename": "w.mp4", "camera_id": cam["id"]}).json()
    put(admin, j["upload_url"], b"video", "video/mp4")
    admin.post(f"/api/objects/jobs/{j['id']}/uploaded")
    with app.state.Session() as s:
        path = s.get(ObjectJob, j["id"]).storage_path
    assert admin.delete(f"/api/clients/{cl['id']}").status_code == 204
    job = next(x for x in admin.get(f"/api/objects/{mid}/jobs").json() if x["id"] == j["id"])
    assert job["status"] == "cancelado" and not app.state.storage.exists(path)
    assert mq.get("/api/agent/objects/next").status_code == 204
