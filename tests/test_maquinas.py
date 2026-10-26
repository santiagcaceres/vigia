"""Máquinas de análisis en la nube: reparto de cámaras, cuántas prender, el encendedor y el modelo por lotes."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from agent import maquina
from agent.escalador import Instance, decide
from server import pool
from server.main import Settings, create_app

ADMIN = ("admin", "clave-admin-123")
MODEL = Path(__file__).resolve().parents[1] / "models" / "personas.onnx"
UTC = timezone.utc


# ---------------------------------------------------------------- peso de cada cámara
def test_peso_segun_tipo_y_horario():
    now = datetime(2026, 10, 26, 15, 0, tzinfo=UTC)  # lunes 12:00 en Montevideo
    w = lambda **kw: pool.camera_weight(**{"fps": 5, "mode": "vivo", "brand": "ip", "hours": None,  # noqa: E731
                                           "tz": "America/Montevideo", "off_hours": "apagada", "now": now, **kw})
    assert w() == pytest.approx(1.0)  # hurto a 5 fps = una unidad
    assert w(fps=2) == pytest.approx(0.52)  # conteo
    assert w(mode="alarmas_equipo") == pytest.approx(0.1)
    abierto, cerrado = {"lun": "09:00-17:00"}, {"lun": "18:00-22:00"}
    assert w(hours=abierto) == pytest.approx(1.0)
    assert w(hours=cerrado) == 0  # local cerrado y «No mirar»: no ocupa nada
    assert w(hours=cerrado, off_hours="alarmas") == pytest.approx(0.05)
    assert w(hours=cerrado, off_hours="alarmas", brand="hikconnect") == 0
    assert w(hours=cerrado, off_hours="vivo") == pytest.approx(1.0)


# ---------------------------------------------------------------- reparto
def test_reparto_respeta_capacidad_no_mueve_de_mas_y_reparte_si_se_cae_una():
    cams = [(i, 1.0, None) for i in range(1, 7)]
    out = pool.assign(cams, {10: 3, 20: 3})
    assert sorted(out.values()).count(10) == 3 and sorted(out.values()).count(20) == 3
    # otra vuelta con lo ya asignado: no se mueve nada
    again = pool.assign([(c, 1.0, out[c]) for c in out], {10: 3, 20: 3})
    assert again == out
    # se cae la 20: todas a la 10 (aunque se pase de capacidad: mejor apretada que sin mirar)
    solo = pool.assign([(c, 1.0, out[c]) for c in out], {10: 3})
    assert set(solo.values()) == {10}
    # llega una tercera: las que no entran van a la más libre; las que entran se quedan
    tres = pool.assign([(c, 1.0, out[c]) for c in out] + [(7, 1.0, None), (8, 1.0, None)], {10: 3, 20: 3, 30: 3})
    assert all(tres[c] == out[c] for c in out) and tres[7] == tres[8] == 30
    assert pool.assign(cams, {}) == {}


def test_plan_cuantas_maquinas_y_cuales_apagar():
    m = lambda key, load, online=True, draining=False: pool.Machine(key, key, [1], online, draining, 30, load)  # noqa: E731
    # 60 unidades con máquinas de 30 al 85 % → 3 máquinas
    p = pool.plan(60, 60, [m("i-1", 30), m("i-2", 30)])
    assert p.desired == 3 and p.stop == [] and not any(p.draining.values())
    # mirando 20 minutos adelante: si van a abrir locales, ya pide máquinas
    assert pool.plan(0, 10, []).desired == 1
    # de noche: 0 unidades → todas se vacían; se apagan las que ya no tienen cámaras
    p = pool.plan(0, 0, [m("i-1", 0), m("i-2", 0.5), m("i-3", 0, online=False)])
    assert p.desired == 0 and p.draining == {"i-1": True, "i-2": True} and p.stop == ["i-1"]
    # sobra una: se vacía la menos cargada, no la que está trabajando
    p = pool.plan(20, 20, [m("i-1", 20), m("i-2", 0)])
    assert p.desired == 1 and p.draining == {"i-1": False, "i-2": True} and p.stop == ["i-2"]
    assert pool.plan(20, 20, [m("i-1", 20)], spare=1).desired == 2


# ---------------------------------------------------------------- encendedor
def test_encendedor_prende_apaga_y_no_se_pasa_del_tope():
    now = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)
    inst = [Instance("i-1", "running", now - timedelta(hours=2)), Instance("i-2", "stopped"),
            Instance("i-3", "stopped")]
    d = decide({"desired": 3, "stop": [], "online_instances": ["i-1"]}, inst, 10, can_launch=False, now=now)
    assert d.start == ["i-2", "i-3"] and d.launch == 0 and d.stop == []
    d = decide({"desired": 5, "stop": []}, inst, 10, can_launch=True, now=now)
    assert d.start == ["i-2", "i-3"] and d.launch == 2
    d = decide({"desired": 5, "stop": []}, inst, 10, can_launch=False, now=now)
    assert d.missing == 2
    assert decide({"desired": 50, "stop": []}, inst, 2, can_launch=True, now=now).start == ["i-2"]  # tope
    # sobran: se apagan solo las que el servidor ya vació, y nunca por debajo de lo pedido
    run = [Instance("i-1", "running", now), Instance("i-2", "running", now), Instance("i-3", "running", now)]
    d = decide({"desired": 1, "stop": ["i-2", "i-3"], "online_instances": ["i-1", "i-2", "i-3"]}, run, 10, False, now)
    assert d.stop == ["i-2", "i-3"]
    d = decide({"desired": 2, "stop": ["i-2", "i-3"], "online_instances": ["i-1", "i-2", "i-3"]}, run, 10, False, now)
    assert d.stop == ["i-2"]
    # una prendida hace rato que nunca se dio de alta (rota) se apaga si sobra
    rota = [Instance("i-1", "running", now - timedelta(hours=1)), Instance("i-9", "running", now - timedelta(hours=1))]
    d = decide({"desired": 1, "stop": [], "online_instances": ["i-1"]}, rota, 10, False, now)
    assert d.stop == ["i-9"]


def test_token_de_la_maquina_se_guarda(tmp_path):
    assert maquina.saved_token(tmp_path, "i-1") is None
    maquina.save_token(tmp_path, "abc", "i-1")
    assert maquina.saved_token(tmp_path, "i-1") == "abc"
    assert maquina.saved_token(tmp_path, "i-2") is None  # una copia clonada se da de alta de nuevo
    assert maquina.machine_name("i-0123", "2") == "maquina-i-0123-2"


# ---------------------------------------------------------------- modelo por lotes
def test_modelo_por_lotes_da_lo_mismo_que_de_a_uno():
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    if not MODEL.exists():
        pytest.skip("sin modelo")
    from core.lotes import make_batch_model

    bm = make_batch_model(MODEL)
    assert bm.col_of == {0: 4, 2: 5, 3: 6, 5: 7, 7: 8} and bm.size == (640, 640)
    orig = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    lote = ort.InferenceSession(bm.data, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(3)
    imgs = rng.integers(0, 256, (3, 640, 640, 3), dtype=np.uint8)  # BGR, como sale de OpenCV
    out = lote.run(None, {bm.input_name: imgs})[0]
    assert out.shape == (3, 9, 8400)
    for i in range(3):
        blob = imgs[i][:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        ref = orig.run(None, {"images": blob})[0][0]
        np.testing.assert_allclose(out[i], ref[[0, 1, 2, 3, 4, 6, 7, 9, 11]], rtol=1e-4, atol=1e-3)


def test_detector_por_lotes_junta_cuadros_de_varias_camaras():
    pytest.importorskip("onnx")
    if not MODEL.exists():
        pytest.skip("sin modelo")
    import threading

    from core.detectors import OnnxSession, OnnxYoloDetector

    sess = OnnxSession(MODEL, max_batch=8, max_wait=0.05)
    det = OnnxYoloDetector(session=sess)
    uno = OnnxYoloDetector(MODEL)
    frame = np.full((360, 640, 3), 120, np.uint8)
    assert det.detect(frame) == uno.detect(frame) == []
    sess.take_stats()
    hilos = [threading.Thread(target=det.detect, args=(frame,)) for _ in range(6)]
    for h in hilos:
        h.start()
    for h in hilos:
        h.join()
    frames, avg = sess.take_stats()
    assert frames == 6 and avg > 1  # se analizaron juntos


def test_maquina_sin_placa_usa_un_modelo_por_nucleo(monkeypatch):
    if not MODEL.exists():
        pytest.skip("sin modelo")
    import threading

    from core import detectors

    monkeypatch.setenv("VIGIA_HILOS", "3")
    monkeypatch.setattr(detectors, "_shared", {})
    monkeypatch.setattr(detectors, "_sessions", {})
    det = detectors.build_detector("yolo", MODEL)
    assert isinstance(det.runner, detectors.SessionPool) and len(det.runner.pool) == 3
    assert detectors.build_detector("yolo", MODEL, vehicles=True).runner is det.runner  # todas las cámaras comparten
    frame = np.full((360, 640, 3), 120, np.uint8)
    out: list = []
    hilos = [threading.Thread(target=lambda: out.append(det.detect(frame))) for _ in range(6)]
    for h in hilos:
        h.start()
    for h in hilos:
        h.join()
    assert out == [[]] * 6 and det.runner.take_stats()[0] == 6


# ---------------------------------------------------------------- servidor
@pytest.fixture
def app(tmp_path, db_url):
    return create_app(Settings(data_dir=tmp_path / "data", database_url=db_url, admin_password=ADMIN[1],
                               serve_public=False))


def test_flujo_maquinas_en_la_nube(app):
    with TestClient(app) as a:
        a.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]})
        dia_cerrado = ("lun", "mar", "mie", "jue", "vie", "sab", "dom")[(datetime.now(UTC).weekday() + 3) % 7]
        abierto = a.post("/api/clients", json={"name": "Abierto", "hours": {d: "00:00-24:00" for d in
                                                                             ("lun", "mar", "mie", "jue", "vie", "sab", "dom")}}).json()
        cerrado = a.post("/api/clients", json={"name": "Cerrado", "hours": {dia_cerrado: "10:00-11:00"}}).json()
        ana = a.post("/api/users", json={"username": "ana", "password": "clave-ana-123", "role": "entrenador"}).json()
        cam = lambda cl, n: a.post("/api/cameras", json={"client_id": cl["id"], "name": n, "brand": "ip",  # noqa: E731
                                                         "host": "dvr", "password": "x", "operator_id": ana["id"]}).json()
        cams = [cam(abierto, f"A{i}") for i in range(4)] + [cam(cerrado, "C1")]
        # modo «pcs» (como hasta ahora): las analiza la PC de quien las revisa
        assert all(c["operator_name"] == "ana" and c["analyzer_name"] == "PC de ana" for c in cams)

        # sin clave no se da de alta ninguna máquina
        assert TestClient(app).post("/api/agent/enroll", json={"key": "x" * 20, "name": "m1"}).status_code == 401
        key = a.post("/api/analysis/pool-key").json()["key"]
        m = {}
        for name, inst in (("m1", "i-1"), ("m2", "i-2")):
            r = TestClient(app).post("/api/agent/enroll", json={"key": key, "name": name, "instance_id": inst,
                                                                "capacity": 3})
            assert r.status_code == 200, r.text
            m[name] = TestClient(app, headers={"X-Agent-Token": r.json()["token"]})
            m[name].post("/api/agent/status", json={"cameras": [], "capacity": 3, "instance_id": inst})
        # en modo «pcs» las máquinas no reciben nada
        assert m["m1"].get("/api/agent/config").json()["cameras"] == []

        # pasar a «servidores»: se reparten entre las dos máquinas; la PC de ana ya no analiza
        r = a.put("/api/analysis", json={"mode": "servidores", "capacity": 3}).json()
        assert r["mode"] == "servidores" and len(r["machines"]) == 2
        tok = a.post("/api/agent/session").json()  # (la PC del admin: no tiene cámaras)
        assert tok["analysis_mode"] == "servidores"
        n1 = len(m["m1"].get("/api/agent/config").json()["cameras"])
        n2 = len(m["m2"].get("/api/agent/config").json()["cameras"])
        assert n1 + n2 == 5 and n1 >= 2 and n2 >= 2
        lst = {c["name"]: c for c in a.get("/api/cameras").json()}
        assert lst["A0"]["operator_name"] == "ana" and lst["A0"]["analyzer_name"] in ("m1", "m2")
        # la empleada sigue viendo las cámaras que revisa
        e = TestClient(app)
        e.post("/api/login", json={"username": "ana", "password": "clave-ana-123"})
        assert len(e.post("/api/agent/session").json()["camera_ids"]) == 5

        # el estado avisa cuándo pedir la configuración: cambia la versión al tocar una cámara
        st1 = m["m1"].post("/api/agent/status", json={"cameras": []}).json()
        a.put(f"/api/cameras/{cams[0]['id']}/analysis", json={"zones": [], "rules": {}})
        st2 = m["m1"].post("/api/agent/status", json={"cameras": []}).json()
        assert st2["rev"] != st1["rev"] and isinstance(st2["cameras"], list)

        # se cae m2 (90 s sin noticias): sus cámaras pasan a m1
        from server.db import Agent
        with app.state.Session() as s:
            ag = s.query(Agent).filter_by(name="m2").one()
            ag.last_seen = datetime.now(UTC).replace(tzinfo=None) - timedelta(minutes=5)
            s.commit()
        a.put("/api/analysis", json={})  # fuerza un reparto
        assert len(m["m1"].get("/api/agent/config").json()["cameras"]) == 5

        # el encendedor: 4 cámaras abiertas (la del local cerrado no cuenta) con capacidad 3 → 2 máquinas
        plan = TestClient(app).get("/api/agent/scale", headers={"X-Pool-Key": key}).json()
        assert plan["cameras_active"] == 4 and plan["plan"]["desired"] == 2
        assert plan["online_instances"] == ["i-1"]
        assert TestClient(app).get("/api/agent/scale", headers={"X-Pool-Key": "otra-clave-x"}).status_code == 401

        # deshabilitar todo lo abierto: no hace falta ninguna máquina → se vacía m1 y aparece para apagar
        for c in cams[:4]:
            a.put(f"/api/cameras/{c['id']}", json={"client_id": abierto["id"], "name": c["name"], "brand": "ip",
                                                   "host": "dvr", "enabled": False})
        plan = TestClient(app).get("/api/agent/scale", headers={"X-Pool-Key": key}).json()
        assert plan["plan"]["desired"] == 0 and plan["plan"]["stop"] == ["i-1"]
        assert m["m1"].post("/api/agent/status", json={"cameras": []}).json()["draining"] is True

        # volver a «pcs»: cada cámara vuelve a la PC de quien la revisa
        a.put("/api/analysis", json={"mode": "pcs"})
        assert {c["analyzer_name"] for c in a.get("/api/cameras").json()} == {"PC de ana"}
        assert m["m1"].get("/api/agent/config").json()["cameras"] == []


def test_volver_a_pcs_devuelve_las_camaras_sin_revisor_a_su_pc_vieja(app):
    """En producción hay cámaras en PCs vinculadas a la antigua (sin usuario que las revise)."""
    with TestClient(app) as a:
        a.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]})
        cl = a.post("/api/clients", json={"name": "X"}).json()
        pc = a.post("/api/agents", json={"name": "PC vieja"}).json()
        c = a.post("/api/cameras", json={"client_id": cl["id"], "name": "Vieja", "brand": "ip", "host": "dvr",
                                         "password": "x", "agent_id": pc["id"]}).json()
        assert c["analyzer_name"] == "PC vieja" and c["operator_name"] is None
        key = a.post("/api/analysis/pool-key").json()["key"]
        tok = TestClient(app).post("/api/agent/enroll", json={"key": key, "name": "m1", "instance_id": "i-1"}).json()
        TestClient(app, headers={"X-Agent-Token": tok["token"]}).post("/api/agent/status", json={"cameras": []})
        a.put("/api/analysis", json={"mode": "servidores"})
        assert a.get("/api/cameras").json()[0]["analyzer_name"] == "m1"
        a.put("/api/analysis", json={"mode": "pcs"})
        assert a.get("/api/cameras").json()[0]["analyzer_name"] == "PC vieja"


def test_id_de_la_instancia_en_aws_y_google_cloud(monkeypatch):
    """Google Cloud también contesta en 169.254.169.254: al PUT de AWS le responde 405 con una página HTML (con
    «’»), que no se puede usar como token. Antes eso tiraba la máquina al arrancar."""
    import httpx

    monkeypatch.delenv("VIGIA_INSTANCE_ID", raising=False)

    def nube(put, get):
        monkeypatch.setattr(maquina.httpx, "put", put)
        monkeypatch.setattr(maquina.httpx, "get", get)
        return maquina.instance_id()

    def gcp_put(url, **_):
        return httpx.Response(405, text="<html>That’s an error. Method not allowed</html>")

    def gcp_get(url, headers=None, **_):
        if url.endswith("/computeMetadata/v1/instance/name") and (headers or {}).get("Metadata-Flavor") == "Google":
            return httpx.Response(200, text="vigia-analisis-1")
        return httpx.Response(404, text="no")

    assert nube(gcp_put, gcp_get) == "vigia-analisis-1"

    def aws_put(url, **_):
        return httpx.Response(200, text="token-aws")

    def aws_get(url, headers=None, **_):
        if url.endswith("/latest/meta-data/instance-id") and (headers or {}).get("X-aws-ec2-metadata-token") == "token-aws":
            return httpx.Response(200, text="i-0abc123")
        return httpx.Response(404, text="no")

    assert nube(aws_put, aws_get) == "i-0abc123"

    def caida(url, **_):
        raise httpx.ConnectTimeout("sin metadata")

    assert nube(caida, caida) is None  # otra nube o una PC: se usa el nombre de la máquina
    monkeypatch.setenv("VIGIA_INSTANCE_ID", "a-mano")
    assert maquina.instance_id() == "a-mano"
