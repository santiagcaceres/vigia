"""Tope de las máquinas (24 nov, decisión del dueño: «para mí lo mejor es poner un tope a las máquinas; meterle cámaras
para que las analice mal no tiene sentido»). El tope se aplica al marcar una cámara para analizar: si no entra, no se
la marca y se explica. El reparto nunca le saca la máquina a una cámara que ya se analiza (un reinicio no puede dejar
afuera a las que andan), y una cámara que no da imagen hace días no ocupa lugar."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from server import pool
from server.main import Settings, create_app

ADMIN = ("admin", "clave-admin-123")
UTC = timezone.utc


def test_los_procesos_de_una_maquina_comparten_el_lugar_y_nadie_queda_sin_mirar():
    """La máquina de AWS: 4 procesos de 8. El lugar es de la máquina entera, no de cada proceso: antes 7 cámaras de
    1,3 no entraban en un proceso (9,1 > 8) y una iba y venía cada vez que un local abría o cerraba."""
    servers = {18: 8.0, 19: 8.0, 20: 8.0, 21: 8.0}
    groups = dict.fromkeys(servers, "i-aws")
    assert pool.machine_caps(servers, groups) == {"i-aws": 32.0}
    assert pool.machine_caps(servers, groups, 30.0) == {"i-aws": 30.0}
    cams = [(i, 1.3, 18 + i % 4) for i in range(24)]  # 6 por proceso = 7,8; en total 31,2
    assert pool.assign(cams, servers, groups) == {i: 18 + i % 4 for i in range(24)}  # nadie se mueve
    # una nueva va al proceso menos cargado; las demás no se mueven
    out = pool.assign([*cams, (99, 0.52, None)], servers, groups)
    assert out[99] in servers and all(out[i] == 18 + i % 4 for i in range(24))
    # aunque la máquina quede pasada (bajaron el tope), el reparto no deja a ninguna sin mirar ni las mueve
    out = pool.assign(cams, servers, groups, cap=20.0)
    assert out == {i: 18 + i % 4 for i in range(24)}
    assert pool.assign(list(zip(out, [1.3] * 24, out.values())), servers, groups, cap=20.0) == out  # y queda quieto
    # un proceso muy cargado (más del 25 % de su parte) le pasa cámaras a otro de la misma máquina
    out = pool.assign([(i, 1.3, 18) for i in range(12)], servers, groups)
    assert len(out) == 12 and sum(1 for a in out.values() if a == 18) == 7 and set(out.values()) > {18}
    # dos máquinas: si una está pasada y la otra tiene lugar, las que sobran pasan a la otra (como siempre)
    two = pool.assign([(i, 1.0, 1) for i in range(5)], {1: 3.0, 2: 3.0})
    assert sorted(two.values()) == [1, 1, 1, 2, 2]
    # y si se cae una, van todas a la que queda (el reparto no deja a ninguna sin mirar)
    assert set(pool.assign([(i, 1.0, two[i]) for i in two], {1: 3.0}).values()) == {1}


def test_las_que_andan_se_quedan_donde_estan():
    assert [pool.life_rank(x) for x in (5, 1799, 1800, 3600 * 71, 3600 * 73, None)] == [0, 0, 1, 1, 2, 2]
    # el caso real del 24 nov: 20 andando, 4 con el local cerrado y 4 que no andan, en un proceso con 10 cámaras
    cams, rank = [], {}
    for i in range(20):
        cams.append((200 + i, 1.3, 18 + i % 4))
        rank[200 + i] = pool.life_rank(60)
    for i in range(4):
        cams.append((100 + i, 1.3, 18))
        rank[100 + i] = pool.life_rank(None)
    for cid, w, age in ((50, 0.52, None), (51, 1.3, 17 * 3600), (52, 0.52, 9 * 3600), (53, 0.52, 9 * 3600)):
        cams.append((cid, w, 19))
        rank[cid] = pool.life_rank(age)
    servers = {18: 8.0, 19: 8.0, 20: 8.0, 21: 8.0}
    groups = dict.fromkeys(servers, "i-aws")
    out = pool.assign(cams, servers, groups, rank, 30.0)
    assert len(out) == 28  # todas siguen con máquina
    assert all(out[c] == a for c, _, a in cams if c >= 200)  # y ninguna de las que andan cambia de proceso
    again = pool.assign([(c, w, out[c]) for c, w, _ in cams], servers, groups, rank, 30.0)
    assert again == out  # otra vuelta: nada se mueve
    # después de un reinicio (nadie dio imagen en la última media hora): tampoco se mueve ninguna de las que andaban
    cold = {c: (1 if r == 0 else r) for c, r in rank.items()}
    assert pool.assign([(c, w, out[c]) for c, w, _ in cams], servers, groups, cold, 30.0) == out


# ---------------------------------------------------------------- en el servidor
@pytest.fixture
def app(tmp_path, db_url):
    return create_app(Settings(data_dir=tmp_path / "data", database_url=db_url, admin_password=ADMIN[1],
                               serve_public=False))


def _machine(app, key: str, name: str, inst: str, capacity: float):
    r = TestClient(app).post("/api/agent/enroll", json={"key": key, "name": name, "instance_id": inst,
                                                        "capacity": capacity})
    assert r.status_code == 200, r.text
    m = TestClient(app, headers={"X-Agent-Token": r.json()["token"]})
    m.post("/api/agent/status", json={"cameras": [], "capacity": capacity, "instance_id": inst})
    return m


def _cams(a, client: dict, n: int) -> list[int]:
    return [a.post("/api/cameras", json={"client_id": client["id"], "name": f"Canal {i}", "brand": "hikvision",
                                         "host": "dvr", "password": "x"}).json()["id"] for i in range(1, n + 1)]


def _old(app, cam_id: int, **days) -> None:
    """Como si hubiera pasado el tiempo: `image` = hace cuántos días dio la última imagen, `since` = hace cuántos se
    la marcó para analizar."""
    from server.db import Camera
    with app.state.Session() as s:
        c = s.get(Camera, cam_id)
        now = datetime.now(UTC).replace(tzinfo=None)
        if "image" in days:
            c.preview_at = None if days["image"] is None else now - timedelta(days=days["image"])
        if "since" in days:
            c.analyze_since = now - timedelta(days=days["since"])
        s.commit()


def test_empezar_a_analizar_con_la_maquina_llena_avisa_y_no_la_marca(app):
    with TestClient(app) as a:
        a.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]})
        ids = _cams(a, a.post("/api/clients", json={"name": "Farmacia"}).json(), 5)
        key = a.post("/api/analysis/pool-key").json()["key"]
        # sin ninguna máquina en línea no se frena nada (empieza sola cuando se prenda una)
        assert a.post("/api/cameras/analyze", json={"camera_ids": ids[:1], "analyze": True}).status_code == 200
        # una máquina de 2 procesos de 2: aguanta 4 (3 cámaras con gestos de 1,3 = 3,9)
        procs = [_machine(app, key, f"maq-{n}", "i-1", 2) for n in (1, 2)]
        assert a.post("/api/cameras/analyze", json={"camera_ids": ids[1:3], "analyze": True}).status_code == 200
        got = {c["id"]: c for c in a.get("/api/cameras").json()}
        assert all(got[i]["agent_id"] for i in ids[:3])
        an = a.get("/api/analysis").json()
        assert an["room"] == {"capacity": 4.0, "reserved": 3.9, "idle": 0}
        assert an["machines"][0]["reserved"] == 3.9 and an["machines"][0]["capacity"] == 4.0
        # la cuarta no entra: 409 con la explicación, y no queda marcada
        r = a.post("/api/cameras/analyze", json={"camera_ids": [ids[3]], "analyze": True})
        assert r.status_code == 409, r.text
        msg = r.json()["detail"]
        assert "No hay lugar" in msg and "el tope es 4" in msg and "3,9 ocupado" in msg and "ocupa 1,3" in msg
        c4 = a.get(f"/api/cameras/{ids[3]}").json()
        assert not c4["analyzed"] and c4["agent_id"] is None
        # tampoco entran dos juntas; volver a marcar las que ya se analizan no se niega
        r = a.post("/api/cameras/analyze", json={"camera_ids": ids[3:], "analyze": True})
        assert r.status_code == 409 and "estas 2 cámaras ocupan 2,6" in r.json()["detail"]
        assert a.post("/api/cameras/analyze", json={"camera_ids": ids[:3], "analyze": True}).status_code == 200

        # una de las tres no anda desde ayer: sigue ocupando su lugar (puede volver) y el aviso la nombra
        _old(app, ids[2], image=1, since=10)
        for i in ids[:2]:
            _old(app, i, image=0, since=10)
        owner = {c["id"]: c["agent_id"] for c in a.get("/api/cameras").json()}
        for p in procs:
            aid = p.get("/api/agent/config").json()["agent_id"]
            p.post("/api/agent/status", json={"instance_id": "i-1", "capacity": 2, "cameras": [
                {"camera_id": i, "state": "sin_conexion" if i == ids[2] else "en_linea", "state_age_s": 90000}
                for i in ids[:3] if owner[i] == aid]})
        r = a.post("/api/cameras/analyze", json={"camera_ids": [ids[3]], "analyze": True})
        assert r.status_code == 409
        assert "«Farmacia · Canal 3» (sin imagen hace 1 día, ocupa 1,3)" in r.json()["detail"]
        # pasan los días sin dar imagen: deja de ocupar lugar (sigue asignada: si vuelve, arranca sola) y la cuarta entra
        _old(app, ids[2], image=4)
        an = a.get("/api/analysis").json()
        assert an["room"] == {"capacity": 4.0, "reserved": 2.6, "idle": 1}
        assert a.post("/api/cameras/analyze", json={"camera_ids": [ids[3]], "analyze": True}).status_code == 200
        got = {c["id"]: c for c in a.get("/api/cameras").json()}
        assert got[ids[3]]["agent_id"] and got[ids[2]]["agent_id"]  # las dos con máquina
        # si la que no andaba vuelve, la máquina queda pasada: se ve, y no se agregan más hasta bajar
        _old(app, ids[2], image=0)
        an = a.get("/api/analysis").json()
        assert an["room"] == {"capacity": 4.0, "reserved": 5.2, "idle": 0}
        assert a.post("/api/cameras/analyze", json={"camera_ids": [ids[4]], "analyze": True}).status_code == 409
        # se saca una de análisis: deja el lugar
        assert a.post("/api/cameras/analyze", json={"camera_ids": ids[2:4], "analyze": False}).status_code == 200
        assert a.post("/api/cameras/analyze", json={"camera_ids": [ids[4]], "analyze": True}).status_code == 200

        # bajar el tope desde la app no le saca la máquina a ninguna: solo frena las nuevas
        an = a.put("/api/analysis", json={"capacity": 3}).json()
        assert an["room"] == {"capacity": 3.0, "reserved": 3.9, "idle": 0}
        assert sum(len(p.get("/api/agent/config").json()["cameras"]) for p in procs) == 3


def test_el_local_cerrado_no_le_deja_su_lugar_a_otra(app):
    """El tope se cuenta con el local abierto: una cámara que hoy no se mira porque el local está cerrado mañana
    vuelve a necesitar su lugar."""
    with TestClient(app) as a:
        a.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]})
        dia = ("lun", "mar", "mie", "jue", "vie", "sab", "dom")[(datetime.now(UTC).weekday() + 3) % 7]
        c1, c2 = _cams(a, a.post("/api/clients", json={"name": "Cerrado", "hours": {dia: "10:00-11:00"}}).json(), 2)
        (c3,) = _cams(a, a.post("/api/clients", json={"name": "Abierto"}).json(), 1)
        key = a.post("/api/analysis/pool-key").json()["key"]
        _machine(app, key, "maq", "i-1", 3)
        assert a.post("/api/cameras/analyze", json={"camera_ids": [c1, c2], "analyze": True}).status_code == 200
        an = a.get("/api/analysis").json()
        assert an["cameras_active"] == 0 and an["room"]["reserved"] == 2.6  # ahora no pesan, pero ocupan su lugar
        assert a.post("/api/cameras/analyze", json={"camera_ids": [c3], "analyze": True}).status_code == 409


def test_un_reinicio_no_le_saca_la_maquina_a_ninguna(app):
    """Los procesos de una máquina vuelven de a uno después de un reinicio: mientras tanto ninguna cámara queda sin
    máquina ni cambia de proceso (las que andan no pueden perder su lugar por el orden en que vuelven)."""
    with TestClient(app) as a:
        a.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]})
        ids = _cams(a, a.post("/api/clients", json={"name": "Bazar"}).json(), 6)
        key = a.post("/api/analysis/pool-key").json()["key"]
        procs = [_machine(app, key, f"maq-{n}", "i-1", 4) for n in (1, 2)]
        assert a.post("/api/cameras/analyze", json={"camera_ids": ids, "analyze": True}).status_code == 200
        before = {c["id"]: c["agent_id"] for c in a.get("/api/cameras").json()}
        assert all(before.values()) and len(set(before.values())) == 2
        from server.db import Agent
        with app.state.Session() as s:  # un proceso tarda más de 90 s en volver
            s.query(Agent).filter_by(name="maq-2").one().last_seen = datetime.now(UTC).replace(tzinfo=None) - timedelta(minutes=5)
            s.commit()
        a.put("/api/analysis", json={})  # fuerza un reparto con un solo proceso en línea
        mid = {c["id"]: c["agent_id"] for c in a.get("/api/cameras").json()}
        assert all(mid.values())  # ninguna quedó sin máquina
        procs[1].post("/api/agent/status", json={"cameras": [], "capacity": 4, "instance_id": "i-1"})
        a.put("/api/analysis", json={})
        after = {c["id"]: c["agent_id"] for c in a.get("/api/cameras").json()}
        assert all(after.values()) and len(set(after.values())) == 2


# ---------------------------------------------------------------- en la app
def test_la_app_muestra_el_tope():
    from agent.oficina import machine_load, room_text

    assert machine_load({"load": 26.0, "reserved": 29.9, "capacity": 30.0}) == "29,9 / 30 · al tope"
    assert machine_load({"load": 26.0, "reserved": 26.0, "capacity": 30.0}) == "26 / 30"
    assert machine_load({"load": 26.0, "reserved": 31.2, "capacity": 30.0}) == "31,2 / 30 · pasada"
    assert machine_load({"load": 9.1, "capacity": 8.0}) == "9,1 / 8"  # (servidor sin actualizar)
    txt = room_text({"room": {"capacity": 30.0, "reserved": 28.34, "idle": 5}})
    assert txt.startswith("Tope: las máquinas en línea aguantan 30 · ocupado 28,3 · libre 1,7.")
    assert "5 cámaras marcadas para analizar no dan imagen hace más de 3 días: no ocupan lugar." in txt
    over = room_text({"room": {"capacity": 30.0, "reserved": 31.2, "idle": 1}})
    assert over.startswith("⚠ Tope:") and "pasadas por 1,2" in over and "1 cámara marcada para analizar no da imagen" in over
    assert "ninguna máquina de análisis en línea" in room_text({"room": {"capacity": 0, "reserved": 5, "idle": 0}})
    assert room_text({}) == ""


def test_el_video_que_llega_y_cuales_pesan_de_mas():
    """Decodificar el video se lleva un tercio del procesador de la máquina: la app dice qué manda cada cámara y marca
    las que pesan mucho más que la segunda calidad que se pide (640×360 a 8 cuadros por segundo)."""
    import cv2

    from agent.oficina import video_text
    from agent.worker import video_info

    class Cap:
        def __init__(self, four: str, w: int, h: int, fps: float):
            self.v = {cv2.CAP_PROP_FOURCC: float(int.from_bytes(four.encode(), "little")),
                      cv2.CAP_PROP_FRAME_WIDTH: float(w), cv2.CAP_PROP_FRAME_HEIGHT: float(h), cv2.CAP_PROP_FPS: fps}

        def get(self, prop):
            return self.v.get(prop, 0.0)

    main = video_info(Cap("hevc", 1920, 1080, 25.0))
    assert main == {"video_codec": "H.265", "video_w": 1920, "video_h": 1080, "video_fps": 25.0}
    assert video_text(main) == ("H.265 1920×1080 · 25 cuadros/s", 28)
    sub = video_info(Cap("avc1", 640, 360, 8.0))
    assert video_text(sub) == ("H.264 640×360 · 8 cuadros/s", 0)
    assert video_text(video_info(Cap("h264", 704, 576, 25.0))) == ("H.264 704×576 · 25 cuadros/s", 6)
    # un grabador que dice cualquier cosa (90000 cuadros/s, sin tamaño) no rompe nada ni marca pesado
    assert video_info(Cap("\x00\x00\x00\x00", 0, 0, 90000.0)) == {}
    assert video_text({}) == ("", 0) and video_text(None) == ("", 0)
    assert video_text({"video_w": 1280, "video_h": 720}) == ("1280×720", 0)
