"""Plan «Gastronomía» (12 nov, el dueño: «contar cuánto demora un mozo en ir a atenderlos, cuánto tiempo promedio están
las personas en las mesas, cuánto demora hasta que les llega la comida»): zona «Mesa» (core/metrics.py)."""
import types
from datetime import datetime, timedelta

from core.metrics import (TABLE_GRACE_SECONDS, MetricsCollector, empty_bucket, merge_bucket, restrict_bucket,
                          table_stats)
from core.rules import parse_config, parse_zones

MESA = {"name": "Mesa 1", "type": "table", "polygon": [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]]}
BARRA = {"name": "Barra", "type": "staff", "polygon": [[0.0, 0.0], [0.2, 0.0], [0.2, 0.2], [0.0, 0.2]]}
W, H = 1000, 1000
T0 = datetime(2026, 11, 12, 12, 0, 0)


def _track(tid, x, y, first=0.0):
    """Una persona con los pies en (x, y) (0-1)."""
    return types.SimpleNamespace(id=tid, label="persona", first_ts=first, visible=True,
                                 bbox=(x * W - 20, y * H - 100, x * W + 20, y * H))


def _run(zones, script, seconds, fps=2):
    """`script(t)` → lista de (id, x, y, desde) a la vista en el segundo t."""
    mc = MetricsCollector(parse_zones(zones))
    for i in range(int(seconds * fps) + 1):
        ts = i / fps
        mc.update([_track(*p) for p in script(ts)], ts, (W, H), T0 + timedelta(seconds=ts))
    live = mc.live_counts()
    total = empty_bucket()
    for b in mc.close():
        total = merge_bucket(total, b["data"])
    return total, live


def test_ocupacion_atencion_y_entrega_con_mozo_del_personal():
    def script(t):
        out = []
        if 10 <= t <= 610:  # dos clientes sentados 10 minutos
            out += [(1, 0.45, 0.5, 10.0), (2, 0.55, 0.5, 10.0)]
        # el mozo: está en la barra, va a la mesa a los 100 s (pedido) y a los 400 s (comida), 6 s cada vez
        if t < 95 or 110 < t < 395 or t > 410:
            out.append((9, 0.1, 0.1, 0.0))
        elif 100 <= t <= 106 or 400 <= t <= 406:
            out.append((9, 0.5, 0.58, 0.0))
        else:
            out.append((9, 0.3, 0.3, 0.0))  # caminando
        return out
    total, _ = _run([MESA, BARRA], script, 610 + TABLE_GRACE_SECONDS + 5)
    st = table_stats(total["tables"]["Mesa 1"])
    assert st["n"] == 1 and 595 <= st["avg_s"] <= 601 and st["party_avg"] == 2
    assert st["attend_n"] == 1 and 89 <= st["attend_avg_s"] <= 91  # se sentaron a los 10 s, el mozo llegó a los 100
    assert st["serve_n"] == 1 and 389 <= st["serve_avg_s"] <= 391  # la segunda visita, a los 400
    assert st["visits"] == 2 and st["no_visit"] == 0
    assert 85 < st["occupied_pct"] < 92  # 600 de ~675 s mirados
    assert total["staff"] == 1 and total["people"] == 2  # el mozo no cuenta como cliente ni ocupa la mesa


def test_sin_zona_de_personal_el_que_llega_y_se_va_es_una_visita():
    def script(t):
        out = []
        if 0 <= t <= 300:
            out.append((1, 0.5, 0.5, 0.0))
        if 60 <= t <= 80:  # alguien llega caminando, se queda 8 s en la mesa y se va
            x = 0.9 if t < 66 or t > 74 else 0.5
            out.append((5, x, 0.55, 60.0))
        if 150 <= t <= 300:  # un comensal que llega tarde y se queda: no es visita
            out.append((6, 0.9 if t < 153 else 0.48, 0.52, 150.0))
        return out
    total, _ = _run([MESA], script, 300 + TABLE_GRACE_SECONDS + 5)
    st = table_stats(total["tables"]["Mesa 1"])
    assert st["n"] == 1 and st["visits"] == 1 and st["attend_n"] == 1 and 64 <= st["attend_avg_s"] <= 68
    assert st["serve_n"] == 0 and st["serve_avg_s"] is None and st["party_avg"] == 2


def test_limpiar_la_mesa_o_pasar_de_largo_no_es_ocuparla():
    def script(t):
        out = []
        if 10 <= t <= 40:  # 30 s acomodando la mesa
            out.append((1, 0.5, 0.5, 10.0))
        if 100 <= t <= 104:  # uno que pasa entre las mesas
            out.append((2, 0.5, 0.45, 95.0))
        return out
    total, live = _run([MESA], script, 250)
    st = table_stats(total["tables"]["Mesa 1"])
    assert st["n"] == 0 and st["occ_s"] == 0 and st["visits"] == 0 and st["obs_s"] > 240
    assert live["zones"]["Mesa 1"]["table"]["busy"] is False


def test_en_vivo_dice_si_esta_ocupada_y_cuanto_lleva():
    total, live = _run([MESA], lambda t: [(1, 0.5, 0.5, 0.0)], 200)
    tb = live["zones"]["Mesa 1"]["table"]
    assert tb["busy"] is True and 195 <= tb["for_s"] <= 201 and tb["n"] == 1
    # al cerrar la cámara, la ocupación en curso queda contada hasta donde se la vio
    assert table_stats(total["tables"]["Mesa 1"])["n"] == 1


def test_las_mesas_viajan_aparte_y_son_del_plan_gastronomia():
    # un agente que no conoce «mesa» no la recibe en `zones` ni en `extra_zones`
    zones, _ = parse_config({"zones": [], "extra_zones": [BARRA], "table_zones": [MESA]})
    assert [z.type for z in zones] == ["staff", "table"]
    b = dict(empty_bucket(), tables={"Mesa 1": {"n": 3, "sum_s": 900.0}}, people=5)
    # el cliente tiene gastronomía y esta cámara no la mide: sus mesas no se suman
    assert restrict_bucket(b, ["comercio", "gastronomia"], ["comercio"])["tables"] == {}
    assert restrict_bucket(b, ["comercio", "gastronomia"], ["gastronomia"])["tables"]["Mesa 1"]["n"] == 3
    # sumar cubos: los máximos se quedan con el mayor y un dato roto no rompe nada
    m = merge_bucket({"tables": {"M": {"n": 1, "attend_max_s": 50, "sum_s": "x"}}},
                     {"tables": {"M": {"n": 2, "attend_max_s": 80, "sum_s": 100}}})
    assert m["tables"]["M"]["n"] == 3 and m["tables"]["M"]["attend_max_s"] == 80 and m["tables"]["M"]["sum_s"] == 100


# ------------------------------------------------------------------ servidor
import csv  # noqa: E402
import io  # noqa: E402
from zoneinfo import ZoneInfo  # noqa: E402

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from server.main import METRICS_COUNT_FPS, TABLE_CSV_HEAD, Settings, create_app  # noqa: E402

ADMIN = ("admin", "clave-admin-123")
TZ = "America/Montevideo"


@pytest.fixture
def app(tmp_path, db_url):
    return create_app(Settings(data_dir=tmp_path / "data", database_url=db_url, admin_password=ADMIN[1],
                               serve_public=False))


def test_plan_gastronomia_de_punta_a_punta_en_el_servidor(app):
    a = TestClient(app)
    assert a.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]}).status_code == 200
    r = a.post("/api/clients", json={"name": "Café Central", "plans": ["gastronomia"], "timezone": TZ})
    assert r.status_code == 201 and r.json()["plans"] == ["gastronomia"]
    cid = r.json()["id"]
    ag = a.post("/api/agents", json={"name": "central", "client_id": None}).json()
    body = {"client_id": cid, "brand": "ip", "host": "dvr", "password": "x", "agent_id": ag["id"]}
    salon = a.post("/api/cameras", json={**body, "name": "Salón"}).json()
    assert salon["measures"] == ["gastronomia"]
    r = a.put(f"/api/cameras/{salon['id']}/analysis", json={"zones": [dict(MESA, wait=True), BARRA], "rules": {}})
    assert r.status_code == 200, r.text
    saved = {z["name"]: z for z in a.get(f"/api/cameras/{salon['id']}").json()["zones"]}
    assert saved["Mesa 1"]["type"] == "table" and "wait" not in saved["Mesa 1"]  # la mesa no mide «espera»
    # a las máquinas: la mesa viaja en su lista (una máquina sin reiniciar no la conoce), solo métricas y a 3 cuadros/s
    cam = a.get("/api/agent/config", headers={"X-Agent-Token": ag["token"]}).json()["cameras"][0]
    an = cam["analysis"]
    assert [z["name"] for z in an["table_zones"]] == ["Mesa 1"] and an["metrics_only"] is True
    assert all(z["type"] != "table" for z in an["zones"] + an.get("extra_zones", []))
    assert cam["process_fps"] == METRICS_COUNT_FPS
    assert {z.type for z in parse_config(an)[0]} == {"table", "staff"}
    # las métricas que manda la máquina → resumen, Excel y PDF
    today = datetime.now(ZoneInfo(TZ)).replace(hour=10, minute=0, second=0, microsecond=0, tzinfo=None)
    data = dict(empty_bucket(), frames=500, tables={"Mesa 1": {
        "n": 4, "sum_s": 7200.0, "max_s": 3000.0, "party": 10, "obs_s": 3600.0, "occ_s": 2700.0, "visits": 7,
        "no_visit": 1, "attend_n": 3, "attend_s": 540.0, "attend_max_s": 300.0, "serve_n": 2, "serve_s": 1800.0,
        "serve_max_s": 1000.0}})
    r = a.post("/api/agent/metrics", headers={"X-Agent-Token": ag["token"]}, json={"buckets": [
        {"camera_id": salon["id"], "bucket_start": today.isoformat(), "data": data}]})
    assert r.json()["saved"] == 1
    m = a.get("/api/metrics", params={"client_id": cid, "days": 1}).json()
    st = table_stats(m["total"]["tables"]["Mesa 1"])
    assert (st["n"], st["avg_s"], st["occupied_pct"], st["party_avg"]) == (4, 1800.0, 75.0, 2.5)
    assert (st["attend_avg_s"], st["serve_avg_s"], st["no_visit"]) == (180.0, 900.0, 1)
    assert m["by_camera"][0]["tables"] == m["total"]["tables"]
    day = today.date().isoformat()
    text = a.get("/api/metrics/export", params={"client_id": cid, "desde": day, "hasta": day}).text.lstrip("﻿")
    row = list(csv.DictReader(io.StringIO(text), delimiter=";"))[0]
    assert [row[h] for h in TABLE_CSV_HEAD] == ["4", "30,0", "75", "2,5", "3,0", "15,0", "1"]
    pdf = a.get("/api/metrics/report.pdf", params={"client_id": cid, "days": 1})
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")
    pypdf = pytest.importorskip("pypdf")
    words = " ".join("".join(p.extract_text() for p in pypdf.PdfReader(io.BytesIO(pdf.content)).pages).split())
    assert "Gastronom" in words and "Mesa 1" in words and "30:00" in words and "3:00" in words and "15:00" in words
    # una cámara que no mide gastronomía (el cliente también tiene comercio): sus mesas no se suman
    a.put(f"/api/clients/{cid}", json={"name": "Café Central", "plans": ["gastronomia", "comercio"], "timezone": TZ,
                                       "known_plans": ["hurto", "comercio", "estacion", "empleados", "gastronomia"]})
    a.post("/api/cameras/plans", json={"camera_ids": [salon["id"]], "plans": ["comercio"],
                                       "known_plans": ["hurto", "comercio", "estacion", "empleados", "gastronomia"]})
    assert a.get("/api/metrics", params={"client_id": cid, "days": 1}).json()["total"]["tables"] == {}
