"""Tiempo de espera por zona y seguimiento de empleados (pedido del dueño, 3 nov: «medir el tiempo de espera promedio
de las personas en una cierta zona, por ejemplo mostradores… y además seguir empleados como otra función de
métricas»). Se mide con los mismos tracks que las demás métricas (core/metrics.py)."""

from datetime import datetime, timedelta

import pytest

from core.metrics import (MetricsCollector, WAIT_MIN_SECONDS, empty_bucket, merge_bucket, post_stats,
                          restrict_bucket, wait_stats)
from core.rules import parse_zones
from core.tracker import Track

W, H = 1000, 1000
ZONES = parse_zones([
    {"name": "Fila caja", "type": "checkout", "wait": True, "polygon": [[0, 0.5], [0.5, 0.5], [0.5, 1], [0, 1]]},
    {"name": "Mostrador", "type": "staff", "polygon": [[0.6, 0.5], [1, 0.5], [1, 1], [0.6, 1]]},
    {"name": "Pasillo", "type": "aisle", "polygon": [[0, 0], [1, 0], [1, 0.4], [0, 0.4]]},
])
WALL0 = datetime(2026, 11, 3, 10, 0, 0)


def person(tid, x, y, first, ts):
    return Track(tid, (x - 20, y - 80, x + 20, y), "persona", first, ts, hits=10)


# (id, desde, hasta, x, y): quién está dónde (2 cuadros por segundo)
PEOPLE = [
    (1, 0, 90, 100, 800),    # A: espera 90 s en la fila (el empleado llega a los 20 s: lo atienden)
    (5, 0, 15, 200, 800),    # E: espera 15 s y se va antes de que llegue nadie al mostrador
    (2, 10, 40, 300, 800),   # B: espera 30 s
    (3, 50, 52, 100, 800),   # C: pasa 2 s por la fila (no es esperar)
    (4, 120, 190, 150, 800),  # D: espera 70 s con el empleado en el mostrador
    (9, 20, 200, 800, 800),  # el empleado, en el mostrador desde los 20 s
    (7, 60, 80, 500, 200),   # un cliente en el pasillo (no espera)
]


def run(zones=ZONES, people=PEOPLE, until=200):
    m = MetricsCollector(zones)
    lives = {}
    for i in range(until * 2):
        ts = i / 2
        tracks = [person(tid, x, y, a, ts) for tid, a, b, x, y in people if a <= ts < b]
        m.update(tracks, ts, (W, H), WALL0 + timedelta(seconds=ts))
        lives[ts] = m.live_counts()
    total = empty_bucket()
    for b in m.close():
        total = merge_bucket(total, b["data"])
    return total, lives


def test_espera_en_la_fila():
    total, _ = run()
    st = wait_stats(total["waits"]["Fila caja"])
    assert st["n"] == 4  # A, E, B y D (C pasó de largo)
    assert st["avg_s"] == pytest.approx((89.5 + 14.5 + 29.5 + 69.5) / 4, abs=0.6)
    assert st["max_s"] == pytest.approx(89.5, abs=0.6)
    assert st["over"] == {60: 2, 180: 0, 300: 0, 600: 0}  # A y D esperaron más de 1 minuto
    # en la fila cuenta quien ya lleva 5 s: a los 10-15 s A y E (B recién llega), después A y B; C pasó de largo
    assert st["queue_max"] == 2
    assert st["queue_avg"] == pytest.approx((84.5 + 9.5 + 24.5 + 64.5) / 199.5, abs=0.02)
    # sin nadie en el mostrador hasta los 20 s, con clientes esperando desde los 5 s: 15 s; E se fue sin que llegara
    # nadie
    assert st["staffed"] is True
    assert st["unattended_s"] == pytest.approx(15.0, abs=0.6)
    assert st["left_alone"] == 1
    # personas: A, E, B, D y el del pasillo (C estuvo 1,5 s: ruido); el empleado va aparte
    assert total["people"] == 5 and total["staff"] == 1


def test_empleados_en_el_puesto():
    total, _ = run()
    post = post_stats(total["posts"]["Mostrador"])
    assert post["obs_s"] == pytest.approx(199.5, abs=0.6)
    assert post["covered_s"] == pytest.approx(179.5, abs=0.6)
    assert post["covered_pct"] == pytest.approx(90.0, abs=0.5)
    assert post["uncovered_s"] == pytest.approx(20.0, abs=0.6)
    assert post["max"] == 1 and post["avg_people"] == pytest.approx(0.9, abs=0.01)
    assert total["staff_max"] == 1
    assert total["staff_s"] == pytest.approx(179.5, abs=0.6)
    assert total["staff_obs_s"] == pytest.approx(199.5, abs=0.6)  # promedio a la vista = 0,9
    assert total["staff_dwell"]["Mostrador"] == pytest.approx(179.5, abs=0.6)  # dónde pasa el tiempo


def test_en_vivo_se_ve_la_espera_y_el_puesto():
    _, lives = run()
    w = lives[12.0]["zones"]["Fila caja"]["wait"]
    assert w["now"] == 2 and w["n"] == 2  # en la fila A, E y B; esperando (5 s o más) A y E
    assert w["max_s"] == pytest.approx(12.0, abs=0.6)
    w = lives[16.0]["zones"]["Fila caja"]["wait"]
    assert w["now"] == 2 and w["n"] == 3  # E ya se fue (su espera quedó); B ya lleva 6 s
    w = lives[100.0]["zones"]["Fila caja"]["wait"]
    assert w["n"] == 3 and w["now"] == 0 and w["max_s"] == pytest.approx(89.5, abs=0.6)
    assert lives[10.0]["zones"]["Mostrador"]["covered_pct"] == 0.0
    assert lives[199.5]["zones"]["Mostrador"]["covered_pct"] == pytest.approx(90.0, abs=0.5)
    assert "wait" not in lives[50.0]["zones"]["Pasillo"]


def test_sin_zona_de_personal_no_hay_atencion():
    zones = [z for z in ZONES if z.type != "staff"]
    people = [p for p in PEOPLE if p[0] != 9]
    total, _ = run(zones, people)
    st = wait_stats(total["waits"]["Fila caja"])
    assert st["n"] == 4 and st["staffed"] is False
    assert st["unattended_s"] is None and st["left_alone"] is None
    assert total["posts"] == {} and total["staff_s"] == 0


def test_se_suma_entre_horas_y_camaras():
    total, _ = run()
    double = merge_bucket(total, total)
    st = wait_stats(double["waits"]["Fila caja"])
    one = wait_stats(total["waits"]["Fila caja"])
    assert st["n"] == 8 and st["avg_s"] == pytest.approx(one["avg_s"])
    assert st["max_s"] == one["max_s"] and st["queue_max"] == one["queue_max"]
    assert st["queue_avg"] == pytest.approx(one["queue_avg"])
    assert post_stats(double["posts"]["Mostrador"])["covered_pct"] == pytest.approx(90.0, abs=0.5)
    assert double["staff_max"] == 1 and double["staff_dwell"]["Mostrador"] == pytest.approx(359.0, abs=1.2)
    # un cubo viejo (agente sin actualizar) no tiene estas claves: suma igual
    old = {k: v for k, v in total.items() if k not in ("waits", "posts", "staff_s", "staff_max", "staff_dwell")}
    mixed = merge_bucket(old, total)
    assert wait_stats(mixed["waits"]["Fila caja"])["n"] == 4


def test_cada_camara_suma_lo_que_mide():
    total, _ = run()
    # el cliente tiene empleados y la cámara no lo mide: lo del personal no se suma
    out = restrict_bucket(total, ["comercio", "empleados"], ["comercio"])
    assert out["posts"] == {} and out["staff_s"] == 0 and out["staff_max"] == 0 and out["staff_dwell"] == {}
    assert out["waits"]  # la espera es de «comercio»
    # la cámara mide empleados y no comercio: la espera (de los clientes) no se suma
    out = restrict_bucket(total, ["comercio", "empleados"], ["empleados"])
    assert out["waits"] == {} and out["posts"]
    # el cliente no tiene «empleados»: no se toca
    assert restrict_bucket(total, ["comercio"], ["comercio"])["posts"] == total["posts"]


def test_zona_de_espera_solo_en_zonas_de_clientes():
    for ztype in ("staff", "pump"):
        with pytest.raises(ValueError, match="espera"):
            parse_zones([{"name": "X", "type": ztype, "wait": True, "polygon": [[0, 0], [1, 0], [1, 1]]}])
    z = parse_zones([{"name": "X", "type": "shelf", "wait": True, "polygon": [[0, 0], [1, 0], [1, 1]]}])[0]
    assert z.wait is True
    assert parse_zones([{"name": "X", "type": "shelf", "polygon": [[0, 0], [1, 0], [1, 1]]}])[0].wait is False


def test_pasar_de_largo_no_es_esperar():
    assert WAIT_MIN_SECONDS == 5.0
    zones = [z for z in ZONES if z.name == "Fila caja"]
    total, _ = run(zones, [(1, 0, 4, 100, 800), (2, 10, 16, 100, 800)])
    assert wait_stats(total["waits"]["Fila caja"])["n"] == 1  # 4 s no; 6 s sí


# ---------------- revisión adversarial (3 nov) ----------------

def collect(zones, frames, flush_every=None):
    """frames: [(ts, [tracks])] → cubos como los guarda el servidor (cada envío reemplaza la hora entera)."""
    m = MetricsCollector(zones)
    server = {}
    for i, (ts, tracks) in enumerate(frames):
        m.update(tracks, ts, (W, H), WALL0 + timedelta(seconds=ts))
        if flush_every and i % flush_every == 0:
            for b in m.flush():
                server[b["bucket_start"]] = b["data"]
    for b in m.close():
        server[b["bucket_start"]] = b["data"]
    return server


def test_la_escena_quieta_cuenta_todo_el_tiempo_mirado():
    # 20 min con el empleado (2 cuadros/s) y 40 min vacío: el filtro de movimiento analiza un cuadro cada 2 s
    frames = [(i / 2, [person(9, 800, 800, 0, i / 2)]) for i in range(1200 * 2)]
    frames += [(1200 + 2 * i, []) for i in range(1, 1201)]
    total = empty_bucket()
    for data in collect(ZONES, frames).values():
        total = merge_bucket(total, data)
    post = post_stats(total["posts"]["Mostrador"])
    assert post["obs_s"] == pytest.approx(3600, abs=2)
    assert post["covered_pct"] == pytest.approx(100 / 3, abs=0.5)
    assert post["uncovered_s"] == pytest.approx(2400, abs=2)


def test_quien_pasa_por_la_fila_no_es_fila():
    # cada 30 s alguien cruza la fila en 3 s: nadie espera, no hay fila ni «esperando sin nadie»
    frames = []
    for i in range(3600 * 2):
        ts = i / 2
        k = int(ts // 30)
        tracks = [person(100 + k, 100, 800, k * 30, ts)] if ts - k * 30 < 3 else []
        frames.append((ts, tracks))
    total = empty_bucket()
    for data in collect(ZONES, frames).values():
        total = merge_bucket(total, data)
    st = wait_stats(total["waits"]["Fila caja"])
    assert st["n"] == 0 and st["queue_max"] == 0 and st["busy_s"] == 0 and st["unattended_s"] == 0


def test_un_turno_largo_no_borra_la_hora_en_que_empezo():
    # la cajera de 10:00 a 13:10 en el mostrador; a las 10:10 un cliente espera 2 min; se manda cada 5 min
    frames = []
    for i in range(int(3.2 * 3600)):
        ts = float(i)
        tracks = [person(9, 800, 800, 0, ts)]
        if 600 <= ts < 720:
            tracks.append(person(1, 100, 800, 600, ts))
        frames.append((ts, tracks))
    server = collect(ZONES, frames, flush_every=300)
    first = server[WALL0.isoformat()]
    assert first["frames"] > 3000 and wait_stats(first["waits"]["Fila caja"])["n"] == 1
    assert post_stats(first["posts"]["Mostrador"])["covered_pct"] == pytest.approx(100, abs=0.5)
    # lo que pasó en el puesto quedó repartido en sus horas (no todo a las 10)
    assert first["staff_dwell"]["Mostrador"] == pytest.approx(3600, abs=5)
    total = empty_bucket()
    for data in server.values():
        total = merge_bucket(total, data)
    assert total["staff"] == 1
    assert total["staff_dwell"]["Mostrador"] == pytest.approx(3.2 * 3600, abs=10)


def test_la_cajera_tapada_un_momento_sigue_en_el_puesto():
    # el detector la pierde 1 de cada 2 s (el cliente adelante): el tracker la sigue informando sin ver
    def cajera(ts):
        t = person(9, 800, 800, 0, ts)
        if int(ts) % 2:
            t.missed, t.last_ts = 1, float(int(ts))
        return t
    frames = [(i / 2, [cajera(i / 2), person(1, 100, 800, 0, i / 2)]) for i in range(600 * 2)]
    total = empty_bucket()
    for data in collect(ZONES, frames).values():
        total = merge_bucket(total, data)
    assert post_stats(total["posts"]["Mostrador"])["covered_pct"] == pytest.approx(100, abs=0.5)
    assert wait_stats(total["waits"]["Fila caja"])["unattended_s"] == pytest.approx(0, abs=0.6)


def test_el_empleado_del_deposito_no_atiende_la_caja():
    zones = parse_zones([
        {"name": "Fila", "type": "checkout", "wait": True, "polygon": [[0, 0.5], [0.3, 0.5], [0.3, 1], [0, 1]]},
        {"name": "Mostrador", "type": "staff", "polygon": [[0.35, 0.5], [0.6, 0.5], [0.6, 1], [0.35, 1]]},
        {"name": "Depósito", "type": "staff", "polygon": [[0.7, 0], [1, 0], [1, 0.4], [0.7, 0.4]]},
    ])
    frames = [(i / 2, [person(9, 850, 300, 0, i / 2)] + ([person(1, 100, 800, 0, i / 2)] if i < 180 else []))
              for i in range(300 * 2)]
    total = empty_bucket()
    for data in collect(zones, frames).values():
        total = merge_bucket(total, data)
    st = wait_stats(total["waits"]["Fila"])
    assert st["n"] == 1 and st["left_alone"] == 1 and st["unattended_s"] == pytest.approx(85, abs=1)


def test_la_camara_que_solo_mide_empleados_cuenta_los_empleados():
    total, _ = run()
    out = restrict_bucket(total, ["comercio", "empleados"], ["empleados"])
    assert out["staff"] == 1 and out["people"] == 0 and out["waits"] == {}
    out = restrict_bucket(total, ["comercio", "empleados"], ["comercio"])
    assert out["staff"] == 1 and out["posts"] == {} and out["staff_obs_s"] == 0
    out = restrict_bucket(total, ["comercio", "empleados", "hurto"], ["hurto"])
    assert out["staff"] == 0 and out["people"] == 0


def test_cambiar_las_zonas_no_pisa_la_hora_en_curso():
    # 3 clientes en la caja a las 10:05; se guardan zonas (se tilda «medir espera») y vienen 2 más: la hora tiene 5
    m = MetricsCollector(ZONES)
    server = {}
    ts = 0.0
    for tid in (1, 2, 3):
        for _ in range(20):
            ts += 0.5
            m.update([person(tid, 100, 800, ts - 0.5 * 19, ts)], ts, (W, H), WALL0 + timedelta(seconds=300 + ts))
        m.update([], ts + 0.5, (W, H), WALL0 + timedelta(seconds=300 + ts + 0.5))
        ts += 0.5
    for b in m.flush():
        server[b["bucket_start"]] = b["data"]
    old = m
    m = old.with_zones(ZONES)
    for b in old.pending:
        server[b["bucket_start"]] = b["data"]
    for tid in (4, 5):
        for _ in range(20):
            ts += 0.5
            m.update([person(tid, 100, 800, ts - 0.5 * 19, ts)], ts, (W, H), WALL0 + timedelta(seconds=300 + ts))
        m.update([], ts + 0.5, (W, H), WALL0 + timedelta(seconds=300 + ts + 0.5))
        ts += 0.5
    for b in m.close():
        server[b["bucket_start"]] = b["data"]
    assert server[WALL0.isoformat()]["people"] == 5


def test_un_puesto_no_cuenta_con_el_local_cerrado():
    sched = {d: "10:00-10:02" for d in ("lun", "mar", "mie", "jue", "vie", "sab", "dom")}
    from core.schedule import parse_hours
    m = MetricsCollector(ZONES, parse_hours(sched))
    for i in range(600):  # 10:00 a 10:05, nadie en el mostrador
        m.update([], i / 2, (W, H), WALL0 + timedelta(seconds=i / 2))
    total = empty_bucket()
    for b in m.close():
        total = merge_bucket(total, b["data"])
    assert post_stats(total["posts"]["Mostrador"])["obs_s"] == pytest.approx(120, abs=1)  # solo lo abierto
