"""Tiempo de espera por zona y «Seguimiento de empleados» en la app de escritorio (pedido del dueño, 3 nov: «medir el
tiempo de espera promedio de las personas en una cierta zona, por ejemplo mostradores… y además seguir empleados como
otra función de métricas»): los textos de «Métricas», lo que se ve en vivo («Personas por zona» y la imagen), el plan
nuevo en la app y la casilla «Medir el tiempo de espera acá» del editor de zonas (esta última con Tk, si hay pantalla)."""

import os
import queue
import types
from datetime import datetime, timedelta

import numpy as np
import pytest

from agent import oficina as of
from agent.ajuste import count_text, draw_counts, wait_clock
from core.detectors import Detection
from core.metrics import MetricsCollector, empty_bucket, merge_bucket, post_stats, wait_stats
from core.rules import Zone, parse_zones
from core.tracker import Track

W, H = 1000, 1000
ZONES = parse_zones([
    {"name": "Caja", "type": "checkout", "wait": True, "polygon": [[0, 0.4], [0.5, 0.4], [0.5, 0.8], [0, 0.8]]},
    {"name": "Mostrador", "type": "staff", "polygon": [[0.6, 0.4], [1, 0.4], [1, 0.8], [0.6, 0.8]]},
])
WAIT = {"n": 3, "sum_s": 240.0, "max_s": 650.0, "hist": [1, 1, 0, 0, 1], "obs_s": 3600.0, "queue_s": 2880.0,
        "queue_max": 4, "busy_s": 1800.0, "unattended_s": 300.0, "left_alone": 1, "staffed": 1}
POST = {"obs_s": 3600.0, "covered_s": 3240.0, "people_s": 4320.0, "max": 2}


def person(tid, x, y, first, ts):
    return Track(tid, (x - 20, y - 80, x + 20, y), "persona", first, ts, hits=10)


def run_scene() -> tuple[MetricsCollector, list[dict]]:
    """2 minutos a 2 cuadros por segundo: un cliente espera 70 s en la caja (los primeros 30 sin nadie atrás del
    mostrador), la empleada está de 30 a 75 s y otro cliente espera 8 s cuando ya no hay nadie."""
    m = MetricsCollector(ZONES)
    wall0 = datetime(2026, 11, 3, 10, 0, 0)
    seen = []
    for i in range(240):
        ts = i / 2
        tracks = []
        if ts < 70:
            tracks.append(person(1, 200, 600, 0.0, ts))
        if 30 <= ts < 75:
            tracks.append(person(2, 800, 600, 30.0, ts))
        if 80 <= ts < 88:
            tracks.append(person(3, 300, 600, 80.0, ts))
        m.update(tracks, ts, (W, H), wall0 + timedelta(seconds=ts))
        seen.append(m.live_counts())
    return m, seen


# ------------------------------------------------------------------ formatos
def test_tiempos_como_reloj_y_decimales_con_coma():
    assert [of.fmt_clock(s) for s in (0, 45, 80, 725.4, 3600, 4320, 7170)] == [
        "0:00", "0:45", "1:20", "12:05", "1 h 0 min", "1 h 12 min", "2 h 0 min"]
    assert of.fmt_clock(None) == "—" and of.fmt_dec(1.25) == "1,2" and of.fmt_dec(0.8) == "0,8"
    assert of.fmt_dec(None) == "—" and of.fmt_dec(92.0, 0) == "92"
    assert [of.fmt_minutes(s) for s in (0, None, 20, 300, 754, 4320)] == [
        "0 min", "0 min", "menos de 1 min", "5 min", "13 min", "1 h 12 min"]
    assert of.fmt_dur(7170) == "2 h 0 min"  # (antes salía «1 h 60 min»)
    assert wait_clock(80) == "1:20" and wait_clock(4320) == "1 h 12 min" and wait_clock(None) == "-"


# ------------------------------------------------------------------ plan «Seguimiento de empleados»
def test_el_plan_empleados_esta_en_la_app():
    assert list(of.PLAN_NAMES) == ["hurto", "comercio", "estacion", "empleados"]
    assert of.PLAN_NAMES["empleados"] == "Seguimiento de empleados" and of.PLAN_SHORT["empleados"] == "Empleados"
    client = {"id": 1, "plans": ["empleados", "comercio", "hurto"]}
    assert of.client_plans(client) == ["hurto", "comercio", "empleados"]
    cam = {"plans": ["comercio", "empleados"], "zones": []}
    assert of.camera_measures(cam, client) == ["comercio", "empleados"]
    assert of.measures_text(cam, client) == "Comercio · Empleados"
    assert of.measures_text(cam, client, short=False) == "Métricas de comercio, Seguimiento de empleados"
    assert of.plans_names(["hurto", "empleados"]) == "Detección de hurtos y Seguimiento de empleados"
    # sin zona «Personal» no se miden los puestos: la app lo avisa
    assert of.staff_zone_missing(cam, client)
    with_post = {**cam, "zones": [{"name": "Mostrador", "type": "staff", "polygon": [[0, 0], [1, 0], [1, 1]]}]}
    assert not of.staff_zone_missing(with_post, client)
    assert not of.staff_zone_missing({"plans": ["comercio"], "zones": []}, client)  # no mide empleados
    assert of.plans_to_save(["hurto", "comercio", "empleados"], of.client_plans(client)) is None  # todo = el cliente


# ------------------------------------------------------------------ textos de «Métricas»
def test_texto_de_la_espera_de_una_zona():
    st = wait_stats(WAIT)
    assert of.wait_lines("Caja", st) == [
        ("Caja: 3 esperas · promedio 1:20 · máximo 10:50 · más de 5 min: 1 · más de 10 min: 1 · fila promedio 0,8, "
         "máxima 4"),
        "con clientes esperando y nadie en su puesto (la zona de personal más cercana): 5 min · se fueron sin que "
        "hubiera nadie: 1"]
    # sin zona de personal en la cámara no se dice nada de «nadie en el puesto»
    alone = wait_stats({**WAIT, "staffed": 0})
    assert len(of.wait_lines("Caja", alone)) == 1
    # mirada pero nadie esperó 5 s o más
    empty = of.wait_lines("Fila", wait_stats({"obs_s": 600.0}))
    assert empty == ["Fila: nadie esperó todavía (5 s o más en la zona) · fila promedio 0,0, máxima 0"]
    # todas las zonas, el segundo renglón sangrado
    sec = of.wait_section({"waits": {"Caja": WAIT, "Fila": {"obs_s": 600.0}}})
    assert sec[0].startswith("Caja: 3 esperas") and sec[1].startswith("  con clientes") and sec[2].startswith("Fila:")
    assert of.has_waits({"waits": {"Fila": {"obs_s": 600.0}}}) and not of.has_waits(empty_bucket())


def test_texto_de_los_empleados():
    assert of.post_text("Mostrador", post_stats(POST)) == (
        "Mostrador: cubierto 90 % · sin nadie 6:00 · en el puesto: promedio 1,2, máximo 2")
    b = {"posts": {"Mostrador": POST}, "staff_s": 5400.0, "staff_obs_s": 3600.0, "staff_max": 3,
         "staff_dwell": {"Mostrador": 4000.0, "Depósito": 900.0, "Pasillo": 0.0}}
    assert of.staff_lines(b) == [
        "Mostrador: cubierto 90 % · sin nadie 6:00 · en el puesto: promedio 1,2, máximo 2",
        "Empleados a la vista (por cámara): promedio 1,5, máximo 3 a la vez",  # 5400 personas-segundo en 3600 s
        "Dónde pasan el tiempo: Mostrador 1 h 7 min · Depósito 15:00"]
    # sin puestos ni tiempo mirado (máquina sin actualizar): no se inventa un «máximo 0», se dice qué falta
    assert of.staff_lines({"staff_s": 0.0, "staff_max": 0}) == [of.NO_POSTS_TEXT]
    assert of.staff_lines({}, hint=False) == []
    # sin puestos pero mirado (empleados que llegaron con el local cerrado)
    lines = of.staff_lines({"staff_s": 100.0, "staff_obs_s": 1000.0, "staff_max": 1})
    assert lines == ["Empleados a la vista (por cámara): promedio 0,1, máximo 1 a la vez", of.NO_POSTS_TEXT]
    # de las 6 zonas donde más tiempo pasan, de mayor a menor
    many = {f"Z{i}": float(i * 60) for i in range(1, 9)}
    dwell = of.staff_lines({"posts": {"Mostrador": POST}, "staff_dwell": many})[-1]
    assert dwell == "Dónde pasan el tiempo: Z8 8:00 · Z7 7:00 · Z6 6:00 · Z5 5:00 · Z4 4:00 · Z3 3:00"


def _resp(total, cams=None, days=None):
    return {"total": total, "by_day": days or [], "by_hour": [], "by_camera": cams or []}


def test_metricas_del_cliente_con_espera_y_empleados():
    m, _ = run_scene()
    data = empty_bucket()
    for b in m.close():
        data = merge_bucket(data, b["data"])
    st = wait_stats(data["waits"]["Caja"])
    assert st["n"] == 2 and st["left_alone"] == 1 and st["staffed"]  # (lo del núcleo, para leer bien el texto)
    client = {"id": 1, "name": "Kiosco", "plans": ["hurto", "comercio", "empleados"]}
    cams = [{"camera_id": 1, "name": "Caja", "measures": ["hurto", "comercio", "empleados"], **data}]
    text = of.metrics_text(client, _resp(data, cams), _resp(data, days=[{"date": "2026-11-03", **data}]))
    assert "TIEMPO DE ESPERA (cada cliente que estuvo 5 s o más en la zona marcada)" in text
    assert "Caja: 2 esperas · promedio " in text and "se fueron sin que hubiera nadie: 1" in text
    assert "EMPLEADOS" in text and "Mostrador: cubierto " in text and "Empleados a la vista (por cámara): promedio " in text
    assert text.count("  Hoy:") == 2 and text.count("  Últimos 7 días:") == 2  # en las dos secciones
    assert of.LIMITS_TEXT in text and of.NO_POSTS_TEXT not in text
    # por cámara: lo que suma cada una
    assert "Caja (mide Hurto · Comercio · Empleados): " in text and "espera en Caja " in text
    assert "Mostrador cubierto " in text

    # sin esos planes, nada de eso
    plain = of.metrics_text({"id": 2, "name": "Minas", "plans": ["hurto"]}, _resp(data), _resp(data))
    assert "TIEMPO DE ESPERA" not in plain and "EMPLEADOS" not in plain and of.LIMITS_TEXT not in plain
    # comercio sin zonas de espera: no hay sección de espera
    no_wait = {**data, "waits": {}}
    shop = of.metrics_text({"id": 3, "name": "Tienda", "plans": ["comercio"]}, _resp(no_wait), _resp(no_wait))
    assert "TIEMPO DE ESPERA" not in shop and "EMPLEADOS" not in shop
    # empleados sin zona de personal: cómo marcarla, una sola vez (no en hoy y en la semana)
    bare = {**empty_bucket(), "frames": 100}
    staff = of.metrics_text({"id": 4, "name": "Bar", "plans": ["empleados"]}, _resp(bare), _resp(bare))
    assert staff.count(of.NO_POSTS_TEXT) == 1 and "EMPLEADOS" in staff


# ------------------------------------------------------------------ en vivo: la tabla y la imagen
COUNTS = {"zones": {"Caja": {"type": "checkout", "visits": 6, "now": 2,
                             "wait": {"n": 5, "avg_s": 80.0, "max_s": 190.0, "now": 2}},
                    "Mostrador": {"type": "staff", "visits": 0, "now": 1, "staff": 2, "covered_pct": 90.0}}}


def test_en_vivo_se_ve_la_espera_y_el_puesto():
    caja, mostrador = ZONES
    assert count_text(caja, COUNTS) == "entraron 6  ahora 2\nesperando 2  prom 1:20  max 3:10"
    assert count_text(mostrador, COUNTS) == "personal 2  cubierto 90%"
    assert of.zone_extra_text(caja, COUNTS["zones"]["Caja"]) == "esperando 2 · prom. 1:20"
    assert of.zone_extra_text(mostrador, COUNTS["zones"]["Mostrador"]) == "cubierto 90 %"
    # sin esperas todavía: solo cuántos esperan
    first = {"zones": {"Caja": {"visits": 0, "now": 1, "wait": {"n": 0, "avg_s": None, "max_s": None, "now": 1}}}}
    assert count_text(caja, first) == "entraron 0  ahora 1\nesperando 1"
    assert of.zone_extra_text(caja, first["zones"]["Caja"]) == "esperando 1"
    # una zona que no mide la espera no la muestra (aunque el conteo la traiga)
    plain = Zone("Caja", "checkout", caja.polygon)
    assert count_text(plain, COUNTS) == "entraron 6  ahora 2" and of.zone_extra_text(plain, COUNTS["zones"]["Caja"]) == ""
    assert of.zone_extra_text(caja, None) == "" and count_text(caja, None) is None
    # en la imagen, la espera en un renglón aparte, debajo
    one = np.zeros((360, 640, 3), np.uint8)
    two = one.copy()
    draw_counts(one, [plain], COUNTS)
    draw_counts(two, [caja], COUNTS)
    rows_one, rows_two = np.where(one.any(axis=2))[0], np.where(two.any(axis=2))[0]
    assert rows_two.max() > rows_one.max() + 10


def test_los_textos_salen_de_lo_que_cuenta_el_nucleo():
    _, seen = run_scene()
    caja, mostrador = ZONES
    at = {round(i / 2, 1): c for i, c in enumerate(seen)}
    # a los 20 s: el cliente lleva 20 s esperando y todavía no llegó la empleada
    mid = at[20.0]
    assert count_text(caja, mid) == "entraron 1  ahora 1\nesperando 1  prom 0:20  max 0:20"
    assert of.zone_extra_text(mostrador, mid["zones"]["Mostrador"]) == "cubierto 0 %"
    # al final: dos esperas (70 y 8 s) y el puesto cubierto una parte del tiempo
    end = seen[-1]
    wz = end["zones"]["Caja"]["wait"]
    assert wz["n"] == 2 and wz["now"] == 0
    assert count_text(caja, end).startswith("entraron 2\nesperando 0  prom ")
    pct = end["zones"]["Mostrador"]["covered_pct"]
    assert 30 < pct < 45 and count_text(mostrador, end) == f"personal 1  cubierto {pct:.0f}%"


def test_tildar_la_espera_en_vivo_recuenta_con_ella():
    """En «● En vivo», guardar las mismas zonas solo con la espera tildada tiene que recontar (si no, la zona no la
    mediría hasta reabrir la ventana)."""
    from agent.ajuste import Tuner

    class Det:
        def detect(self, frame):
            return [Detection((280, 260, 360, 340), "persona", 0.9)]  # pies en (320, 340): en la caja

    zone = {"name": "Caja", "type": "checkout", "polygon": [[0.25, 0.5], [0.75, 0.5], [0.75, 1], [0.25, 1]]}
    t = Tuner({"id": 7, "name": "Caja", "process_fps": 5, "connection": {},
               "analysis": {"zones": [zone], "rules": {}}}, Det())
    assert t.live
    frame = np.full((360, 640, 3), 60, np.uint8)
    wall = datetime(2026, 11, 3, 10, 0, 0)
    for i in range(60):  # 12 s parado en la caja
        t._analyze(frame, i / 5, wall + timedelta(seconds=i / 5), 0)
    assert "wait" not in t.counts_at()["zones"]["Caja"]
    t.set_analysis({"zones": [{**zone, "wait": True}], "rules": {}})
    w = t.counts_at()["zones"]["Caja"]["wait"]
    assert w["n"] == 1 and w["now"] == 1 and w["avg_s"] > 10
    # probar otros tiempos no toca el conteo (sigue el mismo, no se rehace)
    before = t.counts_at()
    t.set_times({"checkout_min_seconds": 2})
    assert t.counts_at() is before


# ------------------------------------------------------------------ editor de zonas (con Tk)
def _tk():
    tk = pytest.importorskip("tkinter")
    if os.name != "nt" and not os.environ.get("DISPLAY"):
        pytest.skip("sin pantalla (correr con xvfb-run)")
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"sin Tk: {exc}")
    return tk, root


def test_el_editor_guarda_y_saca_la_espera():
    tk, root = _tk()
    from tkinter import ttk

    from agent.oficina import ZONE_NAMES, ZonesEditor
    try:
        sent, msgs = [], []
        bg = types.SimpleNamespace(run=lambda fn, done=None, fail=None: done(fn()) if done else fn())
        app = types.SimpleNamespace(
            tk=tk, ttk=ttk, v=types.SimpleNamespace(root=root, log_handler=types.SimpleNamespace(q=queue.Queue())),
            mb=types.SimpleNamespace(showinfo=lambda *a, **k: msgs.append(a), showwarning=lambda *a, **k: msgs.append(a),
                                     showerror=lambda *a, **k: msgs.append(a), askyesno=lambda *a, **k: True),
            bg=bg, api=types.SimpleNamespace(put=lambda path, body: sent.append((path, body)) or {}),
            config_saved=lambda: None, zones_changed=lambda *a, **k: None, _open_window=lambda cid: None,
            agent_of=None)
        app._dialog = types.MethodType(of.Oficina._dialog, app)
        poly = [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]]
        cam = {"id": 5, "name": "Kiosco", "has_preview": False,
               "zones": [{"name": "Fila", "type": "checkout", "polygon": poly},
                         {"name": "Mostrador", "type": "staff", "polygon": [[0.6, 0.1], [0.9, 0.1], [0.9, 0.5]]}]}
        ed = ZonesEditor(app, cam)
        root.update()
        ed.top.update_idletasks()
        assert ed.top.winfo_reqwidth() <= 1366 and ed.top.winfo_reqheight() <= 768 - 70  # entra en una notebook
        # como se abre desde la ventana de la cámara (con el mapa de pisadas) y con una zona de espera elegida:
        # también entra (antes medía 755 de alto y «Guardar zonas» quedaba debajo de la barra de tareas)
        import numpy as np
        from core.pisadas import FootMap
        big = ZonesEditor(app, cam, image=np.zeros((360, 640, 3), np.uint8), footmap=FootMap.empty())
        big.select(0)
        root.update()
        big.top.update_idletasks()
        assert big.top.winfo_reqwidth() <= 1366 and big.top.winfo_reqheight() <= 768 - 70, big.top.winfo_reqheight()
        big.top.destroy()
        # tildar la espera en la «Fila» y aplicar
        ed.select(0)
        assert not ed.zwait.get() and str(ed.waitcb.cget("state")) == "normal"
        ed.zwait.set(True)
        ed._show_help()
        assert of.WAIT_HELP in str(ed.help.cget("text"))
        ed.apply_selected()
        assert ed.zones[0]["wait"] is True and "· espera" in ed.zlist.get(0)
        # en «Personal» no se puede: la casilla se apaga y se saca
        ed.select(1)
        assert not ed.zwait.get() and str(ed.waitcb.cget("state")) == "disabled"
        ed.select(0)
        assert ed.zwait.get()
        ed.ztype.set(ZONE_NAMES["pump"])
        ed._show_help()
        assert not ed.zwait.get() and str(ed.waitcb.cget("state")) == "disabled"
        ed.ztype.set(ZONE_NAMES["checkout"])
        ed._show_help()
        ed.zwait.set(True)
        ed.apply_selected()
        # una zona nueva con la espera tildada
        ed.new_zone()
        ed.zname.set("Mostrador clientes")
        ed.ztype.set(ZONE_NAMES["shop"])
        ed.zwait.set(True)
        ed.draft = [(0.1, 0.6), (0.4, 0.6), (0.4, 0.9)]
        ed.close_zone()
        assert ed.zones[-1]["wait"] is True and ed.zones[-1]["type"] == "shop"
        # arrastrar un punto no cambia las zonas de la cámara antes de guardar (se trabaja sobre una copia)
        ed.zones[0]["polygon"][0] = [0.2, 0.2]
        assert cam["zones"][0]["polygon"][0] == [0.1, 0.1]
        # sacarle la espera a la «Fila»: queda sin la clave
        ed.select(0)
        ed.zwait.set(False)
        ed.apply_selected()
        assert "wait" not in ed.zones[0]
        ed.save()
        path, body = sent[-1]
        assert path == "/api/cameras/5/analysis"
        by_name = {z["name"]: z for z in body["zones"]}
        assert "wait" not in by_name["Fila"] and "wait" not in by_name["Mostrador"]
        assert by_name["Mostrador clientes"]["wait"] is True
        parse_zones(body["zones"])  # lo que se manda lo acepta el núcleo (y el servidor)
    finally:
        root.destroy()


def test_aviso_de_zonas_de_espera_que_se_pisan():
    a = {"name": "Fila", "wait": True, "polygon": [[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5]]}
    b = {"name": "Caja", "wait": True, "polygon": [[0.4, 0.4], [0.8, 0.4], [0.8, 0.8], [0.4, 0.8]]}
    cross = {"name": "Cruz", "wait": True, "polygon": [[0.2, -0.1], [0.3, -0.1], [0.3, 0.9], [0.2, 0.9]]}
    far = {"name": "Lejos", "wait": True, "polygon": [[0.9, 0.9], [1, 0.9], [1, 1]]}
    assert of.overlapping_waits([a, b]) == ("Fila", "Caja")
    assert of.overlapping_waits([a, cross]) == ("Fila", "Cruz")  # se cruzan sin vértices adentro... o con
    assert of.overlapping_waits([a, far]) is None
    assert of.overlapping_waits([a, {**b, "wait": False}]) is None  # solo las que miden la espera


def test_si_ninguna_camara_mide_empleados_lo_dice():
    client = {"id": 1, "name": "Súper", "plans": ["comercio", "empleados"]}
    b = {**empty_bucket(), "frames": 100}
    cams = [{"camera_id": 1, "name": "Caja", "measures": ["comercio"], **b}]
    text = of.metrics_text(client, _resp(b, cams), _resp(b, cams))
    assert of.NO_STAFF_CAMERA_TEXT in text and of.NO_POSTS_TEXT not in text
