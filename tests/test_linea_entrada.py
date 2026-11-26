"""La entrada marcada como una LÍNEA con una flecha (26 nov, el dueño: «cuando marquemos las zonas de entrada, que sea
una línea, con una flecha que apunte a donde es la entrada, no como ahora que es una zona»). Entra quien la cruza hacia
la flecha y sale quien la cruza al revés. Las entradas marcadas como zona (las de antes) siguen contando como contaban."""

import os
import queue
import types
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from core.geometry import line_arrow, line_side, oriented_line
from core.metrics import MetricsCollector, door_counts
from core.rules import parse_config, parse_zones
from core.tracker import Track
from server.main import Settings, create_app

ADMIN = ("admin", "clave-admin-123")
W, H = 640, 360
LINE = {"name": "Puerta", "type": "entrance", "line": True, "polygon": [[0.2, 0.5], [0.8, 0.5]]}  # flecha hacia abajo
T0 = datetime(2026, 11, 26, 10, 0)


def frames_of(paths: dict, fps: int = 5, height: int = 120, label: str = "persona", born: bool = True) -> list:
    """Cuadros como los que da el seguimiento: `paths` = {persona: [(x, y) de los pies por cuadro, None si todavía no
    está o ya se fue, o (x, y, False) si en ese cuadro no se la ve (tapada: el seguimiento la sigue informando)]}."""
    n = max(len(p) for p in paths.values())
    first: dict = {}
    out = []
    for i in range(n + 3):  # (unos cuadros más sin nadie: las visitas se cierran)
        ts = i / fps
        tracks = []
        for tid, path in paths.items():
            if i < len(path) and path[i] is not None:
                x, y = path[i][:2]
                box = (x - 20, y - height, x + 20, y)
                first.setdefault(tid, (ts, box))
                t = Track(tid, box, label, first[tid][0], ts, hits=5, missed=1 if len(path[i]) > 2 else 0)
                t.first_bbox = first[tid][1] if born else None
                tracks.append(t)
        out.append((tracks, ts, (W, H), T0 + timedelta(seconds=ts)))
    return out


def count(zone: dict, paths: dict, **kw) -> tuple[int, int]:
    got = door_counts(frames_of(paths, **kw), parse_zones([zone]))
    assert got["door"] == 1
    return got["entries"], got["exits"]


DOWN = [(320, 100 + 12 * i) for i in range(15)]  # camina hacia abajo de la imagen: cruza a los 180 px
UP = list(reversed(DOWN))


# ------------------------------------------------------------------------------------------------ la línea
def test_la_linea_va_con_dos_puntos_y_solo_es_la_entrada():
    z = parse_zones([LINE])[0]
    assert z.line and z.type == "entrance" and z.polygon == ((0.2, 0.5), (0.8, 0.5)) and z.direction is None
    assert not z.contains(0.5, 0.5)  # no es un lugar: nadie «está» en la línea
    for bad in ({**LINE, "polygon": [[0.2, 0.5], [0.8, 0.5], [0.5, 0.9]]},  # una línea son dos puntas
                {**LINE, "type": "shelf"},  # solo la entrada
                {**LINE, "polygon": [[0.5, 0.5], [0.51, 0.5]]},  # un clic doble
                {"name": "x", "type": "entrance", "polygon": [[0.2, 0.5], [0.8, 0.5]]}):  # 2 puntos sin ser línea
        with pytest.raises(ValueError):
            parse_zones([bad])
    # lo que una app vieja le pueda haber puesto con «Aplicar a la elegida» no cuenta ni da error
    old = parse_zones([{**LINE, "direction": "arriba", "wait": True}])[0]
    assert old.line and old.direction is None and not old.wait
    # a las máquinas viaja en su propia lista (una sin actualizar pide 3 puntos por zona)
    zones, _ = parse_config({"zones": [], "rules": {}, "line_zones": [LINE]})
    assert [z.name for z in zones] == ["Puerta"] and zones[0].line


def test_la_flecha_apunta_al_lado_que_da_el_orden_de_los_puntos():
    a, b = (0.2, 0.5), (0.8, 0.5)  # de izquierda a derecha: la flecha, hacia abajo
    assert line_side(a, b, (0.5, 0.9)) > 0 > line_side(a, b, (0.5, 0.1))
    start, tip = line_arrow(a, b, W, H, 40)
    assert start == (320.0, 180.0) and tip == (320.0, 220.0)
    assert line_arrow(b, a, W, H, 40)[1] == (320.0, 140.0)  # al revés: hacia arriba
    # al marcarla queda hacia abajo (hacia la cámara) sin importar por qué punta se empezó; una vertical, a la derecha
    assert oriented_line(b, a, w=W, h=H) == (a, b) and oriented_line(a, b, w=W, h=H) == (a, b)
    up, down = (0.5, 0.2), (0.5, 0.8)
    va, vb = oriented_line(up, down, w=W, h=H)
    assert line_arrow(va, vb, W, H, 40)[1][0] > 320
    assert oriented_line(a, b, toward=(0.0, -1.0)) == (b, a)


# ------------------------------------------------------------------------------------------------ el conteo
def test_entra_quien_cruza_hacia_la_flecha_y_sale_quien_cruza_al_reves():
    assert count(LINE, {1: DOWN}) == (1, 0)
    assert count(LINE, {1: UP}) == (0, 1)
    assert count(LINE, {1: DOWN + UP}) == (1, 1)  # entró y después se fue
    assert count({**LINE, "polygon": [[0.8, 0.5], [0.2, 0.5]]}, {1: DOWN}) == (0, 1)  # con la flecha dada vuelta
    # varias personas a la vez, cada una por su lado
    assert count(LINE, {1: DOWN, 2: [(200, y) for _, y in UP], 3: [(450, y) for _, y in DOWN]}) == (2, 1)
    # una línea en diagonal: la flecha de (0.3, 0.2)→(0.7, 0.8) apunta hacia abajo a la izquierda
    diag = {**LINE, "polygon": [[0.3, 0.2], [0.7, 0.8]]}
    leftwards = [(520 - 25 * i, 180) for i in range(16)]
    assert count(diag, {1: leftwards}) == (1, 0) and count(diag, {1: list(reversed(leftwards))}) == (0, 1)


def test_lo_que_no_es_cruzar_no_cuenta():
    # parado encima de la línea: el recuadro tiembla unos píxeles para un lado y para el otro
    standing = [(320, 100 + 12 * i) for i in range(7)] + [(320, 180 + (4 if i % 2 else -4)) for i in range(30)]
    assert count(LINE, {1: standing}) == (0, 0)
    # un cuadro en que el recuadro salta (algo le tapó las piernas) y vuelve
    jump = [(320, 150)] * 5 + [(320, 230)] + [(320, 150)] * 5
    assert count(LINE, {1: jump}) == (0, 0)
    # pasa por afuera de las puntas de la línea
    assert count(LINE, {1: [(40, y) for _, y in DOWN]}) == (0, 0)
    # camina a lo largo, del mismo lado
    assert count(LINE, {1: [(150 + 20 * i, 150) for i in range(18)]}) == (0, 0)
    # el seguimiento le dio el mismo número a otra persona que aparece lejos: nadie da ese salto caminando
    assert count(LINE, {1: [(320, 300)] * 6 + [(320, 60)] * 6}) == (0, 0)
    # estuvo 4 s tapada y reaparece del otro lado: no se sabe por dónde anduvo
    assert count(LINE, {1: [(320, 230)] * 5 + [(320, 230, False)] * 20 + [(320, 130)] * 5}) == (0, 0)
    # (tapada medio segundo mientras cruza, sí cuenta: sigue siendo su camino)
    assert count(LINE, {1: [(320, 150)] * 4 + [(320, 150, False)] * 3 + [(320, 215)] * 4}) == (1, 0)
    # un vehículo no es una persona que entra
    got = door_counts(frames_of({1: DOWN}, label="vehiculo"), parse_zones([LINE]))
    assert (got["entries"], got["exits"]) == (0, 0)


def test_quien_cruza_y_se_pierde_enseguida_cuenta_y_quien_aparece_del_otro_lado_tambien():
    # sale por la puerta: cruza y en el cuadro siguiente ya no se lo ve
    out = [(320, 300 - 20 * i) for i in range(8)]  # 300 … 160: cruzó en el último cuadro
    assert out[-1][1] < 180 - 6 < out[-2][1]
    assert count(LINE, {1: out}) == (0, 1)
    # entra rápido: el seguimiento la informa recién a la tercera detección, ya del lado de adentro; se sabe de qué
    # lado apareció por su primera detección
    fast = [None, None] + [(320, 150 + 30 * i) for i in range(2, 9)]  # (informada desde y = 210, ya cruzada)
    frames = frames_of({1: fast})
    for tracks, ts, _size, _wall in frames:
        for t in tracks:
            t.first_ts, t.first_bbox = 0.0, (300, 30, 340, 150)  # se la detectó por primera vez arriba de la línea
    got = door_counts(frames, parse_zones([LINE]))
    assert (got["entries"], got["exits"]) == (1, 0)
    # sin ese dato (o si la persona lleva rato a la vista cuando se guardan las zonas) no se inventa un cruce
    assert count(LINE, {1: [(320, 210 + 10 * i) for i in range(12)]}, born=False) == (0, 0)


def test_el_personal_no_cuenta_como_cliente_y_la_entrada_de_antes_sigue_andando():
    zones = parse_zones([LINE, {"name": "mostrador", "type": "staff",
                                "polygon": [[0, 0.7], [1, 0.7], [1, 1], [0, 1]]}])
    path = DOWN + [(320, 300)] * 15  # entra y se queda detrás del mostrador: es del personal
    got = door_counts(frames_of({1: path}), zones)
    assert (got["entries"], got["staff"], got["staff_entries"]) == (0, 1, 1)
    # en vivo: la línea trae lo suyo (entraron y salieron) y nadie «está» en ella
    mc = MetricsCollector(parse_zones([LINE]))
    for tracks, ts, size, wall in frames_of({1: DOWN, 2: [(450, y) for _, y in UP]}):
        mc.update(tracks, ts, size, wall)
    live = mc.live_counts()
    assert live["door"] and (live["entries"], live["exits"]) == (1, 1)
    assert live["zones"]["Puerta"] == {"type": "entrance", "visits": 0, "now": 0, "entries": 1, "exits": 1}
    data = mc.close()[0]["data"]
    assert (data["entries"], data["exits"], data["door"]) == (1, 1, 1) and "Puerta" not in data["zones"]
    # una entrada marcada como zona con sentido (las de antes) cuenta como contaba
    old = {"name": "puerta", "type": "entrance", "direction": "abajo", "polygon": [[0, 0.3], [1, 0.3], [1, 0.6], [0, 0.6]]}
    got = door_counts(frames_of({1: [(320, 60 + 20 * i) for i in range(14)]}), parse_zones([old]))
    assert (got["entries"], got["exits"], got["door"]) == (1, 0, 1)


# ------------------------------------------------------------------------------------------------ servidor
@pytest.fixture
def app(tmp_path, db_url):
    return create_app(Settings(data_dir=tmp_path / "data", database_url=db_url, admin_password=ADMIN[1],
                               serve_public=False))


def test_el_servidor_guarda_la_linea_y_se_la_manda_aparte_a_las_maquinas(app):
    with TestClient(app) as c:
        c.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]})
        cid = c.post("/api/clients", json={"name": "Kiosco", "plans": ["comercio"]}).json()["id"]
        cam = c.post("/api/cameras", json={"client_id": cid, "name": "Puerta", "brand": "ip", "host": "dvr",
                                           "password": "x"}).json()["id"]
        shelf = {"name": "gondola", "type": "shelf", "polygon": [[0, 0.6], [0.4, 0.6], [0.4, 1], [0, 1]]}
        # (una app vieja le puede haber puesto sentido y espera con «Aplicar a la elegida»: no se guardan)
        r = c.put(f"/api/cameras/{cam}/analysis",
                  json={"zones": [shelf, {**LINE, "direction": "arriba", "wait": True}], "rules": {}})
        assert r.status_code == 200, r.text
        assert r.json()["zones"][1] == LINE and "line" not in r.json()["zones"][0]
        for bad in ({**LINE, "polygon": [[0.2, 0.5], [0.8, 0.5], [0.5, 0.9]]}, {**LINE, "type": "shelf"},
                    {**LINE, "polygon": [[0.2, 0.5]]}, {**shelf, "polygon": [[0, 0.6], [0.4, 0.6]]}):
            assert c.put(f"/api/cameras/{cam}/analysis", json={"zones": [bad], "rules": {}}).status_code == 422
        tok = c.post("/api/agent/session").json()["token"]
        c.put("/api/agent/watch", json={"camera_ids": [cam]}, headers={"X-Agent-Token": tok})
        cfg = c.post("/api/agent/tune", json={"camera_id": cam}, headers={"X-Agent-Token": tok}).json()
        an = cfg["camera"]["analysis"]
        # una máquina sin actualizar lee solo `zones` (y pide 3 puntos por zona): la línea no va ahí
        assert [z["name"] for z in an["zones"]] == ["gondola"] and an["line_zones"] == [LINE]
        assert all(len(z["polygon"]) >= 3 for z in an["zones"])
        assert cfg["camera"]["process_fps"] == 5  # contar la puerta pide 5 cuadros/s aunque sea solo métricas
        zones, _ = parse_config(an)
        assert [(z.name, z.line) for z in zones] == [("gondola", False), ("Puerta", True)]
        assert cfg["raw"]["zones"][1] == LINE  # y el editor la recibe con las demás


# ------------------------------------------------------------------------------------------------ app
def _tk():
    tk = pytest.importorskip("tkinter")
    if os.name != "nt" and not os.environ.get("DISPLAY"):
        pytest.skip("sin pantalla (correr con xvfb-run)")
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"sin Tk: {exc}")
    return tk, root


def _editor(tk, root, zones):
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    from tkinter import ttk

    import agent.oficina as of

    saved, said = {}, []

    def run(fn, done=None, fail=None):
        try:
            res = fn()
        except Exception as exc:  # noqa: BLE001
            return fail(exc) if fail else None
        return done(res) if done else None

    app = types.SimpleNamespace(
        tk=tk, ttk=ttk, v=types.SimpleNamespace(root=root, log_handler=types.SimpleNamespace(q=queue.Queue())),
        mb=types.SimpleNamespace(showinfo=lambda *a, **k: said.append(a[1]), showwarning=lambda *a, **k: None,
                                 showerror=lambda *a, **k: said.append(a[1]), askyesno=lambda *a, **k: True),
        bg=types.SimpleNamespace(run=run), api=types.SimpleNamespace(put=lambda path, body: saved.update(body) or {}),
        config_saved=lambda: None, zones_changed=lambda *a, **k: None, _open_window=lambda cid: None,
        zones_done=lambda *a, **k: None, agent_of=None, _clip_frame=lambda cam: None)
    app._dialog = types.MethodType(of.Oficina._dialog, app)
    cam = {"id": 7, "name": "Entrada", "has_preview": True, "zones": zones}
    ed = of.ZonesEditor(app, cam, image=np.zeros((360, 640, 3), np.uint8))
    return of, ed, saved, said


def _ev(ed, nx, ny):
    return types.SimpleNamespace(x=int(nx * ed.W), y=int(ny * ed.H))


def _click(ed, nx, ny):
    ed.press(_ev(ed, nx, ny))
    ed.release(_ev(ed, nx, ny))


def test_en_la_app_la_entrada_se_marca_con_dos_clics_y_la_flecha_se_da_vuelta():
    tk, root = _tk()
    try:
        old = {"name": "puerta vieja", "type": "entrance", "direction": "arriba",
               "polygon": [[0.6, 0.6], [0.9, 0.6], [0.9, 0.8], [0.6, 0.8]]}
        of, ed, saved, said = _editor(tk, root, [old])
        assert str(ed.flipbtn.cget("state")) == "disabled" and str(ed.tolinebtn.cget("state")) == "disabled"
        ed.ztype.set(of.ZONE_NAMES["entrance"])
        ed._show_help()
        assert "LÍNEA" in ed.help.cget("text")
        ed.new_zone()
        _click(ed, 0.5, 0.4)  # una punta (se empieza por la derecha)
        assert ed.drawing and len(ed.draft) == 1 and len(ed.zones) == 1
        ed.hover(_ev(ed, 0.3, 0.4))  # la línea sigue al mouse
        assert ed.canvas.find_withtag("band")
        _click(ed, 0.2, 0.4)  # la otra punta: queda la línea, sin doble clic
        z = ed.zones[-1]
        assert not ed.drawing and z["type"] == "entrance" and z["line"] is True and "direction" not in z
        (ax, _), (bx, _) = z["polygon"]
        assert ax < bx  # la flecha, hacia abajo (hacia la cámara) aunque se haya empezado por la derecha
        assert ed.sel == 1 and str(ed.flipbtn.cget("state")) == "normal"
        assert "ADENTRO" in ed.info.cget("text") and "línea" in ed.zlist.get(1)
        # darla vuelta: con el botón y con un clic en la flecha
        ed.flip_line()
        assert ed.zones[1]["polygon"][0][0] > ed.zones[1]["polygon"][1][0]
        start, tip = line_arrow(*ed.zones[1]["polygon"], ed.W, ed.H, of.LINE_ARROW)
        assert tip[1] < start[1]  # ahora apunta hacia arriba
        ed.press(types.SimpleNamespace(x=int(tip[0]), y=int(tip[1]) + 4))
        assert ed.zones[1]["polygon"][0][0] < ed.zones[1]["polygon"][1][0]
        # arrastrando de una punta a la otra también se marca
        ed.select(None)
        ed.zname.set("Fondo")
        ed.press(_ev(ed, 0.1, 0.9))
        ed.motion(_ev(ed, 0.3, 0.9))
        ed.release(_ev(ed, 0.4, 0.9))
        assert [z["name"] for z in ed.zones] == ["puerta vieja", "entrada_2", "Fondo"] and ed.zones[2]["line"]
        # corregir: una punta se arrastra; la línea entera también; no se le borran puntas
        ed.press(_ev(ed, 0.1, 0.9))
        ed.motion(_ev(ed, 0.12, 0.8))
        ed.release(_ev(ed, 0.12, 0.8))
        assert ed.zones[2]["polygon"][0] == pytest.approx([0.12, 0.8], abs=0.005)
        before = [list(p) for p in ed.zones[1]["polygon"]]
        ed.press(_ev(ed, 0.35, 0.4))
        ed.motion(_ev(ed, 0.35, 0.5))
        ed.release(_ev(ed, 0.35, 0.5))
        assert ed.sel == 1 and ed.zones[1]["polygon"][0][1] == pytest.approx(before[0][1] + 0.1, abs=0.01)
        ed.right_click(_ev(ed, *ed.zones[1]["polygon"][0]))
        assert len(ed.zones[1]["polygon"]) == 2 and "dos puntas" in said[-1]
        # una línea solo puede ser la entrada, y una zona no pasa a ser la entrada
        ed.ztype.set(of.ZONE_NAMES["shelf"])
        ed.apply_selected()
        assert ed.zones[1]["type"] == "entrance" and "solo puede ser la entrada" in said[-1]
        # la entrada de antes (una zona) se sigue viendo, con su sentido, y se puede pasar a línea
        ed.select(0)
        assert str(ed.tolinebtn.cget("state")) == "normal" and str(ed.flipbtn.cget("state")) == "disabled"
        ed.apply_selected()  # (cambiarle el nombre no le saca el sentido)
        assert ed.zones[0]["direction"] == "arriba"
        ed.to_line()
        z = ed.zones[0]
        assert z["line"] is True and "direction" not in z and len(z["polygon"]) == 2
        start, tip = line_arrow(*z["polygon"], ed.W, ed.H, of.LINE_ARROW)
        assert tip[1] < start[1] and z["polygon"][0][1] == pytest.approx(0.7)  # por el medio, hacia arriba como antes
        ed.save()
        assert [(z["name"], z["type"], z.get("line"), len(z["polygon"])) for z in saved["zones"]] == [
            ("puerta vieja", "entrance", True, 2), ("entrada_2", "entrance", True, 2), ("Fondo", "entrance", True, 2)]
        parse_zones(saved["zones"])  # lo que se manda es lo que el servidor acepta
    finally:
        root.destroy()


def test_la_linea_se_ve_en_la_imagen_analizada_y_en_las_sugerencias():
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    from agent.ajuste import count_text, draw_counts
    from core.analyzer import annotate
    from core.pisadas import FootMap, suggest

    zones = parse_zones([LINE, {"name": "gondola", "type": "shelf", "polygon": [[0, 0.6], [0.4, 0.6], [0.4, 1], [0, 1]]}])
    frame = np.zeros((H, W, 3), np.uint8)
    out = annotate(frame, zones, [])
    assert out[180, 320].any() and out[205, 320].any()  # la línea y, debajo, la flecha
    assert not out[150, 320].any()  # del otro lado no hay flecha
    counts = {"zones": {"Puerta": {"type": "entrance", "visits": 0, "now": 0, "entries": 4, "exits": 3}}}
    assert count_text(zones[0], counts) == "entraron 4  salieron 3"
    draw_counts(out, zones, counts)
    # el reproductor de alertas (las zonas como las guarda el servidor)
    from agent.alerta_video import draw_overlay
    img = draw_overlay(frame.copy(), None, [LINE], None, None)
    assert img[180, 320].any()
    # «Sugerir zonas»: por donde aparece y se va la gente, una línea con la flecha hacia adentro
    fm = FootMap.empty()
    fm.persons = 8
    fm.appear = [(0.4 + 0.02 * i, 0.97, 0.0, -0.05) for i in range(8)]  # aparecen por abajo y caminan hacia arriba
    fm.vanish = []
    door = [z for z in suggest(fm, existing=[]) if z["type"] == "entrance"]
    assert len(door) == 1 and door[0]["line"] is True and len(door[0]["polygon"]) == 2
    a, b = door[0]["polygon"]
    assert line_arrow(a, b, W, H, 40)[1][1] < a[1] * H  # la flecha, hacia arriba (hacia adentro del local)
    assert not [z for z in suggest(fm, existing=door) if z["type"] == "entrance"]  # ya marcada: no la repite
