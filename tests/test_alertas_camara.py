"""«Qué alertas da…» por cámara y «⬇ PDF de métricas» del cliente (9 nov, el dueño: «hay muchas alertas de gente fuera
de horario en Minas, ¿cómo desactivo que me avise esto?» y «dentro de un cliente, descargar su PDF con las métricas y
listo»). En Minas eran todas de los 15 min después de cerrar (los empleados cerrando) en cámaras que no miran con el
local cerrado, y de una cámara a la calle que mira las 24 h."""

import os
import queue
import types
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from core.rules import FUERA_DE_HORARIO, RuleEngine, parse_config
from core.tracker import Track
from server.main import Settings, create_app

ADMIN = ("admin", "clave-admin-123")
TZ = ZoneInfo("America/Montevideo")
HORARIO = {d: "06:00-20:00" for d in ("lun", "mar", "mie", "jue", "vie", "sab", "dom")}


@pytest.fixture
def app(tmp_path, db_url):
    return create_app(Settings(data_dir=tmp_path / "data", database_url=db_url, admin_password=ADMIN[1],
                               serve_public=False))


def _config(a) -> dict:
    tok = a.post("/api/agent/session").json()["token"]
    ag = TestClient(a.app, headers={"X-Agent-Token": tok})
    return {c["name"]: c for c in ag.get("/api/agent/config").json()["cameras"]}


def test_fuera_de_horario_no_avisa_al_abrir_ni_al_cerrar():
    zones, cfg = parse_config({"zones": [], "schedule": {"lun": "09:00-17:00"}})
    eng = RuleEngine(zones, cfg)
    t = Track(1, (100, 100, 150, 250), "persona", 0.0, 0.0, hits=5)
    lun = lambda hh, mm: datetime(2026, 10, 26, hh, mm, tzinfo=TZ)  # noqa: E731
    assert eng.update([t], 0.0, (640, 360), lun(17, 10)) == []  # cerrando: los empleados
    assert eng.update([t], 1.0, (640, 360), lun(8, 50)) == []  # abriendo
    t.last_ts = 400.0
    evs = eng.update([t], 400.0, (640, 360), lun(22, 0))
    assert [e.type for e in evs] == [FUERA_DE_HORARIO]


def test_con_el_local_cerrado_apagada_no_se_manda_la_regla_y_se_elige_que_avisa(app):
    with TestClient(app) as a:
        a.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]})
        me = a.get("/api/me").json()
        minas = a.post("/api/clients", json={"name": "Minas", "hours": HORARIO}).json()
        sin = a.post("/api/clients", json={"name": "Sin horario"}).json()
        cam = lambda cid, n, **kw: a.post("/api/cameras", json={  # noqa: E731
            "client_id": cid, "name": n, "brand": "ip", "host": "dvr", "password": "x", "operator_id": me["id"],
            **kw}).json()
        entrada = cam(minas["id"], "Entrada")
        porton = cam(minas["id"], "Portón calle", profile="exterior", off_hours="vivo")
        otra = cam(sin["id"], "Caja")
        cfg = _config(a)
        # la cámara de adentro no mira con el local cerrado: «persona con el local cerrado» no va
        assert cfg["Entrada"]["analysis"]["rules"]["enabled_rules"] == ["salida_sin_pago", "zona_restringida"]
        assert cfg["Portón calle"]["analysis"]["rules"]["enabled_rules"] == [FUERA_DE_HORARIO]
        assert FUERA_DE_HORARIO in cfg["Caja"]["analysis"]["rules"]["enabled_rules"]  # sin horario, como siempre

        # el portón mira de noche: se le apaga a mano y queda en «solo métricas» (no tenía otra cosa)
        r = a.post("/api/cameras/alerts", json={"camera_ids": [porton["id"]], "rules": []})
        assert r.status_code == 200, r.text
        assert r.json()["cameras"][0]["rules"]["enabled_rules"] == []
        assert _config(a)["Portón calle"]["metrics_only"]

        # a la entrada se le saca un gesto sin tocar las reglas; volver a lo del tipo deja de fijarlo
        before = a.get("/api/cameras").json()
        ent = next(c for c in before if c["id"] == entrada["id"])
        assert ent["gestures_default"] == ["mano_bolsillo", "mano_ropa", "mano_bolsa"]
        a.post("/api/cameras/alerts", json={"camera_ids": [entrada["id"]], "gestures": ["mano_bolsa"]})
        assert _config(a)["Entrada"]["analysis"]["gestures"]["enabled"] == ["mano_bolsa"]
        a.post("/api/cameras/alerts", json={"camera_ids": [entrada["id"]],
                                            "gestures": ["mano_bolsillo", "mano_ropa", "mano_bolsa"]})
        ent = next(c for c in a.get("/api/cameras").json() if c["id"] == entrada["id"])
        assert ent["gestures"] is None and "enabled_rules" not in ent["rules"]

        # apagar todas las reglas no apaga de rebote los gestos que miraba (el tipo ya no los prendería)
        a.post("/api/cameras/alerts", json={"camera_ids": [otra["id"]], "rules": []})
        c = next(c for c in a.get("/api/cameras").json() if c["id"] == otra["id"])
        assert c["gestures"] == {"enabled": ["mano_bolsillo", "mano_ropa", "mano_bolsa"]}
        assert not _config(a)["Caja"]["metrics_only"]

        bad = a.post("/api/cameras/alerts", json={"camera_ids": [otra["id"]], "rules": ["merodeo"]})
        assert bad.status_code == 422 and "merodeo" in bad.json()["detail"]
        assert a.post("/api/cameras/alerts", json={"camera_ids": [9999], "rules": []}).status_code == 404

        # un operador también puede
        a.post("/api/users", json={"username": "lucia", "password": "clave-lucia-123", "role": "entrenador"})
        op = TestClient(app)
        op.post("/api/login", json={"username": "lucia", "password": "clave-lucia-123"})
        assert op.post("/api/cameras/alerts", json={"camera_ids": [otra["id"]], "rules": ["zona_restringida"]}
                       ).status_code == 200


# ---------------------------------------------------------------- app
def _tk():
    tk = pytest.importorskip("tkinter")
    if os.name != "nt" and not os.environ.get("DISPLAY"):
        pytest.skip("sin pantalla (correr con xvfb-run)")
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"sin Tk: {exc}")
    return tk, root


def _office(tk, root, cams, client, posts, gets):
    import agent.oficina as of
    from tkinter import ttk
    o = of.Oficina.__new__(of.Oficina)
    o.tk, o.ttk = tk, ttk
    o.v = types.SimpleNamespace(root=root, log_handler=types.SimpleNamespace(q=queue.Queue()))
    o.mb = types.SimpleNamespace(showinfo=lambda *a, **k: posts.append(("info", a)), showwarning=lambda *a, **k: None,
                                 showerror=lambda *a, **k: posts.append(("error", a)), askyesno=lambda *a, **k: False)
    o.bg = types.SimpleNamespace(run=lambda fn, done=None, fail=None: done(fn()) if done else fn())
    o.api = types.SimpleNamespace(post=lambda path, body: posts.append((path, body)) or {"updated": len(cams)},
                                  get=lambda path, **kw: gets.append((path, kw)) or b"%PDF-1.4 prueba")
    o.cameras, o.clients = cams, [client]
    o._sel_cameras = lambda: cams
    o._sel_client = lambda: client
    o._client_of = lambda c: client
    o.config_saved = lambda: posts.append(("guardado", None))
    return o


def _cam(i, **kw):
    return {"id": i, "client_id": 3, "name": f"Cámara {i}", "plans": None, "measures": ["hurto"], "rules": {},
            "rules_default": ["salida_sin_pago", "zona_restringida", "fuera_de_horario"], "gestures": None,
            "gestures_default": ["mano_bolsillo", "mano_ropa", "mano_bolsa"], **kw}


def _find(widget, cls_name, text=None):
    for w in widget.winfo_children():
        if w.winfo_class() == cls_name and (text is None or str(w.cget("text")) == text):
            return w
        got = _find(w, cls_name, text)
        if got is not None:
            return got
    return None


def test_que_alertas_da_solo_manda_lo_que_se_toco():
    tk, root = _tk()
    try:
        client = {"id": 3, "name": "Minas", "plans": ["hurto"]}
        cams = [_cam(1), _cam(2, rules={"enabled_rules": ["zona_restringida"]})]
        posts, gets = [], []
        o = _office(tk, root, cams, client, posts, gets)
        o.alerts_dialog()
        root.update()
        top = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)][-1]
        top.update_idletasks()
        assert top.winfo_reqwidth() <= 1366 and top.winfo_reqheight() <= 740
        box = _find(top, "TCheckbutton", "Persona con el local cerrado")
        assert box is not None and "selected" not in box.state()  # la 2 no la tiene: no se marca
        box.invoke()  # se la prende a las dos; los gestos no se tocan
        _find(top, "TButton", "Guardar").invoke()
        path, body = next(p for p in posts if p[0] == "/api/cameras/alerts")
        assert body["camera_ids"] == [1, 2] and body["gestures"] is None
        assert body["rules"] == ["zona_restringida", "fuera_de_horario"]  # lo que daban las dos + lo tocado
        assert ("guardado", None) in posts

        # una cámara que no mide hurtos: lo dice y no abre nada
        posts.clear()
        o2 = _office(tk, root, [_cam(5, measures=["comercio"])], client, posts, gets)
        o2.alerts_dialog()
        assert posts and posts[0][0] == "info" and "no mide hurtos" in posts[0][1][1]
    finally:
        root.destroy()


def test_pdf_de_metricas_del_cliente(tmp_path, monkeypatch):
    tk, root = _tk()
    try:
        from tkinter import filedialog
        dest = tmp_path / "reporte.pdf"
        seen = {}

        def ask(**kw):
            seen.update(kw)
            return str(dest)
        monkeypatch.setattr(filedialog, "asksaveasfilename", ask)
        posts, gets = [], []
        o = _office(tk, root, [], {"id": 3, "name": "Minas / Centro", "plans": ["hurto"]}, posts, gets)
        o.client_pdf(7)
        assert dest.read_bytes().startswith(b"%PDF")
        path, kw = gets[-1]
        assert path == "/api/metrics/report.pdf" and kw["client_id"] == 3
        d1, d2 = datetime.fromisoformat(kw["desde"]), datetime.fromisoformat(kw["hasta"])
        assert (d2 - d1).days == 6 and seen["initialfile"].startswith("metricas_Minas_Centro_")
        o.client_pdf(1)
        assert gets[-1][1]["desde"] == gets[-1][1]["hasta"]
    finally:
        root.destroy()
