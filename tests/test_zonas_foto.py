"""«✎ Marcar zonas» de una cámara sin imagen (6 nov, el dueño: «quiero marcar zonas y está totalmente en negro la
cámara, pero en el en vivo funciona; hay que poner alguna opción para tomar foto»). El editor saca solo una foto del
vivo (el cuadro de «En vivo» si se está mirando; si no, abre el video unos segundos), la guarda como imagen de la
cámara y tiene «Tomar foto del vivo» para cambiarla por una de ahora."""

import os
import queue
import types

import numpy as np
import pytest

import agent.oficina as of


def _tk():
    tk = pytest.importorskip("tkinter")
    if os.name != "nt" and not os.environ.get("DISPLAY"):
        pytest.skip("sin pantalla (correr con xvfb-run)")
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"sin Tk: {exc}")
    return tk, root


def _app(tk, root, agent, clip=None):
    from tkinter import ttk
    bg = types.SimpleNamespace(run=lambda fn, done=None, fail=None: _run(fn, done, fail))
    app = types.SimpleNamespace(
        tk=tk, ttk=ttk, v=types.SimpleNamespace(root=root, log_handler=types.SimpleNamespace(q=queue.Queue())),
        mb=types.SimpleNamespace(showinfo=lambda *a, **k: None, showwarning=lambda *a, **k: None,
                                 showerror=lambda *a, **k: None, askyesno=lambda *a, **k: True),
        bg=bg, api=types.SimpleNamespace(put=lambda path, body: {}), config_saved=lambda: None,
        zones_changed=lambda *a, **k: None, _open_window=lambda cid: None,
        agent_of=(lambda: agent) if agent is not None else None, _clip_frame=lambda cam: clip)
    app._dialog = types.MethodType(of.Oficina._dialog, app)
    return app


def _run(fn, done, fail):
    try:
        res = fn()
    except Exception as exc:  # noqa: BLE001
        return fail(exc) if fail else None
    return done(res) if done else None


class FakeAgent:
    def __init__(self, live=None, grab_fails=False):
        self.live, self.uploads, self.views = live, [], []
        self.client = types.SimpleNamespace(view=self._view, preview=lambda cid, jpg: self.uploads.append((cid, jpg)))
        self.grab_fails = grab_fails

    def live_frame(self, cid):
        return None if self.live is None else self.live.copy()

    def _view(self, ids):
        self.views.append(ids)
        return [{"id": i, "connection": {"brand": "hikvision", "host": "x"}} for i in ids]


def _cam():
    return {"id": 145, "name": "Camera 03", "has_preview": False, "zones": []}


def test_sin_imagen_usa_el_cuadro_de_en_vivo_y_lo_guarda():
    tk, root = _tk()
    try:
        live = np.full((360, 640, 3), 200, np.uint8)
        agent = FakeAgent(live=live)
        cam = _cam()
        ed = of.ZonesEditor(_app(tk, root, agent), cam)
        root.update()
        assert ed.base is not None and int(ed.base.mean()) == 200  # dibuja sobre la foto, no sobre el negro
        assert agent.views == []  # la cámara se estaba mirando: no se pide otra transmisión
        assert [c for c, _ in agent.uploads] == [145] and cam["has_preview"]  # queda como imagen de la cámara
        assert "Foto del vivo de las" in str(ed.info.cget("text"))
        ed.top.update_idletasks()
        assert ed.top.winfo_reqwidth() <= 1366  # el botón nuevo no ensancha la ventana
    finally:
        root.destroy()


def test_sin_en_vivo_abre_el_video_y_el_boton_saca_otra(monkeypatch):
    tk, root = _tk()
    from agent import nombres
    shots = [np.full((360, 640, 3), 50, np.uint8), np.full((360, 640, 3), 120, np.uint8)]
    monkeypatch.setattr(nombres, "grab_frame", lambda conn: shots.pop(0))
    try:
        agent = FakeAgent()
        ed = of.ZonesEditor(_app(tk, root, agent), _cam())
        root.update()
        assert agent.views == [[145]] and int(ed.base.mean()) == 50
        ed.snapshot()  # «Tomar foto del vivo»: una de ahora
        root.update()
        assert int(ed.base.mean()) == 120 and len(agent.uploads) == 2
        assert str(ed.snap_btn.cget("state")) == "normal"
    finally:
        root.destroy()


def test_si_el_vivo_no_abre_usa_un_clip_o_lo_dice(monkeypatch):
    tk, root = _tk()
    from agent import nombres

    def no_abre(conn):
        raise RuntimeError("El grabador está sin conexión")
    monkeypatch.setattr(nombres, "grab_frame", no_abre)
    try:
        clip = np.full((360, 640, 3), 90, np.uint8)
        ed = of.ZonesEditor(_app(tk, root, FakeAgent(), clip=clip), _cam())
        root.update()
        assert int(ed.base.mean()) == 90 and "Imagen de un clip" in str(ed.info.cget("text"))
        sin = of.ZonesEditor(_app(tk, root, FakeAgent()), _cam())
        root.update()
        assert sin.base is None and "No se pudo tomar la foto: El grabador está sin conexión" in str(sin.info.cget("text"))
        # la app todavía arrancando: lo dice y se puede dibujar igual
        sola = of.ZonesEditor(_app(tk, root, None), _cam())
        root.update()
        assert "arrancando" in str(sola.info.cget("text"))
    finally:
        root.destroy()


def test_en_vivo_guarda_el_ultimo_cuadro_sin_dibujos():
    from agent.__main__ import Agent
    from agent.vista import AIViewWorker
    from tests.test_ver_con_ia import CFG, _BurstCap, _Det

    # «Ver con la IA» dibuja recuadros y zonas en lo que muestra; para marcar zonas queda el cuadro tal cual
    w = AIViewWorker(CFG, lambda: _Det(), None)
    w._read(_BurstCap(seconds=1.0))
    assert w.raw is not None and int(w.raw.mean()) == 60

    ag = Agent.__new__(Agent)
    frame = np.full((10, 10, 3), 7, np.uint8)
    v = types.SimpleNamespace(state="en_linea", raw=frame)
    ag.viewers = {3: v}
    got = ag.live_frame(3)
    assert got is not None and got is not frame and int(got.mean()) == 7  # una copia
    v.state = "conectando"
    assert ag.live_frame(3) is None and ag.live_frame(9) is None
