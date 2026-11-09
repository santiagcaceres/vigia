"""La app y la detección de personas de cada cámara (pedido del dueño, 6 nov: «¿cómo puedo aumentar la facilidad o
tolerancia para detectar personas? si están lejos no las cuenta»): «Sensibilidad» (1 a 10) y «Personas lejos» en el
formulario de la cámara y en su ventana («● En vivo» / «🎞 Clips»), donde se prueban en el acto y se guardan con
«Guardar» o «Aprobar». Las pruebas con ventana corren con xvfb-run."""

import os
import queue
import time
import types

import numpy as np
import pytest

import agent.oficina as of
from core.deteccion import DetectionConfig


# ------------------------------------------------------------------ sin ventana
def test_la_sensibilidad_del_formulario():
    assert of.detection_from_form("8", True) == DetectionConfig(8, True)
    assert of.detection_from_form(" 5 ", False) == DetectionConfig() and DetectionConfig().is_default
    assert of.detection_from_form(1, 0) == DetectionConfig(1, False)
    for bad in ("0", "11", "", "abc", "7.5", None):
        with pytest.raises(ValueError, match="1 a 10"):
            of.detection_from_form(bad, False)


def test_el_texto_del_detalle():
    assert of.detection_text({}) == "" and of.detection_text(None) == ""
    assert of.detection_text({"detection": {"sensitivity": 5, "far": False}}) == ""  # la de siempre: no se dice
    assert of.detection_text({"detection": {"sensitivity": 8, "far": True}}) == "sensibilidad 8 de 10 · personas lejos"
    assert of.detection_text({"detection": {"far": True}}) == "personas lejos"
    assert of.detection_text({"detection": {"sensitivity": "mucha"}}) == ""  # (tolerante, como los agentes)
    assert "1280x720" in of.DETECTION_HELP and "5 veces" in of.DETECTION_HELP
    assert "como 4 cámaras" in of.DETECTION_HELP and "4 cámaras" in of.FAR_LOAD_WARN


def test_el_peso_de_lejos_en_la_app_es_el_del_reparto():
    from core.deteccion import FAR_WEIGHT
    from server import pool
    assert pool.FAR_FACTOR == FAR_WEIGHT  # (lo que dice la app es lo que cuenta el reparto)


def test_el_detalle_avisa_si_la_maquina_todavia_no_aplica_la_deteccion():
    cam = {"analyzed": True, "mode": "vivo", "detection": {"sensitivity": 8, "far": True},
           "status": {"state": "en_linea"}}
    assert "la toma al reiniciarse" in of.detection_pending(cam)  # máquina vieja: no manda «detection»
    cam["status"]["detection"] = "normal"  # máquina nueva que todavía no pidió la configuración
    assert "en menos de un minuto" in of.detection_pending(cam)
    cam["status"]["detection"] = "sensibilidad 8 de 10 · personas lejos"
    assert of.detection_pending(cam) == ""
    for other in ({"analyzed": False}, {"mode": "alarmas_equipo"}, {"status": {"state": "sin_conexion"}},
                  {"detection": None}):
        assert of.detection_pending({**cam, "status": {"state": "en_linea"}, **other}) == "", other


def test_lo_que_cambio_de_la_deteccion_en_palabras():
    d = DetectionConfig
    assert of.detection_change(d(), d(8)) == "sensibilidad 8 de 10 (acepta personas con confianza desde 0,23 de 1)"
    assert of.detection_change(d(), d(far=True)) == ("«Personas lejos» (la IA mira la imagen entera y en 4 partes: "
                                                     "unas 5 veces el trabajo)")
    assert of.detection_change(d(far=True), d()) == "sin «Personas lejos» (solo la imagen entera)"
    assert of.detection_change(d(8), d()) == ("sensibilidad 5 de 10 (acepta personas con confianza desde 0,35 de 1, "
                                              "como siempre)")
    assert " · " in of.detection_change(d(), d(9, True))


# ------------------------------------------------------------------ con ventana
def _tk():
    tk = pytest.importorskip("tkinter")
    if os.name != "nt" and not os.environ.get("DISPLAY"):
        pytest.skip("sin pantalla (correr con xvfb-run)")
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"sin Tk: {exc}")
    from agent import estilo
    estilo.aplicar(root)  # (las medidas, con las letras y estilos de la app)
    return tk, root


def _run(fn, done=None, fail=None):
    try:
        res = fn()
    except Exception as exc:  # noqa: BLE001
        if fail:
            return fail(exc)
        raise
    return done(res) if done else None


def _walk(w):
    yield w
    for c in w.winfo_children():
        yield from _walk(c)


def _find(top, cls: str, text: str | None = None):
    for w in _walk(top):
        if w.winfo_class() == cls and (text is None or str(w.cget("text")) == text):
            return w
    raise AssertionError(f"no hay {cls} {text!r}")


class Msgs:
    def __init__(self, answer=True):
        self.seen, self.answer = [], answer

    def showinfo(self, *a, **k):
        self.seen.append(("info", a[1] if len(a) > 1 else ""))

    def showwarning(self, *a, **k):
        self.seen.append(("warning", a[1] if len(a) > 1 else ""))

    def showerror(self, *a, **k):
        self.seen.append(("error", a[1] if len(a) > 1 else ""))

    def askyesno(self, *a, **k):
        self.seen.append(("ask", a[1] if len(a) > 1 else ""))
        return self.answer


# ---------- formulario de la cámara
def _oficina(tk, root, cams, msgs):
    from tkinter import ttk
    off = of.Oficina.__new__(of.Oficina)
    off.tk, off.ttk = tk, ttk
    off.v = types.SimpleNamespace(root=root)
    off.mb = msgs
    off.bg = types.SimpleNamespace(run=_run)
    client = {"id": 1, "name": "Depósito Demo", "plans": ["comercio"]}
    off.clients, off.cameras = [client], cams
    off._sel_client = lambda: client
    off.sent = []
    off.api = types.SimpleNamespace(put=lambda path, body: off.sent.append(("PUT", path, body)) or {**body, "id": 9},
                                    post=lambda path, body: off.sent.append(("POST", path, body)) or {**body, "id": 10})
    off.config_saved = lambda: None
    return off


def _cam(**kw):
    return {"id": 9, "client_id": 1, "name": "Depósito fondo", "brand": "hikvision", "host": "192.0.2.10",
            "rtsp_port": 554, "http_port": 80, "channel": 1, "username": "vigia", "has_password": True,
            "mode": "vivo", "profile": "deposito", "process_fps": 5, "detector": "yolo", "enabled": True, **kw}


def _open_dialog(off, root, cam):
    before = set(root.winfo_children())
    off.camera_dialog(cam)
    root.update()
    top = next(w for w in root.winfo_children() if w not in before)
    box = _find(top, "TLabelframe", "Detección de personas")
    return top, _find(box, "TSpinbox"), _find(box, "TCheckbutton")


def test_el_formulario_de_la_camara_manda_la_deteccion():
    tk, root = _tk()
    try:
        msgs = Msgs()
        cam = _cam(detection={"sensitivity": 5, "far": False})
        off = _oficina(tk, root, [cam], msgs)
        # editar sin tocar la detección: no se manda (None en el servidor = no cambia; no pisa lo de otro usuario)
        top, sb, cb = _open_dialog(off, root, cam)
        assert sb.get() == "5" and "selected" not in cb.state()
        _find(top, "TButton", "Guardar").invoke()
        assert off.sent and "detection" not in off.sent[-1][2], msgs.seen
        # subir la sensibilidad y prender «Personas lejos»: va completa
        top, sb, cb = _open_dialog(off, root, cam)
        sb.set("8")
        cb.invoke()
        _find(top, "TButton", "Guardar").invoke()
        assert off.sent[-1][:2] == ("PUT", "/api/cameras/9")
        assert off.sent[-1][2]["detection"] == {"sensitivity": 8, "far": True}
        # una guardada se muestra al editar
        top, sb, cb = _open_dialog(off, root, _cam(detection={"sensitivity": 9, "far": True}))
        assert sb.get() == "9" and "selected" in cb.state()
        # un número fuera de rango no se manda
        sb.set("12")
        n = len(off.sent)
        _find(top, "TButton", "Guardar").invoke()
        assert len(off.sent) == n and msgs.seen[-1] == ("warning", "La sensibilidad va de 1 a 10.")
        top.destroy()
        # cámara nueva (por link): va siempre
        top, sb, cb = _open_dialog(off, root, None)
        assert sb.get() == "5"
        _find(top, "Text").insert("1.0", "rtsp://vigia:clave@192.0.2.20:554/Streaming/Channels/101")
        _find(top, "TButton", "Guardar").invoke()
        assert off.sent[-1][:2] == ("POST", "/api/cameras")
        assert off.sent[-1][2]["detection"] == {"sensitivity": 5, "far": False}
        # el detector de movimiento no tiene sensibilidad
        top, sb, cb = _open_dialog(off, root, _cam(detector="motion"))
        assert "disabled" in sb.state() and "disabled" in cb.state()
        _find(top, "TButton", "Guardar").invoke()
        assert "detection" not in off.sent[-1][2]
    finally:
        root.destroy()


def test_el_formulario_entra_en_la_pantalla():
    """Con «Más opciones» abierto (modo alarmas) y la detección, entra en 1366x768 (menos la barra de tareas)."""
    tk, root = _tk()
    try:
        cam = _cam(mode="alarmas_equipo")
        off = _oficina(tk, root, [cam], Msgs())
        top, _sb, _cb = _open_dialog(off, root, cam)
        top.update_idletasks()
        assert top.winfo_reqwidth() <= 1366 and top.winfo_reqheight() <= 768 - 70, (top.winfo_reqwidth(),
                                                                                      top.winfo_reqheight())
    finally:
        root.destroy()


def test_el_editor_de_zonas_guarda_la_deteccion_que_se_probaba():
    tk, root = _tk()
    from tkinter import ttk
    try:
        sent = []
        app = types.SimpleNamespace(
            tk=tk, ttk=ttk, v=types.SimpleNamespace(root=root, log_handler=types.SimpleNamespace(q=queue.Queue())),
            mb=Msgs(), bg=types.SimpleNamespace(run=_run),
            api=types.SimpleNamespace(put=lambda path, body: sent.append((path, body)) or {}),
            config_saved=lambda: None, zones_changed=lambda *a, **k: None, _open_window=lambda cid: None,
            agent_of=None)
        app._dialog = types.MethodType(of.Oficina._dialog, app)
        cam = {"id": 9, "name": "Depósito fondo", "has_preview": False, "zones": [],
               "save_detection": {"sensitivity": 8, "far": True}}
        ed = of.ZonesEditor(app, cam, image=np.zeros((360, 640, 3), np.uint8), on_saved=lambda body: None)
        root.update()
        ed.save()
        assert sent[-1][1]["detection"] == {"sensitivity": 8, "far": True}
        cam.pop("save_detection")  # (sin cambios: no se manda)
        ed = of.ZonesEditor(app, cam, image=np.zeros((360, 640, 3), np.uint8), on_saved=lambda body: None)
        root.update()
        ed.save()
        assert "detection" not in sent[-1][1]
    finally:
        root.destroy()


# ---------- la ventana de la cámara («● En vivo» / «🎞 Clips»)
class FakeTuner:
    """Lo que la ventana le pide al motor (agent/ajuste.py), sin video ni detector."""
    made: list = []

    def __init__(self, cfg, detector, videos=None):
        self.cfg, self.detector, self.videos = cfg, detector, [str(v) for v in videos or []]
        self.live = not self.videos
        self.analysis = cfg.get("analysis") or {}
        self.detections: list = []
        self.state, self.started, self.stopped = "iniciando", False, False
        self.error, self.seg_now, self.progress, self.speed, self.truncated = None, 0, 0.0, 1.0, False
        self.seg_names = [os.path.basename(v) for v in self.videos]
        self.alerts, self.seg_errors, self.samples = [], {}, []
        FakeTuner.made.append(self)

    def bounds(self):
        return 0, len(self.samples)

    def sample_at(self, i):
        return self.samples[i] if 0 <= i < len(self.samples) else None

    def explain(self, i):
        return []

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def set_detection(self, cfg):
        self.detections.append(cfg)
        return True

    def set_analysis(self, analysis):
        self.analysis = analysis

    def needs_pose(self):
        return False

    def image(self, idx=None):
        return None

    def footmap(self):
        return None

    def segments(self):
        return []

    def hints(self):
        return ["Probá subir la «Sensibilidad»"]

    def stats(self):
        return {"frames": 30, "seconds": 6.0, "persons": 2, "people_pct": 80.0, "per_hour": None, "dropped": 0,
                "detections": 40, "height_pct": 14.0, "conf": 0.49, "alerts": 0, "objects": [], "door": None,
                "gestures": False, "pose": {}, "night_pct": 0, "sensitivity": 9, "far": True}


class FakeDetector:
    name, device, on_gpu, size, tunable = "YOLO11n", "CPU", False, (640, 640), True


class FakeAgent:
    def __init__(self, info):
        self.info = info
        self.client = types.SimpleNamespace(tune=lambda cid: self.info, preview=lambda cid, jpg: None)

    def tune_setup(self, cam_id):
        return self.info, FakeDetector()


def _info(detection=None, detector="yolo"):
    analysis = {"zones": [], "rules": {}, "profile": "deposito", "plans": ["comercio"], "client_plans": ["comercio"],
                "metrics_only": True}
    if detection is not None:
        analysis["detection"] = detection
    return {"camera": {"id": 9, "name": "Depósito fondo", "client_name": "Depósito Demo", "detector": detector,
                       "analysis": analysis},
            "raw": {"zones": [], "rules": {}, "gestures": None}, "analyzed": False, "tuning": "sin_ajustar"}


def _window(tk, root, monkeypatch, tmp_path, info, mode="vivo", videos=(), msgs=None, put=None, kind="hurto",
            bg=None, name="Depósito fondo"):
    from tkinter import ttk

    from agent import ajuste
    from agent.biblioteca import TestVideo
    monkeypatch.setattr(ajuste, "Tuner", FakeTuner)
    monkeypatch.setattr(of.TuneWindow, "tick", lambda self: None)  # (sin dibujar: el motor es falso)
    FakeTuner.made = []
    lib = [TestVideo(str(p), kind=kind) for p in videos]
    sent = []
    app = types.SimpleNamespace(
        tk=tk, ttk=ttk, v=types.SimpleNamespace(root=root, log_handler=types.SimpleNamespace(q=queue.Queue())),
        mb=msgs or Msgs(), bg=types.SimpleNamespace(run=bg or _run), _tune_windows=[],
        api=types.SimpleNamespace(put=put or (lambda path, body: sent.append((path, body)) or dict(body)),
                                  post=lambda path, body: sent.append((path, body)) or {}),
        config_saved=lambda: None, zones_changed=lambda *a, **k: None, _open_window=lambda cid: None,
        _clip_frame=lambda cam: None, library=lambda: types.SimpleNamespace(videos=lambda cid: list(lib)))
    agent = FakeAgent(info)
    app.agent_of = lambda: agent
    app._dialog = types.MethodType(of.Oficina._dialog, app)
    w = of.TuneWindow(app, {"id": 9, "name": name, "has_preview": False, "detector": "yolo"}, mode=mode)
    root.update()
    return w, app, sent, agent


def test_en_vivo_la_sensibilidad_rige_en_el_acto_y_se_guarda(monkeypatch, tmp_path):
    tk, root = _tk()
    try:
        w, app, sent, _agent = _window(tk, root, monkeypatch, tmp_path, _info({"sensitivity": 7}))
        t = FakeTuner.made[-1]
        assert t.live and t.analysis["detection"] == {"sensitivity": 7}  # analiza con la guardada
        assert w.det_sens.get() == "7" and not w.det_far.get() and not w.dirty
        assert str(_find(w.top, "TCheckbutton", "Personas lejos").cget("state")) == "normal"
        # se toca la sensibilidad: espera a que se termine de tocar (las flechitas no rehacen nada a cada paso)
        w.det_sens.set("9")
        root.update()
        assert t.detections == [] and w.dirty and str(w.save_btn.cget("text")) == "Guardar *"
        time.sleep(w.DET_WAIT_MS / 1000 + 0.2)
        root.update()
        assert t.detections == [DetectionConfig(9, False)] and len(FakeTuner.made) == 1  # el vivo sigue, no se rehace
        assert w.analysis["detection"] == {"sensitivity": 9}
        assert w._note == ("desde ahora, sin guardar: sensibilidad 9 de 10 (acepta personas con confianza desde 0,19 "
                           "de 1)")
        w._show_state(t)  # la nota va en el renglón de avisos, no pegada al estado en negrita
        assert "sensibilidad" not in str(w.state_lbl.cget("text"))
        assert str(w.live_warn.cget("text")).startswith("Desde ahora, sin guardar: sensibilidad 9 de 10")
        # «Personas lejos» y «Guardar» enseguida (sin esperar): se aplica antes de guardar, y va completa
        w.det_far.set(True)
        w.save_times()
        assert t.detections[-1] == DetectionConfig(9, True)
        bodies = [b for p, b in sent if p == "/api/cameras/9/analysis"]
        assert bodies[-1]["detection"] == {"sensitivity": 9, "far": True}
        assert not w.dirty and w._saved_det == DetectionConfig(9, True)
        # guardar de nuevo sin tocarla: no se manda
        w.save_times()
        assert "detection" not in [b for p, b in sent if p == "/api/cameras/9/analysis"][-1]
        # «Cómo anda la cámara» dice con qué se detectó
        w._show_stats(t)
        assert "confianza 0.49 · sensibilidad 9 · lejos" in w.stats_txt.get("1.0", "end")
        w.close()
    finally:
        root.destroy()


def test_con_clips_se_vuelve_a_analizar(monkeypatch, tmp_path):
    tk, root = _tk()
    try:
        clip = tmp_path / "Depósito fondo - hurto.mp4"
        clip.write_bytes(b"x")
        w, app, sent, _agent = _window(tk, root, monkeypatch, tmp_path, _info(), mode="clips", videos=[clip])
        first = FakeTuner.made[-1]
        assert not first.live and first.videos == [str(clip)] and "detection" not in first.analysis
        w.det_far.set(True)
        w._flush_detection()
        again = FakeTuner.made[-1]
        assert again is not first and first.stopped and again.started  # el clip se vuelve a analizar…
        assert again.analysis["detection"] == {"far": True} and again.videos == [str(clip)]  # …con «Personas lejos»
        assert w._note == "se vuelve a analizar con personas lejos (5 veces el trabajo)"
        # el mismo valor otra vez no rehace nada
        w.det_far.set(True)
        w._flush_detection()
        assert FakeTuner.made[-1] is again
        # cerrar con la detección sin guardar pregunta
        app.mb.answer = False
        w.close()
        assert not w.closed and "detección de personas" in app.mb.seen[-1][1]
        # «Aprobar» guarda la detección que se estaba probando
        app.mb.answer = True
        w.approve()
        bodies = [b for p, b in sent if p.endswith("/analysis")]
        assert bodies and bodies[-1]["detection"] == {"sensitivity": 5, "far": True}
        assert any("la detección que estás probando" in m for _k, m in app.mb.seen)
    finally:
        root.destroy()


def test_la_ventana_entra_en_la_pantalla(monkeypatch, tmp_path):
    """En 1366x768 (menos la barra de tareas), en vivo y con clips, con una cámara que ya se analiza (el aviso ⚠) y con
    la nota más larga de la detección; la barra para recorrer el video no queda chica (con ella se vuelve al momento
    de un hurto) y la detección va en su renglón, sin ensanchar la imagen."""
    tk, root = _tk()
    try:
        clip = tmp_path / "Minas - Pasillo bebidas - 2026-10-28 08.00-08.30.mp4"
        clip.write_bytes(b"x")
        # (las más largas que arma la ventana)
        notes = ("desde ahora, sin guardar: " + of.detection_change(DetectionConfig(), DetectionConfig(10, True)),
                 "desde ahora, sin guardar: " + of.detection_change(DetectionConfig(), DetectionConfig(far=True)),
                 "se vuelve a analizar con sensibilidad 10 de 10 · personas lejos (5 veces el trabajo), primero el "
                 "clip que mirabas · se pierden 3 etiquetas puestas a mano",
                 "se vuelve a analizar con sensibilidad 10 de 10 · personas lejos (5 veces el trabajo) (cambió la "
                 "guardada) · se pierden 3 etiquetas puestas a mano",
                 ("desde ahora se mira la postura, para los gestos y la IA de hurtos (lo de antes se analizó sin "
                  "mirarla)"))
        for mode, videos in (("vivo", ()), ("clips", (clip,))):  # («🎞 Clips» tiene la lista de clips: más alta)
            sizes = []
            for det in (None, {"sensitivity": 10, "far": True}):
                for analyzed in (False, True):
                    info = _info(det)
                    info["analyzed"] = analyzed
                    w, *_ = _window(tk, root, monkeypatch, tmp_path, info, mode=mode, videos=videos,
                                    name="Depósito estanterías del fondo (galpón 2)")  # (un nombre largo, arriba)
                    w.pos.configure(text="14:37:05 · cuadro 1234 de 3000 · mostrando el último")
                    w.top.update()
                    assert w.scale.master.winfo_reqwidth() <= w.W, w.scale.master.winfo_reqwidth()
                    assert w.scale.winfo_width() >= 280, w.scale.winfo_width()  # (era 287 antes de la detección)
                    sizes.append((w.top.winfo_reqwidth(), w.top.winfo_reqheight()))
                    for note in notes:
                        w._note = note
                        w._show_warn()
                        w.top.update_idletasks()
                        sizes.append((w.top.winfo_reqwidth(), w.top.winfo_reqheight()))
                    w._note = ""
                    w.close(force=True)
            assert all(wd <= 1366 and ht <= 768 - 70 for wd, ht in sizes), (mode, sizes)
    finally:
        root.destroy()


def test_la_deteccion_no_se_toca_hasta_saber_la_guardada(monkeypatch, tmp_path):
    """Mientras dice «Preparando el detector…» la sensibilidad y «Personas lejos» están apagadas: lo tocado ahí lo
    pisaría lo guardado al terminar de preparar."""
    tk, root = _tk()
    try:
        pending = []
        w, *_ = _window(tk, root, monkeypatch, tmp_path, _info({"sensitivity": 7}),
                        bg=lambda fn, done=None, fail=None: pending.append((fn, done, fail)))
        assert all("disabled" in x.state() for x in w.det_widgets) and w.info is None
        while pending:
            fn, done, fail = pending.pop(0)
            _run(fn, done, fail)
        assert w.info is not None and not any("disabled" in x.state() for x in w.det_widgets)
        assert w.det_sens.get() == "7"
    finally:
        root.destroy()


def test_con_clips_se_vuelve_a_analizar_primero_el_que_se_mira(monkeypatch, tmp_path):
    tk, root = _tk()
    try:
        a, b = tmp_path / "Depósito fondo - a.mp4", tmp_path / "Depósito fondo - b.mp4"
        for p in (a, b):
            p.write_bytes(b"x")
        w, app, *_ = _window(tk, root, monkeypatch, tmp_path, _info(), mode="clips", videos=[a, b])
        t = FakeTuner.made[-1]
        assert t.videos == [str(a), str(b)]
        smp = lambda seg, ts: types.SimpleNamespace(seg=seg, ts=ts)  # noqa: E731
        t.samples = [smp(0, 1.0), smp(1, 1.0), smp(1, 2.0), smp(1, 2.2)]
        w.index = 2  # mirando «b» a los 2 s (con la persona del fondo)
        w.labels = {(1, "mano_bolsa", 3, 2.0): "hurto", (0, "mano_ropa", 1, 1.0): None}
        w.det_sens.set("8")
        w._flush_detection()
        again = FakeTuner.made[-1]
        assert again is not t and again.videos == [str(b), str(a)]  # primero el que se miraba; los demás después
        assert [v.path for v in w.run_videos] == [str(b), str(a)]
        assert w._note.startswith("se vuelve a analizar con sensibilidad 8 de 10, primero el clip que mirabas")
        assert w._note.endswith("se pierde 1 etiqueta puesta a mano")
        assert w._goto == (0, 2.0) and w.index is None
        # mientras se analiza muestra lo que va llegando; al llegar a ese momento, vuelve ahí
        again.samples = [smp(0, 1.0)]
        w._goto_moment(again, *again.bounds())
        assert w._goto == (0, 2.0) and w.index is None
        again.samples += [smp(0, 1.8), smp(0, 2.0), smp(0, 2.4)]
        w._goto_moment(again, *again.bounds())
        assert w._goto is None and w.index == 2
    finally:
        root.destroy()


def test_el_resultado_del_clip_dice_con_que_se_analizo_y_pregunta_una_vez(monkeypatch, tmp_path):
    tk, root = _tk()
    try:
        clip = tmp_path / "Depósito fondo - normal.mp4"
        clip.write_bytes(b"x")
        msgs = Msgs(answer=False)
        w, app, *_ = _window(tk, root, monkeypatch, tmp_path, _info(), mode="clips", videos=[clip], msgs=msgs,
                             kind="normal")
        w.clips = list(w.run_videos)  # (como recién subido)
        alert = types.SimpleNamespace(seg=0, type="mano_bolsa", track_id=1, ts=3.0, score=None, kept=True,
                                      description="", wall=None)
        t = FakeTuner.made[-1]
        t.state, t.alerts = "listo", [alert]
        w._clip_verdict(t)
        kind, text = msgs.seen[-1]
        assert kind == "ask" and text.startswith("Resultado con la configuración guardada de «Depósito fondo»")
        assert "¿Mandar esta alerta al entrenamiento" in text
        # se prueba otra sensibilidad: el clip se vuelve a analizar y el resultado lo dice; no vuelve a preguntar
        w.det_sens.set("9")
        w._flush_detection()
        t2 = FakeTuner.made[-1]
        t2.state, t2.alerts = "listo", [alert]
        w._clip_verdict(t2)
        kind, text = msgs.seen[-1]
        assert kind == "info"
        assert text.startswith("Resultado con la sensibilidad 9 que estás probando (sin guardar) de «Depósito fondo»")
        assert "«Mandar a entrenar»" in text
        assert of.TuneWindow._verdict_config(w) == "la sensibilidad 9 que estás probando (sin guardar)"
        w.det_far.set(True)
        w._flush_detection()
        assert w._verdict_config() == "la sensibilidad 9 y «Personas lejos» que estás probando (sin guardar)"
    finally:
        root.destroy()


def test_la_deteccion_guardada_por_fuera_vuelve_a_analizar_los_clips(monkeypatch, tmp_path):
    """Con «🎞 Clips» abierto, la detección guardada cambia desde afuera («Editar», otro usuario) y después se guardan
    zonas en el editor de la barra: lo analizado se detectó con la anterior, así que se vuelve a analizar."""
    tk, root = _tk()
    try:
        clip = tmp_path / "Depósito fondo - hurto.mp4"
        clip.write_bytes(b"x")
        w, app, sent, agent = _window(tk, root, monkeypatch, tmp_path, _info(), mode="clips", videos=[clip])
        t = FakeTuner.made[-1]
        agent.info = _info({"sensitivity": 8, "far": True})
        w.zones_saved({}, external=True)
        again = FakeTuner.made[-1]
        assert again is not t and again.analysis["detection"] == {"sensitivity": 8, "far": True}
        assert w.det_sens.get() == "8" and w.det_far.get() and not w.dirty
        assert "cambió la guardada" in w._note
        # sin cambios en la detección, zonas guardadas no vuelven a analizar (las recalcula el motor)
        n = len(FakeTuner.made)
        w.zones_saved({}, external=True)
        assert len(FakeTuner.made) == n
    finally:
        root.destroy()


def test_guardar_o_aprobar_con_lejos_avisa_cuanto_ocupa_en_la_maquina(monkeypatch, tmp_path):
    tk, root = _tk()
    try:
        msgs = Msgs(answer=False)
        info = _info()
        info["analyzed"] = True
        w, *_ = _window(tk, root, monkeypatch, tmp_path, info, msgs=msgs)
        assert "4 cámaras" not in w._confirm_live()
        w.det_far.set(True)
        w._flush_detection()
        assert of.FAR_LOAD_WARN in w._confirm_live()
        w.approve()
        assert of.FAR_LOAD_WARN in msgs.seen[-1][1]
        w._saved_det = DetectionConfig(far=True)  # ya guardada en una cámara que ya se analiza: no se repite
        assert of.FAR_LOAD_WARN not in w._confirm_live()
        w.approve()
        assert of.FAR_LOAD_WARN not in msgs.seen[-1][1]
        w.det_far.set(False)
        w._flush_detection()
        assert of.FAR_LOAD_WARN not in w._confirm_live()
    finally:
        root.destroy()


def test_las_zonas_guardadas_en_otro_editor_no_pierden_la_deteccion_probada(monkeypatch, tmp_path):
    tk, root = _tk()
    try:
        w, app, sent, agent = _window(tk, root, monkeypatch, tmp_path, _info())
        t = FakeTuner.made[-1]
        w.det_sens.set("8")
        w._flush_detection()
        assert t.detections == [DetectionConfig(8, False)]
        # el editor de la barra guarda zonas (no trae la detección que se probaba acá): se conserva
        info = _info()
        info["camera"]["analysis"]["zones"] = [{"name": "Fondo", "type": "aisle", "polygon": [[0, 0], [1, 0], [1, 1]]}]
        agent.info = info
        w.zones_saved({}, external=True)
        assert w.det_sens.get() == "8" and w.dirty
        assert t.analysis["detection"] == {"sensitivity": 8} and t.analysis["zones"]  # el motor, con las dos cosas
        # «✎ Marcar zonas» desde la ventana lleva la detección que se prueba
        seen = {}
        monkeypatch.setattr(of, "ZonesEditor", lambda app_, cam, **k: seen.setdefault("cam", cam))
        w.edit_zones()
        assert seen["cam"]["save_detection"] == {"sensitivity": 8, "far": False}
    finally:
        root.destroy()


def test_un_servidor_viejo_no_guarda_la_deteccion_y_se_avisa(monkeypatch, tmp_path):
    tk, root = _tk()
    try:
        msgs = Msgs()
        # (responde sin «detection»: no la conoce)
        w, *_ = _window(tk, root, monkeypatch, tmp_path, _info(), msgs=msgs,
                        put=lambda path, body: {k: v for k, v in body.items() if k != "detection"})
        w.det_sens.set("9")
        w.save_times()
        assert any(k == "warning" and "sensibilidad de detección" in m for k, m in msgs.seen)
        assert w._saved_det == DetectionConfig()  # no quedó guardada: la próxima vez se manda de nuevo
    finally:
        root.destroy()


def test_con_el_detector_de_movimiento_no_hay_sensibilidad(monkeypatch, tmp_path):
    tk, root = _tk()
    try:
        w, *_ = _window(tk, root, monkeypatch, tmp_path, _info(detector="motion"))
        assert all("disabled" in x.state() for x in w.det_widgets)
    finally:
        root.destroy()
