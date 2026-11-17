"""Avisos «✔ Cliente guardado» al guardar, editar o borrar (pedido del dueño, 17 nov) y que un cierre al abrir deje
rastro y no se repita (el mismo día: «cuando hago doble clic en el programa se cierra de la nada»)."""

import os
import queue
import types

import pytest

import agent.oficina as of


def test_cada_accion_tiene_su_aviso():
    am = of.action_message
    assert am("POST", "/api/clients", {"name": "x"}) == "Cliente creado"
    assert am("PUT", "/api/clients/4", {}) == "Cliente guardado"
    assert am("DELETE", "/api/clients/4") == "Cliente eliminado"
    assert am("POST", "/api/cameras", {}) == "Cámara creada"
    assert am("PUT", "/api/cameras/9", {}) == "Cámara guardada"
    assert am("PUT", "/api/cameras/9/analysis", {}) == "Zonas y reglas guardadas"
    assert am("POST", "/api/cameras/delete", {"camera_ids": [1]}) == "Cámara eliminada"
    assert am("POST", "/api/cameras/delete", {"camera_ids": [1, 2, 3]}) == "3 cámaras eliminadas"
    assert am("POST", "/api/cameras/analyze", {"camera_ids": [1], "analyze": True}) == "Cámara puesta a analizar"
    assert am("POST", "/api/cameras/analyze", {"camera_ids": [1], "analyze": False}) == "Se dejó de analizar"
    assert am("POST", "/api/cameras/9/tuning", {"status": "aprobada"}) == "Cámara aprobada"
    assert am("POST", "/api/users", {}) == "Usuario creado"
    assert am("DELETE", "/api/users/3") == "Usuario eliminado"
    assert am("PATCH", "/api/events/5", {"label": "falso_positivo"}) == "Alerta marcada: Falsa alarma"
    assert am("PATCH", "/api/events/5", {"label": None}) == "Etiqueta de la alerta quitada"
    assert am("POST", "/api/training/ia/models/2/activate") == "Modelo activado"
    assert am("POST", "/api/models/2/deactivate") == "Modelo apagado"
    assert am("POST", "/api/cloud/action", {"action": "stop"}) == "Apagando la máquina…"
    # lo que la app pide sola no avisa
    for meth, path in (("POST", "/api/cameras/9/live"), ("POST", "/api/cameras/9/probe"), ("POST", "/api/logout"),
                       ("POST", "/api/agent/session"), ("POST", "/api/login"), ("GET", "/api/clients")):
        assert am(meth, path, {}) is None, path


def test_la_api_deja_el_aviso_solo_si_el_servidor_confirmo():
    import httpx

    def handler(req):
        if req.url.path == "/api/clients/4" and req.method == "PUT":
            return httpx.Response(200, json={"id": 4})
        return httpx.Response(409, json={"detail": "ya existe"})
    api = of.Api("http://x")
    api.http = httpx.Client(base_url="http://x", transport=httpx.MockTransport(handler))
    api.put("/api/clients/4", {"name": "Farmacia"})
    with pytest.raises(of.ApiError):
        api.post("/api/clients", {"name": "Otra"})
    assert api.done.get_nowait() == "Cliente guardado"
    with pytest.raises(queue.Empty):
        api.done.get_nowait()  # el que falló no avisa


def _tk():
    tk = pytest.importorskip("tkinter")
    if os.name != "nt" and not os.environ.get("DISPLAY"):
        pytest.skip("sin pantalla (correr con xvfb-run)")
    try:
        return tk, tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"sin Tk: {exc}")


def test_el_cartel_aparece_suma_repetidos_y_se_va_solo():
    tk, root = _tk()
    try:
        root.geometry("900x600+0+0")
        root.update()
        t = of.Toasts(tk, root)
        t.LIFE_MS = 300
        t.show("Cámara creada")
        t.show("Cámara creada")
        t.show("Cliente guardado")
        root.update()
        assert [i["text"] for i in t.items] == ["Cámara creada", "Cliente guardado"]
        assert t.items[0]["label"].cget("text") == "✔  Cámara creada  (×2)"
        assert all(i["win"].winfo_ismapped() for i in t.items)
        # la oficina los toma de la cola de la API en el hilo de la ventana
        off = of.Oficina.__new__(of.Oficina)
        off.api, off.toasts = types.SimpleNamespace(done=queue.Queue()), t
        off.api.done.put("Usuario creado")
        off.show_done()
        assert t.items[-1]["text"] == "Usuario creado"
        import time
        end = time.monotonic() + 3
        while t.items and time.monotonic() < end:
            root.update()
            time.sleep(0.05)
        assert t.items == []
    finally:
        root.destroy()


def test_si_se_cerro_cargando_la_ia_la_proxima_va_en_el_procesador(tmp_path, monkeypatch):
    from agent import __main__ as m
    monkeypatch.delenv("VIGIA_DISPOSITIVO", raising=False)
    m.ai_on_cpu_if_it_crashed(tmp_path)
    assert "VIGIA_DISPOSITIVO" not in os.environ  # abrió bien: nada cambia
    (tmp_path / m.AI_MARK).write_text("cargando la IA")  # quedó la marca: se cerró cargando
    m.ai_on_cpu_if_it_crashed(tmp_path)
    assert os.environ["VIGIA_DISPOSITIVO"] == "procesador"
    assert (tmp_path / m.AI_CPU_FLAG).exists() and not (tmp_path / m.AI_MARK).exists()
    monkeypatch.delenv("VIGIA_DISPOSITIVO")
    m.ai_on_cpu_if_it_crashed(tmp_path)  # las veces siguientes también
    assert os.environ["VIGIA_DISPOSITIVO"] == "procesador"


def test_un_error_al_abrir_queda_en_el_registro_y_en_pantalla(tmp_path, monkeypatch, capsys):
    from agent import __main__ as m
    monkeypatch.setenv("VIGIA_AGENT_DATA", str(tmp_path))
    monkeypatch.setattr(m, "_main", lambda: (_ for _ in ()).throw(RuntimeError("se rompió")))
    monkeypatch.setattr(m.os, "name", "posix")
    with pytest.raises(SystemExit):
        m.main()
    out = capsys.readouterr().err
    assert "no pudo abrir" in out and "se rompió" in out and str(tmp_path / "vigia.log") in out
