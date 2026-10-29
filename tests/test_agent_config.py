"""Configuración del agente empaquetado (.exe): archivo agente.env y pedido interactivo."""

import io

from agent.__main__ import (SERVIDOR_POR_DEFECTO, corregir_servidor, pedir_datos_interactivo, read_env_file,
                            servidor_actual, write_env_file)


def test_archivo_env_ida_y_vuelta(tmp_path):
    p = tmp_path / "agente.env"
    assert read_env_file(p) == {}
    write_env_file(p, {"VIGIA_SERVER": "https://x", "VIGIA_AGENT_TOKEN": "tok-1"})
    assert read_env_file(p) == {"VIGIA_SERVER": "https://x", "VIGIA_AGENT_TOKEN": "tok-1"}
    p.write_text("# comentario\nVIGIA_SERVER = 'https://y' \n\nbasura\n", encoding="utf-8")
    assert read_env_file(p) == {"VIGIA_SERVER": "https://y"}


def test_la_direccion_vieja_del_servidor_pasa_sola_a_la_nueva(tmp_path):
    # el 29 oct se borró vigia-three-ashy.vercel.app: la app y las máquinas que la tenían guardada siguen andando
    assert servidor_actual("https://vigia-three-ashy.vercel.app/") == SERVIDOR_POR_DEFECTO
    assert servidor_actual("http://127.0.0.1:8000") == "http://127.0.0.1:8000"
    assert servidor_actual(None) is None
    p = tmp_path / "agente.env"
    write_env_file(p, {"VIGIA_SERVER": "https://vigia-three-ashy.vercel.app", "VIGIA_SESSION": "s-1"})
    corregir_servidor(p)
    assert read_env_file(p) == {"VIGIA_SERVER": SERVIDOR_POR_DEFECTO, "VIGIA_SESSION": "s-1"}
    write_env_file(p, {"VIGIA_SERVER": "http://otro"})
    corregir_servidor(p)
    assert read_env_file(p) == {"VIGIA_SERVER": "http://otro"}
    corregir_servidor(tmp_path / "no-existe.env")  # sin archivo: nada


def test_pide_token_por_consola_y_lo_guarda(tmp_path, monkeypatch):
    p = tmp_path / "agente.env"

    class Tty(io.StringIO):
        def isatty(self):
            return True
    tok = "t" * 43  # token pegado a mano (sin usuario y clave)
    monkeypatch.setattr("sys.stdin", Tty(f"\n\n{tok}\n"))  # servidor por defecto, vacío, token
    server, token = pedir_datos_interactivo(p, None, None)
    assert server.startswith("https://") and token == tok
    assert read_env_file(p)["VIGIA_AGENT_TOKEN"] == tok
    # sin consola no pide nada
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert pedir_datos_interactivo(p, "s", None) == ("s", "")


def test_mosaico_de_la_ventana():
    import numpy as np
    from agent.ventana import mosaic, TILE_W, TILE_H
    assert mosaic([]).shape == (TILE_H, TILE_W, 3)
    frame = np.zeros((576, 704, 3), np.uint8)
    assert mosaic([("A", frame, "")]).shape == (TILE_H, TILE_W, 3)
    assert mosaic([("A", frame, ""), ("B", None, "Escuchando")]).shape == (TILE_H, 2 * TILE_W, 3)
    assert mosaic([("A", None, "x")] * 5).shape == (2 * TILE_H, 3 * TILE_W, 3)


def test_vincular_pc_con_usuario_y_clave(tmp_path, db_url):
    from fastapi.testclient import TestClient

    from server.main import Settings, create_app

    app = create_app(Settings(data_dir=tmp_path / "data", database_url=db_url, admin_password="clave-admin-123",
                              serve_public=False))
    with TestClient(app) as c:
        r = c.post("/api/agent/login", json={"username": "admin", "password": "mala", "pc_name": "Oficina"})
        assert r.status_code == 401
        r = c.post("/api/agent/login", json={"username": "admin", "password": "clave-admin-123", "pc_name": "Oficina"})
        assert r.status_code == 200 and r.json()["central"] is True and r.json()["name"] == "Oficina"
        tok1 = r.json()["token"]
        assert c.get("/api/agent/config", headers={"X-Agent-Token": tok1}).status_code == 200
        # volver a entrar con el mismo nombre reutiliza la PC y cambia el token (el viejo deja de servir)
        r2 = c.post("/api/agent/login", json={"username": "admin", "password": "clave-admin-123", "pc_name": "Oficina"}).json()
        assert r2["id"] == r.json()["id"] and r2["token"] != tok1
        assert c.get("/api/agent/config", headers={"X-Agent-Token": tok1}).status_code == 401
        # un empleado no puede vincular PCs
        c.post("/api/login", json={"username": "admin", "password": "clave-admin-123"})
        c.post("/api/users", json={"username": "empleado", "password": "clave-empl-1", "role": "entrenador"})
        r = c.post("/api/agent/login", json={"username": "empleado", "password": "clave-empl-1", "pc_name": "X"})
        assert r.status_code == 403
