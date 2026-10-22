"""Configuración del agente empaquetado (.exe): archivo agente.env y pedido interactivo."""

import io

from agent.__main__ import pedir_datos_interactivo, read_env_file, write_env_file


def test_archivo_env_ida_y_vuelta(tmp_path):
    p = tmp_path / "agente.env"
    assert read_env_file(p) == {}
    write_env_file(p, {"VIGIA_SERVER": "https://x", "VIGIA_AGENT_TOKEN": "tok-1"})
    assert read_env_file(p) == {"VIGIA_SERVER": "https://x", "VIGIA_AGENT_TOKEN": "tok-1"}
    p.write_text("# comentario\nVIGIA_SERVER = 'https://y' \n\nbasura\n", encoding="utf-8")
    assert read_env_file(p) == {"VIGIA_SERVER": "https://y"}


def test_pide_token_por_consola_y_lo_guarda(tmp_path, monkeypatch):
    p = tmp_path / "agente.env"

    class Tty(io.StringIO):
        def isatty(self):
            return True
    monkeypatch.setattr("sys.stdin", Tty("\n\ntok-2\n"))  # servidor por defecto, token vacío, token
    server, token = pedir_datos_interactivo(p, None, None)
    assert server.startswith("https://") and token == "tok-2"
    assert read_env_file(p)["VIGIA_AGENT_TOKEN"] == "tok-2"
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
