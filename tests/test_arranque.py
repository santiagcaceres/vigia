"""Pantalla de carga del .exe (agent/arranque.py): mensajes a PyInstaller y cierre, sin trabar nunca el programa."""


import pytest

from agent import arranque


class FakeSplash:
    def __init__(self):
        self.alive, self.texts, self.closed = True, [], 0

    def is_alive(self):
        return self.alive

    def update_text(self, msg):
        self.texts.append(msg)

    def close(self):
        self.closed += 1
        self.alive = False


@pytest.fixture
def splash(monkeypatch):
    fake = FakeSplash()
    monkeypatch.setattr(arranque, "pyi_splash", fake)
    return fake


def test_fuera_del_exe_no_hace_nada(monkeypatch):
    monkeypatch.setattr(arranque, "pyi_splash", None)
    assert not arranque.abierta()
    arranque.aviso("Cargando…", 50)  # no explota
    arranque.cerrar()


def test_los_mensajes_llevan_la_marca_y_el_avance(splash):
    arranque.aviso("Cargando la inteligencia artificial…", 74)
    arranque.aviso("Abriendo\nla   ventana…")  # una sola línea (el Tcl lee de a líneas)
    arranque.aviso("Listo", 150)  # el avance se limita a 0-100
    assert splash.texts == ["=74|Cargando la inteligencia artificial…", "=Abriendo la ventana…", "=100|Listo"]


def test_cerrar_una_sola_vez_y_despues_no_manda_nada(splash):
    arranque.cerrar()
    arranque.cerrar()
    arranque.aviso("tarde", 99)
    assert splash.closed == 1 and splash.texts == []


def test_un_error_de_la_pantalla_de_carga_no_traba_el_programa(splash):
    def boom(*_a):
        raise ConnectionError("se cerró el socket")
    splash.update_text = boom
    splash.close = boom
    arranque.aviso("Cargando…", 10)
    arranque.cerrar()


def test_la_plantilla_de_pyinstaller_se_puede_cambiar():
    """El .spec cambia el procedimiento de Tcl que dibuja el texto: si PyInstaller cambia la plantilla, la compilación
    avisa y queda la pantalla de fábrica. Esta prueba avisa antes (solo si PyInstaller está instalado)."""
    tpl = pytest.importorskip("PyInstaller.building.splash_templates")
    import re
    assert len(re.findall(r"proc canvas_text_update \{.*?\n\}", tpl.image_script, flags=re.S)) == 1
    assert "-image splash_image" in tpl.splash_canvas_setup
    assert "trace add variable status_text write" in tpl.splash_canvas_text


def test_la_imagen_y_las_medidas_coinciden():
    from pathlib import Path
    img = Path(arranque.__file__).parent / "recursos" / "splash.png"
    data = img.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    w, h = int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    assert (w, h) == arranque.SPLASH_SIZE
    x0, y0, x1, y1 = arranque.SPLASH_BAR
    assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h and arranque.SPLASH_TEXT[1] <= h
    assert 0 < arranque.EXTRACT_SHARE < 74  # las etapas de Python arrancan en 74 %
