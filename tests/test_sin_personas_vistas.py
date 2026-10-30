"""Sin «personas vistas»: la suma de personas de todas las cámaras cuenta a la misma persona en cada cámara (y de nuevo si
se pierde y vuelve). En Minas daba 935 con 65 que entraron. La cantidad de gente sale de la puerta (entraron / salieron)."""

import io
from datetime import date
from pathlib import Path

import pytest

from agent.oficina import door_text
from core.metrics import empty_bucket

ROOT = Path(__file__).resolve().parent.parent


def bucket(**kw) -> dict:
    return {**empty_bucket(), "people": 935, "people_dwell_s": 935 * 32.0, "people_max": 6, "frames": 100, **kw}


def test_la_app_dice_cuantos_entraron_y_no_la_suma_de_personas():
    assert door_text(bucket(entries=65, exits=64, door=1)) == "Entraron 65 · salieron 64"
    assert door_text(bucket(entries=65, exits=64, door=1, staff_entries=3), staff=True).endswith(
        "3 de personal descontados")
    assert door_text(bucket(entries=12)).startswith("Pasaron por la entrada 12")  # sin sentido marcado
    sin_puerta = door_text(bucket())
    assert "935" not in sin_puerta and "zona Entrada" in sin_puerta


def test_ni_la_web_ni_el_pdf_ni_el_excel_suman_personas():
    js = (ROOT / "public" / "app.js").read_text(encoding="utf-8")
    assert "Apariciones" not in js and "vistas ·" not in js
    reporte = (ROOT / "server" / "reporte.py").read_text(encoding="utf-8")
    assert "Apariciones" not in reporte


@pytest.mark.parametrize("door", [True, False])
def test_el_pdf_no_muestra_la_suma_de_personas(door):
    pypdf = pytest.importorskip("pypdf")
    pytest.importorskip("fpdf")
    from server.reporte import metrics_pdf

    extra = {"entries": 65, "exits": 64, "door": 1} if door else {}
    day = bucket(**extra, zones={"Caja": {"type": "checkout", "visits": 179, "dwell_s": 1600.0}})
    m = {"total": day, "by_day": [{"date": "2026-10-29", **bucket()}, {"date": "2026-10-30", **day}],
         "by_hour": [{"hour": 9, **day}],
         "by_camera": [{"camera_id": 7, "name": "Entrada", **day}, {"camera_id": 44, "name": "Pasillo", **bucket()}]}
    for since in (date(2026, 10, 30), date(2026, 10, 29)):  # un día (barras por hora) y varios (tabla por día)
        text = "".join(p.extract_text() for p in pypdf.PdfReader(io.BytesIO(
            metrics_pdf("Minas", ["comercio"], since, date(2026, 10, 30), m, None))).pages)
        assert "935" not in text and "Apariciones" not in text
        assert "Entraron" in text and "179" in text  # la puerta (o lo que falta) y las visitas por zona siguen
        assert ("65" in text) == door
