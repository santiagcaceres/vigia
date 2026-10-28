"""Mapa de pisadas y zonas sugeridas: dónde se para y camina la gente, para dibujar las zonas sobre datos."""

import numpy as np

from core.pisadas import GRID, FootMap, overlay, suggest
from core.tracker import Track

W, H = 1920, 1080


def person(tid, fx, fy, ts):
    x, y = fx * W, fy * H  # pies
    return Track(tid, (x - 40, y - 300, x + 40, y), "persona", 0.0, ts, hits=10)


def store(n_people=6, fps=5):
    """Cada persona entra por abajo, camina hasta la góndola de la derecha (0.75, 0.55), se queda 20 s y se va
    por donde vino."""
    frames, ts = [], 0.0
    for k in range(n_people):
        path = [(0.5, 0.98 - 0.4 * i / 20) for i in range(20)]  # sube por el pasillo
        path += [(0.5 + 0.25 * i / 10, 0.58 - 0.03 * i / 10) for i in range(10)]  # va a la góndola
        path += [(0.75 + 0.01 * np.sin(i), 0.55) for i in range(20 * fps)]  # se queda
        path += list(reversed(path[:30]))  # vuelve y se va
        for fx, fy in path:
            frames.append((ts, (W, H), [person(k + 1, fx, fy, ts)]))
            ts += 1 / fps
        frames.append((ts, (W, H), []))
        ts += 1
    return frames


def test_mapa_separa_donde_caminan_de_donde_se_paran():
    fm = FootMap.build(store())
    assert fm.persons == 6 and fm.seconds > 100
    gw, gh = GRID
    still_cell = np.unravel_index(fm.still.argmax(), fm.still.shape)
    assert abs(still_cell[1] / gw - 0.75) < 0.03 and abs(still_cell[0] / gh - 0.55) < 0.03
    assert fm.walk[int(0.8 * gh), int(0.5 * gw)] > 0 and fm.still[int(0.8 * gh), int(0.5 * gw)] == 0
    img = np.full((360, 640, 3), 128, np.uint8)
    out = overlay(img, fm)
    assert out.shape == img.shape and (out != img).any()


def test_sugiere_gondola_pasillo_y_puerta_con_sentido():
    zones = suggest(FootMap.build(store()))
    by_type = {z["type"]: z for z in zones}
    assert set(by_type) == {"shelf", "aisle", "entrance"}
    g = np.array(by_type["shelf"]["polygon"])
    assert g[:, 0].min() < 0.75 < g[:, 0].max() and g[:, 1].min() < 0.55 < g[:, 1].max()
    door = by_type["entrance"]
    assert door["name"] == "Puerta abajo" and door["direction"] == "arriba"  # aparecen abajo y suben
    # lo que ya está marcado no se vuelve a sugerir
    again = suggest(FootMap.build(store()), existing=[by_type["shelf"]])
    assert "shelf" not in {z["type"] for z in again}
    assert suggest(FootMap.empty()) == []


def test_el_pasillo_viaja_aparte_y_no_da_alertas():
    from core.rules import RuleEngine, parse_config
    an = {"zones": [{"name": "gondola", "type": "shelf", "polygon": [[0.6, 0.4], [0.9, 0.4], [0.9, 0.7], [0.6, 0.7]]}],
          "extra_zones": [{"name": "pasillo", "type": "aisle", "polygon": [[0.4, 0], [0.6, 0], [0.6, 1], [0.4, 1]]},
                          {"name": "mostrador", "type": "staff", "polygon": [[0, 0.8], [0.2, 0.8], [0.2, 1], [0, 1]]}],
          "staff_zones": [{"name": "mostrador", "type": "staff", "polygon": [[0, 0.8], [0.2, 0.8], [0.2, 1], [0, 1]]}],
          "rules": {"loiter_seconds": 5}}
    zones, cfg = parse_config(an)
    assert [z.type for z in zones] == ["shelf", "aisle", "staff"]
    engine = RuleEngine(zones, cfg)
    events = []
    for i in range(200):  # 40 s parado en el pasillo
        events += engine.update([person(1, 0.5, 0.5, i / 5)], i / 5, (W, H))
    assert events == []


def test_editor_agarra_puntos_y_mueve_zonas():
    from agent.oficina import moved_polygon, nearest_vertex, zone_at
    zones = [{"polygon": [[0.1, 0.1], [0.4, 0.1], [0.4, 0.4], [0.1, 0.4]]},
             {"polygon": [[0.3, 0.3], [0.6, 0.3], [0.6, 0.6], [0.3, 0.6]]}]
    assert nearest_vertex(zones, 0.402, 0.101, 880, 495, 9) == (0, 1)
    assert nearest_vertex(zones, 0.25, 0.25, 880, 495, 9) is None
    assert zone_at(zones, 0.35, 0.35) == 1  # la de más arriba
    assert zone_at(zones, 0.2, 0.2) == 0 and zone_at(zones, 0.9, 0.9) is None
    moved = moved_polygon(zones[1]["polygon"], 0.6, -0.1)  # no se sale de la imagen
    assert max(x for x, _ in moved) == 1.0 and min(y for _, y in moved) == 0.2
