"""El merodeo ya no existe (29 oct, decisión del dueño): ni alerta, ni parámetro, ni métrica. Lo guardado de antes y
lo que mandan apps y máquinas sin actualizar se limpia; las alertas viejas de merodeo no se listan ni se cuentan."""

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from core.rules import RULE_TYPES, RuleConfig
from server.db import Camera, Event
from server.main import Settings, create_app

ADMIN = ("admin", "clave-admin-123")


@pytest.fixture
def app(tmp_path, db_url):
    return create_app(Settings(data_dir=tmp_path / "data", database_url=db_url, admin_password=ADMIN[1],
                               serve_public=False))


def test_el_merodeo_no_existe_mas(app):
    assert "merodeo" not in RULE_TYPES and not hasattr(RuleConfig(), "loiter_seconds")
    with TestClient(app) as a:
        a.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]})
        cid = a.post("/api/clients", json={"name": "Minas"}).json()["id"]
        new = lambda name: a.post("/api/cameras", json={"client_id": cid, "name": name, "brand": "ip",  # noqa: E731
                                                         "host": "dvr", "password": "x"}).json()["id"]
        vieja, general = new("Pasillo bebidas"), new("Góndolas ventana")
        with app.state.Session() as s:  # guardada antes, con merodeo
            s.get(Camera, vieja).rules = {"loiter_seconds": 45, "enabled_rules": ["merodeo", "salida_sin_pago"],
                                          "cooldown_seconds": {"merodeo": 30, "salida_sin_pago": 9}}
            s.commit()
        h = {"X-Agent-Token": a.post("/api/agent/session").json()["token"]}
        tune = lambda cam: a.post("/api/agent/tune", headers=h, json={"camera_id": cam}).json()  # noqa: E731
        info = tune(vieja)
        rules = info["camera"]["analysis"]["rules"]
        assert rules == {"enabled_rules": ["salida_sin_pago"], "cooldown_seconds": {"salida_sin_pago": 9}}
        assert info["raw"]["rules"] == rules
        assert a.get(f"/api/cameras/{vieja}").json()["rules"] == rules
        # sin elegir a mano, la lista va explícita: una máquina sin actualizar, sin la lista, miraría merodeo
        assert tune(general)["camera"]["analysis"]["rules"]["enabled_rules"] == list(RULE_TYPES)
        # una app vieja que todavía manda el tiempo de merodeo guarda igual (y se descarta)
        r = a.put(f"/api/cameras/{general}/analysis", json={"zones": [], "rules": {"loiter_seconds": 30,
                                                                                 "enabled_rules": ["merodeo"]}})
        assert r.status_code == 200 and r.json()["rules"] == {"enabled_rules": []}

        # las alertas viejas de merodeo no se listan ni cuentan como métrica
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        with app.state.Session() as s:
            for i, t in enumerate(("merodeo", "salida_sin_pago")):
                s.add(Event(uuid=f"e{i}", client_id=cid, camera_id=vieja, type=t, severity="media", description="x",
                            occurred_at=now, alert=True, source="vivo"))
            s.commit()
        assert [e["type"] for e in a.get("/api/events").json()["items"]] == ["salida_sin_pago"]
        assert a.get("/api/stats").json()["by_type"] == {"salida_sin_pago": 1}
        assert a.get("/api/stats/alerts").json()["clients"][str(cid)]["today"] == 1
        assert a.get("/api/events?type=merodeo").json()["total"] == 1  # si se pide a propósito, están
