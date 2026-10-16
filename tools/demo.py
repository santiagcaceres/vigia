"""Carga datos de demostración en un servidor Vigía, sin cámaras reales.

    python -m tools.demo --server https://tu-app.vercel.app --user admin --password <clave>

Crea un cliente "Demo" con una cámara y zonas, sube un video sintético (rectángulos que
recorren un local) y lo analiza en esta PC actuando como agente del cliente. Sirve para ver
el flujo completo: análisis, eventos con clip y etiquetado.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import httpx

from agent.imports import ImportJobRunner
from agent.outbox import ServerClient
from core import synthetic


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--server", default="http://localhost:8000")
    p.add_argument("--user", default="admin")
    p.add_argument("--password", required=True)
    a = p.parse_args()
    api = httpx.Client(base_url=a.server.rstrip("/"), timeout=120, follow_redirects=True)
    api.post("/api/login", json={"username": a.user, "password": a.password}).raise_for_status()

    clients = {c["name"]: c for c in api.get("/api/clients").json()}
    client = clients.get("Demo") or api.post("/api/clients", json={"name": "Demo"}).json()
    cams = [c for c in api.get("/api/cameras", params={"client_id": client["id"]}).json() if c["name"] == "Local simulado"]
    cam = cams[0] if cams else api.post("/api/cameras", json={
        "client_id": client["id"], "name": "Local simulado", "host": "demo", "detector": "motion",
        "process_fps": synthetic.FPS, "nvr_clips": False, "enabled": False}).json()
    api.put(f"/api/cameras/{cam['id']}/analysis", json={"zones": synthetic.STORE_ZONES, "rules": {}}).raise_for_status()
    agent = api.post("/api/agents", json={"client_id": client["id"], "name": "Demo (esta PC)"}).json()

    with tempfile.TemporaryDirectory() as tmp:
        frames = []
        for name in ("hurto", "cliente_paga", "intruso_deposito", "mira_rapido", "hurto", "pasa_de_largo"):
            frames += synthetic.render(name, lead_seconds=2, tail_seconds=2)
        video = synthetic.write_video(frames, Path(tmp) / "demo_local.mp4")
        imp = api.post("/api/imports", json={"camera_id": cam["id"], "filename": "demo_local.mp4",
                                             "size": video.stat().st_size, "recorded_at": "2026-10-15T18:30:00"})
        imp.raise_for_status()
        imp = imp.json()
        url = imp["upload_url"]
        put = api.put(url, content=video.read_bytes(), headers={"Content-Type": "video/mp4"}) if url.startswith("/") \
            else httpx.put(url, content=video.read_bytes(), headers={"Content-Type": "video/mp4"}, timeout=120)
        put.raise_for_status()
        api.post(f"/api/imports/{imp['id']}/uploaded").raise_for_status()
        print("Video subido; analizando en esta PC…")
        runner = ImportJobRunner(ServerClient(a.server, agent["token"]), None, Path(tmp) / "work")
        runner.run_once()
    api.delete(f"/api/agents/{agent['id']}")
    print(f"Listo: mirá {a.server.rstrip('/')}/#/videos")


if __name__ == "__main__":
    main()
