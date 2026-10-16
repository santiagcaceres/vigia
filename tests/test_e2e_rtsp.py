"""Prueba de punta a punta con procesos reales:

  ffmpeg  ->  mediamtx (simula un NVR Hikvision: rutas /Streaming/Channels/302 y /Streaming/tracks/301,
                        con usuario y clave)
          ->  agente (python -m agent)  ->  servidor (uvicorn)  ->  evento con foto y clip del "NVR"

Se saltea si no está el binario de mediamtx (tools/mediamtx o env MEDIAMTX).
"""

import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from core import synthetic

ROOT = Path(__file__).resolve().parents[1]
MEDIAMTX = os.getenv("MEDIAMTX") or str(ROOT.parent / "tools" / "mediamtx")
pytestmark = pytest.mark.skipif(not (Path(MEDIAMTX).exists() and shutil.which("ffmpeg")),
                                reason="mediamtx o ffmpeg no disponibles")

PASSWORD = "Cl@ve#1"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_port(port: int, timeout: float = 20) -> None:
    end = time.time() + timeout
    while time.time() < end:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.2)
    raise TimeoutError(f"puerto {port} no abrió")


@pytest.fixture
def procs():
    started = []
    yield started
    for p in started:
        p.terminate()
    for p in started:
        try:
            p.wait(10)
        except subprocess.TimeoutExpired:
            p.kill()


def test_agente_detecta_en_rtsp_hikvision_y_sube_clip_del_nvr(tmp_path, procs):
    rtsp_port, api_port = free_port(), free_port()
    (tmp_path / "mediamtx.yml").write_text(f"""
logLevel: warn
rtspAddress: :{rtsp_port}
protocols: [tcp]
rtmp: no
hls: no
webrtc: no
srt: no
api: no
metrics: no
playback: no
authInternalUsers:
  - user: any
    ips: ['127.0.0.1', '::1']
    permissions:
      - action: publish
  - user: admin
    pass: "{PASSWORD}"
    permissions:
      - action: read
paths:
  all_others:
""")
    log = open(tmp_path / "mediamtx.log", "w")
    procs.append(subprocess.Popen([MEDIAMTX, str(tmp_path / "mediamtx.yml")], stdout=log, stderr=log))
    wait_port(rtsp_port)

    env = {**os.environ, "DATA_DIR": str(tmp_path / "server"), "ADMIN_PASSWORD": "clave-admin-123",
           "PYTHONPATH": str(ROOT)}
    if os.getenv("E2E_DATABASE_URL"):  # probar contra PostgreSQL, como en Supabase
        env["DATABASE_URL"] = os.environ["E2E_DATABASE_URL"]
    slog = open(tmp_path / "server.log", "w")
    procs.append(subprocess.Popen([sys.executable, "-m", "uvicorn", "server.main:create_app", "--factory",
                                   "--port", str(api_port)], cwd=ROOT, env=env, stdout=slog, stderr=slog))
    wait_port(api_port)
    api = httpx.Client(base_url=f"http://127.0.0.1:{api_port}", timeout=30, follow_redirects=True)
    assert api.post("/api/login", json={"username": "admin", "password": "clave-admin-123"}).status_code == 200
    cid = api.post("/api/clients", json={"name": "Farmacia Sur"}).json()["id"]
    agent = api.post("/api/agents", json={"client_id": cid, "name": "Mini PC"}).json()
    cam = api.post("/api/cameras", json={
        "client_id": cid, "name": "Salón principal", "host": "127.0.0.1", "rtsp_port": rtsp_port,
        "username": "admin", "password": PASSWORD, "channel": 3, "detector": "motion",
        "process_fps": 5, "nvr_clips": True}).json()
    r = api.put(f"/api/cameras/{cam['id']}/analysis", json={"zones": synthetic.STORE_ZONES, "rules": {}})
    assert r.status_code == 200

    alog = open(tmp_path / "agent.log", "w")
    procs.append(subprocess.Popen(
        [sys.executable, "-m", "agent", "--server", f"http://127.0.0.1:{api_port}", "--token", agent["token"],
         "--data", str(tmp_path / "agent"), "--model", str(ROOT / "models" / "personas.onnx")],
        cwd=ROOT, env={**env, "VIGIA_NVR_DELAY": "2"}, stdout=alog, stderr=alog))

    # Escena: 8 s de local vacío + persona que roba + local vacío, en bucle. Se publica recién al
    # lanzar el agente, para que se conecte con el local vacío (lo necesita el detector de movimiento).
    frames = synthetic.render("hurto", lead_seconds=8.0, tail_seconds=3.0)
    video = synthetic.write_video(frames, tmp_path / "escena.mp4")
    for path in ("Streaming/Channels/302", "Streaming/tracks/301"):
        procs.append(subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-re", "-stream_loop", "-1", "-i", str(video), "-c:v", "libx264",
             "-preset", "ultrafast", "-g", "10", "-f", "rtsp", "-rtsp_transport", "tcp",
             f"rtsp://127.0.0.1:{rtsp_port}/{path}"], stdout=subprocess.DEVNULL, stderr=log))

    ev = None
    end = time.time() + 150
    while time.time() < end:
        items = api.get("/api/events").json()["items"]
        with_clip = [e for e in items if e["clip_url"]]
        if with_clip:
            ev = with_clip[0]
            break
        time.sleep(2)
    agent_log = (tmp_path / "agent.log").read_text()
    assert ev is not None, "No llegó ningún evento con clip.\n" + agent_log[-3000:]
    assert ev["type"] == "salida_sin_pago" and ev["camera_name"] == "Salón principal"
    assert ev["clip_source"] == "nvr", agent_log[-3000:]
    clip = api.get(ev["clip_url"])
    assert clip.status_code == 200 and len(clip.content) > 5000
    (tmp_path / "clip.mp4").write_bytes(clip.content)
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=codec_name:format=duration", "-of", "default=nw=1", str(tmp_path / "clip.mp4")],
                           capture_output=True, text=True).stdout
    assert "codec_name=h264" in probe
    dur = float([l for l in probe.splitlines() if l.startswith("duration=")][0].split("=")[1])
    assert 8 <= dur <= 14, probe
    assert api.get(ev["snapshot_url"]).headers["content-type"] == "image/jpeg"

    # el agente reporta estado y manda la vista previa para dibujar zonas
    end = time.time() + 30
    cam_state = None
    while time.time() < end:
        cam_state = api.get(f"/api/cameras/{cam['id']}").json()
        if cam_state["has_preview"] and cam_state["status"]:
            break
        time.sleep(1)
    assert cam_state["has_preview"] and cam_state["status"]["state"] == "en_linea", cam_state
    from urllib.parse import quote

    for secret in (PASSWORD, quote(PASSWORD, safe="")):  # la clave no aparece en los logs, ni codificada
        assert secret not in agent_log
