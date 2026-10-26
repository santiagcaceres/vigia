"""«En vivo» de la app de escritorio: mirar cámaras sin analizarlas.

- `ViewWorker`: la cámara no se analiza. Abre el video (igual que el análisis: RTSP o Hik-Connect) y deja el
  último cuadro para la ventana. No usa el detector, no guarda nada ni sube nada.
- `RemoteView`: la cámara la analiza una máquina de análisis. No se le pide otra transmisión al grabador (por
  Hik-Connect da pocas a la vez): se muestra la imagen que sube la máquina, 1 por segundo y con los recuadros de
  la IA, con el mismo latido que usa el panel web (`POST /api/cameras/{id}/live`).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

import cv2
import httpx
import numpy as np

from .cameras import build_camera

log = logging.getLogger("vigia.vista")
SHOW_INTERVAL = 0.1  # la ventana no necesita más de 10 cuadros por segundo
BEAT_SECONDS = 10.0  # latido «alguien mira»: la máquina sube cuadros mientras lleguen


def _name(cfg: dict) -> str:
    return f"{cfg['client_name']} · {cfg['name']}" if cfg.get("client_name") else cfg["name"]


class ViewWorker:
    """Mirar una cámara que no se analiza."""

    remote = False

    def __init__(self, cfg: dict):
        self.id: int = cfg["id"]
        self.name = _name(cfg)
        self.hik = build_camera(cfg["connection"])
        self.state = "conectando"
        self.error: str | None = None
        self.fps = 0.0
        self.display: np.ndarray | None = None
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self.run, name=f"ver-{self.id}", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self._stop.set()

    def text(self) -> str:
        if self.state == "en_linea":
            return f"Viendo · {self.fps:.0f} fps"
        if self.state == "sin_conexion":
            return "Sin conexión" + (f": {self.error}" if self.error else "")
        return "Conectando..."

    def run(self) -> None:
        backoff = 2.0
        while not self._stop.is_set():
            try:
                url = self.hik.live_url()  # los links de Hik-Connect vencen: uno nuevo en cada conexión
            except Exception as exc:  # noqa: BLE001 - sin link no hay video; se reintenta
                self.state, self.error = "sin_conexion", str(exc)
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 60.0)
                continue
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                why = self.hik.open_failure(url)
                self.state, self.error = "sin_conexion", why or "no se pudo abrir el video"
                log.warning("[%s] Para mirar: %s", self.name, self.error)
                cap.release()
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 60.0)
                continue
            self.state, self.error, backoff = "en_linea", None, 2.0
            try:
                self._read(cap)
            finally:
                cap.release()
            if not self._stop.is_set():
                self.state = "conectando"
                self._stop.wait(2.0)
        self.state = "detenida"

    def _read(self, cap: cv2.VideoCapture) -> None:
        last_show = -1e9
        shown: list[float] = []
        fails = 0
        while not self._stop.is_set():
            if not cap.grab():  # tomar todos los cuadros (no acumular retraso); convertir solo los que se muestran
                fails += 1
                if fails >= 3:
                    return
                continue
            fails = 0
            now = time.monotonic()
            if now - last_show < SHOW_INTERVAL:
                continue
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                continue
            last_show = now
            self.display = frame
            shown = [t for t in shown if now - t < 5] + [now]
            self.fps = len(shown) / 5.0


class RemoteView:
    """Mirar una cámara que analiza una máquina: la imagen que sube la máquina (1 por segundo, con recuadros)."""

    remote = True

    def __init__(self, cfg: dict, beat: Callable[[int], dict], server: str):
        self.id: int = cfg["id"]
        self.name = _name(cfg)
        self.analyzer = cfg.get("analyzer")
        self.beat = beat
        self.server = server.rstrip("/")
        self.state = "esperando"
        self.error: str | None = None
        self.display: np.ndarray | None = None
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self.run, name=f"ver-maquina-{self.id}", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self._stop.set()

    def text(self) -> str:
        if self.state == "en_linea":
            return "Imagen de la máquina (1 por segundo, con recuadros)"
        if self.state == "maquina_apagada":
            return "La máquina que la analiza no está en línea"
        if self.state == "sin_conexion":
            return "Sin conexión con el servidor" + (f": {self.error}" if self.error else "")
        return "Esperando la imagen de la máquina..."

    def run(self) -> None:
        url, last_beat = None, -1e9
        with httpx.Client(timeout=10.0, follow_redirects=True) as http:
            while not self._stop.is_set():
                now = time.monotonic()
                if now - last_beat >= BEAT_SECONDS:
                    last_beat = now
                    try:
                        info = self.beat(self.id)
                        url = info.get("frame_url")
                        if not info.get("agent_online"):
                            self.state = "maquina_apagada"
                    except Exception as exc:  # noqa: BLE001 - se reintenta en el próximo latido
                        self.state, self.error = "sin_conexion", str(exc)
                if url and self.state != "maquina_apagada":
                    full = url if url.startswith("http") else self.server + url
                    try:
                        r = http.get(full + ("&" if "?" in full else "?") + f"t={int(time.time() * 1000)}")
                        img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR) \
                            if r.status_code == 200 else None
                        if img is not None:
                            self.display, self.state = img, "en_linea"
                    except httpx.HTTPError as exc:
                        self.error = str(exc)
                self._stop.wait(1.0)
        self.state = "detenida"
