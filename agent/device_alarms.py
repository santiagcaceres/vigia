"""Escucha de alarmas del propio grabador (modo "alarmas_equipo").

Un hilo por cámara se queda conectado al grabador (Dahua o Hikvision) y, por cada alarma
de video que avisa, la registra en el servidor (`POST /api/agent/alarms`). De ahí en más
es el mismo circuito que las alarmas de Hik-Partner Pro: el servidor agrupa los disparos
seguidos, espera a que el equipo termine de grabar el "después" y `AlarmRunner` baja el
tramo, lo analiza y descarta las que no tienen personas.

No hace falta nada en la nube ni instalar nada en el local: solo llegar al grabador (VPN).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Iterator

import httpx

from .cameras import BaseCamera, build_camera
from .device_events import DeviceEvent, device_events, matches_channel, now_iso
from .outbox import ServerClient

log = logging.getLogger("vigia.alarmas_equipo")

# El grabador suele repetir el mismo evento varias veces por segundo mientras dura el movimiento;
# con avisar una vez cada tanto alcanza (el servidor igual agrupa los disparos seguidos).
REPEAT_SECONDS = 8.0
EventSource = Callable[[BaseCamera], Iterator[DeviceEvent]]


class DeviceAlarmListener:
    def __init__(self, cam_cfg: dict, client: ServerClient, source: EventSource = device_events):
        self.id: int = cam_cfg["id"]
        self.name: str = (f"{cam_cfg['client_name']} · {cam_cfg['name']}" if cam_cfg.get("client_name")
                          else cam_cfg["name"])
        self.fingerprint = self.fingerprint_of(cam_cfg)
        self.cam = build_camera(cam_cfg["connection"])
        self.event_types = set(cam_cfg.get("alarm_cfg", {}).get("event_types") or ())
        self.client = client
        self.source = source
        self._stop = threading.Event()
        self._last: dict[str, float] = {}
        self.thread = threading.Thread(target=self.run, name=f"alarmas-cam-{self.id}", daemon=True)
        self.state = "iniciando"
        self.error: str | None = None
        self.events = 0
        self.last_event_at: float | None = None

    @staticmethod
    def fingerprint_of(cam_cfg: dict) -> str:
        import json
        return json.dumps(cam_cfg, sort_keys=True, default=str)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self._stop.set()

    def status(self) -> dict:
        return {"camera_id": self.id, "state": self.state, "error": self.error, "events": self.events,
                "last_event_age_s": None if self.last_event_at is None else round(time.time() - self.last_event_at, 1)}

    def run(self) -> None:
        backoff = 2.0
        while not self._stop.is_set():
            try:
                self.state, self.error = "escuchando", None
                log.info("[%s] escuchando alarmas del grabador", self.name)
                for ev in self.source(self.cam):
                    if self._stop.is_set():
                        break
                    backoff = 2.0
                    self.handle(ev)
                if self._stop.is_set():
                    break
                self.state, self.error = "reconectando", "El grabador cerró la conexión"
            except httpx.HTTPStatusError as exc:
                self.state = "sin_conexion"
                self.error = (f"El grabador respondió {exc.response.status_code} "
                              f"(¿usuario y clave? ¿la API HTTP está habilitada?)")
            except Exception as exc:
                self.state, self.error = "sin_conexion", f"{type(exc).__name__}: {exc}"
            if not self._stop.is_set():
                log.warning("[%s] %s; reintento en %.0fs", self.name, self.error, backoff)
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 120.0)
        self.state = "detenida"

    def handle(self, ev: DeviceEvent) -> bool:
        """Manda la alarma al servidor si corresponde. Devuelve True si la mandó."""
        if not matches_channel(ev, self.cam.channel) or ev.event_type not in self.event_types:
            return False
        now = time.monotonic()
        if now - self._last.get(ev.event_type, -1e9) < REPEAT_SECONDS:
            return False
        self._last[ev.event_type] = now
        self.events += 1
        self.last_event_at = time.time()
        try:
            self.client.post_device_alarm(self.id, ev.event_type, now_iso(self.cam.tz), ev.target, ev.raw)
        except httpx.HTTPError as exc:
            log.warning("[%s] no se pudo registrar la alarma %s: %s", self.name, ev.event_type, exc)
            return False
        log.info("[%s] alarma %s registrada", self.name, ev.event_type)
        return True
