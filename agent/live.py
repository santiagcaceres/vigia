"""Vista en vivo "liviana": mientras alguien mira una cámara en el panel, la PC central sube
un cuadro por segundo con las personas recuadradas, su número de seguimiento, los segundos
que llevan en escena y la zona en la que están paradas (y, si la cámara mira gestos de
ocultamiento, el esqueleto y el gesto en curso).

Por qué así y no video: el servidor (Vercel) no puede retransmitir video. Los cuadros van
directo al almacenamiento (URL firmada) y el navegador los pide 1 vez por segundo. Cuesta
~30 KB por cuadro, solo mientras hay alguien mirando (el panel manda un latido cada 10 s y la
PC central deja de subir 45 s después del último).

- `LiveStreamer`: lo usa el `CameraWorker` (cámaras en modo vivo): recibe cada cuadro ya
  analizado y sube como máximo uno por segundo.
- `LiveWorker`: para cámaras en modo alarmas (la PC central no las mira en vivo): abre el
  stream solo mientras dure el vivo, detecta y sigue personas, y lo cierra al terminar.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime

import cv2
import numpy as np

from core.analyzer import annotate
from core.detectors import Detector
from core.geometry import bbox_foot
from core.rules import Zone, parse_zones
from core.tracker import CentroidTracker, Track

from .cameras import build_camera
from .outbox import ServerClient

log = logging.getLogger("vigia.live")

FRAME_EVERY_SECONDS = 1.0
MAX_WIDTH = 640
JPEG_QUALITY = 65
LIVE_PROCESS_FPS = 3.0


def gesture_text(gestures, tid: int, ts: float) -> tuple[str, bool]:
    """El gesto en curso de una persona (core/gestos.GestureEngine.progress), corto y sin tildes (OpenCV no las
    dibuja): «mano der bolsillo 0.4/0.6s  tomo algo hace 2s». El segundo valor dice si alguna mano ya llegó al
    tiempo que pide el gesto (se escribe en rojo: posible ocultamiento, para revisar)."""
    pr = gestures.progress(tid, ts) if gestures is not None else None
    if not pr:
        return "", False
    parts = [f"mano {side} {region} {dwell:.1f}/{need:.1f}s"
             for side, (region, dwell, need) in sorted(pr["hands"].items())]
    if pr.get("took_ago") is not None:
        parts.append(f"tomo algo hace {pr['took_ago']:.0f}s")
    hot = any(dwell >= need for region, dwell, need in pr["hands"].values())
    return "  ".join(parts), hot


def hud(frame: np.ndarray, zones: list[Zone], tracks: list[Track], ts: float, name: str,
        wall: datetime | None = None, gestures=None) -> np.ndarray:
    """Cuadro con zonas, recuadros y, por persona, segundos en escena y zona donde está parada. Con `gestures` (el
    motor de gestos de la cámara, si los mira), debajo va el gesto en curso de cada persona."""
    img = annotate(frame, zones, tracks)
    h, w = img.shape[:2]
    thick = max(1, w // 640)
    bar = 22 * thick
    visibles = [t for t in tracks if t.visible]
    for t in visibles:
        fx, fy = bbox_foot(t.bbox)
        zona = next((z.name for z in zones if z.contains(fx / w, fy / h)), None)
        texto = f"#{t.id} {ts - t.first_ts:.0f}s" + (f" · {zona}" if zona else "")
        x1, y1, x2, y2 = (int(v) for v in t.bbox)
        # debajo del recuadro; si no entra (tapado por la barra), adentro del recuadro
        y = y2 + 14 * thick if y2 + 16 * thick < h - bar else y2 - 6 * thick
        cv2.putText(img, texto, (x1 + 2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * thick, (0, 255, 0), thick, cv2.LINE_AA)
        gesto, hot = gesture_text(gestures, t.id, ts)
        if gesto:
            y = y + 14 * thick if y + 16 * thick < h - bar else y - 14 * thick
            cv2.putText(img, gesto, (x1 + 2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * thick,
                        (0, 0, 255) if hot else (0, 165, 255), thick, cv2.LINE_AA)
    cv2.rectangle(img, (0, h - bar), (w, h), (0, 0, 0), -1)
    hora = (wall or datetime.now()).strftime("%H:%M:%S")
    cv2.putText(img, f"EN VIVO {hora}  {name}  personas: {len(visibles)}", (6, h - int(bar * 0.3)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5 * thick, (255, 255, 255), thick, cv2.LINE_AA)
    return img


def encode(img: np.ndarray) -> bytes | None:
    h, w = img.shape[:2]
    if w > MAX_WIDTH:
        img = cv2.resize(img, (MAX_WIDTH, int(h * MAX_WIDTH / w)), interpolation=cv2.INTER_AREA)
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return jpg.tobytes() if ok else None


class LiveStreamer:
    """Sube cuadros de una cámara mientras `until` (reloj monotónico) esté en el futuro."""

    def __init__(self, camera_id: int, name: str, client: ServerClient | None, zones: list[Zone] | None = None):
        self.id = camera_id
        self.name = name
        self.client = client
        self.zones = zones or []
        self.until = 0.0
        self.frames = 0
        self.errors = 0
        self._last = -1e9
        self._busy = threading.Lock()
        self._url: str | None = None

    @property
    def active(self) -> bool:
        return time.monotonic() < self.until

    def extend(self, seconds: float) -> None:
        self.until = max(self.until, time.monotonic() + seconds)

    def push(self, frame: np.ndarray, tracks: list[Track], ts: float, wall: datetime | None = None,
             gestures=None) -> bool:
        """Devuelve True si el cuadro se mandó (como máximo uno por segundo, sin encolar). `gestures`: ver `hud`."""
        now = time.monotonic()
        if not self.active or now - self._last < FRAME_EVERY_SECONDS or self.client is None:
            return False
        if not self._busy.acquire(blocking=False):
            return False  # todavía se está subiendo el anterior: este se salta
        self._last = now
        img = hud(frame, self.zones, tracks, ts, self.name, wall, gestures)
        threading.Thread(target=self._send, args=(img,), name=f"live-{self.id}", daemon=True).start()
        return True

    def _send(self, img: np.ndarray) -> None:
        try:
            data = encode(img)
            if data is None:
                return
            for intento in range(2):
                if self._url is None:
                    self._url = self.client.live_upload_url(self.id)
                try:
                    self.client.upload_bytes(self._url, data, "image/jpeg")
                    self.frames += 1
                    return
                except Exception as exc:  # URL vencida o rechazada: pedir otra una sola vez
                    self._url = None
                    if intento:
                        raise exc
        except Exception as exc:
            self.errors += 1
            if self.errors <= 3 or self.errors % 50 == 0:
                log.warning("[%s] no se pudo subir el cuadro en vivo: %s", self.name, exc)
        finally:
            self._busy.release()


class LiveWorker:
    """Cámara en modo alarmas: abre el stream solo mientras alguien mira, detecta y sigue
    personas y sube los cuadros. Se cierra solo cuando vence el vivo."""

    def __init__(self, cam_cfg: dict, detector: Detector, client: ServerClient):
        self.id: int = cam_cfg["id"]
        self.name: str = (f"{cam_cfg['client_name']} · {cam_cfg['name']}" if cam_cfg.get("client_name")
                          else cam_cfg["name"])
        self.cam = build_camera(cam_cfg["connection"])
        self.detector = detector
        self.streamer = LiveStreamer(self.id, self.name, client, parse_zones((cam_cfg.get("analysis") or {}).get("zones")))
        self.state = "iniciando"
        self.error: str | None = None
        self.show_window = False
        self.display: np.ndarray | None = None
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self.run, name=f"live-cam-{self.id}", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self._stop.set()

    def extend(self, seconds: float) -> None:
        self.streamer.extend(seconds)

    @property
    def active(self) -> bool:
        return self.streamer.active and self.thread.is_alive()

    def run(self) -> None:
        tracker = CentroidTracker()
        t0 = time.monotonic()
        last_proc = -1e9
        while not self._stop.is_set() and self.streamer.active:
            try:
                url = self.cam.live_url()  # en cada conexión: los links de Hik-Connect vencen
            except Exception as exc:  # noqa: BLE001
                self.state, self.error = "sin_conexion", str(exc)
                log.warning("[%s] vivo: %s", self.name, self.error)
                self._stop.wait(5.0)
                continue
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                self.state, self.error = "sin_conexion", f"No se pudo abrir {self.cam.redacted(url)}"
                log.warning("[%s] vivo: %s", self.name, self.error)
                cap.release()
                self._stop.wait(5.0)
                continue
            self.state, self.error = "en_linea", None
            log.info("[%s] vivo: conectada", self.name)
            try:
                fails = 0
                while not self._stop.is_set() and self.streamer.active:
                    ok, frame = cap.read()
                    if not ok:
                        fails += 1
                        if fails >= 3:
                            self.error = "Se perdió la señal"
                            break
                        continue
                    fails = 0
                    now = time.monotonic()
                    if now - last_proc < 1.0 / LIVE_PROCESS_FPS - 1e-3:
                        continue
                    last_proc = now
                    ts = now - t0
                    tracks = tracker.update(self.detector.detect(frame), ts, frame)
                    wall = datetime.now(self.cam.tz)
                    self.streamer.push(frame, tracks, ts, wall)
                    if self.show_window:
                        self.display = hud(frame, self.streamer.zones, tracks, ts, self.name, wall)
            except Exception as exc:
                log.exception("[%s] vivo: error", self.name)
                self.state, self.error = "error", str(exc)
                self._stop.wait(3.0)
            finally:
                cap.release()
        self.state = "detenida"
        log.info("[%s] vivo: terminado (%d cuadros)", self.name, self.streamer.frames)
