"""Escenas sintéticas para probar el sistema sin cámaras reales.

Una "persona" es un rectángulo que recorre una tienda de 640x360 con tres zonas:
góndola (izquierda), caja (centro-abajo) y salida (derecha), más un depósito arriba.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

W, H, FPS = 640, 360, 10
PERSON_W, PERSON_H = 24, 60

# Zonas en coordenadas normalizadas (mismo formato que usa la API).
STORE_ZONES = [
    {"name": "gondola", "type": "shelf", "polygon": [[0.05, 0.4], [0.35, 0.4], [0.35, 0.95], [0.05, 0.95]]},
    {"name": "caja", "type": "checkout", "polygon": [[0.42, 0.6], [0.65, 0.6], [0.65, 0.95], [0.42, 0.95]]},
    {"name": "salida", "type": "exit", "polygon": [[0.82, 0.4], [1.0, 0.4], [1.0, 1.0], [0.82, 1.0]]},
    {"name": "deposito", "type": "restricted", "polygon": [[0.4, 0.0], [0.7, 0.0], [0.7, 0.35], [0.4, 0.35]]},
]

# Cada waypoint: (x_norm, y_norm, segundos_en_llegar, segundos_quieto)
# (x, y) es la posición de los pies de la persona.
SCENARIOS: dict[str, list[tuple[float, float, float, float]]] = {
    # Toma algo de la góndola y se va por la salida sin pasar por caja -> ALERTA
    "hurto": [
        (0.50, 0.50, 0.0, 0.0),
        (0.20, 0.70, 2.0, 4.5),
        (0.50, 0.50, 2.0, 0.0),
        (0.92, 0.55, 2.5, 0.5),
    ],
    # Toma algo, paga en caja y se va -> sin alerta
    "cliente_paga": [
        (0.50, 0.50, 0.0, 0.0),
        (0.20, 0.70, 2.0, 4.5),
        (0.52, 0.78, 2.5, 2.5),
        (0.55, 0.50, 1.5, 0.0),
        (0.92, 0.55, 2.5, 0.5),
    ],
    # Mira rápido la góndola (menos del umbral) y se va -> sin alerta
    "mira_rapido": [
        (0.50, 0.50, 0.0, 0.0),
        (0.25, 0.60, 1.0, 1.0),
        (0.50, 0.50, 1.5, 0.0),
        (0.92, 0.55, 2.5, 0.5),
    ],
    # Solo cruza por el pasillo -> sin alerta
    "pasa_de_largo": [
        (0.50, 0.50, 0.0, 0.0),
        (0.92, 0.55, 3.0, 0.5),
    ],
    # Entra al depósito -> ALERTA zona restringida
    "intruso_deposito": [
        (0.50, 0.50, 0.0, 0.0),
        (0.55, 0.30, 1.5, 2.5),
        (0.50, 0.50, 1.5, 0.0),
    ],
}


def positions(name: str, fps: int = FPS) -> list[tuple[float, float]]:
    """Posición (x, y) normalizada de los pies en cada frame del escenario."""
    wps = SCENARIOS[name]
    out: list[tuple[float, float]] = []
    x, y = wps[0][0], wps[0][1]
    out.append((x, y))
    for nx, ny, travel, hold in wps[1:]:
        steps = max(1, int(round(travel * fps)))
        x0, y0 = x, y
        for i in range(1, steps + 1):
            k = i / steps
            out.append((x0 + (nx - x0) * k, y0 + (ny - y0) * k))
        x, y = nx, ny
        out.extend([(x, y)] * int(round(hold * fps)))
    return out


def _background(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.full((H, W, 3), (55, 52, 50), np.uint8)
    # algo de textura estática para que el fondo no sea perfectamente liso
    noise = rng.integers(0, 6, size=(H, W, 1), dtype=np.uint8)
    return cv2.add(base, np.repeat(noise, 3, axis=2))


def render(name: str, seed: int = 0, lead_seconds: float = 1.5, tail_seconds: float = 1.0) -> list[np.ndarray]:
    """Frames BGR del escenario: escena vacía, persona recorriendo, escena vacía."""
    rng = np.random.default_rng(seed + 1)
    bg = _background(seed)
    frames: list[np.ndarray] = []

    def noisy(img: np.ndarray) -> np.ndarray:
        n = rng.normal(0, 1.5, img.shape)
        return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)

    for _ in range(int(lead_seconds * FPS)):
        frames.append(noisy(bg))
    for nx, ny in positions(name):
        img = bg.copy()
        fx, fy = int(nx * W), int(ny * H)
        cv2.rectangle(img, (fx - PERSON_W // 2, fy - PERSON_H), (fx + PERSON_W // 2, fy), (170, 150, 120), -1)
        frames.append(noisy(img))
    for _ in range(int(tail_seconds * FPS)):
        frames.append(noisy(bg))
    return frames


def write_video(frames: list[np.ndarray], path: str | Path, fps: int = FPS) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frames[0].shape[1], frames[0].shape[0]))
    if not vw.isOpened():
        raise RuntimeError("OpenCV no pudo abrir el VideoWriter (códec mp4v)")
    for f in frames:
        vw.write(f)
    vw.release()
    return path
