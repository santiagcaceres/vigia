"""Qué se ve en la imagen de una cámara: YOLO con todas las clases del modelo (no solo personas), más luz y
color. Sirve para reconocer cuál es cada cámara al ponerle nombre y para sugerir su tipo.

Es una **pista** para la persona que la nombra, no una clasificación: el modelo (COCO) no conoce «góndola» ni «caja
registradora», así que se infiere por objetos (heladeras, botellas, frutas, pantallas, autos) y siempre se dice como
sugerencia. Solo lo usa la app (el servidor no importa este módulo).
"""

from __future__ import annotations

import threading
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .detectors import OnnxYoloDetector

# Clases COCO del modelo que tienen sentido en un comercio o en la calle (las demás, como jirafas o aviones, en una
# cámara de seguridad suelen ser errores y no se muestran).
COCO_ES = {
    0: "persona", 1: "bicicleta", 2: "auto", 3: "moto", 5: "ómnibus", 7: "camión", 9: "semáforo", 13: "banco",
    15: "gato", 16: "perro", 24: "mochila", 25: "paraguas", 26: "cartera", 28: "valija", 32: "pelota",
    39: "botella", 40: "copa", 41: "taza", 45: "bowl", 46: "banana", 47: "manzana", 49: "naranja", 50: "brócoli",
    51: "zanahoria", 56: "silla", 57: "sillón", 58: "planta", 60: "mesa", 62: "pantalla", 63: "notebook",
    64: "mouse", 66: "teclado", 67: "celular", 68: "microondas", 69: "horno", 71: "pileta", 72: "heladera",
    73: "libro", 74: "reloj", 75: "florero",
}
OBJECT_CONF = 0.45  # los objetos (no personas) se muestran solo con confianza alta
VEHICLES = ("auto", "moto", "ómnibus", "camión", "bicicleta")
PRODUCE = ("banana", "manzana", "naranja", "brócoli", "zanahoria")
DRINKS = ("botella", "copa", "taza")
SCREENS = ("pantalla", "notebook", "teclado", "mouse", "celular")

_lock = threading.Lock()
_detectors: dict[str, OnnxYoloDetector] = {}


def scene_detector(model_path: str | Path) -> OnnxYoloDetector:
    """Un detector con todas las clases de `COCO_ES` (sesión propia, de a un cuadro; se arma una vez)."""
    key = str(model_path)
    with _lock:
        if key not in _detectors:
            _detectors[key] = OnnxYoloDetector(model_path, conf=0.35, classes=COCO_ES, max_batch=1)
        return _detectors[key]


def describe(frame: np.ndarray, detector, plans: tuple[str, ...] | list[str] = ()) -> dict:
    """Personas, objetos, tamaño, si la imagen está oscura o en blanco y negro (infrarrojo, de noche) y una sugerencia
    de tipo y nombre (o None)."""
    dets = detector.detect(frame)
    persons = sum(1 for d in dets if d.label == "persona")
    objects = Counter(d.label for d in dets if d.label != "persona" and d.conf >= OBJECT_CONF)
    h, w = frame.shape[:2]
    dark, night = light(frame)
    return {"persons": persons, "objects": dict(objects.most_common()), "size": (w, h),
            "dark": dark, "night": night, "suggestion": suggest(objects, plans)}


def light(frame: np.ndarray) -> tuple[bool, bool]:
    """(imagen oscura, de noche en blanco y negro por el infrarrojo): con poca luz el detector ve menos."""
    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    gray = float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean())
    sat = float(cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 1].mean())
    return gray < 50, sat < 12


def suggest(objects: dict, plans: tuple[str, ...] | list[str] = ()) -> dict | None:
    """Tipo de cámara (perfil) y nombre sugeridos a partir de los objetos, con el porqué."""
    def n(*labels: str) -> int:
        return sum(objects.get(x, 0) for x in labels)

    if n(*VEHICLES):
        if "estacion" in plans:
            return {"profile": "surtidor", "name": "Surtidores", "why": "se ven vehículos"}
        return {"profile": "exterior", "name": "Calle", "why": "se ven vehículos"}
    if n(*PRODUCE) >= 2:
        return {"profile": "gondola", "name": "Verdulería", "why": "se ven frutas o verduras"}
    if n("heladera"):
        return {"profile": "gondola", "name": "Heladeras", "why": "se ve una heladera"}
    if n(*DRINKS) >= 3:
        return {"profile": "gondola", "name": "Bebidas", "why": "se ven varias botellas"}
    if n(*SCREENS):
        return {"profile": "caja", "name": "Caja", "why": "se ve una pantalla, teclado o celular (¿la caja?)"}
    if n("libro") >= 3:
        return {"profile": "gondola", "name": "Libros y revistas", "why": "se ven varios libros"}
    return None


def summary(info: dict) -> str:
    """«2 personas · heladera, 5 botellas · de noche (infrarrojo) · 640×360»"""
    parts = []
    p = info.get("persons") or 0
    parts.append(f"{p} persona{'s' if p != 1 else ''}" if p else "nadie")
    objs = [f"{c} {k}" if c > 1 else k for k, c in (info.get("objects") or {}).items()][:5]
    if objs:
        parts.append(", ".join(objs))
    if info.get("night"):
        parts.append("de noche (infrarrojo)")
    elif info.get("dark"):
        parts.append("imagen oscura")
    w, h = info.get("size") or (0, 0)
    if w:
        parts.append(f"{w}×{h}")
    return " · ".join(parts)
