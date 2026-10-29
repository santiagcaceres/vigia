"""Modelo de filtro de alertas.

Aprende de las etiquetas de la persona de entrenamiento qué eventos valen una alerta
(hurto / sospechoso) y cuáles no (normal / falso positivo). Es una regresión logística
guardada como JSON: el agente la evalúa con numpy, sin scikit-learn ni archivos pickle.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from .gestos import GESTURE_TYPES
from .rules import EVENT_TYPES

log = logging.getLogger("vigia.scorer")

FEATURE_NAMES = (
    "track_age_s", "shelf_total_s", "checkout_total_s", "restricted_total_s", "max_shelf_dwell_s",
    "shelf_visits", "path_len", "mean_speed", "people_in_scene", "bbox_h",
    "has_time", "hour_sin", "hour_cos",
    # gestos (core/gestos.py): 0 en las alertas de zonas
    "gesture_dwell_s", "since_take_s", "takes", "reaches", "wrist_conf_drop", "bag_seen", "pose_ok_ratio",
)
# Duraciones y conteos tienen colas largas: se comprimen con log1p.
_LOG_FEATURES = {"track_age_s", "shelf_total_s", "checkout_total_s", "restricted_total_s",
                 "max_shelf_dwell_s", "shelf_visits", "path_len", "people_in_scene", "gesture_dwell_s", "takes",
                 "reaches"}

VECTOR_NAMES = tuple(FEATURE_NAMES) + tuple(f"regla={r}" for r in EVENT_TYPES)
GESTURE_FEATURES = ("gesture_dwell_s", "since_take_s", "takes", "reaches", "wrist_conf_drop", "bag_seen",
                    "pose_ok_ratio")
# las variables de antes de los gestos (29 oct), en el mismo orden: un modelo entrenado sin ejemplos de gestos usa
# solo estas, así lo siguen usando los agentes que todavía no se actualizaron (piden exactamente esta lista)
LEGACY_NAMES = tuple(n for n in FEATURE_NAMES if n not in GESTURE_FEATURES) + tuple(
    f"regla={r}" for r in EVENT_TYPES if r not in GESTURE_TYPES)

POSITIVE_LABELS = ("hurto", "sospechoso")
NEGATIVE_LABELS = ("normal", "falso_positivo")
LABELS = POSITIVE_LABELS + NEGATIVE_LABELS


def vectorize(event_type: str, features: dict, names: tuple[str, ...] = VECTOR_NAMES) -> np.ndarray:
    """Vector de un evento con las variables `names` (las del modelo): las que falten valen 0."""
    vals = []
    for name in names:
        if name.startswith("regla="):
            vals.append(1.0 if event_type == name[6:] else 0.0)
            continue
        v = float(features.get(name, 0.0) or 0.0)
        if name in _LOG_FEATURES:
            v = math.log1p(max(v, 0.0))
        vals.append(v)
    return np.array(vals, dtype=np.float64)


class Scorer:
    """Un modelo entrenado. Usa las variables con las que se entrenó (un modelo viejo sigue valiendo cuando se suman
    variables o tipos nuevos), y un tipo de evento que el modelo no conoce no se filtra: siempre es alerta."""

    def __init__(self, spec: dict):
        names = tuple(spec.get("vector_names", ()))
        unknown = [n for n in names if not n.startswith("regla=") and n not in FEATURE_NAMES]
        if not names or unknown:
            raise ValueError("El modelo fue entrenado con otras variables; reentrenar")
        self.names = names
        self.known_types = {n[6:] for n in names if n.startswith("regla=")}
        self.spec = spec
        self.mean = np.array(spec["mean"], dtype=np.float64)
        self.scale = np.array(spec["scale"], dtype=np.float64)
        self.coef = np.array(spec["coef"], dtype=np.float64)
        self.intercept = float(spec["intercept"])
        self.threshold = float(spec["threshold"])
        self.id = spec.get("id")

    @classmethod
    def maybe(cls, spec: dict | None) -> "Scorer | None":
        if not spec:
            return None
        try:
            return cls(spec)
        except Exception as exc:
            log.warning("Modelo de filtro ignorado: %s", exc)
            return None

    def score(self, event_type: str, features: dict) -> float:
        if event_type not in self.known_types:
            return 1.0  # tipo nuevo para este modelo: no hay con qué filtrarlo
        x = (vectorize(event_type, features, self.names) - self.mean) / self.scale
        z = float(x @ self.coef + self.intercept)
        return 1.0 / (1.0 + math.exp(-max(min(z, 50.0), -50.0)))

    def is_alert(self, score: float) -> bool:
        return score >= self.threshold
