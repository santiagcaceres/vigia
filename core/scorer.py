"""Modelo de filtro de alertas.

Aprende de las etiquetas de la persona de entrenamiento qué eventos valen una alerta
(hurto / sospechoso) y cuáles no (normal / falso positivo). Es una regresión logística
guardada como JSON: el agente la evalúa con numpy, sin scikit-learn ni archivos pickle.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from .rules import EVENT_TYPES

log = logging.getLogger("vigia.scorer")

FEATURE_NAMES = (
    "track_age_s", "shelf_total_s", "checkout_total_s", "restricted_total_s", "max_shelf_dwell_s",
    "shelf_visits", "path_len", "mean_speed", "people_in_scene", "bbox_h",
    "has_time", "hour_sin", "hour_cos",
)
# Duraciones y conteos tienen colas largas: se comprimen con log1p.
_LOG_FEATURES = {"track_age_s", "shelf_total_s", "checkout_total_s", "restricted_total_s",
                 "max_shelf_dwell_s", "shelf_visits", "path_len", "people_in_scene"}

VECTOR_NAMES = tuple(FEATURE_NAMES) + tuple(f"regla={r}" for r in EVENT_TYPES)

POSITIVE_LABELS = ("hurto", "sospechoso")
NEGATIVE_LABELS = ("normal", "falso_positivo")
LABELS = POSITIVE_LABELS + NEGATIVE_LABELS


def vectorize(event_type: str, features: dict) -> np.ndarray:
    vals = []
    for name in FEATURE_NAMES:
        v = float(features.get(name, 0.0) or 0.0)
        if name in _LOG_FEATURES:
            v = math.log1p(max(v, 0.0))
        vals.append(v)
    vals += [1.0 if event_type == r else 0.0 for r in EVENT_TYPES]
    return np.array(vals, dtype=np.float64)


class Scorer:
    def __init__(self, spec: dict):
        if tuple(spec.get("vector_names", ())) != VECTOR_NAMES:
            raise ValueError("El modelo fue entrenado con otras variables; reentrenar")
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
        x = (vectorize(event_type, features) - self.mean) / self.scale
        z = float(x @ self.coef + self.intercept)
        return 1.0 / (1.0 + math.exp(-max(min(z, 50.0), -50.0)))

    def is_alert(self, score: float) -> bool:
        return score >= self.threshold
