"""Entrenamiento del filtro de alertas con las etiquetas de la persona de entrenamiento.

Regresión logística con regularización L2 y pesos balanceados, implementada con numpy
(sin scikit-learn, para que el servidor entre liviano en Vercel). Se valida separando por
cámara cuando hay suficientes, así la métrica refleja cómo funciona en una cámara nueva.
"""

from __future__ import annotations

import numpy as np

from core.gestos import GESTURE_TYPES
from core.scorer import LEGACY_NAMES, NEGATIVE_LABELS, POSITIVE_LABELS, VECTOR_NAMES, vectorize

MIN_TOTAL = 30
MIN_PER_CLASS = 8
C = 0.5  # inversa de la regularización (como en scikit-learn)


class NotEnoughData(ValueError):
    pass


def vector_names(events: list[dict]) -> tuple[str, ...]:
    """Con qué variables entrenar: sin ejemplos de gestos, las de antes de los gestos (un agente sin actualizar exige
    esa lista exacta y si no, ignora el modelo entero y no filtra nada)."""
    if any(e.get("type") in GESTURE_TYPES for e in events):
        return VECTOR_NAMES
    return LEGACY_NAMES


def build_dataset(events: list[dict], names: tuple[str, ...] = VECTOR_NAMES) -> tuple[np.ndarray, np.ndarray,
                                                                                      np.ndarray]:
    X, y, groups = [], [], []
    for e in events:
        if e["label"] in POSITIVE_LABELS:
            y.append(1)
        elif e["label"] in NEGATIVE_LABELS:
            y.append(0)
        else:
            continue
        X.append(vectorize(e["type"], e["features"] or {}, names))
        groups.append(e["camera_id"])
    return np.array(X).reshape(len(y), len(names)), np.array(y), np.array(groups)


class Standardizer:
    def fit(self, X: np.ndarray) -> "Standardizer":
        self.mean = X.mean(axis=0)
        self.scale = X.std(axis=0)
        self.scale[self.scale == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.scale


def fit_logistic(X: np.ndarray, y: np.ndarray, C: float = C, iters: int = 100) -> tuple[np.ndarray, float]:
    """Newton-Raphson sobre: sum_i w_i * logloss_i + ||coef||² / (2C), con w balanceados."""
    n, d = X.shape
    pos = y.sum()
    w = np.where(y == 1, n / (2 * max(pos, 1)), n / (2 * max(n - pos, 1)))
    Xb = np.hstack([X, np.ones((n, 1))])
    theta = np.zeros(d + 1)
    reg = np.full(d + 1, 1.0 / C)
    reg[-1] = 0.0  # el intercepto no se regulariza
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ theta, -50, 50)))
        grad = Xb.T @ (w * (p - y)) + reg * theta
        H = (Xb * (w * p * (1 - p))[:, None]).T @ Xb + np.diag(reg) + 1e-9 * np.eye(d + 1)
        step = np.linalg.solve(H, grad)
        theta -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return theta[:-1], float(theta[-1])


def predict(X: np.ndarray, coef: np.ndarray, intercept: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(X @ coef + intercept, -50, 50)))


def _folds(y: np.ndarray, groups: np.ndarray) -> tuple[list[np.ndarray], str]:
    cams = sorted(set(groups.tolist()))
    if len(cams) >= 3:
        k = min(5, len(cams))
        # cámaras repartidas en k grupos, de mayor a menor cantidad de eventos
        counts = sorted(cams, key=lambda c: -int(np.sum(groups == c)))
        bins: list[list] = [[] for _ in range(k)]
        sizes = [0] * k
        for c in counts:
            i = sizes.index(min(sizes))
            bins[i].append(c)
            sizes[i] += int(np.sum(groups == c))
        folds = [np.where(np.isin(groups, b))[0] for b in bins]
        if all(len(set(y[np.setdiff1d(np.arange(len(y)), f)].tolist())) == 2 for f in folds):
            return folds, "por_camara"
    rng = np.random.default_rng(0)
    k = int(min(5, y.sum(), len(y) - y.sum()))
    folds = [[] for _ in range(k)]
    for cls in (0, 1):
        idx = rng.permutation(np.where(y == cls)[0])
        for j, i in enumerate(idx):
            folds[j % k].append(i)
    return [np.array(sorted(f)) for f in folds], "aleatoria"


def train(events: list[dict], target_recall: float = 0.9) -> tuple[dict, dict]:
    names = vector_names(events)
    X, y, groups = build_dataset(events, names)
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    if len(y) < MIN_TOTAL or n_pos < MIN_PER_CLASS or n_neg < MIN_PER_CLASS:
        raise NotEnoughData(
            f"Faltan etiquetas: hay {len(y)} ({n_pos} de alerta real, {n_neg} sin importancia). "
            f"Se necesitan al menos {MIN_TOTAL} en total y {MIN_PER_CLASS} de cada tipo.")

    folds, validation = _folds(y, groups)
    oof = np.full(len(y), np.nan)
    for te in folds:
        tr = np.setdiff1d(np.arange(len(y)), te)
        sc = Standardizer().fit(X[tr])
        coef, b = fit_logistic(sc.transform(X[tr]), y[tr])
        oof[te] = predict(sc.transform(X[te]), coef, b)

    threshold = _threshold_for_recall(oof, y, target_recall)
    metrics = _metrics(oof, y, threshold)
    metrics.update(validation=validation, cameras=len(set(groups.tolist())), target_recall=target_recall)

    sc = Standardizer().fit(X)
    coef, b = fit_logistic(sc.transform(X), y)
    spec = {"vector_names": list(names), "mean": sc.mean.tolist(), "scale": sc.scale.tolist(),
            "coef": coef.tolist(), "intercept": b, "threshold": float(threshold)}
    return spec, metrics


def _threshold_for_recall(p: np.ndarray, y: np.ndarray, target: float) -> float:
    """El umbral más alto que todavía deja pasar al menos `target` de las alertas reales."""
    pos = np.sort(p[y == 1])
    k = int(np.floor((1 - target) * len(pos)))
    return float(max(0.0, pos[k] - 1e-6))


def _metrics(p: np.ndarray, y: np.ndarray, thr: float) -> dict:
    pred = p >= thr
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    return {
        "n": int(len(y)), "positivos": int(y.sum()), "negativos": int(len(y) - y.sum()),
        "umbral": round(thr, 4),
        "recall": round(tp / max(tp + fn, 1), 3),
        "precision": round(tp / max(tp + fp, 1), 3),
        "precision_sin_modelo": round(float(y.mean()), 3),
        "falsas_alarmas_eliminadas": round(tn / max(tn + fp, 1), 3),
        "reales_perdidas": fn,
        "matriz": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }
