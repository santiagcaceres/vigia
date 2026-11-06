"""Cómo detecta personas cada cámara: «Sensibilidad» (1 a 10) y «Personas lejos».

Python puro (sin OpenCV ni numpy): lo importa también el servidor de Vercel, que no tiene OpenCV. El detector que lo
aplica está en core/detectors.py (`tuned`, `TunedDetector`).

- **Sensibilidad** = qué tan seguro tiene que estar el detector para aceptar una persona. 5 = como siempre (confianza
  0,35). Más alta ve personas más chicas, lejanas o borrosas, pero también confunde más cosas con personas (cajas,
  maniquíes, carteles); más baja, al revés. Confianza mínima = 0,55 − 0,04 × sensibilidad (10 → 0,15; 1 → 0,51).
- **Personas lejos**: además de la imagen entera, el detector mira la imagen en 4 partes (2x2, que se pisan un poco),
  cada una agrandada a la entrada del modelo: la gente del fondo le llega más grande. Cuesta unas 5 veces el detector
  en esa cámara (`FAR_PASSES`) y sirve más si la imagen tiene detalle (1280x720 o la calidad principal del grabador).

Viaja a las PCs y máquinas en `analysis["detection"]`, compacto (solo lo que difiere de lo de siempre, `to_dict()`), y
los agentes lo leen tolerantes (`from_dict`): un valor raro queda en lo de siempre y no traba la cámara. El servidor lo
valida estricto al guardar (`from_dict(strict=True)`).
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_SENSITIVITY = 5
MIN_SENSITIVITY, MAX_SENSITIVITY = 1, 10
FAR_PASSES = 5  # con «Personas lejos»: la imagen entera + 4 partes (para los textos y para medir)
FAR_WEIGHT = 4  # cuántas cámaras «pesa» una con «Personas lejos» en el reparto de las máquinas (server/pool.FAR_FACTOR)
KEYS = ("sensitivity", "far")

_MSG_OBJ = "detection: va un objeto con «sensitivity» (1 a 10) y «far» (true o false)"
_MSG_SENS = "detection.sensitivity: la sensibilidad va de 1 a 10 (un número entero)"
_MSG_FAR = "detection.far: va true o false"


def _sensitivity(v) -> int | None:
    """La sensibilidad si `v` es un entero de 1 a 10 (también 7.0); si no, None. Un booleano no cuenta."""
    if isinstance(v, bool):
        return None
    if isinstance(v, float):
        if not v.is_integer():
            return None
        v = int(v)
    if isinstance(v, int) and MIN_SENSITIVITY <= v <= MAX_SENSITIVITY:
        return v
    return None


def conf_for(s) -> float:
    """Confianza mínima para aceptar una persona con la sensibilidad `s` (se lleva a 1..10; algo que no es un número
    da la de siempre). Redondeada a 2 decimales: con 5 da 0,35 exacto, como antes de que existiera el ajuste."""
    try:
        s = int(s)
    except (TypeError, ValueError):
        s = DEFAULT_SENSITIVITY
    s = max(MIN_SENSITIVITY, min(MAX_SENSITIVITY, s))
    return round(0.55 - 0.04 * s, 2)


@dataclass(frozen=True)
class DetectionConfig:
    sensitivity: int = DEFAULT_SENSITIVITY
    far: bool = False

    @classmethod
    def from_dict(cls, d, strict: bool = False) -> "DetectionConfig":
        """`strict=False` (agentes, lo leído de la base): todo lo raro queda en lo de siempre (algo que no es un objeto,
        una sensibilidad fuera de 1..10 o que no es un entero, un «far» que no es true/false) y las claves que no se
        conocen se ignoran. `strict=True` (el servidor, al guardar): ValueError con el motivo en castellano.
        `None` (en todo o en una clave) es lo mismo que no mandarlo."""
        if d is None:
            return cls()
        if not isinstance(d, dict):
            if strict:
                raise ValueError(_MSG_OBJ)
            return cls()
        if strict:
            unknown = sorted(str(k) for k in d if k not in KEYS)
            if unknown:
                raise ValueError(f"detection: claves desconocidas {unknown}")
        s, far = d.get("sensitivity"), d.get("far")
        sens = DEFAULT_SENSITIVITY
        if s is not None:
            v = _sensitivity(s)
            if v is None and strict:
                raise ValueError(_MSG_SENS)
            sens = DEFAULT_SENSITIVITY if v is None else v
        lejos = False
        if far is not None:
            if not isinstance(far, bool):
                if strict:
                    raise ValueError(_MSG_FAR)
            else:
                lejos = far
        return cls(sens, lejos)

    def to_dict(self, compact: bool = True) -> dict:
        """`compact=True` (lo que viaja a las máquinas): solo lo que difiere de lo de siempre ({} si es todo lo de
        siempre). `compact=False` (la app y la web): las dos claves."""
        if not compact:
            return {"sensitivity": self.sensitivity, "far": self.far}
        out: dict = {}
        if self.sensitivity != DEFAULT_SENSITIVITY:
            out["sensitivity"] = self.sensitivity
        if self.far:
            out["far"] = True
        return out

    @property
    def conf(self) -> float:
        return conf_for(self.sensitivity)

    @property
    def is_default(self) -> bool:
        return self.sensitivity == DEFAULT_SENSITIVITY and not self.far

    @property
    def passes(self) -> int:
        """Cuántas veces corre el detector por cuadro analizado."""
        return FAR_PASSES if self.far else 1

    def describe(self) -> str:
        """Para el registro y la app: «normal», «sensibilidad 8 de 10», «personas lejos» o las dos."""
        if self.is_default:
            return "normal"
        parts = []
        if self.sensitivity != DEFAULT_SENSITIVITY:
            parts.append(f"sensibilidad {self.sensitivity} de {MAX_SENSITIVITY}")
        if self.far:
            parts.append("personas lejos")
        return " · ".join(parts)
