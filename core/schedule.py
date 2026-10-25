"""Horario del local: cuándo vale la pena mirar las cámaras.

Fuera de horario la PC central **no abre el video** (no gasta internet, ni placa de video, ni
Hik-Connect), salvo que la cámara diga otra cosa (escuchar las alarmas del grabador o seguir 24 h).

Formato (lo que se guarda en `clients.hours`): un texto por día, vacío = cerrado.
    {"lun": "09:00-17:00", "mar": "09:00-13:00, 16:00-20:00", ..., "dom": ""}
- Horario cortado: varios tramos separados por coma.
- Pasando la medianoche: "18:00-02:00" (el tramo es del día en que abre).
- Todo el día: "00:00-24:00".
Sin horario cargado (None o {}) = abierto siempre (como antes).

Solo usa la biblioteca estándar: lo importa también el servidor.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

DAYS = ("lun", "mar", "mie", "jue", "vie", "sab", "dom")  # mismo orden que datetime.weekday()
DAY_NAMES = {"lun": "Lunes", "mar": "Martes", "mie": "Miércoles", "jue": "Jueves", "vie": "Viernes",
             "sab": "Sábado", "dom": "Domingo"}
DEFAULT_MARGIN_MIN = 15  # se empieza a mirar un rato antes de abrir y se sigue un rato después de cerrar

_RANGE = re.compile(r"^\s*(\d{1,2})(?:[:.](\d{2}))?\s*(?:-|a|–)\s*(\d{1,2})(?:[:.](\d{2}))?\s*$")

Parsed = dict[int, list[tuple[int, int]]]  # día (0 = lunes) -> tramos en minutos desde la medianoche


def _ranges(day: str, text: str) -> list[tuple[int, int]]:
    out = []
    for part in [p for p in re.split(r"[,;y]+", text or "") if p.strip()]:
        m = _RANGE.match(part)
        if not m:
            raise ValueError(f"{DAY_NAMES[day]}: «{part.strip()}» no es un horario (ejemplo: 09:00-17:00 o 9-17)")
        h1, m1, h2, m2 = (int(x or 0) for x in m.groups())  # «9-17» = 09:00-17:00
        if h1 > 23 or m1 > 59 or h2 > 24 or m2 > 59 or (h2 == 24 and m2):
            raise ValueError(f"{DAY_NAMES[day]}: hora inválida en «{part.strip()}»")
        start, end = h1 * 60 + m1, h2 * 60 + m2
        if start == end:
            raise ValueError(f"{DAY_NAMES[day]}: el tramo «{part.strip()}» abre y cierra a la misma hora")
        if end < start:
            end += 24 * 60  # cierra pasada la medianoche
        out.append((start, end))
    return sorted(out)


def parse_hours(hours: dict | None) -> Parsed | None:
    """None si no hay horario (abierto siempre). Error en español si algo está mal escrito."""
    if not hours:
        return None
    bad = [k for k in hours if k not in DAYS]
    if bad:
        raise ValueError(f"Días desconocidos en el horario: {bad} (usar {', '.join(DAYS)})")
    parsed = {i: _ranges(d, str(hours.get(d) or "")) for i, d in enumerate(DAYS)}
    if not any(parsed.values()):
        return None  # todo vacío = sin horario (abierto siempre)
    return parsed


def normalize_hours(hours: dict | None) -> dict | None:
    """Lo que se guarda: los 7 días con el texto prolijo («09:00-17:00»), o None si no hay horario."""
    parsed = parse_hours(hours)
    if parsed is None:
        return None

    def fmt(m: int) -> str:
        return "24:00" if m == 24 * 60 else f"{m % (24 * 60) // 60:02d}:{m % 60:02d}"

    return {d: ", ".join(f"{fmt(s)}-{fmt(e)}" for s, e in parsed[i]) for i, d in enumerate(DAYS)}


def _intervals(parsed: Parsed, now: datetime, margin_min: float, days: range):
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    for off in days:
        base = midnight + timedelta(days=off)
        for s, e in parsed[base.weekday()]:
            yield base + timedelta(minutes=s - margin_min), base + timedelta(minutes=e + margin_min)


def is_open(hours: dict | Parsed | None, now: datetime, margin_min: float = DEFAULT_MARGIN_MIN) -> bool:
    parsed = hours if _is_parsed(hours) else parse_hours(hours)  # type: ignore[arg-type]
    if parsed is None:
        return True
    return any(a <= now < b for a, b in _intervals(parsed, now, margin_min, range(-1, 2)))


def next_open(hours: dict | Parsed | None, now: datetime, margin_min: float = DEFAULT_MARGIN_MIN) -> datetime | None:
    """Cuándo vuelve a abrir (con el margen). None si ya está abierto o no hay horario."""
    parsed = hours if _is_parsed(hours) else parse_hours(hours)  # type: ignore[arg-type]
    if parsed is None or is_open(parsed, now, margin_min):
        return None
    starts = [a for a, _ in _intervals(parsed, now, margin_min, range(0, 8)) if a > now]
    return min(starts) if starts else None


def open_hours_per_week(hours: dict | None) -> float:
    """Horas abiertas por semana (sin margen): para estimar cuánto se ahorra."""
    parsed = parse_hours(hours)
    if parsed is None:
        return 24 * 7.0
    return sum(e - s for ranges in parsed.values() for s, e in ranges) / 60.0


def _is_parsed(h) -> bool:
    return isinstance(h, dict) and bool(h) and all(isinstance(k, int) for k in h)
