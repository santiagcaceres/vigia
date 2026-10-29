"""Videos de prueba de cada cámara («Ajustar…» → «Videos de prueba»).

Quedan guardados para volver a probarlos con un clic cada vez que se cambian zonas o tiempos, y sus alertas sirven
para entrenar el modelo del cliente.
- Los videos no se suben: quedan en esta PC. La lista va en `<datos>/videos_prueba.json`, con la ruta de cada uno.
  Por defecto se eligen de la carpeta Descargas\\VIGIA.
- De cada video se dice qué es: «hurto» (con los momentos en que pasa) o «normal» (no debería dar alertas).
- `evaluate` cruza las alertas de prueba con eso: qué hurtos vio la cámara, cuáles no, y alertas de más. Cada tipo de
  alerta tiene su ventana alrededor del momento marcado (los gestos pasan justo en el momento; las reglas, después).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from core.gestos import GESTURE_TYPES

log = logging.getLogger("vigia.biblioteca")

KINDS = {"hurto": "Hurto", "normal": "Normal (sin hurto)", "": "Sin clasificar"}
# una alerta «ve» un hurto marcado si sale hasta 20 s antes o hasta 90 s después: «merodeo» y «salida sin pago»
# saltan después del momento del hurto (cuando la persona cumplió el tiempo o se fue)
MATCH_BEFORE = 20.0
MATCH_AFTER = 90.0
# los gestos de ocultamiento (mano al bolsillo, a la ropa o a la bolsa) saltan en el momento en que se guarda algo:
# una ventana más estrecha, para que un gesto de otro momento del video no cuente como «vio el hurto»
MATCH_WINDOWS = {g: (5.0, 5.0) for g in GESTURE_TYPES}
VIDEO_TYPES = ("Videos", "*.mp4 *.avi *.mkv *.mov *.ts *.dav *.264 *.h264")


@dataclass
class TestVideo:
    path: str
    kind: str = ""  # "hurto" | "normal" | "" (sin clasificar)
    marks: list[float] = field(default_factory=list)  # segundos del video en que pasa el hurto
    note: str = ""
    added: str = ""
    last: str = ""  # resultado de la última prueba, en texto
    last_at: str = ""

    @property
    def name(self) -> str:
        return Path(self.path).name

    def exists(self) -> bool:
        return Path(self.path).is_file()


def default_folder() -> Path:
    """Descargas\\VIGIA (la carpeta que se acordó para los videos de prueba). Si no está, se crea."""
    downloads = Path.home() / "Downloads"
    if not downloads.is_dir():
        downloads = Path.home() / "Descargas" if (Path.home() / "Descargas").is_dir() else Path.home()
    folder = downloads / "VIGIA"
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError:
        return downloads
    return folder


class Library:
    """Lista de videos de prueba por cámara, guardada en un archivo JSON de esta PC."""

    def __init__(self, file: Path):
        self.file = Path(file)
        self._lock = threading.Lock()
        self._data: dict[str, list[dict]] = {}
        try:
            if self.file.exists():
                self._data = json.loads(self.file.read_text(encoding="utf-8")).get("camaras", {})
        except (OSError, ValueError) as exc:
            log.warning("No se pudo leer la lista de videos de prueba (%s): empieza vacía", exc)

    def videos(self, cam_id: int) -> list[TestVideo]:
        with self._lock:
            return [TestVideo(**{k: v for k, v in d.items() if k in TestVideo.__dataclass_fields__})
                    for d in self._data.get(str(cam_id), [])]

    def add(self, cam_id: int, paths: list[str]) -> list[TestVideo]:
        """Suma videos a la cámara (los que ya estaban no se repiten). Devuelve los nuevos."""
        have = {os.path.normcase(os.path.abspath(v.path)) for v in self.videos(cam_id)}
        new = []
        for p in paths:
            key = os.path.normcase(os.path.abspath(p))
            if key in have:
                continue
            have.add(key)
            new.append(TestVideo(os.path.abspath(p), added=datetime.now().isoformat(timespec="seconds")))
        if new:
            with self._lock:
                self._data.setdefault(str(cam_id), []).extend(asdict(v) for v in new)
            self.save()
        return new

    def update(self, cam_id: int, video: TestVideo) -> None:
        with self._lock:
            rows = self._data.setdefault(str(cam_id), [])
            for i, d in enumerate(rows):
                if d.get("path") == video.path:
                    rows[i] = asdict(video)
                    break
            else:
                rows.append(asdict(video))
        self.save()

    def remove(self, cam_id: int, path: str) -> None:
        """Lo saca de la lista (el archivo no se borra)."""
        with self._lock:
            self._data[str(cam_id)] = [d for d in self._data.get(str(cam_id), []) if d.get("path") != path]
        self.save()

    def save(self) -> None:
        with self._lock:
            data = json.dumps({"camaras": self._data}, ensure_ascii=False, indent=1)
        try:
            self.file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.file.with_suffix(".tmp")
            tmp.write_text(data, encoding="utf-8")
            os.replace(tmp, self.file)
        except OSError as exc:
            log.warning("No se pudo guardar la lista de videos de prueba: %s", exc)


def fmt_ts(sec: float) -> str:
    sec = max(0, int(round(sec)))
    return f"{sec // 60}:{sec % 60:02d}"


def matches(mark: float, alert_ts: float, alert_type: str | None = None) -> bool:
    """La alerta (de ese tipo) cae cerca del momento marcado del hurto."""
    before, after = MATCH_WINDOWS.get(alert_type, (MATCH_BEFORE, MATCH_AFTER))
    return mark - before <= alert_ts <= mark + after


def evaluate(video: TestVideo, alerts: list, type_names: dict[str, str] | None = None) -> dict:
    """Cruza las alertas de prueba de un video (objetos con `.ts` y `.type`) con lo que se sabe del video.
    Devuelve `caught` [(momento, alerta)], `missed` [momentos], `extra` [alertas], `ok` (True/False/None) y `text`."""
    names = type_names or {}
    caught, missed, used = [], [], set()
    for m in sorted(video.marks):
        hit = next((a for a in alerts if matches(m, a.ts, a.type)), None)
        if hit is None:
            missed.append(m)
        else:
            caught.append((m, hit))
            used.update(id(a) for a in alerts if matches(m, a.ts, a.type))
    extra = [a for a in alerts if id(a) not in used]

    def what(a) -> str:
        return f"{names.get(a.type, a.type).lower()} {fmt_ts(a.ts)}"

    if video.kind == "normal":
        ok = not alerts
        text = "✔ sin alertas, bien" if ok else (f"✘ {len(alerts)} falsa{'s' if len(alerts) != 1 else ''} "
                                                 f"alarma{'s' if len(alerts) != 1 else ''} ({what(alerts[0])})")
    elif video.kind == "hurto" and video.marks:
        ok = not missed
        if len(video.marks) == 1:
            text = (f"✔ vio el hurto ({what(caught[0][1])})" if caught else
                    f"✘ no vio el hurto de {fmt_ts(missed[0])}")
        else:
            text = f"{'✔' if ok else '✘'} vio {len(caught)} de {len(video.marks)} hurtos" + (
                f" (no vio {', '.join(fmt_ts(m) for m in missed)})" if missed else "")
        if extra:
            text += f" · {len(extra)} alerta{'s' if len(extra) != 1 else ''} más"
    elif video.kind == "hurto":
        ok = None if alerts else False
        text = (f"dio {len(alerts)} alerta{'s' if len(alerts) != 1 else ''} ({what(alerts[0])}); marcá el momento del "
                "hurto para medir mejor" if alerts else "✘ ninguna alerta en un video de hurto")
    else:
        ok = None
        text = (f"{len(alerts)} alerta{'s' if len(alerts) != 1 else ''} ({what(alerts[0])}) · decí si es hurto o normal"
                if alerts else "sin alertas · decí si es hurto o normal")
    return {"caught": caught, "missed": missed, "extra": extra, "ok": ok, "text": text}


def suggested_label(video: TestVideo, alert) -> str | None:
    """Etiqueta para entrenar que se desprende de lo marcado: alerta en el momento de un hurto → «hurto»; alerta en
    un video normal → «falso_positivo». Si no se sabe, None (la elige la persona)."""
    if video.kind == "normal":
        return "falso_positivo"
    if video.kind == "hurto" and any(matches(m, alert.ts, alert.type) for m in video.marks):
        return "hurto"
    return None


def sample_uuid(cam_id: int, video: TestVideo, alert) -> str:
    """Identificador estable de una alerta de prueba (el mismo momento, tipo y persona del mismo video): si se manda
    dos veces al entrenamiento, se actualiza la etiqueta en lugar de repetirla."""
    try:
        size = Path(video.path).stat().st_size
    except OSError:
        size = 0
    key = f"{cam_id}|{video.name}|{size}|{alert.type}|{alert.track_id}|{round(alert.ts, 1)}"
    return hashlib.sha1(key.encode()).hexdigest()[:32]


def live_uuid(cam_id: int, alert) -> str:
    """Igual que `sample_uuid` para una alerta de prueba en vivo (por la hora en que pasó)."""
    when = alert.wall.isoformat() if alert.wall else f"{alert.ts:.1f}"
    return hashlib.sha1(f"{cam_id}|vivo|{when}|{alert.type}|{alert.track_id}".encode()).hexdigest()[:32]
