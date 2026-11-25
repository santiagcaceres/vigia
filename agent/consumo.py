"""En qué se va el procesador de este proceso de análisis, por grupo de hilos (25 nov: para meter más cámaras por
máquina hay que saber qué gasta; antes solo se sabía corriendo `tools/medir_maquina.py` a mano en la máquina).

Lee `/proc/self/task` (Linux; en otro sistema no informa nada) y lo agrupa:

- `video`: decodificar el video (los hilos de FFmpeg «av:…» cuando una cámara usa varios, más lo que cada cámara gasta
  leyendo y pasando a imagen, que con un solo hilo por cámara es donde se decodifica).
- `camaras`: el resto del trabajo de cada cámara: preparar la imagen para el modelo, leer el resultado, seguimiento,
  reglas, gestos, métricas, guardar los cuadros para los clips.
- `modelo`: los hilos que le dan el trabajo a la placa de video (detector y postura por lotes).
- `clips`: armar y subir los clips.
- `nativos`: hilos de las librerías (onnxruntime, CUDA) que no son de Python.
- `otros`: lo demás (hablar con el servidor, imagen en vivo…).

Los números son procesadores enteros (0,42 = el 42 % de un procesador). Solo nombres de grupo: nada de la máquina."""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict

GROUPS = ("video", "camaras", "modelo", "clips", "nativos", "otros")


def thread_cpu() -> dict[int, tuple[str, float]]:
    """Cada hilo del proceso: id → (nombre que le da el sistema, segundos de procesador que lleva gastados)."""
    out: dict[int, tuple[str, float]] = {}
    try:
        hz = float(os.sysconf("SC_CLK_TCK"))
        tids = os.listdir("/proc/self/task")
    except (OSError, ValueError, AttributeError):
        return out
    for tid in tids:
        try:
            with open(f"/proc/self/task/{tid}/stat", "rb") as f:
                data = f.read()
            left, right = data.index(b"("), data.rindex(b")")  # (el nombre puede traer espacios y paréntesis)
            rest = data[right + 2:].split()
            out[int(tid)] = (data[left + 1:right].decode(errors="replace"), (int(rest[11]) + int(rest[12])) / hz)
        except (OSError, ValueError, IndexError):  # el hilo terminó mientras se leía
            continue
    return out


def group_of(comm: str, name: str | None) -> str:
    """`comm`: nombre del hilo en el sistema; `name`: el del hilo de Python (None si no es de Python)."""
    if comm.startswith("av:"):
        return "video"
    if name is None:
        return "nativos"
    if name.startswith("cam-"):
        return "camaras"
    if name.endswith("-lotes"):
        return "modelo"
    if name.startswith("clips"):
        return "clips"
    return "otros"


class CpuMeter:
    """`sample(read_s)` devuelve el consumo por grupo desde la consulta anterior. `read_s`: segundos de procesador que
    las cámaras llevan gastados leyendo el video (acumulado): se pasan de `camaras` a `video`."""

    def __init__(self, min_span: float = 20.0):
        self.min_span = min_span  # (el sistema cuenta de a 10 ms por hilo: con ventanas cortas el número baila)
        self._prev: dict[int, tuple[str, float]] | None = None
        self._read_s = 0.0
        self._at = 0.0
        self.last: dict[str, float] = {}

    def sample(self, read_s: float = 0.0) -> dict[str, float]:
        now = time.monotonic()
        if self._prev is not None and now - self._at < self.min_span:
            return self.last
        cur = thread_cpu()
        if not cur:
            return {}
        if self._prev is not None:
            span = now - self._at
            names = {t.native_id: t.name for t in threading.enumerate()}
            used: dict[str, float] = defaultdict(float)
            for tid, (comm, secs) in cur.items():
                used[group_of(comm, names.get(tid))] += max(secs - self._prev.get(tid, (comm, 0.0))[1], 0.0)
            moved = min(max(read_s - self._read_s, 0.0), used["camaras"])
            used["camaras"] -= moved
            used["video"] += moved
            self.last = {g: round(used[g] / span, 2) for g in GROUPS if used[g] / span >= 0.005}
            self.last["total"] = round(sum(used.values()) / span, 2)
            self.last["cores"] = os.cpu_count() or 0  # procesadores de la máquina (para saber cuánto queda libre)
        self._prev, self._read_s, self._at = cur, read_s, now
        return self.last
