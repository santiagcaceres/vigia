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
- `externos`: los programas que lanza el proceso y ya terminaron (ffmpeg al armar o bajar cada clip; el entrenamiento
  de «Entrenar objetos»). Se cuentan cuando terminan: no están en `/proc/self/task` (26 nov: los clips de una tanda de
  alertas dejaron la máquina sin procesador y el medidor no lo veía).

Los números son procesadores enteros (0,42 = el 42 % de un procesador). Además, de la máquina entera (lo mismo en
todos sus procesos): `load` (la carga del último minuto: cuántos procesos quieren procesador), `wait_pct` (qué parte del
tiempo hubo procesos esperando procesador, `/proc/pressure/cpu`), y de la placa de video `gpu` (% de uso) y `gpu_gb`
(memoria usada, en GB), si la hay. Solo nombres de grupo y números: nada de la máquina."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import time
from collections import defaultdict

GROUPS = ("video", "camaras", "modelo", "clips", "nativos", "otros", "externos")


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


def machine_load() -> dict[str, float]:
    """La carga de la máquina entera: `load` (último minuto) y `wait_pct` (parte del tiempo en que hubo procesos
    esperando procesador, últimos 10 s). Vacío donde no se puede leer (Windows)."""
    out: dict[str, float] = {}
    try:
        out["load"] = round(os.getloadavg()[0], 1)
    except (OSError, AttributeError):
        pass
    try:
        with open("/proc/pressure/cpu", encoding="ascii") as f:
            m = re.search(r"some avg10=([\d.]+)", f.read())
        if m:
            out["wait_pct"] = round(float(m.group(1)), 1)
    except OSError:
        pass
    return out


def gpu_usage(timeout: float = 5.0) -> dict[str, float]:
    """Uso de la placa de video NVIDIA (la primera): `gpu` (%) y `gpu_gb` (memoria usada; en GB porque el servidor no
    guarda números de más de 1000). Vacío sin placa."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return {}
    try:
        r = subprocess.run([smi, "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=timeout, check=False)
        use, mem = (float(v) for v in r.stdout.strip().splitlines()[0].split(",")[:2])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return {}
    return {"gpu": round(use), "gpu_gb": round(mem / 1024, 2)}


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
        self._children = 0.0  # procesador de los programas que lanzó (ffmpeg) y ya terminaron, acumulado
        self._at = 0.0
        self.last: dict[str, float] = {}

    def sample(self, read_s: float = 0.0) -> dict[str, float]:
        now = time.monotonic()
        if self._prev is not None and now - self._at < self.min_span:
            return self.last
        cur = thread_cpu()
        if not cur:
            return {}
        t = os.times()
        children = t.children_user + t.children_system
        if self._prev is not None:
            span = now - self._at
            names = {t.native_id: t.name for t in threading.enumerate()}
            used: dict[str, float] = defaultdict(float)
            for tid, (comm, secs) in cur.items():
                used[group_of(comm, names.get(tid))] += max(secs - self._prev.get(tid, (comm, 0.0))[1], 0.0)
            moved = min(max(read_s - self._read_s, 0.0), used["camaras"])
            used["camaras"] -= moved
            used["video"] += moved
            used["externos"] = max(children - self._children, 0.0)
            self.last = {g: round(used[g] / span, 2) for g in GROUPS if used[g] / span >= 0.005}
            self.last["total"] = round(sum(used.values()) / span, 2)
            self.last["cores"] = os.cpu_count() or 0  # procesadores de la máquina (para saber cuánto queda libre)
            self.last.update(machine_load())
            self.last.update(gpu_usage())
        self._prev, self._read_s, self._at, self._children = cur, read_s, now, children
        return self.last
