"""En qué funciones se va el procesador de los procesos de análisis (25 nov: `medir_maquina.py` dice que el 64 % es
«python», pero no en qué; el detector casi no trabaja —lotes de 1, unas 29 imágenes por segundo entre 20 cámaras— así
que hay que mirar adentro antes de optimizar).

Se corre EN LA MÁQUINA con el análisis andando. No frena las cámaras (mira sin pausar los procesos) y no cambia nada:
baja la herramienta `py-spy` a una carpeta temporal y la borra al terminar.

    curl -fsSL https://raw.githubusercontent.com/santiagcaceres/vigia/main/tools/perfil_maquina.py -o /tmp/perfil.py
    sudo python3 /tmp/perfil.py            # 20 segundos; «sudo python3 /tmp/perfil.py 40» para 40

Muestra, sumando todos los procesos: en qué funciones está parado el programa cuando gasta procesador (las de más
abajo de la pila) y cuánto se lleva cada parte nuestra (leer el video, el detector, la postura, guardar los cuadros de
los clips…). No imprime claves ni direcciones: solo nombres de funciones y archivos."""

from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict

PIP_CANDIDATES = ("/opt/vigia/.venv/bin/pip", "pip3", "pip")
# partes del programa: la primera que aparezca en la pila (de adentro hacia afuera) se lleva la muestra
PARTS = (
    ("escribir el clip (mp4)", ("write_mp4", "_finish_clips", "_upload", "upload_clip")),
    ("postura (modelo)", ("estimate", "estimate_many", "look (")),
    ("detector: hilo de lotes (modelo en la placa)", ("_loop (core/detectors",)),
    ("detector: preparar y leer el resultado", ("_letterbox", "_decode", "_infer", "detect_also", "detect (")),
    ("seguimiento (colores de la ropa)", ("tracker.py",)),
    ("gestos y reglas", ("gestos.py", "rules.py", "hurto_ia.py")),
    ("métricas", ("metrics.py",)),
    ("guardar el cuadro para el clip (JPEG)", ("_small_jpg", "_small (", "_remember", "idle (core/analyzer")),
    ("filtro de movimiento", ("motion.py",)),
    ("imagen en vivo y vista previa", ("live.py", "_send_preview")),
    ("leer el video: esperar y decodificar (grab)", ("grab",)),
    ("leer el video: pasar a imagen (retrieve)", ("retrieve", "square_pixels")),
    ("hablar con el servidor", ("httpx", "httpcore", "outbox.py", "ssl.py")),
)


def run(cmd: list[str], timeout: float = 120) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)


def analysis_pids() -> list[int]:
    out = []
    for p in glob.glob("/proc/[0-9]*"):
        try:
            with open(p + "/cmdline", "rb") as f:
                cmd = f.read().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if "-m agent" in cmd and "perfil" not in cmd:
            out.append(int(p.rsplit("/", 1)[1]))
    return sorted(out)


def get_pyspy(tmp: str) -> str | None:
    found = shutil.which("py-spy")
    if found:
        return found
    for pip in PIP_CANDIDATES:
        if not (os.path.exists(pip) or shutil.which(pip)):
            continue
        r = run([pip, "install", "--quiet", "--target", tmp, "py-spy"], timeout=180)
        exe = os.path.join(tmp, "bin", "py-spy")
        if r.returncode == 0 and os.path.exists(exe):
            return exe
    return None


def short(frame: str) -> str:
    """«estimate (core/pose.py:95)» → «estimate (core/pose.py)»: sin la ruta de instalación ni el número de línea, salvo
    en el bucle que lee el video (agent/worker.py), donde la línea dice si está esperando el cuadro, pasándolo a
    imagen o analizando."""
    frame = frame.strip()
    if "agent/worker.py" not in frame:
        frame = re.sub(r":\d+\)", ")", frame)
    frame = re.sub(r"\((?:/[^ )]*/)?(?:site-packages/|vigia/)", "(", frame)
    return frame if len(frame) <= 110 else frame[:60] + "…" + frame[-48:]


def collect(path: str, leaf: dict, parts: dict, ours: dict) -> int:
    """Suma las pilas de un archivo de py-spy («raíz;…;hoja cantidad» por línea). Devuelve cuántas muestras tenía."""
    total = 0
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.match(r"^(.*) (\d+)$", line.rstrip())
            if not m:
                continue
            stack, n = m.group(1).split(";"), int(m.group(2))
            total += n
            leaf[short(stack[-1])] += n
            parts[part_of(stack)] += n
            for fr in set(stack):
                if "core/" in fr or "agent/" in fr:
                    ours[short(fr)] += n
    return total


def part_of(stack: list[str]) -> str:
    for frame in reversed(stack):  # de adentro hacia afuera
        for name, marks in PARTS:
            if any(m in frame for m in marks):
                return name
    return "otro"


def main() -> None:
    seconds = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].isdigit() else "20"
    pids = analysis_pids()
    if not pids:
        print("No encontré procesos del análisis (¿está prendido «vigia-maquina@1»?).")
        return
    tmp = tempfile.mkdtemp(prefix="perfil-")
    try:
        spy = get_pyspy(tmp)
        if not spy:
            print("No pude bajar py-spy (¿sin internet o sin pip?). Probá: sudo /opt/vigia/.venv/bin/pip install py-spy")
            return
        print(f"== {len(pids)} procesos · mirando {seconds} s (no frena las cámaras)…", flush=True)
        procs = []
        for pid in pids:  # todos a la vez, sin pausarlos
            out = os.path.join(tmp, f"{pid}.txt")
            procs.append((out, subprocess.Popen(
                [spy, "record", "--pid", str(pid), "--duration", seconds, "--rate", "50", "--format", "raw",
                 "--output", out, "--nonblocking"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)))
        leaf: dict[str, int] = defaultdict(int)
        parts: dict[str, int] = defaultdict(int)
        ours: dict[str, int] = defaultdict(int)
        total, errors = 0, []
        for out, p in procs:
            err = p.communicate(timeout=int(seconds) + 90)[1]
            got = collect(out, leaf, parts, ours)
            total += got
            if not got:
                errors.append(((err or "").strip().splitlines() or ["sin datos"])[-1][:120])
        if not total:
            print("No se pudo tomar ninguna muestra:", "; ".join(errors))
            return
        print(f"\n== Por parte del programa ({total} muestras = 100 % del procesador que gastan los hilos de Python)")
        for name, n in sorted(parts.items(), key=lambda kv: -kv[1]):
            print(f"   {100 * n / total:5.1f} %  {name}")
        print("\n== Dónde está parado cuando gasta (función de más adentro)")
        for name, n in sorted(leaf.items(), key=lambda kv: -kv[1])[:26]:
            print(f"   {100 * n / total:5.1f} %  {name}")
        print("\n== Funciones nuestras (con todo lo que llaman)")
        for name, n in sorted(ours.items(), key=lambda kv: -kv[1])[:20]:
            print(f"   {100 * n / total:5.1f} %  {name}")
        # Un solo proceso, pocos segundos, con las funciones de adentro de las librerías (placa de video, decodificador):
        # dice si el tiempo «en el modelo» es trabajo o es esperar a la placa dando vueltas. Esta pasada sí pausa al
        # proceso un instante por muestra: por eso es corta, lenta y de uno solo.
        out = os.path.join(tmp, "nativo.txt")
        r = run([spy, "record", "--pid", str(pids[0]), "--duration", "10", "--rate", "12", "--format", "raw",
                 "--output", out, "--native"], timeout=120)
        nleaf: dict[str, int] = defaultdict(int)
        n_total = collect(out, nleaf, defaultdict(int), defaultdict(int))
        if n_total:
            libs: dict[str, int] = defaultdict(int)
            named: dict[str, int] = defaultdict(int)
            for name, n in nleaf.items():
                m = re.search(r"\(([^()]*)\)\s*$", name)
                lib = re.sub(r"\.cpython.*|\.so[.\d]*$|:\d+$", "", os.path.basename(m.group(1))) if m else "(python)"
                libs[lib] += n
                if not name.startswith("0x"):
                    named[name] += n
            print(f"\n== Por dentro de las librerías (proceso {pids[0]}, {n_total} muestras): en cuál está parado")
            for name, n in sorted(libs.items(), key=lambda kv: -kv[1])[:14]:
                print(f"   {100 * n / n_total:5.1f} %  {name}")
            print("   … y las funciones con nombre que más aparecen")
            for name, n in sorted(named.items(), key=lambda kv: -kv[1])[:14]:
                print(f"   {100 * n / n_total:5.1f} %  {name}")
        else:
            print("\n(sin el detalle de las librerías: "
                  + ((r.stderr or "").strip().splitlines() or ["sin datos"])[-1][:120] + ")")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
