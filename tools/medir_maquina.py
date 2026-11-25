"""En qué se va el procesador de una máquina de análisis (24 nov: la g4dn.xlarge de AWS midió 17 cámaras y el límite es
el procesador, no la placa; antes de optimizar hay que saber cuánto es decodificar el video y cuánto es lo demás).

Se corre EN LA MÁQUINA, con el análisis andando; no instala ni cambia nada y no frena las cámaras:

    curl -fsSL https://raw.githubusercontent.com/santiagcaceres/vigia/main/tools/medir_maquina.py -o /tmp/medir.py
    sudo python3 /tmp/medir.py            # 30 segundos; «sudo python3 /tmp/medir.py 60» para un minuto

Mira dos veces el tiempo de procesador de cada hilo de los procesos del análisis (/proc/<pid>/task/*/stat) y los junta
por nombre: los hilos con que FFmpeg decodifica se llaman «av:h264:df0», «av:hevc:df1»…; los de Python (leer, achicar,
comprimir los cuadros de los clips, seguir a las personas, reglas) quedan como «python». Muestra también cómo está el
sistema entero y la placa de video. No imprime claves (del archivo de configuración solo saca lo que no es secreto).

Desde el 25 nov cada cámara decodifica su video con un solo hilo (el suyo, de Python): ya no hay hilos «av:…» salvo en
videos enormes, y lo de decodificar queda sumado en «python». El reparto de verdad lo informa cada proceso (agent/
consumo.py) y se ve en la app, pestaña «Máquinas», al elegir la máquina."""

from __future__ import annotations

import glob
import os
import re
import subprocess
import sys
import time
from collections import defaultdict

HZ = os.sysconf("SC_CLK_TCK")
SAFE_ENV = ("VIGIA_CAPACIDAD", "VIGIA_HILOS", "VIGIA_LOTE", "VIGIA_FIJA", "VIGIA_TENSORRT", "VIGIA_HILOS_VIDEO",
            "VIGIA_HILOS_OPENCV", "VIGIA_LOTE_POSTURA", "VIGIA_PLACA_ESPERA")  # nada de claves


def read(path: str, binary: bool = False):
    with open(path, "rb" if binary else "r") as f:
        return f.read()


def analysis_pids(pattern: str) -> list[int]:
    out = []
    for p in glob.glob("/proc/[0-9]*"):
        try:
            cmd = read(p + "/cmdline", binary=True).replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if pattern in cmd and "medir" not in cmd:
            out.append(int(p.rsplit("/", 1)[1]))
    return sorted(out)


def threads(pid: int) -> dict[str, tuple[str, int]]:
    """hilo → (nombre, tics de procesador usados: usuario + núcleo)."""
    out = {}
    for t in glob.glob(f"/proc/{pid}/task/*"):
        try:
            s = read(t + "/stat")
        except OSError:
            continue
        name = s[s.index("(") + 1:s.rindex(")")]
        f = s[s.rindex(")") + 2:].split()
        out[t.rsplit("/", 1)[1]] = (name, int(f[11]) + int(f[12]))
    return out


def process_ticks(pid: int) -> int:
    """Tics de todo el proceso (cuenta también los hilos que ya terminaron, p. ej. los de una cámara que se reconectó)."""
    try:
        s = read(f"/proc/{pid}/stat")
    except OSError:
        return 0
    f = s[s.rindex(")") + 2:].split()
    return int(f[11]) + int(f[12])


def system_cpu() -> list[int]:
    return [int(x) for x in read("/proc/stat").splitlines()[0].split()[1:9]]


def group_of(name: str) -> str:
    base = re.sub(r"[\d:]+$", "", name)
    if name.startswith("av:"):
        codec = name.split(":")[1] if ":" in name[3:] else "?"
        return f"decodificar el video ({codec})"
    if base.startswith("python"):
        return "python (leer, achicar, clips, seguimiento, reglas)"
    return base or name


def run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=False).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def main() -> None:
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].replace(".", "").isdigit() else 30.0
    pattern = sys.argv[2] if len(sys.argv) > 2 else "-m agent"
    pids = analysis_pids(pattern)
    cores = os.cpu_count() or 1
    print(f"== Máquina: {cores} procesadores · {len(pids)} procesos de análisis · midiendo {seconds:.0f} s…", flush=True)
    if not pids:
        print("No encontré procesos del análisis (¿está prendido «vigia-maquina@1»?).")
        return
    before = {p: threads(p) for p in pids}
    whole0 = {p: process_ticks(p) for p in pids}
    sys0, t0 = system_cpu(), time.time()
    time.sleep(seconds)
    sys1, dt = system_cpu(), time.time() - t0
    after = {p: threads(p) for p in pids}
    whole1 = {p: process_ticks(p) for p in pids}

    d = [b - a for a, b in zip(sys0, sys1)]
    tot = sum(d) or 1
    names = ("usuario", "prioridad baja", "núcleo", "libre", "esperando disco", "irq", "red (softirq)", "robado")
    print("\n== El sistema entero")
    print("   " + " · ".join(f"{n} {100 * v / tot:.0f} %" for n, v in zip(names, d) if v * 200 >= tot or n == "libre"))
    print("   carga: " + " ".join(read("/proc/loadavg").split()[:3]))

    total_groups: dict[str, float] = defaultdict(float)
    grand = 0.0
    for p in pids:
        groups: dict[str, float] = defaultdict(float)
        per_thread = []
        for tid, (name, ticks) in after[p].items():
            used = (ticks - before[p].get(tid, (name, 0))[1]) / HZ / dt  # procesadores usados por este hilo
            if tid not in before[p]:
                used = ticks / HZ / dt  # (hilo nuevo: todo lo que gastó es de este rato)
            groups[group_of(name)] += used
            per_thread.append((used, name))
        gone = (whole1[p] - whole0[p]) / HZ / dt - sum(groups.values())
        if gone > 0.02:  # lo que gastaron hilos que ya no están (casi siempre, decodificar antes de una reconexión)
            groups["hilos que terminaron en este rato (reconexiones)"] += gone
        used_p = sum(groups.values())
        grand += used_p
        print(f"\n== Proceso {p}: {used_p:.2f} procesadores · {len(after[p])} hilos")
        for g, v in sorted(groups.items(), key=lambda kv: -kv[1]):
            if v >= 0.005:
                print(f"   {v:5.2f}  {100 * v / (used_p or 1):3.0f} %  {g}")
            total_groups[g] += v
        py = sorted((u for u, n in per_thread if n.startswith("python")), reverse=True)
        print("   hilos de python que más gastan: " + " · ".join(f"{100 * u:.0f} %" for u in py[:8]))
        print(f"   suma de los hilos de python: {100 * sum(py):.0f} % de un procesador")

    print(f"\n== Todo el análisis: {grand:.2f} de {cores} procesadores ({100 * grand / cores:.0f} %)")
    for g, v in sorted(total_groups.items(), key=lambda kv: -kv[1]):
        if v >= 0.005:
            print(f"   {v:5.2f}  {100 * v / (grand or 1):3.0f} %  {g}")

    gpu = run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader"])
    if gpu:
        print(f"\n== Placa de video: {gpu}")
        print(run(["nvidia-smi", "dmon", "-c", "3", "-s", "u"]))
    try:
        env = [ln.strip() for ln in read("/etc/vigia/maquina.env").splitlines() if ln.split("=")[0].strip() in SAFE_ENV]
        print("\n== Configuración: " + (" · ".join(env) or "(lo de fábrica)"))
    except OSError:
        pass
    slow = run(["journalctl", "-u", "vigia-maquina@*", "--since", "-15 min", "--no-pager", "-q", "-g", "no da abasto"])
    print(f"== Avisos de «no da abasto» en los últimos 15 minutos: {len(slow.splitlines()) if slow else 0}")


if __name__ == "__main__":
    main()
