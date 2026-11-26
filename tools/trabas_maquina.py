"""Dónde ESPERAN los procesos de análisis (26 nov: al sumar las cámaras de Mayorista Alberto, partes de la máquina
bajaron a 1-3 cuadros/s usando MENOS procesador que antes: algo las tiene esperando, y `perfil_maquina.py` solo mira lo
que gasta procesador). Esto mira el reloj: en qué función está parado cada hilo, gaste o no.

Se corre EN LA MÁQUINA con el análisis andando. No frena las cámaras ni cambia nada: baja `py-spy` a una carpeta
temporal y la borra al terminar.

    curl -fsSL https://raw.githubusercontent.com/santiagcaceres/vigia/main/tools/trabas_maquina.py -o /tmp/trabas.py
    sudo python3 /tmp/trabas.py            # 15 segundos; «sudo python3 /tmp/trabas.py 30» para 30

Muestra: la placa de video (uso por segundo y por proceso), la memoria, si el sistema dice que hay procesos esperando
procesador, disco o memoria, y por cada proceso de análisis en qué pasan el tiempo los hilos de las cámaras (esperando el
cuadro del video, esperando al detector o a la postura, analizando…). No imprime claves ni direcciones: solo nombres de
funciones y archivos."""

from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict

PIP_CANDIDATES = ("/opt/vigia/.venv/bin/pip", "pip3", "pip")
# en qué está un hilo: la primera marca que aparezca en su pila, de adentro hacia afuera
PARTS = (
    ("esperando el resultado del detector o de la postura (cola de lotes)", ("run_many", "_submit", "run (core/detectors")),
    ("en el modelo (hilo de lotes: la placa)", ("_loop (core/detectors", "_run_batch", "Session.run", "onnxruntime")),
    ("postura", ("estimate_many", "estimate (", "look (core/pose")),
    ("detector: preparar y leer el resultado", ("_letterbox", "_decode", "_infer", "detect_also", "detect_groups", "detect (")),
    ("escribir el clip (mp4)", ("write_mp4", "_finish_clips", "clip_timeline")),
    ("bajar el clip del grabador", ("download_nvr_clip", "nvr")),
    ("subir al servidor (alertas, clips, fotos)", ("outbox.py", "_upload", "upload_clip", "httpx", "httpcore")),
    ("seguimiento", ("tracker.py",)),
    ("gestos, reglas e IA de hurtos", ("gestos.py", "rules.py", "hurto_ia.py")),
    ("métricas", ("metrics.py",)),
    ("guardar el cuadro para el clip (JPEG)", ("_small_jpg", "_small (", "_remember", "idle (core/analyzer")),
    ("filtro de movimiento", ("motion.py",)),
    ("imagen en vivo y vista previa", ("live.py", "_send_preview")),
    ("abrir el video (conectando)", ("open_video", "open_failure", "rtsp_diagnose", "diagnose", "_open")),
    ("leer el video: esperar y decodificar (grab)", ("grab",)),
    ("leer el video: pasar a imagen (retrieve)", ("retrieve", "square_pixels")),
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
        if "-m agent" in cmd and "trabas" not in cmd and "perfil" not in cmd:
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
    frame = re.sub(r"\((?:/[^ )]*/)?(?:site-packages/|vigia/)", "(", frame.strip())
    return frame if len(frame) <= 100 else frame[:54] + "…" + frame[-44:]


# esperas de la biblioteca de Python (un candado, una cola): no dicen QUÉ se espera; eso lo dice quien las llamó
GENERIC = ("(threading.py", "(queue.py", "concurrent/futures", "(selectors.py", "(socket.py", "(ssl.py")


def part_of(stack: list[str]) -> str:
    for frame in reversed(stack):  # de adentro hacia afuera
        if any(g in frame for g in GENERIC):
            continue
        for name, marks in PARTS:
            if any(m in frame for m in marks):
                return name
    for frame in reversed(stack):
        if not any(g in frame for g in GENERIC):
            return "otro: " + short(frame)
    return "esperando (un candado o una cola de Python)"


def read_text(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def top_processes(span: float = 2.0, n: int = 8) -> list[tuple[float, str, int]]:
    """Los programas que más procesador gastan en toda la máquina (en procesadores enteros), mirando `span` segundos."""
    def snap() -> dict[int, tuple[str, float]]:
        out = {}
        for path in glob.glob("/proc/[0-9]*/stat"):
            try:
                with open(path, "rb") as f:
                    data = f.read()
                left, right = data.index(b"("), data.rindex(b")")
                rest = data[right + 2:].split()
                out[int(path.split("/")[2])] = (data[left + 1:right].decode(errors="replace"), int(rest[11]) + int(rest[12]))
            except (OSError, ValueError, IndexError):
                continue
        return out
    hz = float(os.sysconf("SC_CLK_TCK"))
    a = snap()
    time.sleep(span)
    b = snap()
    used = [((ticks - a[pid][1]) / hz / span, name, pid) for pid, (name, ticks) in b.items() if pid in a]
    by_name: dict[str, list[float]] = defaultdict(list)
    for cores, name, _pid in used:
        by_name[name].append(cores)
    rows = [(sum(v), f"{name} ×{len(v)}" if len(v) > 1 else name, len(v)) for name, v in by_name.items()]
    return sorted((r for r in rows if r[0] >= 0.02), reverse=True)[:n]


def system_report(seconds: int) -> None:
    load = read_text("/proc/loadavg").split()
    print(f"== Sistema: {os.cpu_count()} procesadores · carga {' '.join(load[:3])}")
    mem = {m.group(1): int(m.group(2)) // 1024 for m in re.finditer(r"^(\w+):\s+(\d+) kB", read_text("/proc/meminfo"), re.M)}
    if mem:
        print(f"   memoria: {mem.get('MemTotal', 0)} MB en total · {mem.get('MemAvailable', 0)} MB disponibles · "
              f"intercambio usado {mem.get('SwapTotal', 0) - mem.get('SwapFree', 0)} MB")
    for what, name in (("cpu", "procesador"), ("io", "disco"), ("memory", "memoria")):
        m = re.search(r"some avg10=([\d.]+) avg60=([\d.]+)", read_text(f"/proc/pressure/{what}"))
        if m:  # parte del tiempo en que algún proceso estuvo esperando ese recurso
            print(f"   esperando {name}: {m.group(1)} % (últimos 10 s) · {m.group(2)} % (último minuto)")
    top = top_processes()
    if top:
        print("   los programas que más procesador gastan (procesadores enteros; «×N» = N procesos con ese nombre):")
        for cores, name, _n in top:
            print(f"      {cores:5.2f}  {name}")
    if not shutil.which("nvidia-smi"):
        print("   (sin nvidia-smi: no hay placa de video o falta el driver)")
        return
    samples = []
    for _ in range(max(3, min(seconds, 8))):
        r = run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                 "--format=csv,noheader,nounits"], timeout=15)
        if r.returncode == 0 and r.stdout.strip():
            samples.append([v.strip() for v in r.stdout.strip().splitlines()[0].split(",")])
        time.sleep(1)
    if samples:
        use = [int(float(s[0])) for s in samples]
        last = samples[-1]
        print(f"== Placa de video: uso {' '.join(f'{u} %' for u in use)}  (promedio {sum(use) / len(use):.0f} %) · "
              f"memoria {last[1]} de {last[2]} MB · {last[3]} °C · {last[4]} W")
    r = run(["nvidia-smi", "pmon", "-c", "3", "-s", "um"], timeout=30)
    if r.returncode == 0:
        print("   por proceso (sm = % de la placa, mem = % de su memoria; 3 muestras):")
        for line in r.stdout.splitlines():
            if line.strip() and not line.startswith("#") or "pid" in line:
                print("   " + line.rstrip())


def main() -> None:
    seconds = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 15
    pids = analysis_pids()
    if not pids:
        print("No encontré procesos del análisis (¿está prendido «vigia-maquina@1»?).")
        return
    tmp = tempfile.mkdtemp(prefix="trabas-")
    try:
        spy = get_pyspy(tmp)
        procs = []
        if spy:
            for pid in pids:  # todos a la vez, sin pausarlos, también los hilos que esperan (--idle)
                out = os.path.join(tmp, f"{pid}.txt")
                procs.append((pid, out, subprocess.Popen(
                    [spy, "record", "--pid", str(pid), "--duration", str(seconds), "--rate", "20", "--format", "raw",
                     "--output", out, "--nonblocking", "--idle", "--threads"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)))
        system_report(seconds)
        if not spy:
            print("\nNo pude bajar py-spy (¿sin internet o sin pip?): falta la parte de los hilos.")
            return
        for pid, out, p in procs:
            err = p.communicate(timeout=seconds + 90)[1]
            by_thread: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
            leaf: dict[str, int] = defaultdict(int)
            for line in read_text(out).splitlines():
                m = re.match(r"^(.*) (\d+)$", line.rstrip())
                if not m:
                    continue
                stack, n = m.group(1).split(";"), int(m.group(2))
                thread = "?"
                if stack and stack[0].startswith("thread"):  # «thread (0x7f…): nombre» con --threads
                    thread = re.sub(r"[-_ ]?\d+$", "", stack[0].split(":", 1)[-1].strip() or "?") or "?"
                    thread = re.sub(r"^thread \(0x[0-9a-f]+\)$", "?", thread)
                    stack = stack[1:]
                if not stack:
                    continue
                by_thread[thread][part_of(stack)] += n
                leaf[short(stack[-1])] += n
            comm = read_text(f"/proc/{pid}/cmdline").replace("\0", " ")
            num = re.search(r"--proceso[= ](\d+)", comm)
            total = sum(sum(d.values()) for d in by_thread.values())
            print(f"\n== Proceso {pid}" + (f" (parte {num.group(1)})" if num else "") + f": {total} muestras")
            if not total:
                print("   sin datos:", ((err or "").strip().splitlines() or ["?"])[-1][:140])
                continue
            groups = sorted(by_thread.items(), key=lambda kv: -sum(kv[1].values()))
            for thread, parts in groups[:7]:
                n = sum(parts.values())
                print(f"   hilos «{thread}» ({100 * n / total:.0f} % de las muestras):")
                for name, k in sorted(parts.items(), key=lambda kv: -kv[1])[:6]:
                    print(f"      {100 * k / n:5.1f} %  {name}")
            print("   función de más adentro (todos los hilos):")
            for name, k in sorted(leaf.items(), key=lambda kv: -kv[1])[:8]:
                print(f"      {100 * k / total:5.1f} %  {name}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
