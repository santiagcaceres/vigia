"""En qué gasta procesador cada ffmpeg que lee el video por la placa (26 nov: con las 4 partes de la máquina de AWS
leyendo por la placa, el video siguió costando lo mismo, ~0,026 procesadores por cámara, sea chica o grande: parece un
costo fijo por ffmpeg y no lo que decodifica). Mira cada hilo de cada ffmpeg un rato y dice cuánto gasta, cuántas
veces se despierta y qué dice la placa (uso, decodificador, memoria).

Se corre EN LA MÁQUINA con el análisis andando; no frena nada ni cambia nada:

    curl -fsSL https://raw.githubusercontent.com/santiagcaceres/vigia/main/tools/ffmpeg_placa.py -o /tmp/fp.py
    sudo python3 /tmp/fp.py            # 15 segundos; «sudo python3 /tmp/fp.py 30» para 30

No imprime links ni claves: de la línea de comando de cada ffmpeg solo el decodificador y el tamaño."""

from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict

HZ = float(os.sysconf("SC_CLK_TCK"))
_CREDENTIALS = re.compile(r"(\w+://)[^/\s]*@")


def read(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def ffmpegs() -> dict[int, dict]:
    """Los ffmpeg que leen por la placa: decodificador y tamaño (nada más de la línea de comando)."""
    out = {}
    for p in glob.glob("/proc/[0-9]*"):
        if read(p + "/comm").strip() != "ffmpeg":
            continue
        args = read(p + "/cmdline").split("\0")
        if not any(a.endswith("_cuvid") for a in args):
            continue
        dec = next((a for a in args if a.endswith("_cuvid")), "?")
        size = args[args.index("-resize") + 1] if "-resize" in args else "?"
        out[int(p.rsplit("/", 1)[1])] = {"dec": dec, "size": size, "select": any("select=" in a for a in args)}
    return out


def threads(pid: int) -> dict[int, tuple[str, float, float, int, int]]:
    """Por hilo: nombre, segundos de usuario y de sistema, veces que se durmió solo y que lo sacaron."""
    out = {}
    for t in glob.glob(f"/proc/{pid}/task/[0-9]*"):
        stat = read(t + "/stat")
        if not stat:
            continue
        rest = stat[stat.rindex(")") + 2:].split()
        status = read(t + "/status")
        vol = re.search(r"^voluntary_ctxt_switches:\s+(\d+)", status, re.M)
        inv = re.search(r"^nonvoluntary_ctxt_switches:\s+(\d+)", status, re.M)
        name = re.sub(r"\d+$", "", read(t + "/comm").strip()) or "?"
        out[int(t.rsplit("/", 1)[1])] = (name, int(rest[11]) / HZ, int(rest[12]) / HZ,
                                         int(vol.group(1)) if vol else 0, int(inv.group(1)) if inv else 0)
    return out


def gpu_report() -> None:
    if not shutil.which("nvidia-smi"):
        print("(sin nvidia-smi)")
        return
    rows = []
    for _ in range(5):
        r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,utilization.decoder,utilization.encoder,"
                            "memory.used,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=15, check=False)
        if r.returncode == 0 and r.stdout.strip():
            rows.append([v.strip() for v in r.stdout.strip().splitlines()[0].split(",")])
        time.sleep(1)
    if rows:
        col = lambda i: " ".join(f"{r[i]} %" for r in rows)  # noqa: E731
        print(f"== Placa: cálculo {col(0)} · decodificador de video {col(1)} · compresor {col(2)} · "
              f"memoria {rows[-1][3]} de {rows[-1][4]} MB")
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,"
                        "nounits"], capture_output=True, text=True, timeout=15, check=False)
    by: dict[str, list[int]] = defaultdict(list)
    for line in r.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3 and parts[2].isdigit():
            by[os.path.basename(parts[1])].append(int(parts[2]))
    for name, mem in sorted(by.items()):
        print(f"   {name}: {len(mem)} programas, {sum(mem)} MB (promedio {sum(mem) / len(mem):.0f} MB)")


def main() -> None:
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
    procs = ffmpegs()
    if not procs:
        print("No hay ningún ffmpeg leyendo por la placa (¿VIGIA_VIDEO_PLACA=si?).")
        return
    load = read("/proc/loadavg").split()[:3]
    print(f"== {len(procs)} ffmpeg leyendo por la placa · carga {' '.join(load)} · mirando {seconds:.0f} s")
    a = {pid: threads(pid) for pid in procs}
    t0 = time.monotonic()
    time.sleep(seconds)
    b = {pid: threads(pid) for pid in procs}
    span = time.monotonic() - t0
    per_proc = []
    by_name: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0])
    for pid, info in procs.items():
        tot = 0.0
        for tid, (name, u1, s1, v1, i1) in b.get(pid, {}).items():
            if tid not in a.get(pid, {}):
                continue
            _n, u0, s0, v0, i0 = a[pid][tid]
            acc = by_name[name]
            acc[0] += (u1 - u0) / span
            acc[1] += (s1 - s0) / span
            acc[2] += (v1 - v0) / span
            acc[3] += (i1 - i0) / span
            acc[4] += 1
            tot += (u1 - u0 + s1 - s0) / span
        per_proc.append((tot, pid, info))
    total = sum(t for t, _p, _i in per_proc)
    user = sum(v[0] for v in by_name.values())
    print(f"   en total {total:.2f} procesadores ({total / len(procs):.3f} por ffmpeg): usuario {user:.2f}, "
          f"sistema {total - user:.2f}")
    print("   por hilo (sumando todos los ffmpeg): procesadores · usuario/sistema · se despierta por segundo · "
          "hilos")
    for name, (u, s, v, i, n) in sorted(by_name.items(), key=lambda kv: -(kv[1][0] + kv[1][1])):
        print(f"      {name:<18} {u + s:5.2f}  ({u:.2f}/{s:.2f})  {v + i:7.0f}/s  ×{n}")
    print("   los que más gastan:")
    for tot, pid, info in sorted(per_proc, key=lambda x: -x[0])[:8]:
        print(f"      {pid:>7}  {tot:.3f}  {info['dec']} {info['size']}{' · baja algunos' if info['select'] else ''}")
    gpu_report()
    if shutil.which("journalctl"):
        r = subprocess.run("journalctl -u 'vigia-maquina@*' --since '-40min' -o cat --no-pager | "
                           "grep -iE 'placa|vuelve a mirar' | tail -15", shell=True, capture_output=True, text=True,
                           timeout=30, check=False)
        lines = [_CREDENTIALS.sub(r"\1***@", line)[:200] for line in r.stdout.splitlines() if line.strip()]
        if lines:
            print("== Registro (lo de la placa, últimos 40 min):")
            for line in lines:
                print("   " + line)


if __name__ == "__main__":
    main()
