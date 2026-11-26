"""¿Cuánto le ahorra TensorRT a la placa de video? (26 nov: con el video ya repartido entre la placa y el procesador, lo
que sigue para que entren más cámaras por máquina es que el detector de personas gaste menos placa: hoy corre con CUDA
en precisión completa; TensorRT arma un «motor» optimizado para esa placa y, en media precisión, suele andar 2-3 veces
más rápido). Esto lo mide EN LA MÁQUINA, en un proceso aparte: no toca las cámaras (comparte la placa unos minutos).

    curl -fsSL https://raw.githubusercontent.com/santiagcaceres/vigia/main/tools/tensorrt_prueba.py -o /tmp/trt.py
    cd /opt/vigia && sudo -u vigia -H nice -n 10 /opt/vigia/.venv/bin/python /tmp/trt.py --instalar

`--instalar` baja las librerías de TensorRT de NVIDIA (pip, ~2 GB) a /var/tmp/vigia-trt si no están; la primera vez
arma el motor (unos minutos) y lo deja en la carpeta que usaría el programa (`VIGIA_TRT_CACHE` o
~/.cache/vigia-tensorrt), así después prenderlo es al instante. Compara, con cuadros del video de prueba
(/tmp/prueba.mp4) o imágenes inventadas si no está: milisegundos por cuadro con lotes de 1, 4 y 8 y si las personas que
encuentra son las mismas."""

from __future__ import annotations

import argparse
import ctypes
import glob
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

TRT_DIR = Path("/var/tmp/vigia-trt")
LIBS = ("libnvinfer.so.10", "libnvinfer_plugin.so.10", "libnvonnxparser.so.10")


def find_trt_libs() -> Path | None:
    for base in [TRT_DIR, *map(Path, sys.path)]:
        d = base / "tensorrt_libs"
        if (d / LIBS[0]).exists():
            return d
    return None


def install_trt() -> Path | None:
    print("Bajando las librerías de TensorRT (puede tardar unos minutos)…", flush=True)
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--target", str(TRT_DIR),
                        "tensorrt-cu12-libs>=10,<11"], capture_output=True, text=True, check=False)
    if r.returncode != 0:
        print("No se pudo bajar TensorRT:", (r.stderr.strip().splitlines() or ["?"])[-1][:300])
        return None
    return find_trt_libs()


def preload(libdir: Path) -> None:
    for name in LIBS:
        path = libdir / name
        if path.exists():
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    os.environ["LD_LIBRARY_PATH"] = f"{libdir}:{os.environ.get('LD_LIBRARY_PATH', '')}"


def letterbox(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    import cv2
    h, w = frame.shape[:2]
    tw, th = size
    r = min(tw / w, th / h)
    nw, nh = max(1, int(round(w * r))), max(1, int(round(h * r)))
    canvas = np.full((th, tw, 3), 114, np.uint8)
    px, py = (tw - nw) // 2, (th - nh) // 2
    canvas[py:py + nh, px:px + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas


def frames(n: int, size: tuple[int, int]) -> tuple[list[np.ndarray], str]:
    import cv2
    out = []
    if os.path.exists("/tmp/prueba.mp4"):
        cap = cv2.VideoCapture("/tmp/prueba.mp4")
        i = 0
        while len(out) < n:
            ok, f = cap.read()
            if not ok:
                break
            if i % 20 == 0:
                out.append(letterbox(f, size))
            i += 1
        cap.release()
    if len(out) >= 4:
        return out, f"{len(out)} cuadros del video de prueba"
    rng = np.random.default_rng(1)
    return [rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8) for _ in range(n)], "imágenes inventadas"


def people(out: np.ndarray, col: int, conf: float = 0.35) -> list[tuple]:
    import cv2
    o = out.T if out.shape[0] < out.shape[1] else out  # (N, 4+C)
    scores = o[:, col]
    keep = scores >= conf
    boxes = o[keep, :4]
    sc = scores[keep]
    xywh = [[float(cx - w / 2), float(cy - h / 2), float(w), float(h)] for cx, cy, w, h in boxes]
    idx = cv2.dnn.NMSBoxes(xywh, sc.tolist(), conf, 0.45) if xywh else []
    return [(*xywh[i], float(sc[i])) for i in np.array(idx).flatten()]


def iou(a: tuple, b: tuple) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    return inter / max(a[2] * a[3] + b[2] * b[3] - inter, 1e-9)


def session(trt: bool):
    os.environ["VIGIA_TENSORRT"] = "1" if trt else "0"
    from core.detectors import OnnxSession
    t0 = time.monotonic()
    s = OnnxSession(Path("models/personas.onnx"), threads=1, max_batch=16)
    return s, time.monotonic() - t0


def bench(s, canvases: list[np.ndarray], batch: int, runs: int) -> tuple[float, float]:
    """(ms de reloj por cuadro, ms de procesador por cuadro) con lotes de `batch`."""
    arr = np.stack([canvases[i % len(canvases)] for i in range(batch)])
    for _ in range(5):
        s.session.run(None, {s.input_name: arr})
    w0, c0 = time.perf_counter(), time.thread_time()
    for _ in range(runs):
        s.session.run(None, {s.input_name: arr})
    w, c = time.perf_counter() - w0, time.thread_time() - c0
    return 1000 * w / (runs * batch), 1000 * c / (runs * batch)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instalar", action="store_true", help="bajar TensorRT si no está")
    ap.add_argument("--vueltas", type=int, default=40)
    args = ap.parse_args()
    import onnxruntime as ort
    print(f"onnxruntime {ort.__version__} · {', '.join(ort.get_available_providers())}")
    if "TensorrtExecutionProvider" not in ort.get_available_providers():
        print("Este onnxruntime no trae TensorRT: no se puede probar.")
        return
    libdir = find_trt_libs() or (install_trt() if args.instalar else None)
    if not libdir:
        print("No están las librerías de TensorRT. Correlo con --instalar para bajarlas.")
        return
    preload(libdir)
    print("TensorRT:", libdir)
    cuda, t_cuda = session(False)
    print(f"CUDA: {cuda.device}, armado en {t_cuda:.1f} s")
    print("Armando el motor de TensorRT (la primera vez, unos minutos)…", flush=True)
    trt, t_trt = session(True)
    print(f"TensorRT: {trt.device}, armado en {t_trt:.0f} s")
    if "Tensorrt" not in trt.device:
        print("No quedó en TensorRT (mirá los avisos de arriba): sigue en", trt.device)
        return
    canvases, what = frames(32, cuda.size)
    print(f"Con {what}:")
    print("   lote   CUDA ms/cuadro (procesador)   TensorRT ms/cuadro (procesador)")
    for b in (1, 4, 8):
        cw, cc = bench(cuda, canvases, b, args.vueltas)
        tw, tc = bench(trt, canvases, b, args.vueltas)
        print(f"   {b:>4}   {cw:8.2f} ({cc:5.2f})              {tw:8.2f} ({tc:5.2f})      ×{cw / tw:.1f} más rápido")
    col = cuda.col_of[0]
    same = total = 0
    ious: list[float] = []
    dconf: list[float] = []
    for c in canvases:
        a = people(cuda.session.run(None, {cuda.input_name: c[None]})[0][0], col)
        b = people(trt.session.run(None, {trt.input_name: c[None]})[0][0], col)
        total += 1
        same += len(a) == len(b)
        for pa in a:
            best = max(b, key=lambda pb: iou(pa, pb), default=None)
            if best:
                ious.append(iou(pa, best))
                dconf.append(abs(pa[4] - best[4]))
    print(f"Personas: misma cantidad en {same} de {total} cuadros", end="")
    if ious:
        print(f" · las mismas cajas (coincidencia promedio {np.mean(ious):.3f}, peor {min(ious):.3f}) · confianza "
              f"distinta en {np.mean(dconf):.3f} de promedio (peor {max(dconf):.3f})")
    else:
        print(" (no había personas en esos cuadros)")
    cache = os.getenv("VIGIA_TRT_CACHE") or str(Path.home() / ".cache" / "vigia-tensorrt")
    print("Motor guardado en", cache, f"({len(glob.glob(cache + '/*'))} archivos)")


if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    main()
