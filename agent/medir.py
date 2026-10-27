"""Cuántas cámaras aguanta esta máquina: `python -m agent.medir` (`--segundos 20`, `--nucleos 2` para usar menos).

Mide por separado lo que gasta cada cámara, con el mismo código que usan las máquinas de análisis:
- **analizar** los cuadros (detector de personas + seguimiento + reglas + guardar el cuadro para el clip): 5 por
  segundo en una cámara de hurto. En procesador usa una sesión de 1 hilo por núcleo (`VIGIA_HILOS`); con placa de
  video, la placa;
- **decodificar** el video: todos los cuadros que manda el grabador (8 por segundo en el substream recomendado).
Con eso estima cuántas cámaras de hurto entran, dejando un 30 % de margen. No usa cámaras ni internet. El video de
prueba es MPEG-4 de 640x360; el H.264 de los grabadores suele costar algo más al decodificar.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ANALYZE_FPS = 5.0  # cámaras de hurto
DECODE_FPS = 8.0  # substream recomendado
MARGIN = 0.7  # 30 % libre para lo que no se mide (red, subir clips, métricas, picos)


def test_video(path: Path, seconds: int = 10, fps: int = 8) -> Path:
    """Un video de 640x360 con «personas» que se mueven (el detector trabaja igual con o sin gente real)."""
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 360))
    rng = np.random.default_rng(0)
    base = rng.integers(40, 90, (360, 640, 3), dtype=np.uint8)
    for i in range(seconds * fps):
        img = base.copy()
        for k in range(3):
            x = int((i * 7 + k * 200) % 600)
            cv2.rectangle(img, (x, 120 + k * 40), (x + 30, 200 + k * 40), (150, 140, 120), -1)
        vw.write(img)
    vw.release()
    return path


def frames_of(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out


def run_threads(n: int, seconds: float, work) -> float:
    """`n` hilos haciendo `work(i, contador)` durante `seconds`; devuelve cuántas veces por segundo en total."""
    counts = [0] * n
    stop = time.monotonic() + seconds

    def loop(i: int) -> None:
        while time.monotonic() < stop:
            work(i)
            counts[i] += 1
    ts = [threading.Thread(target=loop, args=(i,), daemon=True) for i in range(n)]
    t0 = time.monotonic()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return sum(counts) / (time.monotonic() - t0)


def measure(model: str, seconds: float = 15.0, cores: int | None = None) -> dict:
    if cores and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:cores]))
    ncpu = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    import onnxruntime as ort

    nvidia = any(p in ort.get_available_providers() for p in ("CUDAExecutionProvider", "TensorrtExecutionProvider"))
    if not nvidia:  # como en las máquinas sin placa: una sesión de 1 hilo por núcleo (con placa: lotes en la placa)
        os.environ["VIGIA_HILOS"] = str(ncpu)
    cv2.setNumThreads(1)
    from core.analyzer import Analyzer
    from core.detectors import build_detector
    from core.synthetic import STORE_ZONES

    with tempfile.TemporaryDirectory() as tmp:
        video = test_video(Path(tmp) / "prueba.mp4")
        frames = frames_of(video)
        det = build_detector("yolo", model)
        det.detect(frames[0])  # la primera vez arma el modelo
        gpu = bool(getattr(det, "on_gpu", False))
        workers = max(ncpu, 16 if gpu else ncpu)  # con placa hacen falta muchas cámaras a la vez para llenarla
        analyzers = [Analyzer({"zones": STORE_ZONES, "rules": {}}, det) for _ in range(workers)]
        pos = [0] * workers

        def analyze(i: int) -> None:
            pos[i] += 1
            analyzers[i].process(frames[pos[i] % len(frames)], pos[i] / ANALYZE_FPS)
        analyzed = run_threads(workers, seconds, analyze)

        caps = [cv2.VideoCapture(str(video)) for _ in range(ncpu)]

        def decode(i: int) -> None:
            if not caps[i].grab():
                caps[i].set(cv2.CAP_PROP_POS_FRAMES, 0)
        decoded = run_threads(ncpu, seconds, decode)
        for c in caps:
            c.release()

    if gpu:  # detecta la placa; el procesador decodifica y hace el resto
        cams = MARGIN * min(analyzed / ANALYZE_FPS, decoded / DECODE_FPS)
    else:  # todo en el procesador: analizar y decodificar se reparten los mismos núcleos
        cams = MARGIN / (ANALYZE_FPS / analyzed + DECODE_FPS / decoded)
    return {"cores": ncpu, "device": getattr(det, "device", "?"), "analyzed_fps": analyzed,
            "decoded_fps": decoded, "cameras": cams}


def main() -> None:
    p = argparse.ArgumentParser(description="Cuántas cámaras de hurto aguanta esta máquina")
    p.add_argument("--modelo", default=str(Path(__file__).resolve().parent.parent / "models" / "personas.onnx"))
    p.add_argument("--segundos", type=float, default=15.0)
    p.add_argument("--nucleos", type=int, default=None, help="usar solo estos núcleos (para comparar tamaños)")
    a = p.parse_args()
    r = measure(a.modelo, a.segundos, a.nucleos)
    print(f"Núcleos: {r['cores']} · detector en {r['device']}")
    print(f"Análisis completo (detector + seguimiento + reglas + clip): {r['analyzed_fps']:.1f} cuadros por segundo")
    print(f"Decodificar video 640x360: {r['decoded_fps']:.0f} cuadros por segundo")
    print(f"→ Entran unas {r['cameras']:.0f} cámaras de hurto (5 cuadros por segundo analizados, 8 por segundo de "
          f"video, 30 % de margen)")


if __name__ == "__main__":
    main()
