"""Cuántas cámaras aguanta esta máquina: `python -m agent.medir` (`--segundos 20`, `--nucleos 2` para usar menos;
`--numero` solo escribe la cantidad: así la instalación de una máquina de análisis le pone su capacidad medida;
`--gestos` suma lo que cuesta mirar la postura para los gestos de ocultamiento).

Mide por separado lo que gasta cada cámara, con el mismo código que usan las máquinas de análisis:
- **analizar** los cuadros (detector de personas + seguimiento + reglas + guardar el cuadro para el clip): 5 por
  segundo en una cámara de hurto. En procesador usa una sesión de 1 hilo por núcleo (`VIGIA_HILOS`); con placa de
  video, la placa;
- **decodificar** el video: todos los cuadros que manda el grabador (8 por segundo en el substream recomendado).
Con eso estima cuántas cámaras de hurto entran, dejando un 30 % de margen. No usa cámaras ni internet. El video de
prueba es MPEG-4 de 640x360; el H.264 de los grabadores suele costar algo más al decodificar.

Con `--gestos` mide además la **postura** (core/pose.py, YOLO11n-pose sobre el recorte de cada persona): cuánto tarda
por persona y cuántas por segundo da la máquina, y lo suma como si en cada cuadro analizado hubiera ~2 personas a
las que mirarles la postura (en el video de prueba no hay personas de verdad: se mide sobre recortes del cuadro, que
cuestan lo mismo porque el modelo siempre recibe 256x256). Con más gente en el local cuesta más.
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
POSE_PEOPLE = 2.0  # con gestos: a cuántas personas por cuadro se les mira la postura (promedio supuesto)
# recuadros de «personas» del tamaño de una real en el cuadro de 640x360 (para recortar y medir la postura)
POSE_BOXES = ((120.0, 30.0, 210.0, 340.0), (380.0, 60.0, 460.0, 330.0))


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


def measure_pose(est, frame: np.ndarray, workers: int, seconds: float) -> tuple[float, float]:
    """(milisegundos por persona, de a una; personas por segundo en total con `workers` hilos) de la postura."""
    est.estimate(frame, POSE_BOXES[0])  # la primera vez arma el modelo
    n, t0 = 0, time.perf_counter()
    while n < 5 or time.perf_counter() - t0 < min(3.0, seconds / 3):
        est.estimate(frame, POSE_BOXES[n % len(POSE_BOXES)])
        n += 1
    ms = 1000.0 * (time.perf_counter() - t0) / n
    rate = run_threads(workers, seconds, lambda i: est.estimate(frame, POSE_BOXES[i % len(POSE_BOXES)]))
    return ms, rate


def measure(model: str, seconds: float = 15.0, cores: int | None = None, gestures: bool = False,
            pose_model: str | None = None) -> dict:
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

        pose_ms = pose_rate = None
        if gestures:
            from core.pose import build_pose

            est = build_pose(pose_model)  # el mismo estimador que usan las máquinas (sin el modelo: None)
            if est is not None:
                pose_ms, pose_rate = measure_pose(est, frames[0], workers, seconds)

    def cameras(analyzed_fps: float) -> float:
        if gpu:  # detecta la placa; el procesador decodifica y hace el resto
            return MARGIN * min(analyzed_fps / ANALYZE_FPS, decoded / DECODE_FPS)
        # todo en el procesador: analizar y decodificar se reparten los mismos núcleos
        return MARGIN / (ANALYZE_FPS / analyzed_fps + DECODE_FPS / decoded)

    out = {"cores": ncpu, "device": getattr(det, "device", "?"), "analyzed_fps": analyzed,
           "decoded_fps": decoded, "cameras": cameras(analyzed)}
    if gestures:
        # cada cuadro analizado cuesta además la postura de ~2 personas (comparten los mismos núcleos o la placa)
        with_pose = 1.0 / (1.0 / analyzed + POSE_PEOPLE / pose_rate) if pose_rate else analyzed
        out.update(pose_ms=pose_ms, pose_fps=pose_rate, analyzed_fps_gestures=with_pose,
                   cameras_gestures=cameras(with_pose))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Cuántas cámaras de hurto aguanta esta máquina")
    p.add_argument("--modelo", default=str(Path(__file__).resolve().parent.parent / "models" / "personas.onnx"))
    p.add_argument("--segundos", type=float, default=15.0)
    p.add_argument("--nucleos", type=int, default=None, help="usar solo estos núcleos (para comparar tamaños)")
    p.add_argument("--numero", action="store_true",
                   help="solo el número de cámaras (lo usa deploy/maquina-analisis/instalar.sh)")
    p.add_argument("--gestos", action="store_true",
                   help="sumar la postura de ~2 personas por cuadro (gestos de ocultamiento); con --numero, el "
                        "número con gestos")
    p.add_argument("--postura", default=str(Path(__file__).resolve().parent.parent / "models" / "postura.onnx"),
                   help="modelo de postura (por defecto models/postura.onnx)")
    a = p.parse_args()
    r = measure(a.modelo, a.segundos, a.nucleos, gestures=a.gestos, pose_model=a.postura)
    if a.numero:
        print(max(1, int(r["cameras_gestures"] if a.gestos else r["cameras"])))
        return
    print(f"Núcleos: {r['cores']} · detector en {r['device']}")
    print(f"Análisis completo (detector + seguimiento + reglas + clip): {r['analyzed_fps']:.1f} cuadros por segundo")
    print(f"Decodificar video 640x360: {r['decoded_fps']:.0f} cuadros por segundo")
    print(f"→ Entran unas {r['cameras']:.0f} cámaras de hurto (5 cuadros por segundo analizados, 8 por segundo de "
          f"video, 30 % de margen)")
    if a.gestos and r["pose_fps"] is None:
        print(f"Gestos: no está el modelo de postura ({a.postura}); esta máquina analizaría sin gestos")
    elif a.gestos:
        print(f"Postura (gestos): {r['pose_ms']:.0f} ms por persona de a una; {r['pose_fps']:.0f} personas por "
              f"segundo en total")
        print(f"Con gestos (~{POSE_PEOPLE:.0f} personas por cuadro): {r['analyzed_fps_gestures']:.1f} cuadros por "
              f"segundo de análisis completo")
        print(f"→ Con gestos entran unas {r['cameras_gestures']:.0f} cámaras de hurto (con más gente en el local, "
              f"menos)")


if __name__ == "__main__":
    main()
