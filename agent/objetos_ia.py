"""Lo de «Entrenar objetos» que necesita Ultralytics (y torch): corre APARTE del análisis de las cámaras, en otro proceso
que lanza agent/objetos_maquina.py, así una falla, la memoria de torch o un entrenamiento cancelado no tocan las cámaras.

    python -m agent.objetos_ia train   <config.json>
    python -m agent.objetos_ia predict <config.json>

Va contando lo que hace en renglones «@@VIGIA {json}» por la salida estándar (lo demás que escribe Ultralytics se ignora,
salvo para explicar un error). Solo en las máquinas instaladas con ENTRENAR=si (requirements-entrenar.txt); la app de
Windows no lo trae.

- **train**: afina YOLO11n (o la versión anterior del objeto) con las fotos marcadas, lo mide con las fotos de prueba y lo
  exporta a ONNX a 640 (lo mismo que el detector de personas: lo lee core/objetos.ObjectDetector).
- **predict**: propone recuadros en fotos sin marcar, con la versión activa del objeto o, si todavía no hay, con YOLOE y
  las palabras en inglés de cada clase («pallet»): una persona después los corrige.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

MARK = "@@VIGIA "
YOLOE_WEIGHTS = "yoloe-11s-seg.pt"


def emit(**data) -> None:
    sys.stdout.write(MARK + json.dumps(data, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def device_info() -> tuple[str | int, str, bool]:
    """(lo que se le pasa a Ultralytics, el nombre para mostrar, si es placa de video)."""
    import torch

    if torch.cuda.is_available():
        return 0, torch.cuda.get_device_name(0), True
    n = os.cpu_count() or 2
    torch.set_num_threads(max(1, n // 2))  # deja la mitad para el análisis de las cámaras
    try:
        os.nice(10)
    except (AttributeError, OSError):
        pass
    return "cpu", f"procesador ({n} núcleos)", False


def _num(v) -> float | None:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, 4) if v == v else None


def val_metrics(metrics, keys: list[str], n_test: int) -> dict:
    """Lo que mide Ultralytics con las fotos de prueba, por clase y en total."""
    box = metrics.box
    per = {}
    idx = getattr(box, "ap_class_index", None)
    for j, ci in enumerate(list(idx) if idx is not None else []):
        if 0 <= int(ci) < len(keys):
            p, r, ap50, _ = box.class_result(j)
            per[keys[int(ci)]] = {"precision": _num(p), "recall": _num(r), "map50": _num(ap50)}
    return {"map50": _num(box.map50), "map50_95": _num(box.map), "precision": _num(box.mp), "recall": _num(box.mr),
            "test_images": n_test, "per_class": per}


def train(cfg: dict) -> None:
    from ultralytics import YOLO

    dev, dev_name, gpu = device_info()
    keys = cfg["keys"]
    epochs = int(cfg.get("epochs") or 100)
    emit(progress=0.06, message=f"Entrenando en {dev_name}")
    model = YOLO(cfg["base"])
    started = time.monotonic()

    def on_epoch(trainer) -> None:
        total = max(1, int(trainer.epochs))
        e = min(int(trainer.epoch) + 1, total)  # (la última vuelta avisa dos veces)
        m = trainer.metrics or {}
        rec = _num(m.get("metrics/recall(B)"))
        found = f" · ahora encuentra {int(round(rec * 10))} de cada 10" if rec is not None else ""
        emit(progress=round(0.06 + 0.84 * e / total, 3), message=f"Vuelta {e} de {total}{found}", epoch=e, epochs=total,
             map50=_num(m.get("metrics/mAP50(B)")))

    model.add_callback("on_fit_epoch_end", on_epoch)
    out = Path(cfg["out"])
    model.train(data=cfg["data"], epochs=epochs, imgsz=int(cfg.get("imgsz") or 640),
                patience=int(cfg.get("patience") or 25), project=str(out), name="run", exist_ok=True, device=dev,
                batch=int(cfg.get("batch") or (16 if gpu else 8)), workers=int(cfg.get("workers") or 2), plots=False,
                verbose=False, amp=gpu, cache=False, seed=0, deterministic=False)
    trainer = model.trainer
    best = Path(getattr(trainer, "best", "") or out / "run" / "weights" / "best.pt")
    if not best.exists():
        best = Path(getattr(trainer, "last", "") or out / "run" / "weights" / "last.pt")
    emit(progress=0.92, message="Midiendo con las fotos de prueba")
    final = YOLO(str(best))
    metrics = final.val(data=cfg["data"], split="val", imgsz=int(cfg.get("imgsz") or 640), device=dev, plots=False,
                        verbose=False, batch=8, project=str(out), name="prueba", exist_ok=True)
    result = val_metrics(metrics, keys, int(cfg.get("test_images") or 0))
    emit(progress=0.96, message="Exportando el modelo")
    onnx = final.export(format="onnx", imgsz=int(cfg.get("imgsz") or 640), opset=12, dynamic=False,
                        simplify=bool(cfg.get("simplify", True)), device="cpu")
    emit(done={"metrics": result, "onnx": str(onnx), "pt": str(best), "device": dev_name,
               "epochs_run": int(getattr(trainer, "epoch", epochs - 1)) + 1,
               "minutes": round((time.monotonic() - started) / 60, 1)})


def predict(cfg: dict) -> None:
    images = cfg["images"]  # [{"id", "path"}]
    conf = float(cfg.get("conf") or 0.15)
    dev, dev_name, _ = device_info()
    if cfg["method"] == "yoloe":
        from ultralytics import YOLOE

        prompts: dict[str, str] = cfg["prompts"]  # clave → palabra
        names, owner = [], []
        for k, p in prompts.items():
            if p not in names:
                names.append(p)
                owner.append(k)
        model = YOLOE(cfg.get("weights") or YOLOE_WEIGHTS)
        model.set_classes(names)
        key_of = owner
    else:
        from ultralytics import YOLO

        model = YOLO(cfg["weights"])
        key_of = list(cfg["keys"])
    emit(progress=0.1, message=f"La IA mira las fotos en {dev_name}")
    out = {}
    step = 16
    for i in range(0, len(images), step):
        part = images[i:i + step]
        results = model.predict([x["path"] for x in part], conf=conf, device=dev, verbose=False,
                                imgsz=int(cfg.get("imgsz") or 640))
        for x, r in zip(part, results):
            boxes = []
            b = r.boxes
            if b is not None and len(b):
                for xy, c, p in zip(b.xyxyn.tolist(), b.cls.tolist(), b.conf.tolist()):
                    ci = int(c)
                    k = key_of[ci] if 0 <= ci < len(key_of) else None
                    boxes.append({"k": k, "x1": round(xy[0], 4), "y1": round(xy[1], 4), "x2": round(xy[2], 4),
                                  "y2": round(xy[3], 4), "p": round(float(p), 3)})
            out[str(x["id"])] = boxes
        emit(progress=round(0.1 + 0.85 * min(len(images), i + step) / max(1, len(images)), 3),
             message=f"Pre-marcadas {min(len(images), i + step)} de {len(images)} fotos")
    Path(cfg["out"]).write_text(json.dumps(out))
    emit(done={"images": len(images), "with_boxes": sum(1 for v in out.values() if v)})


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[0] not in ("train", "predict"):
        print(__doc__)
        return 2
    os.environ.setdefault("YOLO_VERBOSE", "False")
    cfg = json.loads(Path(argv[1]).read_text())
    try:
        (train if argv[0] == "train" else predict)(cfg)
    except Exception as exc:  # noqa: BLE001 - se informa entero al proceso que lo lanzó
        emit(error=f"{type(exc).__name__}: {exc}"[:400])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
