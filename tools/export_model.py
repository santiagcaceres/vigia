"""Exporta un modelo YOLO (.pt) a ONNX para el agente y el servidor.

Se usa para regenerar models/personas.onnx o para usar un modelo afinado con tus cámaras.
Requiere: pip install ultralytics   (no hace falta en el agente ni en el servidor)

    python -m tools.export_model yolo11n.pt models/personas.onnx --imgsz 640

Licencia: los modelos de Ultralytics (YOLOv8/YOLO11) son AGPL-3.0. Para un producto comercial
cerrado hace falta su licencia Enterprise, o cambiar a un detector con licencia permisiva.
"""

import argparse
import shutil
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("weights")
    p.add_argument("out")
    p.add_argument("--imgsz", type=int, default=640, help="640 = más preciso; 480 o 416 = más rápido en CPU")
    a = p.parse_args()
    from ultralytics import YOLO

    path = YOLO(a.weights).export(format="onnx", imgsz=a.imgsz, dynamic=False, simplify=True, opset=12)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(path, a.out)
    print("Modelo exportado en", a.out)


if __name__ == "__main__":
    main()
