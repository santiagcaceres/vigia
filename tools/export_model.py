"""Exporta un modelo YOLO (.pt) a ONNX para el agente y el servidor.

Se usa para regenerar models/personas.onnx (el detector) y models/postura.onnx (la postura de cada persona, para
los gestos de ocultamiento), o para usar un modelo afinado con tus cámaras.
Requiere: pip install ultralytics   (no hace falta en el agente ni en el servidor)

    python -m tools.export_model yolo11n.pt models/personas.onnx --imgsz 640
    python -m tools.export_model yolo11n-pose.pt models/postura.onnx --imgsz 256

La postura va a 256x256 porque no mira la imagen entera sino el recorte de cada persona que encontró el detector
(core/pose.py): en recortes, sus puntos coinciden con los de la versión de 640 a ~0,08 largos de torso, y cuesta
~18 ms por persona con 1 hilo. Sobre la imagen entera se pierde a la gente chica (AP50 de personas 0,53 contra 0,76 del
detector, 29 oct). Después de regenerarla, verificar la salida (1, 56, 1344): 4 del recuadro + puntaje + 17 puntos
x (x, y, confianza). Va dentro del .exe (agente_pc.spec).

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
    p.add_argument("--imgsz", type=int, default=640,
                   help="640 = más preciso; 480 o 416 = más rápido en CPU; 256 para la postura (recortes)")
    a = p.parse_args()
    from ultralytics import YOLO

    path = YOLO(a.weights).export(format="onnx", imgsz=a.imgsz, dynamic=False, simplify=True, opset=12)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(path, a.out)
    print("Modelo exportado en", a.out)


if __name__ == "__main__":
    main()
