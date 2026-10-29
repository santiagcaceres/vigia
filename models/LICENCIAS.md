# Modelos y sus licencias

| Archivo | Qué es | De dónde sale | Licencia |
|---|---|---|---|
| `personas.onnx` | YOLO11n (COCO, 80 clases), el detector de las máquinas y la app | Ultralytics | **AGPL-3.0**: para cobrar, el código del sistema tiene que publicarse bajo AGPL o hay que comprar la licencia Enterprise de Ultralytics |
| `postura.onnx` | YOLO11n-pose (COCO-pose, 17 puntos del cuerpo) exportado a 256x256: la postura de cada persona para los gestos de ocultamiento (`core/pose.py`, `core/gestos.py`) | Ultralytics | **AGPL-3.0** (igual que `personas.onnx`) |

Comparación del 28 oct 2026 (128 imágenes de COCO con 254 personas, procesador, 1 hilo):

| Detector | AP50 personas | Ve (con ~91 % de precisión) | ms por cuadro | Cámaras de hurto en 2 núcleos (`agent.medir`) |
|---|---|---|---|---|
| YOLO11n | 0,763 | 59 % (umbral 0,35) | 72-87 | 3 |
| D-FINE-N | 0,809 | 63-66 % (umbral 0,55-0,60) | 124-145 | 1 |
| YOLOX-s (Apache) | 0,779 | — | 228 | — |
| YOLOX-tiny / nano (Apache) | 0,709 / 0,624 | — | 60 / 16 | — |

D-FINE-N a 480x480 (sin reentrenar) baja a 0,699: no conviene. En video real de Minas (28 oct, Dell): tarda 1,5-1,8
veces lo de YOLO11n y casi no ve más (sostiene mejor a una 2.ª persona del fondo; objetos con falsos). **Decisión del
dueño (29 oct): solo YOLO.** D-FINE salió de la app y del repo (`dfine_n_coco.pth` se exporta de nuevo con
`tools/deployment/export_onnx.py` de D-FINE si algún día hiciera falta).

**Postura (`postura.onnx`), medido el 29 oct 2026:** sobre la imagen entera el modelo de pose encuentra muchas menos
personas que el detector (AP50 de personas 0,53 contra 0,76 de `personas.onnx`, en las mismas 128 imágenes de COCO: se
pierde a la gente chica), por eso **va sobre el recorte de cada persona** que ya encontró el detector, que llega al
modelo agrandada. En recortes, los puntos coinciden con los de la versión de 640x640 a ~0,08 largos de torso (de los
hombros a las caderas). Cuesta **~18 ms por persona con 1 hilo** de procesador (`python -m agent.medir --gestos` lo
mide en cada máquina). Se regenera con `python -m tools.export_model yolo11n-pose.pt models/postura.onnx --imgsz 256`.
