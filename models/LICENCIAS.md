# Modelos y sus licencias

| Archivo | Qué es | De dónde sale | Licencia |
|---|---|---|---|
| `personas.onnx` | YOLO11n (COCO, 80 clases), el detector de las máquinas y la app | Ultralytics | **AGPL-3.0**: para cobrar, el código del sistema tiene que publicarse bajo AGPL o hay que comprar la licencia Enterprise de Ultralytics |
| `dfine_n.onnx` | D-FINE-N (COCO, 80 clases), opción de prueba en «Ajustar…» | [Peterande/D-FINE](https://github.com/Peterande/D-FINE), `dfine_n_coco.pth` exportado el 28 oct 2026 con `tools/deployment/export_onnx.py` (640x640, salidas `labels`/`boxes`/`scores`) | **Apache-2.0**: libre para vender (conservar este aviso) |

Comparación del 28 oct 2026 (128 imágenes de COCO con 254 personas, procesador, 1 hilo):

| Detector | AP50 personas | Ve (con ~91 % de precisión) | ms por cuadro | Cámaras de hurto en 2 núcleos (`agent.medir`) |
|---|---|---|---|---|
| YOLO11n | 0,763 | 59 % (umbral 0,35) | 72-87 | 3 |
| D-FINE-N | 0,809 | 63-66 % (umbral 0,55-0,60) | 124-145 | 1 |
| YOLOX-s (Apache) | 0,779 | — | 228 | — |
| YOLOX-tiny / nano (Apache) | 0,709 / 0,624 | — | 60 / 16 | — |

D-FINE-N a 480x480 (sin reentrenar) baja a 0,699: no conviene. Con placa de video la diferencia de velocidad casi no
importa. Falta compararlos con video real de cámaras de techo (Minas) en «Ajustar…».
