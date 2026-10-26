"""Modelo de personas preparado para analizar **muchas cámaras juntas** en la placa de video (servidores).

El ONNX exportado viene para una imagen por vez, con la imagen ya convertida a números decimales
(RGB, 0..1) y devolviendo las 80 clases COCO. Para un servidor con GPU NVIDIA eso desperdicia:
- de a una imagen la GPU queda casi sin usar (cada llamada tiene un costo fijo);
- convertir cada cuadro a decimales en el procesador mueve 5 MB por cuadro;
- bajar las 80 clases de la GPU cuando solo se usan 5.

`make_batch_model` arma, en memoria y sin volver a exportar, una variante que:
1. acepta un **lote** de N imágenes (los `Reshape` internos copian el tamaño del lote en vez de fijarlo
   en 1; se verificó que da exactamente lo mismo que el original);
2. recibe las imágenes **tal como salen de OpenCV** (uint8, BGR, alto x ancho x 3): la conversión a
   decimales, el orden de colores y la transposición se hacen en la GPU;
3. devuelve solo las cajas y las clases pedidas (persona y vehículos).

Necesita el paquete `onnx` (está en requirements-servidor.txt; la app de Windows no lo usa).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

KEEP_CLASSES = (0, 2, 3, 5, 7)  # persona; auto, moto, ómnibus, camión
INPUT_NAME = "imagenes"
OUTPUT_NAME = "salida"


@dataclass
class BatchModel:
    data: bytes  # el ONNX serializado
    input_name: str
    size: tuple[int, int]  # ancho, alto de entrada
    col_of: dict[int, int]  # clase COCO -> columna en la salida


def make_batch_model(model_path: str | Path, keep_classes: tuple[int, ...] = KEEP_CLASSES) -> BatchModel:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    m = onnx.load(str(model_path))
    g = m.graph
    inits = {i.name: i for i in g.initializer}

    # 1) lote variable: en los Reshape con forma fija [1, ...], el 1 pasa a 0 (= copiar la dimensión del lote)
    made: dict[str, str] = {}
    for n in g.node:
        if n.op_type != "Reshape" or n.input[1] not in inits:
            continue
        src = n.input[1]
        arr = numpy_helper.to_array(inits[src]).copy()
        if arr.ndim != 1 or arr[0] != 1:
            continue
        if src not in made:
            arr[0] = 0
            made[src] = src + "_lote"
            g.initializer.append(numpy_helper.from_array(arr, made[src]))
        n.input[1] = made[src]
    used = {i for n in g.node for i in n.input}
    for name in list(made):  # sacar las constantes viejas que ya nadie usa
        if name not in used:
            g.initializer.remove(inits[name])

    old_in = g.input[0]
    dims = old_in.type.tensor_type.shape.dim
    h = dims[2].dim_value or 640
    w = dims[3].dim_value or 640
    old_out = g.output[0]
    n_out = old_out.type.tensor_type.shape.dim[1].dim_value or 84

    # 2) entrada uint8 BGR (lote, alto, ancho, 3) -> decimales RGB (lote, 3, alto, ancho) dentro del modelo
    new_in = helper.make_tensor_value_info(INPUT_NAME, TensorProto.UINT8, ["lote", h, w, 3])
    pre = [
        helper.make_node("Cast", [INPUT_NAME], ["_pre_f"], to=TensorProto.FLOAT),
        helper.make_node("Div", ["_pre_f", "_pre_255"], ["_pre_n"]),
        helper.make_node("Transpose", ["_pre_n"], ["_pre_t"], perm=[0, 3, 1, 2]),
        helper.make_node("Gather", ["_pre_t", "_pre_rgb"], [old_in.name], axis=1),
    ]
    g.initializer.extend([numpy_helper.from_array(np.array(255.0, np.float32), "_pre_255"),
                          numpy_helper.from_array(np.array([2, 1, 0], np.int64), "_pre_rgb")])
    # 3) salida: solo las 4 coordenadas y las clases que se usan
    keep = [k for k in keep_classes if 4 + k < n_out]
    cols = np.array([0, 1, 2, 3] + [4 + k for k in keep], np.int64)
    g.initializer.append(numpy_helper.from_array(cols, "_post_cols"))
    post = helper.make_node("Gather", [old_out.name, "_post_cols"], [OUTPUT_NAME], axis=1)
    new_out = helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, ["lote", len(cols), None])

    nodes = list(g.node)
    del g.node[:]
    g.node.extend(pre + nodes + [post])
    g.input.remove(old_in)
    g.input.insert(0, new_in)
    g.output.remove(old_out)
    g.output.append(new_out)
    del g.value_info[:]  # las formas intermedias tenían el lote fijo en 1
    onnx.checker.check_model(m)
    return BatchModel(m.SerializeToString(), INPUT_NAME, (w, h), {k: 4 + i for i, k in enumerate(keep)})
