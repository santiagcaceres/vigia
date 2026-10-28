"""Genera los logos e íconos de Analítica Crítica a partir de la imagen original (docs/marca/).

    python -m tools.logo

Usa solo el símbolo (el ojo con la mira): el texto de la imagen original dice «Análitica» (con la tilde en la
primera a); en el sistema el nombre se escribe aparte, bien: «Analítica Crítica».
Salida:
- agent/recursos/: logo para la app de escritorio (PNG para Tk, que no escala bien: un archivo por tamaño) e
  `icono.ico` (ícono del .exe, de la ventana y de la barra de tareas).
- public/img/: logo y favicon para la web.
Necesita Pillow (solo para generar; la app y el servidor no lo usan).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "docs" / "marca" / "analitica-critica-original.png"
NAVY = (1, 27, 93)  # el azul del logo (promedio de la imagen original)
SYMBOL_ROWS = (0, 700)  # el símbolo ocupa las filas 58-675; el texto, 754-831


def symbol_mask() -> Image.Image:
    """Máscara del símbolo (255 = tinta) recortada a su contorno, cuadrada."""
    rgb = np.asarray(Image.open(SOURCE).convert("RGB")).astype(np.float32)
    ink = np.clip((255.0 - rgb.mean(axis=2)) / (255.0 - 60.0), 0.0, 1.0)  # el fondo es blanco, la tinta azul oscuro
    ink[SYMBOL_ROWS[1]:] = 0.0  # sin el texto
    ys, xs = np.where(ink > 0.02)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = ink[y0:y1, x0:x1]
    side = max(crop.shape)
    sq = np.zeros((side, side), np.float32)
    oy, ox = (side - crop.shape[0]) // 2, (side - crop.shape[1]) // 2
    sq[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
    return Image.fromarray((sq * 255).astype(np.uint8), "L")


def colored(mask: Image.Image, size: int, color: tuple[int, int, int], pad: float = 0.0) -> Image.Image:
    """El símbolo de `color` sobre fondo transparente, en un cuadrado de `size` con margen `pad` (0-0.5)."""
    inner = max(1, round(size * (1 - 2 * pad)))
    m = mask.resize((inner, inner), Image.LANCZOS)
    out = Image.new("RGBA", (size, size), color + (0,))
    layer = Image.new("RGBA", (inner, inner), color + (255,))
    layer.putalpha(m)
    off = (size - inner) // 2
    out.alpha_composite(layer, (off, off))
    return out


def tile(mask: Image.Image, size: int) -> Image.Image:
    """Ícono: el símbolo azul sobre un cuadrado blanco con puntas redondeadas (se ve en barras claras y oscuras)."""
    big = size * 4  # se dibuja grande y se achica: bordes suaves
    bg = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    d = ImageDraw.Draw(bg)
    border = max(4, big // 64)
    d.rounded_rectangle((0, 0, big - 1, big - 1), radius=big * 0.22, fill=(214, 220, 232, 255))  # borde gris azulado
    d.rounded_rectangle((border, border, big - 1 - border, big - 1 - border), radius=big * 0.22 - border,
                        fill=(255, 255, 255, 255))
    pad = 0.10 if size <= 24 else 0.13
    bg.alpha_composite(colored(mask, big, NAVY, pad))
    return bg.resize((size, size), Image.LANCZOS)


def main() -> None:
    mask = symbol_mask()
    app = ROOT / "agent" / "recursos"
    web = ROOT / "public" / "img"
    app.mkdir(parents=True, exist_ok=True)
    web.mkdir(parents=True, exist_ok=True)
    sizes = [16, 20, 24, 32, 40, 48, 64, 96, 128, 256]
    tiles = {s: tile(mask, s) for s in sizes}
    tiles[256].save(app / "icono.ico", sizes=[(s, s) for s in sizes], append_images=[tiles[s] for s in sizes[:-1]])
    for s in (32, 64, 256):  # ventana (iconphoto) y diálogos
        tiles[s].save(app / f"icono-{s}.png")
    for s in (28, 40, 96):  # encabezado y pantalla de entrada (sin cuadrado, sobre el fondo de la app)
        colored(mask, s, NAVY).save(app / f"logo-{s}.png")
    colored(mask, 512, NAVY).save(web / "logo.png")
    colored(mask, 512, (255, 255, 255)).save(web / "logo-blanco.png")
    tiles[32].save(web / "favicon-32.png")
    tile(mask, 180).save(web / "apple-touch-icon.png")
    tiles[48].save(ROOT / "public" / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48)],
                   append_images=[tiles[16], tiles[32]])
    print("Listo:", app, web)


if __name__ == "__main__":
    main()
