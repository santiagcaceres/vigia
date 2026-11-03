# -*- mode: python ; coding: utf-8 -*-
# PyInstaller: genera dist/VigiaAgente.exe (la app de Analítica Crítica) con los modelos ONNX, el logo y el ícono.
#   pip install pyinstaller && pyinstaller agente_pc.spec
# Se sigue llamando VigiaAgente.exe adentro del .zip para que las versiones viejas (hasta la 1.21) puedan
# actualizarse; al arrancar, el programa deja una copia «Analítica Crítica.exe» y la abre (agent/actualizar.py).
import re
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

# personas.onnx: el detector; postura.onnx: la postura de cada persona para los gestos de ocultamiento (core/pose.py
# la busca en models/ dentro del .exe: sin ella, «Configurar…» y el análisis siguen, sin gestos)
datas = [("models/personas.onnx", "models"), ("models/postura.onnx", "models"), ("agent/recursos", "agent/recursos")]
binaries, hiddenimports = [], []
for pkg in ("onnxruntime", "cv2", "tzdata"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h


def version_file() -> str:
    """Datos que muestra Windows (Propiedades del archivo, Administrador de tareas): nombre y versión."""
    text = Path("agent/version.py").read_text(encoding="utf-8-sig")
    m = re.search(r'VERSION\s*=\s*"([^"]*)"', text)
    version = m.group(1) if m else "dev"
    nums = [int(x) for x in re.findall(r"\d+", version)][:3] if re.match(r"\d", version) else []
    nums = tuple((nums + [0, 0, 0, 0])[:4])
    name = "Anal\\u00edtica Cr\\u00edtica"  # el archivo se evalúa como Python: así queda en ASCII
    out = Path("build") / "version_info.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text(f"""VSVersionInfo(
  ffi=FixedFileInfo(filevers={nums}, prodvers={nums}, mask=0x3f, flags=0x0, OS=0x40004, fileType=0x1,
                    subtype=0x0, date=(0, 0)),
  kids=[
    StringFileInfo([StringTable('380A04B0', [
      StringStruct('CompanyName', '{name}'),
      StringStruct('FileDescription', '{name}'),
      StringStruct('FileVersion', '{version}'),
      StringStruct('InternalName', 'AnaliticaCritica'),
      StringStruct('OriginalFilename', 'VigiaAgente.exe'),
      StringStruct('ProductName', '{name}'),
      StringStruct('ProductVersion', '{version}')])]),
    VarFileInfo([VarStruct('Translation', [0x380A, 1200])])
  ]
)
""", encoding="ascii")
    return str(out)


a = Analysis(
    ["agente_pc.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["matplotlib", "PyQt5", "PySide2", "scipy", "pandas", "sklearn"],
    noarchive=False,
)
pyz = PYZ(a.pure)


# ---- pantalla de carga (pedido del dueño, 3 nov): la muestra PyInstaller apenas se abre el .exe, antes de descomprimir
# (lo que más tarda) y de que arranque Python. agent/arranque.py le cambia el texto y la cierra cuando la ventana está.
# De fábrica muestra el nombre de cada archivo que descomprime; acá se cambia el procedimiento de Tcl que dibuja el
# texto para que muestre «Preparando el programa…» con una barra que avanza (hasta EXTRACT_SHARE %) y nuestros mensajes
# («=NN|texto»: NN mueve la barra). Si una versión futura de PyInstaller cambia esa plantilla, se avisa y queda la
# pantalla de fábrica (con los nombres de archivo): no rompe la compilación.
def splash_screen():
    import os
    import sys

    sys.path.insert(0, ".")
    from PyInstaller.building import splash_templates as tpl

    from agent.arranque import EXTRACT_SHARE, SPLASH_BAR, SPLASH_TEXT

    # la barra avanza por tamaño, no por cantidad: primero salen los archivos grandes (el detector, OpenCV,
    # onnxruntime), que son los que tardan. PyInstaller manda el nombre de cada archivo al empezarlo; al llegar el
    # siguiente, el anterior ya está. Los grandes van con su tamaño en el programa de Tcl; el resto, con el promedio.
    sizes = {}
    for name, src, _kind in list(a.binaries) + list(a.datas):
        try:
            sizes[name.replace("\\", "/")] = os.path.getsize(src)
        except OSError:
            pass
    total = max(1, sum(sizes.values()))
    big = {n: b for n, b in sizes.items() if b >= 256 * 1024 and not set(n) & set("{}[]$\\\" ")}
    small = max(1, (total - sum(big.values())) // max(1, len(sizes) - len(big)))
    print(f"Pantalla de carga: {len(sizes)} archivos, {total / 1e6:.0f} MB, {len(big)} grandes")
    x0, y0, x1, y1 = SPLASH_BAR
    proc = r"""
set _ac_done 0
set _ac_prev ""
set _ac_p 0
array set _ac_big {%(big)s}
proc _ac_bar {canvas p} {
    $canvas coords acbar %(x0)d %(y0)d [expr {%(x0)d + (%(x1)d - %(x0)d) * $p / 100}] %(y1)d
}
proc canvas_text_update {canvas tag _var - -} {
    upvar $_var var
    global _ac_done _ac_prev _ac_p _ac_big
    if {[regexp {^=(\d+)\|(.*)$} $var -> p txt]} {
        _ac_bar $canvas $p
        $canvas itemconfigure $tag -text $txt
    } elseif {[string index $var 0] eq "="} {
        $canvas itemconfigure $tag -text [string range $var 1 end]
    } else {
        if {$_ac_prev ne ""} {
            if {[info exists _ac_big($_ac_prev)]} {
                set _ac_done [expr {$_ac_done + $_ac_big($_ac_prev)}]
            } else {
                set _ac_done [expr {$_ac_done + %(small)d}]
            }
        }
        set _ac_prev [string map {\\ /} $var]
        set p [expr {min(%(share)d, int(%(share)d.0 * $_ac_done / %(total)d))}]
        if {$p > $_ac_p} {
            set _ac_p $p
            _ac_bar $canvas $p
        }
        $canvas itemconfigure $tag -text "Preparando el programa\u2026"
    }
}""" % {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "share": EXTRACT_SHARE, "total": total, "small": small,
        "big": " ".join(f"{{{n}}} {b}" for n, b in big.items())}
    bar = f"\n.root.canvas create rectangle {x0} {y0} {x0} {y1} -fill #011b5d -width 0 -tags acbar\n"
    image_script, n = re.subn(r"proc canvas_text_update \{.*?\n\}", lambda m: proc.strip(), tpl.image_script,
                              flags=re.S)
    if n == 1 and "-image splash_image" in tpl.splash_canvas_setup:
        tpl.image_script = image_script
        tpl.splash_canvas_setup = tpl.splash_canvas_setup + bar
    else:
        print("AVISO: cambió la plantilla de la pantalla de carga de PyInstaller: queda la de fábrica")
    return Splash(
        "agent/recursos/splash.png", binaries=a.binaries, datas=a.datas,
        text_pos=SPLASH_TEXT, text_size=10, text_font="{Segoe UI}", text_color="#5b6470",
        text_default="=Abriendo Anal\\u00edtica Cr\\u00edtica\\u2026",  # (Tcl: \\u en vez de tildes)
        always_on_top=True, center="active",  # en Windows, en la pantalla donde está el mouse
    )


splash = splash_screen()
exe = EXE(
    pyz, a.scripts, splash, splash.binaries, a.binaries, a.datas, [],
    name="VigiaAgente",
    console=False,  # sin ventana negra: la interfaz es la ventana de la app (registro en agent_data/vigia.log)
    icon="agent/recursos/icono.ico",
    version=version_file(),
    upx=False,
)
