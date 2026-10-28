# -*- mode: python ; coding: utf-8 -*-
# PyInstaller: genera dist/VigiaAgente.exe (la app de Analítica Crítica) con el modelo ONNX, el logo y el ícono adentro.
#   pip install pyinstaller && pyinstaller agente_pc.spec
# Se sigue llamando VigiaAgente.exe adentro del .zip para que las versiones viejas (hasta la 1.21) puedan
# actualizarse; al arrancar, el programa se renombra solo a «Analítica Crítica.exe» (agent/actualizar.py).
import re
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

datas = [("models/personas.onnx", "models"), ("agent/recursos", "agent/recursos")]
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
exe = EXE(
    pyz, a.scripts, a.binaries, a.datas, [],
    name="VigiaAgente",
    console=False,  # sin ventana negra: la interfaz es la ventana de la app (registro en agent_data/vigia.log)
    icon="agent/recursos/icono.ico",
    version=version_file(),
    upx=False,
)
