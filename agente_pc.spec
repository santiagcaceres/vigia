# -*- mode: python ; coding: utf-8 -*-
# PyInstaller: genera dist/VigiaAgente.exe con el modelo ONNX adentro.
#   pip install pyinstaller && pyinstaller agente_pc.spec
from PyInstaller.utils.hooks import collect_all

datas = [("models/personas.onnx", "models")]
binaries, hiddenimports = [], []
for pkg in ("onnxruntime", "cv2", "tzdata"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

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
    console=True,
    upx=False,
)
