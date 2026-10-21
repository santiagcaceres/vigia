"""Punto de entrada del agente para empaquetarlo como .exe (PyInstaller).

    pyinstaller agente_pc.spec

Equivale a `python -m agent`. El .exe lee `agente.env` (servidor y token) de su misma carpeta
y, si no existe, los pide por consola la primera vez.
"""

from agent.__main__ import main

if __name__ == "__main__":
    main()
