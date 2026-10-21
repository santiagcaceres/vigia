@echo off
REM Inicia el agente Vigía usando el Python instalado (alternativa al .exe).
REM Si falta el entorno, lo crea e instala las dependencias (primera vez: unos minutos).
cd /d %~dp0\..
if not exist .venv\Scripts\python.exe (
  echo Preparando el entorno por primera vez...
  py -3.12 -m venv .venv 2>nul || python -m venv .venv || goto :error
  .venv\Scripts\python -m pip install --upgrade pip
  .venv\Scripts\python -m pip install -r requirements-agent.txt || goto :error
)
.venv\Scripts\python -m agent %*
if errorlevel 1 pause
goto :eof
:error
echo Fallo la instalacion. Instala Python 3.12 desde python.org (marcar "Add python.exe to PATH").
pause
exit /b 1
