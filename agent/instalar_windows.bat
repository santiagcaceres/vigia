@echo off
REM Instalación del agente Vigía en una PC con Windows (puede ser la misma donde corre iVMS-4200).
REM Requisitos: Python 3.11 o 3.12 instalado ("Add python.exe to PATH" marcado).
cd /d %~dp0\..
python -m venv .venv || goto :error
call .venv\Scripts\activate.bat
pip install -r requirements-agent.txt || goto :error
where ffmpeg >nul 2>nul || (
  echo Instalando ffmpeg...
  winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
)
echo.
echo Listo. Para iniciar el agente:
echo   .venv\Scripts\python -m agent --server https://TU-SERVIDOR --token TOKEN_DEL_PANEL
echo Para que arranque solo con Windows, ver README (seccion "Agente como servicio").
goto :eof
:error
echo Fallo la instalacion.
exit /b 1
