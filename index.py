"""Punto de entrada para Vercel: Vercel busca una instancia de FastAPI llamada `app`."""

from server.main import create_app

app = create_app()
