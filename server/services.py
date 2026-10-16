"""Alertas por Telegram y webhook.

En Vercel la función termina al responder, así que las alertas se envían antes de
responder al agente, con tiempos de espera cortos para no demorarlo.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from .db import Client, Event, FilterModel, utcnow

log = logging.getLogger("vigia.server")

LABELS = {"salida_sin_pago": "Posible hurto: salió sin pasar por caja", "merodeo": "Merodeo prolongado",
          "zona_restringida": "Ingreso a zona restringida", "fuera_de_horario": "Persona fuera de horario"}


class Alerts:
    def __init__(self, telegram_token: str | None, public_url: str | None, timeout: float = 6.0):
        self.token = telegram_token
        self.public_url = (public_url or "").rstrip("/")
        self.timeout = timeout

    def send(self, client: Client, camera_name: str, ev: Event, snapshot: bytes | None) -> None:
        if not (client.webhook_url or (self.token and client.telegram_chat_id)):
            return
        payload = {
            "event_id": ev.id, "client": client.name, "camera": camera_name, "type": ev.type,
            "severity": ev.severity, "zone": ev.zone, "description": ev.description, "score": ev.score,
            "occurred_at": ev.occurred_at.isoformat() + "Z",
            "url": f"{self.public_url}/#/evento/{ev.id}" if self.public_url else None,
        }
        if client.webhook_url:
            try:
                httpx.post(client.webhook_url, json=payload, timeout=self.timeout).raise_for_status()
            except Exception as exc:
                log.warning("Falló el webhook de alertas: %s", exc)
        if self.token and client.telegram_chat_id:
            text = (f"🚨 {LABELS.get(ev.type, ev.type)}\n{client.name} · {camera_name}\n{ev.description}"
                    + (f"\n{payload['url']}" if payload["url"] else ""))
            base = f"https://api.telegram.org/bot{self.token}"
            try:
                if snapshot:
                    r = httpx.post(f"{base}/sendPhoto", data={"chat_id": client.telegram_chat_id, "caption": text},
                                   files={"photo": ("evento.jpg", snapshot, "image/jpeg")}, timeout=self.timeout)
                else:
                    r = httpx.post(f"{base}/sendMessage", json={"chat_id": client.telegram_chat_id, "text": text},
                                   timeout=self.timeout)
                r.raise_for_status()
            except Exception as exc:
                log.warning("Falló el envío a Telegram: %s", exc)


def active_model(s, client_id: int) -> FilterModel | None:
    m = s.query(FilterModel).filter_by(client_id=client_id, active=True).first()
    return m or s.query(FilterModel).filter_by(client_id=None, active=True).first()


def parse_occurred_at(value: str | None) -> datetime:
    if not value:
        return utcnow()
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)
