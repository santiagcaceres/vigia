"""Notificaciones.

Dos cosas distintas:
- OperatorAlerts: aviso interno a TU equipo de que llegó un clip para revisar.
- ClientNotifier: aviso que un empleado le manda al CLIENTE con el clip, por los
  canales que el cliente tenga configurados (Telegram, WhatsApp, email, webhook).

En Vercel la función termina al responder, así que todo se envía de forma sincrónica,
con tiempos de espera cortos para no demorar la respuesta.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from .db import Client, Event, FilterModel, utcnow

log = logging.getLogger("vigia.server")

LABELS = {"salida_sin_pago": "Posible hurto: salió sin pasar por caja", "merodeo": "Merodeo prolongado",
          "zona_restringida": "Ingreso a zona restringida", "fuera_de_horario": "Persona fuera de horario"}


def _telegram(token: str, chat_id: str, text: str, snapshot: bytes | None, timeout: float) -> None:
    base = f"https://api.telegram.org/bot{token}"
    if snapshot:
        r = httpx.post(f"{base}/sendPhoto", data={"chat_id": chat_id, "caption": text[:1024]},
                       files={"photo": ("evento.jpg", snapshot, "image/jpeg")}, timeout=timeout)
    else:
        r = httpx.post(f"{base}/sendMessage", json={"chat_id": chat_id, "text": text}, timeout=timeout)
    r.raise_for_status()


class OperatorAlerts:
    """Avisa a TU equipo que hay un clip nuevo para revisar. No le escribe al cliente."""

    def __init__(self, telegram_token: str | None, operator_chat_id: str | None,
                 operator_webhook: str | None, public_url: str | None, timeout: float = 6.0):
        self.token = telegram_token
        self.chat_id = operator_chat_id
        self.webhook = operator_webhook
        self.public_url = (public_url or "").rstrip("/")
        self.timeout = timeout

    def new_clip(self, client_name: str, camera_name: str, ev: Event, snapshot: bytes | None) -> None:
        if not (self.webhook or (self.token and self.chat_id)):
            return
        url = f"{self.public_url}/#/revisar" if self.public_url else None
        if self.webhook:
            try:
                httpx.post(self.webhook, json={
                    "event_id": ev.id, "client": client_name, "camera": camera_name, "type": ev.type,
                    "severity": ev.severity, "description": ev.description, "score": ev.score,
                    "occurred_at": ev.occurred_at.isoformat() + "Z", "review_url": url,
                }, timeout=self.timeout).raise_for_status()
            except Exception as exc:
                log.warning("Falló el webhook del operador: %s", exc)
        if self.token and self.chat_id:
            text = (f"🔎 Clip para revisar\n{client_name} · {camera_name}\n"
                    f"{LABELS.get(ev.type, ev.type)}\n{ev.description}" + (f"\n{url}" if url else ""))
            try:
                _telegram(self.token, self.chat_id, text, snapshot, self.timeout)
            except Exception as exc:
                log.warning("Falló el aviso al operador por Telegram: %s", exc)


class ClientNotifier:
    """Manda el aviso al cliente con el link al clip, por los canales disponibles.

    El 'portal' siempre está disponible (el link existe aunque no se envíe nada).
    Devuelve (channels_intentados, delivery_por_canal, status).
    """

    def __init__(self, telegram_token: str | None = None, twilio_sid: str | None = None,
                 twilio_token: str | None = None, twilio_from: str | None = None,
                 resend_key: str | None = None, email_from: str | None = None, timeout: float = 8.0):
        self.telegram_token = telegram_token
        self.twilio_sid = twilio_sid
        self.twilio_token = twilio_token
        self.twilio_from = twilio_from
        self.resend_key = resend_key
        self.email_from = email_from
        self.timeout = timeout

    def available(self, client: Client) -> list[str]:
        """Canales que se pueden usar para este cliente, dado lo configurado."""
        out = ["portal"]
        if self.telegram_token and client.telegram_chat_id:
            out.append("telegram")
        if self.twilio_sid and self.twilio_token and self.twilio_from and client.contact_whatsapp:
            out.append("whatsapp")
        if self.resend_key and self.email_from and client.contact_email:
            out.append("email")
        if client.webhook_url:
            out.append("webhook")
        return out

    def send(self, client: Client, camera_name: str, ev: Event, message: str, link: str,
             snapshot: bytes | None, channels: list[str] | None) -> tuple[list[str], dict, str]:
        usable = set(self.available(client))
        wanted = [c for c in (channels or list(usable)) if c in usable and c != "portal"]
        delivery: dict[str, dict] = {"portal": {"ok": True, "detail": "link disponible"}}
        header = f"🚨 {client.name} · {camera_name}"
        full = f"{header}\n\n{message}\n\n▶ Ver el clip: {link}"
        for ch in wanted:
            try:
                if ch == "telegram":
                    _telegram(self.telegram_token, client.telegram_chat_id, full, snapshot, self.timeout)
                elif ch == "whatsapp":
                    self._whatsapp(client.contact_whatsapp, full)
                elif ch == "email":
                    self._email(client.contact_email, client.name, message, link)
                elif ch == "webhook":
                    httpx.post(client.webhook_url, json={
                        "kind": "aviso_cliente", "event_id": ev.id, "client": client.name,
                        "camera": camera_name, "message": message, "link": link,
                        "occurred_at": ev.occurred_at.isoformat() + "Z",
                    }, timeout=self.timeout).raise_for_status()
                delivery[ch] = {"ok": True, "detail": "enviado"}
            except Exception as exc:
                log.warning("Falló el aviso al cliente por %s: %s", ch, exc)
                delivery[ch] = {"ok": False, "detail": str(exc)[:200]}
        attempted = list(delivery.keys())
        failed = [c for c, r in delivery.items() if not r["ok"]]
        status = "error" if wanted and len(failed) == len(wanted) else ("parcial" if failed else "enviado")
        return attempted, delivery, status

    def _whatsapp(self, to: str, body: str) -> None:
        sender = self.twilio_from if self.twilio_from.startswith("whatsapp:") else f"whatsapp:{self.twilio_from}"
        dest = to if to.startswith("whatsapp:") else f"whatsapp:{to}"
        r = httpx.post(f"https://api.twilio.com/2010-04-01/Accounts/{self.twilio_sid}/Messages.json",
                       data={"From": sender, "To": dest, "Body": body},
                       auth=(self.twilio_sid, self.twilio_token), timeout=self.timeout)
        r.raise_for_status()

    def _email(self, to: str, client_name: str, message: str, link: str) -> None:
        html = (f"<p>{message}</p>"
                f'<p><a href="{link}" style="background:#b91c1c;color:#fff;padding:10px 16px;'
                f'border-radius:8px;text-decoration:none">Ver el clip</a></p>'
                f'<p style="color:#666;font-size:12px">Vigía · {client_name}</p>')
        r = httpx.post("https://api.resend.com/emails",
                       headers={"Authorization": f"Bearer {self.resend_key}"},
                       json={"from": self.email_from, "to": [to],
                             "subject": f"Vigía · posible hurto en {client_name}", "html": html},
                       timeout=self.timeout)
        r.raise_for_status()


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
