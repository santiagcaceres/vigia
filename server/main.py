from __future__ import annotations

import json
import logging
import os
import re
import secrets
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import httpx

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, or_, select, update
from sqlalchemy.exc import IntegrityError

from core.rules import EVENT_TYPES, SEVERITY, parse_config
from core.scorer import FEATURE_NAMES, LABELS

from . import hpp, probe, training
from .db import (ROLES, Agent, Alarm, Camera, Client, ClientReport, Event, FilterModel, Incident, SessionToken, User,
                 VideoImport, iso, make_session_factory, utcnow)
from .security import SecretBox, hash_password, new_token, token_hash, verify_password
from .services import ClientNotifier, OperatorAlerts, active_model, parse_occurred_at
from .storage import LocalStorage, Storage, StorageError, SupabaseStorage

log = logging.getLogger("vigia.server")
COOKIE = "vigia_session"
PUBLIC_DIR = Path(__file__).resolve().parents[1] / "public"
VIDEO_EXT = {".mp4", ".mkv", ".avi", ".mov", ".ts", ".m4v"}
STALE_CLAIM = timedelta(minutes=20)
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class Settings:
    data_dir: Path = Path("data")
    database_url: str | None = None
    secret_key: str | None = None
    supabase_url: str | None = None
    supabase_key: str | None = None
    bucket: str = "vigia"
    telegram_token: str | None = None
    public_url: str | None = None
    admin_user: str = "admin"
    admin_password: str | None = None
    cron_secret: str | None = None
    retention_days: int = 60
    max_import_bytes: int = 5 * 1024**3
    session_days: int = 14
    clip_pre_seconds: float = 6.0   # mínimo antes de la alerta (el clip arranca cuando apareció la persona)
    clip_post_seconds: float = 6.0  # después de que la persona se fue
    clip_max_seconds: float = 180.0  # tope de duración de un clip
    serve_public: bool = True
    on_vercel: bool = False
    # Aviso interno al operador (equipo propio) de que hay clips para revisar
    operator_chat_id: str | None = None
    operator_webhook: str | None = None
    # Canales para avisar al cliente (opcionales, se activan si están configurados)
    twilio_sid: str | None = None
    twilio_token: str | None = None
    twilio_whatsapp_from: str | None = None
    resend_api_key: str | None = None
    email_from: str | None = None
    report_link_days: int = 14
    # Hik-Partner Pro OpenAPI (alarmas de los DVR por webhook)
    hpp_app_key: str | None = None
    hpp_secret_key: str | None = None
    hpp_sign_secret: str | None = None
    alarm_timezone: str = "America/Montevideo"  # hora de los DVR que todavía no tienen cámara asignada
    alarm_merge_seconds: float = 20.0  # disparos seguidos de la misma cámara se juntan en una alarma
    alarm_max_window_seconds: float = 120.0
    alarm_ready_delay_seconds: float = 20.0  # el DVR tarda en tener disponible lo recién grabado
    live_seconds: float = 45.0  # cuánto sigue "en vivo" una cámara después del último latido del panel
    missing: list[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> "Settings":
        st = cls(
            data_dir=Path(os.getenv("DATA_DIR", "data")),
            database_url=os.getenv("DATABASE_URL") or None,
            secret_key=os.getenv("SECRET_KEY") or None,
            supabase_url=os.getenv("SUPABASE_URL") or None,
            supabase_key=os.getenv("SUPABASE_SECRET_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY") or None,
            bucket=os.getenv("SUPABASE_BUCKET", "vigia"),
            telegram_token=os.getenv("TELEGRAM_BOT_TOKEN") or None,
            public_url=os.getenv("PUBLIC_URL") or None,
            admin_user=os.getenv("ADMIN_USER", "admin"),
            admin_password=os.getenv("ADMIN_PASSWORD") or None,
            cron_secret=os.getenv("CRON_SECRET") or None,
            retention_days=int(os.getenv("RETENTION_DAYS", "60")),
            clip_pre_seconds=float(os.getenv("CLIP_PRE_SECONDS", "6")),
            clip_post_seconds=float(os.getenv("CLIP_POST_SECONDS", "6")),
            clip_max_seconds=float(os.getenv("CLIP_MAX_SECONDS", "180")),
            alarm_merge_seconds=float(os.getenv("ALARM_MERGE_SECONDS", "20")),
            alarm_max_window_seconds=float(os.getenv("ALARM_MAX_WINDOW_SECONDS", "120")),
            on_vercel=bool(os.getenv("VERCEL")),
            operator_chat_id=os.getenv("OPERATOR_TELEGRAM_CHAT_ID") or None,
            operator_webhook=os.getenv("OPERATOR_WEBHOOK_URL") or None,
            twilio_sid=os.getenv("TWILIO_ACCOUNT_SID") or None,
            twilio_token=os.getenv("TWILIO_AUTH_TOKEN") or None,
            twilio_whatsapp_from=os.getenv("TWILIO_WHATSAPP_FROM") or None,
            resend_api_key=os.getenv("RESEND_API_KEY") or None,
            email_from=os.getenv("EMAIL_FROM") or None,
            report_link_days=int(os.getenv("REPORT_LINK_DAYS", "14")),
            hpp_app_key=os.getenv("HPP_APP_KEY") or None,
            hpp_secret_key=os.getenv("HPP_SECRET_KEY") or None,
            hpp_sign_secret=os.getenv("HPP_WEBHOOK_SECRET") or None,
            alarm_timezone=os.getenv("ALARM_TIMEZONE", "America/Montevideo"),
        )
        if st.on_vercel:
            st.serve_public = False  # Vercel sirve public/ desde su CDN
            if not st.public_url and os.getenv("VERCEL_PROJECT_PRODUCTION_URL"):
                st.public_url = "https://" + os.environ["VERCEL_PROJECT_PRODUCTION_URL"]
            for name, val in (("DATABASE_URL", st.database_url), ("SECRET_KEY", st.secret_key),
                              ("SUPABASE_URL", st.supabase_url), ("SUPABASE_SECRET_KEY", st.supabase_key),
                              ("ADMIN_PASSWORD", st.admin_password)):
                if not val:
                    st.missing.append(name)
        return st


# ---------------------------------------------------------------- esquemas
class LoginIn(BaseModel):
    username: str
    password: str


class ClientIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    timezone: str = "America/Argentina/Buenos_Aires"
    telegram_chat_id: str | None = None
    contact_whatsapp: str | None = Field(None, max_length=32)
    contact_email: str | None = Field(None, max_length=200)
    webhook_url: str | None = None


class UserIn(BaseModel):
    username: str = Field(min_length=3, max_length=80)
    password: str | None = Field(None, min_length=8)
    role: Literal["admin", "entrenador", "cliente"]
    client_id: int | None = None
    active: bool = True


class PasswordIn(BaseModel):
    current: str
    new: str = Field(min_length=8)


class AgentIn(BaseModel):
    client_id: int | None = None  # None = agente central (atiende cámaras asignadas de cualquier cliente)
    name: str = Field(min_length=1, max_length=120)
    process_imports: bool = True


class AlarmCfgIn(BaseModel):
    event_types: list[Literal[hpp.VIDEO_EVENT_TYPES]] = list(hpp.DEFAULT_EVENT_TYPES)  # type: ignore[valid-type]
    pre_seconds: float = Field(10.0, ge=0, le=60)
    post_seconds: float = Field(20.0, ge=3, le=90)


class CameraIn(BaseModel):
    client_id: int
    agent_id: int | None = None
    name: str = Field(min_length=1, max_length=120)
    brand: Literal["hikvision", "dahua"] = "hikvision"
    host: str = Field("", max_length=255)
    username: str = "admin"
    password: str | None = None
    channel: int = Field(1, ge=1, le=256)
    rtsp_port: int = Field(554, ge=1, le=65535)
    http_port: int = Field(80, ge=1, le=65535)
    detect_stream: Literal["sub", "main"] = "sub"
    nvr_clips: bool = True
    playback_time: Literal["local", "utc"] = "local"
    source_url: str | None = None
    detector: Literal["yolo", "motion"] = "yolo"
    process_fps: float = Field(5.0, gt=0, le=15)
    enabled: bool = True
    # vivo: el agente mira el stream 24 h. alarmas: Hik-Partner Pro avisa (webhook). alarmas_equipo:
    # el propio grabador avisa (conexión larga desde el agente); vale para cualquier marca.
    mode: Literal["vivo", "alarmas", "alarmas_equipo"] = "vivo"
    hpp_serial: str | None = Field(None, max_length=32)
    alarm_cfg: AlarmCfgIn = Field(default_factory=AlarmCfgIn)


class ZoneIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    type: Literal["shelf", "checkout", "exit", "restricted"]
    polygon: list[tuple[float, float]] = Field(min_length=3)


class AnalysisIn(BaseModel):
    zones: list[ZoneIn] = []
    rules: dict = {}


class LabelIn(BaseModel):
    label: Literal["hurto", "sospechoso", "normal", "falso_positivo"] | None
    note: str | None = Field(None, max_length=500)


class ReportIn(BaseModel):
    message: str | None = Field(None, max_length=1000)
    channels: list[str] | None = None
    expires_days: int | None = Field(None, ge=1, le=90)


class ImportIn(BaseModel):
    camera_id: int
    filename: str = Field(min_length=1, max_length=255)
    size: int | None = Field(None, ge=0)
    recorded_at: str | None = None


class IncidentIn(BaseModel):
    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)
    note: str | None = Field(None, max_length=300)


class TrainIn(BaseModel):
    client_id: int | None = None
    target_recall: float = Field(0.9, ge=0.5, le=1.0)


class DeviceAlarmIn(BaseModel):
    camera_id: int
    event_type: Literal[hpp.VIDEO_EVENT_TYPES]  # type: ignore[valid-type]
    occurred_at: str  # ISO con zona (hora de la PC central en la zona del cliente)
    target: str | None = Field(None, max_length=20)
    raw: dict = {}


class AlarmDoneIn(BaseModel):
    result: Literal["con_persona", "sin_persona"]
    event_uuid: str | None = Field(None, max_length=40)
    stats: dict = {}


class AlarmFailIn(BaseModel):
    error: str = Field(max_length=500)
    retry: bool = True


class HppSetupIn(BaseModel):
    all_devices: bool = False  # True: suscribir todos los equipos de la cuenta HPP


class ManualClipIn(BaseModel):
    ext: Literal["webm", "mp4"] = "webm"


class ClipDoneIn(BaseModel):
    source: Literal["nvr", "buffer", "video"] = "buffer"


class ProgressIn(BaseModel):
    progress: float = Field(ge=0, le=1)


class DoneIn(BaseModel):
    duration_s: float | None = None


class FailIn(BaseModel):
    error: str = Field(max_length=500)


# ---------------------------------------------------------------- app
def create_app(settings: Settings | None = None) -> FastAPI:
    st = settings or Settings.from_env()
    if st.missing:
        return _misconfigured_app(st.missing)

    if not st.on_vercel:
        st.data_dir.mkdir(parents=True, exist_ok=True)
    db_url = st.database_url or f"sqlite:///{(st.data_dir / 'vigia.db').as_posix()}"
    _, Session = make_session_factory(db_url)
    box = SecretBox(st.secret_key) if st.secret_key else SecretBox.from_env_or_file(st.data_dir)
    if st.supabase_url and st.supabase_key:
        storage: Storage = SupabaseStorage(st.supabase_url, st.supabase_key, st.bucket)
        try:
            storage.ensure_bucket()
        except Exception as exc:
            log.warning("No se pudo verificar el bucket de Supabase: %s", exc)
    else:
        storage = LocalStorage(st.data_dir / "files", box.secret + ":files")
    operator_alerts = OperatorAlerts(st.telegram_token, st.operator_chat_id, st.operator_webhook, st.public_url)
    notifier = ClientNotifier(st.telegram_token, st.twilio_sid, st.twilio_token, st.twilio_whatsapp_from,
                              st.resend_api_key, st.email_from)
    secure_cookie = bool(st.public_url and st.public_url.startswith("https")) or st.on_vercel

    with Session() as s:  # usuario administrador inicial
        if not s.scalar(select(func.count()).select_from(User)):
            pw = st.admin_password or secrets.token_urlsafe(12)
            s.add(User(username=st.admin_user, password_hash=hash_password(pw), role="admin"))
            try:
                s.commit()
                if not st.admin_password:
                    (st.data_dir / "admin_password.txt").write_text(f"{st.admin_user}\n{pw}\n")
                    log.warning("Usuario inicial: %s / %s (también en %s)", st.admin_user, pw,
                                st.data_dir / "admin_password.txt")
            except Exception:
                s.rollback()  # otra instancia lo creó al mismo tiempo

    app = FastAPI(title="Vigía API", version="2.0.0")
    app.state.Session = Session
    app.state.storage = storage
    app.state.settings = st

    # ------------------------------------------------------------ helpers
    def db():
        with Session() as s:
            yield s

    def current_user(request: Request, s=Depends(db)) -> User:
        token = request.cookies.get(COOKIE)
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            token = auth[7:].strip()
        if not token:
            raise HTTPException(401, "Iniciá sesión")
        sess = s.query(SessionToken).filter_by(token_hash=token_hash(token)).first()
        if not sess or sess.expires_at < utcnow():
            raise HTTPException(401, "Sesión vencida")
        user = s.get(User, sess.user_id)
        if not user or not user.active:
            raise HTTPException(401, "Usuario inactivo")
        return user

    def require(*roles: str):
        def dep(user: User = Depends(current_user)) -> User:
            if user.role not in roles:
                raise HTTPException(403, "No tenés permiso para esta acción")
            return user
        return dep

    # admin: usuarios, agentes y conexión con HPP. entrenador (= "empleado" en el panel): revisa y etiqueta,
    # crea clientes y configura cámaras. cliente: solo ve lo suyo.
    admin_only = require("admin")
    staff = require("admin", "entrenador")

    def current_agent(x_agent_token: str = Header(...), s=Depends(db)) -> Agent:
        ag = s.query(Agent).filter_by(token_hash=token_hash(x_agent_token)).first()
        if not ag:
            raise HTTPException(401, "Token de agente inválido")
        if not ag.last_seen or utcnow() - ag.last_seen > timedelta(seconds=20):
            ag.last_seen = utcnow()
            s.commit()
        return ag

    def scope_client(user: User, client_id: int | None) -> int | None:
        if user.role == "cliente":
            if client_id is not None and client_id != user.client_id:
                raise HTTPException(403, "No tenés acceso a ese cliente")
            return user.client_id
        return client_id

    def get_or_404(s, model, id_: int, what: str):
        obj = s.get(model, id_)
        if not obj:
            raise HTTPException(404, f"{what} no encontrado")
        return obj

    def event_visible(user: User, ev: Event) -> None:
        if user.role == "cliente" and ev.client_id != user.client_id:
            raise HTTPException(404, "Evento no encontrado")

    def redirect_to(path: str | None) -> RedirectResponse:
        if not path:
            raise HTTPException(404, "Archivo todavía no disponible")
        try:
            url = storage.signed_download_url(path, 3600)
        except StorageError as exc:
            raise HTTPException(502, str(exc)) from exc
        return RedirectResponse(url, status_code=307, headers={"Cache-Control": "private, max-age=600"})

    def client_dict(c: Client) -> dict:
        return {"id": c.id, "name": c.name, "timezone": c.timezone, "telegram_chat_id": c.telegram_chat_id,
                "contact_whatsapp": c.contact_whatsapp, "contact_email": c.contact_email,
                "webhook_url": c.webhook_url, "notify_channels": notifier.available(c),
                "created_at": iso(c.created_at)}

    def user_dict(u: User) -> dict:
        return {"id": u.id, "username": u.username, "role": u.role, "client_id": u.client_id, "active": u.active}

    def agent_dict(a: Agent) -> dict:
        online = bool(a.last_seen and utcnow() - a.last_seen < timedelta(minutes=3))
        return {"id": a.id, "client_id": a.client_id, "central": a.client_id is None, "name": a.name,
                "last_seen": iso(a.last_seen), "online": online, "status": a.last_status or {},
                "process_imports": a.process_imports}

    def camera_dict(c: Camera, user: User, status: dict | None = None) -> dict:
        d = {"id": c.id, "client_id": c.client_id, "agent_id": c.agent_id, "name": c.name, "enabled": c.enabled,
             "detector": c.detector, "process_fps": c.process_fps, "zones": c.zones or [], "rules": c.rules or {},
             "has_preview": bool(c.preview_path), "preview_at": iso(c.preview_at), "status": status,
             "live": bool(status and status.get("live")), "mode": c.mode or "vivo"}
        if user.role != "cliente":
            d.update(brand=c.brand, host=c.host, username=c.username, has_password=bool(c.password_enc),
                     channel=c.channel,
                     rtsp_port=c.rtsp_port, http_port=c.http_port, detect_stream=c.detect_stream,
                     nvr_clips=c.nvr_clips, playback_time=c.playback_time, has_source_url=bool(c.source_url_enc),
                     hpp_serial=c.hpp_serial, alarm_cfg=alarm_cfg_of(c))
        return d

    def alarm_cfg_of(c: Camera) -> dict:
        return {**AlarmCfgIn().model_dump(), **(c.alarm_cfg or {})}

    def event_dict(e: Event, cam_name: str | None = None, client_name: str | None = None) -> dict:
        return {
            "id": e.id, "uuid": e.uuid, "client_id": e.client_id, "client_name": client_name,
            "camera_id": e.camera_id, "camera_name": cam_name, "source": e.source, "import_id": e.import_id,
            "type": e.type, "severity": e.severity, "zone": e.zone, "description": e.description,
            "video_ts": e.video_ts, "occurred_at": iso(e.occurred_at), "score": e.score, "alert": e.alert,
            "features": e.features or {}, "label": e.label, "note": e.note, "labeled_at": iso(e.labeled_at),
            "snapshot_url": f"/api/events/{e.id}/snapshot" if e.snapshot_path else None,
            "clip_url": f"/api/events/{e.id}/clip" if e.clip_path else None, "clip_source": e.clip_source,
            "alarm": (e.meta or {}).get("alarm"),
        }

    def import_dict(i: VideoImport, cam_name=None) -> dict:
        return {"id": i.id, "client_id": i.client_id, "camera_id": i.camera_id, "camera_name": cam_name,
                "filename": i.filename, "size_bytes": i.size_bytes,
                "recorded_at": i.recorded_at.isoformat() if i.recorded_at else None,
                "duration_s": i.duration_s, "status": i.status, "progress": i.progress, "error": i.error,
                "created_at": iso(i.created_at), "video_url": f"/api/imports/{i.id}/video"}

    def delete_event_media(events: list[Event]) -> None:
        storage.remove([p for e in events for p in (e.snapshot_path, e.clip_path) if p])

    def report_link(token: str) -> str:
        base = (st.public_url or "").rstrip("/")
        return f"{base}/aviso.html?t={token}"

    def report_dict(r: ClientReport, for_client: bool, ev: Event | None = None,
                    cam_name: str | None = None, client_name: str | None = None) -> dict:
        d = {"id": r.id, "event_id": r.event_id, "client_id": r.client_id, "client_name": client_name,
             "camera_name": cam_name, "message": r.message, "status": r.status,
             "sent_at": iso(r.sent_at), "viewed_at": iso(r.viewed_at), "expires_at": iso(r.expires_at),
             "occurred_at": iso(ev.occurred_at) if ev else None,
             "has_clip": bool(ev and ev.clip_path), "link": report_link(r.public_token)}
        if not for_client:  # el cliente no ve a qué canales ni con qué resultado se envió
            d.update(channels=r.channels or [], delivery=r.delivery or {})
        return d

    # ------------------------------------------------------------ archivos locales (solo sin Supabase)
    if isinstance(storage, LocalStorage):
        @app.get("/api/_files/{path:path}", include_in_schema=False)
        def local_get(path: str, exp: int, sig: str):
            if not storage.verify("get", path, exp, sig) or not storage.exists(path):
                raise HTTPException(404, "No disponible")
            return FileResponse(storage.file(path))

        @app.put("/api/_files/upload/{path:path}", include_in_schema=False)
        async def local_put(path: str, exp: int, sig: str, request: Request):
            if not storage.verify("put", path, exp, sig):
                raise HTTPException(403, "Firma inválida")
            f = storage.file(path)
            f.parent.mkdir(parents=True, exist_ok=True)
            with open(f, "wb") as out:
                async for chunk in request.stream():
                    out.write(chunk)
            return {"ok": True}

    # ------------------------------------------------------------ sesión
    @app.post("/api/login")
    def login(body: LoginIn, response: Response, s=Depends(db)) -> dict:
        u = s.query(User).filter_by(username=body.username).first()
        if not u or not u.active or not verify_password(body.password, u.password_hash):
            raise HTTPException(401, "Usuario o clave incorrectos")
        token = new_token()
        s.add(SessionToken(token_hash=token_hash(token), user_id=u.id,
                           expires_at=utcnow() + timedelta(days=st.session_days)))
        s.commit()
        response.set_cookie(COOKIE, token, httponly=True, samesite="lax", secure=secure_cookie,
                            max_age=st.session_days * 86400)
        return {"user": user_dict(u), "token": token}

    @app.post("/api/logout")
    def logout(request: Request, response: Response, s=Depends(db)) -> dict:
        token = request.cookies.get(COOKIE)
        if token:
            s.query(SessionToken).filter_by(token_hash=token_hash(token)).delete()
            s.commit()
        response.delete_cookie(COOKIE)
        return {"ok": True}

    @app.get("/api/me")
    def me(user: User = Depends(current_user)) -> dict:
        return user_dict(user)

    @app.post("/api/me/password")
    def change_password(body: PasswordIn, user: User = Depends(current_user), s=Depends(db)) -> dict:
        u = s.get(User, user.id)
        if not verify_password(body.current, u.password_hash):
            raise HTTPException(422, "La clave actual no es correcta")
        u.password_hash = hash_password(body.new)
        s.commit()
        return {"ok": True}

    @app.get("/api/health")
    def health(s=Depends(db)) -> dict:
        s.execute(select(1))
        return {"status": "ok", "storage": "supabase" if isinstance(storage, SupabaseStorage) else "local",
                "database": s.get_bind().dialect.name}

    @app.get("/api/meta")
    def meta(_: User = Depends(current_user)) -> dict:
        return {"rule_types": list(EVENT_TYPES), "alarm_types": list(hpp.VIDEO_EVENT_TYPES), "severity": SEVERITY,
                "labels": list(LABELS),
                "zone_types": ["shelf", "checkout", "exit", "restricted"], "roles": list(ROLES)}

    # ------------------------------------------------------------ clientes y usuarios
    @app.get("/api/clients")
    def list_clients(user: User = Depends(current_user), s=Depends(db)) -> list[dict]:
        q = s.query(Client).order_by(Client.name)
        if user.role == "cliente":
            q = q.filter(Client.id == user.client_id)
        return [client_dict(c) for c in q.all()]

    @app.post("/api/clients", status_code=201)
    def create_client(body: ClientIn, _: User = Depends(staff), s=Depends(db)) -> dict:
        if s.query(Client).filter_by(name=body.name).first():
            raise HTTPException(409, "Ya existe un cliente con ese nombre")
        c = Client(**body.model_dump())
        s.add(c)
        s.commit()
        return client_dict(c)

    @app.put("/api/clients/{cid}")
    def update_client(cid: int, body: ClientIn, _: User = Depends(staff), s=Depends(db)) -> dict:
        c = get_or_404(s, Client, cid, "Cliente")
        for k, v in body.model_dump().items():
            setattr(c, k, v)
        s.commit()
        return client_dict(c)

    @app.get("/api/users")
    def list_users(_: User = Depends(admin_only), s=Depends(db)) -> list[dict]:
        return [user_dict(u) for u in s.query(User).order_by(User.username).all()]

    @app.post("/api/users", status_code=201)
    def create_user(body: UserIn, _: User = Depends(admin_only), s=Depends(db)) -> dict:
        if not body.password:
            raise HTTPException(422, "La clave es obligatoria (mínimo 8 caracteres)")
        if body.role == "cliente" and not body.client_id:
            raise HTTPException(422, "Un usuario de cliente necesita client_id")
        if s.query(User).filter_by(username=body.username).first():
            raise HTTPException(409, "Ese usuario ya existe")
        u = User(username=body.username, password_hash=hash_password(body.password), role=body.role,
                 client_id=body.client_id if body.role == "cliente" else None, active=body.active)
        s.add(u)
        s.commit()
        return user_dict(u)

    @app.put("/api/users/{uid}")
    def update_user(uid: int, body: UserIn, me_: User = Depends(admin_only), s=Depends(db)) -> dict:
        u = get_or_404(s, User, uid, "Usuario")
        if u.id == me_.id and (body.role != "admin" or not body.active):
            raise HTTPException(422, "No podés quitarte el rol de administrador ni desactivarte")
        if body.role == "cliente" and not body.client_id:
            raise HTTPException(422, "Un usuario de cliente necesita client_id")
        other = s.query(User).filter(User.username == body.username, User.id != uid).first()
        if other:
            raise HTTPException(409, "Ese usuario ya existe")
        u.username, u.role, u.active = body.username, body.role, body.active
        u.client_id = body.client_id if body.role == "cliente" else None
        if body.password:
            u.password_hash = hash_password(body.password)
        if body.password or not body.active:
            s.query(SessionToken).filter_by(user_id=u.id).delete()
        s.commit()
        return user_dict(u)

    # ------------------------------------------------------------ agentes
    @app.get("/api/agents")
    def list_agents(_: User = Depends(staff), s=Depends(db)) -> list[dict]:
        return [agent_dict(a) for a in s.query(Agent).order_by(Agent.id).all()]

    @app.post("/api/agents", status_code=201)
    def create_agent(body: AgentIn, _: User = Depends(admin_only), s=Depends(db)) -> dict:
        if body.client_id is not None:
            get_or_404(s, Client, body.client_id, "Cliente")
        token = new_token()
        a = Agent(client_id=body.client_id, name=body.name, token_hash=token_hash(token),
                  process_imports=body.process_imports)
        s.add(a)
        s.commit()
        return {**agent_dict(a), "token": token}

    @app.post("/api/agents/{aid}/token")
    def rotate_agent_token(aid: int, _: User = Depends(admin_only), s=Depends(db)) -> dict:
        a = get_or_404(s, Agent, aid, "Agente")
        token = new_token()
        a.token_hash = token_hash(token)
        s.commit()
        return {**agent_dict(a), "token": token}

    @app.delete("/api/agents/{aid}", status_code=204)
    def delete_agent(aid: int, _: User = Depends(admin_only), s=Depends(db)) -> None:
        a = get_or_404(s, Agent, aid, "Agente")
        s.query(Camera).filter_by(agent_id=aid).update({"agent_id": None})
        s.query(VideoImport).filter_by(claimed_by=aid).update({"claimed_by": None})
        s.delete(a)
        s.commit()

    # ------------------------------------------------------------ cámaras
    def agent_status_map(s) -> dict[int, dict]:
        out = {}
        for a in s.query(Agent).all():
            online = bool(a.last_seen and utcnow() - a.last_seen < timedelta(minutes=3))
            live = set((a.last_status or {}).get("live") or [])
            for cs in (a.last_status or {}).get("cameras", []):
                out[cs["camera_id"]] = {**cs, "agent_online": online, "live": cs["camera_id"] in live}
            for cid in live:  # cámaras en modo alarmas no figuran en cameras[] pero sí pueden estar en vivo
                out.setdefault(cid, {"camera_id": cid, "state": "escuchando", "agent_online": online, "live": True})
        return out

    @app.get("/api/cameras")
    def list_cameras(client_id: int | None = None, user: User = Depends(current_user), s=Depends(db)) -> list[dict]:
        client_id = scope_client(user, client_id)
        q = s.query(Camera).order_by(Camera.client_id, Camera.name)
        if client_id is not None:
            q = q.filter(Camera.client_id == client_id)
        statuses = agent_status_map(s)
        return [camera_dict(c, user, statuses.get(c.id)) for c in q.all()]

    @app.get("/api/cameras/{cam_id}")
    def get_camera(cam_id: int, user: User = Depends(current_user), s=Depends(db)) -> dict:
        c = get_or_404(s, Camera, cam_id, "Cámara")
        scope_client(user, c.client_id)
        return camera_dict(c, user, agent_status_map(s).get(c.id))

    def check_camera_body(s, body: CameraIn) -> None:
        get_or_404(s, Client, body.client_id, "Cliente")
        if body.agent_id is not None:
            ag = get_or_404(s, Agent, body.agent_id, "Agente")
            if ag.client_id is not None and ag.client_id != body.client_id:
                raise HTTPException(422, "El agente pertenece a otro cliente")
        if not body.host and not body.source_url:
            raise HTTPException(422, "Indicá la IP del NVR/cámara o una URL RTSP")
        body.hpp_serial = (body.hpp_serial or "").strip().upper() or None
        if body.mode == "alarmas" and not body.hpp_serial:
            raise HTTPException(422, "En modo alarmas indicá el número de serie del DVR en Hik-Partner Pro")
        if body.mode in ("alarmas", "alarmas_equipo") and not body.alarm_cfg.event_types:
            raise HTTPException(422, "Elegí al menos un tipo de alarma")
        if body.mode == "alarmas" and body.brand != "hikvision":
            raise HTTPException(422, "Las alarmas por Hik-Partner Pro son solo para Hikvision; "
                                     "usá «alarmas del propio grabador»")

    @app.post("/api/cameras", status_code=201)
    def create_camera(body: CameraIn, user: User = Depends(staff), s=Depends(db)) -> dict:
        check_camera_body(s, body)
        c = Camera(**body.model_dump(exclude={"password", "source_url"}), password_enc=box.encrypt(body.password),
                   source_url_enc=box.encrypt(body.source_url))
        s.add(c)
        s.commit()
        return camera_dict(c, user)

    @app.put("/api/cameras/{cam_id}")
    def update_camera(cam_id: int, body: CameraIn, user: User = Depends(staff), s=Depends(db)) -> dict:
        c = get_or_404(s, Camera, cam_id, "Cámara")
        check_camera_body(s, body)
        for k, v in body.model_dump(exclude={"password", "source_url"}).items():
            setattr(c, k, v)
        if body.password:
            c.password_enc = box.encrypt(body.password)
        c.source_url_enc = box.encrypt(body.source_url) if body.source_url else None
        s.commit()
        return camera_dict(c, user)

    @app.put("/api/cameras/{cam_id}/analysis")
    def update_analysis(cam_id: int, body: AnalysisIn, user: User = Depends(staff), s=Depends(db)) -> dict:
        c = get_or_404(s, Camera, cam_id, "Cámara")
        cfg = {"zones": [z.model_dump() for z in body.zones], "rules": body.rules}
        try:
            parse_config(cfg)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        c.zones, c.rules = cfg["zones"], cfg["rules"]
        s.commit()
        return camera_dict(c, user)

    @app.delete("/api/cameras/{cam_id}", status_code=204)
    def delete_camera(cam_id: int, _: User = Depends(admin_only), s=Depends(db)) -> None:
        c = get_or_404(s, Camera, cam_id, "Cámara")
        if s.query(Event).filter_by(camera_id=cam_id).first() or s.query(VideoImport).filter_by(camera_id=cam_id).first():
            raise HTTPException(409, "La cámara tiene eventos o videos: deshabilitala en lugar de borrarla")
        storage.remove(([c.preview_path] if c.preview_path else []) + [live_path(c)])
        s.delete(c)
        s.commit()

    @app.post("/api/cameras/{cam_id}/probe")
    def camera_probe(cam_id: int, _: User = Depends(staff), s=Depends(db)) -> dict:
        """Prueba desde el servidor que el grabador responda por HTTP con el usuario y la clave cargados.
        Sirve para los grabadores expuestos por el router del cliente (IP pública o DDNS)."""
        c = get_or_404(s, Camera, cam_id, "Cámara")
        if not c.host:
            raise HTTPException(422, "La cámara no tiene IP o DDNS cargado")
        res = probe.probe(c.brand, c.host, c.http_port, c.username, box.decrypt(c.password_enc))
        res["public"] = probe.is_public_host(c.host)
        res["rtsp_url"] = probe.rtsp_url(c.brand, c.host, c.rtsp_port, c.channel, c.username, "CLAVE")
        res["standard_ports"] = res["public"] and (c.rtsp_port == 554 or c.http_port == 80)
        return res

    @app.get("/api/cameras/{cam_id}/preview")
    def camera_preview(cam_id: int, user: User = Depends(current_user), s=Depends(db)):
        c = get_or_404(s, Camera, cam_id, "Cámara")
        scope_client(user, c.client_id)
        resp = redirect_to(c.preview_path)
        resp.headers["Cache-Control"] = "no-store"
        return resp

    def live_path(c: Camera) -> str:
        return f"c{c.client_id}/live/{c.id}.jpg"

    @app.post("/api/cameras/{cam_id}/live")
    def camera_live(cam_id: int, user: User = Depends(current_user), s=Depends(db)) -> dict:
        """Latido del panel: "alguien está mirando esta cámara". La PC central lo ve en la respuesta
        de /api/agent/status y sube cuadros con los recuadros mientras los latidos sigan llegando.
        Devuelve la URL firmada del cuadro (el navegador la pide 1 vez por segundo con ?t=)."""
        c = get_or_404(s, Camera, cam_id, "Cámara")
        scope_client(user, c.client_id)
        c.live_requested_at = utcnow()
        s.commit()
        st_cam = agent_status_map(s).get(c.id) or {}
        ag = s.get(Agent, c.agent_id) if c.agent_id else None
        agent_online = bool(ag and ag.last_seen and utcnow() - ag.last_seen < timedelta(minutes=3))
        frame_url, frame_age = None, None
        try:
            mtime = storage.modified_at(live_path(c))
            if mtime is not None:
                frame_url = storage.signed_download_url(live_path(c), 600)
                frame_age = max(0.0, round(datetime.now(timezone.utc).timestamp() - mtime, 1))
        except StorageError:
            frame_url = None
        # "activo" si la PC central dice que transmite o si el último cuadro es de hace pocos segundos
        fresh = frame_age is not None and frame_age < 8
        return {"active": agent_online and (bool(st_cam.get("live")) or fresh), "agent_online": agent_online,
                "frame_url": frame_url, "frame_age_s": frame_age,
                "heartbeat_seconds": 10, "live_seconds": st.live_seconds}

    # ------------------------------------------------------------ API del agente
    def agent_owns(ag: Agent, c: Camera) -> bool:
        """Un agente de local atiende las cámaras de su cliente (no asignadas a otro agente);
        un agente central solo las que se le asignaron explícitamente."""
        if ag.client_id is None:
            return c.agent_id == ag.id
        return c.client_id == ag.client_id and c.agent_id in (None, ag.id)

    def agent_cameras_q(s, ag: Agent):
        if ag.client_id is None:
            return s.query(Camera).filter(Camera.agent_id == ag.id)
        return s.query(Camera).filter(Camera.client_id == ag.client_id,
                                      or_(Camera.agent_id == ag.id, Camera.agent_id.is_(None)))

    def agent_camera(s, ag: Agent, cam_id: int) -> Camera:
        c = s.get(Camera, cam_id)
        if not c or not agent_owns(ag, c):
            raise HTTPException(404, "Cámara no encontrada para este agente")
        return c

    def agent_event_or_404(s, ag: Agent, uuid: str) -> Event:
        ev = s.query(Event).filter_by(uuid=uuid).first()
        cam = s.get(Camera, ev.camera_id) if ev else None
        if not ev or not cam or not agent_owns(ag, cam):
            raise HTTPException(404, "Evento no encontrado")
        return ev

    def camera_analysis(c: Camera) -> dict:
        return {"zones": c.zones or [], "rules": c.rules or {}}

    def model_spec(s, client_id: int) -> dict | None:
        m = active_model(s, client_id)
        return dict(m.spec, id=m.id) if m else None

    @app.get("/api/agent/config")
    def agent_config(ag: Agent = Depends(current_agent), s=Depends(db)) -> dict:
        cams = agent_cameras_q(s, ag).order_by(Camera.id).all()
        clients = {c.id: c for c in s.query(Client).filter(Client.id.in_({c.client_id for c in cams}))} if cams else {}
        models = {cid: model_spec(s, cid) for cid in clients}  # cada cliente puede tener su propio modelo
        return {
            "agent_id": ag.id, "client_id": ag.client_id, "central": ag.client_id is None,
            "process_imports": ag.process_imports,
            "settings": {"clip_pre_seconds": st.clip_pre_seconds, "clip_post_seconds": st.clip_post_seconds,
                         "clip_max_seconds": st.clip_max_seconds},
            "filter_model": model_spec(s, ag.client_id) if ag.client_id is not None else None,
            "cameras": [{
                **agent_camera_cfg(c, clients[c.client_id]), "filter_model": models[c.client_id],
            } for c in cams],
        }

    def agent_camera_cfg(c: Camera, client: Client) -> dict:
        return {
            "id": c.id, "name": c.name, "client_id": c.client_id, "client_name": client.name,
            "enabled": c.enabled, "detector": c.detector, "process_fps": c.process_fps,
            "mode": c.mode or "vivo", "alarm_cfg": alarm_cfg_of(c),
            "connection": {"brand": c.brand, "host": c.host, "username": c.username,
                           "password": box.decrypt(c.password_enc),
                           "channel": c.channel, "rtsp_port": c.rtsp_port, "http_port": c.http_port,
                           "detect_stream": c.detect_stream, "nvr_clips": c.nvr_clips,
                           "playback_time": c.playback_time, "timezone": client.timezone,
                           "source_url": box.decrypt(c.source_url_enc) or None},
            "analysis": camera_analysis(c),
        }

    @app.post("/api/agent/status")
    def agent_status(payload: dict, ag: Agent = Depends(current_agent), s=Depends(db)) -> dict:
        a = s.get(Agent, ag.id)
        a.last_status = {"cameras": payload.get("cameras", [])[:200],
                         "live": [int(x) for x in (payload.get("live") or [])[:200] if isinstance(x, int)],
                         "outbox_pending": payload.get("outbox_pending"), "version": payload.get("version"),
                         "at": iso(utcnow()),
                         **{k: payload[k] for k in ("alarms_with_person", "alarms_discarded", "alarms_failed")
                            if isinstance(payload.get(k), int)}}
        a.last_seen = utcnow()
        s.commit()
        since = utcnow() - timedelta(seconds=st.live_seconds)
        live = [c.id for c in agent_cameras_q(s, ag).filter(Camera.live_requested_at >= since, Camera.enabled.is_(True))]
        return {"ok": True, "live": live}

    @app.post("/api/agent/cameras/{cam_id}/live-url")
    def agent_live_url(cam_id: int, ag: Agent = Depends(current_agent), s=Depends(db)) -> dict:
        """URL firmada para que la PC central suba el cuadro en vivo directo al almacenamiento
        (sin pasar por el servidor: 1 cuadro por segundo mientras alguien mira)."""
        c = agent_camera(s, ag, cam_id)
        try:
            return {"upload_url": storage.signed_upload_url(live_path(c)), "content_type": "image/jpeg"}
        except StorageError as exc:
            raise HTTPException(502, str(exc)) from exc

    @app.post("/api/agent/events", status_code=201)
    def agent_event(meta: str = Form(...), snapshot: UploadFile = File(...),
                          ag: Agent = Depends(current_agent), s=Depends(db)) -> dict:
        try:
            m = json.loads(meta)
            uuid = str(m["uuid"])
            cam = agent_camera(s, ag, int(m["camera_id"]))
            if m["type"] not in EVENT_TYPES:
                raise ValueError(f"tipo desconocido {m['type']}")
            if not uuid.isalnum() or len(uuid) > 40:
                raise ValueError("uuid inválido")
            source = m.get("source", "vivo")
            if source not in ("vivo", "importado", "alarma"):
                raise ValueError("source inválido")
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(422, f"Metadatos inválidos: {exc}") from exc
        import_id = None
        if source == "importado":
            imp = s.get(VideoImport, int(m.get("import_id") or 0))
            if not imp or imp.claimed_by != ag.id or imp.camera_id != cam.id:
                raise HTTPException(404, "Importación no encontrada para este agente")
            import_id = imp.id
        alarm = None
        if source == "alarma":
            alarm = s.get(Alarm, int(m.get("alarm_id") or 0))
            if not alarm or alarm.claimed_by != ag.id or alarm.camera_id != cam.id:
                raise HTTPException(404, "Alarma no encontrada para este agente")
        existing = s.query(Event).filter_by(uuid=uuid).first()
        if existing:
            return {"id": existing.id, "duplicate": True}
        data = snapshot.file.read(5 * 1024 * 1024 + 1)
        if len(data) > 5 * 1024 * 1024:
            raise HTTPException(413, "Imagen demasiado grande")
        snap_path = f"c{cam.client_id}/cam{cam.id}/{uuid}.jpg"
        try:
            storage.put(snap_path, data, "image/jpeg")
        except StorageError as exc:
            raise HTTPException(502, str(exc)) from exc
        ev = Event(uuid=uuid, client_id=cam.client_id, camera_id=cam.id, source=source, import_id=import_id,
                   type=m["type"], severity=SEVERITY[m["type"]], zone=m.get("zone"),
                   description=str(m.get("description", ""))[:500], track_id=int(m.get("track_id") or 0),
                   video_ts=float(m.get("video_ts") or 0), occurred_at=parse_occurred_at(m.get("occurred_at")),
                   features={k: float(v) for k, v in (m.get("features") or {}).items() if k in FEATURE_NAMES},
                   meta={"bbox": m.get("bbox"), "frame_size": m.get("frame_size"),
                         **({"alarm": {"id": alarm.id, "dvr_event": alarm.event_type, "triggers": alarm.triggers,
                                       "stats": m.get("clip_stats") or {}}} if alarm else {})},
                   score=m.get("score"), alert=bool(m.get("alert", True)), snapshot_path=snap_path)
        s.add(ev)
        s.commit()
        if ev.alert and source in ("vivo", "alarma"):
            operator_alerts.new_clip(s.get(Client, cam.client_id).name, cam.name, ev, data)
        return {"id": ev.id}

    @app.post("/api/agent/events/{uuid}/clip-upload")
    def agent_clip_upload(uuid: str, ag: Agent = Depends(current_agent), s=Depends(db)) -> dict:
        ev = agent_event_or_404(s, ag, uuid)
        path = f"c{ev.client_id}/cam{ev.camera_id}/{ev.uuid}.mp4"
        try:
            return {"upload_url": storage.signed_upload_url(path), "content_type": "video/mp4"}
        except StorageError as exc:
            raise HTTPException(502, str(exc)) from exc

    @app.post("/api/agent/events/{uuid}/clip-done")
    def agent_clip_done(uuid: str, body: ClipDoneIn, ag: Agent = Depends(current_agent), s=Depends(db)) -> dict:
        ev = agent_event_or_404(s, ag, uuid)
        path = f"c{ev.client_id}/cam{ev.camera_id}/{ev.uuid}.mp4"
        if not storage.exists(path):
            raise HTTPException(409, "El clip no llegó al almacenamiento")
        ev.clip_path, ev.clip_source = path, body.source
        s.commit()
        return {"ok": True}

    @app.post("/api/agent/cameras/{cam_id}/preview")
    def agent_preview(cam_id: int, image: UploadFile = File(...), ag: Agent = Depends(current_agent),
                            s=Depends(db)) -> dict:
        c = agent_camera(s, ag, cam_id)
        data = image.file.read(5 * 1024 * 1024 + 1)
        if len(data) > 5 * 1024 * 1024:
            raise HTTPException(413, "Imagen demasiado grande")
        path = f"c{c.client_id}/previews/{c.id}.jpg"
        storage.put(path, data, "image/jpeg")
        c.preview_path, c.preview_at = path, utcnow()
        s.commit()
        return {"ok": True}

    @app.get("/api/agent/jobs")
    def agent_next_job(ag: Agent = Depends(current_agent), s=Depends(db)):
        """Entrega al agente el próximo video importado de su cliente para analizar."""
        if not ag.process_imports:
            return Response(status_code=204)
        stale = utcnow() - STALE_CLAIM
        mine = agent_cameras_q(s, ag).with_entities(Camera.id)
        candidates = (s.query(VideoImport)
                      .filter(VideoImport.camera_id.in_(mine),
                              or_(VideoImport.status == "en_cola",
                                  (VideoImport.status == "procesando") & (VideoImport.claimed_at < stale)))
                      .order_by(VideoImport.id).limit(5).all())
        for imp in candidates:
            prev_status, prev_claim = imp.status, imp.claimed_at
            res = s.execute(update(VideoImport)
                            .where(VideoImport.id == imp.id, VideoImport.status == prev_status,
                                   (VideoImport.claimed_at == prev_claim) if prev_claim else VideoImport.claimed_at.is_(None))
                            .values(status="procesando", claimed_by=ag.id, claimed_at=utcnow(), progress=0.0, error=None))
            s.commit()
            if res.rowcount != 1:
                continue  # otro agente lo tomó primero
            old = s.query(Event).filter_by(import_id=imp.id).all()  # restos de un intento anterior
            delete_event_media(old)
            for e in old:
                s.delete(e)
            s.commit()
            cam = s.get(Camera, imp.camera_id)
            client = s.get(Client, imp.client_id)
            return {"import_id": imp.id, "filename": imp.filename,
                    "download_url": storage.signed_download_url(imp.storage_path, 6 * 3600),
                    "recorded_at": imp.recorded_at.isoformat() if imp.recorded_at else None,
                    "timezone": client.timezone,
                    "camera": {"id": cam.id, "name": cam.name, "detector": cam.detector,
                               "process_fps": cam.process_fps, "analysis": camera_analysis(cam)},
                    "filter_model": model_spec(s, imp.client_id)}
        return Response(status_code=204)

    def claimed_import(s, ag: Agent, iid: int) -> VideoImport:
        imp = s.get(VideoImport, iid)
        if not imp or imp.claimed_by != ag.id or imp.status != "procesando":
            raise HTTPException(409, "Esta importación ya no está asignada a este agente")
        return imp

    @app.post("/api/agent/imports/{iid}/progress")
    def agent_import_progress(iid: int, body: ProgressIn, ag: Agent = Depends(current_agent), s=Depends(db)) -> dict:
        imp = claimed_import(s, ag, iid)
        imp.progress, imp.claimed_at = round(body.progress, 3), utcnow()
        s.commit()
        return {"ok": True}

    @app.post("/api/agent/imports/{iid}/done")
    def agent_import_done(iid: int, body: DoneIn, ag: Agent = Depends(current_agent), s=Depends(db)) -> dict:
        imp = claimed_import(s, ag, iid)
        imp.status, imp.progress, imp.duration_s = "listo", 1.0, body.duration_s
        s.commit()
        return {"ok": True}

    @app.post("/api/agent/imports/{iid}/fail")
    def agent_import_fail(iid: int, body: FailIn, ag: Agent = Depends(current_agent), s=Depends(db)) -> dict:
        imp = claimed_import(s, ag, iid)
        imp.status, imp.error = "error", body.error
        s.commit()
        return {"ok": True}

    # ------------------------------------------------------------ alarmas de Hik-Partner Pro
    hpp_sign_secret = st.hpp_sign_secret or st.hpp_secret_key

    def hpp_client() -> hpp.HppClient:
        if not (st.hpp_app_key and st.hpp_secret_key):
            raise HTTPException(503, "Falta configurar HPP_APP_KEY y HPP_SECRET_KEY en Vercel")
        return hpp.HppClient(st.hpp_app_key, st.hpp_secret_key)

    def webhook_url() -> str:
        if not st.public_url:
            raise HTTPException(503, "Falta PUBLIC_URL (la dirección pública del panel)")
        return st.public_url.rstrip("/") + "/api/hpp/webhook"

    def alarm_dict(a: Alarm, cam_name: str | None = None, client_name: str | None = None) -> dict:
        return {"id": a.id, "client_id": a.client_id, "client_name": client_name, "camera_id": a.camera_id,
                "camera_name": cam_name, "device_serial": a.device_serial, "channel": a.channel, "source": a.source,
                "event_type": a.event_type, "target": a.target, "first_at": iso(a.first_at),
                "last_at": iso(a.last_at), "triggers": a.triggers, "status": a.status, "attempts": a.attempts,
                "error": a.error, "stats": a.stats or {}, "event_id": a.event_id, "received_at": iso(a.received_at)}

    def ingest_alarm(s, al: hpp.HppAlarm) -> str:
        """Guarda una alarma recibida. Devuelve qué se hizo con ella."""
        serial = al.device_serial.strip().upper()
        cam = (s.query(Camera).filter(Camera.hpp_serial == serial, Camera.channel == al.channel)
               .order_by(Camera.id).first())
        client = s.get(Client, cam.client_id) if cam else None
        occurred = al.occurred_utc(client.timezone if client else st.alarm_timezone)
        if cam is not None:
            if not cam.enabled or (cam.mode or "vivo") != "alarmas":
                return "ignorada"
            if al.event_type not in alarm_cfg_of(cam)["event_types"]:
                return "ignorada"
        elif al.event_type not in hpp.VIDEO_EVENT_TYPES:
            return "ignorada"
        return store_alarm(s, cam, serial, al.channel, al.event_type, occurred, al.target, al.raw,
                           dedup_key=f"{serial}:{al.channel}:{al.event_type}:{al.trigger_time}", source="hpp")

    def store_alarm(s, cam: Camera | None, serial: str, channel: int, event_type: str, occurred: datetime,
                    target: str | None, raw: dict, dedup_key: str, source: str) -> str:
        """Guarda una alarma (de HPP o del propio grabador) agrupando los disparos seguidos."""
        if target and target.lower() not in ("human", "person", "persona"):
            return "no_persona_dvr"  # el propio equipo (AcuSense / SMD) dice que fue un vehículo u otra cosa
        status = "pendiente" if cam else "sin_camara"
        merge = timedelta(seconds=st.alarm_merge_seconds)
        q = s.query(Alarm).filter(Alarm.device_serial == serial, Alarm.channel == channel,
                                  Alarm.status == status, Alarm.last_at >= occurred - merge,
                                  Alarm.first_at <= occurred + merge,
                                  Alarm.first_at >= occurred - timedelta(seconds=st.alarm_max_window_seconds))
        prev = q.order_by(Alarm.id.desc()).first()
        if prev is not None:  # disparos seguidos: una sola alarma con una ventana más larga
            prev.first_at, prev.last_at = min(prev.first_at, occurred), max(prev.last_at, occurred)
            prev.triggers += 1
            if event_type != prev.event_type and event_type not in prev.event_type:
                prev.event_type = (prev.event_type + "+" + event_type)[:40]
            s.commit()
            return "agrupada"
        a = Alarm(dedup_key=dedup_key[:160], client_id=cam.client_id if cam else None,
                  camera_id=cam.id if cam else None, device_serial=serial, channel=channel,
                  event_type=event_type, target=target, first_at=occurred, last_at=occurred, status=status,
                  raw=raw, source=source)
        s.add(a)
        try:
            s.commit()
        except IntegrityError:  # reenvío de la misma alarma
            s.rollback()
            return "duplicada"
        return "nueva" if cam else "sin_camara"

    @app.post("/api/agent/alarms", status_code=201)
    def agent_device_alarm(body: DeviceAlarmIn, ag: Agent = Depends(current_agent), s=Depends(db)) -> dict:
        """El agente avisa una alarma que le mandó el propio grabador (modo alarmas_equipo)."""
        cam = agent_camera(s, ag, body.camera_id)
        if (cam.mode or "vivo") != "alarmas_equipo" or not cam.enabled:
            raise HTTPException(409, "La cámara no está en modo «alarmas del propio grabador»")
        if body.event_type not in alarm_cfg_of(cam)["event_types"]:
            return {"result": "ignorada"}
        client = s.get(Client, cam.client_id)
        occurred = hpp.parse_device_time(body.occurred_at, client.timezone)
        serial = cam.hpp_serial or f"CAM{cam.id}"
        result = store_alarm(s, cam, serial, cam.channel, body.event_type, occurred, body.target,
                             {"brand": cam.brand, **body.raw}, dedup_key=f"equipo:{cam.id}:{body.event_type}:{body.occurred_at}",
                             source="equipo")
        return {"result": result}

    @app.get("/api/hpp/webhook")
    def hpp_webhook_check(x_hook_batch_id: str = Header(""), x_hook_timestamp: str = Header("")):
        """HPP valida la URL al registrarla: espera la firma en el encabezado X-Hook-Signature."""
        if not hpp_sign_secret:
            raise HTTPException(503, "Hik-Partner Pro no configurado")
        return Response(status_code=200, headers={
            "X-Hook-Signature": hpp.sign(hpp_sign_secret, x_hook_timestamp, x_hook_batch_id)})

    @app.post("/api/hpp/webhook")
    async def hpp_webhook(request: Request, x_hook_batch_id: str = Header(""), x_hook_timestamp: str = Header(""),
                          x_hook_signature: str = Header("")):
        if not hpp_sign_secret:
            raise HTTPException(503, "Hik-Partner Pro no configurado")
        if not hpp.verify(hpp_sign_secret, x_hook_timestamp, x_hook_batch_id, x_hook_signature):
            raise HTTPException(401, "Firma inválida")
        try:
            body = json.loads(await request.body())
        except ValueError as exc:
            raise HTTPException(400, "JSON inválido") from exc
        _, messages = hpp.extract_messages(body)
        result: Counter = Counter()
        with Session() as s:
            for msg in messages[:500]:
                al = hpp.parse_message(msg)
                if al is None:
                    result["sin_datos"] += 1
                    continue
                try:
                    result[ingest_alarm(s, al)] += 1
                except Exception:
                    log.exception("No se pudo guardar una alarma de HPP")
                    s.rollback()
                    result["error"] += 1
        if result.get("sin_datos"):
            log.info("Webhook HPP con mensajes no reconocidos: %s", json.dumps(body)[:1000])
        return {"ok": True, **result}

    @app.get("/api/hpp/status")
    def hpp_status(check: bool = False, _: User = Depends(staff), s=Depends(db)) -> dict:
        last = s.query(Alarm).order_by(Alarm.id.desc()).first()
        d = {"configured": bool(st.hpp_app_key and st.hpp_secret_key), "webhook_url":
             (st.public_url.rstrip("/") + "/api/hpp/webhook") if st.public_url else None,
             "last_alarm_at": iso(last.received_at) if last else None,
             "cameras": s.query(Camera).filter(Camera.mode == "alarmas").count()}
        if check and d["configured"]:
            try:
                d["remote"] = hpp_client().webhook_config()
            except (hpp.HppError, httpx.HTTPError) as exc:
                d["error"] = str(exc)
        return d

    @app.post("/api/hpp/setup")
    def hpp_setup(body: HppSetupIn, _: User = Depends(admin_only), s=Depends(db)) -> dict:
        """Registra el webhook en HPP y suscribe los DVR de las cámaras en modo alarmas."""
        serials = sorted({c.hpp_serial for c in s.query(Camera).filter(Camera.hpp_serial.isnot(None))})
        if not serials and not body.all_devices:
            raise HTTPException(422, "No hay cámaras con número de serie de HPP; cargalas o suscribí todos los equipos")
        try:
            hpp_client().setup_webhook(webhook_url(), st.hpp_sign_secret, None if body.all_devices else serials)
        except (hpp.HppError, httpx.HTTPError) as exc:
            raise HTTPException(502, f"Hik-Partner Pro: {exc}") from exc
        return {"ok": True, "webhook_url": webhook_url(), "serials": None if body.all_devices else serials}

    @app.get("/api/hpp/devices")
    def hpp_devices(_: User = Depends(staff), s=Depends(db)) -> list[dict]:
        """Equipos y canales de la cuenta HPP, para dar de alta las cámaras sin tipear series."""
        hc = hpp_client()
        known = {(c.hpp_serial, c.channel): c.id for c in s.query(Camera).filter(Camera.hpp_serial.isnot(None))}
        out = []
        try:
            for d in hc.devices()[:60]:
                serial = str(d.get("deviceSerial") or "").upper()
                chans = []
                if d.get("deviceCategory") in (None, 1, 2, 7):  # cámaras, NVR/DVR y domos
                    try:
                        chans = [{"channel": int(c.get("channelNo") or 1), "name": c.get("channelName"),
                                  "online": c.get("status") == 1,
                                  "camera_id": known.get((serial, int(c.get("channelNo") or 1)))}
                                 for c in hc.channels(serial)]
                    except hpp.HppError as exc:
                        chans = [{"error": str(exc)}]
                out.append({"serial": serial, "name": d.get("deviceName"), "model": d.get("deviceType"),
                            "online": d.get("deviceOnlineStatus") == 1, "channels": chans})
        except (hpp.HppError, httpx.HTTPError) as exc:
            raise HTTPException(502, f"Hik-Partner Pro: {exc}") from exc
        return out

    @app.get("/api/alarms")
    def list_alarms(status: str | None = None, client_id: int | None = None, camera_id: int | None = None,
                    limit: int = Query(100, ge=1, le=500), _: User = Depends(staff), s=Depends(db)) -> dict:
        conds = []
        if status:
            conds.append(Alarm.status == status)
        if client_id is not None:
            conds.append(Alarm.client_id == client_id)
        if camera_id is not None:
            conds.append(Alarm.camera_id == camera_id)
        rows = s.execute(select(Alarm, Camera.name, Client.name).outerjoin(Camera, Camera.id == Alarm.camera_id)
                         .outerjoin(Client, Client.id == Alarm.client_id).where(*conds)
                         .order_by(Alarm.id.desc()).limit(limit)).all()
        since = utcnow() - timedelta(days=7)
        cq = select(Alarm.status, func.count()).where(Alarm.received_at >= since)
        if client_id is not None:
            cq = cq.where(Alarm.client_id == client_id)
        counts = dict(s.execute(cq.group_by(Alarm.status)).all())
        return {"items": [alarm_dict(a, cn, kn) for a, cn, kn in rows], "counts_7d": counts}

    @app.post("/api/alarms/{aid}/retry")
    def retry_alarm(aid: int, _: User = Depends(staff), s=Depends(db)) -> dict:
        a = get_or_404(s, Alarm, aid, "Alarma")
        if a.camera_id is None:
            cam = (s.query(Camera).filter(Camera.hpp_serial == a.device_serial, Camera.channel == a.channel)
                   .order_by(Camera.id).first())
            if not cam:
                raise HTTPException(409, "Todavía no hay una cámara con esa serie y canal")
            a.camera_id, a.client_id = cam.id, cam.client_id
        if a.status in ("procesando", "con_persona"):
            raise HTTPException(409, "Esa alarma ya fue analizada o se está analizando")
        a.status, a.attempts, a.error, a.claimed_by, a.claimed_at = "pendiente", 0, None, None, None
        s.commit()
        return alarm_dict(a)

    @app.get("/api/agent/alarms/next")
    def agent_next_alarm(ag: Agent = Depends(current_agent), s=Depends(db)):
        """Entrega a la PC central la próxima alarma lista para bajar su tramo de video."""
        now = utcnow()
        mine = agent_cameras_q(s, ag).filter(Camera.mode.in_(("alarmas", "alarmas_equipo")), Camera.enabled.is_(True))
        cams = {c.id: c for c in mine}
        if not cams:
            return Response(status_code=204)
        stale = now - STALE_CLAIM
        candidates = (s.query(Alarm)
                      .filter(Alarm.camera_id.in_(list(cams)),
                              or_(Alarm.status == "pendiente",
                                  (Alarm.status == "procesando") & (Alarm.claimed_at < stale)))
                      .order_by(Alarm.first_at).limit(20).all())
        for a in candidates:
            cam = cams[a.camera_id]
            cfg = alarm_cfg_of(cam)
            if a.last_at + timedelta(seconds=cfg["post_seconds"] + st.alarm_ready_delay_seconds) > now:
                continue  # todavía se está grabando el "después"
            if a.attempts >= 3:
                a.status, a.error = "error", a.error or "Se agotaron los intentos"
                s.commit()
                continue
            prev_status, prev_claim = a.status, a.claimed_at
            res = s.execute(update(Alarm)
                            .where(Alarm.id == a.id, Alarm.status == prev_status,
                                   (Alarm.claimed_at == prev_claim) if prev_claim else Alarm.claimed_at.is_(None))
                            .values(status="procesando", claimed_by=ag.id, claimed_at=now, attempts=a.attempts + 1))
            s.commit()
            if res.rowcount != 1:
                continue
            client = s.get(Client, cam.client_id)
            tz = ZoneInfo(client.timezone)
            start = (a.first_at - timedelta(seconds=cfg["pre_seconds"])).replace(tzinfo=timezone.utc).astimezone(tz)
            end = (a.last_at + timedelta(seconds=cfg["post_seconds"])).replace(tzinfo=timezone.utc).astimezone(tz)
            return {"alarm_id": a.id, "event_type": a.event_type, "triggers": a.triggers,
                    "start": start.isoformat(), "end": end.isoformat(),
                    "camera": agent_camera_cfg(cam, client), "filter_model": model_spec(s, cam.client_id)}
        return Response(status_code=204)

    def claimed_alarm(s, ag: Agent, aid: int) -> Alarm:
        a = s.get(Alarm, aid)
        if not a or a.claimed_by != ag.id or a.status != "procesando":
            raise HTTPException(409, "Esta alarma ya no está asignada a este agente")
        return a

    @app.post("/api/agent/alarms/{aid}/done")
    def agent_alarm_done(aid: int, body: AlarmDoneIn, ag: Agent = Depends(current_agent), s=Depends(db)) -> dict:
        a = claimed_alarm(s, ag, aid)
        if body.result == "con_persona":
            ev = s.query(Event).filter_by(uuid=body.event_uuid or "").first()
            if not ev or ev.camera_id != a.camera_id:
                raise HTTPException(422, "Falta el evento de la alarma")
            a.event_id = ev.id
        a.status, a.stats, a.error = body.result, body.stats, None
        s.commit()
        return {"ok": True}

    @app.post("/api/agent/alarms/{aid}/fail")
    def agent_alarm_fail(aid: int, body: AlarmFailIn, ag: Agent = Depends(current_agent), s=Depends(db)) -> dict:
        a = claimed_alarm(s, ag, aid)
        a.error = body.error
        a.status = "pendiente" if body.retry and a.attempts < 3 else "error"
        a.claimed_by = a.claimed_at = None
        s.commit()
        return {"ok": True, "status": a.status}

    # ------------------------------------------------------------ eventos
    @app.get("/api/events")
    def list_events(client_id: int | None = None, camera_id: int | None = None, type: str | None = None,
                    label: str | None = None, source: Literal["vivo", "importado", "alarma", "prueba"] | None = None,
                    alert: bool | None = None, since: datetime | None = None, until: datetime | None = None,
                    import_id: int | None = None, order: Literal["reciente", "antiguo"] = "reciente",
                    limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0),
                    user: User = Depends(current_user), s=Depends(db)) -> dict:
        client_id = scope_client(user, client_id)
        conds = []
        if client_id is not None:
            conds.append(Event.client_id == client_id)
        if camera_id is not None:
            conds.append(Event.camera_id == camera_id)
        if type:
            conds.append(Event.type == type)
        if label == "sin_etiqueta":
            conds.append(Event.label.is_(None))
        elif label:
            conds.append(Event.label == label)
        if source:
            conds.append(Event.source == source)
        if alert is not None:
            conds.append(Event.alert == alert)
        if since:
            conds.append(Event.occurred_at >= _naive_utc(since))
        if until:
            conds.append(Event.occurred_at <= _naive_utc(until))
        if import_id is not None:
            conds.append(Event.import_id == import_id)
        total = s.scalar(select(func.count()).select_from(Event).where(*conds)) or 0
        ordering = (Event.occurred_at.desc(), Event.id.desc()) if order == "reciente" else (Event.occurred_at, Event.id)
        rows = s.execute(select(Event, Camera.name, Client.name).join(Camera, Camera.id == Event.camera_id)
                         .join(Client, Client.id == Event.client_id).where(*conds)
                         .order_by(*ordering).limit(limit).offset(offset)).all()
        return {"items": [event_dict(e, cn, kn) for e, cn, kn in rows], "total": total, "limit": limit,
                "offset": offset}

    @app.post("/api/events/manual", status_code=201)
    def manual_event(meta: str = Form(...), snapshot: UploadFile = File(...), user: User = Depends(staff),
                     s=Depends(db)) -> dict:
        """Evento creado desde la pestaña Prueba (análisis en el navegador de la pantalla de HPP).
        Devuelve una URL firmada para subir el clip grabado; después se llama a clip-done."""
        try:
            m = json.loads(meta)
            cam = get_or_404(s, Camera, int(m["camera_id"]), "Cámara")
            etype = m.get("type") if m.get("type") in EVENT_TYPES else "alarma_dvr"
            ext = m.get("ext") if m.get("ext") in ("webm", "mp4") else "webm"
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(422, f"Metadatos inválidos: {exc}") from exc
        data = snapshot.file.read(5 * 1024 * 1024 + 1)
        if len(data) > 5 * 1024 * 1024:
            raise HTTPException(413, "Imagen demasiado grande")
        uid = secrets.token_hex(16)
        snap_path = f"c{cam.client_id}/cam{cam.id}/{uid}.jpg"
        clip_path = f"c{cam.client_id}/cam{cam.id}/{uid}.{ext}"
        try:
            storage.put(snap_path, data, "image/jpeg")
            upload_url = storage.signed_upload_url(clip_path)
        except StorageError as exc:
            raise HTTPException(502, str(exc)) from exc
        ev = Event(uuid=uid, client_id=cam.client_id, camera_id=cam.id, source="prueba", type=etype,
                   severity=SEVERITY[etype], zone=(m.get("zone") or None),
                   description=str(m.get("description", "Prueba desde la pantalla"))[:500],
                   occurred_at=parse_occurred_at(m.get("occurred_at")),
                   features={k: float(v) for k, v in (m.get("features") or {}).items() if k in FEATURE_NAMES},
                   meta={"bbox": m.get("bbox"), "frame_size": m.get("frame_size"), "prueba": True,
                         "clip_path": clip_path}, score=None, alert=True, snapshot_path=snap_path)
        s.add(ev)
        s.commit()
        return {**event_dict(ev, cam.name, s.get(Client, cam.client_id).name), "upload_url": upload_url,
                "content_type": "video/webm" if ext == "webm" else "video/mp4"}

    @app.post("/api/events/{eid}/clip-done")
    def manual_clip_done(eid: int, body: ManualClipIn, _: User = Depends(staff), s=Depends(db)) -> dict:
        e = get_or_404(s, Event, eid, "Evento")
        path = (e.meta or {}).get("clip_path") if e.source == "prueba" else None
        if not path:
            raise HTTPException(409, "Este evento no espera un clip")
        if not storage.exists(path):
            raise HTTPException(409, "El clip no llegó al almacenamiento")
        e.clip_path, e.clip_source = path, "prueba"
        s.commit()
        return event_dict(e, s.get(Camera, e.camera_id).name, s.get(Client, e.client_id).name)

    @app.get("/api/events/{eid}")
    def get_event(eid: int, user: User = Depends(current_user), s=Depends(db)) -> dict:
        e = get_or_404(s, Event, eid, "Evento")
        event_visible(user, e)
        return event_dict(e, s.get(Camera, e.camera_id).name, s.get(Client, e.client_id).name)

    @app.patch("/api/events/{eid}")
    def label_event(eid: int, body: LabelIn, user: User = Depends(staff), s=Depends(db)) -> dict:
        e = get_or_404(s, Event, eid, "Evento")
        e.label, e.note = body.label, body.note
        e.labeled_by, e.labeled_at = (user.id, utcnow()) if body.label else (None, None)
        s.commit()
        return event_dict(e, s.get(Camera, e.camera_id).name, s.get(Client, e.client_id).name)

    @app.get("/api/events/{eid}/snapshot")
    def event_snapshot(eid: int, user: User = Depends(current_user), s=Depends(db)):
        e = get_or_404(s, Event, eid, "Evento")
        event_visible(user, e)
        return redirect_to(e.snapshot_path)

    @app.get("/api/events/{eid}/clip")
    def event_clip(eid: int, user: User = Depends(current_user), s=Depends(db)):
        e = get_or_404(s, Event, eid, "Evento")
        event_visible(user, e)
        return redirect_to(e.clip_path)

    # ------------------------------------------------------------ avisos al cliente
    def local_dt(dt: datetime, tz: str) -> datetime:
        try:
            from zoneinfo import ZoneInfo
            return dt.replace(tzinfo=timezone.utc).astimezone(ZoneInfo(tz))
        except Exception:
            return dt

    def default_message(ev: Event, cam_name: str, client: Client) -> str:
        when = local_dt(ev.occurred_at, client.timezone).strftime("%d/%m/%Y %H:%M")
        return (f"Detectamos una situación sospechosa (posible hurto) en la cámara «{cam_name}» "
                f"el {when}. Te compartimos el clip para que lo revises.")

    @app.post("/api/events/{eid}/report", status_code=201)
    def send_report(eid: int, body: ReportIn, user: User = Depends(staff), s=Depends(db)) -> dict:
        ev = get_or_404(s, Event, eid, "Evento")
        if not ev.clip_path:
            raise HTTPException(409, "El evento todavía no tiene clip para enviar")
        client = s.get(Client, ev.client_id)
        cam = s.get(Camera, ev.camera_id)
        message = (body.message or "").strip() or default_message(ev, cam.name, client)
        token = secrets.token_urlsafe(24)
        days = body.expires_days or st.report_link_days
        rep = ClientReport(event_id=ev.id, client_id=ev.client_id, public_token=token, message=message,
                           sent_by=user.id, sent_at=utcnow(), expires_at=utcnow() + timedelta(days=days))
        s.add(rep)
        s.commit()
        # la miniatura solo se usa si se va a enviar por Telegram
        wants = set(body.channels) if body.channels is not None else set(notifier.available(client))
        snapshot = None
        if ev.snapshot_path and "telegram" in wants and "telegram" in notifier.available(client):
            try:
                snapshot = storage.get(ev.snapshot_path)
            except Exception:
                snapshot = None
        channels, delivery, status = notifier.send(client, cam.name, ev, message, report_link(token),
                                                    snapshot, body.channels)
        rep.channels, rep.delivery, rep.status = channels, delivery, status
        s.commit()
        return report_dict(rep, for_client=False, ev=ev, cam_name=cam.name, client_name=client.name)

    @app.get("/api/reports")
    def list_reports(client_id: int | None = None, limit: int = Query(50, ge=1, le=200),
                     offset: int = Query(0, ge=0), user: User = Depends(current_user), s=Depends(db)) -> dict:
        client_id = scope_client(user, client_id)
        conds = []
        if client_id is not None:
            conds.append(ClientReport.client_id == client_id)
        total = s.scalar(select(func.count()).select_from(ClientReport).where(*conds)) or 0
        rows = s.execute(select(ClientReport, Event, Camera.name, Client.name)
                         .join(Event, Event.id == ClientReport.event_id)
                         .join(Camera, Camera.id == Event.camera_id)
                         .join(Client, Client.id == ClientReport.client_id).where(*conds)
                         .order_by(ClientReport.sent_at.desc(), ClientReport.id.desc())
                         .limit(limit).offset(offset)).all()
        for_client = user.role == "cliente"
        return {"items": [report_dict(r, for_client, ev, cn, kn) for r, ev, cn, kn in rows],
                "total": total, "limit": limit, "offset": offset}

    @app.delete("/api/reports/{rid}", status_code=204)
    def revoke_report(rid: int, _: User = Depends(staff), s=Depends(db)) -> None:
        r = get_or_404(s, ClientReport, rid, "Aviso")
        s.delete(r)
        s.commit()

    # --- vista pública del clip (sin login, por link con token; solo lectura) ---
    def _report_by_token(s, token: str) -> ClientReport:
        r = s.query(ClientReport).filter_by(public_token=token).first()
        if not r:
            raise HTTPException(404, "El enlace no existe")
        if r.expires_at < utcnow():
            raise HTTPException(410, "El enlace venció")
        return r

    @app.get("/api/r/{token}")
    def public_report(token: str, s=Depends(db)) -> dict:
        r = _report_by_token(s, token)
        if not r.viewed_at:
            r.viewed_at = utcnow()
            s.commit()
        ev = s.get(Event, r.event_id)
        cam = s.get(Camera, ev.camera_id) if ev else None
        client = s.get(Client, r.client_id)
        return {"client_name": client.name if client else "", "camera_name": cam.name if cam else "",
                "message": r.message, "occurred_at": iso(ev.occurred_at) if ev else None,
                "has_clip": bool(ev and ev.clip_path), "has_snapshot": bool(ev and ev.snapshot_path)}

    @app.get("/api/r/{token}/clip")
    def public_report_clip(token: str, s=Depends(db)):
        r = _report_by_token(s, token)
        ev = s.get(Event, r.event_id)
        return redirect_to(ev.clip_path if ev else None)

    @app.get("/api/r/{token}/snapshot")
    def public_report_snapshot(token: str, s=Depends(db)):
        r = _report_by_token(s, token)
        ev = s.get(Event, r.event_id)
        return redirect_to(ev.snapshot_path if ev else None)

    @app.get("/api/stats")
    def stats(client_id: int | None = None, days: int = Query(30, ge=1, le=365),
              user: User = Depends(current_user), s=Depends(db)) -> dict:
        client_id = scope_client(user, client_id)
        q = (select(Event.type, Event.label, Event.alert, Event.occurred_at, Camera.name)
             .join(Camera, Camera.id == Event.camera_id)
             .where(Event.occurred_at >= utcnow() - timedelta(days=days), Event.source.in_(("vivo", "alarma", "prueba"))))
        if client_id is not None:
            q = q.where(Event.client_id == client_id)
        rows = s.execute(q).all()
        by_day = Counter(r.occurred_at.date().isoformat() for r in rows)
        return {"days": days, "total": len(rows), "alerts": sum(1 for r in rows if r.alert),
                "by_type": dict(Counter(r.type for r in rows)),
                "by_label": dict(Counter(r.label or "sin_etiqueta" for r in rows)),
                "by_camera": dict(Counter(r.name for r in rows)),
                "by_day": [{"date": d, "count": n} for d, n in sorted(by_day.items())]}

    # ------------------------------------------------------------ videos importados
    @app.post("/api/imports", status_code=201)
    def create_import(body: ImportIn, user: User = Depends(staff), s=Depends(db)) -> dict:
        cam = get_or_404(s, Camera, body.camera_id, "Cámara")
        name = Path(body.filename).name
        if Path(name).suffix.lower() not in VIDEO_EXT:
            raise HTTPException(422, f"Formato no soportado. Usar: {', '.join(sorted(VIDEO_EXT))}")
        if body.size and body.size > st.max_import_bytes:
            raise HTTPException(413, "El video es demasiado grande")
        rec = None
        if body.recorded_at:
            try:
                rec = datetime.fromisoformat(body.recorded_at).replace(tzinfo=None)
            except ValueError as exc:
                raise HTTPException(422, "recorded_at inválido (usar AAAA-MM-DDTHH:MM)") from exc
        imp = VideoImport(client_id=cam.client_id, camera_id=cam.id, filename=name, storage_path="",
                          size_bytes=body.size, recorded_at=rec, created_by=user.id)
        s.add(imp)
        s.flush()
        safe = _SAFE_NAME.sub("_", name)[-120:]
        imp.storage_path = f"imports/{imp.id}/{safe}"
        try:
            upload_url = storage.signed_upload_url(imp.storage_path)
        except StorageError as exc:
            s.rollback()
            raise HTTPException(502, str(exc)) from exc
        s.commit()
        return {**import_dict(imp, cam.name), "upload_url": upload_url}

    @app.post("/api/imports/{iid}/uploaded")
    def import_uploaded(iid: int, _: User = Depends(staff), s=Depends(db)) -> dict:
        i = get_or_404(s, VideoImport, iid, "Video")
        if i.status != "subiendo":
            raise HTTPException(409, "El video ya fue registrado")
        if not storage.exists(i.storage_path):
            raise HTTPException(409, "El archivo no llegó al almacenamiento; volvé a subirlo")
        i.status = "en_cola"
        s.commit()
        return import_dict(i)

    @app.get("/api/imports")
    def list_imports(client_id: int | None = None, _: User = Depends(staff), s=Depends(db)) -> list[dict]:
        q = s.query(VideoImport, Camera.name).join(Camera, Camera.id == VideoImport.camera_id)
        if client_id is not None:
            q = q.filter(VideoImport.client_id == client_id)
        out = []
        online = [a for a in s.query(Agent).all()
                  if a.process_imports and a.last_seen and utcnow() - a.last_seen < timedelta(minutes=3)]
        for i, n in q.order_by(VideoImport.id.desc()).all():
            d = import_dict(i, n)
            cam = s.get(Camera, i.camera_id)
            d["agent_available"] = any(agent_owns(a, cam) for a in online)
            out.append(d)
        return out

    @app.get("/api/imports/{iid}")
    def get_import(iid: int, _: User = Depends(staff), s=Depends(db)) -> dict:
        i = get_or_404(s, VideoImport, iid, "Video")
        d = import_dict(i, s.get(Camera, i.camera_id).name)
        d["incidents"] = [{"id": x.id, "start_s": x.start_s, "end_s": x.end_s, "note": x.note}
                          for x in s.query(Incident).filter_by(import_id=iid).order_by(Incident.start_s)]
        return d

    @app.get("/api/imports/{iid}/video")
    def import_video(iid: int, _: User = Depends(staff), s=Depends(db)):
        i = get_or_404(s, VideoImport, iid, "Video")
        return redirect_to(i.storage_path if i.status != "subiendo" else None)

    @app.post("/api/imports/{iid}/reprocess")
    def reprocess_import(iid: int, _: User = Depends(staff), s=Depends(db)) -> dict:
        i = get_or_404(s, VideoImport, iid, "Video")
        if i.status in ("subiendo", "en_cola", "procesando"):
            raise HTTPException(409, "Ya está en proceso")
        old = s.query(Event).filter_by(import_id=iid).all()
        delete_event_media(old)
        for e in old:
            s.delete(e)
        i.status, i.progress, i.error, i.claimed_by, i.claimed_at = "en_cola", 0.0, None, None, None
        s.commit()
        return import_dict(i)

    @app.delete("/api/imports/{iid}", status_code=204)
    def delete_import(iid: int, _: User = Depends(staff), s=Depends(db)) -> None:
        i = get_or_404(s, VideoImport, iid, "Video")
        if i.status == "procesando":
            raise HTTPException(409, "Esperá a que termine de procesarse")
        old = s.query(Event).filter_by(import_id=iid).all()
        delete_event_media(old)
        for e in old:
            s.delete(e)
        s.query(Incident).filter_by(import_id=iid).delete()
        storage.remove([i.storage_path])
        s.delete(i)
        s.commit()

    @app.post("/api/imports/{iid}/incidents", status_code=201)
    def add_incident(iid: int, body: IncidentIn, _: User = Depends(staff), s=Depends(db)) -> dict:
        get_or_404(s, VideoImport, iid, "Video")
        if body.end_s < body.start_s:
            raise HTTPException(422, "El fin no puede ser anterior al inicio")
        x = Incident(import_id=iid, **body.model_dump())
        s.add(x)
        s.commit()
        return {"id": x.id, "start_s": x.start_s, "end_s": x.end_s, "note": x.note}

    @app.delete("/api/imports/{iid}/incidents/{xid}", status_code=204)
    def delete_incident(iid: int, xid: int, _: User = Depends(staff), s=Depends(db)) -> None:
        x = s.get(Incident, xid)
        if not x or x.import_id != iid:
            raise HTTPException(404, "Incidente no encontrado")
        s.delete(x)
        s.commit()

    @app.get("/api/imports/{iid}/evaluation")
    def evaluate_import(iid: int, tolerance_s: float = Query(10.0, ge=0, le=120), _: User = Depends(staff),
                        s=Depends(db)) -> dict:
        i = get_or_404(s, VideoImport, iid, "Video")
        incidents = s.query(Incident).filter_by(import_id=iid).order_by(Incident.start_s).all()
        events = s.query(Event).filter_by(import_id=iid).order_by(Event.video_ts).all()
        rows, matched = [], set()
        for x in incidents:
            hits = [e for e in events if x.start_s - tolerance_s <= e.video_ts <= x.end_s + tolerance_s]
            matched.update(e.id for e in hits)
            rows.append({"id": x.id, "start_s": x.start_s, "end_s": x.end_s, "note": x.note,
                         "detected": bool(hits), "alerted": any(e.alert for e in hits),
                         "event_ids": [e.id for e in hits]})
        outside = [e for e in events if e.id not in matched]
        hours = (i.duration_s or 0) / 3600
        return {"import_id": iid, "tolerance_s": tolerance_s, "duration_s": i.duration_s,
                "incidents": len(incidents), "detected": sum(r["detected"] for r in rows),
                "alerted": sum(r["alerted"] for r in rows), "events": len(events),
                "false_alarms": len(outside), "false_alerts": sum(1 for e in outside if e.alert),
                "false_alarms_per_hour": round(len(outside) / hours, 2) if hours else None,
                "detail": rows, "false_alarm_event_ids": [e.id for e in outside]}

    # ------------------------------------------------------------ entrenamiento
    @app.get("/api/training/summary")
    def training_summary(_: User = Depends(staff), s=Depends(db)) -> dict:
        rows = s.execute(select(Event.client_id, Event.label, func.count()).group_by(Event.client_id, Event.label)).all()
        names = {c.id: c.name for c in s.query(Client).all()}
        per: dict[int, dict] = {}
        for cid, label, n in rows:
            d = per.setdefault(cid, {"client_id": cid, "client_name": names.get(cid), "labels": {}, "total": 0})
            d["labels"][label or "sin_etiqueta"] = n
            d["total"] += n
        return {"clients": list(per.values()), "min_total": training.MIN_TOTAL,
                "min_per_class": training.MIN_PER_CLASS}

    def model_dict(m: FilterModel, names: dict) -> dict:
        return {"id": m.id, "client_id": m.client_id, "client_name": names.get(m.client_id) if m.client_id else None,
                "scope": "cliente" if m.client_id else "global", "metrics": m.metrics, "n_samples": m.n_samples,
                "active": m.active, "created_at": iso(m.created_at), "threshold": m.spec.get("threshold")}

    @app.post("/api/training/run", status_code=201)
    def run_training(body: TrainIn, user: User = Depends(staff), s=Depends(db)) -> dict:
        q = s.query(Event).filter(Event.label.isnot(None))
        if body.client_id is not None:
            get_or_404(s, Client, body.client_id, "Cliente")
            q = q.filter(Event.client_id == body.client_id)
        events = [{"type": e.type, "features": e.features, "label": e.label, "camera_id": e.camera_id} for e in q]
        try:
            spec, metrics = training.train(events, body.target_recall)
        except training.NotEnoughData as exc:
            raise HTTPException(422, str(exc)) from exc
        m = FilterModel(client_id=body.client_id, spec=spec, metrics=metrics, n_samples=metrics["n"],
                        created_by=user.id)
        s.add(m)
        s.commit()
        return model_dict(m, {c.id: c.name for c in s.query(Client).all()})

    @app.get("/api/models")
    def list_models(_: User = Depends(staff), s=Depends(db)) -> list[dict]:
        names = {c.id: c.name for c in s.query(Client).all()}
        return [model_dict(m, names) for m in s.query(FilterModel).order_by(FilterModel.id.desc()).all()]

    @app.post("/api/models/{mid}/activate")
    def activate_model(mid: int, _: User = Depends(staff), s=Depends(db)) -> dict:
        m = get_or_404(s, FilterModel, mid, "Modelo")
        s.query(FilterModel).filter(FilterModel.client_id.is_(None) if m.client_id is None
                                    else FilterModel.client_id == m.client_id).update({"active": False})
        m.active = True
        s.commit()
        return model_dict(m, {c.id: c.name for c in s.query(Client).all()})

    @app.post("/api/models/{mid}/deactivate")
    def deactivate_model(mid: int, _: User = Depends(staff), s=Depends(db)) -> dict:
        m = get_or_404(s, FilterModel, mid, "Modelo")
        m.active = False
        s.commit()
        return model_dict(m, {c.id: c.name for c in s.query(Client).all()})

    @app.get("/api/training/export.csv")
    def export_labels(_: User = Depends(staff), s=Depends(db)) -> Response:
        import csv
        import io

        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["event_id", "client_id", "camera_id", "source", "type", "label", "score", *FEATURE_NAMES])
        for e in s.query(Event).filter(Event.label.isnot(None)).order_by(Event.id):
            w.writerow([e.id, e.client_id, e.camera_id, e.source, e.type, e.label, e.score,
                        *[(e.features or {}).get(k, "") for k in FEATURE_NAMES]])
        return Response(buf.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": "attachment; filename=etiquetas.csv"})

    # ------------------------------------------------------------ limpieza diaria (Vercel Cron)
    @app.get("/api/cron/cleanup")
    def cron_cleanup(authorization: str = Header(""), s=Depends(db)) -> dict:
        if not st.cron_secret or not secrets.compare_digest(authorization, f"Bearer {st.cron_secret}"):
            raise HTTPException(401, "No autorizado")
        cutoff = utcnow() - timedelta(days=st.retention_days)
        old = (s.query(Event).filter(Event.source.in_(("vivo", "alarma", "prueba")), Event.received_at < cutoff)
               .order_by(Event.id).limit(2000).all())
        unlabeled = [e for e in old if not e.label]
        labeled = [e for e in old if e.label and (e.snapshot_path or e.clip_path)]
        # sin etiqueta: se borran; etiquetados: se borra el video pero se conservan los datos para entrenar
        delete_event_media(unlabeled + labeled)
        for e in unlabeled:
            s.delete(e)
        for e in labeled:
            e.snapshot_path = e.clip_path = None
        # alarmas sin evento (sin persona, sin cámara, error): solo sirven de estadística, se guardan 14 días
        s.query(Alarm).filter(Alarm.event_id.is_(None), Alarm.status != "procesando",
                              Alarm.received_at < utcnow() - timedelta(days=14)).delete(synchronize_session=False)
        s.query(SessionToken).filter(SessionToken.expires_at < utcnow()).delete()
        s.commit()
        return {"deleted": len(unlabeled), "media_removed": len(labeled), "retention_days": st.retention_days}

    if not st.serve_public:
        @app.get("/", include_in_schema=False)
        def root():
            return RedirectResponse("/index.html", status_code=307)

    if st.serve_public and PUBLIC_DIR.exists():
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="public")
    return app


def _misconfigured_app(missing: list[str]) -> FastAPI:
    app = FastAPI(title="Vigía (falta configuración)")
    msg = ("Falta configurar en Vercel (Settings > Environment Variables) y volver a desplegar: "
           + ", ".join(missing))
    log.error(msg)

    @app.api_route("/api/{rest:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    def not_configured(rest: str):
        raise HTTPException(503, msg)

    return app


def _naive_utc(dt: datetime | None) -> datetime | None:
    if dt is None or dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)
