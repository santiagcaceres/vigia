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

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, or_, select, update

from core.rules import RULE_TYPES, SEVERITY, parse_config
from core.scorer import FEATURE_NAMES, LABELS

from . import training
from .db import (ROLES, Agent, Camera, Client, ClientReport, Event, FilterModel, Incident, SessionToken, User,
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
    clip_pre_seconds: float = 6.0
    clip_post_seconds: float = 6.0
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
            on_vercel=bool(os.getenv("VERCEL")),
            operator_chat_id=os.getenv("OPERATOR_TELEGRAM_CHAT_ID") or None,
            operator_webhook=os.getenv("OPERATOR_WEBHOOK_URL") or None,
            twilio_sid=os.getenv("TWILIO_ACCOUNT_SID") or None,
            twilio_token=os.getenv("TWILIO_AUTH_TOKEN") or None,
            twilio_whatsapp_from=os.getenv("TWILIO_WHATSAPP_FROM") or None,
            resend_api_key=os.getenv("RESEND_API_KEY") or None,
            email_from=os.getenv("EMAIL_FROM") or None,
            report_link_days=int(os.getenv("REPORT_LINK_DAYS", "14")),
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
    client_id: int
    name: str = Field(min_length=1, max_length=120)
    process_imports: bool = True


class CameraIn(BaseModel):
    client_id: int
    agent_id: int | None = None
    name: str = Field(min_length=1, max_length=120)
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
        return {"id": a.id, "client_id": a.client_id, "name": a.name, "last_seen": iso(a.last_seen),
                "online": online, "status": a.last_status or {}, "process_imports": a.process_imports}

    def camera_dict(c: Camera, user: User, status: dict | None = None) -> dict:
        d = {"id": c.id, "client_id": c.client_id, "agent_id": c.agent_id, "name": c.name, "enabled": c.enabled,
             "detector": c.detector, "process_fps": c.process_fps, "zones": c.zones or [], "rules": c.rules or {},
             "has_preview": bool(c.preview_path), "preview_at": iso(c.preview_at), "status": status}
        if user.role != "cliente":
            d.update(host=c.host, username=c.username, has_password=bool(c.password_enc), channel=c.channel,
                     rtsp_port=c.rtsp_port, http_port=c.http_port, detect_stream=c.detect_stream,
                     nvr_clips=c.nvr_clips, playback_time=c.playback_time, has_source_url=bool(c.source_url_enc))
        return d

    def event_dict(e: Event, cam_name: str | None = None, client_name: str | None = None) -> dict:
        return {
            "id": e.id, "uuid": e.uuid, "client_id": e.client_id, "client_name": client_name,
            "camera_id": e.camera_id, "camera_name": cam_name, "source": e.source, "import_id": e.import_id,
            "type": e.type, "severity": e.severity, "zone": e.zone, "description": e.description,
            "video_ts": e.video_ts, "occurred_at": iso(e.occurred_at), "score": e.score, "alert": e.alert,
            "features": e.features or {}, "label": e.label, "note": e.note, "labeled_at": iso(e.labeled_at),
            "snapshot_url": f"/api/events/{e.id}/snapshot" if e.snapshot_path else None,
            "clip_url": f"/api/events/{e.id}/clip" if e.clip_path else None, "clip_source": e.clip_source,
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
        return {"rule_types": list(RULE_TYPES), "severity": SEVERITY, "labels": list(LABELS),
                "zone_types": ["shelf", "checkout", "exit", "restricted"], "roles": list(ROLES)}

    # ------------------------------------------------------------ clientes y usuarios
    @app.get("/api/clients")
    def list_clients(user: User = Depends(current_user), s=Depends(db)) -> list[dict]:
        q = s.query(Client).order_by(Client.name)
        if user.role == "cliente":
            q = q.filter(Client.id == user.client_id)
        return [client_dict(c) for c in q.all()]

    @app.post("/api/clients", status_code=201)
    def create_client(body: ClientIn, _: User = Depends(admin_only), s=Depends(db)) -> dict:
        if s.query(Client).filter_by(name=body.name).first():
            raise HTTPException(409, "Ya existe un cliente con ese nombre")
        c = Client(**body.model_dump())
        s.add(c)
        s.commit()
        return client_dict(c)

    @app.put("/api/clients/{cid}")
    def update_client(cid: int, body: ClientIn, _: User = Depends(admin_only), s=Depends(db)) -> dict:
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
            for cs in (a.last_status or {}).get("cameras", []):
                out[cs["camera_id"]] = {**cs, "agent_online": online}
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
            if ag.client_id != body.client_id:
                raise HTTPException(422, "El agente pertenece a otro cliente")
        if not body.host and not body.source_url:
            raise HTTPException(422, "Indicá la IP del NVR/cámara o una URL RTSP")

    @app.post("/api/cameras", status_code=201)
    def create_camera(body: CameraIn, user: User = Depends(admin_only), s=Depends(db)) -> dict:
        check_camera_body(s, body)
        c = Camera(**body.model_dump(exclude={"password", "source_url"}), password_enc=box.encrypt(body.password),
                   source_url_enc=box.encrypt(body.source_url))
        s.add(c)
        s.commit()
        return camera_dict(c, user)

    @app.put("/api/cameras/{cam_id}")
    def update_camera(cam_id: int, body: CameraIn, user: User = Depends(admin_only), s=Depends(db)) -> dict:
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
        storage.remove([c.preview_path] if c.preview_path else [])
        s.delete(c)
        s.commit()

    @app.get("/api/cameras/{cam_id}/preview")
    def camera_preview(cam_id: int, user: User = Depends(current_user), s=Depends(db)):
        c = get_or_404(s, Camera, cam_id, "Cámara")
        scope_client(user, c.client_id)
        resp = redirect_to(c.preview_path)
        resp.headers["Cache-Control"] = "no-store"
        return resp

    # ------------------------------------------------------------ API del agente
    def agent_camera(s, ag: Agent, cam_id: int) -> Camera:
        c = s.get(Camera, cam_id)
        if not c or c.client_id != ag.client_id:
            raise HTTPException(404, "Cámara no encontrada para este agente")
        return c

    def agent_event_or_404(s, ag: Agent, uuid: str) -> Event:
        ev = s.query(Event).filter_by(uuid=uuid).first()
        if not ev or ev.client_id != ag.client_id:
            raise HTTPException(404, "Evento no encontrado")
        return ev

    def camera_analysis(c: Camera) -> dict:
        return {"zones": c.zones or [], "rules": c.rules or {}}

    def model_spec(s, client_id: int) -> dict | None:
        m = active_model(s, client_id)
        return dict(m.spec, id=m.id) if m else None

    @app.get("/api/agent/config")
    def agent_config(ag: Agent = Depends(current_agent), s=Depends(db)) -> dict:
        client = s.get(Client, ag.client_id)
        cams = s.query(Camera).filter(Camera.client_id == ag.client_id,
                                      or_(Camera.agent_id == ag.id, Camera.agent_id.is_(None))).all()
        return {
            "agent_id": ag.id, "client_id": ag.client_id, "process_imports": ag.process_imports,
            "settings": {"clip_pre_seconds": st.clip_pre_seconds, "clip_post_seconds": st.clip_post_seconds},
            "filter_model": model_spec(s, ag.client_id),
            "cameras": [{
                "id": c.id, "name": c.name, "enabled": c.enabled, "detector": c.detector,
                "process_fps": c.process_fps,
                "connection": {"host": c.host, "username": c.username, "password": box.decrypt(c.password_enc),
                               "channel": c.channel, "rtsp_port": c.rtsp_port, "http_port": c.http_port,
                               "detect_stream": c.detect_stream, "nvr_clips": c.nvr_clips,
                               "playback_time": c.playback_time, "timezone": client.timezone,
                               "source_url": box.decrypt(c.source_url_enc) or None},
                "analysis": camera_analysis(c),
            } for c in cams],
        }

    @app.post("/api/agent/status")
    def agent_status(payload: dict, ag: Agent = Depends(current_agent), s=Depends(db)) -> dict:
        a = s.get(Agent, ag.id)
        a.last_status = {"cameras": payload.get("cameras", [])[:200],
                         "outbox_pending": payload.get("outbox_pending"), "version": payload.get("version"),
                         "at": iso(utcnow())}
        a.last_seen = utcnow()
        s.commit()
        return {"ok": True}

    @app.post("/api/agent/events", status_code=201)
    def agent_event(meta: str = Form(...), snapshot: UploadFile = File(...),
                          ag: Agent = Depends(current_agent), s=Depends(db)) -> dict:
        try:
            m = json.loads(meta)
            uuid = str(m["uuid"])
            cam = agent_camera(s, ag, int(m["camera_id"]))
            if m["type"] not in RULE_TYPES:
                raise ValueError(f"tipo desconocido {m['type']}")
            if not uuid.isalnum() or len(uuid) > 40:
                raise ValueError("uuid inválido")
            source = m.get("source", "vivo")
            if source not in ("vivo", "importado"):
                raise ValueError("source inválido")
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(422, f"Metadatos inválidos: {exc}") from exc
        import_id = None
        if source == "importado":
            imp = s.get(VideoImport, int(m.get("import_id") or 0))
            if not imp or imp.client_id != ag.client_id or imp.claimed_by != ag.id or imp.camera_id != cam.id:
                raise HTTPException(404, "Importación no encontrada para este agente")
            import_id = imp.id
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
                   meta={"bbox": m.get("bbox"), "frame_size": m.get("frame_size")},
                   score=m.get("score"), alert=bool(m.get("alert", True)), snapshot_path=snap_path)
        s.add(ev)
        s.commit()
        if ev.alert and source == "vivo":
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
        candidates = (s.query(VideoImport)
                      .filter(VideoImport.client_id == ag.client_id,
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

    # ------------------------------------------------------------ eventos
    @app.get("/api/events")
    def list_events(client_id: int | None = None, camera_id: int | None = None, type: str | None = None,
                    label: str | None = None, source: Literal["vivo", "importado"] | None = None,
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
             .where(Event.occurred_at >= utcnow() - timedelta(days=days), Event.source == "vivo"))
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
        online = {a.client_id for a in s.query(Agent).all()
                  if a.process_imports and a.last_seen and utcnow() - a.last_seen < timedelta(minutes=3)}
        for i, n in q.order_by(VideoImport.id.desc()).all():
            d = import_dict(i, n)
            d["agent_available"] = i.client_id in online
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
        old = (s.query(Event).filter(Event.source == "vivo", Event.received_at < cutoff)
               .order_by(Event.id).limit(2000).all())
        unlabeled = [e for e in old if not e.label]
        labeled = [e for e in old if e.label and (e.snapshot_path or e.clip_path)]
        # sin etiqueta: se borran; etiquetados: se borra el video pero se conservan los datos para entrenar
        delete_event_media(unlabeled + labeled)
        for e in unlabeled:
            s.delete(e)
        for e in labeled:
            e.snapshot_path = e.clip_path = None
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
