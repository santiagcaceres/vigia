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

from core.metrics import empty_bucket, merge_bucket
from core.rules import EVENT_TYPES, SEVERITY, parse_config
from core.schedule import DEFAULT_MARGIN_MIN, normalize_hours
from core.scorer import FEATURE_NAMES, LABELS

from . import hikconnect, hpp, pool, probe, training
from .db import (ROLES, Agent, Alarm, AppSetting, Camera, CameraMetric, Client, ClientReport, Event, FilterModel, Incident, SessionToken, User,
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
    # Actualización de la app de escritorio: última versión publicada en GitHub Releases
    github_repo: str = "santiagcaceres/vigia"
    github_token: str | None = None  # solo si el repositorio pasa a privado: token de solo lectura (Contents: read)
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
            github_repo=os.getenv("GITHUB_REPO", "santiagcaceres/vigia"),
            github_token=os.getenv("GITHUB_TOKEN") or None,
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
    plans: list[Literal["hurto", "comercio", "estacion"]] = ["hurto"]
    # Acceso a la web del cliente (opcional). Al crear: usuario y clave. Al editar: cambia el usuario
    # existente (la clave solo si viene) o lo crea si el cliente no tenía.
    web_username: str | None = Field(None, max_length=80)
    web_password: str | None = Field(None, max_length=200)
    # Hik-Connect for Teams (video de Hikvision sin abrir puertos). Solo se tocan si vienen en el pedido;
    # el SecretKey vacío deja el que estaba.
    hc_app_key: str | None = Field(None, max_length=128)
    hc_secret: str | None = Field(None, max_length=256)
    hc_api_url: str | None = Field(None, max_length=255)
    hc_code: str | None = Field(None, max_length=64)  # código de cifrado de video; vacío = no cambia, "-" = borrar
    # horario del local: {"lun": "09:00-17:00", ..., "dom": ""} (core/schedule.py). Solo se toca si viene;
    # {} o todo vacío = sin horario (se mira las 24 h)
    hours: dict[str, str] | None = None
    # dónde se analizan sus cámaras: general (ajuste de Administración) | pcs | servidores. None = no cambia
    analysis_location: Literal["general", "pcs", "servidores"] | None = None

    def client_fields(self) -> dict:
        return self.model_dump(exclude={"web_username", "web_password", "hc_app_key", "hc_secret", "hc_api_url",
                                        "hc_code", "hours", "analysis_location"})


PLAN_NAMES = {"hurto": "Detección de hurtos", "comercio": "Métricas de comercio", "estacion": "Estación de servicio"}


class MetricBucketIn(BaseModel):
    camera_id: int
    bucket_start: str
    data: dict


class MetricsIn(BaseModel):
    buckets: list[MetricBucketIn] = Field(default_factory=list, max_length=500)


class UserIn(BaseModel):
    username: str = Field(min_length=3, max_length=80)
    password: str | None = Field(None, min_length=8)
    role: Literal["admin", "entrenador", "cliente"]
    client_id: int | None = None
    active: bool = True


class PasswordIn(BaseModel):
    current: str
    new: str = Field(min_length=8)


class UserCamerasIn(BaseModel):
    camera_ids: list[int] = Field(default_factory=list, max_length=2000)


class CamerasDeleteIn(BaseModel):
    camera_ids: list[int] = Field(min_length=1, max_length=2000)
    con_alertas: bool = False  # borrar también sus alertas, clips y videos subidos


class EnrollIn(BaseModel):
    """Alta de una máquina de análisis (servidor) con la clave de máquinas, sin usuario ni clave de nadie."""
    key: str = Field(min_length=10, max_length=200)
    name: str = Field(min_length=1, max_length=120)
    instance_id: str | None = Field(None, max_length=80)
    capacity: float | None = Field(None, gt=0, le=1000)
    process_imports: bool = False


class AnalysisSettingsIn(BaseModel):
    # pcs: cada usuario analiza en su PC las cámaras que revisa. servidores: las analizan las máquinas de
    # análisis (en la nube) y los usuarios solo revisan
    mode: Literal["pcs", "servidores"] | None = None
    capacity: float | None = Field(None, gt=0, le=1000)  # cámaras de hurto a 5 fps por máquina
    spare: int | None = Field(None, ge=0, le=20)  # máquinas de reserva prendidas de más


class AgentLoginIn(BaseModel):
    username: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=1, max_length=200)
    pc_name: str = Field(min_length=1, max_length=120)


class AgentIn(BaseModel):
    client_id: int | None = None  # None = agente central (atiende cámaras asignadas de cualquier cliente)
    name: str = Field(min_length=1, max_length=120)
    process_imports: bool = True


class AlarmCfgIn(BaseModel):
    event_types: list[Literal[hpp.VIDEO_EVENT_TYPES]] = list(hpp.DEFAULT_EVENT_TYPES)  # type: ignore[valid-type]
    pre_seconds: float = Field(10.0, ge=0, le=60)
    post_seconds: float = Field(20.0, ge=3, le=90)


MIN_FPS = 5.0  # cuadros por segundo analizados, como mínimo, en las cámaras que buscan hurtos
# Cámaras solo de métricas (cliente sin plan «hurto» o tipo «Solo métricas»): contar gente y medir tiempos
# no necesita 5 fps ni guardar clips. Con zona de entrada o surtidor un poco más, para no perder a quien cruza.
METRICS_FPS = 2.0
METRICS_COUNT_FPS = 3.0

# Tipo de cámara (qué mira): qué reglas de hurto se aplican si no se eligieron a mano en «Zonas y reglas».
CAMERA_PROFILES = {
    "general": {"name": "General (todas las reglas)", "rules": None},
    "gondola": {"name": "Góndola / pasillo", "rules": ["merodeo", "salida_sin_pago", "zona_restringida", "fuera_de_horario"]},
    "caja": {"name": "Caja", "rules": ["zona_restringida", "fuera_de_horario"]},
    "entrada": {"name": "Entrada / puerta", "rules": ["salida_sin_pago", "fuera_de_horario"]},
    "deposito": {"name": "Depósito / trastienda", "rules": ["zona_restringida", "fuera_de_horario"],
                 "whole_zone": "restricted"},
    "exterior": {"name": "Calle / exterior", "rules": ["fuera_de_horario"], "vehicles": True},
    "surtidor": {"name": "Surtidor (estación)", "rules": [], "vehicles": True},
    "metricas": {"name": "Solo métricas (conteo y tiempos, sin hurto)", "rules": []},
}
OFF_HOURS = ("apagada", "alarmas", "vivo")


class CameraIn(BaseModel):
    client_id: int
    agent_id: int | None = None
    name: str = Field(min_length=1, max_length=120)
    # hikconnect = Hikvision por la nube; ip = por IP y puertos, la PC central detecta si es Hikvision o Dahua
    brand: Literal["hikvision", "dahua", "hikconnect", "ip"] = "hikvision"
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
    process_fps: float = Field(5.0, ge=MIN_FPS, le=15)  # mínimo 5 para todas (decisión del 24 oct)
    enabled: bool = True
    # vivo: el agente mira el stream 24 h. alarmas: Hik-Partner Pro avisa (webhook). alarmas_equipo:
    # el propio grabador avisa (conexión larga desde el agente); vale para cualquier marca.
    mode: Literal["vivo", "alarmas", "alarmas_equipo"] = "vivo"
    hpp_serial: str | None = Field(None, max_length=64)  # la larga de Hik-Connect se acorta al validar
    hc_resource_id: str | None = Field(None, max_length=64)
    profile: Literal["general", "gondola", "caja", "entrada", "deposito", "exterior", "surtidor", "metricas"] = "general"
    # con el local cerrado (según el horario del cliente). None = no cambia (al crear: apagada)
    off_hours: Literal["apagada", "alarmas", "vivo"] | None = None
    # dónde se analiza: general (lo que diga su cliente) | pcs | servidores. None = no cambia
    analysis_location: Literal["general", "pcs", "servidores"] | None = None
    # usuario que la ve y la analiza en su PC (lo elige el admin). None = no cambia; 0 = sin asignar
    operator_id: int | None = None
    alarm_cfg: AlarmCfgIn = Field(default_factory=AlarmCfgIn)


class StreamUrlIn(BaseModel):
    kind: Literal["live", "playback", "cloud"] = "live"
    quality: Literal["sub", "main"] = "sub"
    start: str | None = None  # hora local del grabador, ISO
    end: str | None = None


class ZoneIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    type: Literal["shelf", "checkout", "exit", "restricted", "entrance", "pump", "shop"]
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
                "plans": list(c.plans or ["hurto"]), "hours": c.hours,
                "analysis_location": c.analysis_location or "general", "created_at": iso(c.created_at)}

    def user_dict(u: User) -> dict:
        return {"id": u.id, "username": u.username, "role": u.role, "client_id": u.client_id, "active": u.active}

    def agent_dict(a: Agent) -> dict:
        online = bool(a.last_seen and utcnow() - a.last_seen < timedelta(minutes=3))
        return {"id": a.id, "client_id": a.client_id, "central": a.client_id is None, "name": a.name,
                "last_seen": iso(a.last_seen), "online": online, "status": a.last_status or {},
                "process_imports": a.process_imports, "kind": a.kind or "pc", "capacity": a.capacity,
                "instance_id": a.instance_id, "draining": bool(a.draining),
                "version": (a.last_status or {}).get("version")}

    def operator_map(s) -> dict:
        """Nombres para mostrar en cada cámara: quién la revisa (usuario) y qué la analiza (PC o máquina)."""
        return {"users": dict(s.query(User.id, User.username).all()),
                "agents": dict(s.query(Agent.id, Agent.name).all())}

    def user_agent(s, u: User) -> Agent:
        """La PC de un usuario: se crea sola la primera vez (no hay que vincular ni nombrar PCs)."""
        a = s.query(Agent).filter_by(user_id=u.id).order_by(Agent.id).first()
        if a is None:
            a = Agent(client_id=None, name=f"PC de {u.username}", user_id=u.id, token_hash=token_hash(new_token()),
                      process_imports=True)
            s.add(a)
            s.flush()
        return a

    def assigned_camera_ids(s, u: User) -> list[int]:
        """Las cámaras que revisa el usuario (en modo «pcs», además, las analiza su PC)."""
        return [cid for (cid,) in s.query(Camera.id).filter(Camera.operator_id == u.id).order_by(Camera.id)]

    # ------------------------------------------------------------ máquinas de análisis (server/pool.py)
    def setting(s, key: str, default: str = "") -> str:
        row = s.get(AppSetting, key)
        return row.value if row is not None and row.value is not None else default

    def set_setting(s, key: str, value) -> None:
        row = s.get(AppSetting, key)
        if row is None:
            s.add(AppSetting(key=key, value=str(value)))
        else:
            row.value = str(value)

    def analysis_mode(s) -> str:
        return setting(s, "analysis_mode", "pcs")

    def locations(s) -> dict[int, str]:
        """Dónde se analiza cada cámara: lo que diga la cámara; si no, su cliente; si no, el ajuste general."""
        mode = analysis_mode(s)
        clients = dict(s.query(Client.id, Client.analysis_location).all())
        out = {}
        for cid, client_id, loc in s.query(Camera.id, Camera.client_id, Camera.analysis_location):
            cl = clients.get(client_id) or "general"
            out[cid] = loc if loc in ("pcs", "servidores") else (cl if cl in ("pcs", "servidores") else mode)
        return out

    def camera_on_pcs(s, c: Camera) -> bool:
        if c.analysis_location in ("pcs", "servidores"):
            return c.analysis_location == "pcs"
        cl = s.get(Client, c.client_id)
        loc = cl.analysis_location if cl is not None else "general"
        return (loc if loc in ("pcs", "servidores") else analysis_mode(s)) == "pcs"

    def sync_locations(s, before: dict[int, str]) -> int:
        """Después de cambiar dónde se analiza (en general, un cliente o una cámara): las cámaras que pasan a
        las máquinas guardan qué PC tenían; las que vuelven a «pcs» van a la PC de quien las revisa o a la
        que tenían antes. Las que van a máquinas las reparte `rebalance`."""
        after = locations(s)
        changed = [cid for cid, loc in after.items() if before.get(cid) != loc]
        if not changed:
            return 0
        try:
            backup = {int(k): v for k, v in json.loads(setting(s, "pcs_backup", "{}") or "{}").items()}
        except ValueError:
            backup = {}
        servers = {a.id for a in server_agents(s)}
        alive = {a for (a,) in s.query(Agent.id)}
        for c in s.query(Camera).filter(Camera.id.in_(changed)):
            if after[c.id] == "servidores":
                if c.agent_id not in servers:
                    backup[c.id] = c.agent_id
            elif c.operator_id:
                op = s.get(User, c.operator_id)
                c.agent_id = user_agent(s, op).id if op else None
            elif c.agent_id in servers or c.agent_id is None:
                prev = backup.get(c.id)
                c.agent_id = prev if prev in alive and prev not in servers else None
        set_setting(s, "pcs_backup", json.dumps({str(k): v for k, v in backup.items()}))
        bump_rev(s)
        return len(changed)

    def pool_capacity(s) -> float:
        try:
            return float(setting(s, "pool_capacity") or pool.DEFAULT_CAPACITY)
        except ValueError:
            return pool.DEFAULT_CAPACITY

    def bump_rev(s) -> None:
        """Algo cambió en cámaras, zonas o clientes: las PCs y máquinas piden la configuración ya."""
        try:
            set_setting(s, "config_rev", int(setting(s, "config_rev", "0") or 0) + 1)
        except ValueError:
            set_setting(s, "config_rev", 1)

    def server_agents(s) -> list[Agent]:
        return s.query(Agent).filter(Agent.kind == "servidor").order_by(Agent.id).all()

    def is_online(a: Agent, now: datetime) -> bool:
        return bool(a.last_seen and now - a.last_seen < pool.ONLINE)

    def camera_weights(s, times: list[datetime]) -> tuple[list[Camera], list[dict[int, float]]]:
        """Peso de cada cámara habilitada que se analiza en las máquinas, en cada momento pedido (server/pool.py)."""
        locs = locations(s)
        cams = [c for c in s.query(Camera).filter(Camera.enabled.is_(True)).order_by(Camera.id)
                if locs.get(c.id) == "servidores"]
        clients = ({c.id: c for c in s.query(Client).filter(Client.id.in_({c.client_id for c in cams}))}
                   if cams else {})
        out: list[dict[int, float]] = [{} for _ in times]
        for c in cams:
            cl = clients.get(c.client_id)
            if cl is None:
                continue
            fps = camera_fps(c, camera_analysis(c, cl))
            for i, t in enumerate(times):
                out[i][c.id] = pool.camera_weight(fps=fps, mode=c.mode or "vivo", brand=c.brand, hours=cl.hours,
                                                  tz=cl.timezone, off_hours=c.off_hours or "apagada", now=t)
        return cams, out

    def machines_of(s, now: datetime, weights: dict[int, float]) -> list[pool.Machine]:
        agents = server_agents(s)
        cap = pool_capacity(s)
        load: dict[int, float] = {}
        if agents:
            for cid, aid in s.query(Camera.id, Camera.agent_id).filter(Camera.agent_id.in_([a.id for a in agents])):
                load[aid] = load.get(aid, 0.0) + weights.get(cid, 0.0)
        groups: dict[str, pool.Machine] = {}
        for a in agents:  # varios procesos en la misma máquina = una sola máquina
            key = a.instance_id or f"agente-{a.id}"
            m = groups.setdefault(key, pool.Machine(key, a.instance_id, [], False, True, 0.0, 0.0))
            m.agent_ids.append(a.id)
            m.names.append(a.name)
            m.online = m.online or is_online(a, now)
            m.draining = m.draining and bool(a.draining)
            m.capacity += a.capacity or cap
            m.load += load.get(a.id, 0.0)
        return list(groups.values())

    def try_lock(s) -> bool:
        """Un solo reparto a la vez (Vercel puede atender dos pedidos juntos)."""
        if s.get_bind().dialect.name != "postgresql":
            return True
        return bool(s.execute(select(func.pg_try_advisory_xact_lock(727001))).scalar())

    def rebalance(s, force: bool = False) -> bool:
        """Reparte las cámaras entre las máquinas de análisis en línea. Como mucho cada 20 s (salvo `force`)."""
        if "servidores" not in set(locations(s).values()):
            return False
        now = utcnow()
        last = setting(s, "balance_at")
        if not force and last:
            try:
                if now - datetime.fromisoformat(last) < timedelta(seconds=20):
                    return False
            except ValueError:
                pass
        if not try_lock(s):
            return False
        set_setting(s, "balance_at", now.isoformat())
        cap = pool_capacity(s)
        servers = {a.id: (a.capacity or cap) for a in server_agents(s) if is_online(a, now) and not a.draining}
        if not servers:
            s.commit()
            return False
        cams, (w_now,) = camera_weights(s, [now])
        result = pool.assign([(c.id, w_now.get(c.id, 0.0), c.agent_id) for c in cams], servers)
        moved = 0
        for c in cams:
            new = result.get(c.id)
            if new is not None and c.agent_id != new:
                c.agent_id = new
                moved += 1
        if moved:
            bump_rev(s)
            log.info("Reparto de cámaras: %d movidas entre %d máquinas", moved, len(servers))
        s.commit()
        return moved > 0

    def pc_mode(s, ag: Agent) -> str:
        """Para la nota de la app: «servidores» si las cámaras que revisa este usuario las analizan las máquinas."""
        if ag.user_id is None:
            return analysis_mode(s)
        locs = locations(s)
        mine = [cid for (cid,) in s.query(Camera.id).filter(Camera.operator_id == ag.user_id)]
        if mine and all(locs.get(c) == "servidores" for c in mine):
            return "servidores"
        return analysis_mode(s)

    def check_pool_key(s, key: str) -> None:
        stored = setting(s, "pool_key_hash")
        if not stored or not secrets.compare_digest(token_hash(key.strip()), stored):
            raise HTTPException(401, "Clave de máquinas inválida (se crea en Administración → Máquinas de análisis)")

    def analysis_overview(s, apply_draining: bool = False) -> dict:
        now = utcnow()
        cams, (w_now, w_soon) = camera_weights(s, [now, pool.soon(now)])
        ms = machines_of(s, now, w_now)
        spare = int(setting(s, "pool_spare", "0") or 0)
        mode = analysis_mode(s)
        # solo cuentan las cámaras que se analizan en máquinas (en general, por cliente o por cámara)
        p = pool.plan(sum(w_now.values()), sum(w_soon.values()), ms, pool_capacity(s), spare if cams else 0)
        if apply_draining:
            by_id = {a.id: a for a in server_agents(s)}
            for m in ms:
                if m.key in p.draining:
                    for aid in m.agent_ids:
                        by_id[aid].draining = p.draining[m.key]
                    m.draining = p.draining[m.key]
            s.commit()
        cam_count: dict[int, int] = {}
        for c in cams:
            if c.agent_id is not None:
                cam_count[c.agent_id] = cam_count.get(c.agent_id, 0) + 1
        open_now = sum(1 for w in w_now.values() if w > 0)
        clients_on = sorted({c.client_id for c in cams})
        return {"mode": mode, "capacity": pool_capacity(s), "spare": spare, "clients_on_servers": clients_on,
                "has_pool_key": bool(setting(s, "pool_key_hash")), "plan": p.as_dict(),
                "cameras": len(cams), "cameras_active": open_now,
                "online_instances": [m.instance_id for m in ms if m.online and m.instance_id],
                "machines": [{"key": m.key, "instance_id": m.instance_id, "names": m.names, "agent_ids": m.agent_ids,
                              "online": m.online, "draining": m.draining, "capacity": m.capacity,
                              "load": round(m.load, 2), "cameras": sum(cam_count.get(a, 0) for a in m.agent_ids)}
                             for m in ms]}

    def camera_dict(c: Camera, user: User, status: dict | None = None, ops: dict | None = None) -> dict:
        d = {"id": c.id, "client_id": c.client_id, "agent_id": c.agent_id, "name": c.name, "enabled": c.enabled,
             "detector": c.detector, "process_fps": c.process_fps, "zones": c.zones or [], "rules": c.rules or {},
             "has_preview": bool(c.preview_path), "preview_at": iso(c.preview_at), "status": status,
             "live": bool(status and status.get("live")), "mode": c.mode or "vivo"}
        if user.role != "cliente":
            d.update(brand=c.brand, host=c.host, username=c.username, has_password=bool(c.password_enc),
                     channel=c.channel,
                     rtsp_port=c.rtsp_port, http_port=c.http_port, detect_stream=c.detect_stream,
                     nvr_clips=c.nvr_clips, playback_time=c.playback_time, has_source_url=bool(c.source_url_enc),
                     hpp_serial=c.hpp_serial, hc_resource_id=c.hc_resource_id, alarm_cfg=alarm_cfg_of(c),
                     profile=c.profile or "general", off_hours=c.off_hours or "apagada",
                     analysis_location=c.analysis_location or "general")
            ops = ops or {}
            op_name = (ops.get("users") or {}).get(c.operator_id) if c.operator_id else None
            d.update(operator_id=c.operator_id if op_name else None, operator_name=op_name,
                     analyzer_name=(ops.get("agents") or {}).get(c.agent_id) if c.agent_id else None)
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

    @app.get("/api/app/download")
    def app_download(user: User = Depends(staff)) -> RedirectResponse:
        """Descarga del .zip de la app para el personal logueado en la web (sirve aunque el
        repositorio sea privado: no hace falta cuenta de GitHub)."""
        return RedirectResponse(app_update(download=True, _=user)["url"], status_code=302,
                                headers={"Cache-Control": "no-store"})

    @app.get("/api/app/update")
    def app_update(download: bool = False, _: User = Depends(staff)) -> dict:
        """Última versión de la app de escritorio publicada en GitHub Releases. Con `download=1`
        devuelve además un link de descarga directo que vence en unos minutos. Si el repositorio
        es privado hace falta GITHUB_TOKEN en el servidor (el programa nunca ve ese token)."""
        gh = "https://api.github.com/repos/" + st.github_repo
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        if st.github_token:
            headers["Authorization"] = "Bearer " + st.github_token
        try:
            with httpx.Client(timeout=15, headers=headers) as h:
                r = h.get(gh + "/releases/latest")
                if r.status_code in (401, 403, 404):
                    raise HTTPException(503, "El servidor no puede leer las versiones publicadas en GitHub (si el "
                                             "repositorio es privado, falta GITHUB_TOKEN en Vercel o venció).")
                r.raise_for_status()
                rel = r.json()
                asset = next((x for x in rel.get("assets") or [] if x.get("name", "").endswith("windows.zip")), None)
                tag = rel.get("tag_name") or ""
                out = {"version": tag.removeprefix("agente-v").removeprefix("v"), "tag": tag,
                       "published_at": rel.get("published_at"), "notes": rel.get("body") or "",
                       "size": (asset or {}).get("size"), "url": None}
                if download:
                    if not asset:
                        raise HTTPException(404, "La última versión no tiene el .zip para Windows")
                    d = h.get(f"{gh}/releases/assets/{asset['id']}", headers={"Accept": "application/octet-stream"},
                              follow_redirects=False)
                    if d.status_code in (301, 302, 303, 307, 308):
                        out["url"] = d.headers["location"]
                    else:
                        raise HTTPException(502, f"GitHub no dio el link de descarga ({d.status_code})")
                return out
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"No se pudo consultar GitHub: {exc}") from exc

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
        web = {}
        for u in s.query(User).filter_by(role="cliente").order_by(User.id.desc()):
            web[u.client_id] = u.username  # el primero creado queda último y gana
        return [{**client_dict(c), "web_username": web.get(c.id), **client_hc(c)} for c in q.all()]

    def client_web_user(s, cid: int) -> User | None:
        return s.query(User).filter_by(client_id=cid, role="cliente").order_by(User.id).first()

    def client_out(s, c: Client) -> dict:
        u = client_web_user(s, c.id)
        return {**client_dict(c), "web_username": u.username if u else None, **client_hc(c)}

    def client_hc(c: Client) -> dict:
        """Datos de Hik-Connect para el personal (el SecretKey nunca sale del servidor)."""
        return {"hc_app_key": c.hc_app_key, "hc_api_url": c.hc_api_url, "hc_has_secret": bool(c.hc_secret_enc),
                "hc_has_code": bool(c.hc_code_enc)}

    def set_client_hours(c: Client, body: ClientIn) -> None:
        if "hours" not in body.model_fields_set:
            return
        try:
            c.hours = normalize_hours(body.hours)
        except ValueError as exc:
            raise HTTPException(422, f"Horario: {exc}") from exc

    def set_client_hc(c: Client, body: ClientIn) -> None:
        sent = body.model_fields_set
        if "hc_app_key" in sent:
            c.hc_app_key = (body.hc_app_key or "").strip() or None
            if not c.hc_app_key:
                c.hc_secret_enc = None
        if "hc_secret" in sent and (body.hc_secret or "").strip():
            c.hc_secret_enc = box.encrypt(body.hc_secret.strip())
        if "hc_api_url" in sent:
            c.hc_api_url = hikconnect.normalize_url(body.hc_api_url)
        code = (body.hc_code or "").strip() if "hc_code" in sent else ""
        if code == "-":
            c.hc_code_enc = None
        elif code:
            c.hc_code_enc = box.encrypt(code)

    def hc_client(c: Client) -> hikconnect.HikConnect:
        if not c.hc_app_key or not c.hc_secret_enc:
            raise HTTPException(422, f"El cliente «{c.name}» no tiene cargada la clave de la API de Hik-Connect")
        return hikconnect.HikConnect(c.hc_app_key, box.decrypt(c.hc_secret_enc), c.hc_api_url)

    def set_client_web_user(s, c: Client, body: ClientIn) -> None:
        """Crea o actualiza el usuario con el que el cliente entra a la web (rol cliente)."""
        username = (body.web_username or "").strip()
        password = body.web_password or ""
        if not username:
            if password:
                raise HTTPException(422, "Falta el usuario para la web")
            return
        if len(username) < 3:
            raise HTTPException(422, "El usuario para la web necesita al menos 3 letras")
        if password and len(password) < 8:
            raise HTTPException(422, "La clave para la web necesita al menos 8 caracteres")
        u = client_web_user(s, c.id) if c.id else None
        other = s.query(User).filter(User.username == username)
        if u:
            other = other.filter(User.id != u.id)
        if other.first():
            raise HTTPException(409, f"El usuario «{username}» ya existe: elegí otro")
        if u is None:
            if not password:
                raise HTTPException(422, "Falta la clave para la web (mínimo 8 caracteres)")
            s.add(User(username=username, password_hash=hash_password(password), role="cliente", client_id=c.id,
                       active=True))
            return
        u.username = username
        if password:
            u.password_hash = hash_password(password)
            s.query(SessionToken).filter_by(user_id=u.id).delete()

    @app.post("/api/clients", status_code=201)
    def create_client(body: ClientIn, _: User = Depends(staff), s=Depends(db)) -> dict:
        if s.query(Client).filter_by(name=body.name).first():
            raise HTTPException(409, "Ya existe un cliente con ese nombre")
        c = Client(**body.client_fields())
        c.analysis_location = body.analysis_location or "general"
        set_client_hours(c, body)
        set_client_hc(c, body)
        s.add(c)
        s.flush()
        set_client_web_user(s, c, body)  # si falla, no queda el cliente a medias
        s.commit()
        return client_out(s, c)

    @app.put("/api/clients/{cid}")
    def update_client(cid: int, body: ClientIn, _: User = Depends(staff), s=Depends(db)) -> dict:
        c = get_or_404(s, Client, cid, "Cliente")
        other = s.query(Client).filter(Client.name == body.name, Client.id != cid).first()
        if other:
            raise HTTPException(409, "Ya existe un cliente con ese nombre")
        before = locations(s)
        for k, v in body.client_fields().items():
            setattr(c, k, v)
        if body.analysis_location is not None:
            c.analysis_location = body.analysis_location
        set_client_hours(c, body)
        set_client_hc(c, body)
        set_client_web_user(s, c, body)
        s.flush()
        sync_locations(s, before)
        bump_rev(s)  # horario y planes cambian qué hacen sus cámaras
        s.commit()
        rebalance(s, force=True)
        return client_out(s, c)

    @app.delete("/api/clients/{cid}", status_code=204)
    def delete_client(cid: int, _: User = Depends(admin_only), s=Depends(db)) -> None:
        """Borra el cliente con todo lo suyo: cámaras, alertas y clips, alarmas, métricas, videos, avisos,
        modelos y los usuarios de rol cliente. No se puede deshacer (el panel pide escribir el nombre)."""
        c = get_or_404(s, Client, cid, "Cliente")
        files: list[str] = []
        cam_ids = [x.id for x in s.query(Camera).filter_by(client_id=cid)]
        s.query(ClientReport).filter_by(client_id=cid).delete(synchronize_session=False)
        for e in s.query(Event).filter_by(client_id=cid):
            files += [p for p in (e.snapshot_path, e.clip_path) if p]
        s.query(Event).filter_by(client_id=cid).delete(synchronize_session=False)
        s.query(Alarm).filter_by(client_id=cid).delete(synchronize_session=False)
        s.query(CameraMetric).filter_by(client_id=cid).delete(synchronize_session=False)
        imps = s.query(VideoImport).filter_by(client_id=cid).all()
        if imps:
            s.query(Incident).filter(Incident.import_id.in_([i.id for i in imps])).delete(synchronize_session=False)
            files += [i.storage_path for i in imps if i.storage_path]
            s.query(VideoImport).filter_by(client_id=cid).delete(synchronize_session=False)
        s.query(FilterModel).filter_by(client_id=cid).delete(synchronize_session=False)
        for cam in s.query(Camera).filter_by(client_id=cid):
            files += ([cam.preview_path] if cam.preview_path else []) + [live_path(cam)]
        s.query(Camera).filter_by(client_id=cid).delete(synchronize_session=False)
        users = [u.id for u in s.query(User).filter_by(client_id=cid)]
        if users:
            s.query(SessionToken).filter(SessionToken.user_id.in_(users)).delete(synchronize_session=False)
            s.query(User).filter(User.id.in_(users)).delete(synchronize_session=False)
        for a in s.query(Agent).filter_by(client_id=cid):  # agentes "de local" de ese cliente
            s.delete(a)
        s.delete(c)
        s.commit()
        try:
            storage.remove(files)
        except StorageError:
            log.warning("No se pudieron borrar %d archivos del cliente %s", len(files), cid)
        log.info("Cliente %s borrado (%d cámaras)", cid, len(cam_ids))

    @app.get("/api/users")
    def list_users(_: User = Depends(admin_only), s=Depends(db)) -> list[dict]:
        cams: dict[int, list[int]] = {}
        for cid, uid in s.query(Camera.id, Camera.operator_id).filter(Camera.operator_id.is_not(None)):
            cams.setdefault(uid, []).append(cid)
        return [{**user_dict(u), "camera_ids": cams.get(u.id, [])} for u in s.query(User).order_by(User.username).all()]

    @app.put("/api/users/{uid}/cameras")
    def set_user_cameras(uid: int, body: UserCamerasIn, _: User = Depends(admin_only), s=Depends(db)) -> dict:
        """Las cámaras que revisa un administrador o empleado (en modo «pcs», además, las analiza su PC).
        Una cámara la revisa un solo usuario."""
        u = get_or_404(s, User, uid, "Usuario")
        if u.role not in ("admin", "entrenador"):
            raise HTTPException(422, "Las cámaras se asignan a administradores o empleados")
        a = user_agent(s, u)
        want = set(body.camera_ids)
        locs = locations(s)
        for c in s.query(Camera).filter(or_(Camera.id.in_(want), Camera.operator_id == u.id)):
            mine = c.id in want
            c.operator_id = u.id if mine else None
            if locs.get(c.id) == "pcs":
                c.agent_id = a.id if mine else (None if c.agent_id == a.id else c.agent_id)
        bump_rev(s)
        s.commit()
        return {"user_id": u.id, "camera_ids": assigned_camera_ids(s, u)}

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

    @app.delete("/api/users/{uid}", status_code=204)
    def delete_user(uid: int, me_: User = Depends(admin_only), s=Depends(db)) -> None:
        """Borra un usuario. Lo que hizo (etiquetas, avisos, videos subidos) queda, sin su nombre; su PC se borra
        y las cámaras que tenía abiertas quedan cerradas. No se puede borrar el propio usuario ni el último admin."""
        u = get_or_404(s, User, uid, "Usuario")
        if u.id == me_.id:
            raise HTTPException(409, "No podés borrar tu propio usuario")
        if u.role == "admin" and not s.query(User).filter(User.role == "admin", User.active.is_(True),
                                                          User.id != u.id).first():
            raise HTTPException(409, "Es el único administrador activo: creá otro antes de borrarlo")
        pcs = [a for (a,) in s.query(Agent.id).filter(Agent.user_id == uid)]
        if pcs:
            s.query(Camera).filter(Camera.agent_id.in_(pcs)).update({"agent_id": None}, synchronize_session=False)
            s.query(VideoImport).filter(VideoImport.claimed_by.in_(pcs)).update({"claimed_by": None},
                                                                              synchronize_session=False)
            s.query(Alarm).filter(Alarm.claimed_by.in_(pcs)).update({"claimed_by": None}, synchronize_session=False)
            s.query(Agent).filter(Agent.id.in_(pcs)).delete(synchronize_session=False)
        s.query(Camera).filter(Camera.operator_id == uid).update({"operator_id": None}, synchronize_session=False)
        s.query(VideoImport).filter(VideoImport.created_by == uid).update({"created_by": None}, synchronize_session=False)
        s.query(Event).filter(Event.labeled_by == uid).update({"labeled_by": None}, synchronize_session=False)
        s.query(ClientReport).filter(ClientReport.sent_by == uid).update({"sent_by": None}, synchronize_session=False)
        s.query(FilterModel).filter(FilterModel.created_by == uid).update({"created_by": None},
                                                                          synchronize_session=False)
        s.query(SessionToken).filter(SessionToken.user_id == uid).delete(synchronize_session=False)
        name = u.username
        s.delete(u)
        bump_rev(s)
        s.commit()
        log.info("Usuario %s borrado por %s", name, me_.username)

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

    @app.post("/api/agent/session")
    def agent_session(user: User = Depends(staff), s=Depends(db)) -> dict:
        """La app de escritorio, al entrar un administrador o empleado, pide el token de la PC de ese usuario:
        con él analiza y muestra las cámaras que el admin le asignó. Se genera uno nuevo cada vez (si el
        mismo usuario entra en otra PC, la anterior deja de analizar)."""
        u = s.get(User, user.id)
        a = user_agent(s, u)
        token = new_token()
        a.token_hash = token_hash(token)
        s.commit()
        return {**agent_dict(a), "token": token, "camera_ids": assigned_camera_ids(s, u),
                "analysis_mode": analysis_mode(s)}

    @app.post("/api/agent/login")
    def agent_login(body: AgentLoginIn, s=Depends(db)) -> dict:
        """Vincula una PC central con usuario y clave de administrador (sin copiar tokens a mano).
        Si ya existe una PC central con ese nombre, se le genera un token nuevo (el anterior deja de
        servir); si no, se crea. Devuelve el token, que el programa guarda en agente.env."""
        u = s.query(User).filter_by(username=body.username.strip()).first()
        if not u or not u.active or not verify_password(body.password, u.password_hash):
            raise HTTPException(401, "Usuario o clave incorrectos")
        if u.role != "admin":
            raise HTTPException(403, "Solo un administrador puede vincular una PC central")
        name = body.pc_name.strip()
        a = s.query(Agent).filter(Agent.client_id.is_(None), Agent.name == name).order_by(Agent.id).first()
        token = new_token()
        if a is None:
            a = Agent(client_id=None, name=name, token_hash=token_hash(token), process_imports=True)
            s.add(a)
        else:
            a.token_hash = token_hash(token)
        s.commit()
        log.info("PC central %r vinculada por %s", name, u.username)
        return {**agent_dict(a), "token": token}

    @app.post("/api/agent/enroll")
    def agent_enroll(body: EnrollIn, s=Depends(db)) -> dict:
        """Una máquina de análisis se da de alta sola al arrancar, con la clave de máquinas. Si ya existía
        (mismo nombre), se le genera un token nuevo."""
        check_pool_key(s, body.key)
        name = body.name.strip()
        a = s.query(Agent).filter(Agent.kind == "servidor", Agent.name == name).order_by(Agent.id).first()
        token = new_token()
        if a is None:
            a = Agent(client_id=None, name=name, kind="servidor", token_hash=token_hash(token),
                      process_imports=body.process_imports)
            s.add(a)
        else:
            a.token_hash = token_hash(token)
        a.instance_id = body.instance_id or a.instance_id
        a.capacity = body.capacity or a.capacity
        a.draining = False
        a.last_seen = utcnow()
        s.commit()
        log.info("Máquina de análisis %r dada de alta (instancia %s)", name, a.instance_id)
        rebalance(s, force=True)  # que reciba cámaras ya, en su primer pedido de configuración
        return {**agent_dict(a), "token": token}

    @app.get("/api/agent/scale")
    def agent_scale(x_pool_key: str = Header(...), s=Depends(db)) -> dict:
        """El encendedor (agent/escalador.py) pregunta cuántas máquinas tienen que estar prendidas y cuáles se
        pueden apagar. Marca para vaciar las que sobran; se apagan cuando ya no tienen cámaras."""
        check_pool_key(s, x_pool_key)
        rebalance(s)
        out = analysis_overview(s, apply_draining=True)
        rebalance(s, force=True)  # sacarle ya las cámaras a las que se están vaciando
        return out

    @app.get("/api/analysis")
    def get_analysis(_: User = Depends(admin_only), s=Depends(db)) -> dict:
        return analysis_overview(s)

    @app.put("/api/analysis")
    def put_analysis(body: AnalysisSettingsIn, _: User = Depends(admin_only), s=Depends(db)) -> dict:
        if body.capacity is not None:
            set_setting(s, "pool_capacity", body.capacity)
        if body.spare is not None:
            set_setting(s, "pool_spare", body.spare)
        if body.mode is not None and body.mode != analysis_mode(s):
            before = locations(s)
            set_setting(s, "analysis_mode", body.mode)
            s.flush()
            n = sync_locations(s, before)
            log.info("Ajuste general: las cámaras se analizan en %s (%d cambian)", body.mode, n)
        s.commit()
        rebalance(s, force=True)
        return analysis_overview(s)

    @app.post("/api/analysis/pool-key")
    def new_pool_key(_: User = Depends(admin_only), s=Depends(db)) -> dict:
        """Clave para dar de alta máquinas de análisis (va en el archivo de configuración de cada máquina).
        Se muestra una sola vez; crear otra invalida la anterior para máquinas nuevas (las ya dadas de alta
        siguen con su token)."""
        key = "vgm_" + new_token()
        set_setting(s, "pool_key_hash", token_hash(key))
        s.commit()
        return {"key": key}

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
        bump_rev(s)
        s.commit()
        rebalance(s, force=True)

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
        statuses, ops = agent_status_map(s), operator_map(s)
        return [camera_dict(c, user, statuses.get(c.id), ops) for c in q.all()]

    @app.get("/api/cameras/{cam_id}")
    def get_camera(cam_id: int, user: User = Depends(current_user), s=Depends(db)) -> dict:
        c = get_or_404(s, Camera, cam_id, "Cámara")
        scope_client(user, c.client_id)
        return camera_dict(c, user, agent_status_map(s).get(c.id), operator_map(s))

    def camera_fields(body: CameraIn) -> dict:
        f = body.model_dump(exclude={"password", "source_url", "operator_id"})
        if "agent_id" not in body.model_fields_set:
            f.pop("agent_id")  # la asignación se hace por usuario (operator_id) y solo la toca el admin
        if f.get("off_hours") is None:
            f.pop("off_hours")  # una app vieja no lo manda: queda como estaba
        if f.get("analysis_location") is None:
            f.pop("analysis_location")
        return f

    def apply_operator(s, c: Camera, body: CameraIn, user: User) -> None:
        """Quién revisa la cámara. En modo «pcs» también la analiza su PC; en modo «servidores» la máquina
        que la analiza la elige el reparto (server/pool.py)."""
        if body.operator_id is None or user.role != "admin":
            return  # sin dato (o un empleado): la asignación queda como estaba
        pcs = camera_on_pcs(s, c)
        if body.operator_id == 0:
            c.operator_id = None
            if pcs:
                c.agent_id = None
            return
        op = get_or_404(s, User, body.operator_id, "Usuario")
        if op.role not in ("admin", "entrenador"):
            raise HTTPException(422, "Las cámaras se asignan a administradores o empleados")
        c.operator_id = op.id
        if pcs:
            c.agent_id = user_agent(s, op).id

    def check_camera_body(s, body: CameraIn) -> None:
        get_or_404(s, Client, body.client_id, "Cliente")
        if body.agent_id is not None:
            ag = get_or_404(s, Agent, body.agent_id, "Agente")
            if ag.client_id is not None and ag.client_id != body.client_id:
                raise HTTPException(422, "El agente pertenece a otro cliente")
        body.hpp_serial = (body.hpp_serial or "").strip().upper() or None
        if body.brand == "hikconnect" and body.hpp_serial:
            body.hpp_serial = hikconnect.short_serial(body.hpp_serial) or None
        body.hc_resource_id = (body.hc_resource_id or "").strip() or None
        if body.brand == "hikconnect":
            if not body.hpp_serial:
                raise HTTPException(422, "Por Hik-Connect hace falta el número de serie del grabador")
            if body.mode != "vivo":
                raise HTTPException(422, "Por Hik-Connect, por ahora, solo el modo En vivo 24 h")
            return
        if not body.host and not body.source_url:
            raise HTTPException(422, "Indicá la IP del NVR/cámara o una URL RTSP")
        if body.mode == "alarmas" and not body.hpp_serial:
            raise HTTPException(422, "En modo alarmas indicá el número de serie del DVR en Hik-Partner Pro")
        if body.mode in ("alarmas", "alarmas_equipo") and not body.alarm_cfg.event_types:
            raise HTTPException(422, "Elegí al menos un tipo de alarma")
        if body.mode == "alarmas" and body.brand not in ("hikvision", "ip"):
            raise HTTPException(422, "Las alarmas por Hik-Partner Pro son solo para Hikvision; "
                                     "usá «alarmas del propio grabador»")

    def check_not_repeated(s, body: CameraIn, exclude: int | None = None) -> None:
        """Por Hik-Connect, el mismo grabador y canal no se carga dos veces en un cliente (se repetía al elegir
        varias veces las mismas de la lista, y cada copia pedía su propia transmisión al grabador)."""
        if body.brand != "hikconnect":
            return
        q = s.query(Camera).filter(Camera.client_id == body.client_id, Camera.brand == "hikconnect",
                                   Camera.hpp_serial == body.hpp_serial, Camera.channel == body.channel)
        dup = q.filter(Camera.id != exclude).first() if exclude else q.first()
        if dup:
            raise HTTPException(409, f"Esa cámara ya está cargada en este cliente: «{dup.name}»")

    @app.post("/api/cameras", status_code=201)
    def create_camera(body: CameraIn, user: User = Depends(staff), s=Depends(db)) -> dict:
        check_camera_body(s, body)
        check_not_repeated(s, body)
        c = Camera(**camera_fields(body), password_enc=box.encrypt(body.password),
                   source_url_enc=box.encrypt(body.source_url))
        apply_operator(s, c, body, user)
        s.add(c)
        bump_rev(s)
        s.commit()
        rebalance(s, force=True)
        return camera_dict(c, user, ops=operator_map(s))

    @app.put("/api/cameras/{cam_id}")
    def update_camera(cam_id: int, body: CameraIn, user: User = Depends(staff), s=Depends(db)) -> dict:
        c = get_or_404(s, Camera, cam_id, "Cámara")
        check_camera_body(s, body)
        check_not_repeated(s, body, exclude=cam_id)
        before = locations(s)
        for k, v in camera_fields(body).items():
            setattr(c, k, v)
        s.flush()
        sync_locations(s, before)
        apply_operator(s, c, body, user)
        if body.password:
            c.password_enc = box.encrypt(body.password)
        c.source_url_enc = box.encrypt(body.source_url) if body.source_url else None
        bump_rev(s)
        s.commit()
        rebalance(s, force=True)
        return camera_dict(c, user, ops=operator_map(s))

    @app.put("/api/cameras/{cam_id}/analysis")
    def update_analysis(cam_id: int, body: AnalysisIn, user: User = Depends(staff), s=Depends(db)) -> dict:
        c = get_or_404(s, Camera, cam_id, "Cámara")
        cfg = {"zones": [z.model_dump() for z in body.zones], "rules": body.rules}
        try:
            parse_config(cfg)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        c.zones, c.rules = cfg["zones"], cfg["rules"]
        bump_rev(s)
        s.commit()
        return camera_dict(c, user, ops=operator_map(s))

    def remove_cameras(s, cams: list[Camera], with_alerts: bool) -> None:
        """Borra cámaras con sus métricas y alarmas. Si tienen alertas o videos subidos, solo con `with_alerts`
        (se borran también, con sus fotos, clips y avisos). No se puede deshacer."""
        ids = [c.id for c in cams]
        n_events = s.query(func.count(Event.id)).filter(Event.camera_id.in_(ids)).scalar() or 0
        imps = s.query(VideoImport).filter(VideoImport.camera_id.in_(ids)).all()
        if (n_events or imps) and not with_alerts:
            raise HTTPException(409, f"{'La cámara tiene' if len(ids) == 1 else 'Estas cámaras tienen'} {n_events} "
                                     f"alerta(s){f' y {len(imps)} video(s) subidos' if imps else ''}. Para borrar"
                                     f"{'la' if len(ids) == 1 else 'las'} hay que borrar eso también.")
        files: list[str] = []
        imp_ids = [i.id for i in imps]
        ev_ids = [eid for (eid,) in s.query(Event.id).filter(or_(Event.camera_id.in_(ids), Event.import_id.in_(imp_ids)))]
        if ev_ids:
            for e in s.query(Event).filter(Event.id.in_(ev_ids)):
                files += [p for p in (e.snapshot_path, e.clip_path) if p]
            s.query(ClientReport).filter(ClientReport.event_id.in_(ev_ids)).delete(synchronize_session=False)
            s.query(Alarm).filter(Alarm.event_id.in_(ev_ids)).update({"event_id": None}, synchronize_session=False)
            s.query(Event).filter(Event.id.in_(ev_ids)).delete(synchronize_session=False)
        if imps:
            s.query(Incident).filter(Incident.import_id.in_(imp_ids)).delete(synchronize_session=False)
            files += [i.storage_path for i in imps if i.storage_path]
            s.query(VideoImport).filter(VideoImport.id.in_(imp_ids)).delete(synchronize_session=False)
        s.query(CameraMetric).filter(CameraMetric.camera_id.in_(ids)).delete(synchronize_session=False)
        s.query(Alarm).filter(Alarm.camera_id.in_(ids)).delete(synchronize_session=False)
        for c in cams:
            files += ([c.preview_path] if c.preview_path else []) + [live_path(c)]
            s.delete(c)
        bump_rev(s)
        s.commit()
        try:
            storage.remove(files)
        except StorageError:
            log.warning("No se pudieron borrar %d archivos de cámaras borradas", len(files))
        log.info("Cámaras borradas: %s (%d alertas)", ids, n_events)

    @app.delete("/api/cameras/{cam_id}", status_code=204)
    def delete_camera(cam_id: int, con_alertas: bool = False, _: User = Depends(admin_only), s=Depends(db)) -> None:
        remove_cameras(s, [get_or_404(s, Camera, cam_id, "Cámara")], con_alertas)

    @app.post("/api/cameras/delete")
    def delete_cameras(body: CamerasDeleteIn, _: User = Depends(admin_only), s=Depends(db)) -> dict:
        """Borrar varias cámaras de una vez (por ejemplo, las de un grabador que se cargaron de más)."""
        cams = s.query(Camera).filter(Camera.id.in_(set(body.camera_ids))).all()
        if not cams:
            raise HTTPException(404, "No se encontraron esas cámaras")
        remove_cameras(s, cams, body.con_alertas)
        return {"deleted": len(cams)}

    @app.post("/api/cameras/{cam_id}/probe")
    def camera_probe(cam_id: int, _: User = Depends(staff), s=Depends(db)) -> dict:
        """Prueba desde el servidor que el grabador responda por HTTP con el usuario y la clave cargados.
        Sirve para los grabadores expuestos por el router del cliente (IP pública o DDNS)."""
        c = get_or_404(s, Camera, cam_id, "Cámara")
        if c.brand == "hikconnect":
            try:
                url = hc_stream_url(s, c, "live")
            except hikconnect.HikConnectError as exc:
                return {"ok": False, "hikconnect": True, "public": True, "error": str(exc)}
            return {"ok": True, "hikconnect": True, "public": True, "resource_id": c.hc_resource_id,
                    "detail": "Hik-Connect entregó el link de video (" + url.split("?")[0][:80] + "…)"}
        if not c.host:
            raise HTTPException(422, "La cámara no tiene IP o DDNS cargado")
        brand = c.brand
        if brand == "ip":  # marca a detectar: primero como Hikvision, si no responde, como Dahua
            res = probe.probe("hikvision", c.host, c.http_port, c.username, box.decrypt(c.password_enc))
            brand = "hikvision"
            if not res.get("ok"):
                alt = probe.probe("dahua", c.host, c.http_port, c.username, box.decrypt(c.password_enc))
                if alt.get("ok"):
                    res, brand = alt, "dahua"
            res["brand_detected"] = brand if res.get("ok") else None
        else:
            res = probe.probe(c.brand, c.host, c.http_port, c.username, box.decrypt(c.password_enc))
        res["public"] = probe.is_public_host(c.host)
        res["rtsp_url"] = probe.rtsp_url(brand, c.host, c.rtsp_port, c.channel, c.username, "CLAVE")
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

    def camera_analysis(c: Camera, client: Client | None = None) -> dict:
        plans = list((client.plans if client is not None else None) or [])
        prof = CAMERA_PROFILES.get(c.profile or "general", CAMERA_PROFILES["general"])
        zones, rules = list(c.zones or []), dict(c.rules or {})
        if prof["rules"] is not None and "enabled_rules" not in rules:  # lo elegido a mano manda
            rules["enabled_rules"] = list(prof["rules"])
        whole = prof.get("whole_zone")
        if whole and not any((z or {}).get("type") == whole for z in zones):  # ej. depósito: toda la imagen
            zones.append({"name": prof["name"], "type": whole, "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]})
        vehicles = ("estacion" in plans or bool(prof.get("vehicles"))
                    or any((z or {}).get("type") == "pump" for z in zones))
        # sin plan de hurto (o sin ninguna regla): la cámara solo mide; no hay alertas ni clips
        # (sin cliente = video subido para entrenar: siempre con las reglas de hurto)
        metrics_only = (client is not None and "hurto" not in plans) or rules.get("enabled_rules") == []
        if metrics_only:
            rules["enabled_rules"] = []
        out = {"zones": zones, "rules": rules, "vehicles": vehicles, "plans": plans, "profile": c.profile or "general",
               "metrics_only": metrics_only}
        if client is not None and client.hours:
            out["schedule"] = client.hours  # para la regla «fuera de horario»
        return out

    def camera_fps(c: Camera, analysis: dict) -> float:
        if analysis.get("metrics_only"):
            counting = any((z or {}).get("type") in ("entrance", "pump") for z in analysis.get("zones") or [])
            return METRICS_COUNT_FPS if counting else METRICS_FPS
        return max(MIN_FPS, c.process_fps or MIN_FPS)

    def model_spec(s, client_id: int) -> dict | None:
        m = active_model(s, client_id)
        return dict(m.spec, id=m.id) if m else None

    def watcher(s, ag: Agent) -> User:
        """La PC de un usuario activo del equipo (admin o empleado): la que abre cámaras a demanda."""
        u = s.get(User, ag.user_id) if ag.user_id else None
        if u is None or not u.active or u.role not in ("admin", "entrenador"):
            raise HTTPException(403, "Solo la app de un usuario del equipo abre cámaras a demanda")
        return u

    @app.get("/api/agent/catalog")
    def agent_catalog(ag: Agent = Depends(current_agent), s=Depends(db)) -> dict:
        """Todas las cámaras, para la lista de «En vivo»: el usuario elige cuáles abrir (sin asignar nada)."""
        watcher(s, ag)
        locs = locations(s)
        clients = dict(s.query(Client.id, Client.name).all())
        names = dict(s.query(Agent.id, Agent.name).all())
        cams = s.query(Camera).filter(Camera.enabled.is_(True)).order_by(Camera.client_id, Camera.name, Camera.id)
        return {"cameras": [{
            "id": c.id, "client_id": c.client_id, "client_name": clients.get(c.client_id, ""), "name": c.name,
            "here": c.agent_id == ag.id, "on_servers": locs.get(c.id) == "servidores",
            "elsewhere": names.get(c.agent_id) if c.agent_id not in (None, ag.id) else None,
        } for c in cams]}

    @app.put("/api/agent/watch")
    def agent_watch(body: UserCamerasIn, ag: Agent = Depends(current_agent), s=Depends(db)) -> dict:
        """Las cámaras que el usuario tiene abiertas en «En vivo»: las analiza su PC mientras estén abiertas.
        Las que dejó de mirar quedan libres. Las que van a las máquinas de análisis no se tocan."""
        watcher(s, ag)
        locs = locations(s)
        want = {cid for cid in body.camera_ids if locs.get(cid) == "pcs"}
        changed = 0
        for c in s.query(Camera).filter(or_(Camera.agent_id == ag.id, Camera.id.in_(want))):
            new = ag.id if c.id in want else None
            if c.agent_id != new:
                c.agent_id = new
                changed += 1
        if changed:
            bump_rev(s)
        s.commit()
        return {"camera_ids": sorted(want), "on_servers": sorted(set(body.camera_ids) - want)}

    @app.get("/api/agent/config")
    def agent_config(ag: Agent = Depends(current_agent), s=Depends(db)) -> dict:
        cams = agent_cameras_q(s, ag).order_by(Camera.id).all()
        clients = {c.id: c for c in s.query(Client).filter(Client.id.in_({c.client_id for c in cams}))} if cams else {}
        models = {cid: model_spec(s, cid) for cid in clients}  # cada cliente puede tener su propio modelo
        return {
            "agent_id": ag.id, "client_id": ag.client_id, "central": ag.client_id is None,
            "process_imports": ag.process_imports, "kind": ag.kind or "pc", "analysis_mode": pc_mode(s, ag),
            "rev": setting(s, "config_rev", "0"),
            "settings": {"clip_pre_seconds": st.clip_pre_seconds, "clip_post_seconds": st.clip_post_seconds,
                         "clip_max_seconds": st.clip_max_seconds},
            "filter_model": model_spec(s, ag.client_id) if ag.client_id is not None else None,
            "cameras": [{
                **agent_camera_cfg(c, clients[c.client_id]), "filter_model": models[c.client_id],
            } for c in cams],
        }

    def agent_camera_cfg(c: Camera, client: Client) -> dict:
        analysis = camera_analysis(c, client)
        return {
            "id": c.id, "name": c.name, "client_id": c.client_id, "client_name": client.name,
            "enabled": c.enabled, "detector": c.detector, "process_fps": camera_fps(c, analysis),
            "metrics_only": analysis["metrics_only"],
            # con el local cerrado la PC central no abre el video (salvo que la cámara diga otra cosa)
            "schedule": {"hours": client.hours, "margin_min": DEFAULT_MARGIN_MIN, "off_hours": c.off_hours or "apagada"},
            "mode": c.mode or "vivo", "alarm_cfg": alarm_cfg_of(c),
            "connection": {"brand": c.brand, "host": c.host, "username": c.username,
                           "password": box.decrypt(c.password_enc),
                           "channel": c.channel, "rtsp_port": c.rtsp_port, "http_port": c.http_port,
                           "detect_stream": c.detect_stream, "nvr_clips": c.nvr_clips,
                           "playback_time": c.playback_time, "timezone": client.timezone,
                           "source_url": box.decrypt(c.source_url_enc) or None,
                           "camera_id": c.id, "serial": c.hpp_serial},
            "analysis": analysis,
        }

    def hc_stream_url(s, c: Camera, kind: str, quality: str = "sub", start: datetime | None = None,
                      end: datetime | None = None) -> str:
        """Link de video de una cámara por Hik-Connect. Si no se eligió la cámara de Hik-Connect, la busca
        por serie del grabador y canal (y la guarda)."""
        client = s.get(Client, c.client_id)
        api = hc_client(client)
        try:
            if not c.hc_resource_id:
                cams = [x for x in api.cameras() if x["serial"].upper() == (c.hpp_serial or "").upper()]
                pick = next((x for x in cams if x["channel"] == c.channel), None)
                if pick is None and len(cams) >= c.channel and all(x["channel"] is None for x in cams):
                    pick = cams[c.channel - 1]  # sin número de canal en la respuesta: por orden
                if pick is None:
                    raise hikconnect.HikConnectError(
                        f"No encontré el canal {c.channel} del grabador {c.hpp_serial} en Hik-Connect "
                        f"({len(cams)} cámaras de esa serie). ¿Está agregado al equipo?")
                c.hc_resource_id = pick["id"]
                s.commit()
            # sin `code`: con el cifrado de video prendido Hikvision no entrega HLS aunque se mande el código
            # (EVZ60019; solo sirve para su reproductor EZOPEN). Hay que apagar el cifrado en el grabador.
            url = api.stream_url(c.hc_resource_id, c.hpp_serial or "", kind, quality, start, end)
            if client.hc_api_url is None and api.server:
                client.hc_api_url = api.server  # recordar la región que anduvo
                s.commit()
            return url
        finally:
            api.close()

    @app.get("/api/clients/{cid}/hikconnect/cameras")
    def hikconnect_cameras(cid: int, _: User = Depends(staff), s=Depends(db)) -> dict:
        """Cámaras que ve la API de Hik-Connect del cliente (para elegir al cargar una cámara)."""
        client = get_or_404(s, Client, cid, "Cliente")
        api = hc_client(client)
        try:
            cams = api.cameras()
        except hikconnect.HikConnectError as exc:
            raise HTTPException(502, str(exc)) from exc
        finally:
            api.close()
        if client.hc_api_url is None and api.server:
            client.hc_api_url = api.server
            s.commit()
        return {"server": api.server, "cameras": cams}

    @app.post("/api/agent/cameras/{cam_id}/stream-url")
    def agent_stream_url(cam_id: int, body: StreamUrlIn, ag: Agent = Depends(current_agent), s=Depends(db)) -> dict:
        """La PC central pide un link de video nuevo (las cámaras por Hik-Connect no tienen IP ni puertos)."""
        c = get_or_404(s, Camera, cam_id, "Cámara")
        if not agent_owns(ag, c):
            raise HTTPException(403, "La cámara no es de esta PC")
        if c.brand != "hikconnect":
            raise HTTPException(422, "La cámara no está configurada por Hik-Connect")
        try:
            start = datetime.fromisoformat(body.start).replace(tzinfo=None) if body.start else None
            end = datetime.fromisoformat(body.end).replace(tzinfo=None) if body.end else None
            url = hc_stream_url(s, c, body.kind, body.quality, start, end)
        except ValueError as exc:
            raise HTTPException(422, f"Hora mal formada: {exc}") from exc
        except hikconnect.HikConnectError as exc:
            raise HTTPException(502, str(exc)) from exc
        return {"url": url}

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
        if (a.kind or "pc") == "servidor":  # máquina de análisis: capacidad e instancia, y repartir si hace falta
            if isinstance(payload.get("capacity"), (int, float)) and 0 < payload["capacity"] <= 1000:
                a.capacity = float(payload["capacity"])
            if isinstance(payload.get("instance_id"), str) and payload["instance_id"]:
                a.instance_id = payload["instance_id"][:80]
            for k in ("batch_avg", "frames_per_s"):
                if isinstance(payload.get(k), (int, float)):
                    a.last_status = {**a.last_status, k: payload[k]}
        s.commit()
        if (a.kind or "pc") == "servidor":
            rebalance(s)
        since = utcnow() - timedelta(seconds=st.live_seconds)
        live = [c.id for c in agent_cameras_q(s, ag).filter(Camera.live_requested_at >= since, Camera.enabled.is_(True))]
        # para que la PC pida la configuración apenas cambie algo (sin esperar al minuto)
        mine = sorted(cid for (cid,) in agent_cameras_q(s, ag).with_entities(Camera.id))
        return {"ok": True, "live": live, "rev": setting(s, "config_rev", "0"), "cameras": mine,
                "draining": bool(a.draining)}

    @app.post("/api/agent/metrics")
    def agent_metrics(body: MetricsIn, ag: Agent = Depends(current_agent), s=Depends(db)) -> dict:
        """La PC central manda los cubos horarios de métricas (core/metrics.py); se reemplazan enteros."""
        saved = 0
        for b in body.buckets:
            c = s.get(Camera, b.camera_id)
            if not c or not agent_owns(ag, c):
                continue
            try:
                start = datetime.fromisoformat(b.bucket_start).replace(tzinfo=None, minute=0, second=0, microsecond=0)
            except ValueError:
                continue
            data = merge_bucket(empty_bucket(), b.data)  # normaliza claves y tipos
            row = s.query(CameraMetric).filter_by(camera_id=c.id, bucket_start=start).first()
            if row is None:
                row = CameraMetric(client_id=c.client_id, camera_id=c.id, bucket_start=start)
                s.add(row)
            row.data, row.updated_at = data, utcnow()
            saved += 1
        s.commit()
        return {"ok": True, "saved": saved}

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
        listens = (cam.mode or "vivo") == "alarmas_equipo" or (cam.off_hours == "alarmas" and cam.brand != "hikconnect")
        if not listens or not cam.enabled:
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
        mine = agent_cameras_q(s, ag).filter(or_(Camera.mode.in_(("alarmas", "alarmas_equipo")), Camera.off_hours == "alarmas"),
                                             Camera.enabled.is_(True))
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
        if user.role == "entrenador":  # un empleado ve las alertas de las cámaras que le asignaron
            conds.append(Event.camera_id.in_(assigned_camera_ids(s, user) or [-1]))
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

    @app.get("/api/metrics")
    def metrics(client_id: int | None = None, days: int = Query(1, ge=1, le=90), camera_id: int | None = None,
                user: User = Depends(current_user), s=Depends(db)) -> dict:
        """Resumen de métricas de un cliente: totales del período, por día, por hora y por cámara.
        Las horas son las del local (los cubos ya vienen en hora local)."""
        client_id = scope_client(user, client_id)
        if client_id is None:
            raise HTTPException(422, "Falta client_id")
        client = get_or_404(s, Client, client_id, "Cliente")
        tz = ZoneInfo(client.timezone)
        today = datetime.now(tz).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
        since = today - timedelta(days=days - 1)
        q = s.query(CameraMetric).filter(CameraMetric.client_id == client_id, CameraMetric.bucket_start >= since)
        if camera_id is not None:
            q = q.filter(CameraMetric.camera_id == camera_id)
        rows = q.all()
        cams = {c.id: c.name for c in s.query(Camera).filter(Camera.client_id == client_id)}
        total = empty_bucket()
        by_day: dict[str, dict] = {}
        by_hour: dict[int, dict] = {}
        by_cam: dict[int, dict] = {}
        for r in rows:
            total = merge_bucket(total, r.data)
            d = r.bucket_start.date().isoformat()
            by_day[d] = merge_bucket(by_day.get(d, empty_bucket()), r.data)
            by_hour[r.bucket_start.hour] = merge_bucket(by_hour.get(r.bucket_start.hour, empty_bucket()), r.data)
            by_cam[r.camera_id] = merge_bucket(by_cam.get(r.camera_id, empty_bucket()), r.data)
        return {"client_id": client_id, "client_name": client.name, "plans": list(client.plans or ["hurto"]),
                "days": days, "since": since.date().isoformat(), "until": today.date().isoformat(),
                "total": total,
                "by_day": [{"date": d, **by_day[d]} for d in sorted(by_day)],
                "by_hour": [{"hour": h, **by_hour[h]} for h in sorted(by_hour)],
                "by_camera": [{"camera_id": cid, "name": cams.get(cid, f"Cámara {cid}"), **by_cam[cid]} for cid in sorted(by_cam)]}

    @app.get("/api/stats/alerts")
    def alert_counts(_: User = Depends(staff), s=Depends(db)) -> dict:
        """Alertas por cámara y por cliente: hoy (desde la medianoche del local), últimos 7 días y sin revisar
        (sin etiqueta, últimos 60 días). Solo números: no baja fotos ni clips."""
        now = utcnow()
        base = (Event.source.in_(("vivo", "alarma", "prueba")), Event.alert.is_(True))
        cams: dict[int, dict] = {}

        def add(q, key: str) -> None:
            for cam_id, client_id, n in q.group_by(Event.camera_id, Event.client_id):
                d = cams.setdefault(cam_id, {"client_id": client_id, "today": 0, "week": 0, "unreviewed": 0})
                d[key] += n

        def q():
            return s.query(Event.camera_id, Event.client_id, func.count(Event.id)).filter(*base)

        add(q().filter(Event.occurred_at >= now - timedelta(days=7)), "week")
        add(q().filter(Event.label.is_(None), Event.occurred_at >= now - timedelta(days=60)), "unreviewed")
        by_tz: dict[str, list[int]] = {}
        for cid, tz in s.query(Client.id, Client.timezone):
            by_tz.setdefault(tz or "America/Montevideo", []).append(cid)
        for tz, ids in by_tz.items():  # «hoy» empieza a distinta hora según el país del local
            try:
                local = datetime.now(ZoneInfo(tz))
            except Exception:  # noqa: BLE001 - zona mal cargada
                local = datetime.now(ZoneInfo("America/Montevideo"))
            start = local.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc).replace(tzinfo=None)
            add(q().filter(Event.client_id.in_(ids), Event.occurred_at >= start), "today")
        clients: dict[int, dict] = {}
        for d in cams.values():
            t = clients.setdefault(d["client_id"], {"today": 0, "week": 0, "unreviewed": 0})
            for k in ("today", "week", "unreviewed"):
                t[k] += d[k]
        return {"cameras": {str(k): v for k, v in cams.items()}, "clients": {str(k): v for k, v in clients.items()}}

    @app.get("/api/metrics/export")
    def metrics_export(client_id: int | None = None, desde: str | None = None, hasta: str | None = None,
                       agrupar: Literal["hora", "dia"] = "dia", user: User = Depends(current_user),
                       s=Depends(db)) -> Response:
        """Métricas en CSV para Excel (separado por «;», decimales con coma, con BOM): por cámara, por día u
        hora, en hora del local. Sin cliente = todos (solo el personal). Fechas AAAA-MM-DD; por defecto la
        última semana."""
        client_id = scope_client(user, client_id)
        try:
            d2 = datetime.strptime(hasta, "%Y-%m-%d") if hasta else datetime.now().replace(
                hour=0, minute=0, second=0, microsecond=0)
            d1 = datetime.strptime(desde, "%Y-%m-%d") if desde else d2 - timedelta(days=6)
        except ValueError as exc:
            raise HTTPException(422, "Fechas con formato AAAA-MM-DD") from exc
        if d1 > d2:
            d1, d2 = d2, d1
        if (d2 - d1).days > 366:
            raise HTTPException(422, "Como mucho un año por vez")
        q = s.query(CameraMetric).filter(CameraMetric.bucket_start >= d1, CameraMetric.bucket_start < d2 + timedelta(days=1))
        if client_id is not None:
            q = q.filter(CameraMetric.client_id == client_id)
        rows = q.order_by(CameraMetric.bucket_start, CameraMetric.camera_id).all()
        cams = dict(s.query(Camera.id, Camera.name).all())
        cls = dict(s.query(Client.id, Client.name).all())
        groups: dict[tuple, dict] = {}
        for r in rows:
            key = (r.bucket_start if agrupar == "hora" else r.bucket_start.replace(hour=0), r.client_id, r.camera_id)
            groups[key] = merge_bucket(groups.get(key, empty_bucket()), r.data)

        def num(v: float, dec: int = 1) -> str:
            return f"{v:.{dec}f}".replace(".", ",")

        head = ["Fecha"] + (["Hora"] if agrupar == "hora" else []) + [
            "Cliente", "Cámara", "Personas", "Tiempo promedio en escena (min)", "Máximo a la vez", "Ingresos",
            "Vehículos", "Vehículos en surtidor", "Espera promedio en surtidor (min)", "Surtidor sin atender",
            "Visitas a la tienda", "Zonas (visitas · minutos)"]
        lines = [";".join(head)]
        for (when, cl_id, cam_id), b in sorted(groups.items(), key=lambda kv: (kv[0][0], cls.get(kv[0][1], ""),
                                                                                  cams.get(kv[0][2], ""))):
            zones = " | ".join(f"{name}: {z.get('visits', 0)} · {num(z.get('dwell_s', 0) / 60)}"
                               for name, z in sorted((b.get("zones") or {}).items()))
            cells = [when.strftime("%d/%m/%Y")] + ([when.strftime("%H:00")] if agrupar == "hora" else []) + [
                cls.get(cl_id, str(cl_id)), cams.get(cam_id, f"Cámara {cam_id}"), str(b["people"]),
                num(b["people_dwell_s"] / b["people"] / 60) if b["people"] else "", str(b["people_max"]),
                str(b["entries"]), str(b["vehicles"]), str(b["pump_vehicles"]),
                num(b["pump_wait_s"] / b["pump_wait_n"] / 60) if b["pump_wait_n"] else "", str(b["pump_unattended"]),
                str(b["shop_visits"]), zones]
            lines.append(";".join(c.replace(";", ",") for c in cells))
        name = (cls.get(client_id, "cliente") if client_id else "todos").replace(" ", "_")
        fname = f"metricas_{name}_{d1:%Y-%m-%d}_a_{d2:%Y-%m-%d}.csv"
        return Response(content="\ufeff" + "\r\n".join(lines) + "\r\n", media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="{fname}"'})

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
