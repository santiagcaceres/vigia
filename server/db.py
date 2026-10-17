from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, create_engine, event, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.pool import NullPool

log = logging.getLogger("vigia.db")
ROLES = ("admin", "entrenador", "cliente")


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def iso(dt: datetime | None) -> str | None:
    return dt.isoformat(timespec="seconds") + "Z" if dt else None


class Base(DeclarativeBase):
    pass


class Client(Base):
    __tablename__ = "clients"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    timezone: Mapped[str] = mapped_column(String(64), default="America/Argentina/Buenos_Aires")
    # Contacto del cliente: se usa SOLO cuando un empleado decide avisarle de un hurto.
    telegram_chat_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    contact_whatsapp: Mapped[str | None] = mapped_column(String(32), nullable=True)  # formato +5989...
    contact_email: Mapped[str | None] = mapped_column(String(200), nullable=True)
    webhook_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(80), unique=True)
    password_hash: Mapped[str] = mapped_column(String(300))
    role: Mapped[str] = mapped_column(String(20))
    client_id: Mapped[int | None] = mapped_column(ForeignKey("clients.id"), nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class SessionToken(Base):
    __tablename__ = "sessions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    expires_at: Mapped[datetime] = mapped_column(DateTime)


class Agent(Base):
    """PC que corre el análisis. Con client_id es el agente de un local; con client_id NULL es
    un agente *central* (la PC de la empresa) que atiende las cámaras que se le asignen, de
    cualquier cliente, llegando a cada DVR por VPN."""
    __tablename__ = "agents"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    client_id: Mapped[int | None] = mapped_column(ForeignKey("clients.id"), nullable=True, index=True)
    name: Mapped[str] = mapped_column(String(120))
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    process_imports: Mapped[bool] = mapped_column(Boolean, default=True)
    last_seen: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_status: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Camera(Base):
    __tablename__ = "cameras"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    client_id: Mapped[int] = mapped_column(ForeignKey("clients.id"), index=True)
    agent_id: Mapped[int | None] = mapped_column(ForeignKey("agents.id", ondelete="SET NULL"), nullable=True)
    name: Mapped[str] = mapped_column(String(120))
    host: Mapped[str] = mapped_column(String(255), default="")
    username: Mapped[str] = mapped_column(String(120), default="admin")
    password_enc: Mapped[str | None] = mapped_column(Text, nullable=True)
    channel: Mapped[int] = mapped_column(Integer, default=1)
    rtsp_port: Mapped[int] = mapped_column(Integer, default=554)
    http_port: Mapped[int] = mapped_column(Integer, default=80)
    detect_stream: Mapped[str] = mapped_column(String(10), default="sub")
    nvr_clips: Mapped[bool] = mapped_column(Boolean, default=True)
    playback_time: Mapped[str] = mapped_column(String(10), default="local")
    source_url_enc: Mapped[str | None] = mapped_column(Text, nullable=True)
    detector: Mapped[str] = mapped_column(String(20), default="yolo")
    process_fps: Mapped[float] = mapped_column(Float, default=5.0)
    zones: Mapped[list] = mapped_column(JSON, default=list)
    rules: Mapped[dict] = mapped_column(JSON, default=dict)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    preview_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    preview_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class VideoImport(Base):
    __tablename__ = "imports"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    client_id: Mapped[int] = mapped_column(ForeignKey("clients.id"), index=True)
    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id"), index=True)
    filename: Mapped[str] = mapped_column(String(255))
    storage_path: Mapped[str] = mapped_column(String(500))
    size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    recorded_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)  # hora local de inicio
    duration_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    # subiendo | en_cola | procesando | listo | error
    status: Mapped[str] = mapped_column(String(20), default="subiendo", index=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[str | None] = mapped_column(String(500), nullable=True)
    claimed_by: Mapped[int | None] = mapped_column(ForeignKey("agents.id", ondelete="SET NULL"), nullable=True)
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Event(Base):
    __tablename__ = "events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    uuid: Mapped[str] = mapped_column(String(40), unique=True, index=True)
    client_id: Mapped[int] = mapped_column(ForeignKey("clients.id"), index=True)
    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id"), index=True)
    source: Mapped[str] = mapped_column(String(12), default="vivo")  # vivo | importado
    import_id: Mapped[int | None] = mapped_column(ForeignKey("imports.id"), nullable=True, index=True)
    type: Mapped[str] = mapped_column(String(40), index=True)
    severity: Mapped[str] = mapped_column(String(10))
    zone: Mapped[str | None] = mapped_column(String(120), nullable=True)
    description: Mapped[str] = mapped_column(String(500))
    track_id: Mapped[int] = mapped_column(Integer, default=0)
    video_ts: Mapped[float] = mapped_column(Float, default=0.0)
    occurred_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    received_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    features: Mapped[dict] = mapped_column(JSON, default=dict)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    alert: Mapped[bool] = mapped_column(Boolean, default=True)
    snapshot_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    clip_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    clip_source: Mapped[str | None] = mapped_column(String(10), nullable=True)
    label: Mapped[str | None] = mapped_column(String(20), nullable=True, index=True)
    note: Mapped[str | None] = mapped_column(String(500), nullable=True)
    labeled_by: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    labeled_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class Incident(Base):
    """Hurto real marcado a mano en un video importado (para medir qué detecta el sistema)."""
    __tablename__ = "incidents"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    import_id: Mapped[int] = mapped_column(ForeignKey("imports.id", ondelete="CASCADE"), index=True)
    start_s: Mapped[float] = mapped_column(Float)
    end_s: Mapped[float] = mapped_column(Float)
    note: Mapped[str | None] = mapped_column(String(300), nullable=True)


class ClientReport(Base):
    """Aviso que un empleado le manda al cliente con el clip de un posible hurto.

    Genera un link público, temporal y sin login, para que el dueño vea el clip desde
    el celular. Se envía por los canales que tenga configurados el cliente (portal siempre,
    + Telegram / WhatsApp / email / webhook si están disponibles).
    """
    __tablename__ = "client_reports"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_id: Mapped[int] = mapped_column(ForeignKey("events.id", ondelete="CASCADE"), index=True)
    client_id: Mapped[int] = mapped_column(ForeignKey("clients.id"), index=True)
    public_token: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    message: Mapped[str] = mapped_column(String(1000), default="")
    channels: Mapped[list] = mapped_column(JSON, default=list)   # canales intentados
    delivery: Mapped[dict] = mapped_column(JSON, default=dict)   # resultado por canal
    status: Mapped[str] = mapped_column(String(12), default="enviado")  # enviado | parcial | error
    sent_by: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    sent_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    viewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime)


class FilterModel(Base):
    __tablename__ = "filter_models"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    client_id: Mapped[int | None] = mapped_column(ForeignKey("clients.id"), nullable=True)  # None = global
    spec: Mapped[dict] = mapped_column(JSON)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    n_samples: Mapped[int] = mapped_column(Integer, default=0)
    active: Mapped[bool] = mapped_column(Boolean, default=False)
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


def normalize_url(url: str) -> str:
    """Acepta la cadena de conexión tal como la copia Supabase (postgresql://…)."""
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


def make_session_factory(url: str, create_schema: bool = True):
    url = normalize_url(url)
    if url.startswith("sqlite"):
        engine = create_engine(url, connect_args={"check_same_thread": False, "timeout": 30})

        @event.listens_for(engine, "connect")
        def _pragma(dbapi_conn, _):
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()
    else:
        # Funciones serverless: sin pool propio (lo hace el pooler de Supabase, puerto 6543)
        # y sin sentencias preparadas, que el pooler en modo transacción no admite.
        engine = create_engine(url, poolclass=NullPool, connect_args={"prepare_threshold": None})
    if create_schema:
        try:
            Base.metadata.create_all(engine)
            if engine.dialect.name == "postgresql":
                with engine.begin() as conn:
                    # Las tablas quedan cerradas para la API pública de Supabase (anon/authenticated):
                    # solo este servidor, conectado como dueño de la base, puede leerlas.
                    for table in Base.metadata.sorted_tables:
                        conn.execute(text(f'ALTER TABLE "{table.name}" ENABLE ROW LEVEL SECURITY'))
        except Exception:
            log.exception("No se pudo crear o verificar el esquema (¿otra instancia lo está creando?)")
    return engine, sessionmaker(engine, expire_on_commit=False)
