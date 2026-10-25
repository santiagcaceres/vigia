from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import (JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, create_engine,
                        event, inspect, text)
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
    # planes contratados: "hurto" (alertas), "comercio" (gente, tiempos, destinos), "estacion" (vehículos, pistero, tienda)
    plans: Mapped[list] = mapped_column(JSON, default=lambda: ["hurto"])
    # horario del local por día (core/schedule.py): fuera de horario la PC central no abre el video
    hours: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Hik-Connect for Teams (video de Hikvision sin abrir puertos): clave de la API del equipo del cliente
    hc_app_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    hc_secret_enc: Mapped[str | None] = mapped_column(Text, nullable=True)
    hc_api_url: Mapped[str | None] = mapped_column(String(255), nullable=True)  # vacío = se busca la región
    hc_code_enc: Mapped[str | None] = mapped_column(Text, nullable=True)  # código de cifrado de video de sus grabadores
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
    # la PC de un usuario (admin o empleado): analiza y muestra las cámaras que el admin le asigna
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
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
    brand: Mapped[str] = mapped_column(String(20), default="hikvision")  # hikvision | dahua
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
    # "vivo": el agente mira el stream las 24 h. "alarmas": solo analiza el tramo de video de cada
    # alarma que el DVR manda a Hik-Partner Pro (mucho más barato: una PC atiende muchos locales).
    mode: Mapped[str] = mapped_column(String(20), default="vivo")  # vivo | alarmas | alarmas_equipo
    hpp_serial: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)  # serie del DVR en HPP
    hc_resource_id: Mapped[str | None] = mapped_column(String(64), nullable=True)  # cámara en Hik-Connect
    # qué mira la cámara: decide qué reglas se aplican (góndola, caja, entrada, depósito, calle, surtidor)
    profile: Mapped[str] = mapped_column(String(20), default="general")
    # con el local cerrado: apagada (no se abre el video) | alarmas (escucha al grabador) | vivo (sigue 24 h)
    off_hours: Mapped[str] = mapped_column(String(20), default="apagada")
    alarm_cfg: Mapped[dict] = mapped_column(JSON, default=dict)  # tipos de alarma, segundos antes/después
    preview_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    # vista en vivo: el panel marca que alguien la está mirando; la PC central sube cuadros mientras dure
    live_requested_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
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


class Alarm(Base):
    """Alarma de un DVR recibida desde Hik-Partner Pro (webhook). La PC central baja el tramo
    de video, lo analiza y, si hay una persona, lo convierte en un evento para revisar.

    Estados: pendiente -> procesando -> con_persona | sin_persona | error.
    Sin cámara configurada para esa serie/canal: sin_camara. Tipo no vigilado: ignorada.
    Origen (`source`): "hpp" (webhook de Hik-Partner Pro) o "equipo" (el grabador le avisó al agente).
    """
    __tablename__ = "alarms"
    __table_args__ = (UniqueConstraint("dedup_key", name="uq_alarms_dedup"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dedup_key: Mapped[str] = mapped_column(String(160))
    client_id: Mapped[int | None] = mapped_column(ForeignKey("clients.id"), nullable=True, index=True)
    camera_id: Mapped[int | None] = mapped_column(ForeignKey("cameras.id", ondelete="SET NULL"), nullable=True,
                                                  index=True)
    device_serial: Mapped[str] = mapped_column(String(32), index=True)
    channel: Mapped[int] = mapped_column(Integer, default=1)
    event_type: Mapped[str] = mapped_column(String(40))
    source: Mapped[str] = mapped_column(String(10), default="hpp")  # hpp | equipo
    target: Mapped[str | None] = mapped_column(String(20), nullable=True)  # human / vehicle / others (si el DVR lo dice)
    first_at: Mapped[datetime] = mapped_column(DateTime, index=True)  # UTC, primer disparo
    last_at: Mapped[datetime] = mapped_column(DateTime)  # UTC, último disparo agrupado en esta alarma
    triggers: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(20), default="pendiente", index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(String(500), nullable=True)
    stats: Mapped[dict] = mapped_column(JSON, default=dict)
    raw: Mapped[dict] = mapped_column(JSON, default=dict)
    event_id: Mapped[int | None] = mapped_column(ForeignKey("events.id", ondelete="SET NULL"), nullable=True)
    claimed_by: Mapped[int | None] = mapped_column(ForeignKey("agents.id", ondelete="SET NULL"), nullable=True)
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    received_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)


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


def add_missing_columns(engine) -> list[str]:
    """create_all no agrega columnas nuevas a tablas que ya existen: las agrega acá (solo columnas
    que admiten NULL o tienen valor por defecto simple). Así no hace falta migrar a mano."""
    added = []
    insp = inspect(engine)
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if not insp.has_table(table.name):
                continue
            have = {c["name"] for c in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name in have:
                    continue
                ddl = col.type.compile(engine.dialect)
                default = col.default.arg if col.default is not None and col.default.is_scalar else None
                if isinstance(default, bool):
                    ddl += " DEFAULT " + ("TRUE" if default else "FALSE")
                elif isinstance(default, (int, float)):
                    ddl += f" DEFAULT {default}"
                elif isinstance(default, str):
                    ddl += " DEFAULT '" + default.replace("'", "''") + "'"
                conn.execute(text(f'ALTER TABLE "{table.name}" ADD COLUMN "{col.name}" {ddl}'))
                added.append(f"{table.name}.{col.name}")
    if added:
        log.warning("Columnas agregadas a la base: %s", ", ".join(added))
    return added


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
        ensure_schema(engine)
    return engine, sessionmaker(engine, expire_on_commit=False)


# Subir este número cada vez que cambie el esquema (tabla o columna nueva): el servidor solo
# vuelve a correr create_all / ALTER cuando la versión guardada en la base no coincide. Así un
# arranque en frío en Vercel hace una sola consulta en vez de inspeccionar todas las tablas.
SCHEMA_VERSION = "2026-10-25.1"


class CameraMetric(Base):
    """Métricas de una cámara por hora (hora local del cliente). `data` es el cubo de core/metrics.py.
    La PC central lo manda cada 5 minutos y se reemplaza entero (upsert por cámara y hora)."""
    __tablename__ = "camera_metrics"
    __table_args__ = (UniqueConstraint("camera_id", "bucket_start", name="uq_metric_camera_bucket"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    client_id: Mapped[int] = mapped_column(ForeignKey("clients.id"), index=True)
    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id", ondelete="CASCADE"), index=True)
    bucket_start: Mapped[datetime] = mapped_column(DateTime, index=True)  # hora local del cliente, sin tz
    data: Mapped[dict] = mapped_column(JSON, default=dict)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class SchemaMeta(Base):
    __tablename__ = "schema_meta"
    key: Mapped[str] = mapped_column(String(40), primary_key=True)
    value: Mapped[str] = mapped_column(String(80))


def ensure_schema(engine) -> bool:
    """Crea o actualiza el esquema si hace falta. Devuelve True si tuvo que tocar algo."""
    try:
        with engine.connect() as conn:
            row = conn.execute(text("SELECT value FROM schema_meta WHERE key = 'version'")).first()
        if row and row[0] == SCHEMA_VERSION:
            return False
    except Exception:
        pass  # la tabla todavía no existe: hay que crear todo
    try:
        Base.metadata.create_all(engine)
        add_missing_columns(engine)
        with engine.begin() as conn:
            if engine.dialect.name == "postgresql":
                # Las tablas quedan cerradas para la API pública de Supabase (anon/authenticated):
                # solo este servidor, conectado como dueño de la base, puede leerlas.
                for table in Base.metadata.sorted_tables:
                    conn.execute(text(f'ALTER TABLE "{table.name}" ENABLE ROW LEVEL SECURITY'))
            conn.execute(text("DELETE FROM schema_meta WHERE key = 'version'"))
            conn.execute(text("INSERT INTO schema_meta (key, value) VALUES ('version', :v)"), {"v": SCHEMA_VERSION})
        return True
    except Exception:
        log.exception("No se pudo crear o verificar el esquema (¿otra instancia lo está creando?)")
        return False
