"""«Entrenar objetos» en el servidor (18 nov): objetos con sus clases, fotos en el Storage, marcas, y los trabajos que
hacen las máquinas de análisis (sacar fotos de un video, pre-marcar con la IA, entrenar) con sus versiones.

La app (staff) usa `/api/objects/...`; las máquinas, `/api/agent/objects/...` con su token. Las fotos y los videos van
directo al Storage con URLs firmadas (Vercel no acepta pedidos de más de 4,5 MB). Vercel no entrena ni toca imágenes: solo
guarda datos y reparte trabajo. La lógica de los datos está en core/objetos.py (sin OpenCV: se puede importar acá).
"""

from __future__ import annotations

import logging
import re
from datetime import timedelta
from pathlib import Path

from fastapi import Depends, HTTPException, Response
from pydantic import BaseModel, Field
from sqlalchemy import func, or_, update

from core import objetos as ob

from .db import Agent, Camera, Client, ObjectImage, ObjectJob, ObjectModel, ObjectVersion, User, iso, utcnow
from .storage import StorageError

log = logging.getLogger("vigia.objetos")

MACHINE_ONLINE = timedelta(minutes=3)
STALE_JOB = timedelta(minutes=20)  # una máquina que dejó de avisar: el trabajo vuelve a la cola
MAX_IMAGES_PER_MODEL = 20000
MAX_BATCH = 50
VIDEO_EXT = {".mp4", ".avi", ".mkv", ".mov", ".m4v", ".ts", ".webm", ".dav", ".264", ".h264"}
PHOTO_SOURCES = ("foto", "video", "camara")
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")
_UID = re.compile(r"^[0-9a-f]{40}$")


class ClassIn(BaseModel):
    name: str = Field(max_length=80)
    prompt: str | None = Field(None, max_length=120)
    key: str | None = Field(None, max_length=10)


class ObjectIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    classes: list[ClassIn] = Field(min_length=1, max_length=20)
    plan: str = Field("objetos", max_length=20)
    client_id: int | None = None
    notes: str | None = Field(None, max_length=1000)


class PhotoIn(BaseModel):
    uuid: str = Field(max_length=40)
    name: str | None = Field(None, max_length=200)
    width: int = Field(ge=1, le=20000)
    height: int = Field(ge=1, le=20000)
    source: str = Field("foto", max_length=12)
    camera_id: int | None = None
    video_ts: float | None = Field(None, ge=0, le=10 * 24 * 3600)


class PhotosIn(BaseModel):
    images: list[PhotoIn] = Field(max_length=MAX_BATCH)


class IdsIn(BaseModel):
    ids: list[int] = Field(max_length=500)


class LabelIn(BaseModel):
    boxes: list[dict] | None = Field(None, max_length=ob.MAX_BOXES)
    status: str | None = Field(None, max_length=12)


class VideoIn(BaseModel):
    filename: str = Field(max_length=255)
    size: int | None = Field(None, ge=0)
    every_s: float = Field(ob.VIDEO_EVERY_S, ge=0.5, le=600)
    max_images: int = Field(ob.VIDEO_MAX_IMAGES, ge=1, le=2000)
    camera_id: int | None = None


class TrainIn(BaseModel):
    epochs: int | None = Field(None, ge=1, le=500)
    fresh: bool = False  # True = empezar del YOLO de fábrica (no desde la versión anterior)


class ProgressIn(BaseModel):
    progress: float = Field(0.0, ge=0, le=1)
    message: str | None = Field(None, max_length=300)


class DoneIn(BaseModel):
    result: dict = Field(default_factory=dict)


class FailIn(BaseModel):
    error: str = Field(max_length=500)


class ProposalsIn(BaseModel):
    items: list[dict] = Field(max_length=500)
    by: str = Field(max_length=80)


def install(app, *, db, staff, admin_only, current_agent, storage, get_or_404, plan_names: dict[str, str],
            max_video_bytes: int) -> None:
    """Agrega los pedidos de «Entrenar objetos» a la app de FastAPI (los helpers vienen de server/main.py)."""

    def bad(msg: str, code: int = 422):
        raise HTTPException(code, msg)

    def data_error(fn, *a, **k):
        try:
            return fn(*a, **k)
        except ob.ObjectDataError as exc:
            raise HTTPException(422, str(exc)) from exc

    def keys_of(m: ObjectModel) -> list[str]:
        return [c["key"] for c in m.classes or []]

    def image_path(m: ObjectModel, uid: str) -> str:
        return f"training/objects/{m.id}/img/{uid}.jpg"

    # ---------------------------------------------------------------- máquinas
    def capable(a: Agent, kind: str) -> bool:
        """Si esta máquina puede hacer ese trabajo: sacar fotos de un video lo hace cualquiera con la versión nueva;
        pre-marcar y entrenar, solo las que tienen Ultralytics (instaladas con ENTRENAR=si)."""
        caps = (a.last_status or {}).get("objects")
        if not isinstance(caps, dict) or a.draining or (a.last_status or {}).get("leaving"):
            return False
        return bool(caps.get("video")) if kind == "video" else bool(caps.get("ultra"))

    def online_machines(s) -> list[Agent]:
        since = utcnow() - MACHINE_ONLINE
        return [a for a in s.query(Agent).filter(Agent.kind == "servidor", Agent.last_seen >= since)]

    def machines_info(s) -> list[dict]:
        out = []
        for a in online_machines(s):
            caps = (a.last_status or {}).get("objects")
            if not isinstance(caps, dict):
                continue
            out.append({"id": a.id, "name": a.name, "video": bool(caps.get("video")),
                        "train": bool(caps.get("ultra")), "gpu": bool(caps.get("gpu")),
                        "device": str(caps.get("device") or "")[:60]})
        return out

    # ---------------------------------------------------------------- lo que se muestra
    def counts(s, mid: int) -> dict:
        rows = s.query(ObjectImage.status, ObjectImage.split, func.count()).filter(
            ObjectImage.model_id == mid).group_by(ObjectImage.status, ObjectImage.split).all()
        by = {st: 0 for st in ob.STATUSES}
        test_marked = 0
        for st, sp, n in rows:
            by[st] = by.get(st, 0) + n
            if st == "marcada" and sp == "prueba":
                test_marked += n
        proposed = s.query(func.count()).select_from(ObjectImage).filter(
            ObjectImage.model_id == mid, ObjectImage.status == "sin_marcar",
            ObjectImage.proposed_by.isnot(None)).scalar() or 0
        return {**by, "total": sum(by.values()) - by["subiendo"], "test_marked": test_marked,
                "with_proposals": proposed}

    def box_counts(s, m: ObjectModel) -> dict[str, int]:
        out = {k: 0 for k in keys_of(m)}
        for (boxes,) in s.query(ObjectImage.boxes).filter(ObjectImage.model_id == m.id,
                                                          ObjectImage.status == "marcada"):
            for b in boxes or []:
                if b.get("k") in out:
                    out[b["k"]] += 1
        return out

    def job_dict(j: ObjectJob, names: dict[int, str] | None = None) -> dict:
        return {"id": j.id, "model_id": j.model_id, "kind": j.kind, "kind_name": ob.JOB_NAMES.get(j.kind, j.kind),
                "status": j.status, "status_name": ob.JOB_STATE_NAMES.get(j.status, j.status),
                "progress": round(j.progress or 0.0, 3), "message": j.message, "error": j.error,
                "result": j.result or {}, "params": j.params or {}, "filename": j.filename,
                "machine": (names or {}).get(j.claimed_by) if j.claimed_by else None,
                "created_at": iso(j.created_at), "started_at": iso(j.started_at), "finished_at": iso(j.finished_at)}

    def version_dict(v: ObjectVersion, m: ObjectModel) -> dict:
        note, todo = ob.grade((v.metrics or {}).get("map50"))
        return {"id": v.id, "model_id": v.model_id, "number": v.number, "classes": v.classes or [],
                "metrics": v.metrics or {}, "grade": note, "advice": todo,
                "text": ob.metrics_text(v.metrics or {}, v.classes or []), "n_images": v.n_images,
                "n_boxes": v.n_boxes, "trained_on": v.trained_on, "active": m.active_version_id == v.id,
                "created_at": iso(v.created_at)}

    def model_dict(s, m: ObjectModel, full: bool = False) -> dict:
        c = counts(s, m.id)
        boxes = box_counts(s, m)
        d = {"id": m.id, "name": m.name, "classes": m.classes or [], "plan": m.plan,
             "plan_name": plan_names.get(m.plan, m.plan), "client_id": m.client_id,
             "client_name": (s.get(Client, m.client_id).name if m.client_id and s.get(Client, m.client_id) else None),
             "notes": m.notes, "active_version_id": m.active_version_id, "counts": c, "boxes": boxes,
             "check": ob.train_check(c["marcada"], boxes, m.classes or [], c["test_marked"]),
             "versions": s.query(func.count()).select_from(ObjectVersion).filter(
                 ObjectVersion.model_id == m.id).scalar() or 0,
             "created_at": iso(m.created_at), "updated_at": iso(m.updated_at)}
        busy = s.query(ObjectJob).filter(ObjectJob.model_id == m.id,
                                         ObjectJob.status.in_(("subiendo", "en_cola", "procesando"))).all()
        d["busy"] = {j.kind: j.id for j in busy}
        if full:
            names = {a.id: a.name for a in s.query(Agent.id, Agent.name)}
            d["jobs"] = [job_dict(j, names) for j in s.query(ObjectJob).filter(ObjectJob.model_id == m.id)
                         .order_by(ObjectJob.id.desc()).limit(15)]
            d["version_list"] = [version_dict(v, m) for v in s.query(ObjectVersion).filter(
                ObjectVersion.model_id == m.id).order_by(ObjectVersion.number.desc())]
            d["machines"] = machines_info(s)
        return d

    def image_dict(i: ObjectImage) -> dict:
        return {"id": i.id, "uuid": i.uuid, "width": i.width, "height": i.height, "source": i.source,
                "source_name": i.source_name, "camera_id": i.camera_id, "video_ts": i.video_ts,
                "status": i.status, "split": i.split, "boxes": i.boxes or [], "proposed": i.proposed or [],
                "proposed_by": i.proposed_by, "doubt": i.doubt or 0.0, "labeled_at": iso(i.labeled_at),
                "created_at": iso(i.created_at)}

    def clean_model_body(s, body: ObjectIn, old: ObjectModel | None) -> tuple[str, list[dict]]:
        name = " ".join(body.name.split())
        if not name:
            bad("Poné el nombre del objeto (por ejemplo «Pallets»)")
        if body.plan not in plan_names:
            bad("Ese plan de métricas no existe")
        if body.client_id is not None:
            get_or_404(s, Client, body.client_id, "Cliente")
        classes = data_error(ob.clean_classes, [c.model_dump() for c in body.classes], old.classes if old else None)
        q = s.query(ObjectModel).filter(func.lower(ObjectModel.name) == name.lower())
        if old is not None:
            q = q.filter(ObjectModel.id != old.id)
        if q.first():
            bad(f"Ya hay un objeto que se llama «{name}»", 409)
        return name, classes

    def remove_files(paths: list[str]) -> None:
        paths = [p for p in paths if p]
        if not paths:
            return
        try:
            storage.remove(paths)
        except StorageError:
            log.warning("No se pudieron borrar %d archivos de objetos", len(paths))

    # ---------------------------------------------------------------- objetos
    @app.get("/api/objects")
    def list_objects(_: User = Depends(staff), s=Depends(db)) -> list[dict]:
        return [model_dict(s, m) for m in s.query(ObjectModel).order_by(ObjectModel.name)]

    @app.post("/api/objects", status_code=201)
    def create_object(body: ObjectIn, user: User = Depends(staff), s=Depends(db)) -> dict:
        name, classes = clean_model_body(s, body, None)
        m = ObjectModel(name=name, classes=classes, plan=body.plan, client_id=body.client_id,
                        notes=(body.notes or "").strip() or None, created_by=user.id)
        s.add(m)
        s.commit()
        log.info("Objeto nuevo «%s» (%s) por %s: %s", name, m.id, user.username, [c["name"] for c in classes])
        return model_dict(s, m, full=True)

    @app.get("/api/objects/{mid}")
    def get_object(mid: int, _: User = Depends(staff), s=Depends(db)) -> dict:
        return model_dict(s, get_or_404(s, ObjectModel, mid, "Objeto"), full=True)

    @app.put("/api/objects/{mid}")
    def update_object(mid: int, body: ObjectIn, user: User = Depends(staff), s=Depends(db)) -> dict:
        m = get_or_404(s, ObjectModel, mid, "Objeto")
        name, classes = clean_model_body(s, body, m)
        gone = set(keys_of(m)) - {c["key"] for c in classes}
        if gone:
            used = {k: n for k, n in box_counts(s, m).items() if k in gone and n}
            if used:
                names = ob.class_names(m.classes)
                bad("No se puede sacar " + ", ".join(f"«{names[k]}» ({n} recuadros marcados)" for k, n in used.items())
                    + ": primero borrá esos recuadros (o renombrala en vez de sacarla)", 409)
        m.name, m.classes, m.plan, m.client_id = name, classes, body.plan, body.client_id
        m.notes, m.updated_at = (body.notes or "").strip() or None, utcnow()
        s.commit()
        log.info("Objeto «%s» (%s) editado por %s", name, m.id, user.username)
        return model_dict(s, m, full=True)

    @app.delete("/api/objects/{mid}", status_code=204)
    def delete_object(mid: int, user: User = Depends(admin_only), s=Depends(db)) -> None:
        m = get_or_404(s, ObjectModel, mid, "Objeto")
        if s.query(ObjectJob).filter(ObjectJob.model_id == mid, ObjectJob.status == "procesando").first():
            bad("Una máquina está trabajando con este objeto: cancelá el trabajo y esperá a que termine", 409)
        paths = [p for (p,) in s.query(ObjectImage.path).filter(ObjectImage.model_id == mid)]
        for v in s.query(ObjectVersion).filter(ObjectVersion.model_id == mid):
            paths += [v.onnx_path, v.pt_path]
        paths += [j.storage_path for j in s.query(ObjectJob).filter(ObjectJob.model_id == mid) if j.storage_path]
        for tbl in (ObjectImage, ObjectJob, ObjectVersion):
            s.query(tbl).filter(tbl.model_id == mid).delete(synchronize_session=False)
        s.delete(m)
        s.commit()
        remove_files(paths)
        log.info("Objeto %s borrado por %s (%d archivos)", mid, user.username, len(paths))

    # ---------------------------------------------------------------- fotos
    @app.get("/api/objects/{mid}/images")
    def list_images(mid: int, status: str | None = None, order: str = "nuevas", limit: int = 200, offset: int = 0,
                    _: User = Depends(staff), s=Depends(db)) -> dict:
        get_or_404(s, ObjectModel, mid, "Objeto")
        q = s.query(ObjectImage).filter(ObjectImage.model_id == mid, ObjectImage.status != "subiendo")
        if status:
            if status not in ob.STATUSES:
                bad("Ese estado no existe")
            q = q.filter(ObjectImage.status == status)
        total = q.count()
        if order == "dudosas":  # las que la IA duda primero (aprende más con esas)
            q = q.order_by(ObjectImage.doubt.desc(), ObjectImage.id)
        elif order == "viejas":
            q = q.order_by(ObjectImage.id)
        else:
            q = q.order_by(ObjectImage.id.desc())
        items = q.offset(max(0, offset)).limit(max(1, min(limit, 500))).all()
        return {"total": total, "items": [image_dict(i) for i in items]}

    @app.post("/api/objects/images/urls")
    def image_urls(body: IdsIn, _: User = Depends(staff), s=Depends(db)) -> dict:
        """Las URLs firmadas (1 hora) para bajar varias fotos de una vez."""
        rows = s.query(ObjectImage.id, ObjectImage.path).filter(ObjectImage.id.in_(body.ids[:200])).all()
        try:
            urls = storage.signed_download_urls([p for _, p in rows], 3600)
        except StorageError as exc:
            raise HTTPException(502, str(exc)) from exc
        return {"urls": {str(i): urls.get(p.lstrip("/")) or urls.get(p) for i, p in rows}}

    def register_photos(s, m: ObjectModel, photos: list[PhotoIn], user_id: int | None,
                        job: ObjectJob | None = None) -> list[dict]:
        """Crea las fotos que no estaban («subiendo») y devuelve dónde subir cada una. Las que ya estaban (la misma foto
        subida de nuevo) no se tocan."""
        have = s.query(func.count()).select_from(ObjectImage).filter(ObjectImage.model_id == m.id).scalar() or 0
        if have + len(photos) > MAX_IMAGES_PER_MODEL:
            bad(f"Este objeto ya tiene {have} fotos: el máximo es {MAX_IMAGES_PER_MODEL}", 413)
        uids = [p.uuid for p in photos]
        if any(not _UID.match(u) for u in uids):
            bad("Cada foto va con su huella (40 letras de 0 a 9 y a-f)")
        old = {i.uuid: i for i in s.query(ObjectImage).filter(ObjectImage.model_id == m.id,
                                                             ObjectImage.uuid.in_(uids))}
        out, new = [], []
        for p in photos:
            if p.source not in PHOTO_SOURCES:
                bad("Origen de la foto desconocido")
            if p.camera_id is not None:
                get_or_404(s, Camera, p.camera_id, "Cámara")
            i = old.get(p.uuid)
            if i is not None and i.status != "subiendo":
                out.append({"uuid": p.uuid, "id": i.id, "existed": True, "upload_url": None})
                continue
            if i is None:
                i = ObjectImage(model_id=m.id, uuid=p.uuid, path=image_path(m, p.uuid), split=ob.split_of(p.uuid),
                                created_by=user_id, status="subiendo")
                s.add(i)
                old[p.uuid] = i
            i.width, i.height, i.source, i.camera_id = p.width, p.height, p.source, p.camera_id
            i.source_name = (p.name or "").strip()[:200] or None
            i.video_ts, i.job_id = p.video_ts, job.id if job else None
            new.append((p.uuid, i))
        s.flush()
        for uid, i in new:
            try:
                url = storage.signed_upload_url(i.path)
            except StorageError as exc:
                s.rollback()
                raise HTTPException(502, str(exc)) from exc
            out.append({"uuid": uid, "id": i.id, "existed": False, "upload_url": url})
        s.commit()
        return out

    def confirm_photos(s, m: ObjectModel, ids: list[int]) -> int:
        res = s.execute(update(ObjectImage).where(ObjectImage.model_id == m.id, ObjectImage.id.in_(ids),
                                                  ObjectImage.status == "subiendo").values(status="sin_marcar"))
        m.updated_at = utcnow()
        s.commit()
        return res.rowcount or 0

    @app.post("/api/objects/{mid}/images")
    def add_images(mid: int, body: PhotosIn, user: User = Depends(staff), s=Depends(db)) -> dict:
        """Fotos nuevas (la app las achica a 1280 px y las sube directo al Storage con `upload_url`, PUT image/jpeg);
        después, `/images/uploaded` con los ids que subió."""
        m = get_or_404(s, ObjectModel, mid, "Objeto")
        return {"items": register_photos(s, m, body.images, user.id)}

    @app.post("/api/objects/{mid}/images/uploaded")
    def images_uploaded(mid: int, body: IdsIn, _: User = Depends(staff), s=Depends(db)) -> dict:
        m = get_or_404(s, ObjectModel, mid, "Objeto")
        return {"ok": confirm_photos(s, m, body.ids)}

    @app.put("/api/objects/images/{iid}")
    def label_image(iid: int, body: LabelIn, user: User = Depends(staff), s=Depends(db)) -> dict:
        """Lo que marcó una persona. Con `boxes` (aunque sea una lista vacía = «no hay ninguno») queda «marcada»;
        `status` «descartada» la saca del entrenamiento (foto mala o repetida) y «sin_marcar» la devuelve."""
        i = get_or_404(s, ObjectImage, iid, "Foto")
        m = s.get(ObjectModel, i.model_id)
        status = body.status
        if status is not None and status not in ("sin_marcar", "marcada", "descartada"):
            bad("Ese estado no existe")
        if body.boxes is not None:
            i.boxes = data_error(ob.clean_boxes, body.boxes, keys_of(m))
            status = status or "marcada"
        if status:
            i.status = status
        if i.status == "marcada":
            i.labeled_by, i.labeled_at = user.id, utcnow()
        m.updated_at = utcnow()
        s.commit()
        return image_dict(i)

    @app.delete("/api/objects/images/{iid}", status_code=204)
    def delete_image(iid: int, _: User = Depends(staff), s=Depends(db)) -> None:
        i = get_or_404(s, ObjectImage, iid, "Foto")
        path = i.path
        s.delete(i)
        s.commit()
        remove_files([path])

    # ---------------------------------------------------------------- trabajos
    def busy_job(s, mid: int, kind: str) -> ObjectJob | None:
        return s.query(ObjectJob).filter(ObjectJob.model_id == mid, ObjectJob.kind == kind,
                                         ObjectJob.status.in_(("en_cola", "procesando"))).first()

    def no_machine_note(s, kind: str) -> str | None:
        ok = [a for a in online_machines(s) if capable(a, kind)]
        if ok:
            return None
        if kind == "video":
            return ("Ahora no hay ninguna máquina de análisis en línea con la versión nueva: el video queda en cola y "
                    "se procesa apenas haya una.")
        return ("Ahora no hay ninguna máquina que pueda entrenar (hace falta una instalada con ENTRENAR=si, ver "
                "deploy/maquina-analisis/README.md): queda en cola y arranca apenas haya una.")

    @app.post("/api/objects/{mid}/videos", status_code=201)
    def add_video(mid: int, body: VideoIn, user: User = Depends(staff), s=Depends(db)) -> dict:
        """Un video para que una máquina saque las fotos: se sube entero al Storage (`upload_url`, PUT) y después
        `/api/objects/jobs/{id}/uploaded`."""
        m = get_or_404(s, ObjectModel, mid, "Objeto")
        name = Path(body.filename).name
        if Path(name).suffix.lower() not in VIDEO_EXT:
            bad(f"Formato no soportado. Usar: {', '.join(sorted(VIDEO_EXT))}")
        if body.size and body.size > max_video_bytes:
            bad(f"El video pesa {body.size / 1e6:.0f} MB y a la nube se suben hasta {max_video_bytes / 1e6:.0f} MB: "
                f"sacá las fotos en esta PC (sube solo las fotos)", 413)
        if body.camera_id is not None:
            get_or_404(s, Camera, body.camera_id, "Cámara")
        j = ObjectJob(model_id=m.id, kind="video", status="subiendo", filename=name, camera_id=body.camera_id,
                      params={"every_s": body.every_s, "max_images": body.max_images, "size": body.size},
                      created_by=user.id)
        s.add(j)
        s.flush()
        j.storage_path = f"training/objects/{m.id}/videos/{j.id}-{_SAFE.sub('_', name)[-100:]}"
        try:
            url = storage.signed_upload_url(j.storage_path)
        except StorageError as exc:
            s.rollback()
            raise HTTPException(502, str(exc)) from exc
        s.commit()
        return {**job_dict(j), "upload_url": url}

    @app.post("/api/objects/jobs/{jid}/uploaded")
    def video_uploaded(jid: int, _: User = Depends(staff), s=Depends(db)) -> dict:
        j = get_or_404(s, ObjectJob, jid, "Trabajo")
        if j.kind != "video" or j.status != "subiendo":
            bad("Ese video ya estaba registrado", 409)
        try:
            ok = storage.exists(j.storage_path)
        except StorageError as exc:
            raise HTTPException(502, str(exc)) from exc
        if not ok:
            bad("El video no llegó a la nube: volvé a subirlo", 409)
        j.status = "en_cola"
        s.commit()
        return {**job_dict(j), "note": no_machine_note(s, "video")}

    def prelabel_spec(s, m: ObjectModel) -> dict | None:
        """Con qué pre-marcar: la versión activa (el modelo propio) o, si no hay, YOLOE con las palabras en inglés de las
        clases. None si no hay ninguna de las dos."""
        v = s.get(ObjectVersion, m.active_version_id) if m.active_version_id else None
        if v is not None and v.pt_path:
            return {"method": "version", "version_id": v.id, "number": v.number, "keys": [c["key"] for c in v.classes]}
        prompts = {c["key"]: c["prompt"] for c in m.classes or [] if c.get("prompt")}
        if prompts:
            return {"method": "yoloe", "prompts": prompts}
        return None

    @app.post("/api/objects/{mid}/prelabel", status_code=201)
    def request_prelabel(mid: int, user: User = Depends(staff), s=Depends(db)) -> dict:
        m = get_or_404(s, ObjectModel, mid, "Objeto")
        if busy_job(s, mid, "premarcar"):
            bad("Ya se están pre-marcando las fotos de este objeto", 409)
        spec = prelabel_spec(s, m)
        if spec is None:
            bad("Para pre-marcar hace falta una versión entrenada y activa, o la palabra en inglés de alguna clase "
                "(«Editar objeto»)")
        todo = s.query(func.count()).select_from(ObjectImage).filter(ObjectImage.model_id == mid,
                                                                     ObjectImage.status == "sin_marcar").scalar()
        if not todo:
            bad("No hay fotos sin marcar")
        j = ObjectJob(model_id=mid, kind="premarcar", params=spec, created_by=user.id)
        s.add(j)
        s.commit()
        return {**job_dict(j), "note": no_machine_note(s, "premarcar")}

    @app.post("/api/objects/{mid}/train", status_code=201)
    def request_training(mid: int, body: TrainIn, user: User = Depends(staff), s=Depends(db)) -> dict:
        m = get_or_404(s, ObjectModel, mid, "Objeto")
        if busy_job(s, mid, "entrenar"):
            bad("Ya hay un entrenamiento de este objeto en cola o en marcha", 409)
        c = counts(s, mid)
        check = ob.train_check(c["marcada"], box_counts(s, m), m.classes or [], c["test_marked"])
        if not check["ok"]:
            bad(" ".join(check["problems"]))
        params = {"epochs": body.epochs or ob.auto_epochs(c["marcada"]), "imgsz": ob.IMGSZ, "patience": 25,
                  "base": "yolo11n.pt"}
        prev = s.query(ObjectVersion).filter(ObjectVersion.model_id == mid).order_by(
            ObjectVersion.number.desc()).first()
        if prev is not None and prev.pt_path and not body.fresh and \
                [x["key"] for x in prev.classes or []] == keys_of(m):
            params.update(base="version", base_version_id=prev.id, base_number=prev.number)
        j = ObjectJob(model_id=mid, kind="entrenar", params=params, created_by=user.id)
        s.add(j)
        s.commit()
        log.info("Entrenamiento del objeto «%s» pedido por %s: %d fotos marcadas", m.name, user.username, c["marcada"])
        return {**job_dict(j), "note": no_machine_note(s, "entrenar"), "warnings": check["warnings"]}

    @app.get("/api/objects/{mid}/jobs")
    def list_jobs(mid: int, _: User = Depends(staff), s=Depends(db)) -> list[dict]:
        get_or_404(s, ObjectModel, mid, "Objeto")
        names = {a.id: a.name for a in s.query(Agent.id, Agent.name)}
        return [job_dict(j, names) for j in s.query(ObjectJob).filter(ObjectJob.model_id == mid)
                .order_by(ObjectJob.id.desc()).limit(30)]

    @app.post("/api/objects/jobs/{jid}/cancel")
    def cancel_job(jid: int, user: User = Depends(staff), s=Depends(db)) -> dict:
        j = get_or_404(s, ObjectJob, jid, "Trabajo")
        if j.status not in ("subiendo", "en_cola", "procesando"):
            bad("Ese trabajo ya terminó", 409)
        j.status, j.finished_at = "cancelado", utcnow()
        j.message = f"Cancelado por {user.username}"
        s.commit()
        if j.kind == "video" and j.storage_path:
            remove_files([j.storage_path])
        return job_dict(j)

    # ---------------------------------------------------------------- versiones
    @app.post("/api/objects/versions/{vid}/activate")
    def activate_version(vid: int, user: User = Depends(staff), s=Depends(db)) -> dict:
        v = get_or_404(s, ObjectVersion, vid, "Versión")
        m = s.get(ObjectModel, v.model_id)
        if [c["key"] for c in v.classes or []] != keys_of(m):
            log.info("Versión %s de «%s» activada con otras clases que las de ahora", v.number, m.name)
        m.active_version_id, m.updated_at = v.id, utcnow()
        s.commit()
        log.info("Versión %s de «%s» activada por %s", v.number, m.name, user.username)
        return version_dict(v, m)

    @app.post("/api/objects/versions/{vid}/deactivate")
    def deactivate_version(vid: int, user: User = Depends(staff), s=Depends(db)) -> dict:
        v = get_or_404(s, ObjectVersion, vid, "Versión")
        m = s.get(ObjectModel, v.model_id)
        if m.active_version_id == v.id:
            m.active_version_id, m.updated_at = None, utcnow()
            s.commit()
        log.info("Versión %s de «%s» apagada por %s", v.number, m.name, user.username)
        return version_dict(v, m)

    @app.get("/api/objects/versions/{vid}/model")
    def version_model(vid: int, _: User = Depends(staff), s=Depends(db)) -> dict:
        """El .onnx de una versión (link firmado por 1 hora) para probarlo en la PC."""
        v = get_or_404(s, ObjectVersion, vid, "Versión")
        try:
            url = storage.signed_download_url(v.onnx_path, 3600)
        except StorageError as exc:
            raise HTTPException(502, str(exc)) from exc
        return {"url": url, "classes": v.classes or [], "number": v.number, "conf": ob.DEFAULT_CONF}

    # ---------------------------------------------------------------- las máquinas
    def claimed(s, ag: Agent, jid: int) -> ObjectJob:
        j = s.get(ObjectJob, jid)
        if j is None or j.claimed_by != ag.id or j.status != "procesando":
            raise HTTPException(409, "Este trabajo ya no está asignado a esta máquina")
        return j

    def model_brief(m: ObjectModel) -> dict:
        return {"id": m.id, "name": m.name, "classes": m.classes or []}

    def version_files(s, vid: int | None) -> dict | None:
        v = s.get(ObjectVersion, vid) if vid else None
        if v is None or not v.pt_path:
            return None
        return {"id": v.id, "number": v.number, "keys": [c["key"] for c in v.classes or []],
                "pt_url": storage.signed_download_url(v.pt_path, 6 * 3600)}

    def job_payload(s, j: ObjectJob) -> dict:
        m = s.get(ObjectModel, j.model_id)
        out = {"job_id": j.id, "kind": j.kind, "model": model_brief(m), "params": j.params or {}}
        if j.kind == "video":
            out.update(filename=j.filename, download_url=storage.signed_download_url(j.storage_path, 6 * 3600),
                       camera_id=j.camera_id, prelabel=prelabel_spec(s, m))
        elif j.kind == "premarcar":
            spec = dict(j.params or {})
            out["prelabel"] = spec
        if j.kind == "entrenar" and (j.params or {}).get("base") == "version":
            out["base"] = version_files(s, j.params.get("base_version_id"))
        p = out.get("prelabel")
        if p and p.get("method") == "version":
            vf = version_files(s, p.get("version_id"))
            out["prelabel"] = {**p, **(vf or {})} if vf else None
        return out

    @app.get("/api/agent/objects/next")
    def agent_next_object_job(ag: Agent = Depends(current_agent), s=Depends(db)):
        """El próximo trabajo de objetos que puede hacer esta máquina (204 si no hay). Entrenar va primero a las
        máquinas con placa de video: una sin placa lo toma solo si no hay ninguna con placa en línea."""
        if (ag.kind or "pc") != "servidor":
            return Response(status_code=204)
        me = s.get(Agent, ag.id)
        kinds = [k for k in ob.JOB_KINDS if capable(me, k)]
        if not kinds:
            return Response(status_code=204)
        gpu_here = bool(((me.last_status or {}).get("objects") or {}).get("gpu"))
        gpu_elsewhere = any(capable(a, "entrenar") and ((a.last_status or {}).get("objects") or {}).get("gpu")
                            for a in online_machines(s) if a.id != me.id)
        if "entrenar" in kinds and not gpu_here and gpu_elsewhere:
            kinds.remove("entrenar")
        stale = utcnow() - STALE_JOB
        cands = (s.query(ObjectJob).filter(ObjectJob.kind.in_(kinds),
                                           or_(ObjectJob.status == "en_cola",
                                               (ObjectJob.status == "procesando") & (ObjectJob.claimed_at < stale)))
                 .order_by(ObjectJob.id).limit(5).all())
        for j in cands:
            prev_status, prev_claim = j.status, j.claimed_at
            cond = [ObjectJob.id == j.id, ObjectJob.status == prev_status]
            cond.append(ObjectJob.claimed_at == prev_claim if prev_claim else ObjectJob.claimed_at.is_(None))
            res = s.execute(update(ObjectJob).where(*cond).values(
                status="procesando", claimed_by=me.id, claimed_at=utcnow(), started_at=utcnow(), progress=0.0,
                error=None, message="Empezando"))
            s.commit()
            if res.rowcount != 1:
                continue  # otra máquina lo tomó primero
            s.refresh(j)
            try:
                payload = job_payload(s, j)
            except StorageError as exc:
                j.status, j.error = "en_cola", None
                s.commit()
                raise HTTPException(502, str(exc)) from exc
            log.info("Trabajo de objetos %s (%s) para la máquina %s", j.id, j.kind, me.name)
            return payload
        return Response(status_code=204)

    @app.post("/api/agent/objects/jobs/{jid}/progress")
    def agent_job_progress(jid: int, body: ProgressIn, ag: Agent = Depends(current_agent), s=Depends(db)) -> dict:
        """Cuánto va. Responde `cancel: true` si una persona lo canceló (la máquina corta)."""
        j = s.get(ObjectJob, jid)
        if j is None or j.claimed_by != ag.id:
            return {"cancel": True}
        if j.status != "procesando":
            return {"cancel": True}
        j.progress, j.claimed_at = round(body.progress, 3), utcnow()
        if body.message:
            j.message = body.message[:300]
        s.commit()
        return {"cancel": False}

    @app.post("/api/agent/objects/jobs/{jid}/fail")
    def agent_job_fail(jid: int, body: FailIn, ag: Agent = Depends(current_agent), s=Depends(db)) -> dict:
        j = claimed(s, ag, jid)
        j.status, j.error, j.finished_at = "error", body.error[:500], utcnow()
        s.commit()
        log.warning("Trabajo de objetos %s (%s) falló en %s: %s", jid, j.kind, ag.name, body.error[:200])
        return {"ok": True}

    @app.post("/api/agent/objects/jobs/{jid}/images")
    def agent_add_images(jid: int, body: PhotosIn, ag: Agent = Depends(current_agent), s=Depends(db)) -> dict:
        """Las fotos que la máquina sacó del video: igual que las de la app (`upload_url`)."""
        j = claimed(s, ag, jid)
        if j.kind != "video":
            bad("Este trabajo no saca fotos", 409)
        m = s.get(ObjectModel, j.model_id)
        for p in body.images:
            p.source, p.camera_id = "video", j.camera_id
            p.name = p.name or j.filename
        return {"items": register_photos(s, m, body.images, j.created_by, job=j)}

    @app.post("/api/agent/objects/jobs/{jid}/images/uploaded")
    def agent_images_uploaded(jid: int, body: IdsIn, ag: Agent = Depends(current_agent), s=Depends(db)) -> dict:
        j = claimed(s, ag, jid)
        return {"ok": confirm_photos(s, s.get(ObjectModel, j.model_id), body.ids)}

    @app.get("/api/agent/objects/jobs/{jid}/todo")
    def agent_job_todo(jid: int, ag: Agent = Depends(current_agent), s=Depends(db)) -> dict:
        """Pre-marcar: las fotos sin marcar (con su link) para que la IA proponga recuadros."""
        j = claimed(s, ag, jid)
        q = s.query(ObjectImage).filter(ObjectImage.model_id == j.model_id, ObjectImage.status == "sin_marcar")
        if j.kind == "video":
            q = q.filter(ObjectImage.job_id == j.id)
        rows = q.order_by(ObjectImage.id).limit(3000).all()
        try:
            urls = storage.signed_download_urls([i.path for i in rows], 6 * 3600)
        except StorageError as exc:
            raise HTTPException(502, str(exc)) from exc
        return {"items": [{"id": i.id, "url": urls.get(i.path), "width": i.width, "height": i.height}
                          for i in rows if urls.get(i.path)]}

    @app.post("/api/agent/objects/jobs/{jid}/proposals")
    def agent_proposals(jid: int, body: ProposalsIn, ag: Agent = Depends(current_agent), s=Depends(db)) -> dict:
        """Lo que propone la IA para cada foto (`items: [{id, boxes}]`). No toca lo que ya marcó una persona."""
        j = claimed(s, ag, jid)
        m = s.get(ObjectModel, j.model_id)
        keys = keys_of(m)
        by_id = {it.get("id"): it for it in body.items if type(it.get("id")) is int}
        n = 0
        for i in s.query(ObjectImage).filter(ObjectImage.model_id == m.id, ObjectImage.id.in_(list(by_id))):
            if i.status == "marcada":
                continue
            boxes = data_error(ob.clean_boxes, by_id[i.id].get("boxes") or [], keys, with_conf=True)
            i.proposed, i.proposed_by, i.doubt = boxes, body.by[:80], ob.doubt_of(boxes)
            n += 1
        s.commit()
        return {"ok": n}

    @app.get("/api/agent/objects/jobs/{jid}/dataset")
    def agent_dataset(jid: int, ag: Agent = Depends(current_agent), s=Depends(db)) -> dict:
        """Entrenar: las fotos marcadas con sus recuadros, si son de prueba o de entrenar, y el link de cada una."""
        j = claimed(s, ag, jid)
        if j.kind != "entrenar":
            bad("Este trabajo no entrena", 409)
        m = s.get(ObjectModel, j.model_id)
        rows = s.query(ObjectImage).filter(ObjectImage.model_id == m.id, ObjectImage.status == "marcada").all()
        try:
            urls = storage.signed_download_urls([i.path for i in rows], 6 * 3600)
        except StorageError as exc:
            raise HTTPException(502, str(exc)) from exc
        return {"classes": m.classes or [], "items": [
            {"id": i.id, "uuid": i.uuid, "url": urls.get(i.path), "split": i.split, "boxes": i.boxes or []}
            for i in rows if urls.get(i.path)]}

    @app.post("/api/agent/objects/jobs/{jid}/files")
    def agent_version_files(jid: int, ag: Agent = Depends(current_agent), s=Depends(db)) -> dict:
        """Dónde subir el modelo entrenado (.onnx y .pt) antes de avisar que terminó."""
        j = claimed(s, ag, jid)
        if j.kind != "entrenar":
            bad("Este trabajo no entrena", 409)
        base = f"training/objects/{j.model_id}/versiones/job{j.id}"
        try:
            return {"onnx_url": storage.signed_upload_url(base + ".onnx"),
                    "pt_url": storage.signed_upload_url(base + ".pt")}
        except StorageError as exc:
            raise HTTPException(502, str(exc)) from exc

    @app.post("/api/agent/objects/jobs/{jid}/done")
    def agent_job_done(jid: int, body: DoneIn, ag: Agent = Depends(current_agent), s=Depends(db)) -> dict:
        j = claimed(s, ag, jid)
        m = s.get(ObjectModel, j.model_id)
        res = body.result if isinstance(body.result, dict) else {}
        out: dict = {"ok": True}
        if j.kind == "entrenar":
            base = f"training/objects/{j.model_id}/versiones/job{j.id}"
            try:
                if not storage.exists(base + ".onnx"):
                    bad("El modelo entrenado (.onnx) no llegó a la nube: subilo antes de avisar", 409)
                has_pt = storage.exists(base + ".pt")
            except StorageError as exc:
                raise HTTPException(502, str(exc)) from exc
            metrics = clean_metrics(res.get("metrics"), keys_of(m))
            number = (s.query(func.max(ObjectVersion.number)).filter(ObjectVersion.model_id == m.id).scalar() or 0) + 1
            v = ObjectVersion(model_id=m.id, number=number, job_id=j.id, classes=m.classes or [], metrics=metrics,
                              n_images=int(res.get("n_images") or 0), n_boxes=int(res.get("n_boxes") or 0),
                              onnx_path=base + ".onnx", pt_path=base + ".pt" if has_pt else None,
                              trained_on=str(res.get("device") or ag.name)[:120], created_by=j.created_by)
            s.add(v)
            s.flush()
            out["version_id"] = v.id
            res = {"version_id": v.id, "number": number, "map50": metrics.get("map50"),
                   "epochs_run": res.get("epochs_run"), "minutes": res.get("minutes")}
            log.info("Objeto «%s»: versión %s entrenada en %s (mAP50 %s)", m.name, number, ag.name,
                     metrics.get("map50"))
        elif j.kind == "video":
            res = {k: res.get(k) for k in ("images", "skipped", "duration_s", "proposed") if k in res}
            if j.storage_path:  # el video ya no hace falta: quedan las fotos
                remove_files([j.storage_path])
                j.storage_path = None
        else:
            res = {k: res.get(k) for k in ("images", "proposed") if k in res}
        j.status, j.progress, j.result, j.finished_at = "listo", 1.0, res, utcnow()
        j.message = None
        m.updated_at = utcnow()
        s.commit()
        return out

    def clean_metrics(raw, keys: list[str]) -> dict:
        def num(v):
            return round(float(v), 4) if isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v <= 1 \
                else None
        raw = raw if isinstance(raw, dict) else {}
        out = {k: num(raw.get(k)) for k in ("map50", "map50_95", "precision", "recall")}
        n = raw.get("test_images")
        out["test_images"] = n if type(n) is int and 0 <= n < 10**6 else None
        per = raw.get("per_class") if isinstance(raw.get("per_class"), dict) else {}
        out["per_class"] = {k: {x: num((per.get(k) or {}).get(x)) for x in ("precision", "recall", "map50")}
                            for k in keys if isinstance(per.get(k), dict)}
        return out
