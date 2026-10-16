"""Almacenamiento de fotos, clips y videos.

- SupabaseStorage: producción. Bucket privado; el navegador y los agentes suben y bajan
  archivos directo a Supabase con URLs firmadas (así no pasan por Vercel, que limita el
  tamaño de cada pedido a 4,5 MB).
- LocalStorage: desarrollo y pruebas. Misma interfaz, con URLs firmadas que atiende la
  propia API en /api/_files.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
from pathlib import Path
from urllib.parse import quote, urlencode

import httpx

log = logging.getLogger("vigia.storage")


class StorageError(RuntimeError):
    pass


class Storage:
    def put(self, path: str, data: bytes, content_type: str) -> None: ...
    def signed_download_url(self, path: str, expires_in: int = 3600) -> str: ...
    def signed_upload_url(self, path: str) -> str: ...
    def exists(self, path: str) -> bool: ...
    def remove(self, paths: list[str]) -> None: ...


class SupabaseStorage(Storage):
    def __init__(self, supabase_url: str, key: str, bucket: str = "vigia", timeout: float = 30.0,
                 transport: httpx.BaseTransport | None = None):
        self.base = supabase_url.rstrip("/") + "/storage/v1"
        self.bucket = bucket
        headers = {"apikey": key}
        if key.startswith("eyJ"):  # clave service_role clásica (JWT)
            headers["Authorization"] = f"Bearer {key}"
        self.http = httpx.Client(headers=headers, timeout=timeout, transport=transport)

    def _p(self, path: str) -> str:
        return quote(path.lstrip("/"), safe="/")

    def _check(self, r: httpx.Response, what: str) -> dict:
        if r.status_code >= 400:
            raise StorageError(f"Supabase Storage ({what}) respondió {r.status_code}: {r.text[:200]}")
        try:
            return r.json()
        except ValueError:
            return {}

    def ensure_bucket(self, file_size_limit: int | None = None) -> None:
        body = {"id": self.bucket, "name": self.bucket, "public": False}
        if file_size_limit:
            body["file_size_limit"] = file_size_limit
        r = self.http.post(f"{self.base}/bucket", json=body)
        if r.status_code >= 400 and "already exists" not in r.text.lower() and r.status_code != 409:
            self._check(r, "crear bucket")

    def put(self, path: str, data: bytes, content_type: str) -> None:
        r = self.http.post(f"{self.base}/object/{self.bucket}/{self._p(path)}", content=data,
                           headers={"Content-Type": content_type, "x-upsert": "true"})
        self._check(r, "subir")

    def signed_download_url(self, path: str, expires_in: int = 3600) -> str:
        r = self.http.post(f"{self.base}/object/sign/{self.bucket}/{self._p(path)}", json={"expiresIn": expires_in})
        data = self._check(r, "firmar descarga")
        signed = data.get("signedURL") or data.get("signedUrl")
        if not signed:
            raise StorageError("Supabase no devolvió signedURL")
        return self.base + signed

    def signed_upload_url(self, path: str) -> str:
        r = self.http.post(f"{self.base}/object/upload/sign/{self.bucket}/{self._p(path)}",
                           headers={"x-upsert": "true"})
        data = self._check(r, "firmar subida")
        if not data.get("url"):
            raise StorageError("Supabase no devolvió url de subida")
        return self.base + data["url"]

    def exists(self, path: str) -> bool:
        r = self.http.head(f"{self.base}/object/{self.bucket}/{self._p(path)}")
        return r.status_code == 200

    def remove(self, paths: list[str]) -> None:
        paths = [p for p in paths if p]
        for i in range(0, len(paths), 500):
            r = self.http.request("DELETE", f"{self.base}/object/{self.bucket}", json={"prefixes": paths[i:i + 500]})
            if r.status_code >= 400:
                log.warning("No se pudieron borrar %d archivos: %s", len(paths[i:i + 500]), r.text[:200])


class LocalStorage(Storage):
    """Guarda en disco. Las URLs firmadas apuntan a /api/_files y expiran."""

    def __init__(self, root: Path, secret: str, base_url: str = ""):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.secret = secret.encode()
        self.base_url = base_url.rstrip("/")

    def file(self, path: str) -> Path:
        p = (self.root / path).resolve()
        if not p.is_relative_to(self.root.resolve()):
            raise StorageError("Ruta inválida")
        return p

    def sign(self, action: str, path: str, exp: int) -> str:
        return hmac.new(self.secret, f"{action}:{path}:{exp}".encode(), hashlib.sha256).hexdigest()

    def verify(self, action: str, path: str, exp: int, sig: str) -> bool:
        return exp >= time.time() and hmac.compare_digest(self.sign(action, path, exp), sig)

    def _url(self, action: str, path: str, expires_in: int) -> str:
        exp = int(time.time()) + expires_in
        q = urlencode({"exp": exp, "sig": self.sign(action, path, exp)})
        prefix = "/api/_files/" if action == "get" else "/api/_files/upload/"
        return f"{self.base_url}{prefix}{quote(path, safe='/')}?{q}"

    def put(self, path: str, data: bytes, content_type: str) -> None:
        f = self.file(path)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(data)

    def signed_download_url(self, path: str, expires_in: int = 3600) -> str:
        return self._url("get", path, expires_in)

    def signed_upload_url(self, path: str) -> str:
        return self._url("put", path, 7200)

    def exists(self, path: str) -> bool:
        return self.file(path).exists()

    def remove(self, paths: list[str]) -> None:
        for p in paths:
            if p:
                self.file(p).unlink(missing_ok=True)
