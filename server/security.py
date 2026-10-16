from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return f"scrypt${base64.b64encode(salt).decode()}${base64.b64encode(dk).decode()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, salt_b64, dk_b64 = stored.split("$")
        salt, dk = base64.b64decode(salt_b64), base64.b64decode(dk_b64)
    except ValueError:
        return False
    test = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=len(dk))
    return hmac.compare_digest(test, dk)


def new_token() -> str:
    return secrets.token_urlsafe(32)


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class SecretBox:
    """Cifra las claves de las cámaras guardadas en la base de datos."""

    def __init__(self, secret: str):
        self.secret = secret
        key = base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest())
        self._f = Fernet(key)

    @classmethod
    def from_env_or_file(cls, data_dir: Path) -> "SecretBox":
        secret = os.getenv("SECRET_KEY")
        if not secret:
            path = Path(data_dir) / "secret.key"
            if path.exists():
                secret = path.read_text().strip()
            else:
                secret = secrets.token_urlsafe(48)
                path.write_text(secret)
                try:
                    path.chmod(0o600)
                except OSError:
                    pass
        return cls(secret)

    def encrypt(self, value: str | None) -> str | None:
        if value is None or value == "":
            return None
        return self._f.encrypt(value.encode()).decode()

    def decrypt(self, value: str | None) -> str:
        if not value:
            return ""
        try:
            return self._f.decrypt(value.encode()).decode()
        except InvalidToken as exc:
            raise RuntimeError("No se pudo descifrar una clave de cámara: ¿cambió SECRET_KEY?") from exc
