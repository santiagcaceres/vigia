from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import unicodedata
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

# scrypt con lo que pide OWASP (Password Storage Cheat Sheet): N=2^17, r=8, p=1 (~128 MB y unas décimas de segundo por
# clave). Las claves guardadas antes (N=2^14, formato de 3 partes) se siguen aceptando y se rehacen solas al entrar
# (`needs_rehash`). VIGIA_SCRYPT_LOG2N lo baja solo para las pruebas (que crean cientos de usuarios).
SCRYPT_LOG2N = int(os.getenv("VIGIA_SCRYPT_LOG2N", "17"))
SCRYPT_R, SCRYPT_P = 8, 1
LEGACY_LOG2N = 14


def _scrypt(password: str, salt: bytes, log2n: int, r: int, p: int, dklen: int = 32) -> bytes:
    n = 2 ** log2n
    return hashlib.scrypt(password.encode(), salt=salt, n=n, r=r, p=p, dklen=dklen, maxmem=256 * n * r + 2 ** 24)


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    dk = _scrypt(password, salt, SCRYPT_LOG2N, SCRYPT_R, SCRYPT_P)
    return (f"scrypt${SCRYPT_LOG2N}${SCRYPT_R}${SCRYPT_P}$"
            f"{base64.b64encode(salt).decode()}${base64.b64encode(dk).decode()}")


def _parse(stored: str) -> tuple[int, int, int, bytes, bytes] | None:
    parts = (stored or "").split("$")
    try:
        if len(parts) == 3 and parts[0] == "scrypt":  # formato viejo: N=2^14, r=8, p=1
            return LEGACY_LOG2N, 8, 1, base64.b64decode(parts[1]), base64.b64decode(parts[2])
        if len(parts) == 6 and parts[0] == "scrypt":
            log2n, r, p = int(parts[1]), int(parts[2]), int(parts[3])
            if not (10 <= log2n <= 20 and 1 <= r <= 16 and 1 <= p <= 4):
                return None
            return log2n, r, p, base64.b64decode(parts[4]), base64.b64decode(parts[5])
    except ValueError:
        return None
    return None


def verify_password(password: str, stored: str) -> bool:
    parsed = _parse(stored)
    if parsed is None:
        return False
    log2n, r, p, salt, dk = parsed
    test = _scrypt(password, salt, log2n, r, p, dklen=len(dk))
    return hmac.compare_digest(test, dk)


def needs_rehash(stored: str) -> bool:
    """La clave guardada es de antes (más barata de adivinar que lo de ahora): se rehace al entrar."""
    parsed = _parse(stored)
    return (parsed is None or stored.count("$") != 5
            or (parsed[0], parsed[1], parsed[2]) != (SCRYPT_LOG2N, SCRYPT_R, SCRYPT_P))


# Una clave que no protege nada: las más usadas (listas públicas de filtraciones, también en castellano) y variantes del
# nombre de la empresa. Se compara sin mayúsculas, tildes ni espacios.
COMMON_PASSWORDS = {
    "12345678", "123456789", "1234567890", "123123123", "11111111", "00000000", "87654321", "12341234", "123qweasd",
    "password", "password1", "password123", "passw0rd", "qwerty123", "qwertyui", "qwerty12", "1q2w3e4r", "1qaz2wsx",
    "abc12345", "abcd1234", "iloveyou", "sunshine", "princess", "football", "baseball", "welcome1", "letmein1",
    "admin123", "admin1234", "administrador", "contrasena", "contrasena1", "contrasena123", "clave123", "clave1234",
    "micontrasena", "uruguay1", "uruguay123", "montevideo", "argentina", "teamo123", "analiticacritica", "analitica",
    "analitica123", "vigia123", "vigia1234", "camaras123", "hikvision", "hikvision123", "dahua123", "seguridad",
    "seguridad123", "cambiame", "cambiar123",
}


def _plain(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode().lower()
    return "".join(ch for ch in text if not ch.isspace())


def password_problem(password: str, username: str | None = None) -> str | None:
    """Por qué una clave nueva no sirve (None si sirve): muy corta o muy larga, igual o parecida al usuario, un solo
    carácter repetido o una de las más usadas."""
    if len(password) < 8:
        return "La clave tiene que tener al menos 8 caracteres"
    if len(password) > 200:
        return "La clave puede tener hasta 200 caracteres"
    plain, user = _plain(password), _plain(username or "")
    if len(set(plain)) <= 2:
        return "La clave no puede ser un mismo carácter repetido"
    if user and (plain == user or user in plain and len(plain) - len(user) < 4):
        return "La clave no puede ser el nombre de usuario (o casi)"
    if plain in COMMON_PASSWORDS:
        return "Esa clave es de las más usadas: elegí otra (una frase de 3 o 4 palabras es fácil de recordar)"
    return None


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
