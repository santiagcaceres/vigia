"""Máquinas virtuales en Google Cloud: ver cuáles están prendidas y prenderlas, apagarlas o reiniciarlas desde la app.

Se usa una **cuenta de servicio** del proyecto (el archivo .json de su clave). El administrador la elige en la app
(Máquinas → «Conectar Google Cloud…») y el servidor la guarda cifrada (`app_settings.gcp_key_enc`); nunca vuelve a
salir del servidor. Con el rol «Visualizador de Compute» solo se ve el estado; con «Administrador de instancias de
Compute (v1)» también se prende, se apaga y se reinicia.

Sin librerías de Google: el token se pide con un JWT firmado (RS256, `cryptography`) y la API de Compute Engine se
llama con httpx. El token dura una hora y se reutiliza mientras el proceso viva."""

from __future__ import annotations

import base64
import json
import logging
import re
import threading
import time

import httpx

log = logging.getLogger("vigia.nube")

TOKEN_URL = "https://oauth2.googleapis.com/token"
SCOPE = "https://www.googleapis.com/auth/compute"
API = "https://compute.googleapis.com/compute/v1"
ACTIONS = ("start", "stop", "reset", "resume")  # resume: una suspendida se prende así
NAME_RE = re.compile(r"^[a-z]([-a-z0-9]{0,61}[a-z0-9])?$")  # nombres de instancia y zona de Google

_tokens: dict[str, tuple[str, float]] = {}  # cuenta → (token, vence)
_lock = threading.Lock()


class CloudError(Exception):
    """Un error para mostrar tal cual en la app (en castellano)."""

    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


def parse_key(text: str) -> dict:
    """El .json de la clave de una cuenta de servicio. ValueError (en castellano) si no lo es."""
    try:
        key = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise ValueError("El archivo no es un JSON: elegí el .json que bajaste al crear la clave.") from exc
    if not isinstance(key, dict) or key.get("type") != "service_account":
        raise ValueError("No es la clave de una cuenta de servicio (en Google Cloud: IAM y administración → Cuentas de "
                         "servicio → Claves → Agregar clave → JSON).")
    for field in ("client_email", "private_key", "project_id"):
        if not key.get(field):
            raise ValueError(f"A la clave le falta «{field}»: bajala de nuevo desde Google Cloud.")
    return key


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def signed_jwt(key: dict, now: float | None = None) -> str:
    """El pedido de token firmado con la clave privada de la cuenta de servicio (RS256)."""
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding

    now = time.time() if now is None else now
    header = {"alg": "RS256", "typ": "JWT"}
    if key.get("private_key_id"):
        header["kid"] = key["private_key_id"]
    claims = {"iss": key["client_email"], "scope": SCOPE, "aud": TOKEN_URL, "iat": int(now), "exp": int(now) + 3600}
    signing_input = f"{_b64(json.dumps(header).encode())}.{_b64(json.dumps(claims).encode())}".encode()
    try:
        private = serialization.load_pem_private_key(key["private_key"].encode(), password=None)
    except (ValueError, TypeError) as exc:
        raise ValueError("La clave privada del archivo no se puede leer: bajá una clave nueva.") from exc
    signature = private.sign(signing_input, padding.PKCS1v15(), hashes.SHA256())
    return f"{signing_input.decode()}.{_b64(signature)}"


class GoogleCloud:
    """Las instancias de Compute Engine de un proyecto."""

    def __init__(self, key: dict, http: httpx.Client | None = None):
        self.key = key
        self.project = key["project_id"]
        self.account = key["client_email"]
        self.http = http or httpx.Client(timeout=8.0)  # Vercel corta los pedidos largos

    def _token(self) -> str:
        with _lock:
            cached = _tokens.get(self.account)
            if cached and cached[1] > time.time() + 60:
                return cached[0]
        try:
            r = self.http.post(TOKEN_URL, data={"grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                                                "assertion": signed_jwt(self.key)})
        except httpx.HTTPError as exc:
            raise CloudError(f"No se pudo conectar con Google ({exc.__class__.__name__}).") from exc
        if r.status_code != 200:
            detail = _detail(r)
            log.warning("Google rechazó la cuenta de servicio %s: %s %s", self.account, r.status_code, detail)
            raise CloudError("Google no aceptó la clave de la cuenta de servicio (¿la borraron o la desactivaron?): "
                             f"{detail}", 502)
        data = r.json()
        token = data["access_token"]
        with _lock:
            _tokens[self.account] = (token, time.time() + float(data.get("expires_in", 3600)))
        return token

    def _call(self, method: str, url: str) -> dict:
        try:
            r = self.http.request(method, url, headers={"Authorization": f"Bearer {self._token()}"})
        except httpx.HTTPError as exc:
            raise CloudError(f"No se pudo conectar con Google ({exc.__class__.__name__}).") from exc
        if r.status_code == 401:
            with _lock:
                _tokens.pop(self.account, None)
        if r.status_code in (401, 403):
            detail = _detail(r)
            if "has not been used" in detail or "is disabled" in detail:
                raise CloudError("La API de Compute Engine está apagada en el proyecto: prendela en Google Cloud "
                                 "(APIs y servicios → Compute Engine API → Habilitar).", 403)
            raise CloudError(f"La cuenta de servicio {self.account} no tiene permiso para esto en el proyecto "
                             f"{self.project}. Dale el rol «Administrador de instancias de Compute (v1)» (o "
                             "«Visualizador de Compute» si solo querés ver).", 403)
        if r.status_code == 404:
            raise CloudError("Google no encontró esa máquina (¿la borraron?).", 404)
        if r.status_code >= 400:
            raise CloudError(f"Google respondió {r.status_code}: {_detail(r)}", 502)
        return r.json()

    def instances(self) -> list[dict]:
        """Todas las instancias del proyecto, en todas las zonas."""
        url = f"{API}/projects/{self.project}/aggregated/instances?returnPartialSuccess=true"
        out: list[dict] = []
        while url:
            data = self._call("GET", url)
            for scope in (data.get("items") or {}).values():
                for inst in scope.get("instances") or []:
                    out.append(instance_dict(inst))
            token = data.get("nextPageToken")
            url = (f"{API}/projects/{self.project}/aggregated/instances?returnPartialSuccess=true&pageToken={token}"
                   if token else None)
        return sorted(out, key=lambda i: i["name"])

    def action(self, zone: str, name: str, action: str) -> dict:
        """Prender (start; resume si está suspendida), apagar (stop) o reiniciar (reset) una instancia. Google lo
        hace en segundo plano: la respuesta es la operación, no el resultado."""
        if action not in ACTIONS:
            raise CloudError("Acción desconocida.", 422)
        if not NAME_RE.match(zone or "") or not NAME_RE.match(name or ""):
            raise CloudError("Zona o nombre de máquina inválidos.", 422)
        op = self._call("POST", f"{API}/projects/{self.project}/zones/{zone}/instances/{name}/{action}")
        return {"operation": op.get("name"), "status": op.get("status")}


def instance_dict(inst: dict) -> dict:
    """Lo que la app muestra de una instancia (sin IPs ni datos de más)."""
    sched = inst.get("scheduling") or {}
    return {"name": inst.get("name"), "zone": (inst.get("zone") or "").rsplit("/", 1)[-1],
            "status": inst.get("status"), "machine_type": (inst.get("machineType") or "").rsplit("/", 1)[-1],
            "spot": sched.get("provisioningModel") == "SPOT" or bool(sched.get("preemptible")),
            "created": inst.get("creationTimestamp"), "last_start": inst.get("lastStartTimestamp"),
            "last_stop": inst.get("lastStopTimestamp")}


def _detail(r: httpx.Response) -> str:
    try:
        data = r.json()
    except ValueError:
        return r.text[:200]
    err = data.get("error")
    if isinstance(err, dict):
        return str(err.get("message") or err)[:300]
    return str(data.get("error_description") or err or data)[:300]
