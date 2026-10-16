import json

import httpx
import numpy as np
import pytest

from server import training
from server.storage import LocalStorage, StorageError, SupabaseStorage


def fake_supabase(log):
    def handler(req: httpx.Request) -> httpx.Response:
        log.append(req)
        p = req.url.path
        if p == "/storage/v1/bucket":
            return httpx.Response(400, json={"statusCode": "409", "error": "Duplicate", "message": "The resource already exists"})
        if p.startswith("/storage/v1/object/upload/sign/"):
            return httpx.Response(200, json={"url": p[len("/storage/v1"):] + "?token=TKN"})
        if p.startswith("/storage/v1/object/sign/"):
            return httpx.Response(200, json={"signedURL": p[len("/storage/v1"):] + "?token=DL"})
        if req.method == "HEAD":
            return httpx.Response(200 if "existe" in p else 400)
        if req.method == "POST" and p.startswith("/storage/v1/object/vigia/"):
            return httpx.Response(200, json={"Key": p})
        if req.method == "DELETE":
            return httpx.Response(200, json=[])
        return httpx.Response(500, text="inesperado " + p)
    return httpx.MockTransport(handler)


def test_supabase_storage_usa_los_endpoints_correctos():
    log = []
    st = SupabaseStorage("https://abc.supabase.co/", "sb_secret_xyz", transport=fake_supabase(log))
    st.ensure_bucket()  # "ya existe" no es error
    st.put("c1/cam2/u 1.jpg", b"jpg", "image/jpeg")
    up = st.signed_upload_url("imports/5/video.mp4")
    dl = st.signed_download_url("c1/cam2/x.mp4", 600)
    assert st.exists("c1/existe.mp4") and not st.exists("c1/no.mp4")
    st.remove(["a.jpg", "", "b.mp4"])
    assert up == "https://abc.supabase.co/storage/v1/object/upload/sign/vigia/imports/5/video.mp4?token=TKN"
    assert dl == "https://abc.supabase.co/storage/v1/object/sign/vigia/c1/cam2/x.mp4?token=DL"
    put = log[1]
    assert put.method == "POST" and put.url.raw_path == b"/storage/v1/object/vigia/c1/cam2/u%201.jpg"
    assert put.headers["content-type"] == "image/jpeg" and put.headers["x-upsert"] == "true"
    assert all(r.headers["apikey"] == "sb_secret_xyz" for r in log)
    assert "authorization" not in log[1].headers  # las claves nuevas no son JWT
    assert json.loads(log[3].content) == {"expiresIn": 600}
    assert json.loads(log[-1].content) == {"prefixes": ["a.jpg", "b.mp4"]}


def test_supabase_storage_clave_jwt_y_errores():
    log = []
    st = SupabaseStorage("https://abc.supabase.co", "eyJhbGciOi.test", transport=fake_supabase(log))
    st.put("x.jpg", b"1", "image/jpeg")
    assert log[0].headers["authorization"] == "Bearer eyJhbGciOi.test"
    bad = SupabaseStorage("https://abc.supabase.co", "k", transport=httpx.MockTransport(
        lambda r: httpx.Response(403, json={"message": "invalid signature"})))
    with pytest.raises(StorageError, match="403"):
        bad.signed_download_url("x")


def test_local_storage_firmas(tmp_path):
    st = LocalStorage(tmp_path, "secreto")
    url = st.signed_download_url("a/b.jpg", 60)
    q = dict(x.split("=") for x in url.split("?")[1].split("&"))
    assert st.verify("get", "a/b.jpg", int(q["exp"]), q["sig"])
    assert not st.verify("get", "a/otro.jpg", int(q["exp"]), q["sig"])
    assert not st.verify("put", "a/b.jpg", int(q["exp"]), q["sig"])
    assert not st.verify("get", "a/b.jpg", 1, st.sign("get", "a/b.jpg", 1))  # vencida
    with pytest.raises(StorageError):
        st.file("../fuera.txt")


def test_regresion_logistica_coincide_con_scikit_learn():
    sk = pytest.importorskip("sklearn.linear_model")
    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 6))
    y = (X[:, 0] - 0.7 * X[:, 2] + rng.normal(scale=0.8, size=200) > 0.4).astype(int)
    coef, b = training.fit_logistic(X, y, C=0.5)
    ref = sk.LogisticRegression(C=0.5, class_weight="balanced", max_iter=5000, tol=1e-10).fit(X, y)
    assert np.allclose(coef, ref.coef_[0], atol=1e-4)
    assert abs(b - ref.intercept_[0]) < 1e-4


def test_validacion_por_camara_cuando_hay_varias():
    rng = np.random.default_rng(0)
    events = []
    for i in range(80):
        real = i % 2 == 0
        events.append({"type": "salida_sin_pago", "label": "hurto" if real else "falso_positivo", "camera_id": i % 4,
                       "features": {"shelf_total_s": float(rng.normal(8 if real else 2, 2))}})
    spec, m = training.train(events)
    assert m["validation"] == "por_camara" and m["cameras"] == 4
    assert m["recall"] >= 0.9 and len(spec["coef"]) == len(spec["mean"])
