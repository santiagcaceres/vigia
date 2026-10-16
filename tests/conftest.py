import os
import uuid

import pytest

PG = os.getenv("VIGIA_TEST_PG")  # ej. postgresql://postgres@127.0.0.1:54329/postgres


@pytest.fixture(params=["sqlite", "postgres"])
def db_url(request, tmp_path):
    """Las pruebas del servidor corren con SQLite y, si está disponible, con PostgreSQL (como Supabase)."""
    if request.param == "sqlite":
        yield f"sqlite:///{tmp_path / 'test.db'}"
        return
    if not PG:
        pytest.skip("VIGIA_TEST_PG no definido")
    import psycopg

    name = "t_" + uuid.uuid4().hex[:12]
    with psycopg.connect(PG, autocommit=True) as c:
        c.execute(f"CREATE DATABASE {name}")
    base = PG.rsplit("/", 1)[0]
    yield f"{base}/{name}"
    with psycopg.connect(PG, autocommit=True) as c:
        c.execute(f"DROP DATABASE IF EXISTS {name} WITH (FORCE)")
