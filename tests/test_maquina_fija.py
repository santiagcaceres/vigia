"""AWS con Tigo (9 nov): una máquina fija siempre prendida (la de la reserva o «Savings Plan») y una segunda que
nuestro sistema prende y detiene según el horario de los locales. La fija no se vacía ni se apaga; el encendedor (que
corre en la fija) la descuenta y maneja solo la segunda. Y el instalador de las máquinas con placa de video: exige la
placa, mide, respeta el tope y deja andando el encendedor (se prueba con programas de mentira: no instala nada)."""

import json
import os
import shutil
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent.escalador import AwsPool, Instance, decide, decide_asg, tick
from server import pool
from server.main import Settings, create_app

ADMIN = ("admin", "clave-admin-123")
UTC = timezone.utc
RAIZ = Path(__file__).resolve().parents[1]
INSTALAR = RAIZ / "deploy" / "maquina-analisis" / "instalar.sh"
PLANTILLA = RAIZ / "deploy" / "aws" / "tigo-virginia.yaml"


# ---------------------------------------------------------------- el plan
def test_plan_la_fija_se_queda_y_se_vacia_la_otra():
    def m(key, load, fixed=False, online=True):
        return pool.Machine(key, key, [1], online, False, 30, load, fixed=fixed)
    # alcanza con una: se queda la fija aunque la otra esté más cargada (la otra se paga por hora)
    p = pool.plan(20, 20, [m("i-fija", 5, fixed=True), m("i-2", 15)])
    assert p.desired == 1 and p.draining == {"i-fija": False, "i-2": True} and p.stop == []
    p = pool.plan(20, 20, [m("i-fija", 20, fixed=True), m("i-2", 0)])
    assert p.stop == ["i-2"]
    # de noche (nada que mirar): la fija ni se vacía ni figura para apagar
    p = pool.plan(0, 0, [m("i-fija", 0, fixed=True), m("i-2", 0)])
    assert p.desired == 0 and p.draining == {"i-fija": False, "i-2": True} and p.stop == ["i-2"]
    # hacen falta las dos: ninguna se vacía
    p = pool.plan(40, 40, [m("i-fija", 25, fixed=True), m("i-2", 15)])
    assert p.desired == 2 and not any(p.draining.values()) and p.stop == []
    # sin fijas, como siempre: se queda la más cargada
    p = pool.plan(20, 20, [m("i-1", 5), m("i-2", 15)])
    assert p.draining == {"i-1": True, "i-2": False}


# ---------------------------------------------------------------- el encendedor
def test_encendedor_descuenta_la_fija_y_no_la_toca():
    now = datetime(2026, 11, 9, 12, 0, tzinfo=UTC)
    seg = lambda state: [Instance("i-2", state, now)]  # noqa: E731  (lo que ve el encendedor: solo la segunda)
    fija = {"fixed_instances": ["i-fija"], "online_instances": ["i-fija"]}
    # con la fija alcanza: la segunda queda detenida
    assert decide({"desired": 1, "stop": [], **fija}, seg("stopped"), 1, False, now).start == []
    # hacen falta dos: prende la segunda
    assert decide({"desired": 2, "stop": [], **fija}, seg("stopped"), 1, False, now).start == ["i-2"]
    # aunque el plan pida 5, el tope es de las que maneja (1) y no avisa que faltan
    d = decide({"desired": 5, "stop": [], **fija}, seg("running"), 1, False, now)
    assert d.start == [] and d.missing == 0 and d.stop == []
    # ya no hace falta y el servidor la vació: la detiene
    plan = {"desired": 1, "stop": ["i-2"], "fixed_instances": ["i-fija"], "online_instances": ["i-fija", "i-2"]}
    assert decide(plan, seg("running"), 1, False, now).stop == ["i-2"]
    # todavía tiene cámaras (el servidor no la dio para apagar): se espera
    assert decide({**plan, "stop": []}, seg("running"), 1, False, now).stop == []
    # la fija se cayó (no figura en línea): la segunda la cubre
    assert decide({"desired": 1, "stop": []}, seg("stopped"), 1, False, now).start == ["i-2"]
    # si por error la fija tiene la etiqueta del grupo, igual no se toca
    juntas = [Instance("i-fija", "running", now), Instance("i-2", "running", now)]
    d = decide({"desired": 0, "stop": ["i-fija", "i-2"], "fixed_instances": ["i-fija"],
                "online_instances": ["i-fija", "i-2"]}, juntas, 5, False, now)
    assert d.stop == ["i-2"]
    # con grupo de Auto Scaling: también se descuenta
    assert decide_asg({"desired": 3, "fixed_instances": ["i-fija"]}, 0, 5, [], 5, now).desired == 2


class FakeEc2:
    """Lo mínimo de EC2 que usa AwsPool, con la segunda máquina detenida."""

    def __init__(self, state="stopped"):
        self.state, self.calls = state, []

    def get_paginator(self, _name):
        outer = self

        class P:
            def paginate(self, Filters):  # noqa: N803
                outer.calls.append(("describe", {f["Name"]: f["Values"] for f in Filters}))
                return [{"Reservations": [{"Instances": [{
                    "InstanceId": "i-2", "State": {"Name": outer.state},
                    "LaunchTime": datetime(2026, 11, 9, 11, 0, tzinfo=UTC)}]}]}]
        return P()

    def start_instances(self, InstanceIds):  # noqa: N803
        self.calls.append(("start", InstanceIds))
        self.state = "running"

    def stop_instances(self, InstanceIds):  # noqa: N803
        self.calls.append(("stop", InstanceIds))
        self.state = "stopped"


@pytest.fixture
def app(tmp_path, db_url):
    return create_app(Settings(data_dir=tmp_path / "data", database_url=db_url, admin_password=ADMIN[1],
                               serve_public=False))


def test_la_fija_analiza_sola_y_la_segunda_se_prende_cuando_no_alcanza(app):
    with TestClient(app) as a:
        a.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]})
        todo_el_dia = {d: "00:00-24:00" for d in ("lun", "mar", "mie", "jue", "vie", "sab", "dom")}
        cl = a.post("/api/clients", json={"name": "Abierto", "hours": todo_el_dia}).json()
        cam = lambda n: a.post("/api/cameras", json={"client_id": cl["id"], "name": n, "brand": "ip",  # noqa: E731
                                                     "host": "dvr", "password": "x", "profile": "caja"}).json()
        key = a.post("/api/analysis/pool-key").json()["key"]
        m = {}
        for name, inst, extra in (("fija", "i-fija", {"fixed": True}), ("segunda", "i-2", {})):
            r = TestClient(app).post("/api/agent/enroll", json={"key": key, "name": name, "instance_id": inst,
                                                                "capacity": 10})
            m[name] = TestClient(app, headers={"X-Agent-Token": r.json()["token"]})
            m[name].post("/api/agent/status", json={"cameras": [], "capacity": 10, "instance_id": inst, **extra})
        a.put("/api/analysis", json={"mode": "servidores", "capacity": 10})
        cams = [cam(f"C{i}") for i in range(4)]  # 4 cámaras de hurto: entran en una máquina de 10
        http = TestClient(app)
        ec2 = FakeEc2("running")  # (la segunda recién creada: prendida y dada de alta)
        nube = AwsPool("us-east-1", "analisis", None, ec2=ec2)

        def revisar():
            return tick("http://testserver", key, nube, 1, http)

        revisar()  # el servidor marca la segunda para vaciar y le saca las cámaras
        plan = http.get("/api/agent/scale", headers={"X-Pool-Key": key}).json()
        assert plan["plan"]["desired"] == 1 and plan["fixed_instances"] == ["i-fija"]
        by = {x["instance_id"]: x for x in plan["machines"]}
        assert by["i-fija"]["fixed"] and not by["i-fija"]["draining"] and by["i-fija"]["cameras"] == 4
        assert by["i-2"]["draining"] and by["i-2"]["cameras"] == 0 and plan["plan"]["stop"] == ["i-2"]
        assert len(m["fija"].get("/api/agent/config").json()["cameras"]) == 4
        revisar()
        assert ("stop", ["i-2"]) in ec2.calls and ("start", ["i-2"]) not in ec2.calls
        assert ec2.calls[0] == ("describe", {"tag:vigia-grupo": ["analisis"],
                                             "instance-state-name": ["pending", "running", "stopping", "stopped"]})

        # abren más locales: 12 cámaras no entran en la fija → prende la segunda, que recibe las que no entran
        cams += [cam(f"D{i}") for i in range(8)]
        ec2.calls.clear()
        revisar()
        assert not any(c[0] == "start" for c in ec2.calls)  # (una revisión sola no alcanza: puede ser un momento)
        revisar()
        assert ("start", ["i-2"]) in ec2.calls
        m["segunda"].post("/api/agent/status", json={"cameras": [], "capacity": 10, "instance_id": "i-2",
                                                     "fresh": True})
        m["fija"].post("/api/agent/status", json={"cameras": [], "capacity": 10, "instance_id": "i-fija",
                                                  "fixed": True})
        plan = http.get("/api/agent/scale", headers={"X-Pool-Key": key}).json()
        assert plan["plan"]["desired"] == 2 and not any(x["draining"] for x in plan["machines"])
        n1 = len(m["fija"].get("/api/agent/config").json()["cameras"])
        n2 = len(m["segunda"].get("/api/agent/config").json()["cameras"])
        # la fija se queda con las suyas (sin pasarse de las 10 que aguanta) y la segunda recibe las que no entran
        assert n1 + n2 == 12 and 8 <= n1 <= 10 and n2 >= 2

        # de noche (nada que analizar): la fija queda prendida y sin vaciar; la segunda se detiene
        a.post("/api/cameras/analyze", json={"camera_ids": [c["id"] for c in cams], "analyze": False})
        ec2.calls.clear()
        revisar()
        revisar()
        plan = http.get("/api/agent/scale", headers={"X-Pool-Key": key}).json()
        by = {x["instance_id"]: x for x in plan["machines"]}
        assert plan["plan"]["desired"] == 0 and not by["i-fija"]["draining"] and by["i-2"]["draining"]
        assert ("stop", ["i-2"]) in ec2.calls and not any(c[0] == "stop" and "i-fija" in c[1] for c in ec2.calls)


def test_si_la_fija_se_cae_la_otra_maquina_recibe_las_camaras_sin_esperar_al_encendedor(app):
    """La mudanza desde Google Cloud: la máquina de antes queda prendida y vacía (la fija se lleva todo). Si la
    fija se apaga o se cae, y con ella el encendedor, las cámaras vuelven solas a la de antes."""
    with TestClient(app) as a:
        a.post("/api/login", json={"username": ADMIN[0], "password": ADMIN[1]})
        todo_el_dia = {d: "00:00-24:00" for d in ("lun", "mar", "mie", "jue", "vie", "sab", "dom")}
        cl = a.post("/api/clients", json={"name": "Abierto", "hours": todo_el_dia}).json()
        key = a.post("/api/analysis/pool-key").json()["key"]
        m = {}
        for name, inst, extra in (("google", "vigia-analisis-1", {}), ("fija", "i-fija", {"fixed": True})):
            r = TestClient(app).post("/api/agent/enroll", json={"key": key, "name": name, "instance_id": inst,
                                                                "capacity": 10})
            m[name] = TestClient(app, headers={"X-Agent-Token": r.json()["token"]})
            m[name].post("/api/agent/status", json={"cameras": [], "capacity": 10, "instance_id": inst, **extra})
        a.put("/api/analysis", json={"mode": "servidores", "capacity": 10})
        for i in range(3):
            a.post("/api/cameras", json={"client_id": cl["id"], "name": f"C{i}", "brand": "ip", "host": "dvr",
                                         "password": "x", "profile": "caja"})
        cuantas = lambda n: len(m[n].get("/api/agent/config").json()["cameras"])  # noqa: E731
        # el encendedor de la fija pregunta: la fija se queda con todo y la de Google queda vacía, marcada para vaciar
        http = TestClient(app)
        http.get("/api/agent/scale", headers={"X-Pool-Key": key})
        assert (cuantas("fija"), cuantas("google")) == (3, 0)
        # la fija se va (y con ella el encendedor: nadie más pregunta el plan)
        assert m["fija"].post("/api/agent/leaving", json={"reason": "prueba"}).status_code == 200
        assert (cuantas("fija"), cuantas("google")) == (0, 3)
        by = {x["key"]: x for x in a.get("/api/analysis").json()["machines"]}
        assert not by["vigia-analisis-1"]["draining"] and by["vigia-analisis-1"]["cameras"] == 3


def test_la_maquina_avisa_que_es_fija():
    from agent.__main__ import Agent

    ag = Agent.__new__(Agent)
    ag._stats_at, ag._fresh = 0.0, False
    ag.machine = {"capacity": 30, "instance_id": "i-fija", "fixed": True}
    assert ag.machine_status()["fixed"] is True
    ag.machine = {"capacity": 30, "instance_id": "i-2", "fixed": False}
    assert "fixed" not in ag.machine_status()


# ---------------------------------------------------------------- el instalador (con programas de mentira)
def _stub(path: Path, body: str) -> None:
    path.write_text("#!/bin/bash\n" + body + "\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _instalar(tmp: Path, env: dict, *, placa: bool = True, nucleos: int = 4, medir: dict | None = None,
              medir_despues: dict | None = None):
    """Corre instalar.sh con las carpetas del sistema cambiadas por carpetas de prueba y con apt, git, systemctl,
    nvidia-smi y Python de mentira (anotan lo que se les pide). Devuelve (código de salida, salida, carpeta)."""
    if os.name == "nt" or not shutil.which("bash"):
        pytest.skip("hace falta bash")
    t = tmp / "maq"
    binp = t / "bin"
    binp.mkdir(parents=True)
    script = INSTALAR.read_text(encoding="utf-8")
    for real, fake in (("/opt/vigia", f"{t}/opt/vigia"), ("/etc/vigia", f"{t}/etc/vigia"),
                       ("/etc/systemd/system", f"{t}/systemd")):
        script = script.replace(real, fake)
    (t / "instalar.sh").write_text(script, encoding="utf-8")
    (t / "systemd").mkdir()
    log = t / "pedidos.log"
    primera = medir if medir is not None else {"cameras": 143.2, "device": "GPU (CUDA)"}
    (t / "medir.1.json").write_text(json.dumps(primera), encoding="utf-8")
    (t / "medir.2.json").write_text(json.dumps(medir_despues if medir_despues is not None else primera),
                                    encoding="utf-8")
    # (la segunda medición —después de cambiar onnxruntime-gpu— contesta con el otro archivo)
    py = (f'case "$*" in\n  *agent.medir*) echo "aviso de onnxruntime"\n'
          f'    if [ -e {t}/medido ]; then cat {t}/medir.2.json; else cat {t}/medir.1.json; fi; echo; touch {t}/medido ;;\n'
          f'  *) exec {sys.executable} "$@" ;;\nesac')
    for name, body in {
        "apt-get": f'echo "apt-get $*" >> {log}',
        "id": "exit 0", "useradd": "exit 0", "chown": "exit 0", "sleep": "exit 0",
        "nproc": f"echo {nucleos}",
        "systemctl": f'echo "systemctl $*" >> {log}',
        "sudo": 'shift 2; exec "$@"',  # «sudo -u vigia programa …»
        "git": (f'echo "git $*" >> {log}\nfor last; do :; done\nmkdir -p "$last/.git" "$last/deploy"\n'
                f'cp -r {RAIZ}/deploy/maquina-analisis "$last/deploy/"'),
        # «python3 -m venv CARPETA»: deja un python y un pip de mentira en CARPETA/bin
        "python3": (f'for last; do :; done\nmkdir -p "$last/bin"\ncat > "$last/bin/python" <<\'PY\'\n#!/bin/bash\n{py}\nPY\n'
                    f'printf \'#!/bin/bash\\necho "pip $*" >> {log}\\n\' > "$last/bin/pip"\n'
                    'chmod +x "$last/bin/python" "$last/bin/pip"'),
    }.items():
        _stub(binp / name, body)
    if placa:
        _stub(binp / "nvidia-smi", 'echo "Tesla T4, 550.0, 15360 MiB"')
    r = subprocess.run(["bash", str(t / "instalar.sh")], capture_output=True, text=True, timeout=120,
                       env={"PATH": f"{binp}:/usr/bin:/bin", "HOME": str(t), **env})
    return r.returncode, r.stdout + r.stderr, t


def _env(path: Path) -> dict:
    return dict(line.split("=", 1) for line in path.read_text(encoding="utf-8").splitlines() if "=" in line)


def test_instalar_maquina_fija_con_placa_y_encendedor(tmp_path):
    code, out, t = _instalar(tmp_path, {
        "VIGIA_POOL_KEY": "vgm_clave-de-prueba", "VIGIA_GPU": "si", "VIGIA_FIJA": "si", "CAPACIDAD_TOPE": "60",
        "ACTUALIZAR_DIARIO": "si", "ACTUALIZAR_HORA": "06:15", "ENCENDEDOR": "si", "VIGIA_NUBE": "aws",
        "VIGIA_GRUPO": "analisis", "VIGIA_MAX_MAQUINAS": "1", "VIGIA_RESPALDO": "no", "AWS_REGION": "us-east-1"})
    assert code == 0, out
    maq = _env(t / "etc/vigia/maquina.env")
    # midió 143 pero anota el tope (60 por máquina, en 2 procesos) hasta comprobarlo con cámaras de verdad
    assert maq["VIGIA_CAPACIDAD"] == "30" and maq["VIGIA_FIJA"] == "si" and "VIGIA_HILOS" not in maq
    assert "Midió 143; se anota 60" in out and "detector en GPU (CUDA)" in out
    enc = _env(t / "etc/vigia/escalador.env")
    assert (enc["VIGIA_NUBE"], enc["VIGIA_GRUPO"], enc["VIGIA_MAX_MAQUINAS"], enc["AWS_REGION"]) == \
        ("aws", "analisis", "1", "us-east-1")
    assert enc["VIGIA_RESPALDO"] == "no" and enc["VIGIA_PLANTILLA"] == ""  # nunca crea máquinas: prende y detiene
    pedidos = (t / "pedidos.log").read_text(encoding="utf-8")
    for unit in ("vigia-maquina@1", "vigia-maquina@2", "vigia-escalador", "vigia-actualizar-diario.timer"):
        assert f"enable --now {unit}" in pedidos, pedidos
    assert "vigia-maquina@3" not in pedidos and "requirements-servidor.txt" in pedidos
    # si el driver de la placa no carga, el análisis no arranca (no queda analizando con el procesador)
    assert "ExecStartPre=" in (t / "systemd/vigia-maquina@.service.d/placa.conf").read_text(encoding="utf-8")
    assert "OnCalendar=*-*-* 06:15:00 UTC" in (t / "systemd/vigia-actualizar-diario.timer").read_text(encoding="utf-8")


def test_instalar_con_placa_pedida_no_sigue_si_el_detector_va_por_el_procesador(tmp_path):
    code, out, t = _instalar(tmp_path, {"VIGIA_POOL_KEY": "vgm_clave-de-prueba", "VIGIA_GPU": "si"},
                             medir={"cameras": 4.0, "device": "procesador"})
    assert code == 1 and "NO la está usando" in out and "Instalación detenida" in out
    assert not (t / "etc/vigia/maquina.env").exists()
    pedidos = (t / "pedidos.log").read_text(encoding="utf-8")
    assert "vigia-maquina" not in pedidos and "onnxruntime-gpu[cuda,cudnn]==1.22.*" in pedidos  # (antes probó eso)
    # el driver de la imagen es más viejo que el que pide el último onnxruntime-gpu: con el 1.22 anda y sigue
    code, out, t = _instalar(tmp_path / "viejo", {"VIGIA_POOL_KEY": "vgm_clave-de-prueba", "VIGIA_GPU": "si"},
                             medir={"cameras": 4.0, "device": "procesador"},
                             medir_despues={"cameras": 80.0, "device": "GPU (CUDA)"})
    assert code == 0 and _env(t / "etc/vigia/maquina.env")["VIGIA_CAPACIDAD"] == "30", out
    # y si directamente no hay placa (tipo de máquina o imagen equivocados), lo dice y no instala nada
    code, out, t = _instalar(tmp_path / "sin", {"VIGIA_POOL_KEY": "vgm_clave-de-prueba", "VIGIA_GPU": "si"},
                             placa=False)
    assert code == 1 and "nvidia-smi no anda" in out and not (t / "pedidos.log").exists()


def test_instalar_segunda_maquina_grande_y_maquina_sin_placa(tmp_path):
    # g4dn.2xlarge (8 vCPU): 4 procesos; no es fija ni lleva encendedor
    code, out, t = _instalar(tmp_path / "a", {"VIGIA_POOL_KEY": "vgm_clave-de-prueba", "VIGIA_GPU": "si",
                                              "CAPACIDAD_TOPE": "60"}, nucleos=8)
    assert code == 0, out
    maq = _env(t / "etc/vigia/maquina.env")
    assert maq["VIGIA_CAPACIDAD"] == "15" and "VIGIA_FIJA" not in maq
    pedidos = (t / "pedidos.log").read_text(encoding="utf-8")
    assert "enable --now vigia-maquina@4" in pedidos and "vigia-escalador" not in pedidos
    assert not (t / "etc/vigia/escalador.env").exists() and "timer" not in pedidos
    # sin placa (como la de Google Cloud): mide con el procesador, sin tope, un modelo por núcleo
    code, out, t = _instalar(tmp_path / "b", {"VIGIA_POOL_KEY": "vgm_clave-de-prueba"}, placa=False,
                             medir={"cameras": 6.7, "device": "procesador"})
    assert code == 0, out
    maq = _env(t / "etc/vigia/maquina.env")
    assert maq["VIGIA_CAPACIDAD"] == "6" and maq["VIGIA_HILOS"] == "4"
    assert not (t / "systemd/vigia-maquina@.service.d").exists()
    assert "requirements-servidor-cpu.txt" in (t / "pedidos.log").read_text(encoding="utf-8")
    # si no se pudo medir: con placa, 30 por máquina; sin placa, una por núcleo
    code, out, t = _instalar(tmp_path / "c", {"VIGIA_POOL_KEY": "vgm_clave-de-prueba"}, medir={})
    assert code == 0 and _env(t / "etc/vigia/maquina.env")["VIGIA_CAPACIDAD"] == "15", out
    # solo el encendedor (máquina chica): no instala el análisis
    code, out, t = _instalar(tmp_path / "d", {"VIGIA_POOL_KEY": "vgm_clave-de-prueba", "MODO": "escalador"},
                             placa=False)
    assert code == 0 and (t / "etc/vigia/escalador.env").exists() and not (t / "etc/vigia/maquina.env").exists()


# ---------------------------------------------------------------- la plantilla de AWS
def test_plantilla_de_tigo_coincide_con_el_instalador_y_el_encendedor():
    yaml = pytest.importorskip("yaml")

    class Cfn(yaml.SafeLoader):
        pass

    def tag(loader, suffix, node):  # !Ref, !Sub, !If, !GetAtt…: alcanza con el contenido
        if isinstance(node, yaml.ScalarNode):
            return loader.construct_scalar(node)
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node, deep=True)
        return loader.construct_mapping(node, deep=True)
    Cfn.add_multi_constructor("!", tag)
    tpl = yaml.load(PLANTILLA.read_text(encoding="utf-8"), Loader=Cfn)  # noqa: S506
    res = tpl["Resources"]
    fija, segunda = res["MaquinaFija"]["Properties"], res["MaquinaSegunda"]["Properties"]
    tags = lambda p: {t["Key"]: t["Value"] for t in p["Tags"]}  # noqa: E731
    ud_fija, ud_seg = fija["UserData"]["Fn::Base64"], segunda["UserData"]["Fn::Base64"]
    # el encendedor (en la fija) maneja las máquinas con la etiqueta de la segunda, y solo esas
    assert tags(segunda)["vigia-grupo"] == "analisis" and "VIGIA_GRUPO=analisis" in ud_fija
    assert tags(fija)["vigia-grupo"] != "analisis"
    assert "VIGIA_FIJA=si" in ud_fija and "ENCENDEDOR=si" in ud_fija and "VIGIA_MAX_MAQUINAS=1" in ud_fija
    assert "VIGIA_FIJA" not in ud_seg and "ENCENDEDOR" not in ud_seg
    script = INSTALAR.read_text(encoding="utf-8")
    for ud in (ud_fija, ud_seg):
        assert "VIGIA_GPU=si" in ud and "VIGIA_POOL_KEY_SSM=" in ud
        for var in ("VIGIA_GPU", "VIGIA_FIJA", "CAPACIDAD_TOPE", "ACTUALIZAR_DIARIO", "ACTUALIZAR_HORA", "ENCENDEDOR",
                    "VIGIA_NUBE", "VIGIA_GRUPO", "VIGIA_MAX_MAQUINAS", "VIGIA_RESPALDO", "VIGIA_POOL_KEY_SSM"):
            assert var in script  # (todo lo que pasa la plantilla lo entiende el instalador)
    # el permiso de prender y detener es solo para las máquinas con esa etiqueta
    pol = next(p for p in res["RolFija"]["Properties"]["Policies"] if p["PolicyName"] == "prender-y-apagar-la-segunda")
    st = next(s for s in pol["PolicyDocument"]["Statement"] if "ec2:StartInstances" in s["Action"])
    assert st["Condition"] == {"StringEquals": {"aws:ResourceTag/vigia-grupo": "analisis"}}
    assert "RolSegunda" in res and len(res["RolSegunda"]["Properties"]["Policies"]) == 1
    # sin puertos de entrada, discos cifrados, IP fija en cada máquina, y nada de lo que se sacó de la cotización
    assert "SecurityGroupIngress" not in res["SinEntrada"]["Properties"]
    for p in (fija, segunda):
        assert p["BlockDeviceMappings"][0]["Ebs"]["Encrypted"] is True
        assert p["MetadataOptions"]["HttpTokens"] == "required"
    types = {r["Type"] for r in res.values()}
    assert {"AWS::EC2::EIP", "AWS::EC2::EIPAssociation", "AWS::EC2::InternetGateway"} <= types
    assert not {t for t in types if any(x in t for x in ("NatGateway", "VPN", "LoadBalancing", "S3"))}
    assert tpl["Parameters"]["Imagen"]["Default"].endswith("base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id")
