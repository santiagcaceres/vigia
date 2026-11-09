#!/usr/bin/env bash
# Instala Vigía en una máquina de análisis: Ubuntu 22.04 o 24.04, x86 o ARM (AWS Graviton).
# - Con placa NVIDIA y su driver (en AWS: «Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)»).
# - Sin placa: se da cuenta solo y analiza con el procesador.
# En los dos casos MIDE cuántas cámaras aguanta (agent/medir.py), salvo que se le diga con CAPACIDAD.
#
#   sudo VIGIA_POOL_KEY=vgm_... bash instalar.sh            # máquina de análisis
#   sudo VIGIA_POOL_KEY=vgm_... PROCESOS=3 CAPACIDAD=12 bash instalar.sh
#   sudo VIGIA_POOL_KEY=vgm_... MODO=escalador bash instalar.sh   # solo el encendedor (máquina chica)
#   sudo VIGIA_POOL_KEY_SSM=/vigia/clave-maquinas bash instalar.sh   # en AWS: la clave sale del Parameter Store
#                                                                    # (así la usan las plantillas de deploy/aws/)
# Otras opciones (las usa deploy/aws/tigo-virginia.yaml):
#   VIGIA_GPU=si          tiene que haber placa de video: si no anda, la instalación falla (y lo dice en el registro)
#                         en vez de quedar analizando con el procesador sin que nadie se entere
#   VIGIA_FIJA=si         máquina fija: siempre prendida (la de la reserva); el servidor no la vacía ni se apaga
#   ENCENDEDOR=si         además de analizar, esta máquina prende y apaga las otras (VIGIA_NUBE, VIGIA_GRUPO…)
#   CAPACIDAD_TOPE=60     no decir que aguanta más que esto (por máquina), mida lo que mida
#   ACTUALIZAR_DIARIO=si  bajar la última versión cada noche (una máquina que no se apaga nunca no se actualiza sola);
#                         ACTUALIZAR_HORA=07:30 es la hora UTC (04:30 en Uruguay)
set -euo pipefail
MODO=${MODO:-maquina}
ENCENDEDOR=${ENCENDEDOR:-no}
EXIGIR_GPU=${VIGIA_GPU:-no}
if [ "$MODO" = "maquina" ] && [ "$EXIGIR_GPU" = "si" ]; then
  # la imagen con el driver a veces tarda un poco en dejarlo listo la primera vez que prende
  for _ in $(seq 1 60); do nvidia-smi >/dev/null 2>&1 && break; sleep 5; done
  if ! nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: se pidió placa de video (VIGIA_GPU=si) y nvidia-smi no anda. ¿La máquina es de un tipo con placa" \
         "(g4dn, g5, g6) y la imagen trae el driver de NVIDIA (Deep Learning Base OSS Nvidia Driver GPU AMI)?" >&2
    exit 1
  fi
fi
if nvidia-smi >/dev/null 2>&1; then
  GPU=si; REQ=requirements-servidor.txt
  # la placa detecta; recibir y decodificar el video lo hace el procesador: un proceso cada 2 núcleos (de 2 a 4)
  NUCLEOS=$(nproc)
  PROCESOS=${PROCESOS:-$(( NUCLEOS >= 8 ? 4 : 2 ))}
  CAPACIDAD_TOPE=${CAPACIDAD_TOPE:-60}   # hasta comprobarlo con cámaras de verdad
else
  GPU=no; REQ=requirements-servidor-cpu.txt
  PROCESOS=${PROCESOS:-1}
  CAPACIDAD_TOPE=${CAPACIDAD_TOPE:-}
fi
CAPACIDAD=${CAPACIDAD:-}     # vacío: se mide al final de la instalación
SERVIDOR=${VIGIA_SERVER:-https://analitica-critica.vercel.app}
REPO=${VIGIA_REPO:-https://github.com/santiagcaceres/vigia.git}
if [ -z "${VIGIA_POOL_KEY:-}" ] && [ -z "${VIGIA_POOL_KEY_SSM:-}" ]; then
  echo "Falta VIGIA_POOL_KEY (se crea en la app: Máquinas → «Más» → «Clave para máquinas nuevas»)" >&2
  exit 1
fi
APT="apt-get -q -y -o DPkg::Lock::Timeout=600"  # al prender una máquina nueva, Ubuntu se actualiza solo un rato

$APT update
$APT install git python3-venv ffmpeg curl
id vigia >/dev/null 2>&1 || useradd --system --create-home --home-dir /var/lib/vigia vigia
[ -d /opt/vigia/.git ] || git clone -q --depth 1 "$REPO" /opt/vigia
chown -R vigia:vigia /opt/vigia
sudo -u vigia python3 -m venv /opt/vigia/.venv
sudo -u vigia /opt/vigia/.venv/bin/pip install -q -U pip
mkdir -p /etc/vigia
PY=/opt/vigia/.venv/bin/python

if [ -z "${VIGIA_POOL_KEY:-}" ]; then
  # AWS: la clave vive en el Parameter Store (SecureString) y la máquina la lee con su rol de IAM
  sudo -u vigia /opt/vigia/.venv/bin/pip install -q boto3
  TOKEN=$(curl -fsS -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
  REGION=${AWS_REGION:-$(curl -fsS -H "X-aws-ec2-metadata-token: $TOKEN" \
    http://169.254.169.254/latest/meta-data/placement/region)}
  VIGIA_POOL_KEY=$(sudo -u vigia "$PY" -c 'import sys, boto3
print(boto3.client("ssm", region_name=sys.argv[1]).get_parameter(Name=sys.argv[2], WithDecryption=True)["Parameter"]["Value"])' \
    "$REGION" "$VIGIA_POOL_KEY_SSM")
  AWS_REGION=$REGION
fi

instalar_encendedor() {
  sudo -u vigia /opt/vigia/.venv/bin/pip install -q httpx boto3
  cat > /etc/vigia/escalador.env <<CFG
VIGIA_SERVER=$SERVIDOR
VIGIA_POOL_KEY=$VIGIA_POOL_KEY
VIGIA_NUBE=${VIGIA_NUBE:-aws}
AWS_REGION=${AWS_REGION:-sa-east-1}
VIGIA_ASG=${VIGIA_ASG:-}
VIGIA_GRUPO=${VIGIA_GRUPO:-analisis}
VIGIA_PLANTILLA=${VIGIA_PLANTILLA:-}
VIGIA_SPOT=${VIGIA_SPOT:-no}
VIGIA_TIPOS=${VIGIA_TIPOS:-}
VIGIA_RESPALDO=${VIGIA_RESPALDO:-si}
VIGIA_MAX_MAQUINAS=${VIGIA_MAX_MAQUINAS:-10}
CFG
  chown root:vigia /etc/vigia/escalador.env && chmod 640 /etc/vigia/escalador.env
  cp /opt/vigia/deploy/maquina-analisis/vigia-escalador.service /etc/systemd/system/
  systemctl daemon-reload
  systemctl enable --now vigia-escalador
  echo "Encendedor andando. Registro: journalctl -u vigia-escalador -f"
}

if [ "$MODO" = "escalador" ]; then
  instalar_encendedor
  exit 0
fi

sudo -u vigia /opt/vigia/.venv/bin/pip install -q -r "/opt/vigia/$REQ"
echo "VIGIA_REQ=$REQ" > /etc/vigia/actualizar.env

# Medir: cuántas cámaras de hurto entran (detector + seguimiento + decodificar, 30 % de margen) y dónde corre el
# detector. Cada tipo de máquina rinde distinto: por eso se mide en cada una.
medir() {
  echo "Midiendo cuántas cámaras aguanta esta máquina (unos 30 segundos)…"
  MEDIR=$(cd /opt/vigia && sudo -u vigia "$PY" -m agent.medir --segundos 12 --json 2>/dev/null | tail -1 || true)
  MEDIDO=$("$PY" -c 'import json, sys; print(max(1, int(json.loads(sys.argv[1])["cameras"])))' "$MEDIR" 2>/dev/null || true)
  DONDE=$("$PY" -c 'import json, sys; print(json.loads(sys.argv[1])["device"])' "$MEDIR" 2>/dev/null || true)
  echo "Medido: ${MEDIDO:-no se pudo} cámaras de hurto · detector en ${DONDE:-no se sabe} · $MEDIR"
}
medir
if [ "$GPU" = "si" ]; then
  case "$DONDE" in
    GPU*) ;;
    *) # lo más común: el driver de NVIDIA de la imagen es más viejo que el que pide el último onnxruntime-gpu
       echo "Hay placa de video pero el detector no la usa (corre en «${DONDE:-?}»): pruebo con onnxruntime-gpu 1.22"
       sudo -u vigia /opt/vigia/.venv/bin/pip install -q "onnxruntime-gpu[cuda,cudnn]==1.22.*" || true
       medir ;;
  esac
  case "$DONDE" in
    GPU*) ;;
    *) echo "ATENCIÓN: hay placa de video pero el detector NO la está usando (corre en «${DONDE:-?}»)." \
            "Revisar el driver de NVIDIA (nvidia-smi) y onnxruntime-gpu (pip list | grep onnxruntime)." >&2
       if [ "$EXIGIR_GPU" = "si" ]; then
         echo "ERROR: con VIGIA_GPU=si la máquina no se pone a analizar con el procesador. Instalación detenida." >&2
         exit 1
       fi ;;
  esac
fi
if [ -z "$CAPACIDAD" ]; then
  case "$MEDIDO" in ''|*[!0-9]*) if [ "$GPU" = "si" ]; then MEDIDO=30; else MEDIDO=$(nproc); fi ;; esac
  if [ -n "$CAPACIDAD_TOPE" ] && [ "$MEDIDO" -gt "$CAPACIDAD_TOPE" ]; then
    echo "Midió $MEDIDO; se anota $CAPACIDAD_TOPE (CAPACIDAD_TOPE) hasta comprobarlo con cámaras de verdad."
    MEDIDO=$CAPACIDAD_TOPE
  fi
  CAPACIDAD=$(( MEDIDO / PROCESOS )); [ "$CAPACIDAD" -ge 1 ] || CAPACIDAD=1
fi
cat > /etc/vigia/maquina.env <<CFG
VIGIA_SERVER=$SERVIDOR
VIGIA_POOL_KEY=$VIGIA_POOL_KEY
VIGIA_CAPACIDAD=$CAPACIDAD
CFG
[ "${VIGIA_FIJA:-no}" = "si" ] && echo "VIGIA_FIJA=si" >> /etc/vigia/maquina.env
# sin placa de video: un modelo por núcleo, repartidos entre los procesos
HILOS=$(( $(nproc) / PROCESOS )); [ "$HILOS" -ge 1 ] || HILOS=1
[ "$GPU" = "no" ] && echo "VIGIA_HILOS=$HILOS" >> /etc/vigia/maquina.env
chown root:vigia /etc/vigia/maquina.env && chmod 640 /etc/vigia/maquina.env
cp /opt/vigia/deploy/maquina-analisis/vigia-actualizar.service /opt/vigia/deploy/maquina-analisis/vigia-maquina@.service \
   /etc/systemd/system/
if [ "$GPU" = "si" ]; then
  # si un día el driver de la placa no carga (por ejemplo, después de una actualización de Ubuntu), el análisis no
  # arranca y la máquina figura «sin conexión» en la app, en vez de analizar con el procesador a la décima parte
  mkdir -p /etc/systemd/system/vigia-maquina@.service.d
  printf '[Service]\nExecStartPre=%s -L\n' "$(command -v nvidia-smi)" \
    > /etc/systemd/system/vigia-maquina@.service.d/placa.conf
fi
if [ "${ACTUALIZAR_DIARIO:-no}" = "si" ]; then
  cp /opt/vigia/deploy/maquina-analisis/vigia-actualizar-diario.service /etc/systemd/system/
  sed "s/^OnCalendar=.*/OnCalendar=*-*-* ${ACTUALIZAR_HORA:-07:30}:00 UTC/" \
    /opt/vigia/deploy/maquina-analisis/vigia-actualizar-diario.timer > /etc/systemd/system/vigia-actualizar-diario.timer
fi
systemctl daemon-reload
systemctl enable vigia-actualizar
[ "${ACTUALIZAR_DIARIO:-no}" = "si" ] && systemctl enable --now vigia-actualizar-diario.timer
for i in $(seq 1 "$PROCESOS"); do systemctl enable --now "vigia-maquina@$i"; done
if [ "$GPU" = "si" ]; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
  echo "Con placa de video: $PROCESOS procesos de $CAPACIDAD cámaras de hurto cada uno."
  echo "Listo. Registro: journalctl -u 'vigia-maquina@*' -f   (tiene que decir «Detector de personas en GPU»)"
else
  echo "Sin placa NVIDIA: analiza con el procesador ($(nproc) núcleos, $(uname -m)): $CAPACIDAD cámaras de hurto por proceso."
  echo "Listo. Registro: journalctl -u 'vigia-maquina@*' -f"
fi
if [ "$ENCENDEDOR" = "si" ]; then
  instalar_encendedor
fi
