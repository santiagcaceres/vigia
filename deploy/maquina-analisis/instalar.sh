#!/usr/bin/env bash
# Instala Vigía en una máquina de análisis: Ubuntu 22.04 o 24.04, x86 o ARM (AWS Graviton).
# - Con placa NVIDIA y su driver (en AWS: «Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)»).
# - Sin placa: se da cuenta solo, analiza con el procesador y MIDE cuántas cámaras aguanta (agent/medir.py).
#
#   sudo VIGIA_POOL_KEY=vgm_... bash instalar.sh            # máquina de análisis
#   sudo VIGIA_POOL_KEY=vgm_... PROCESOS=3 CAPACIDAD=12 bash instalar.sh
#   sudo VIGIA_POOL_KEY=vgm_... MODO=escalador bash instalar.sh   # el encendedor (máquina chica)
#   sudo VIGIA_POOL_KEY_SSM=/vigia/clave-maquinas bash instalar.sh   # en AWS: la clave sale del Parameter Store
#                                                                    # (así la usa deploy/aws/vigia-analisis.yaml)
set -euo pipefail
MODO=${MODO:-maquina}
if nvidia-smi >/dev/null 2>&1; then
  GPU=si; REQ=requirements-servidor.txt
  PROCESOS=${PROCESOS:-2}
  CAPACIDAD=${CAPACIDAD:-15}   # cámaras de hurto a 5 fps por proceso (2 x 15 = 30 por máquina, hasta medir)
else
  GPU=no; REQ=requirements-servidor-cpu.txt
  PROCESOS=${PROCESOS:-1}
  CAPACIDAD=${CAPACIDAD:-}     # sin placa: se mide al final de la instalación
fi
SERVIDOR=${VIGIA_SERVER:-https://vigia-three-ashy.vercel.app}
REPO=${VIGIA_REPO:-https://github.com/santiagcaceres/vigia.git}
if [ -z "${VIGIA_POOL_KEY:-}" ] && [ -z "${VIGIA_POOL_KEY_SSM:-}" ]; then
  echo "Falta VIGIA_POOL_KEY (se crea en la app: Administración → Máquinas de análisis)" >&2
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

if [ "$MODO" = "escalador" ]; then
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
  echo "Listo: encendedor andando. Registro: journalctl -u vigia-escalador -f"
  exit 0
fi

sudo -u vigia /opt/vigia/.venv/bin/pip install -q -r "/opt/vigia/$REQ"
echo "VIGIA_REQ=$REQ" > /etc/vigia/actualizar.env
if [ -z "$CAPACIDAD" ]; then
  # sin placa: medir cuántas cámaras de hurto entran (detector + seguimiento + decodificar, 30 % de margen). Cada
  # tipo de máquina rinde distinto (en AWS el grupo mezcla tipos): por eso se mide en cada una.
  echo "Midiendo cuántas cámaras aguanta esta máquina (unos 30 segundos)…"
  MEDIDO=$(cd /opt/vigia && sudo -u vigia "$PY" -m agent.medir --segundos 12 --numero 2>/dev/null | tail -1 || true)
  case "$MEDIDO" in ''|*[!0-9]*) MEDIDO=$(nproc) ;; esac
  CAPACIDAD=$(( MEDIDO / PROCESOS )); [ "$CAPACIDAD" -ge 1 ] || CAPACIDAD=1
fi
cat > /etc/vigia/maquina.env <<CFG
VIGIA_SERVER=$SERVIDOR
VIGIA_POOL_KEY=$VIGIA_POOL_KEY
VIGIA_CAPACIDAD=$CAPACIDAD
CFG
# sin placa de video: un modelo por núcleo, repartidos entre los procesos
HILOS=$(( $(nproc) / PROCESOS )); [ "$HILOS" -ge 1 ] || HILOS=1
[ "$GPU" = "no" ] && echo "VIGIA_HILOS=$HILOS" >> /etc/vigia/maquina.env
chown root:vigia /etc/vigia/maquina.env && chmod 640 /etc/vigia/maquina.env
cp /opt/vigia/deploy/maquina-analisis/vigia-actualizar.service /opt/vigia/deploy/maquina-analisis/vigia-maquina@.service \
   /etc/systemd/system/
systemctl daemon-reload
systemctl enable vigia-actualizar
for i in $(seq 1 "$PROCESOS"); do systemctl enable --now "vigia-maquina@$i"; done
if [ "$GPU" = "si" ]; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
  echo "Listo. Registro: journalctl -u 'vigia-maquina@*' -f   (tiene que decir «Detector de personas en GPU»)"
else
  echo "Sin placa NVIDIA: analiza con el procesador ($(nproc) núcleos, $(uname -m)): $CAPACIDAD cámaras de hurto por proceso."
  echo "Listo. Registro: journalctl -u 'vigia-maquina@*' -f"
fi
