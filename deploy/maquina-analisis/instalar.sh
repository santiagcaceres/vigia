#!/usr/bin/env bash
# Instala Vigía en una máquina de análisis: Ubuntu 22.04 o 24.04 con placa NVIDIA y su driver
# (en AWS: la imagen «Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)» ya lo trae).
#
#   sudo VIGIA_POOL_KEY=vgm_... bash instalar.sh            # máquina de análisis (2 procesos)
#   sudo VIGIA_POOL_KEY=vgm_... PROCESOS=3 CAPACIDAD=12 bash instalar.sh
#   sudo VIGIA_POOL_KEY=vgm_... MODO=escalador bash instalar.sh   # el encendedor (máquina chica)
set -euo pipefail
MODO=${MODO:-maquina}
PROCESOS=${PROCESOS:-2}
CAPACIDAD=${CAPACIDAD:-15}   # cámaras de hurto a 5 fps por proceso (2 x 15 = 30 por máquina, hasta medir)
SERVIDOR=${VIGIA_SERVER:-https://vigia-three-ashy.vercel.app}
REPO=${VIGIA_REPO:-https://github.com/santiagcaceres/vigia.git}
: "${VIGIA_POOL_KEY:?Falta VIGIA_POOL_KEY (se crea en la app: Administración → Máquinas de análisis)}"

apt-get update -q
apt-get install -y -q git python3-venv ffmpeg
id vigia >/dev/null 2>&1 || useradd --system --create-home --home-dir /var/lib/vigia vigia
[ -d /opt/vigia/.git ] || git clone -q --depth 1 "$REPO" /opt/vigia
chown -R vigia:vigia /opt/vigia
sudo -u vigia python3 -m venv /opt/vigia/.venv
sudo -u vigia /opt/vigia/.venv/bin/pip install -q -U pip
mkdir -p /etc/vigia

if [ "$MODO" = "escalador" ]; then
  sudo -u vigia /opt/vigia/.venv/bin/pip install -q httpx boto3
  cat > /etc/vigia/escalador.env <<CFG
VIGIA_SERVER=$SERVIDOR
VIGIA_POOL_KEY=$VIGIA_POOL_KEY
VIGIA_NUBE=${VIGIA_NUBE:-aws}
AWS_REGION=${AWS_REGION:-sa-east-1}
VIGIA_GRUPO=${VIGIA_GRUPO:-analisis}
VIGIA_PLANTILLA=${VIGIA_PLANTILLA:-}
VIGIA_MAX_MAQUINAS=${VIGIA_MAX_MAQUINAS:-10}
CFG
  chown root:vigia /etc/vigia/escalador.env && chmod 640 /etc/vigia/escalador.env
  cp /opt/vigia/deploy/maquina-analisis/vigia-escalador.service /etc/systemd/system/
  systemctl daemon-reload
  systemctl enable --now vigia-escalador
  echo "Listo: encendedor andando. Registro: journalctl -u vigia-escalador -f"
  exit 0
fi

sudo -u vigia /opt/vigia/.venv/bin/pip install -q -r /opt/vigia/requirements-servidor.txt
cat > /etc/vigia/maquina.env <<CFG
VIGIA_SERVER=$SERVIDOR
VIGIA_POOL_KEY=$VIGIA_POOL_KEY
VIGIA_CAPACIDAD=$CAPACIDAD
CFG
chown root:vigia /etc/vigia/maquina.env && chmod 640 /etc/vigia/maquina.env
cp /opt/vigia/deploy/maquina-analisis/vigia-actualizar.service /opt/vigia/deploy/maquina-analisis/vigia-maquina@.service \
   /etc/systemd/system/
systemctl daemon-reload
systemctl enable vigia-actualizar
for i in $(seq 1 "$PROCESOS"); do systemctl enable --now "vigia-maquina@$i"; done
if nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
else
  echo "OJO: no se ve la placa NVIDIA (nvidia-smi). Va a analizar con el procesador (mucho más lento)."
fi
echo "Listo. Registro: journalctl -u 'vigia-maquina@*' -f   (tiene que decir «Detector de personas en GPU»)"
