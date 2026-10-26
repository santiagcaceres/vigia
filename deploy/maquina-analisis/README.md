# Máquinas de análisis en la nube

Las cámaras se analizan en **máquinas en la nube con placa NVIDIA** y los empleados solo revisan las alertas
desde la app (sirve cualquier notebook). Las máquinas se **prenden y apagan solas** según el horario de los
locales: de noche no se pagan.

```
Grabadores ── Hik-Connect (São Paulo) ──► máquinas de análisis (AWS São Paulo / Tigo) ──► Vigía (Vercel + Supabase)
                                              ▲                                              │
                                   encendedor (máquina chica, siempre prendida) ◄── «cuántas hacen falta»
```

- **Vigía** reparte las cámaras entre las máquinas en línea (si una se cae, en ~2 minutos sus cámaras pasan a
  las otras) y calcula cuántas hacen falta ahora y en los próximos 20 minutos (`server/pool.py`).
- **Cada máquina** se da de alta sola con la *clave de máquinas*, recibe sus cámaras y las analiza juntando
  los cuadros de todas en la placa de video (`core/lotes.py`). Al arrancar baja la última versión del programa.
- **El encendedor** (`agent/escalador.py`) pregunta cada minuto cuántas máquinas hacen falta y las prende o
  apaga en la nube. Antes de apagar una, Vigía le saca las cámaras.

## 0. En la app

Administración → **Clave para máquinas nuevas** → copiala (no se vuelve a mostrar).

## 1. La primera máquina (AWS, región São Paulo `sa-east-1`)

1. EC2 → **Lanzar instancia**:
   - Imagen: **Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)** (ya trae el driver de NVIDIA).
   - Tipo: **g6.xlarge** para probar (1 NVIDIA L4, 4 vCPU). Para producción, g6.2xlarge o g6.4xlarge (más
     procesador para decodificar más cámaras).
   - Disco: 60 GB gp3. Grupo de seguridad: **sin puertos de entrada** (solo SSH desde tu IP para instalar).
   - Etiqueta: `vigia-grupo` = `analisis`.
2. Entrá por SSH y corré:

   ```bash
   curl -fsSL https://raw.githubusercontent.com/santiagcaceres/vigia/main/deploy/maquina-analisis/instalar.sh -o instalar.sh
   sudo VIGIA_POOL_KEY=vgm_LA_CLAVE bash instalar.sh
   ```

   Corre 2 procesos de 15 cámaras cada uno (`PROCESOS=3 CAPACIDAD=12` para cambiarlo).
3. Mirá el registro: `journalctl -u 'vigia-maquina@*' -f`. Tiene que decir **«Detector de personas en GPU
   (CUDA) (lotes de hasta 16 cuadros)»** y «Máquina … dada de alta». Si dice «en procesador», la placa no se
   está usando: revisá `nvidia-smi` (el driver tiene que ser 580 o más nuevo; si es más viejo, instalá
   `onnxruntime-gpu[cuda,cudnn]==1.22.*` en `/opt/vigia/.venv`).
4. En la app, Administración → «Dónde se analizan las cámaras» → **En las máquinas de análisis** → Guardar.
   Mientras no haya ninguna máquina en línea, las cámaras siguen en la PC de cada usuario.

## 2. Medir cuántas cámaras aguanta una máquina

Con las cámaras ya repartidas, mirá en la app los fps de cada cámara (tienen que dar 5) y en la máquina
`nvidia-smi` (uso de la placa) y `top` (procesador). Subí las cámaras de a poco hasta que los fps bajen y
dejá «Cámaras por máquina» (Administración) un 15 % por debajo de ese número. **Esto todavía no se midió.**

## 3. Más máquinas: imagen y plantilla

1. Con la primera andando: EC2 → la instancia → Acciones → Imagen y plantillas → **Crear imagen**.
2. EC2 → Plantillas de lanzamiento → **Crear**: esa imagen, el mismo tipo, disco y grupo de seguridad, y la
   etiqueta `vigia-grupo=analisis` para instancias. Anotá su id (`lt-…`).
3. Cada copia se da de alta sola como máquina nueva (el token se guarda por instancia).

## 4. El encendedor

1. Una instancia chica siempre prendida: **t4g.nano** (Ubuntu 24.04, ~US$ 3 por mes), sin puertos de entrada.
2. Dale un **rol de IAM** con este permiso (sin claves en ningún archivo):

   ```json
   {"Version": "2012-10-17", "Statement": [
     {"Effect": "Allow", "Action": ["ec2:DescribeInstances"], "Resource": "*"},
     {"Effect": "Allow", "Action": ["ec2:StartInstances", "ec2:StopInstances"], "Resource": "*",
      "Condition": {"StringEquals": {"aws:ResourceTag/vigia-grupo": "analisis"}}},
     {"Effect": "Allow", "Action": ["ec2:RunInstances", "ec2:CreateTags"], "Resource": "*"}
   ]}
   ```

   (La última línea solo si querés que cree máquinas nuevas con la plantilla.)
3. Instalalo:

   ```bash
   sudo VIGIA_POOL_KEY=vgm_LA_CLAVE MODO=escalador VIGIA_PLANTILLA=lt-XXXX VIGIA_MAX_MAQUINAS=10 bash instalar.sh
   journalctl -u vigia-escalador -f
   ```

   `VIGIA_MAX_MAQUINAS` es un tope para que un error nunca prenda máquinas de más.

## Tigo u otra nube

`instalar.sh` sirve en cualquier Ubuntu con placa NVIDIA. Si la nube no se puede manejar por programa,
instalá el encendedor con `VIGIA_NUBE=ninguna`: no toca nada y solo registra cuántas máquinas harían falta
(las prendés a mano, o las dejás siempre prendidas).

## Seguridad

- Las máquinas no necesitan puertos de entrada: solo salen a internet (Hik-Connect y Vigía).
- La clave de máquinas solo sirve para dar de alta máquinas. Si se filtra, creá otra en la app (las ya dadas
  de alta siguen andando) y borrá las máquinas que no reconozcas.
- Las claves de AWS no van en ningún archivo: el encendedor usa el rol de IAM de su instancia.
