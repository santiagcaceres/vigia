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

## Prueba gratis con pocas cámaras (sin placa de video)

Para probar el circuito completo con un cliente (por ejemplo 5 cámaras de Minas). Sin placa de video se analiza con
el procesador: medido en 4 núcleos x86, unos 42 cuadros por segundo (≈ 8 cámaras de hurto a 5 fps sin contar la
decodificación); en ARM rinde menos. No hace falta el encendedor: la máquina queda siempre prendida.

**Antes, en la app:** Administración → **Clave para máquinas nuevas** (copiala). En el cliente de prueba, dejá solo
las cámaras a probar o elegí «Dónde se analiza: En las máquinas de análisis» **cámara por cámara**: si lo ponés en el
cliente, se van a la máquina todas sus cámaras. El ajuste general queda en «En la PC de cada usuario».

### Opción A: Google Cloud (US$ 300 gratis por 90 días, la recomendada para 5 cámaras)

1. console.cloud.google.com → **Probar gratis** (pide tarjeta para verificar; no cobra nada salvo que actives la
   cuenta paga). Creá un proyecto `vigia`.
2. Menú → Compute Engine → **Instancias de VM** → Habilitar (tarda un minuto) → **Crear instancia**:
   - Nombre `vigia-analisis-1`. Región **southamerica-east1 (São Paulo)**.
   - Máquina: E2 → **e2-standard-4** (4 vCPU, 16 GB; a la derecha muestra el costo por mes, ~US$ 150 prendida
     las 24 h). Si los fps no llegan a 5, e2-standard-8.
   - Disco de arranque → Cambiar → **Ubuntu 24.04 LTS (x86/64)**, 30 GB.
   - Firewall: sin tildar HTTP ni HTTPS (no necesita puertos de entrada).
3. En la lista de instancias, botón **SSH** (abre una terminal en el navegador) y corré:

   ```bash
   curl -fsSL https://raw.githubusercontent.com/santiagcaceres/vigia/main/deploy/maquina-analisis/instalar.sh -o instalar.sh
   sudo VIGIA_POOL_KEY=vgm_LA_CLAVE bash instalar.sh
   journalctl -u 'vigia-maquina@*' -f
   ```

   Tiene que decir «Detector de personas en procesador (4 cuadros a la vez)», «Máquina … dada de alta» y, al rato,
   «[Cliente · Cámara] conectada».
4. Al terminar la prueba: **Detener** la instancia (no gasta procesador; el disco cuesta centavos) o borrarla.

**Probado el 26 oct de 2026:** una e2-standard-4 en São Paulo analizó 3 cámaras de Minas por Hik-Connect a 5 fps con
~11 % de procesador. La máquina toma como id el nombre de la instancia (ya no hace falta `VIGIA_INSTANCE_ID`). Los
avisos de ffmpeg «[tls] IO error: End of file» cada tanto son normales: Hik-Connect corta la conexión y ffmpeg
la reabre sin perder el video.

### Opción B: Oracle Cloud «Always Free» (gratis para siempre, pero chica)

Desde agosto de 2026 (según varios sitios; Oracle no lo confirma en su página) lo gratis de ARM bajó a **2 núcleos y
12 GB**: alcanza para 2 o 3 cámaras de hurto a 5 fps (más si son de conteo).

1. Creá la cuenta en oracle.com/cloud/free. **Elegí como región «Brazil East (São Paulo)»**: lo gratis solo
   anda en la región que elegís al crear la cuenta y no se puede cambiar. Pide tarjeta para verificar (no cobra).
2. Menú → Compute → Instances → **Create instance**:
   - Image: **Canonical Ubuntu 24.04** (la versión aarch64 aparece al elegir la forma Ampere).
   - Shape: **Ampere → VM.Standard.A1.Flex**, con todos los núcleos y la memoria que deje gratis.
   - Bajá la clave SSH que te ofrece. Lo demás, como viene.
   - Si dice «Out of capacity», probá otro «availability domain» o reintentá más tarde (pasa seguido).
3. Entrá por SSH (`ssh -i clave.key ubuntu@IP_PUBLICA`; en Windows, desde PowerShell) y corré los mismos comandos
   que en Google Cloud. En la app aparece en Administración → Máquinas de análisis.

Oracle puede recuperar una máquina gratis que pase una semana casi sin usar el procesador; analizando no pasa.

### Qué mirar en la prueba

- En la app, columna «Analiza» de las cámaras = la máquina, y estado «en línea» con 5 fps.
- En la máquina, `top`: si el procesador está al 100 % y los fps bajan de 5, sacale cámaras o usá una más grande.
- Una alerta de punta a punta: foto, clip dentro de la app, etiqueta y «Avisar al cliente».

## AWS (Tigo) u otra nube

**En AWS, lo armado es `deploy/aws/`**: una plantilla de CloudFormation crea el grupo de máquinas Spot Graviton, el
encendedor y los permisos, sin puertos de entrada (ver `deploy/aws/README.md`). Lo de arriba (máquinas a mano con
placa de video) queda para cuando se pruebe GPU.

`instalar.sh` sirve en cualquier Ubuntu 22.04/24.04, x86 o ARM, con o sin placa NVIDIA. Sin placa **mide** cuántas
cámaras aguanta la máquina al instalarse (`python -m agent.medir`) y usa ese número como capacidad (antes: 1 por
vCPU). Si la nube no se puede manejar por programa, instalá el encendedor con `VIGIA_NUBE=ninguna`: no toca nada y
solo registra cuántas máquinas harían falta (las prendés a mano, o las dejás siempre prendidas).

**Máquinas Spot (AWS o Google):** la máquina mira cada 5 segundos si la nube avisó que se la lleva (AWS avisa 2
minutos antes; Google, 30 s). Si avisó, se lo dice al servidor (sus cámaras pasan a las otras en el momento),
suelta las transmisiones de Hik-Connect y manda lo que tenga en la cola. En la app figura «Se va (la nube la
reclamó)».

## Seguridad

- Las máquinas no necesitan puertos de entrada: solo salen a internet (Hik-Connect y Vigía).
- La clave de máquinas solo sirve para dar de alta máquinas. Si se filtra, creá otra en la app (las ya dadas
  de alta siguen andando) y borrá las máquinas que no reconozcas.
- Las claves de AWS no van en ningún archivo: el encendedor usa el rol de IAM de su instancia.
