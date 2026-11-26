# Mudanza a AWS con Tigo (N. Virginia)

Lo que se acordó con Tigo (noviembre 2026) y cómo se arma. El panel, la base y la app no cambian (Vercel y
Supabase); lo único que se muda son las **máquinas que analizan las cámaras**, hoy en Google Cloud.

```
Grabadores ── Hik-Connect o puertos ──►  máquina fija (g4dn, siempre prendida, IP fija) ──► Analítica Crítica
                                    └─►  segunda máquina (g4dn, IP fija)                     (Vercel + Supabase)
                                              ▲ la prende y la detiene la fija, según el horario de los locales
```

- **Dos máquinas con placa de video** (g4dn, NVIDIA T4) en `us-east-1`, cada una con su **IP pública fija**
  (Elastic IP), en una subred pública **sin ningún puerto de entrada**. Sin VPN, sin balanceador (NLB), sin NAT
  Gateway y sin S3: las máquinas se conectan ellas a los grabadores y a nuestro servidor, y el reparto de cámaras lo
  hace nuestro sistema.
- **La máquina fija** es la de la reserva (Savings Plan): está siempre prendida, analiza, y además corre el
  *encendedor*, que cada minuto le pregunta al servidor cuántas máquinas hacen falta.
- **La segunda** se prende sola unos 20 minutos antes de que la fija no alcance (según el horario de cada local) y
  se detiene cuando el servidor ya le sacó las cámaras. Detenida no paga máquina (sí el disco y la IP).
- Para entrar a una máquina no hay SSH ni puertos: **Session Manager** desde la consola de AWS.

Todo lo crea una plantilla de CloudFormation: [`tigo-virginia.yaml`](tigo-virginia.yaml).

## Lo que todavía no se probó (leer antes de empezar)

- **Nunca corrimos el análisis en una placa de video de verdad.** El código está (lotes de cuadros en la placa) y
  probado sin placa, pero la primera máquina es la primera prueba real. Por eso el instalador **no deja analizando
  una máquina que no use la placa**: si algo falla, la máquina no aparece en la app y el motivo queda en el registro.
- **Cuántas cámaras aguanta cada máquina no se sabe.** La máquina lo mide al instalarse y anota como mucho 60
  (parámetro «Tope de cámaras») hasta comprobarlo con cámaras de verdad. La placa detecta; **recibir y decodificar
  el video lo hace el procesador**, que puede ser lo primero en llenarse (por eso existe la opción g4dn.2xlarge).
- La plantilla pasa la validación (`cfn-lint`) pero **no se creó nunca en una cuenta**. La primera vez puede pedir
  algún ajuste.

## Lo que firmamos con Tigo: el acta de riesgo (19 nov 2026)

Como la cuenta es de **co-administración** (nosotros administradores, Tigo solo la infraestructura), la empresa firmó
que asume tres riesgos y que Tigo no responde por ellos:

1. **Los aumentos de costo** por lo que despleguemos.
2. **Las brechas de seguridad** por desplegar sin buenas prácticas.
3. **La gestión de identidad** (usuarios, roles, políticas): lo que hagamos ahí no lo supervisa el equipo de AWS de Tigo.

O sea: los costos, la seguridad y los accesos de la cuenta son **nuestros**. Estas son las reglas para trabajar en
ella (valen para personas y para las sesiones de Claude):

**Costos**
- **Todo lo crea la plantilla** ([`tigo-virginia.yaml`](tigo-virginia.yaml)). A mano en la consola solo lo que dice
  esta guía (cupo, cifrado por defecto, el parámetro con la clave). Así todo tiene dueño y «Delete» de la pila borra
  todo.
- **Crear, agrandar o sumar una máquina lo aprueba el dueño sabiendo el costo** (punto 10). Solo en `us-east-1` y
  solo g4dn: la reserva cubre esa familia; cualquier otra se paga a precio de lista.
- **Aviso de costo desde el día 1:** parámetros «Aviso si el mes pasa de» y «Mail para los avisos de costo» de la
  plantilla (mail al 80 %, al 100 % y si el pronóstico pasa del 100 %; no apaga nada). Además, Billing → **Cost
  Anomaly Detection** (sin cargo) con aviso por mail.
- **Revisar los gastos** cada semana el primer mes y después cada mes: Billing → Bills, y en EC2 lo que quedó suelto
  y se sigue pagando: Elastic IPs sin máquina, Volumes «available», Snapshots. Lo que no es de la pila se borra.
- El encendedor **no crea máquinas**: solo prende y detiene la segunda de la pila.

**Seguridad**
- **Ningún puerto de entrada** (grupo `vigia-sin-entrada`), sin SSH ni archivos .pem: se entra con **Session
  Manager**. Sin S3, sin balanceador, discos cifrados, IMDSv2 (todo ya en la plantilla).
- **La clave de máquinas vive solo en Parameter Store** (SecureString): nunca en la plantilla, el repo (es público),
  un mail o un chat.
- **Quién hizo qué:** CloudTrail → **Event history** guarda 90 días de lo que se hace en la cuenta, sin cargo y sin
  armar nada. Preguntarle a Tigo si además guardan un registro más largo de la organización.
- **Parches:** el programa se actualiza solo cada noche; los parches de seguridad de Ubuntu los instala
  `unattended-upgrades`. Comprobarlo una vez en cada máquina: `systemctl status unattended-upgrades`.
- **Cambiar la plantilla:** antes, `cfn-lint` y que la lea otra persona. Nada de cambios a mano que la pila no sepa.

**Accesos (identidad)**
- **Usuario raíz (root) de la cuenta:** con MFA, sin claves de acceso, guardado por el dueño y sin usar en el día a
  día. Si la cuenta quedó dentro de la organización de Tigo, preguntarles quién tiene el root.
- **Cada persona con su propio acceso y MFA obligatorio** (IAM Identity Center si Tigo lo habilita, o un usuario de
  IAM). Nada de usuarios compartidos.
- **Sin claves de acceso (access keys) para personas.** Las máquinas usan roles con el mínimo permiso (ya está: solo
  leer la clave y prender o detener la segunda). Si algún día un programa necesita claves: de mínimo permiso, con
  dueño y fecha anotados, y nunca pegadas en un chat.
- **Administradores, solo quienes manejan AWS.** Para mirar costos alcanza un permiso de solo lectura de facturación.
- **Bajas el mismo día:** quien deja de trabajar con nosotros pierde el acceso ese día.
- **Cada 3 meses:** IAM → usuarios, claves (último uso) y roles; **IAM Access Analyzer** de acceso externo (sin cargo)
  para ver si algo quedó abierto hacia afuera. Lo que no se usa, se borra.
- **El acceso de Tigo** es para la infraestructura: sin entrar a las máquinas ni copiar sus discos (punto 1.4).

## Cómo quedó de verdad (24 nov 2026)

Tigo no usó nuestra plantilla: entregó **un usuario de IAM administrador** y **una g4dn.xlarge ya creada** por ellos
(Ubuntu 24.04 común, sin el driver de NVIDIA), en una **subred privada sin IP pública**, que sale a internet por **un NAT
Gateway** suyo con una Elastic IP. Esa IP del NAT es la que ven los grabadores (la que se autoriza en los routers de los
clientes). Ojo con el costo: el NAT cobra por hora y **por GB** (todo el video de las cámaras pasa por ahí); lo acordado
era subred pública con una Elastic IP por máquina y sin NAT (se lo pedimos a Tigo el 24 nov).

Se instaló a mano, por **Session Manager** (Conectar → «Administrador de sesiones de SSM»; el rol de la máquina ya lo
permite):

```bash
# 1. el driver de la placa (la imagen común no lo trae; nuestro instalador no lo instala)
sudo apt-get update && sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers install --gpgpu nvidia:580-server-open && sudo apt-get install -y nvidia-utils-580-server
sudo reboot            # y después: nvidia-smi → «Tesla T4», driver 580, CUDA 13
# 2. nuestro programa (la clave de máquinas sale de la app: Máquinas → Más → Clave para máquinas nuevas)
curl -fsSL https://raw.githubusercontent.com/santiagcaceres/vigia/main/deploy/maquina-analisis/instalar.sh -o /tmp/instalar.sh
sudo nohup sh -c 'VIGIA_POOL_KEY=vgm_… VIGIA_GPU=si VIGIA_FIJA=si CAPACIDAD_TOPE=60 ACTUALIZAR_DIARIO=si ENTRENAR=si \
  bash /tmp/instalar.sh; echo "== TERMINO, codigo $?"' > /tmp/instalar.log 2>&1 &
tail -f /tmp/instalar.log
```

(En segundo plano con `nohup`: si se corta la sesión, la instalación sigue.) Resultado: **midió 17 cámaras de hurto**
(GPU CUDA, 122 cuadros/s analizados con 4 vCPU: el límite es el procesador, no la placa), 2 procesos de 8, Ultralytics
para entrenar objetos con la placa. A los minutos el reparto le pasó solo 22 cámaras desde la máquina de Google (la fija
va primero): la farmacia pasó de 2-3 cuadros/s salteando a ~4 sin saltear. Sin encendedor (no hay segunda máquina).
Si con todas las cámaras saltea mucho, pasar a g4dn.2xlarge (el doble de procesador; la reserva cubre la familia).

### Decodificar el video en la placa (26 nov 2026)

El driver «para cálculo» (`--gpgpu`) no trae las librerías de video de NVIDIA, y sin ellas ffmpeg decodifica con el
procesador aunque se le pida la placa. Se instalaron a mano el 26 nov, de la **misma versión** que el driver que anda
(desde esa fecha `instalar.sh` lo hace solo):

```bash
v=$(dpkg-query -W -f='${Package}\n' 'libnvidia-compute-*' | sed -n 's/^libnvidia-compute-//p' | head -1)
ver=$(dpkg-query -W -f='${Version}' "libnvidia-compute-$v")
sudo apt-get install -y --no-install-recommends "libnvidia-decode-$v=$ver" "libnvidia-encode-$v=$ver"
```

Medido después (1 minuto de H.265 960x576 a 12 cuadros/s): decodificar con el procesador, 2,3-2,5 s de procesador;
con la placa al tamaño original (`hevc_cuvid`, NV12), 0,33 s. El programa lo usa con `VIGIA_VIDEO_PLACA=si`
(`agent/video_placa.py`, apagado de fábrica). Para probarlo en una sola de las 4 partes de la máquina y comparar:

```bash
sudo mkdir -p /etc/systemd/system/vigia-maquina@4.service.d
printf '[Service]\nEnvironment=VIGIA_VIDEO_PLACA=si\n' | sudo tee /etc/systemd/system/vigia-maquina@4.service.d/placa-video.conf
sudo systemctl daemon-reload && sudo systemctl start vigia-actualizar-diario.service   # trae el código y reinicia
```

Deshacer: `sudo rm /etc/systemd/system/vigia-maquina@4.service.d/placa-video.conf && sudo systemctl daemon-reload &&
sudo systemctl restart vigia-maquina@4`.

**Resultado en la parte 4 (26 nov, 12:46, 6 cámaras, con el arreglo de los cuadros repetidos y bajando de la placa
solo los que se usan):** leer el video costó ~la mitad que con el procesador (0,19 procesadores las 5 que andaban por
la placa, contra ~0,40 estimado con el procesador), a los mismos 3,9-4,1 cuadros/s analizados. Una cámara que manda 12
cuadros/s llegaba con 25 (13 eran copias de ffmpeg): ahora con los 12 de verdad. La memoria de la placa: ~120 MiB por
cámara. Para prenderlo en **las 4 partes** (un archivo para todas, en vez del de la parte 4):

```bash
sudo mkdir -p /etc/systemd/system/vigia-maquina@.service.d
printf '[Service]\nEnvironment=VIGIA_VIDEO_PLACA=si\n' | sudo tee /etc/systemd/system/vigia-maquina@.service.d/placa-video.conf
sudo rm -f /etc/systemd/system/vigia-maquina@4.service.d/placa-video.conf
sudo systemctl daemon-reload && sudo systemctl start vigia-actualizar-diario.service   # trae el código y reinicia
```

Deshacer en las 4: `sudo rm /etc/systemd/system/vigia-maquina@.service.d/placa-video.conf && sudo systemctl
daemon-reload && sudo systemctl restart 'vigia-maquina@*'`.

## 1. Antes de crear nada (con Tigo)

1. Cuenta de AWS con nosotros como administradores, en **N. Virginia (`us-east-1`)**.
2. **Cupo de máquinas con placa:** Service Quotas → Amazon EC2 → «Running On-Demand G and VT instances» ≥ **16 vCPU**
   (dos g4dn.xlarge usan 8; dos g4dn.2xlarge, 16). En cuentas nuevas viene en 0: lo pide Tigo.
3. **La red:** si Tigo ya armó la VPC, pedirles el id de la VPC y el de la **subred pública** (con ruta a un
   Internet Gateway). Si no armaron nada, la plantilla crea una red chica propia.
4. Pedirles que su acceso de cogestión **no incluya entrar a las máquinas** (Session Manager) ni copiar sus discos.
5. EC2 → Settings → **EBS encryption → «Always encrypt new EBS volumes»** (la plantilla igual cifra los discos).
6. **Avisos de costo:** completar en la plantilla «Aviso si el mes pasa de» y el mail (punto 3), y prender Billing →
   **Cost Anomaly Detection**. Antes de crear la pila: MFA en el root y un acceso propio con MFA para cada uno.

## 2. La clave de máquinas

1. En la app: Máquinas → «Más» → **Clave para máquinas nuevas** → copiarla. (Ojo: crear una clave nueva deja sin
   efecto la anterior para máquinas nuevas; la de Google ya dada de alta sigue andando.)
2. En AWS (Virginia): Systems Manager → **Parameter Store** → Create parameter → Name `/vigia/clave-maquinas` · Type
   **SecureString** · Value: la clave. No queda escrita en la plantilla ni en ningún archivo.

## 3. Crear la pila: primero una sola máquina

CloudFormation (Virginia) → **Create stack** → Upload a template file → `tigo-virginia.yaml` → nombre
`analitica-critica` →

| Parámetro | Para empezar |
|---|---|
| Máquina fija | `g4dn.xlarge` (se cambia después en 10 minutos, ver punto 8) |
| ¿Crear la segunda máquina? | **no** (primero se prueba una) |
| Tope de cámaras por máquina | 60 |
| VPC y subred que ya existen | los de Tigo, o **vacíos** para que la plantilla cree la red |
| Zona | `us-east-1a` (solo si la red la crea la plantilla) |
| Aviso si el mes pasa de · Mail | lo esperado por mes (punto 10) · el mail del dueño |

Tildar «I acknowledge that AWS CloudFormation might create IAM resources» → Create stack. La pila tarda unos 3
minutos; la máquina, **entre 10 y 15 más** en instalarse (baja las librerías de la placa, mide y se da de alta).

Si la pila falla diciendo que no hay máquinas de ese tipo en la zona, borrarla y crearla con otra zona.

## 4. Qué tiene que pasar

1. En la app, **Máquinas**: aparece la máquina nueva («maquina-i-0…», con sus dos procesos) **En línea**.
2. **La máquina nueva se lleva sola las cámaras que hoy analiza la de Google** (es la fija: el sistema la prefiere).
   La de Google queda prendida y vacía, de respaldo. No hay que mover nada a mano.
3. Mirar el registro de la instalación: EC2 → la instancia `vigia-fija` → **Connect → Session Manager** →

   ```bash
   sudo tail -40 /var/log/vigia-instalar.log
   ```

   Tiene que decir `Medido: N cámaras de hurto · detector en GPU (CUDA)` y terminar en `INSTALACIÓN TERMINADA BIEN`.
   Anotar ese N.
4. En la app, **En vivo**: destildar «Ver con la IA de esta PC», elegir una cámara que se analice y mirar que la
   imagen de la máquina tenga los recuadros sobre las personas. En Clientes, que cada cámara diga 5 cuadros por
   segundo.
5. En la máquina, con las cámaras andando: `nvidia-smi` (uso de la placa) y `top` (procesador).

**Si algo anda mal:** EC2 → `vigia-fija` → Instance state → **Stop**. En dos minutos las cámaras vuelven solas a la
máquina de Google (mientras siga prendida). Si la máquina nunca apareció en la app, el motivo está en
`/var/log/vigia-instalar.log` (por ejemplo «el detector NO la está usando»).

Dejarla así **uno o dos días** antes de seguir.

## 5. Apagar la de Google

Cuando la de AWS anduvo bien: en la app, Máquinas → la de Google → **■ Apagar**. Unos días después, borrarla en
Google Cloud (apagada, el disco sigue gastando crédito) y quitarla de la lista en la app.

## 6. Agregar la segunda máquina

CloudFormation → la pila → **Update** → Use existing template → «¿Crear la segunda máquina?» = **si** → Update. No
toca la fija. La segunda se instala (10-15 minutos), se da de alta y, si la fija alcanza sola, **a los pocos minutos
se detiene**: es lo esperado. Se va a prender sola cuando las cámaras no entren en la fija.

Para verlo: en la fija, `sudo journalctl -u vigia-escalador -n 30 --no-pager` («Prendiendo 1 máquina», «Apagando 1
máquina sin cámaras»).

## 7. Las IP fijas

CloudFormation → la pila → **Outputs**: `IpDeLaMaquinaFija` e `IpDeLaSegundaMaquina`. Son las direcciones desde las
que nuestras máquinas llegan a los grabadores. En los locales que se conectan **por puertos**, el técnico puede
dejar en el router que esos puertos solo acepten esas dos IP.

**Ojo:** «Ver con la IA de esta PC», «Tomar foto del vivo», «Probar conexión» y «Bajar grabación» abren el video
desde la PC de quien usa la app, no desde AWS. Si el router del cliente solo acepta las IP de AWS, eso deja de andar
desde las PC (la imagen de la máquina en «En vivo» sí sigue). Si la oficina tiene IP fija, sumarla también.

## 8. Cambiar el tamaño de una máquina

CloudFormation → Update → «Máquina fija» (o «Segunda máquina») = `g4dn.2xlarge` → Update. AWS la apaga, la cambia y
la prende (menos de 10 minutos; la IP no cambia; mientras tanto sus cámaras pasan a la otra, si está prendida).

Después, para que use los 8 procesadores, en la máquina:

```bash
sudo systemctl enable --now vigia-maquina@3 vigia-maquina@4     # de 2 a 4 procesos
sudo nano /etc/vigia/maquina.env                                # VIGIA_CAPACIDAD = cámaras por proceso
sudo systemctl restart 'vigia-maquina@*'
```

**Cuándo hace falta:** si con la placa todavía holgada (`nvidia-smi` por debajo del 70 %) el procesador está lleno
(`top` cerca del 100 % en todos) y las cámaras bajan de 5 cuadros por segundo, falta procesador: 2xlarge. Si lo que
está llena es la placa, no sirve agrandar: hace falta otra máquina.

**Subir el tope de cámaras** cuando ya se comprobó cuánto aguanta: mismo archivo, `VIGIA_CAPACIDAD` (por proceso) y
reiniciar. Si pasa de 60 por proceso, subir también «Cámaras por máquina» en la app (Máquinas): el servidor no le
cree a una máquina más del doble de ese número por proceso.

## 9. El día a día

- **Actualizaciones:** las dos máquinas bajan la última versión del programa cada noche a las 04:30 de Uruguay y
  cada vez que prenden. No hay que reiniciar nada a mano.
- **Si la máquina fija se cae**, se cae también el encendedor: la segunda no se prende sola. Prenderla a mano (EC2 →
  `vigia-segunda` → Start): recibe las cámaras en un par de minutos. Después, prender la fija.
- **Si AWS no tiene lugar para prender la segunda** (pasa a veces con las máquinas con placa), el encendedor
  reintenta cada minuto y mientras tanto la fija se queda con todas las cámaras, apretada.
- **Dónde mirar:**

  ```bash
  sudo tail -50 /var/log/vigia-instalar.log                  # la instalación
  sudo journalctl -u 'vigia-maquina@*' -n 100 --no-pager     # el análisis
  sudo journalctl -u vigia-escalador -n 100 --no-pager       # el encendedor (solo en la fija)
  nvidia-smi ; top                                           # placa y procesador
  ```

## 10. Lo que se paga (precios de lista de AWS en Virginia, noviembre 2026; Tigo cobra los suyos)

| Pieza | Precio de lista |
|---|---|
| g4dn.xlarge (4 vCPU, 16 GB, T4) | US$ 0,526 por hora a demanda (≈ 384 por mes prendida siempre) |
| g4dn.2xlarge (8 vCPU, 32 GB, T4) | US$ 0,752 por hora a demanda (≈ 549 por mes prendida siempre) |
| IP pública fija | US$ 0,005 por hora cada una (≈ 3,65 por mes), prendida o detenida la máquina |
| Disco | US$ 0,08 por GB y por mes, también con la máquina detenida |
| Tráfico de salida | los primeros 100 GB por mes sin cargo; después US$ 0,09 por GB (subimos clips y fotos, poco) |

La reserva (Savings Plan de la familia g4dn) es un compromiso de dinero por hora durante 12 meses: se aplica a
cualquier g4dn que esté prendida en la región, del tamaño que sea, y **no se puede cancelar**. Se puede comprar
después de crear las máquinas, sin tocarlas.

## 11. Borrar todo

En la app, «✘ No analizar» en las cámaras (o tener otra máquina prendida). CloudFormation → la pila → **Delete**:
borra máquinas, IP, roles y, si la creó la plantilla, la red. El parámetro con la clave se borra a mano.
