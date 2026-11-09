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

## 1. Antes de crear nada (con Tigo)

1. Cuenta de AWS con nosotros como administradores, en **N. Virginia (`us-east-1`)**.
2. **Cupo de máquinas con placa:** Service Quotas → Amazon EC2 → «Running On-Demand G and VT instances» ≥ **16 vCPU**
   (dos g4dn.xlarge usan 8; dos g4dn.2xlarge, 16). En cuentas nuevas viene en 0: lo pide Tigo.
3. **La red:** si Tigo ya armó la VPC, pedirles el id de la VPC y el de la **subred pública** (con ruta a un
   Internet Gateway). Si no armaron nada, la plantilla crea una red chica propia.
4. Pedirles que su acceso de cogestión **no incluya entrar a las máquinas** (Session Manager) ni copiar sus discos.
5. EC2 → Settings → **EBS encryption → «Always encrypt new EBS volumes»** (la plantilla igual cifra los discos).
6. Billing → **Budgets**: un aviso por mail si el mes pasa de lo esperado.

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
