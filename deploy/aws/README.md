# Analítica Crítica en AWS (pensado para Tigo)

Una plantilla de CloudFormation (`vigia-analisis.yaml`) arma todo lo que hace falta en una cuenta de AWS para
analizar las cámaras en la nube. El panel, la base y la app no cambian: siguen en Vercel y Supabase.

```
Grabadores ── Hik-Connect ──► máquinas de análisis (Spot, Graviton) ──► Analítica Crítica (Vercel + Supabase)
                                   ▲  grupo de Auto Scaling (0 a N)                │
                                   └──── encendedor (t4g.nano, siempre prendido) ◄─┘ «cuántas hacen falta ahora»
```

- **Máquinas de análisis**: un grupo de Auto Scaling de máquinas **Spot** (hasta 70-90 % más baratas que a demanda)
  con procesador **Graviton** (ARM: cada vCPU es un núcleo real, que es lo que rinde para este trabajo). Al prender,
  cada máquina instala el programa, **mide cuántas cámaras aguanta** y se da de alta sola en el servidor.
- **El encendedor** pregunta cada minuto cuántas máquinas hacen falta (según el horario de cada local, mirando 20
  minutos adelante) y ajusta el grupo. De noche, con los locales cerrados, quedan **cero máquinas**: no se pagan.
- **Si AWS reclama una máquina Spot** (avisa 2 minutos antes), la máquina se lo dice al servidor, que pasa sus
  cámaras a las otras en el momento, y el grupo pone otra sola.
- **Sin puertos de entrada**, discos cifrados, sin claves de AWS en ningún archivo (roles de IAM). Para entrar a
  una máquina se usa Session Manager (desde la consola, sin SSH).

## Lo que cuesta (São Paulo, `sa-east-1`, octubre 2026)

| Pieza | Tipo | A demanda | Spot |
|---|---|---|---|
| Máquina de análisis | c7g.xlarge (4 núcleos Graviton3, 8 GB) | US$ 0,2222 por hora | ≈ US$ 0,05 por hora |
| Encendedor | t4g.nano, siempre prendido | ≈ US$ 4 por mes | — |
| Disco e IP pública | por máquina prendida | centavos por día | igual |

**Cuántas cámaras por máquina:** cada máquina lo mide al prender (`agent/medir.py`: detector, seguimiento, reglas
y decodificación, con 30 % de margen) y aparece en la app, Administración → máquinas («carga / capacidad»). En
Google Cloud medimos unos **10-11 cuadros analizados por segundo por núcleo real** (una cámara de hurto usa 5): una
máquina de 4 núcleos reales ≈ **6 cámaras de hurto**. **Graviton todavía no se midió**; la primera máquina lo dice.

Con 6 cámaras por c7g.xlarge Spot: **≈ US$ 0,008 por cámara y hora**. Una cámara mirada 12 horas por día ≈ **US$ 3
por mes** (a demanda, ≈ US$ 13). Las cámaras de conteo (2 cuadros por segundo) pesan la mitad. Precios para
confirmar en la consola (Spot cambia según la demanda).

## 1. Antes de crear la pila

1. **La clave de máquinas.** En la app: Administración → **Clave para máquinas nuevas** → copiala.
2. **Guardarla en AWS** (región São Paulo): Systems Manager → **Parameter Store** → Create parameter:
   - Name: `/vigia/clave-maquinas`
   - Tier: Standard · Type: **SecureString** · KMS key: la que viene (`alias/aws/ssm`)
   - Value: la clave. (Es gratis. No queda en la plantilla ni en los datos de las máquinas al crearlas.)
3. **Cupo de Spot.** Service Quotas → Amazon EC2 → «All Standard (A, C, D, H, I, M, R, T, Z) Spot Instance
   Requests»: tiene que ser al menos 4 × el tope de máquinas (en vCPU). Las cuentas nuevas a veces tienen poco: se
   pide más ahí mismo.

## 2. Crear la pila

CloudFormation (São Paulo) → **Create stack** → With new resources → Upload a template file → `vigia-analisis.yaml`
→ nombre `analitica-critica` → parámetros:

| Parámetro | Para empezar |
|---|---|
| Procesador | `arm64` (Graviton) |
| Tipos | `c7g.xlarge`, `c8g.xlarge`, `c6g.xlarge` (si no hay Spot de uno, usa otro) |
| Tope de máquinas | **1** para la primera prueba (después, lo que haga falta: nunca se pasa) |
| Máquinas siempre a demanda | 0 (todas Spot). Con 1, siempre hay una que AWS no reclama |
| VPC y subredes | la VPC «default» y **todas** sus subredes (más zonas = más lugar para Spot) |

Tildá «I acknowledge that AWS CloudFormation might create IAM resources» → **Create stack** (tarda unos 3 minutos).

Por consola de comandos es lo mismo:

```bash
aws cloudformation deploy --region sa-east-1 --stack-name analitica-critica \
  --template-file deploy/aws/vigia-analisis.yaml --capabilities CAPABILITY_IAM \
  --parameter-overrides VpcId=vpc-XXXX Subredes=subnet-AAAA,subnet-BBBB,subnet-CCCC MaxMaquinas=1
```

## 3. Qué tiene que pasar

1. El encendedor se instala solo (unos 3 minutos) y empieza a preguntar cada minuto.
2. Hace falta alguna máquina solo si hay cámaras para analizar: en la app, **Clientes → elegir cámaras → ✔ Analizar**
   (y el local abierto según su horario).
3. Aparece una máquina en EC2 (`vigia-analisis`). En unos 5-7 minutos se instala, mide y se da de alta: en la app,
   Administración → máquinas → «maquina-i-0…» **En línea**, con su capacidad. Las cámaras pasan a ella.
4. Al cerrar el local (más 15 minutos), sus cámaras dejan de pesar, el servidor vacía la máquina y el encendedor la
   termina. 20 minutos antes de abrir, la vuelve a crear.

**Si algo no anda**, entrá a la máquina: EC2 → la instancia → **Connect → Session Manager** →

```bash
sudo tail -50 /var/log/vigia-instalar.log            # la instalación (y cuántas cámaras midió)
sudo journalctl -u 'vigia-maquina@*' -n 100 --no-pager   # el análisis
sudo journalctl -u vigia-escalador -n 100 --no-pager     # en el encendedor
```

## 4. Cambiar algo

CloudFormation → la pila → **Update** → Use current template → cambiar parámetros (tope, tipos, a demanda). Las
máquinas nuevas toman los cambios; las que están prendidas siguen hasta que se terminen.

## 5. Borrar todo

CloudFormation → la pila → **Delete**. Borra máquinas, grupo, encendedor, roles y grupo de seguridad (el parámetro
con la clave queda: borralo a mano si no se usa más). Antes, en la app, marcá «✘ No analizar» en las cámaras que
estaban en la nube (si no, quedan esperando una máquina).

## Detalles

- **Por qué Graviton:** las medidas de octubre dieron que el «hyperthreading» casi no suma para este trabajo; en
  Intel y AMD de AWS (c6a, c7i, c5a) cada vCPU es medio núcleo, en Graviton es uno entero. `c7a` (AMD sin
  hyperthreading) no existe en São Paulo.
- **Máquinas con placa de video** (g4dn, g6: estimado 40-80 cámaras por máquina) necesitan otra imagen (Deep
  Learning AMI con el driver de NVIDIA) y x86: hoy se arman a mano con `deploy/maquina-analisis/README.md`.
  **Sin probar.**
- **Sin el grupo de Auto Scaling:** el encendedor también maneja máquinas EC2 sueltas (`VIGIA_NUBE=aws`, con
  `VIGIA_SPOT=si`, `VIGIA_TIPOS=c7g.xlarge,c6g.xlarge` y `VIGIA_RESPALDO=si` para crear a demanda si no hay Spot).
- **Las máquinas viejas** (terminadas, que no se ven hace 3 días) se borran solas de la lista de la app.
- **La plantilla pasa `cfn-lint` pero todavía no se creó en una cuenta real.** Primera prueba: tope 1, un cliente.
