# Vigía: detección de hurtos en cámaras Hikvision y Dahua

Sistema para varios clientes, con varias cámaras por cliente. Mira las cámaras en tiempo real y,
cuando detecta un comportamiento sospechoso, guarda el clip y avisa. Incluye un panel donde una
persona revisa y etiqueta los clips, y con esas etiquetas el sistema aprende a separar las
alertas reales de las falsas.

```
 LOCAL DEL CLIENTE                               VERCEL                        SUPABASE
 ┌───────────────────────────┐                   ┌────────────────────┐        ┌───────────────────┐
 │ NVR / cámaras Hikvision   │                   │ Panel web (public/)│        │ Base de datos     │
 │   │ RTSP                  │  HTTPS saliente   │ API (FastAPI)      │◄──────►│ (Postgres)        │
 │   ▼                       │ ────────────────► │  clientes, cámaras │        │                   │
 │ Agente Vigía (una PC)     │  eventos, estado  │  eventos, etiquetas│        │ Storage (privado) │
 │  • detecta personas       │ ◄──────────────── │  entrenamiento     │        │  fotos, clips,    │
 │  • aplica las reglas      │  configuración    │  alertas Telegram  │        │  videos           │
 │  • baja el clip del NVR   │                   └────────────────────┘        └───────────────────┘
 │  • analiza videos subidos │  clips y videos: directo a Supabase con URL firmada ──────────▲
 └───────────────────────────┘
```

- **El agente** corre en una PC dentro de la red del cliente (puede ser la PC con Windows donde
  está iVMS-4200). Solo hace conexiones **salientes**: no hay que abrir puertos en el router.
- **iVMS-4200 sigue funcionando igual.** Vigía no pasa por iVMS: se conecta al NVR con RTSP,
  igual que iVMS.
- **Vercel** publica el panel y la API. Como Vercel no puede procesar video, los videos que se
  suben para entrenar los analiza el agente de ese cliente.
- **Supabase** guarda la base de datos y los archivos. Las tablas quedan cerradas a la API
  pública de Supabase (RLS activado, sin políticas): solo el servidor puede leerlas.

## Modo alarmas: el grabador avisa, Vigía analiza solo ese tramo (lo recomendado)

Es el modo más barato y el que reemplaza el trabajo de "mirar los clips de las alarmas". Hay dos
formas de enterarse de una alarma; las dos terminan en el mismo análisis:

- **Alarmas del propio grabador** (Hikvision y Dahua, sin nube ni API key, **el modo por defecto**):
  la PC central se queda conectada al grabador por la VPN y el equipo le avisa cada alarma
  (movimiento, intrusión, cruce de línea). Solo hay que activar esas detecciones en el grabador.
- **Alarmas por Hik-Partner Pro** (solo Hikvision): HPP avisa por webhook. Sirve cuando no se
  quiere que la PC esté conectada todo el tiempo al equipo; igual necesita VPN para bajar el video.

```
DVR del cliente ──alarma (movimiento / intrusión / línea)──► Hik-Partner Pro ──webhook──► Vigía (Vercel)
                                                                                            │
PC central ◄──── "bajá 30 s de la cámara 2 del DVR E44…" ───────────────────────────────────┘
   │  baja SOLO ese tramo del DVR (VPN), lo analiza
   ├─ sin personas ─► se descarta sola (nadie la mira)
   └─ con persona  ─► evento con clip ─► empleados etiquetan ─► entrena el modelo de ese cliente
                                                     └─► una persona decide "Avisar al cliente"
```

- **El DVR ya hace de primer filtro**: solo se analizan los segundos alrededor de cada alarma, no
  24 h de video. Una PC central alcanza para muchos locales y el consumo de internet del local es
  mínimo (solo el tramo de cada alarma).
- **Lo que más trabajo ahorra**: las alarmas de movimiento sin ninguna persona (luces, sombras,
  insectos) se descartan solas.
- **Configuración por cámara**: qué alarmas del DVR se analizan, cuántos segundos antes y después,
  zonas (góndola, caja, salida) y reglas. Cada cliente tiene su propio modelo entrenado.

**Lo que da y lo que no da Hik-Partner Pro** (guía oficial *OpenAPI V2.15.500*):
- Da **las alarmas** (equipo, canal, hora, tipo) por webhook, gratis, con la API key que se saca
  sola en el portal: *Site & Device → My Service → API Integration*.
- **No da el video.** Reproducir y descargar video requiere el kit HPNetSDK con una clave "VSaaS"
  que se pide **por mail a Hikvision** (pendiente). Mientras tanto, la PC central baja el tramo de
  video **del DVR por VPN** (ver Paso 5b). Cuando Hikvision dé la clave, se agrega esa fuente en
  `agent/alarms.py` (`ClipFetcher`) sin cambiar nada más.

**Puesta en marcha:**
1. En Vercel, cargar `HPP_APP_KEY`, `HPP_SECRET_KEY` (de *API Integration* en HPP) y, opcional,
   `HPP_WEBHOOK_SECRET` (8 a 32 letras y números; si no, se usa la SecretKey). `PUBLIC_URL` tiene
   que ser la dirección HTTPS del panel. Volver a desplegar.
2. En el panel, **Alarmas HPP → Ver equipos de HPP**: lista los DVR y canales de la cuenta, con un
   link "cargar" que crea la cámara con la serie y el canal ya puestos. Completar la IP de VPN del
   DVR, usuario y clave, y el **agente = el central**.
3. **Alarmas HPP → Conectar**: registra el webhook en HPP y suscribe los DVR cargados.
4. En el DVR (desde HPP, configuración remota): activar la detección de movimiento o, mejor,
   **intrusión / cruce de línea** en las cámaras que miran góndolas, con "Notificar al centro de
   vigilancia" activado. Si el DVR es AcuSense (filtra personas/vehículos), las alarmas de
   vehículos se descartan solas.
5. Dibujar las zonas de cada cámara (Paso 6).

En **Alarmas HPP** se ve cada alarma y qué pasó con ella (descartada, con persona, error, cámara no
cargada) y cuántas se descartaron solas en la semana.

## Puesta en marcha

### Paso 1. Supabase (5 min)

1. En <https://supabase.com/dashboard>, crear un proyecto (**New project**):
   - **Región:** São Paulo (la más cercana a Argentina).
   - **Clave de la base:** anotala. Que tenga **solo letras y números**, así no hay que
     codificarla en la URL.
2. Copiar estos tres datos:
   - **DATABASE_URL:** botón **Connect** → **Connection string** → modo **Transaction pooler**
     (puerto **6543**). Reemplazá `[YOUR-PASSWORD]` por tu clave.
   - **SUPABASE_URL:** *Project Settings → Data API*, la "Project URL" (`https://xxxx.supabase.co`).
   - **SUPABASE_SECRET_KEY:** *Project Settings → API Keys*, la **secret key** (`sb_secret_…`).
     Si tu proyecto muestra claves "legacy", usá **service_role**. **Nunca** la publiques ni la
     pegues en el panel.
3. **No hace falta crear tablas ni buckets:** el servidor los crea solo la primera vez.
4. **Límite de archivos:** en el plan gratis cada archivo puede pesar hasta 50 MB. En Pro se sube
   en *Storage → Settings → Global file size limit*.

### Paso 2. Vercel (10 min)

1. En <https://vercel.com/new>, importar el repositorio de GitHub `vigia`. Vercel detecta
   FastAPI solo. No cambies "Build" ni "Output".
2. Antes de **Deploy**, en **Environment Variables** cargar:

   | Variable | Valor |
   |---|---|
   | `DATABASE_URL` | La del paso 1 (pooler, puerto 6543) |
   | `SUPABASE_URL` | La del paso 1 |
   | `SUPABASE_SECRET_KEY` | La secret key del paso 1 |
   | `SECRET_KEY` | Texto al azar (ver abajo). **No lo pierdas**: cifra las claves de las cámaras |
   | `ADMIN_PASSWORD` | La clave con la que vas a entrar como `admin` |
   | `CRON_SECRET` | Otro texto al azar (protege la limpieza diaria) |
   | `TELEGRAM_BOT_TOKEN` | Opcional. Para avisar a tu equipo y/o a los clientes por Telegram |
   | `OPERATOR_TELEGRAM_CHAT_ID` | Opcional. Chat de **tu equipo** que recibe "hay clip para revisar" |
   | `TWILIO_ACCOUNT_SID` / `TWILIO_AUTH_TOKEN` / `TWILIO_WHATSAPP_FROM` | Opcional. Para avisar a clientes por WhatsApp |
   | `RESEND_API_KEY` / `EMAIL_FROM` | Opcional. Para avisar a clientes por email |
   | `HPP_APP_KEY` / `HPP_SECRET_KEY` | Opcional. API key de Hik-Partner Pro para recibir las alarmas de los DVR |
   | `HPP_WEBHOOK_SECRET` | Opcional. Clave para firmar el webhook de HPP (si no, se usa la SecretKey) |
   | `PUBLIC_URL` | La dirección HTTPS del panel (para los links de aviso y el webhook de HPP) |

   Para generar un texto al azar: en el navegador, F12 → Consola, pegar
   `crypto.randomUUID() + crypto.randomUUID()` y Enter.
3. **Deploy.** Al terminar, entrá a `https://tu-proyecto.vercel.app` con `admin` y la clave
   `ADMIN_PASSWORD`.
4. Si falta alguna variable, `https://tu-proyecto.vercel.app/api/health` dice cuál.

> El plan Hobby de Vercel es solo para uso personal **no comercial**. Para vender el servicio
> se necesita Vercel Pro.

### Paso 3. Preparar el grabador (en cada cliente)

> La guía completa para conectar el grabador de un cliente **a través de su router** (qué pedirle,
> usuario `vigia`, puertos en el router de Antel, DDNS, prueba en VLC, carga en el panel, seguridad y
> fallas comunes) está en el panel: menú **Guía** (`/guia.html`). Lo que sigue es el resumen.


Vigía funciona con **Hikvision** y **Dahua**. Lo más simple es entrar por el navegador a
`http://IP-DEL-GRABADOR`; también sirve iVMS-4200 (Hikvision) o SmartPSS Lite (Dahua).

1. **Crear un usuario solo para Vigía**, con permiso de "Vista en vivo" y "Reproducción". No uses `admin`.
2. **Activar RTSP** y verificar el puerto (554 por defecto).
   - Hikvision: habilitar ISAPI si aparece en *Red > Avanzado > Integración*.
   - Dahua: *Red > Puerto* (RTSP). La API CGI viene habilitada.
3. **Configurar el stream secundario (sub) en H.264** (no H.265), con 640x360 o 704x576 y 10 a 15 fps.
   **Desactivar H.264+ / H.265+ (Dahua: "Smart Codec")**.
4. **Anotar el canal de cada cámara:** la cámara 1 es el canal 1, la 2 el canal 2, etc.

Probar la conexión desde la PC del agente:

```bash
python -m agent.hikvision --host IP --user vigia --password ****   # Hikvision
python -m agent.dahua     --host IP --user vigia --password ****   # Dahua
```

**Las apps de nube no sirven para esto.** Ni Hik-Connect (Hikvision) ni DMSS/SmartPSS (Dahua)
entregan el video a software de terceros: son visores. Vigía se conecta al grabador, en la red
del local o por VPN. Esas apps sí sirven para *configurar* los equipos a distancia.

### Paso 4. Cargar cliente, agente y cámaras en el panel

1. En **Administración**:
   - Crear el **cliente**.
   - Crear un **agente** para ese cliente. Copiá el token: se muestra una sola vez.
2. En **Cámaras > Agregar cámara**: **marca** (Hikvision o Dahua), IP del grabador, canal, usuario y clave
   (los mismos datos que usás en iVMS-4200 o SmartPSS Lite).

### Paso 5. Instalar el agente (en tu PC central)

**Windows, opción fácil (sin instalar Python):** descargar `VigiaAgente-windows.zip` desde
GitHub → **Actions → "Agente para Windows"** (artefacto de la última corrida) o desde
**Releases**. Descomprimir, abrir `VigiaAgente.exe`, Enter para el servidor y pegar el token.
Queda guardado en `agente.env` al lado del .exe. Si Defender lo frena: "Más información →
Ejecutar de todas formas". Abre además la ventana **"Vigia - PC central"**: mosaico con las cámaras y
los recuadros de la IA en tiempo real (`--sin-ventana` para un servidor sin pantalla). Para generar un .exe nuevo: Actions → Run workflow (o `git tag
agente-v1.0.0 && git push --tags`, que además lo publica en Releases).

**Windows, con Python:**
1. Instalar [Python 3.12](https://www.python.org/downloads/) y marcar "Add python.exe to PATH".
2. Bajar este repositorio (botón **Code → Download ZIP** en GitHub) y descomprimirlo, por ejemplo
   en `C:\vigia`.
3. Doble clic en `agent\iniciar_windows.bat`: la primera vez instala las dependencias y pide el
   servidor y el token (quedan en `agente.env`). Las siguientes veces arranca directo.
4. Probar la conexión con el grabador, si hace falta:

   ```bat
   .venv\Scripts\python -m agent.hikvision --host 203.0.113.25 --port 38080 --user vigia --password ****
   ```

5. Para que arranque solo con Windows: Programador de tareas → Crear tarea → "Al iniciar
   sesión", programa `VigiaAgente.exe` (o `agent\iniciar_windows.bat`) y "Reiniciar si falla".

**Linux:** ver `agent/vigia-agent.service`.

En menos de un minuto la cámara aparece **En línea** en el panel y llega una imagen para
dibujar las zonas.

**Capacidad:** el detector de personas tardó unos 120 ms por cuadro en la CPU de prueba. Una PC
de oficina maneja de 2 a 4 cámaras a 5 cuadros por segundo. Para más cámaras, bajar a 3 cuadros
por segundo, usar una PC con GPU NVIDIA, o exportar el modelo a 416 con `tools/export_model.py`.

### Paso 5b. Modo central: una sola PC para todos los clientes (recomendado)

En vez de una PC por local, **tu PC central corre un único agente** que atiende las cámaras de
todos los clientes. En cada local no va ninguna computadora: solo hace falta que tu PC llegue al
DVR del cliente por **VPN**.

**1. Crear el agente central.** Administración → Agentes → Tipo **"Central"** → Crear. Copiá el token.

**2. En tu PC central** (Windows): los mismos pasos del Paso 5, con el token del agente central.
Instalá además [Tailscale](https://tailscale.com/download) (VPN; gratis hasta 100 equipos) e iniciá
sesión.

**3. En cada local, llegar al DVR por VPN.** Elegí una:
- **Router del cliente con VPN (US$0):** muchos routers (MikroTik, Ubiquiti, TP-Link Omada,
  GL.iNet, OpenWrt) traen WireGuard o Tailscale. Se configura como *subnet router* o se mapea el
  puerto 554 del DVR.
- **Cajita VPN (~US$30–40, una vez):** un mini router **GL.iNet** (p. ej. GL-MT3000 o GL-SFT1200)
  enchufado por cable al router del cliente, con Tailscale activado en su firmware y un
  **mapeo de puertos** 554 (RTSP) y 80 (ISAPI) → IP del DVR. En Vigía la cámara se carga con
  **host = IP de Tailscale de la cajita**. Así no importa que dos locales usen la misma red
  (192.168.1.x), porque cada cajita tiene su propia IP de VPN.
- **Nunca** abrir los puertos del DVR a internet: los DVR expuestos se atacan constantemente.

**4. Preparar el grabador sin ir al local:** con **Hik-Partner Pro** (Hikvision) o **DoLynk Care**
(Dahua) entrás a la configuración remota y hacés el Paso 3 (usuario `vigia`, RTSP, flujo
secundario H.264) desde tu oficina. También podés hacerlo por la web del equipo una vez
montada la VPN.

**5. Cargar la cámara** en el panel con el **agente = el central**, host = IP de VPN, canal, usuario
`vigia`. En un minuto aparece En línea.

**Ancho de banda:** cada cámara consume ~0,25–0,5 Mbps de **subida** del local (flujo secundario),
las 24 h. Con 4 cámaras, 1–2 Mbps (unos 400–500 GB/mes). Fibra: sin problema. Internet móvil con
tope de datos: no sirve.

**Capacidad de la PC central:** el detector tarda ~120 ms por cuadro en CPU. A 5 cuadros/seg por
cámara, una PC de oficina maneja 2–4 cámaras; con más locales, bajar a 3 cuadros/seg o usar una
placa NVIDIA (decenas de cámaras). **Medilo** con tus cámaras antes de prometer cupo.

**Futuro sin VPN:** si Hikvision te aprueba como Technology Partner, su protocolo **ISUP** permite
que el DVR se conecte *saliente* a tu PC (sin VPN ni cajita). Requiere su SDK; se evalúa cuando
tengas el acceso.

### Paso 6. Zonas

En **Cámaras > Zonas y reglas**, dibujá sobre la imagen las zonas:
- **Góndola:** donde está la mercadería.
- **Caja:** donde se paga.
- **Salida:** la puerta.
- **Restringida:** depósito, detrás del mostrador.

Se evalúan **los pies** de la persona, así que las zonas van **sobre el piso**. La regla de
salida sin pago necesita góndola, caja y salida vistas por la misma cámara.

## Qué detecta y qué no

| Regla | Qué marca |
|---|---|
| Salida sin pago | Estuvo en una góndola y salió sin pasar por la caja |
| Zona restringida | Entró al depósito, detrás del mostrador, etc. |
| Merodeo | Permanencia muy larga frente a una góndola |
| Fuera de horario | Persona en el local con el negocio cerrado |

**No detecta ocultamiento de mercadería** (guardar algo en la ropa o la mochila). Cada evento es
un indicio para revisar, no una prueba. La precisión todavía **no se midió con video real**:
es lo primero que hay que hacer en el piloto (ver "Cómo medir").

## Usuarios y roles

| Rol | Qué hace |
|---|---|
| **Administrador** | Crea usuarios (otros administradores, empleados, clientes) y agentes; conecta Hik-Partner Pro. |
| **Empleado** | Revisa y etiqueta los clips; crea clientes y configura sus cámaras (alta, edición, zonas). |
| **Cliente** | El dueño del comercio: solo ve sus eventos y los avisos que le mandaron. |

Los usuarios se crean en **Administración → Usuarios**. Ahí también se le cambia la clave a alguien
o se lo desactiva. Cada uno puede cambiar su propia clave con **Mi clave**.

## Guía para la persona que entrena

**Revisar y etiquetar (todos los días):** mirar el clip y apretar una tecla:

| Tecla | Etiqueta | Cuándo |
|---|---|---|
| 1 | Hurto | Se ve el hurto |
| 2 | Sospechoso | No es seguro, pero merece la alerta |
| 3 | Normal | Persona real sin problema (repositor, cliente que salió sin comprar) |
| 4 | Falso positivo | Error de detección: sombra, reflejo, no es una persona |

`S` saltea, `R` repite el clip. Si dudás, saltealo.

**Videos (grabaciones viejas):**
1. Exportá la grabación en **MP4** (iVMS-4200 en Hikvision, SmartPSS Lite en Dahua).
2. Subila eligiendo la cámara y la hora de inicio. El agente de ese cliente la analiza; tiene
   que estar encendido.
3. Marcá los hurtos reales que viste: el sistema te dice cuántos detectó y cuántas falsas
   alarmas dio por hora.
4. Etiquetá los eventos que generó.

**Entrenamiento:**
- **Cuándo:** con 30 etiquetas o más (8 de alerta real y 8 sin importancia), hacé clic en
  **Entrenar**.
- **Activar:** solo si el resultado no pierde casos reales que te importen.
- **Efecto:** con el modelo activo, los eventos poco probables se guardan igual pero no
  generan alerta.

## Avisar al cliente ("nos robaron")

El ciclo pensado es: el clip sospechoso llega a **tu** panel → tu equipo lo revisa y etiqueta
(eso entrena al agente) → si es un hurto, **una persona** decide avisarle al cliente. Nunca se
le avisa al cliente de forma automática: así un falso positivo no termina en una acusación.

**Cómo se avisa:** en el detalle de un evento con clip, botón **"Avisar al cliente"**. Se puede
editar el texto (por defecto dice "posible hurto", no una acusación) y elegir los canales. Se
genera un **link privado y temporal** (14 días por defecto) a una página donde el dueño ve el
clip desde el celular, sin instalar nada ni iniciar sesión.

**Canales** (se usan los que el cliente tenga cargados en Administración):
- **Portal / link:** siempre disponible, gratis. El cliente también ve sus avisos entrando al
  panel como usuario "cliente" (menú **Incidentes**).
- **Telegram:** gratis. Requiere `TELEGRAM_BOT_TOKEN` y el chat del cliente.
- **WhatsApp:** lo mejor para un comercio. Requiere cuenta de WhatsApp Business API (Twilio) y
  las variables `TWILIO_*`.
- **Email:** requiere una cuenta de Resend (`RESEND_API_KEY`, `EMAIL_FROM`).

**Avisos a clientes** (menú del equipo): lista todo lo enviado, si el cliente ya lo abrió, y
permite **revocar** un link.

> El aviso interno "hay un clip para revisar" le llega a **tu equipo**, no al cliente, si cargás
> `OPERATOR_TELEGRAM_CHAT_ID` (o `OPERATOR_WEBHOOK_URL`).

## Cómo medir antes de prometer

1. Para cada tipo de local, subí grabaciones con hurtos conocidos y también horas normales.
2. Marcá los hurtos reales y mirá **hurtos detectados** y **falsas alarmas por hora**.
3. Si hay muchas falsas alarmas, primero ajustá zonas y tiempos ("Volver a analizar"). Después
   etiquetá y entrená.

## Seguridad, privacidad y costos

- **Claves de cámaras:** cifradas con `SECRET_KEY`. Nunca vuelven al panel; solo las recibe el
  agente de ese cliente, por HTTPS.
- **Aislamiento entre clientes:** cada agente solo ve las cámaras de su cliente. El rol "cliente"
  solo ve sus eventos y no ve datos de conexión.
- **Archivos privados:** el bucket es privado y los archivos se entregan con URLs firmadas que
  vencen en una hora.
- **Retención:** todos los días se borran los eventos sin etiquetar con más de `RETENTION_DAYS`
  días (60 por defecto). De los etiquetados se borra el video pero se conservan los datos para
  entrenar.
- **Autorizaciones:** necesitás autorización de cada cliente para procesar sus grabaciones, y
  que el contrato contemple usar sus videos para entrenar. Además: cartel de videovigilancia en
  el local, y una alerta nunca alcanza para acusar a nadie.
- **Licencia del detector:** `models/personas.onnx` es YOLO11n de Ultralytics, con licencia
  **AGPL-3.0**. Para un producto comercial cerrado necesitás su licencia Enterprise, o cambiar a
  un detector permisivo como YOLOX (Apache-2.0).
- **Costos de referencia:** Vercel Pro US$20/mes y Supabase Pro US$25/mes, que incluye 100 GB de
  archivos. Cada clip ocupa de 1 a 3 MB.

## Desarrollo y pruebas

```bash
pip install -r requirements.txt -r requirements-agent.txt pytest
uvicorn server.main:create_app --factory --port 8000   # sin variables: SQLite y disco local
python -m pytest                                        # 104 pruebas (unas 30 piden Postgres)
```

- **Postgres:** con `VIGIA_TEST_PG=postgresql://…` las pruebas del servidor corren también
  contra PostgreSQL.
- **Prueba de punta a punta (`tests/test_e2e_rtsp.py`):** simula un NVR Hikvision con mediamtx,
  usando sus mismas rutas RTSP y con usuario y clave. Encima corre el servidor y el agente
  reales. Necesita `mediamtx` y se saltea si no está.
- **Demo sin cámaras:** `python -m tools.demo --server https://tu-proyecto.vercel.app --password <clave>`.

**No probado contra equipos Hikvision reales ni contra un proyecto real de Supabase:** las rutas
siguen la documentación de Hikvision y de la API de Supabase Storage. Lo primero a verificar en
el piloto es la hora de las grabaciones del NVR (opción "Hora de grabación" en la cámara).

## Estructura

```
index.py           punto de entrada de Vercel
server/            API (FastAPI): base de datos, almacenamiento, entrenamiento, alertas
public/            panel web
core/              detección, seguimiento, reglas, modelo de filtro (compartido)
agent/             agente (cámaras en vivo, alarmas de HPP, videos importados)
models/            detector de personas (ONNX) para el agente
tools/             demo y exportación de modelos
tests/             pruebas
```
