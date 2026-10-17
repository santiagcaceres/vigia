# Vigía: detección de hurtos en cámaras Hikvision

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

   Para generar un texto al azar: en el navegador, F12 → Consola, pegar
   `crypto.randomUUID() + crypto.randomUUID()` y Enter.
3. **Deploy.** Al terminar, entrá a `https://tu-proyecto.vercel.app` con `admin` y la clave
   `ADMIN_PASSWORD`.
4. Si falta alguna variable, `https://tu-proyecto.vercel.app/api/health` dice cuál.

> El plan Hobby de Vercel es solo para uso personal **no comercial**. Para vender el servicio
> se necesita Vercel Pro.

### Paso 3. Preparar el NVR Hikvision (en cada cliente)

Desde la web del NVR o desde iVMS-4200 (Configuración remota):

1. **Crear un usuario solo para Vigía**, con permiso de "Vista en vivo" y "Reproducción". No uses `admin`.
2. **Activar RTSP** y verificar el puerto (554 por defecto). Habilitar ISAPI si aparece en
   *Red > Avanzado > Integración*.
3. **Configurar el stream secundario en H.264** (no H.265), con 640x360 o 704x576 y 10 a 15 fps.
   **Desactivar H.264+ / H.265+**.
4. **Anotar el canal de cada cámara en el NVR:** la cámara 1 es el canal 1, la 2 el canal 2, etc.

### Paso 4. Cargar cliente, agente y cámaras en el panel

1. En **Administración**:
   - Crear el **cliente**.
   - Crear un **agente** para ese cliente. Copiá el token: se muestra una sola vez.
2. En **Cámaras > Agregar cámara**: IP del NVR, canal, usuario y clave (los mismos datos que usa iVMS).

### Paso 5. Instalar el agente en la PC del local

**Windows:**
1. Instalar [Python 3.12](https://www.python.org/downloads/) y marcar "Add python.exe to PATH".
2. Bajar este repositorio (botón **Code → Download ZIP** en GitHub) y descomprimirlo, por ejemplo
   en `C:\vigia`.
3. Ejecutar `agent\instalar_windows.bat`. Instala las dependencias y ffmpeg.
4. Probar la conexión con el NVR:

   ```bat
   .venv\Scripts\python -m agent.hikvision --host 192.168.1.64 --user vigia --password ****
   ```

5. Iniciar el agente:

   ```bat
   .venv\Scripts\python -m agent --server https://tu-proyecto.vercel.app --token TOKEN
   ```

6. Para que arranque solo con Windows: Programador de tareas → Crear tarea → "Al iniciar el
   sistema", con la misma línea del paso 5 y "Reiniciar si falla".

**Linux:** ver `agent/vigia-agent.service`.

En menos de un minuto la cámara aparece **En línea** en el panel y llega una imagen para
dibujar las zonas.

**Capacidad:** el detector de personas tardó unos 120 ms por cuadro en la CPU de prueba. Una PC
de oficina maneja de 2 a 4 cámaras a 5 cuadros por segundo. Para más cámaras, bajar a 3 cuadros
por segundo, usar una PC con GPU NVIDIA, o exportar el modelo a 416 con `tools/export_model.py`.

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
1. En iVMS-4200, exportá la grabación en **MP4**.
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
python -m pytest                                        # 52 pruebas
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
agent/             agente para la PC del local (cámaras + videos importados)
models/            detector de personas (ONNX) para el agente
tools/             demo y exportación de modelos
tests/             pruebas
```
