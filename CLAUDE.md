# Vigía — contexto del proyecto

Sistema comercial que vigila cámaras **Hikvision y Dahua**, detecta comportamiento sospechoso (posible
hurto), guarda el clip y lo manda a revisión humana. Multi-cliente. Todo en español:
código, comentarios, interfaz y trato con el usuario.

## Estado: EN PRODUCCIÓN

| Pieza | Dónde |
|---|---|
| Panel + API | https://vigia-three-ashy.vercel.app (proyecto Vercel `vigia`, región gru1) |
| Base + archivos | Supabase proyecto **CAMARAS**, ref `cnwhpaqvdjabntteacrj` (sa-east-1 no: us-east-1) |
| Repo | github.com/santiagcaceres/vigia (privado) |
| Dueño | Santiago Cáceres, Uruguay (America/Montevideo) |

**Secretos:** viven en las variables de entorno de Vercel (`DATABASE_URL`, `SUPABASE_URL`,
`SUPABASE_SECRET_KEY`, `SECRET_KEY`, `ADMIN_PASSWORD`, `CRON_SECRET`, `PUBLIC_URL`).
Nunca los pongas en el repo. El usuario tiene la clave de `admin` y los tokens de agente.
`SECRET_KEY` cifra las claves de las cámaras: si se pierde, hay que recargarlas.

**Ya creado en la base:** cliente `Casa (prueba)` (id 1); agente `PC de casa` (id 1, de local);
agente `PC central` (id 2, `client_id` NULL); cámara `Demo noticiero (video)` (id 1, deshabilitada,
asignada al agente central, con zonas y reglas para probar con videos subidos).
En CAMARAS quedaron tablas de una app anterior (`profiles`, `videos`, `video_batches`,
`video_detections`, y `clients_v0_old` renombrada); son inertes, no las usa Vigía.

## Arquitectura

```
DVR Hikvision --RTSP--> agente (Python) --HTTPS--> servidor FastAPI en Vercel --> Supabase
                                                          |
                                      empleados revisan y etiquetan (entrena)
                                                          |
                                      "Avisar al cliente" -> link seguro al clip
```

- **server/**: FastAPI, sin estado (Vercel). `index.py` exporta `app`. Postgres por el
  *transaction pooler* (6543, `NullPool`, `prepare_threshold=None`). Archivos a Supabase
  Storage con URLs firmadas (Vercel no acepta subidas > 4,5 MB).
- **agent/**: corre en una PC. Lee RTSP, detecta, arma clips, sube con reintentos (outbox en
  disco). También analiza los videos que se suben para entrenar (Vercel no procesa video).
- **agent/cameras.py**: capa común de marcas. `BaseCamera` (datos de conexión, auth, horario),
  `build_camera(cfg)` elige por `brand`, y `download_nvr_clip` / `test_stream` sirven para
  cualquier marca. `agent/hikvision.py` (RTSP `/Streaming/Channels/<canal>01|02`, playback
  `/Streaming/tracks/`, ISAPI XML) y `agent/dahua.py` (RTSP `/cam/realmonitor?channel=N&subtype=0|1`,
  playback `/cam/playback?...starttime=AAAA_MM_DD_HH_MM_SS`, API CGI `clave=valor`).
  Agregar una marca = una clase con `live_url`/`playback_url` + `probe`, y sumarla a la fábrica.
- **core/**: detector YOLO (ONNX, CPU), tracker, zonas/reglas, scorer, clips. El servidor solo
  importa `core.rules`, `core.scorer` (y sus deps `geometry`, `tracker`): **no** mete OpenCV ni
  onnxruntime en el build de Vercel. No romper eso.
- **public/**: panel (SPA sin framework) + `aviso.html`, la página pública del clip.

### Modo alarmas de Hik-Partner Pro (el camino elegido, oct 2026)
Reemplaza a los empleados del socio que hoy miran los clips en el portal de HPP. Flujo:
alarma del DVR → HPP → **webhook** `POST /api/hpp/webhook` (firmado HMAC) → tabla `alarms`
(agrupa disparos seguidos de la misma cámara) → la PC central la pide (`/api/agent/alarms/next`),
**baja solo ese tramo** del DVR, lo analiza con `core/clip_analysis.py` → sin personas: se
descarta sola; con persona: evento `source="alarma"` con clip → empleados etiquetan → modelo.
Cámara con `mode="alarmas"`, `hpp_serial` (serie del DVR en HPP), `channel` y `alarm_cfg`
(tipos de alarma, segundos antes/después). El agente no mira en vivo esas cámaras.
- **La OpenAPI de HPP (guía V2.15.500, copia en github SYSCOM-Labs/API-DOCS) da alarmas y fotos,
  NO video.** Video = HPNetSDK con clave "VSaaS" que se pide por mail a Hikvision (pendiente).
  Hasta entonces el tramo se baja **por VPN** del DVR (`download_nvr_clip`). Para sumar HPNetSDK:
  otro `ClipFetcher` en `agent/alarms.py`.
- API key de HPP: el socio la saca solo en el portal (Site & Device → My Service → API
  Integration). Variables `HPP_APP_KEY`, `HPP_SECRET_KEY`, `HPP_WEBHOOK_SECRET` (opcional).
- El formato exacto del cuerpo del webhook no está documentado: se parsea tolerante y se guarda
  `raw`. **Verificar con la primera alarma real** (logs de Vercel + `alarms.raw`).

### Modo central
`Agent.client_id` NULL = agente **central**: corre en la PC de la empresa y atiende las cámaras
que se le asignen, **de cualquier cliente**, llegando a cada DVR por **VPN** (Tailscale /
WireGuard; cajita GL.iNet ~US$30 si el router no soporta). Evita poner una PC por local.
`agent_owns()` / `agent_cameras_q()` deciden qué ve cada agente. Cada cámara recibe el
`filter_model` de **su** cliente, así un agente corre varios modelos a la vez.

### Roles
- **admin** (habrá 3: Santiago, el socio y uno más): crean usuarios y agentes, conectan HPP.
- **entrenador** (en el panel se llama **"Empleado"**; el valor interno sigue siendo `entrenador`):
  revisan y etiquetan, **crean clientes y configuran cámaras** (alta, edición, zonas). No borran
  cámaras ni ven usuarios/agentes.
- **cliente**: el dueño del comercio, solo ve lo suyo.

### Flujo de aviso al cliente (clave del producto)
El clip llega al operador → empleados revisan y etiquetan (1 hurto, 2 sospechoso, 3 normal,
4 falso positivo) → eso entrena el modelo → **una persona** decide avisar al cliente. **Nunca
automático**: evita acusar por un falso positivo. El texto por defecto dice "posible hurto", no
una acusación. Genera un link privado que vence (14 días). Canales: portal (siempre, gratis),
Telegram, WhatsApp (Twilio) y email (Resend), según lo configurado por cliente.
La alerta automática de "hay clip para revisar" va al **operador**, no al cliente.

## Decisiones importantes (no re-litigar)

- **Las apps de nube NO sirven** para leer video: Hik-Connect (Hikvision) y DMSS / SmartPSS Lite
  (Dahua) son visores cerrados, no exponen el stream. Sí sirven para configurar los equipos a
  distancia (Hik-Partner Pro / DoLynk Care).
- **Hik-Partner Pro** (el cliente ya lo usa) sirve para **configurar los DVR a distancia** y su
  OpenAPI REST da **las alarmas** (webhook) pero **no video** (confirmado en la guía V2.15.500:
  live/playback/descarga = HPNetSDK con clave VSaaS por mail; sacaron `video/by/time`).
  Dahua no tiene equivalente implementado: sus cámaras se analizan en vivo por RTSP.
- **La cámara analógica va al DVR**; el agente se conecta al DVR, no a la cámara.
- **Dahua y la hora del playback:** algunos equipos interpretan starttime/endtime en hora local
  y otros en UTC. Se elige por cámara con `playback_time`; si el clip llega corrido, probar la otra.
- **Nunca** abrir puertos del DVR a internet: siempre VPN.
- El detector es **YOLO11n, licencia AGPL-3.0**. Para vender hay que comprar licencia
  Enterprise de Ultralytics o cambiar a un modelo Apache-2.0 (YOLOX, RT-DETR). **Pendiente.**

## Limitaciones honestas (decirlas siempre, no prometer de más)

- **No detecta ocultamiento** (guardar algo en la ropa/mochila). Marca patrones por zonas y
  tiempos. Cada evento es un indicio para revisar, **no una prueba**.
- **La precisión nunca se midió con video real.** Es lo primero del piloto.
- Probado con un video real de noticiero: detecta a la persona en el 86 % de los cuadros y
  dispara "merodeo", pero no "ve" el gesto del hurto.
- Capacidad: ~120 ms/cuadro en CPU → 2–4 cámaras por PC a 5 fps. Medir antes de prometer cupo.
- Ancho de banda del local: 0,25–0,5 Mbps de subida por cámara, 24 h.

## Desarrollo

```bash
python -m pytest -q                 # 79 tests (24 se saltean sin Postgres)
VIGIA_TEST_PG="postgresql://postgres@127.0.0.1:54329/vigia_test" python -m pytest -q   # Postgres
python -m agent --server https://vigia-three-ashy.vercel.app --token TOKEN   # agente
python -m agent.hikvision --host IP --user vigia --password ****             # probar DVR Hikvision
python -m agent.dahua     --host IP --user vigia --password ****             # probar DVR Dahua
```

- Sin `DATABASE_URL` usa SQLite local; sin Supabase guarda archivos en disco. Así corre local.
- **Deploy:** `git push` + lanzar el deploy en Vercel (el proyecto no tiene auto-deploy por git
  configurado; se dispara con la API/MCP de Vercel o desde el panel de Vercel).
- Las tablas se crean solas (`create_all`) y quedan con RLS activo. Las **columnas nuevas** también
  se agregan solas al arrancar (`add_missing_columns` en `server/db.py`); renombrar o cambiar el
  tipo de una columna sí requiere migración manual.
- Ojo: `requirements.txt` es del servidor; `requirements-agent.txt` del agente (OpenCV, onnxruntime).

## Próximos pasos

1. **Conectar HPP:** el socio saca la API key, se carga en Vercel, "Conectar" en Alarmas HPP, y
   revisar la primera alarma real (formato del webhook, hora del DVR, serie/canal).
2. **Video de las alarmas:** pedir a Hikvision la clave VSaaS/HPNetSDK (sin VPN), o poner VPN en
   cada local (router con WireGuard/Tailscale o cajita GL.iNet ~US$30).
3. **Medir:** % de alarmas descartadas solas y cuántas con persona eran hurto (con las etiquetas).
4. **Detector de ocultamiento** (mano góndola→cuerpo): necesita 30–50 videos etiquetados.
5. **Licencia del detector** (AGPL) antes de cobrar.
6. **Preguntas a Hikvision:** ¿requisitos y costo de la clave VSaaS / HPNetSDK? ¿permite descargar
   la grabación por horario (o el clip de alarma) por equipo? ¿corre en Windows/Linux? ¿UY/AR?
7. Antes de cobrar: Vercel Pro (el plan gratis es no comercial) y Supabase Pro (el gratis limita
   los archivos a 50 MB).

## Estilo

Español rioplatense, claro y sin jerga técnica innecesaria: el usuario es desarrollador pero
el producto lo usan comerciantes. Explicar los límites con honestidad antes que vender humo.
