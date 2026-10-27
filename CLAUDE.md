# Vigía — contexto del proyecto

Sistema comercial que vigila cámaras **Hikvision y Dahua**, detecta comportamiento sospechoso (posible
hurto), guarda el clip y lo manda a revisión humana. Multi-cliente. Todo en español:
código, comentarios, interfaz y trato con el usuario.

## Estado: EN PRODUCCIÓN

| Pieza | Dónde |
|---|---|
| Panel + API | https://vigia-three-ashy.vercel.app (proyecto Vercel `vigia`, región iad1 desde el 22 oct) |
| Base + archivos | Supabase proyecto **CAMARAS**, ref `cnwhpaqvdjabntteacrj` (sa-east-1 no: us-east-1) |
| Repo | github.com/santiagcaceres/vigia (**público**, verificado el 23 oct: nada de claves ni IPs reales) |
| Dueño | Santiago Cáceres, Uruguay (America/Montevideo) |

**Publicado:** ver `git log origin/main` (el último deploy de producción apunta a `main`). El proyecto Vercel
**no** tiene auto-deploy: después de `git push` hay que lanzar el deploy a mano (MCP de Vercel
con `gitSource` = rama + sha, o desde el panel de Vercel) y verificar `/api/health`.
**Arranque en frío:** `ensure_schema` (server/db.py) guarda `SCHEMA_VERSION` en la tabla
`schema_meta` y solo corre `create_all`/`ALTER` cuando cambia. **Subir `SCHEMA_VERSION` al agregar
tablas o columnas**, si no, en producción no se crean.

**Secretos:** viven en las variables de entorno de Vercel (`DATABASE_URL`, `SUPABASE_URL`,
`SUPABASE_SECRET_KEY`, `SECRET_KEY`, `ADMIN_PASSWORD`, `CRON_SECRET`, `PUBLIC_URL`).
Nunca los pongas en el repo. El usuario tiene la clave de `admin` y los tokens de agente.
`SECRET_KEY` cifra las claves de las cámaras: si se pierde, hay que recargarlas.
**Todavía no cargadas:** `HPP_APP_KEY`, `HPP_SECRET_KEY`, `HPP_WEBHOOK_SECRET` (opcional); sin
ellas la parte de Hik-Partner Pro queda apagada y el resto funciona igual.

**Ya creado en la base:** cliente `Casa (prueba)` (id 1); agente `PC de casa` (id 1, de local);
agente `PC central` (id 2, `client_id` NULL); cámara `Demo noticiero (video)` (id 1, deshabilitada,
asignada al agente central, con zonas y reglas para probar con videos subidos). Usuarios: solo
`admin` (faltan crear los otros 2 administradores y los empleados; lo hace un admin desde el panel).
Tablas `alarms` y columnas `cameras.mode/hpp_serial/alarm_cfg/brand`, `alarms.source` ya existen.
`cameras.mode` se amplió a varchar(20) con migración manual (19 oct).
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
- **public/**: panel (SPA sin framework) + `aviso.html` (página pública del clip) + `guia.html` +
  `prueba.js`. Menú (pedido del dueño, 22 oct): **Resumen, Clientes, Alertas** y Administración
  (solo admin) — **desde el 22 oct el staff usa la app de escritorio; ver más abajo**. "Cámaras" y "Prueba" salieron del menú (siguen por URL: `#/camaras`, `#/prueba`). Las demás vistas siguen existiendo por URL y se llega desde botones:
  `#/revisar` (desde Alertas y Resumen), `#/alarmas` (alarmas del grabador), `#/avisos`, `#/videos`,
  `#/entrenamiento` (links "Más" al pie del Resumen). `#/eventos` fue renombrado `#/alertas`.
  Clientes y cámaras se cachean 20 s en el navegador (`loadClients`/`loadCameras`) para que
  cambiar de pestaña no vuelva a pedir todo.
- **Cámaras por cliente (21 oct, pedido del dueño):** "Cámaras" salió del menú. Desde Clientes se entra a
  `#/cliente/:id` ("Cámaras de <cliente>": miniaturas con la última imagen y estado, botón Agregar cámara)
  y un clic abre la ficha `#/camara/:id` (imagen grande que se refresca cada 30 s, datos de conexión,
  Zonas y reglas, Editar, Probar conexión, Alertas). `#/camaras` sigue existiendo como tabla general (link
  desde el Resumen). El formulario y "Probar conexión" son funciones compartidas (`cameraForm`,
  `probeCamera`). El editor de zonas espera la primera imagen sin recargar (consulta cada 10 s).
- **Vista en vivo liviana (22 oct):** en la ficha de la cámara. El panel late `POST /api/cameras/{id}/live`
  cada 10 s; el servidor guarda `cameras.live_requested_at` y se lo devuelve a la PC central en la respuesta
  de `/api/agent/status` (`live: [ids]`, el agente lo pide cada 10 s). La PC central (`agent/live.py`:
  `LiveStreamer` dentro del `CameraWorker` en modo vivo; `LiveWorker` abre el stream solo mientras dure
  el vivo en modo alarmas) sube **1 cuadro por segundo** (640 px, JPEG 65, ~30 KB) con zonas, recuadros,
  `#id`, segundos en escena y zona, **directo al Storage** con la URL firmada de
  `POST /api/agent/cameras/{id}/live-url` (`c{cliente}/live/{id}.jpg`, upsert, `cache-control: no-cache`).
  El navegador pide la URL firmada del cuadro con `?t=` cada segundo. Corta 45 s después del último
  latido; con la pestaña oculta no late. **Costo:** no pasa por Vercel; sí consume egress de Supabase
  (~100 MB por hora de visualización por cámara; el plan gratis tiene 5 GB/mes → Supabase Pro si se mira
  mucho). Si el CDN de Supabase devolviera cuadros viejos, pasar a `GET /api/cameras/{id}/live-frame`
  con redirección firmada por cuadro (cuesta invocaciones de Vercel).
- **Clips de estadía completa (22 oct):** el clip de una alerta en modo vivo arranca cuando la persona
  **apareció** (`RuleEvent.track_start_ts`, con 3 s de margen; mínimo `clip_pre`) y sigue **mientras se la
  vea** hasta `clip_post` después de que se fue, con tope `clip_max_seconds` (180, env `CLIP_MAX_SECONDS`,
  viaja en `settings` al agente). El buffer del `Analyzer` guarda JPEG de 640 px (3 min ≈ 36 MB por cámara);
  `ClipReady.frames` son `(ts, bytes)` y `write_mp4` decodifica de a uno. Con `nvr_clips`, el tramo que se
  baja del grabador es el mismo (`ClipReady.start_ts/end_ts`).
- **Ventana de la PC central (22 oct):** `agent/ventana.py`, Tkinter (sin Pillow: las imágenes van como PPM).
  Izquierda: árbol de clientes y cámaras con estado (`Agent.window_cameras`); se eligen cámaras o un cliente
  entero y el mosaico de la derecha muestra solo esas con los recuadros de la IA a la velocidad del stream
  (`CameraWorker.display` / `LiveWorker.display`, que solo se dibujan para las elegidas). La selección se
  guarda en `agent_data/ventana.json`. `--ventana` / `--sin-ventana`; en el .exe viene activada (env
  `VIGIA_VENTANA=0` la apaga); sin pantalla o sin Tk, avisa y sigue. Cerrar la ventana no detiene el agente.
  Para probarla en la sesión cloud: `python3.12` del sistema con `python3-tk` (apt) + venv con
  `--system-site-packages` y `xvfb-run`; captura con `import -window root`.
- **La gestión se hace en la app de escritorio (22 oct, decisión del dueño):** la web queda para los
  **clientes** (su resumen, alertas, incidentes y cámaras). Admin y empleados que entran a la web ven "Usá la
  app de escritorio" (con un link de emergencia "Entrar igual" que guarda `sessionStorage.vigia_web_staff`).
  La app (`VigiaAgente.exe`) entra con **usuario y clave** (`POST /api/login`, sesión Bearer guardada en
  `agente.env` como `VIGIA_SESSION`) y tiene pestañas (`agent/oficina.py`): **Clientes y cámaras** (alta,
  edición, borrado, probar conexión, `ZonesEditor` sobre la imagen de la cámara, métricas del día),
  **Alertas** (lista con filtros, foto, reproducción del clip dentro de la app con OpenCV, etiquetas con
  teclas 1-4, «Avisar al cliente» que copia el link), **Administración** (solo admin: usuarios y PCs). Con
  «Esta PC analiza las cámaras» tildado (solo admin) además vincula la PC (`/api/agent/login`) y suma la
  pestaña **En vivo** con el análisis; sin tildar es un puesto de empleado (no analiza). Las llamadas a la
  red corren en hilos (`Bg`) y vuelven por `Ventana.hooks`. "Cerrar sesión" borra la sesión y reabre el
  programa. **Vercel pasó a `iad1`** (vercel.json `regions`), junto a Supabase us-east-1: antes estaba en
  gru1 y cada consulta a la base cruzaba el continente (la web era lenta por eso).
- **Zonas en caliente y prueba desde la PC (23 oct):** al guardar una cámara o zonas en la app, `Oficina.config_saved`
  pide la configuración ya (`Agent.refresh_config`); además se pide cada 20 s. `CameraWorker.fingerprint_of` ya no
  incluye `analysis` (solo si hay que detectar vehículos): zonas y reglas nuevas se aplican en caliente
  (`update_analysis` → `_apply_analysis` en el hilo de la cámara: Analyzer.zones/engine, métricas y vivo) **sin
  reconectar el video**. "Probar conexión" en la app, si la cámara la analiza esta PC, prueba **desde la PC**
  (`Agent.local_probe`: página del grabador + RTSP); si no, desde el servidor (que no llega a IPs internas
  192.168.x: el mensaje lo explica). La lista de cámaras se refresca sola cada 15 s.
- **Actualizar la app con un botón (23 oct):** arriba a la derecha, «Buscar actualización» (se vuelve
  «★ Actualizar a X» si al abrir detecta una nueva). `GET /api/app/update` lee la última Release de GitHub
  (`GITHUB_REPO`; `GITHUB_TOKEN` solo si el repo pasa a privado) y con `?download=1` da el link firmado del
  .zip (redirect de la API de assets). `agent/actualizar.py` lo baja, saca `VigiaAgente.exe` como
  `.nuevo.exe`, renombra el que corre a `VigiaAgente.viejo.exe` (Windows deja renombrar un .exe abierto),
  pone el nuevo y lo reabre (`restart_app`, con `PYINSTALLER_RESET_ENVIRONMENT=1`); el viejo se borra al
  arrancar. `agente.env` no se toca: vuelve con la misma sesión. La versión la graba el workflow en
  `agent/version.py` (en el código dice `dev`). **Publicar una versión = correr el workflow con `version`.**
- **Descarga de la app desde la web (23 oct):** `GET /api/app/download` (staff logueado) redirige al .zip de la
  última Release; el botón de la pantalla «Usá la app de escritorio» apunta ahí (sirve aunque el repo sea privado).
  Guía del técnico para instalar un local: `docs/guia-tecnico-instalacion.html` + `.pdf` (DVR → IP fija → router
  ZTE F6600 → DDNS No-IP → prueba con reinicio → app).
  **Tutorial para clientes Hikvision (26 oct):** `docs/tutorial-hikvision-analitica-critica.html` + `.pdf` («Tutorial
  Hikvision · Analítica Crítica · Cómo darnos acceso a sus cámaras», de usted): Hik-Connect → apagar cifrado (app,
  grabador o web) → substream H.264 640×360 8 fps 384 kbps → Hik-Connect for Teams → AppKey/SecretKey → planilla de
  datos y cámaras. Los nombres de menú no se verificaron en todos los modelos. PDF con Playwright
  (`executable_path="/opt/pw-browsers/chromium"`, A4, `print_background=True`).
- **Usuario web al crear el cliente (23 oct):** `ClientIn.web_username/web_password` → crea (o edita) el
  usuario de rol `cliente` en la misma transacción; staff (admin y empleado) puede. `GET /api/clients` trae
  `web_username`. En la app: recuadro «Acceso a la web del cliente» (usuario sugerido del nombre, «Generar»
  clave) y al guardar muestra el texto para pasarle al cliente con «Copiar».
- **Usuarios con cámaras asignadas, sin elegir PC (24 oct, decisión del dueño; app 1.13.0):** el admin crea usuarios
  (admin o empleado) y **les asigna cámaras** (Administración → «Editar y asignar cámaras», o «Asignada a» en la
  cámara). Cada usuario entra a la app con su usuario y clave y su PC **analiza y muestra solo sus cámaras**; ya no
  hay casilla «Esta PC analiza» ni nombre de PC ni lista de PCs. Por dentro: `agents.user_id` (la «PC de <usuario>»,
  la crea `user_agent()` sola), `POST /api/agent/session` (staff; rota el token: si el mismo usuario entra en otra
  PC, la anterior deja de analizar), `PUT /api/users/{id}/cameras`, `CameraIn.operator_id` (None = no cambia,
  0 = sin asignar; solo admin), `camera_dict.operator_id/operator_name`, `GET /api/users` trae `camera_ids`.
  **Un empleado ve en Alertas solo las de sus cámaras** (`list_events`). «Clientes y cámaras» sigue mostrando todo.
  `/api/agent/login` (admin + nombre de PC) queda para el agente sin ventana.
- **Tipo de cámara (24 oct):** `cameras.profile` = general | gondola | caja | entrada | deposito | exterior | surtidor
  (`CAMERA_PROFILES` en `server/main.py`). Decide qué reglas se aplican si en «Zonas y reglas» no se eligieron a
  mano (`enabled_rules`), si detecta vehículos (exterior, surtidor) y, en depósito sin zonas, toda la imagen es
  zona restringida. Lo arma `camera_analysis()`; el agente no cambia.
- **Seguimiento con memoria (24 oct, `core/tracker.py`):** una persona perdida se recuerda 30 s; si reaparece cerca
  de donde se la perdió (más lejos cuanto más tiempo pasó, hasta 4 veces su tamaño), **no entrando por el borde**, y
  con ropa parecida (histograma de color HSV, correlación ≥ 0,35), conserva el número y el tiempo en escena. Quien
  aparece tocando el borde es nuevo; quien se perdió tocando el borde se da por ido. `update(dets, ts, frame)`.
  Siguiente paso si no alcanza: ByteTrack/BoT-SORT con un modelo de re-identificación (OSNet).
- **Ahorro de insumos (25 oct, pedido del dueño: «ahorrar lo máximo posible»; app 1.14.0):**
  - **Horario del local** (`clients.hours`, `core/schedule.py`): un texto por día (`{"lun": "09:00-17:00", "mie":
    "09:00-13:00, 16:00-20:00", "vie": "18:00-02:00", "dom": ""}`; acepta «9-17» y «9 a 17»; vacío = cerrado; sin
    horario = 24 h). Se mira desde 15 min antes de abrir hasta 15 min después de cerrar (`DEFAULT_MARGIN_MIN`).
    Con el local cerrado la PC central **no abre el video** (`agent.__main__.effective_mode`, revisado cada 30 s
    sin pedir nada al servidor); la cámara decide qué hacer en `cameras.off_hours`: `apagada` (por defecto),
    `alarmas` (escucha al grabador con `DeviceAlarmListener` aunque su modo sea vivo; solo por IP y puertos: por
    Hik-Connect todavía no llegan alarmas) o `vivo` (24 h). El servidor acepta esas alarmas (`agent_device_alarm`,
    `agent_next_alarm` miran `off_hours`). Estado `fuera_de_horario` («Local cerrado · vuelve 08:45»). La regla
    «fuera de horario» usa el horario del cliente (`analysis.schedule`, aparte de `rules` para que un agente viejo
    no falle). En la app: recuadro «Horario del local» en el cliente (dice cuántas horas por semana se mira) y
    «Con el local cerrado» en la cámara. El formulario del cliente pasó a dos columnas (entra en 1366×768).
  - **Filtro de movimiento** (`core/motion.py`, `MotionGate`): si la escena está quieta (96×54 en gris contra lo
    de hace 1 s) y no se ve a nadie (`tracker.seeing`, `Analyzer.busy`), no se usa el detector; igual se analiza
    1 cuadro cada 2 s. El buffer de clips guarda 1 cuadro/s en esos ratos (`Analyzer.idle`). Ahorra placa de video
    (no internet ni decodificación). En un video sintético con 70 de 81 s vacíos salteó el 71 % y la alerta salió
    igual. La app muestra «· 80 % quieta» (`idle_pct` en el estado).
  - **Solo métricas** (cliente sin plan «hurto», o tipo de cámara «Solo métricas» / sin reglas): `METRICS_FPS` 2
    (3 con zona de entrada o surtidor), `enabled_rules=[]`, `Analyzer(clips=False)` (no comprime ni guarda cuadros).
    Viaja como `metrics_only` y `process_fps` en `/api/agent/config`. El mínimo de 5 fps sigue para hurto.
  - **Menos pedidos al servidor:** métricas de todas las cámaras en un solo envío cada 5 min (`MetricsQueue`); la
    configuración se pide cada 60 s (antes 20; al guardar en la app se pide ya); alarmas pendientes cada 20 s y solo
    si hay cámaras en modo alarmas (más 10 min después de abrir); videos para entrenar cada 120 s; la lista de
    cámaras de la app cada 30 s y solo con la pestaña a la vista.
  - Arreglado: el «Asignada a» del formulario de cámara no se mandaba (se agregaba después de armar el pedido).
  - **Lo que tiene que hacer el dueño:** cargar el horario de cada cliente; sacar «Detección de hurtos» a los clientes
    que son solo métricas; en los grabadores, substream a 640×360, 6-8 fps, 256-384 kbps (desde Hik-Partner Pro):
    eso baja internet y decodificación, que el filtro de movimiento no ahorra.
- **Máquinas de análisis en la nube (26 oct, decisión del dueño: sin inversión inicial, opciones C/D; app 1.15.0):**
  - **Quién revisa y quién analiza están separados.** `cameras.operator_id` = usuario que la revisa (Alertas del
    empleado, «Asignada a»); `cameras.agent_id` = qué la analiza. Ajuste global `app_settings.analysis_mode`
    (Administración → «Dónde se analizan las cámaras»): `pcs` (como antes: la PC de quien la revisa) o
    `servidores` (máquinas de análisis; los usuarios solo revisan, cualquier notebook sirve). Al pasar a
    `servidores`, mientras no haya máquinas en línea las cámaras siguen en las PCs. La migración llenó
    `operator_id` desde la PC que analizaba (`ensure_schema`).
  - **Máquina de análisis** = `Agent.kind="servidor"` (`capacity`, `instance_id`, `draining`). Se da de alta sola:
    `python -m agent --clave-maquinas vgm_…` → `POST /api/agent/enroll` (clave de máquinas: Administración →
    «Clave para máquinas nuevas», se guarda solo su hash en `app_settings.pool_key_hash`) → token en
    `<datos>/token-maquina-<instancia>` (una copia clonada se da de alta de nuevo). Si la borran, se vuelve a
    dar de alta. `agent/maquina.py` (id de instancia por el metadata de AWS, IMDSv2) y `run_machine`.
  - **Reparto** (`server/pool.py`, `rebalance` en main.py, como mucho cada 20 s, con `pg_try_advisory_xact_lock`):
    peso por cámara (hurto 5 fps = 1,0; conteo 2 fps ≈ 0,5; alarmas 0,1; local cerrado y «No mirar» = 0); cada
    cámara se queda en su máquina si entra, si no va a la menos cargada (si no entra en ninguna, igual va: mejor
    apretada que sin mirar). Máquina sin noticias 90 s = caída → sus cámaras pasan a las otras. `/api/agent/status`
    devuelve `rev` (`app_settings.config_rev`, se sube al tocar cámaras/zonas/clientes) y los ids de sus cámaras:
    si cambian, el agente pide la configuración ya (la config se pide cada 60 s igual).
  - **Plan y encendedor:** `GET /api/agent/scale` (cabecera `X-Pool-Key`) calcula cuántas máquinas hacen falta
    ahora y dentro de 20 min (horarios de los locales), marca para vaciar las que sobran y lista las que ya se
    pueden apagar (sin cámaras). `agent/escalador.py` (corre en una máquina chica siempre prendida) prende/apaga
    instancias EC2 con la etiqueta `vigia-grupo` (o crea con una plantilla de lanzamiento), con tope
    `VIGIA_MAX_MAQUINAS`; `VIGIA_NUBE=ninguna` solo informa (Tigo mientras no se sepa su API). Sin claves de AWS en
    Vigía: rol de IAM en la instancia del encendedor.
  - **GPU por lotes** (`core/lotes.py` + `OnnxSession` en `core/detectors.py`): en NVIDIA (CUDA/TensorRT) el modelo
    se transforma en memoria (paquete `onnx`) para recibir **lotes** de cuadros uint8 BGR tal cual salen de OpenCV
    (la conversión a decimales y RGB la hace la GPU) y devolver solo persona y vehículos; un hilo junta cuadros de
    todas las cámaras (hasta 16, espera máx. 8 ms). Verificado: mismas detecciones que de a uno (bus.jpg y
    zidane.jpg). `VIGIA_LOTE=N` lo fuerza (o lo apaga con 1); `VIGIA_TENSORRT=1` usa TensorRT FP16 (motor guardado
    en `~/.cache/vigia-tensorrt`). La app de Windows (DirectML) sigue de a un cuadro.
  - **Instalación:** `deploy/maquina-analisis/` (`README.md` paso a paso para AWS São Paulo, `instalar.sh` para
    Ubuntu con driver NVIDIA, systemd `vigia-maquina@N` con varios procesos por máquina, `vigia-actualizar` que
    hace `git pull` en cada arranque, `vigia-escalador`). `requirements-servidor.txt` (onnxruntime-gpu con CUDA y
    cuDNN por pip: solo hace falta el driver). **Sin probar en una máquina con GPU real todavía**: medir cuántas
    cámaras aguanta (Administración → «Cámaras por máquina», 30 por defecto).
- **Panel de datos y prueba por cliente (26 oct, app 1.16.0):** la PC del dueño (Dell sin placa de video) queda
  como panel: no analiza ni recibe video, solo datos.
  - **Dónde se analiza, por cliente y por cámara:** `clients.analysis_location` y `cameras.analysis_location`
    (`general` | `pcs` | `servidores`; la cámara manda sobre el cliente y el cliente sobre el ajuste general).
    `locations()` / `sync_locations()` en main.py hacen la transición (guardan la PC vieja en
    `app_settings.pcs_backup` y la restauran al volver). `camera_weights`, el reparto y el plan solo cuentan las
    cámaras que van a máquinas. Así se prueba **un cliente** (Minas) en una máquina sin mover las 129 cámaras.
  - **Alertas por cliente y cámara:** `GET /api/stats/alerts` (hoy desde la medianoche del local, 7 días, sin
    revisar en 60 días; solo `alert=True` y fuentes vivo/alarma/prueba). En la app, columna «Alertas» (hoy · 7 días ·
    sin revisar) en clientes y cámaras, y columnas «Revisa» / «Analiza».
  - **Exportar métricas:** `GET /api/metrics/export?client_id&desde&hasta&agrupar=dia|hora` → CSV para Excel (BOM,
    `;`, decimales con coma), una fila por cámara y día u hora. En la app, botón «Exportar métricas…».
  - **Árbol máquina → cliente → cámara** en Administración (máquinas de análisis y PCs).
  - **Máquinas sin placa de video** (Google Cloud con los US$ 300 de prueba, u Oracle «Always Free» ARM): `instalar.sh`
    lo detecta (sin `nvidia-smi`), instala `requirements-servidor-cpu.txt`, 1 proceso con capacidad = núcleos y
    `VIGIA_HILOS` = núcleos / procesos → `SessionPool` en `core/detectors.py`: **una sesión de 1 hilo por núcleo**
    compartidas por todas las cámaras (26 oct: en 4 núcleos x86, 42 cuadros/s contra 29 con una sola sesión de 4
    hilos; ≈ 8 cámaras de hurto a 5 fps sin contar decodificar; ARM rinde menos). Pasos en
    `deploy/maquina-analisis/README.md` («Prueba gratis con pocas cámaras»: Google Cloud e2-standard-4 en São Paulo,
    Ubuntu 24.04, botón SSH del navegador; Oracle bajó lo gratis a 2 núcleos/12 GB desde el 18 ago 2026 según
    terceros → 2-3 cámaras). Probado de punta a punta con un venv que solo tiene `requirements-servidor-cpu.txt`
    (servidor local + 5 cámaras): se da de alta, recibe las cámaras y las analiza.
- **Cámaras a demanda: el admin abre y listo (26 oct, decisión del dueño; app 1.18.0).** «Nada de PCs ni de asignar
  cámaras»: se entra a la app con el usuario admin y la pestaña **En vivo** lista **todas** las cámaras de todos los
  clientes (`GET /api/agent/catalog`, token de la PC del usuario). Clic en una cámara = se abre en esa PC (se ve y se
  analiza: recuadros, alertas, métricas); sacarla de la selección = se cierra. La app avisa qué tiene abierto con
  `PUT /api/agent/watch` (pone `cameras.agent_id` = la PC del usuario y libera las que dejó; las que van a las
  máquinas de análisis no se tocan) y pide la configuración ya. Espera 0,8 s sin cambios de selección antes de
  avisar; al entrar no abre nada hasta avisar; al cerrar la app libera todo (`watch([])`). Clic en un cliente no
  abre todas sus cámaras y no hay botón «Todas» (sí «Cerrar todas»). `Agent(on_demand=True)` en `run_app`,
  `Ventana.build_main(on_demand=True)`. Fuera de la app: sin «Asignada a» en la cámara, sin «Cámaras asignadas» en
  los usuarios ni columna «Revisa»; «Abierta en» dice qué PC la tiene; estado «Cerrada». Una cámara que nunca se
  abrió no tiene imagen para dibujar zonas: abrirla antes. `operator_id` y `PUT /api/users/{id}/cameras` siguen en
  el servidor pero la app ya no los usa. Ese día se pidió borrar las 8 PCs viejas sin usuario y las cámaras
  repetidas de Minas (128 → 35, una por grabador y canal); el conector de Supabase no pudo (la confirmación no
  llega): se le pasó el SQL al dueño para el SQL Editor (ojo: ese editor corre cada sentencia aparte, una
  `temp table ... on commit drop` no sirve; usar subconsultas). Quedó 1 PC («PC de admin») y Minas con 35 cámaras.
  **Borrar todo desde la app (pedido del dueño, app 1.18.1):** clientes (ya estaba: escribir el nombre), cámaras
  (varias), **usuarios** (`DELETE /api/users/{id}`: no el propio ni el último admin activo; su PC se borra y lo que
  hizo queda sin nombre) y **PCs** («Quitar máquina o PC» en Administración; la lista muestra todas las PCs aunque
  no tengan cámaras). **No se repiten cámaras de Hik-Connect** (`check_not_repeated`: mismo cliente, serie y canal
  → 409; la app, al agregar varias, saltea las que ya estaban y lo dice). En Administración los botones de máquinas
  quedaban tapados en pantallas de 768 de alto: ahora van antes que la lista.
  **Google Cloud probado (26 oct, otra sesión de Claude con el dueño; app 1.18.2):** VM `vigia-analisis-1`
  (proyecto `vigia-512822`, southamerica-east1-c, e2-standard-4, Ubuntu 24.04) con `instalar.sh`: 3 cámaras de Minas
  (camera 3/5/13@E02380030) a 5 fps, ~11 % de procesador. Se caía al arrancar: Google contesta en 169.254.169.254 un
  405 con HTML («’») al PUT de AWS y `instance_id()` lo usaba como token (UnicodeEncodeError, que no es
  httpx.HTTPError). Ahora mira el código, prueba `computeMetadata/v1/instance/name` (`Metadata-Flavor: Google`) y
  atrapa `UnicodeError`. Los «[tls] IO error: End of file» de ffmpeg con el HLS de Hik-Connect son normales (reabre).
  **Una sola app por PC** (`single_instance`: traba `vigia.lock` en la carpeta de datos, msvcrt en Windows y flock en
  Linux; espera 8 s por si la anterior se está cerrando; `restart_app` la suelta): con dos copias abiertas cada una
  pedía el token de la PC del usuario y la otra quedaba con 401 y se cerraba la sesión. En Administración los
  usuarios y máquinas se cargan después de los clientes (antes a veces salía «—» en vez del nombre del cliente).
  **App 1.18.3:** «Buscar actualización» en la 1.18.1 no mostró nada (los registros de Vercel muestran que esos clics
  no llegaron al servidor: la ventana estaba trabada). Causa encontrada: al cerrar una cámara, `apply_config` hacía
  `thread.join(15)` en el hilo de la ventana (si el video estaba abriendo, hasta 15 s por cámara sin responder). Con
  ventana ya no se espera (`Agent._stopping`; se esperan al cerrar la app); las máquinas sin ventana siguen esperando.
  El botón ahora dice «Buscando…», deja la respuesta en el registro y avisa si no hay respuesta en 30 s. Queda: el
  bucle de la ventana todavía hace pedidos de red (config, estado, lista de cámaras) en el mismo hilo.
- **«En vivo» solo para mirar; qué se analiza se elige en «Clientes» (26 oct, decisión del dueño; app 1.19.0).**
  Reemplaza a «cámaras a demanda» (1.18): **la app ya no analiza nada** (`apply_config` con `on_demand` deja las
  cámaras vacías; al entrar avisa `watch([])` para soltar lo que tuviera). Pestañas: **En vivo** (lista «Cámaras» con
  todas las de todos los clientes; clic = mirar, sin analizar) y **Clientes** (antes «Clientes y cámaras»).
  - **Qué se analiza:** en Clientes, columna «Analizar» (Sí · máquina / No) y botones «✔ Analizar» / «✘ No analizar»
    (varias a la vez) → `POST /api/cameras/analyze` (`camera_ids`, `analyze`): `analysis_location` = `servidores`
    o `pcs` (con «No», `agent_id` y `operator_id` en NULL). Lo analizan las máquinas de análisis. En el formulario de
    la cámara, casilla «Analizar en las máquinas de análisis» (usa el mismo pedido). `camera_dict.analyzed` = dónde se
    analiza en efecto (`locations()`). Cámara nueva = no se analiza (sigue el ajuste general, «pcs»). Ajuste general
    renombrado: «Solo las elegidas en «Clientes»» (pcs) / «Todas las cámaras» (servidores); en el cliente: «Solo las
    cámaras marcadas» / «Todas sus cámaras».
  - **Mirar** (`agent/vista.py`): `POST /api/agent/view` (PC de un usuario del equipo) da la conexión de las elegidas.
    Si no se analiza → `ViewWorker` (abre el video directo, sin detector, 10 cuadros/s a la ventana). Si la analiza
    una máquina → `RemoteView`: la imagen que sube la máquina (1 por segundo, con recuadros) con el latido del panel
    (`POST /api/cameras/{id}/live` con la sesión del usuario, `Agent(beat=...)`), para no pedirle otra transmisión
    al grabador (Hik-Connect da pocas a la vez). El link de Hik-Connect (`stream-url`) también se da a la PC de un
    usuario del equipo para mirar. Estados en la lista: «Viendo · N fps», «Imagen de la máquina», «Se analiza ·
    máquina», «No disponible». Probado de punta a punta con servidor, máquina y app locales (captura).
- **«Ajustar…»: poner a punto cada cámara antes de analizarla (27 oct, pedido del dueño; app 1.20.0).** En Clientes,
  botón «Ajustar…» (de a una cámara) → `TuneWindow` (agent/oficina.py) con el motor `Tuner` (agent/ajuste.py): la
  cámara se analiza **en la PC del usuario** (no en las máquinas), en vivo o con **videos grabados** («Probar con
  videos…», varios a la vez, uno tras otro; tope 30 min; .dav/.264 que OpenCV no abra → exportar MP4/AVI). Se ve lo que
  ve el análisis: zonas, personas con número y, por zona, segundos contra el tiempo de la regla («gondola 32/45s», en
  rojo al llegar) y «TOMO ALGO». **Alertas de prueba**: quedan en la ventana (no van a empleados, cliente ni
  entrenamiento); clic = ir al momento; barra para recorrer y ▶. **Probar tiempos** (merodeo, «tomó algo», caja,
  restringida) recalcula al instante todo lo analizado: se guardan por cuadro los tracks (`Sample`) y solo se vuelven a
  pasar las reglas (`RuleEngine.progress` da zonas y segundos por persona). En vivo se recuerdan 10 min. «Zonas…» abre
  el `ZonesEditor` sobre la imagen del ajuste. «Cómo anda la cámara»: tamaño medio de las personas, confianza del
  detector, apariciones de menos de 2 s, cuadros por segundo, cortes, alertas por hora, y por video qué alertas dio,
  más avisos (`Tuner.hints`: falta zona Salida/Caja/Góndola, personas chicas < 10 % del alto, detector que duda < 0,5…).
  «Guardar tiempos» (y pasa a «En ajuste») · «✔ Aprobar y analizar» → `POST /api/cameras/{id}/tuning` (`status`,
  `notes`, `analyze`) guarda quién y cuándo y la manda a las máquinas. Columnas: `cameras.tuning` (None/en_ajuste/
  aprobada), `tuned_by`, `tuned_at`, `tuning_notes`; columna «Ajuste» en Clientes; «✔ Analizar» pregunta si hay
  cámaras sin aprobar. La PC pide la cámara con `POST /api/agent/tune` (PC de un usuario del equipo: config como la de
  /config + `raw` = zonas y reglas guardadas) y puede dejar su imagen (`/api/agent/cameras/{id}/preview` acepta la PC
  de un usuario del equipo). El ajuste en vivo usa una transmisión del grabador mientras la ventana esté abierta.
  Probado con videos sintéticos (detector de movimiento) y con YOLO sobre una imagen real (capturas en 1366x768).
- **Programa de la PC central amigable (22 oct):** el .exe ya no abre consola (`console=False`); la ventana
  pide **usuario y clave de administrador** y un nombre de PC → `POST /api/agent/login` crea (o reutiliza por
  nombre, rotando el token) un agente central y devuelve el token, que se guarda en `agente.env`
  (`VIGIA_PC_NAME` también). Si el servidor responde 401 (agente borrado), vuelve a pedir usuario y clave.
  Cabecera con estado, botón "Cambiar usuario", registro abajo (`LogHandler`) y a archivo
  `agent_data/vigia.log`. Cerrar la ventana **cierra el programa** (pide confirmación). Sin ventana, la consola
  pide usuario/clave o un token pegado. Un solo programa atiende todos los clientes.
- **Borrar desde el panel (22 oct):** admin puede eliminar clientes (`DELETE /api/clients/{id}`, borra todo lo
  suyo; pide escribir el nombre), cámaras y agentes. **Cámaras (26 oct, app 1.17.0):** en la app se eligen varias
  (Ctrl/Mayús o «Elegir todas») → `POST /api/cameras/delete` (`camera_ids`, `con_alertas`); si tienen alertas o
  videos responde 409 y la app pregunta antes de borrar todo (`remove_cameras`: alertas, avisos, incidentes, videos,
  métricas, alarmas y sus archivos). `DELETE /api/cameras/{id}?con_alertas=true` hace lo mismo con una. El conector MCP de Supabase no deja hacer
  DELETE/UPDATE (pide una confirmación que no llega): usar el panel o el SQL Editor de Supabase.
- **Planes y métricas por cliente (22 oct, pedido del dueño):** `Client.plans` (JSON: `hurto`, `comercio`,
  `estacion`; lo elige el staff en el formulario del cliente). `core/metrics.py` (`MetricsCollector`) usa los
  mismos tracks que las reglas y arma **cubos horarios** por cámara (hora local): personas vistas, tiempo en
  escena, máximo simultáneo, `entries` (zona `entrance`), visitas y tiempo por zona ("a dónde van"),
  vehículos, `pump_*` (vehículos en zona `pump`, espera hasta que una persona pisa el surtidor = "pistero",
  `pump_unattended`), `shop_visits` (zona `shop`). El `CameraWorker` los manda cada 5 min a
  `POST /api/agent/metrics` → tabla `camera_metrics` (upsert por cámara y hora). `GET /api/metrics?client_id&days`
  suma total, por día, por hora y por cámara. Panel: `#/metricas?client_id=` (staff, desde Clientes) y
  **Resumen** del rol cliente (`#/resumen` → `metricsView`), con secciones según plan y gráficos SVG propios.
  Zonas nuevas: `entrance`, `pump`, `shop` (las reglas de hurto las ignoran). **Vehículos:** el ONNX trae las
  80 clases COCO; `build_detector(..., vehicles=True)` (plan `estacion` o zona `pump`, vía
  `analysis.vehicles`) devuelve `label="vehiculo"` (auto/moto/ómnibus/camión); el tracker no mezcla etiquetas
  y las reglas solo miran personas. **Límites honestos:** solo cámaras en modo vivo generan métricas (en modo
  alarmas no se mira continuo); no reconoce caras ni distingue empleados; una persona que se pierde y
  reaparece cuenta dos veces; gestos (mano al bolsillo, botella al cuerpo, lavar vidrios) **no** se detectan:
  requieren modelo propio entrenado con clips etiquetados (pendiente, punto 6 de Próximos pasos).
- **Pestaña Prueba** (`public/prueba.js`): comparte la pantalla (`getDisplayMedia`) de la ventana de
  HPP / iVMS / SmartPSS con la cámara en vivo y corre el detector de personas **en el navegador**
  con ONNX Runtime Web (jsdelivr 1.19.2, wasm 1 hilo) sobre `public/models/personas.onnx` (copia
  del de `models/`, 10 MB; si se reexporta el modelo, copiar a los dos lados). El usuario marca el
  recuadro de la cámara y una zona; si una persona (punto de los pies) queda en la zona más de N s,
  guarda una alerta: `POST /api/events/manual` (foto) → PUT del clip webm de 12 s a la URL firmada →
  `POST /api/events/{id}/clip-done`. `Event.source="prueba"`, `clip_source="prueba"`. Era un puente
  para probar sin puertos abiertos; el 22 oct el dueño la sacó del menú (todos los clientes van por
  router + agente). Sigue en `#/prueba`.

`server/probe.py` + `POST /api/cameras/{id}/probe`: prueba desde Vercel que el DVR responda por HTTP
(ISAPI deviceInfo / CGI magicBox) con el usuario y la clave cargados, lista los canales y avisa si
está expuesto con puertos estándar. Botón "Probar conexión" en Cámaras. `public/guia.html` es la
guía paso a paso (datos a pedir, usuario en el DVR, puertos en el router de Antel, DDNS, prueba en
VLC, carga en el panel, seguridad, fallas comunes); link "Guía" en el menú.

Archivos clave del modo alarmas: `server/hpp.py` (firma del webhook, parseo de alarmas, cliente
de la OpenAPI), `core/clip_analysis.py` (un clip → un resultado), `agent/alarms.py` (cola en la
PC central, `--alarm-workers N`), `tests/test_alarms.py` (flujo completo).

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
- Tipos de evento: los 4 de reglas + `alarma_dvr` (alarma con persona pero sin regla disparada,
  severidad baja). `core.scorer.VECTOR_NAMES` cambió: un modelo viejo se ignora y hay que
  reentrenar (hoy no hay ninguno entrenado, así que no afecta).
- En clips cortos la regla "merodeo" (45 s en góndola) casi nunca llega a dispararse; igual se
  miden los tiempos en góndola/caja y los usa el modelo.
- Alarmas sin evento (sin persona, sin cámara, error) se borran a los 14 días en el cron diario.

### Alarmas desde el equipo (`mode="alarmas_equipo"`, IMPLEMENTADO 19 oct, el modo por defecto)
Criterio elegido: **mínimo costo y mínima intrusión en el local del cliente.**
- **En el local no se instala nada** (ni PC ni software): solo hace falta llegar al grabador
  por VPN, que es necesaria igual para bajar el video.
- **El propio grabador avisa sus alarmas** por una conexión HTTP larga. No hace falta ninguna
  nube, ni API key, ni que el cliente esté en HPP o DoLynk Care.
- **Funciona igual para las dos marcas** (según documentación; **confirmar con un equipo real**):
  - Dahua: `GET /cgi-bin/eventManager.cgi?action=attach&codes=[VideoMotion,CrossLineDetection,...]`
    (digest). Líneas `Code=VideoMotion;action=Start;index=0`. Se toma `index+1` como canal (ojo:
    verificar que sea así en el DVR real); solo `action=Start`.
  - Hikvision: `GET /ISAPI/Event/notification/alertStream` (digest), multipart XML por evento;
    los latidos `videoloss inactive` se ignoran; se usa `channelID`/`dynChannelID`.
- Cómo quedó: `agent/device_events.py` (parsers puros + streams, normaliza a los nombres de HPP:
  VMD / fielddetection / linedetection...), `agent/device_alarms.py` (`DeviceAlarmListener`, un hilo
  por cámara, reconecta con backoff, no repite el mismo tipo en 8 s) → `POST /api/agent/alarms`
  (`store_alarm`, el mismo que usa el webhook de HPP, con `Alarm.source="equipo"`) → el resto es
  idéntico: agrupado, espera del "después", `/api/agent/alarms/next`, `AlarmRunner`, `analyze_clip`.
- El agente informa el estado del listener en `cameras[].state` = `escuchando` / `sin_conexion`
  (se ve en la pantalla Cámaras). En el panel: modo "Alarmas del propio grabador (recomendado)".
- Si el grabador no distingue persona/vehículo, igual filtra el detector de personas al analizar.
- **Por qué antes que HPP:** mientras no llegue la clave VSaaS, el modo HPP **no ahorra la VPN**,
  porque el video sale del grabador igual. HPP se mantiene para cuando llegue esa clave y porque
  reemplaza a los empleados del socio que miran el portal. El modo `alarmas` (HPP) sigue siendo
  solo Hikvision y exige `hpp_serial`; `alarmas_equipo` no exige nada más que host/usuario/clave.

### Modo central
`Agent.client_id` NULL = agente **central**: corre en la PC de la empresa y atiende las cámaras
que se le asignen, **de cualquier cliente**, llegando a cada DVR por internet (puertos abiertos en
el router del cliente). El panel solo crea agentes centrales (los "de local" siguen en el modelo de
datos pero ya no se ofrecen).
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
- **Hik-Connect for Teams (ex HikCentral Connect) SÍ tiene OpenAPI con video (hallado 23 oct, SIN PROBAR):**
  distinta de la de HPP (`/api/hccgw/...` en vez de `/api/hpcgw/...`). Desde 2026 cualquier cuenta en modo
  equipo genera AppKey/Secret en Team Management → API Integration (gratis según usuarios). Token:
  `POST https://<región>.hikcentralconnect.com/api/hccgw/platform/v1/token/get` → `accessToken` + `areaDomain`.
  Video: `POST /api/hccgw/video/v1/live/address/get` (`resourceId`, `deviceSerial`, `type` 1 vivo / 2 nube /
  3 grabación local con `startTime`/`stopTime`, `protocol` 1 EZOPEN / 2 HLS, `quality` 1/2, `expireTime`).
  HLS solo H.264 y sin cifrado de stream; con cifrado solo EZOPEN (SDK EZUIKit). También
  `/api/hccgw/video/v1/isapi/proxypass` (ISAPI por la nube) y `/api/hccgw/resource/v1/areas/cameras/get`.
  Fuente: integración de Home Assistant `Frens98/hikconnect-nvr` (vivo por HLS, funciona) y una referencia de
  terceros de la API V2.11.800 (tipos de playback, sin verificar). Si anda, **evita abrir puertos** en los locales
  Hikvision. El sandbox de Claude no llega a `*.hikcentralconnect.com`: hay que probar desde la PC central.
  **Implementado el 23 oct (app 1.9.0, sin probar con equipo real):** marca `hikconnect` («Hikvision por
  Hik-Connect (sin puertos)»). Cliente: `hc_app_key`, `hc_secret_enc` (cifrado), `hc_api_url` (se completa solo con
  la región que anduvo; `REGION_URLS` en `server/hikconnect.py`). Cámara: `hpp_serial` = serie del grabador,
  `channel`, `hc_resource_id` (se busca solo por serie+canal o se elige con «Elegir de Hik-Connect…»). La PC
  central pide el link a `POST /api/agent/cameras/{id}/stream-url` (el SecretKey nunca sale del servidor) y lo abre
  con OpenCV; cada reconexión pide link nuevo (`agent/hikconnect.py`, `CameraWorker.run`). Solo modo `vivo`.
  `GET /api/clients/{id}/hikconnect/cameras` lista lo que ve la API.
  **Probado el 23 oct con el DVR de Minas (E02380030, DS-7116HGHI-F1):** token OK en `isa`, lista de 42 cámaras OK,
  el link de video da **EVZ60019 (cifrado de video prendido)**. Mandar `code` (el código de verificación) NO lo
  arregla para HLS: solo sirve para el reproductor EZOPEN de Hikvision. **Hay que apagar el cifrado en el
  grabador.** (`Client.hc_code_enc` quedó en la base pero no se usa.) Formulario de cámara (app 1.10.1): «Conexión»
  = Hikvision por Hik-Connect (por defecto si el cliente tiene clave; sin IP ni puertos; se eligen varias cámaras
  a la vez) / Hikvision por IP y puertos / Dahua por IP y puertos. **Dahua no tiene API pública equivalente**
  (DoLynk Care no publica una; IMOU Open Platform da HLS pero solo para equipos en cuentas Imou, sin verificar con
  NVR de DMSS). **24 oct: anda el vivo por Hik-Connect** (Minas, cámaras 1, 5 y 13 en línea con recuadros) una vez
  apagado el cifrado. El HLS llega en ráfagas de 1-4 s: el `CameraWorker` usa `MediaClock` (marcas de tiempo del
  video, o cuadros/fps) en vez del reloj de la PC para elegir qué cuadros analizar (antes salía 0-1 fps). Los errores de Hik-Connect quedan en los
  logs de Vercel (`vigia.hikconnect`): mirarlos en la primera prueba.
  **26 oct (app 1.17.1):** si OpenCV no abre el HLS («No se pudo abrir …m3u8»), `BaseCamera.open_failure` →
  `agent/hikconnect.diagnose` pide la lista directo y suma el motivo al error (código HTTP y cuerpo de Hik-Connect,
  no se pudo conectar al puerto 4443, o lista OK = video no legible, ¿H.265?). Ese día, en Minas, los canales 3, 5
  y 13 de E02380030 andaban en DESKTOP-JL2DU44 y los 1 y 2 no abrían en la PC de admin (misma versión de librerías
  desde la 1.13: no era la app); sospecha: límite de transmisiones simultáneas por grabador en la nube. El segundo
  grabador de Minas (GE7132986) tiene el cifrado prendido (EVZ60019).
- **La cámara analógica va al DVR**; el agente se conecta al DVR, no a la cámara.
- **Dahua y la hora del playback:** algunos equipos interpretan starttime/endtime en hora local
  y otros en UTC. Se elige por cámara con `playback_time`; si el clip llega corrido, probar la otra.
- **Cómo se llega al DVR (decisión del dueño, 19 oct 2026):** la cajita GL.iNet quedó descartada
  (en Uruguay cuesta ~12.000 UYU). Se eligió **exponer el DVR por el router del cliente** (IP pública
  o DDNS + dos puertos externos no estándar redirigidos a 554 y 80) y usar el mismo enlace RTSP de
  siempre. Es la opción más riesgosa de las que se compararon; las mitigaciones están en
  `public/guia.html` (usuario `vigia` mínimo, clave larga, puertos altos, firmware, bloqueo por
  intentos). **No va ninguna PC ni aparato en el local del cliente** (el dueño lo descartó): la PC
  central llega directo al DVR por los puertos del router de Antel. La captura de pantalla de HPP
  quedó como pestaña **Prueba** (puente para probar, no servicio). Descartadas por ahora: Tailscale
  en PC del local, RTMP saliente, FTP de clips, foto de alarma de HPP. La definitiva para Hikvision
  sigue siendo la clave VSaaS/HPNetSDK (pendiente de pedir).
- El detector es **YOLO11n, licencia AGPL-3.0**. Para vender hay que comprar licencia
  Enterprise de Ultralytics o cambiar a un modelo Apache-2.0 (YOLOX, RT-DETR). **Pendiente.**

## Limitaciones honestas (decirlas siempre, no prometer de más)

- **No detecta ocultamiento** (guardar algo en la ropa/mochila). Marca patrones por zonas y
  tiempos. Cada evento es un indicio para revisar, **no una prueba**.
- **La precisión nunca se midió con video real.** Es lo primero del piloto.
- Probado con un video real de noticiero: detecta a la persona en el 86 % de los cuadros y
  dispara "merodeo", pero no "ve" el gesto del hurto.
- Capacidad: ~120 ms/cuadro en CPU → 2–4 cámaras por PC a 5 fps. Medir antes de prometer cupo.
  **24 oct:** en un i5-11400 + RTX 3050 daba 15 fps con 1 cámara y 5 fps con 5 porque **la GPU no se usaba**
  (el .exe traía `onnxruntime` solo CPU y cada cámara tenía su sesión). Desde la app 1.11.0: el workflow instala
  `onnxruntime-directml` (GPU en Windows por DirectX 12, cae a CPU si no hay), `build_detector` comparte **una**
  sesión en GPU entre todas las cámaras (con lock) y el registro dice «Detector de personas en GPU (Dml)» o «en
  procesador». El `CameraWorker` hace `grab()` de todos los cuadros y `retrieve()` solo de los que analiza o
  muestra (ventana a 10 fps máx.). Falta medir cuántas cámaras aguanta así (estimado 20-30 a 5 fps).
  **Decisión del dueño (24 oct): mínimo 5 fps para todas las cámaras** (`MIN_FPS` en el servidor, `CameraIn`
  `ge=5`; el agente también hace `max(5, …)`; en la app un Spinbox 5–15).
- **Conexión «Por IP y puertos» (24 oct, app 1.12.0):** en el formulario quedan dos opciones: Hik-Connect o «Por IP y
  puertos (Hikvision o Dahua: la marca se detecta sola)» → `brand="ip"`. La PC central (`AutoCamera` en
  `agent/cameras.py`) prueba ISAPI (Hikvision) y CGI (Dahua), y si la página no contesta, el RTSP de cada marca;
  recuerda la marca detectada. «Probar conexión» desde el servidor también prueba las dos. Las cámaras viejas con
  `hikvision`/`dahua` conservan su marca.
- Ancho de banda del local: 0,25–0,5 Mbps de subida por cámara, 24 h.

## Desarrollo

```bash
python -m pytest -q                 # 189 tests (unos 40 se saltean sin Postgres)
VIGIA_TEST_PG="postgresql://postgres@127.0.0.1:54329/vigia_test" python -m pytest -q   # Postgres
python -m agent --server https://vigia-three-ashy.vercel.app --token TOKEN   # agente
python -m agent.hikvision --host IP --user vigia --password ****             # probar DVR Hikvision
python -m agent.dahua     --host IP --user vigia --password ****             # probar DVR Dahua
```

- Sin `DATABASE_URL` usa SQLite local; sin Supabase guarda archivos en disco. Así corre local.
- **Agente como .exe (21 oct):** `.github/workflows/agente-windows.yml` compila `VigiaAgente.exe` con
  PyInstaller (`agente_pc.py` + `agente_pc.spec`, modelo ONNX adentro) en `windows-latest`; se lanza a
  mano (Actions → Run workflow) o con un tag `agente-v*` (también publica en Releases). El agente lee
  `agente.env` (`VIGIA_SERVER`, `VIGIA_AGENT_TOKEN`) de la carpeta del .exe/repo y, si falta el token y
  hay consola, lo pide y lo guarda. `agent/iniciar_windows.bat` hace lo mismo con Python instalado.
- **Deploy:** `git push` + lanzar el deploy en Vercel (el proyecto no tiene auto-deploy por git
  configurado; se dispara con la API/MCP de Vercel o desde el panel de Vercel).
- Las tablas se crean solas (`create_all`) y quedan con RLS activo. Las **columnas nuevas** también
  se agregan solas al arrancar (`add_missing_columns`), pero **solo si se sube `SCHEMA_VERSION`** en
  `server/db.py`. Renombrar o cambiar el tipo de una columna sí requiere migración manual.
- Ojo: `requirements.txt` es del servidor; `requirements-agent.txt` del agente (OpenCV, onnxruntime).
- En la sesión cloud de Claude Code: `pip install -r requirements.txt -r requirements-agent.txt
  pytest scikit-learn`; Postgres local con `initdb` como usuario `postgres` en `/var/tmp/vigia-pg`
  (puerto 54329); capturas del panel con Playwright usando
  `executable_path="/opt/pw-browsers/chromium"` contra `uvicorn ... --port 8011` con SQLite.

## Pendiente del socio (pedido por WhatsApp el 17 oct)

1. AppKey/SecretKey de la OpenAPI de HPP (por privado, no por grupo).
2. Local para el piloto: nombre, serie del DVR, canales que miran góndola/caja/puerta, tipo de
   internet y router (para decidir VPN en router o cajita GL.iNet).
3. Usuario `vigia` en ese DVR con vista en vivo + reproducción (creado a distancia desde HPP).
4. Cómo están configuradas las alarmas del DVR (movimiento / intrusión / cruce de línea, en qué
   cámaras).
5. Clips de hurtos reales (ideal ≥ 30) y de falsas alarmas, con fecha, local y cámara.
6. Cómo trabajan hoy los empleados: alarmas por día por local, % falsas, a quién avisan y cómo.
7. Consulta a Hikvision por la clave VSaaS / HPNetSDK (requisitos, costo, Uruguay).
8. Si el contrato con los clientes permite usar sus videos para entrenar.

**Hecho el 21 oct (equipo propio de Santiago, cliente "Prueba"):** DVR DS-7104HGHI-M1/T en
192.168.1.6 (reservada por MAC en el ZTE F6600 de Antel), reenvío TCP 38554→554 y 38080→80,
IP pública de Antel real (sin CG-NAT, PPPoE, puede cambiar: falta DDNS), RTSP verificado en VLC
desde datos móviles y otra red, cámara "Habitacion" cargada en modo **vivo** (el DVR no tiene
disco) y "Probar conexión" devuelve modelo y 4 canales. Agente `vigia` (id 3) corriendo como
`VigiaAgente.exe` en la PC de Santiago: cámara en línea a 4 fps y vista previa recibida.
El .exe se publica en GitHub Releases (tag `agente-v*`); link en Administración → Agentes.

**Lunes 20 oct:** Santiago tendrá acceso al video en vivo de un DVR. Plan: desde la PC central
`python -m agent.hikvision` (o `agent.dahua`) `--host IP --user vigia --password ****` (lista
canales y prueba el stream); cargar la cámara con agente = central en modo **"Alarmas del propio
grabador"**; activar movimiento/intrusión en el grabador; dibujar zonas; provocar una alarma y
seguirla en la pantalla Alarmas. Verificar la hora de la grabación (`playback_time`).

## Próximos pasos

0. **Verificar el vivo en producción** con la cámara "Habitacion": que el cuadro cambie cada segundo
   (si el CDN de Supabase cachea, ver nota en "Vista en vivo liviana"). Después: **métricas por cliente**
   (personas por hora, tiempo en góndola/caja, alertas por día: la PC central ya mide los tiempos por
   persona; falta guardarlos cada 5 min en una tabla y graficarlos) y renombrar "Agente" → "PC central".
1. **Probar `alarmas_equipo` con un grabador real** (lunes 20 oct, con el acceso en vivo): cargar
   la cámara en ese modo, generar movimiento y mirar en Alarmas que llegue, que el canal coincida
   y que el clip bajado sea del momento correcto (`playback_time`). Si Dahua trae `index` con otra
   numeración, ajustar `parse_dahua_line`.
2. **Conectar cada local según `public/guia.html`:** DVR por el router (IP pública/DDNS + puertos
   externos). Hasta que un local tenga los puertos abiertos, probar con la pestaña **Prueba**.
3. **Piloto y medición:** % de alarmas descartadas solas y, de las que tienen persona, cuántas
   eran hurto (con las etiquetas de los empleados). Sin esto no se puede prometer precisión.
4. **Conectar HPP** (solo Hikvision, opcional): el socio saca la API key, se carga en Vercel,
   "Conectar" en Alarmas HPP, y revisar la primera alarma real (formato del webhook, hora del
   DVR, serie/canal).
5. **Preguntas a Hikvision:** ¿requisitos y costo de la clave VSaaS / HPNetSDK? ¿permite bajar la
   grabación por horario (o el clip de alarma)? ¿corre en Windows/Linux? ¿UY/AR? Si la dan,
   los clientes Hikvision se quedan sin VPN (otro `ClipFetcher` en `agent/alarms.py`).
6. **Detector de gestos** (mano al bolsillo en caja, botella/elemento al cuerpo; lavado de vidrios en
   estaciones): necesita 30–50 clips etiquetados por gesto y un modelo de pose (YOLO-pose) o de acción;
   hoy el sistema solo mide posiciones y tiempos.
7. **Licencia del detector** (AGPL) antes de cobrar.
8. Antes de cobrar: Vercel Pro (el plan gratis es no comercial) y Supabase Pro (el gratis limita
   los archivos a 50 MB).

## Estilo

Español rioplatense, claro y sin jerga técnica innecesaria: el usuario es desarrollador pero
el producto lo usan comerciantes. Explicar los límites con honestidad antes que vender humo.
