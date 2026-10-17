# Vigía — contexto del proyecto

Sistema comercial que vigila cámaras Hikvision, detecta comportamiento sospechoso (posible
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
- **core/**: detector YOLO (ONNX, CPU), tracker, zonas/reglas, scorer, clips. El servidor solo
  importa `core.rules`, `core.scorer` (y sus deps `geometry`, `tracker`): **no** mete OpenCV ni
  onnxruntime en el build de Vercel. No romper eso.
- **public/**: panel (SPA sin framework) + `aviso.html`, la página pública del clip.

### Modo central (lo que se usa)
`Agent.client_id` NULL = agente **central**: corre en la PC de la empresa y atiende las cámaras
que se le asignen, **de cualquier cliente**, llegando a cada DVR por **VPN** (Tailscale /
WireGuard; cajita GL.iNet ~US$30 si el router no soporta). Evita poner una PC por local.
`agent_owns()` / `agent_cameras_q()` deciden qué ve cada agente. Cada cámara recibe el
`filter_model` de **su** cliente, así un agente corre varios modelos a la vez.

### Flujo de aviso al cliente (clave del producto)
El clip llega al operador → empleados revisan y etiquetan (1 hurto, 2 sospechoso, 3 normal,
4 falso positivo) → eso entrena el modelo → **una persona** decide avisar al cliente. **Nunca
automático**: evita acusar por un falso positivo. El texto por defecto dice "posible hurto", no
una acusación. Genera un link privado que vence (14 días). Canales: portal (siempre, gratis),
Telegram, WhatsApp (Twilio) y email (Resend), según lo configurado por cliente.
La alerta automática de "hay clip para revisar" va al **operador**, no al cliente.

## Decisiones importantes (no re-litigar)

- **Hik-Connect NO sirve** para leer video: es un visor cerrado, no expone el stream. Verificado
  contra la documentación de Hikvision. El software no puede "correr sobre Hik-Connect".
- **Hik-Partner Pro** (el cliente ya lo usa) sirve para **configurar los DVR a distancia**. Su
  OpenAPI *podría* dar clips de alarma de 7 s, pero **no está confirmado** que entregue video
  continuo analizable. Hay preguntas pendientes para Hikvision (ver abajo).
- **La cámara analógica va al DVR**; el agente se conecta al DVR, no a la cámara.
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
python -m pytest -q                 # 43 tests (SQLite)
VIGIA_TEST_PG="postgresql://postgres@127.0.0.1:54329/vigia_test" python -m pytest -q   # Postgres
python -m agent --server https://vigia-three-ashy.vercel.app --token TOKEN   # agente
python -m agent.hikvision --host IP --user vigia --password ****             # probar DVR
```

- Sin `DATABASE_URL` usa SQLite local; sin Supabase guarda archivos en disco. Así corre local.
- **Deploy:** `git push` + lanzar el deploy en Vercel (el proyecto no tiene auto-deploy por git
  configurado; se dispara con la API/MCP de Vercel o desde el panel de Vercel).
- Las tablas se crean solas (`create_all`) y quedan con RLS activo. Agregar una **columna** a una
  tabla existente sí requiere migración manual (`ALTER TABLE`) en Supabase.
- Ojo: `requirements.txt` es del servidor; `requirements-agent.txt` del agente (OpenCV, onnxruntime).

## Próximos pasos

1. **Piloto real:** conectar el DVR de prueba (DS-7104HGHI-M1/T + cámara analógica
   DS-2CE16D0T-LFS) y medir detección y falsas alarmas por hora.
2. **Detector de ocultamiento** (mano góndola→cuerpo): necesita 30–50 videos etiquetados.
3. **Licencia del detector** (AGPL) antes de cobrar.
4. **Preguntas a Hikvision** (TPP): ¿la OpenAPI da video continuo por URL analizable? ¿los clips
   de alarma pueden durar más de 7 s? ¿costos por equipo/tráfico? ¿sirve en UY/AR?
5. Antes de cobrar: Vercel Pro (el plan gratis es no comercial) y Supabase Pro (el gratis limita
   los archivos a 50 MB).

## Estilo

Español rioplatense, claro y sin jerga técnica innecesaria: el usuario es desarrollador pero
el producto lo usan comerciantes. Explicar los límites con honestidad antes que vender humo.
