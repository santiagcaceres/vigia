# Banco de pruebas en casa (cliente de prueba)

Idea: usar estos equipos como si fueran un cliente más y correr Vigía (agente + servidor) sobre ellos.
Estado: **solo anotado, sin hacer nada todavía** (pedido del usuario, 2026-10-17).

## Equipos (confirmados por etiqueta)
- **Grabador DVR Hikvision DS-7104HGHI-M1/T** (SN FZ4798546): 4 canales Turbo HD, hasta 1080p, H.265, 1 bahía de disco, HDMI, compatible con Hik-Connect. Híbrido "5 en 1" (IP / HDTVI / HDCVI / AHD / CVBS).
- **Cámara Hikvision DS-2CE16D0T-LFS 2.8 mm** (SN GK5172281): bullet **analógica Turbo HD (HDTVI)** 2 MP, Smart Hybrid Light (IR 30 m + luz blanca 20 m), IP67, 12 V 275 mA. **No es IP**: va al DVR por coaxial o por UTP con balun; el agente lee el canal desde el DVR.
- **Baluns de video pasivos** (CYGNUS BP-01B, par) y un **conector de alimentación DC** (hembra de tornillo).
- **Cable UTP Cat6** en rollo (para pasar video con los baluns o para la red).
- Aplicación del cliente: **Hik-Connect** (móvil), además de iVMS-4200 en la PC.

## Para tener en cuenta cuando se retome
- Si la cámara es analógica, va conectada al grabador (por coaxial o por UTP con balun) y **el agente se conecta al grabador**, no a la cámara. Ya está contemplado: el agente lee `rtsp://usuario:clave@IP:554/Streaming/Channels/<canal>01` del grabador.
- Hace falta conectar el grabador a la red de la casa (cable al router) y darle IP, activarlo (clave de admin) y habilitar RTSP/ISAPI. Verificar con `python -m agent.hikvision`.
- Un grabador híbrido de entrada puede tener resolución o fps limitados en el flujo secundario (sub-stream). Hay que medir.
- Un DVR mini sin PoE no alimenta cámaras IP; las analógicas se alimentan con fuente 12 V aparte (conector DC de la foto).
- Pendiente de confirmar: disco duro instalado (¿tiene?), PC donde correr el agente.

## Hik-Connect: qué es y qué se puede hacer con él (investigado 2026-10-17)
- **Hik-Connect** es el servicio de nube P2P gratuito de Hikvision y su app para el usuario final. El DVR se registra en la nube de Hikvision (código de verificación / serie) y la app ve el video a través de un relay de Hikvision, sin abrir puertos. **No expone RTSP ni ninguna URL de video pública**: con usuario y clave de Hik-Connect no hay forma documentada de que un programa propio lea el video.
- **Hik-Partner Pro** (antes Hik-ProConnect) es la versión para instaladores: una cuenta donde se administran los sitios y equipos de muchos clientes, con el cliente autorizando al instalador. Es lo más parecido a "ver todas las cámaras de todos los clientes desde una app", pero es un producto de Hikvision, no un lugar donde correr software ajeno.
- **Hik-Connect OpenAPI / Hik-Partner Pro OpenAPI**: la vía oficial para terceros. Requiere registrarse como Technology Partner (tpp.hikvision.com), pedir AK/SK; "sin costo adicional de licencia". Ofrece gestión de equipos, alarmas, vista en vivo, reproducción y descarga. El video llega por la nube de Hikvision y la documentación pública apunta a un decodificador JS para navegador; **no está confirmado** que entreguen un stream (HLS/RTMP/RTSP) que un servidor pueda analizar cuadro a cuadro. Hay que preguntarlo al soporte regional de integración antes de apostar a esto.
- **ISUP (EHome)**: protocolo para que el DVR se conecte *saliente* a un servidor propio por internet (sin IP fija ni puertos abiertos). Es la alternativa "sin PC en el local", pero exige implementar el protocolo (Device Gateway/SDK) y un servidor con capacidad de video.
- Librerías no oficiales que imitan la app de Hik-Connect existen, pero son frágiles y pueden violar los términos de Hikvision: **no las usamos** para un producto comercial.

### Plan de prueba con estos equipos
1. Conectar la cámara DS-2CE16D0T-LFS al canal 1 del DVR (coaxial, o UTP con los baluns).
2. Conectar el DVR al router por cable; activarlo (clave de admin) y, en Red, habilitar RTSP (554)
   e ISAPI. Confirmar que tenga disco (si no graba, igual sirve el buffer del agente).
3. En la misma red, probar: `python -m agent.hikvision --host <IP-del-DVR> --user vigia --password ****`.
4. Cargar en el panel: cliente "Casa (prueba)", agente (una PC de la casa), cámara (IP del DVR,
   canal 1). Dibujar una zona de góndola/caja/salida de prueba.
5. Simular un "hurto" caminando frente a la cámara y saliendo; revisar el evento, etiquetar y
   probar el botón **Avisar al cliente** (a tu propio Telegram/WhatsApp como si fueras el dueño).

### Conclusión para Vigía
- Hoy: el agente en la red del local lee RTSP del DVR (ya hecho, sin permisos de Hikvision). El panel de Vigía es donde se ven los eventos de todos los clientes. Hik-Connect sigue siendo la app con la que el cliente mira sus cámaras; conviven sin problema.
- Futuro "sin PC en el local": pedir acceso a Hik-Partner Pro OpenAPI y confirmar formato de stream; si sirve, un agente único en la nube (servidor con GPU, no Vercel) consumiría los streams. El análisis por cámara (YOLO) sigue costando lo mismo donde sea que corra, y el ancho de banda de subida de cada local pasa a ser el límite.
