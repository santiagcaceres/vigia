"""Arma docs/conectar-camaras-hikvision.html y docs/conectar-camaras-dahua.html (y sus PDF) para el técnico del
cliente: qué necesitamos, cómo se arma el enlace, qué configurar en el grabador y en el router, y cómo probar."""
import base64
import sys
from pathlib import Path

REPO = Path("/home/user/vigia")
LOGO = "data:image/png;base64," + base64.b64encode((REPO / "public/img/logo.png").read_bytes()).decode()

CSS = """
@page { size: A4; margin: 12mm 13mm 10mm; }
* { box-sizing: border-box; }
body { font-family: "Segoe UI", Inter, Arial, sans-serif; color: #1b1f24; font-size: 9.6pt; line-height: 1.38; margin: 0; }
.head { display: flex; align-items: center; gap: 12px; background: #011b5d; color: #fff; border-radius: 10px;
        padding: 12px 16px; margin-bottom: 12px; }
.head img { width: 40px; height: 40px; background: #fff; border-radius: 8px; padding: 4px; }
.head h1 { font-size: 17pt; margin: 0; line-height: 1.15; }
.head p { margin: 2px 0 0; font-size: 9.5pt; opacity: .85; }
h2 { font-size: 12.2pt; color: #011b5d; margin: 11px 0 5px; padding-bottom: 3px; border-bottom: 2px solid #e3e8f0; }
h2 .n { display: inline-block; width: 21px; height: 21px; border-radius: 50%; background: #011b5d; color: #fff;
        text-align: center; font-size: 10pt; line-height: 21px; margin-right: 6px; }
p { margin: 4px 0; }
.lead { font-size: 10.3pt; margin: 0 0 8px; }
.box { border: 1px solid #d9dee5; background: #f6f8fa; border-radius: 8px; padding: 8px 11px; margin: 6px 0; }
.warn { border-color: #f0c78a; background: #fff8ec; }
.ok { border-color: #9fd3ae; background: #f0faf2; }
ul { margin: 4px 0; padding-left: 18px; } li { margin: 2px 0; }
table { width: 100%; border-collapse: collapse; margin: 5px 0; font-size: 9.1pt; }
th { background: #011b5d; color: #fff; text-align: left; padding: 5px 8px; font-weight: 600; }
td { border-bottom: 1px solid #e3e8f0; padding: 5px 8px; vertical-align: top; }
tr:nth-child(even) td { background: #fafbfc; }
td:first-child { font-weight: 600; width: 27%; }
code { font-family: Consolas, "DejaVu Sans Mono", monospace; background: #eef1f5; padding: 0 4px; border-radius: 3px;
       font-size: 8.8pt; }
.off { background: #ffe4e6; color: #b4233c; font-weight: 700; padding: 0 5px; border-radius: 3px; font-size: 8.3pt; }
.on { background: #e3f6ea; color: #1a7f43; font-weight: 700; padding: 0 5px; border-radius: 3px; font-size: 8.3pt; }
.link { display: flex; flex-wrap: nowrap; justify-content: center; font-family: Consolas, "DejaVu Sans Mono", monospace;
        font-weight: 700; font-size: 12.2pt; margin: 10px 0 30px; }
.seg { position: relative; padding: 5px 2px; border-radius: 5px; margin: 0 1px; white-space: nowrap; }
.sep { padding: 5px 0; color: #8a96a8; }
.seg i { position: absolute; left: 50%; transform: translateX(-50%); top: 33px; width: 18px; height: 18px;
         border-radius: 50%; color: #fff; font: 700 9pt Arial, sans-serif; line-height: 18px; text-align: center;
         font-style: normal; }
.legend { display: grid; grid-template-columns: 1fr 1fr; gap: 4px 14px; font-size: 9pt; }
.legend div { display: flex; gap: 7px; align-items: flex-start; }
.legend i { flex: none; width: 17px; height: 17px; border-radius: 50%; color: #fff; font: 700 8.5pt Arial, sans-serif;
            line-height: 17px; text-align: center; font-style: normal; margin-top: 1px; }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.pb { break-before: page; }
.foot { margin-top: 12px; font-size: 8.3pt; color: #6b7480; border-top: 1px solid #e3e8f0; padding-top: 6px; }
"""

COL = {1: "#8a96a8", 2: "#1f6feb", 3: "#8b5cf6", 4: "#2ea043", 5: "#d9730d", 6: "#8a96a8", 7: "#e11d48", 8: "#0e9aa7"}
BG = {1: "#eceff3", 2: "#e3edff", 3: "#f3e8ff", 4: "#e3f6ea", 5: "#fff0e0", 6: "#eceff3", 7: "#ffe4e6", 8: "#dff5f7"}


def seg(n, text):
    return f'<span class="seg" style="background:{BG[n]};color:{COL[n]}">{text}<i style="background:{COL[n]}">{n}</i></span>'


def sep(t):
    return f'<span class="sep">{t}</span>'


def leg(n, title, text):
    return f'<div><i style="background:{COL[n]}">{n}</i><span><b>{title}:</b> {text}</span></div>'


BRANDS = {
    "hikvision": {
        "name": "Hikvision",
        "link": [seg(1, "rtsp://"), seg(2, "ac"), sep(":"), seg(3, "Clave1234"), sep("@"), seg(4, "nombre.dyndns.org"),
                 sep(":"), seg(5, "21554"), seg(6, "/Streaming/Channels/"), seg(7, "1"), seg(8, "02")],
        "legend": [
            (1, "rtsp://", "siempre igual."),
            (5, "Puerto de afuera", "el que se abre en el router hacia el <code>554</code> del grabador. Uno distinto por grabador."),
            (2, "Usuario", "el que se crea para nosotros (en el ejemplo, <code>ac</code>)."),
            (6, "/Streaming/Channels/", "siempre igual en Hikvision."),
            (3, "Clave", "de ese usuario. Solo letras y números (los símbolos rompen el enlace)."),
            (7, "Cámara", "<code>1</code> = cámara 1 (o D1 en un NVR), <code>2</code> = cámara 2…"),
            (4, "Dirección del local", "el nombre fijo (DynDNS) o la IP pública fija del local."),
            (8, "Calidad", "<code>02</code> = segunda calidad (la que usamos), <code>01</code> = principal."),
        ],
        "examples": [("Grabador 1, cámara 1", "rtsp://ac:Clave1234@nombre.dyndns.org:21554/Streaming/Channels/102"),
                     ("Grabador 1, cámara 12", "rtsp://ac:Clave1234@nombre.dyndns.org:21554/Streaming/Channels/1202"),
                     ("Grabador 2, cámara 3", "rtsp://ac:Clave1234@nombre.dyndns.org:22554/Streaming/Channels/302")],
        "config": [
            ("IP fija", "Configuración → Red → General (TCP/IP): <b>DHCP</b> <span class=off>APAGADO</span>, una IP fija "
             "distinta para cada grabador, máscara <code>255.255.255.0</code>, puerta de enlace = el router, DNS "
             "<code>8.8.8.8</code>."),
            ("Puerto RTSP", "Red → Más ajustes: <code>554</code> (el de fábrica). Si es otro, avisarnos cuál."),
            ("Usuario para nosotros", "Sistema → Usuario → Agregar: tipo <b>Operador</b>, solo <b>vista en vivo</b> y "
             "<b>reproducción</b> (local y remota) en todos los canales. Clave de 8 a 16, solo letras y números."),
            ("Segunda calidad", "Grabar → Parámetros (Codificación) → <b>Substream</b>: <b>H.264</b> (no H.265, ni "
             "H.264+ ni H.265+), <b>640x360</b> o la más parecida, <b>8</b> cuadros/s, <b>384</b> kbps. «Copiar» a "
             "todas las cámaras."),
            ("Cifrado de transmisión", "Red → Acceso a plataforma (Hik-Connect): «Cifrado de transmisión» "
             "<span class=off>APAGADO</span>. Si está prendido, el video no se puede leer."),
            ("H.265 automático", "En un NVR con cámaras IP: «Habilitar H.265 (para uso inicial)» "
             "<span class=off>APAGADO</span> (si no, vuelve a pasar las cámaras a H.265)."),
            ("UPnP", "Red → NAT: «Activar» <span class=off>APAGADO</span> (que el grabador no abra puertos solo)."),
            ("Bloqueo por intentos", "Sistema → Seguridad: «Bloqueo de inicio de sesión ilegal» "
             "<span class=on>PRENDIDO</span>."),
        ],
        "noports": "ni el 8000 (el de iVMS), ni el 80, ni el 443", "linksize": "12.2pt",
    },
    "dahua": {
        "name": "Dahua",
        "link": [seg(1, "rtsp://"), seg(2, "ac"), sep(":"), seg(3, "Clave1234"), sep("@"), seg(4, "nombre.dyndns.org"),
                 sep(":"), seg(5, "21554"), seg(6, "/cam/realmonitor?channel="), seg(7, "1"), seg(6, "&amp;subtype="),
                 seg(8, "1")],
        "legend": [
            (1, "rtsp://", "siempre igual."),
            (5, "Puerto de afuera", "el que se abre en el router hacia el <code>554</code> del grabador. Uno distinto por grabador."),
            (2, "Usuario", "el que se crea para nosotros (en el ejemplo, <code>ac</code>)."),
            (6, "/cam/realmonitor?channel= y &amp;subtype=", "siempre igual en Dahua."),
            (3, "Clave", "de ese usuario. Solo letras y números (los símbolos rompen el enlace)."),
            (7, "Cámara", "<code>1</code> = canal 1, <code>2</code> = canal 2…"),
            (4, "Dirección del local", "el nombre fijo (DynDNS) o la IP pública fija del local."),
            (8, "Calidad", "<code>1</code> = segunda calidad (la que usamos), <code>0</code> = principal."),
        ],
        "examples": [("Grabador 1, canal 1", "rtsp://ac:Clave1234@nombre.dyndns.org:21554/cam/realmonitor?channel=1&amp;subtype=1"),
                     ("Grabador 1, canal 12", "rtsp://ac:Clave1234@nombre.dyndns.org:21554/cam/realmonitor?channel=12&amp;subtype=1"),
                     ("Grabador 2, canal 3", "rtsp://ac:Clave1234@nombre.dyndns.org:22554/cam/realmonitor?channel=3&amp;subtype=1")],
        "config": [
            ("IP fija", "Menú principal → Red → TCP/IP: modo <b>Estático</b> (DHCP <span class=off>APAGADO</span>), una "
             "IP fija distinta para cada grabador, máscara <code>255.255.255.0</code>, puerta de enlace = el router, "
             "DNS <code>8.8.8.8</code>."),
            ("Puerto RTSP", "Red → Puerto (Conexión): RTSP <code>554</code> (el de fábrica). Si es otro, avisarnos cuál."),
            ("Usuario para nosotros", "Sistema → Cuenta → Agregar usuario: grupo <b>user</b> (no admin), permisos solo "
             "<b>Vista en vivo / Monitor</b> y <b>Reproducción</b> en todos los canales. Clave de 8 a 32, solo letras y "
             "números."),
            ("Segunda calidad", "Cámara → Codificación → <b>Flujo secundario</b> (Sub Stream / Extra Stream): "
             "<b>H.264</b> (H.264H o H.264B también sirven; no H.265), <b>Codificación inteligente</b> (Smart Codec) "
             "<span class=off>APAGADA</span>, <b>640x360</b> o la más parecida (VGA, D1 o CIF), <b>8</b> cuadros/s, "
             "<b>384</b> kbps. «Copiar» a los demás canales."),
            ("Cifrado del video", "Si el equipo tiene «Transmisión cifrada», «RTSP sobre TLS» o «Cifrado de video» "
             "(en Seguridad), dejarlo <span class=off>APAGADO</span>."),
            ("UPnP", "Red → UPnP: <span class=off>APAGADO</span> (que el grabador no abra puertos solo)."),
            ("Bloqueo por intentos", "Seguridad (o Sistema → Cuenta): bloqueo de cuenta por intentos fallidos "
             "<span class=on>PRENDIDO</span> (de fábrica suele estar)."),
        ],
        "noports": "ni el 37777 / 37778 (los de DMSS y SmartPSS), ni el 80, ni el 443", "linksize": "9.4pt",
    },
}


def page(key):
    b = BRANDS[key]
    ex = "".join(f"<tr><td>{t}</td><td><code>{u}</code></td></tr>" for t, u in b["examples"])
    cfg = "".join(f"<tr><td>{a}</td><td>{c}</td></tr>" for a, c in b["config"])
    leg_html = "".join(leg(*x) for x in b["legend"])
    return f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<title>Conectar cámaras {b['name']} · Analítica Crítica</title><style>{CSS}</style></head><body>
<div class="head"><img src="{LOGO}" alt=""><div><h1>Conectar las cámaras {b['name']} a Analítica Crítica</h1>
<p>Qué necesitamos del grabador y del router para ver las cámaras · Guía para el técnico del local</p></div></div>

<p class="lead">Nuestro sistema se conecta al grabador por internet con un enlace <b>RTSP</b> por cámara (el mismo que
abre VLC). No hace falta instalar nada en el local: solo un usuario en el grabador, unos ajustes y <b>un puerto abierto
en el router por cada grabador</b>.</p>

<div class="box ok"><b>En resumen, necesitamos:</b>
<ul>
<li>Cada grabador con <b>IP fija</b> y el puerto RTSP <code>554</code>.</li>
<li>Un <b>usuario para nosotros</b> con permiso solo de <b>vista en vivo y reproducción</b>.</li>
<li>La <b>segunda calidad</b> de las cámaras en <b>H.264</b>, 640x360, 8 cuadros por segundo.</li>
<li>En el router, <b>un puerto de afuera por grabador</b> hacia su <code>554</code> (TCP).</li>
<li>Que nos manden los datos del punto 5, por privado.</li>
</ul></div>

<h2><span class="n">1</span>Cómo se arma el enlace de cada cámara</h2>
<p>Ejemplo con datos inventados. Cada color es una parte:</p>
<div class="link" style="font-size:{b['linksize']}">{''.join(b['link'])}</div>
<div class="legend">{leg_html}</div>
<table><tr><th>Ejemplo</th><th>Enlace</th></tr>{ex}</table>

<h2><span class="n">2</span>Configuración de cada grabador</h2>
<p>Con monitor y mouse en el grabador, entrando con el usuario administrador. Los nombres de los menús pueden variar un
poco según el modelo y la versión.</p>
<table><tr><th>Qué</th><th>Cómo dejarlo</th></tr>{cfg}</table>

<h2><span class="n">3</span>Puertos en el router</h2>
<p>En el router del local (menú <b>Reenvío de puertos</b>, <b>Port Forwarding</b>, <b>Servidor virtual</b> o
<b>NAT</b>), una regla por grabador, protocolo <b>TCP</b>:</p>
<table><tr><th>Puerto de afuera</th><th>Va a</th><th>Puerto del grabador</th></tr>
<tr><td>21554</td><td>IP fija del grabador 1</td><td>554</td></tr>
<tr><td>22554</td><td>IP fija del grabador 2</td><td>554</td></tr>
<tr><td>23554 …</td><td>IP fija del grabador 3 (y así)</td><td>554</td></tr></table>
<ul>
<li><b>Dos routers en fila</b> (por ejemplo, el de Antel y otro conectado a él): se abre en los dos. En el de la calle,
los mismos puertos hacia la IP del segundo router (con el mismo número de los dos lados), y en el segundo, hacia cada
grabador. Al segundo router hay que dejarle una IP fija.</li>
<li>Pueden usar otros números de afuera (entre 10000 y 65535): avísennos cuáles quedaron.</li>
<li>No hace falta abrir ningún otro puerto: {b['noports']}.</li>
</ul>

<h2><span class="n">4</span>Dirección del local y prueba</h2>
<ul>
<li><b>Dirección:</b> si el local tiene <b>IP pública fija</b> contratada, usamos esa. Si no (la IP común cambia sola),
usamos un <b>nombre fijo de DynDNS</b> que ponemos nosotros: lo cargamos en el grabador 1 o les pasamos los datos para el
router.</li>
<li><b>Prueba:</b> desde un celular <b>con datos móviles y el wifi apagado</b>, abrir la app <b>VLC</b> → Red → pegar el
enlace de la cámara 1. Si se ve, está listo. Desde adentro del mismo local la prueba puede fallar aunque esté todo bien.</li>
</ul>

<h2><span class="n">5</span>Qué mandarnos</h2>
<table><tr><th>Dato</th><th>Ejemplo</th></tr>
<tr><td>Marca y modelo de cada grabador</td><td>{b['name']} · etiqueta de abajo del equipo</td></tr>
<tr><td>Puerto de afuera de cada grabador</td><td>Grabador 1 → 21554, grabador 2 → 22554</td></tr>
<tr><td>IP fija de cada grabador</td><td>192.168.1.201, 192.168.1.202</td></tr>
<tr><td>Usuario y clave para nosotros</td><td><b>Por privado</b>, nunca por un grupo</td></tr>
<tr><td>IP pública fija (si tiene)</td><td>La que figura en el contrato con Antel u otro proveedor</td></tr>
<tr><td>Canales que se usan y qué mira cada uno</td><td>1 Caja · 2 Entrada · 3 Góndola bebidas…</td></tr>
</table>

<div class="box warn"><b>Seguridad:</b> el usuario que nos crean solo puede ver, no cambiar nada. Con puertos altos
(no el 554 de fábrica), una clave sin datos personales y el bloqueo por intentos prendido, el grabador queda mucho más
protegido. Si cambian la clave, avísennos para actualizarla.</div>

<p class="foot">Analítica Crítica · Guía para el técnico del local · Los datos de los ejemplos son inventados.</p>
</body></html>"""


def main():  # el PDF: python -m tools.guias_conexion (o Playwright de Node con el HTML)
    from playwright.sync_api import sync_playwright
    out = REPO / "docs"
    with sync_playwright() as pw:
        br = pw.chromium.launch(executable_path="/opt/pw-browsers/chromium")
        pg = br.new_page()
        for key in BRANDS:
            html = out / f"conectar-camaras-{key}.html"
            html.write_text(page(key), encoding="utf-8")
            pg.goto(html.as_uri())
            pg.pdf(path=str(out / f"conectar-camaras-{key}.pdf"), format="A4", print_background=True,
                   margin={"top": "0", "bottom": "0", "left": "0", "right": "0"}, prefer_css_page_size=True)
            print("ok", key)
        br.close()


if __name__ == "__main__":
    main()
