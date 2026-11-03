"""Video para clientes (python -m tools.video_clientes): video explicativo para clientes (sin voz, con subtítulos grabados en la imagen y un .srt aparte).

Cada escena es una pantalla con la marca y un dibujo simple; cada subtítulo es un cuadro (escena + frase). Playwright
saca las imágenes a 1920x1080 y ffmpeg las une (fundido al entrar y salir de cada escena)."""
import base64
import subprocess
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "docs/video-clientes"
WORK = Path("/tmp/video-clientes-cuadros")
WORK.mkdir(parents=True, exist_ok=True)
LOGO = base64.b64encode((REPO / "docs/guia-operadores/logo.png").read_bytes()).decode()
LOGO_W = base64.b64encode((REPO / "public/img/logo-blanco.png").read_bytes()).decode()

NAVY, OK, WARN, BAD, MUTED = "#011b5d", "#2f8f46", "#b7791f", "#c0392b", "#5b6470"

ICONS = {
    "camara": '<svg viewBox="0 0 120 90"><rect x="8" y="22" width="78" height="50" rx="10" fill="{c}"/>'
              '<polygon points="86,38 114,24 114,70 86,56" fill="{c}"/><circle cx="34" cy="47" r="12" fill="#fff"/>'
              '<circle cx="34" cy="47" r="6" fill="{c}"/></svg>',
    "ojo": '<svg viewBox="0 0 120 90"><path d="M8 45 Q60 -5 112 45 Q60 95 8 45Z" fill="{c}"/>'
           '<circle cx="60" cy="45" r="20" fill="#fff"/><circle cx="60" cy="45" r="10" fill="{c}"/></svg>',
    "persona": '<svg viewBox="0 0 120 90"><circle cx="60" cy="26" r="17" fill="{c}"/>'
               '<path d="M26 88 Q26 50 60 50 Q94 50 94 88Z" fill="{c}"/></svg>',
    "aviso": '<svg viewBox="0 0 120 90"><rect x="36" y="4" width="48" height="82" rx="8" fill="{c}"/>'
             '<rect x="42" y="14" width="36" height="56" rx="3" fill="#fff"/><rect x="47" y="22" width="26" height="6" rx="3" '
             'fill="{c}"/><rect x="47" y="33" width="20" height="6" rx="3" fill="{c}"/><circle cx="60" cy="78" r="4" '
             'fill="#fff"/></svg>',
    "router": '<svg viewBox="0 0 120 90"><rect x="10" y="46" width="100" height="32" rx="8" fill="{c}"/>'
              '<rect x="28" y="14" width="6" height="34" fill="{c}"/><rect x="86" y="14" width="6" height="34" fill="{c}"/>'
              '<circle cx="30" cy="62" r="5" fill="#fff"/><circle cx="48" cy="62" r="5" fill="#fff"/></svg>',
    "nube": '<svg viewBox="0 0 120 90"><path d="M30 76 Q4 76 8 54 Q12 36 32 38 Q38 12 64 14 Q88 16 92 40 Q116 40 114 60 Q112 '
            '76 92 76Z" fill="{c}"/><path d="M44 52 L56 64 L80 38" stroke="#fff" stroke-width="9" fill="none" '
            'stroke-linecap="round" stroke-linejoin="round"/></svg>',
    "llave": '<svg viewBox="0 0 120 90"><circle cx="34" cy="45" r="24" fill="{c}"/><circle cx="34" cy="45" r="10" fill="#fff"/>'
             '<rect x="54" y="39" width="60" height="12" rx="4" fill="{c}"/><rect x="92" y="51" width="10" height="16" '
             'fill="{c}"/><rect x="76" y="51" width="10" height="12" fill="{c}"/></svg>',
    "candado": '<svg viewBox="0 0 120 90"><path d="M40 40 V28 Q40 8 60 8 Q80 8 80 28 V40" stroke="{c}" stroke-width="10" '
               'fill="none"/><rect x="28" y="38" width="64" height="48" rx="8" fill="{c}"/><circle cx="60" cy="58" r="7" '
               'fill="#fff"/><rect x="57" y="60" width="6" height="14" fill="#fff"/></svg>',
    "basura": '<svg viewBox="0 0 120 90"><rect x="34" y="22" width="52" height="64" rx="6" fill="{c}"/><rect x="26" y="12" '
              'width="68" height="8" rx="4" fill="{c}"/><rect x="50" y="4" width="20" height="8" rx="3" fill="{c}"/>'
              '<rect x="46" y="32" width="6" height="44" rx="3" fill="#fff"/><rect x="68" y="32" width="6" height="44" '
              'rx="3" fill="#fff"/></svg>',
}


def icon(name: str, color: str = NAVY, size: int = 150) -> str:
    return f'<div class="ic" style="width:{size}px;height:{size * 0.75}px">{ICONS[name].format(c=color)}</div>'


def step(name: str, title: str, active: bool, color: str = NAVY) -> str:
    c = color if active else "#c4ccd8"
    return (f'<div class="step {"on" if active else ""}">{icon(name, c, 140)}<div class="st" '
            f'style="color:{c}">{title}</div></div>')


def flow(active: int) -> str:
    items = [("camara", "Sus cámaras"), ("ojo", "Analiza"), ("persona", "Revisamos"), ("aviso", "Le avisamos")]
    out = []
    for i, (n, t) in enumerate(items):
        out.append(step(n, t, i <= active))
        if i < len(items) - 1:
            out.append(f'<div class="arrow" style="color:{"#011b5d" if i < active else "#c4ccd8"}">→</div>')
    return '<div class="flow">' + "".join(out) + "</div>"


def two(left: str, right: str) -> str:
    return f'<div class="two">{left}{right}</div>'


def card(icon_name: str, color: str, title: str, lines: list[str], dim: bool = False) -> str:
    lis = "".join(f"<li>{x}</li>" for x in lines)
    return (f'<div class="card" style="border-top-color:{color};opacity:{0.35 if dim else 1}">{icon(icon_name, color, 120)}'
            f'<div class="ct" style="color:{color}">{title}</div><ul>{lis}</ul></div>')


SCENES = [
    {"title": None, "visual": lambda i: (
        f'<div class="intro"><img src="data:image/png;base64,{LOGO}"><div class="name">Analítica Crítica</div>'
        f'<div class="tag">Análisis de video para comercios, con revisión humana</div></div>'),
     "caps": ["Analítica Crítica mira las cámaras de su comercio por usted.",
              "Y le avisa cuando pasa algo que puede ser un hurto."]},
    {"title": "Cómo funciona", "visual": lambda i: flow(i),
     "caps": ["Usted ya tiene cámaras y un grabador. No hay que cambiar nada.",
              "Nuestro sistema analiza el video mientras pasa: salir sin pagar, entrar a zonas privadas, "
              "esconder un producto.",
              "Una persona de nuestro equipo revisa cada alerta. Las falsas alarmas no le llegan.",
              "Y le avisamos con un link privado al clip. Es un aviso para que mire, no una acusación."]},
    {"title": "Qué guardamos y qué no", "visual": lambda i: two(
        card("basura", BAD, "No grabamos", ["El video se analiza y se descarta", "No reconocemos caras",
                                            "Sus grabaciones siguen en su grabador"], dim=i == 1),
        card("candado", OK, "Solo lo necesario", ["El clip corto de cada alerta", "Se borra a los 60 días",
                                                  "Números de su local, sin imágenes"], dim=i == 0)),
     "caps": ["No grabamos sus cámaras: el video se analiza y se descarta.",
              "Solo guardamos el clip corto de cada alerta, y se borra a los 60 días.",
              "Sus grabaciones siguen en su grabador, como siempre. Y no reconocemos caras."]},
    {"title": "Dos formas de darnos acceso", "visual": lambda i: two(
        card("router", WARN, "Abrir puertos", ["Un técnico toca su router", "Su grabador queda expuesto a internet",
                                               "Para cortarlo, hay que volver a tocarlo"], dim=i == 2),
        card("nube", OK, "Clave de Hik-Connect", ["No se toca el router", "No se instala nada",
                                                  "La borra cuando quiera"], dim=i == 1)),
     "caps": ["Para analizar el video, nuestro sistema tiene que poder verlo. Hay dos formas.",
              "Abrir puertos en el router deja su grabador expuesto a internet y necesita un técnico.",
              "Con una clave de Hik-Connect no se toca el router ni se instala nada. Es la que le recomendamos."]},
    {"title": "¿Qué es la clave de API?", "visual": lambda i: (
        '<div class="keys">' + icon("llave", NAVY, 220) +
        '<div class="kk"><div class="k"><b>AppKey</b><span>el nombre de la llave</span></div>'
        '<div class="k"><b>SecretKey</b><span>la llave secreta</span></div></div></div>'),
     "caps": ["La clave tiene dos partes: la AppKey, que es el nombre, y la SecretKey, la llave secreta.",
              "Es una llave aparte, solo para nosotros. No es la contraseña de su cuenta.",
              "La puede borrar cuando quiera, y en ese mismo momento dejamos de ver sus cámaras."]},
    {"title": "Qué tiene que hacer usted", "visual": lambda i: (
        '<ol class="steps">' + "".join(
            f'<li class="{"on" if (i == 0 and n < 3) or (i == 1 and 3 <= n) or i == 2 else ""}"><span>{n + 1}</span>{t}</li>'
            for n, t in enumerate(["Tener el grabador en la app Hik-Connect", "Apagar el «cifrado de video»",
                                   "Pasar la cuenta al «modo equipo»", "Crear la clave en «Integración de API»",
                                   "Mandarnos la AppKey y la SecretKey por privado"])) + "</ol>"),
     "caps": ["Para darnos acceso: tenga su grabador en Hik-Connect, apague el cifrado de video y pase la cuenta al "
              "modo equipo.",
              "Cree la clave en «Integración de API» y mándenos la AppKey y la SecretKey, por privado.",
              "Le damos un tutorial con cada paso y, si prefiere, lo hacemos con usted."]},
    {"title": None, "visual": lambda i: (
        f'<div class="outro"><img src="data:image/png;base64,{LOGO_W}"><div class="name">Analítica Crítica</div>'
        f'<div class="tag">Nunca le vamos a pedir la contraseña de su cuenta.</div></div>'), "dark": True,
     "caps": ["Nunca le vamos a pedir la contraseña de su cuenta.",
              "Analítica Crítica: cuidamos su comercio, con revisión humana."]},
]

CSS = f"""
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ width: 1280px; height: 720px; font-family: Inter, "DejaVu Sans", sans-serif; background: #fff; color: #1b1f24;
       overflow: hidden; position: relative; }}
body.dark {{ background: {NAVY}; color: #fff; }}
.top {{ position: absolute; top: 0; left: 0; right: 0; height: 74px; display: flex; align-items: center; gap: 14px;
        padding: 0 40px; border-bottom: 3px solid {NAVY}; }}
.top img {{ width: 44px; height: 44px; }}
.top .b {{ font-weight: 800; letter-spacing: 3px; text-transform: uppercase; color: {NAVY}; font-size: 15px; }}
.top h1 {{ margin-left: auto; font-size: 30px; color: {NAVY}; font-weight: 800; }}
.stage {{ position: absolute; top: 74px; left: 0; right: 0; bottom: 150px; display: flex; align-items: center;
          justify-content: center; }}
.cap {{ position: absolute; left: 70px; right: 70px; bottom: 34px; min-height: 92px; display: flex; align-items: center;
        justify-content: center; text-align: center; background: rgba(10,14,30,.86); color: #fff; font-size: 30px;
        line-height: 1.3; font-weight: 600; border-radius: 14px; padding: 14px 28px; }}
body.dark .cap {{ background: rgba(255,255,255,.14); }}
.ic svg {{ width: 100%; height: 100%; }}
.intro, .outro {{ text-align: center; }}
.intro img, .outro img {{ width: 190px; height: 190px; }}
.intro .name, .outro .name {{ font-size: 58px; font-weight: 800; color: {NAVY}; margin-top: 6px; }}
.outro .name {{ color: #fff; }}
.intro .tag, .outro .tag {{ font-size: 25px; color: {MUTED}; margin-top: 8px; }}
.outro .tag {{ color: #dfe6f3; }}
.flow {{ display: flex; align-items: center; gap: 14px; }}
.step {{ width: 220px; text-align: center; }}
.step .ic {{ margin: 0 auto; }}
.step .st {{ font-size: 25px; font-weight: 700; margin-top: 10px; }}
.arrow {{ font-size: 54px; font-weight: 800; margin-top: -40px; }}
.two {{ display: flex; gap: 40px; }}
.card {{ width: 480px; border: 2px solid #d9dee5; border-top: 8px solid; border-radius: 18px; padding: 20px 28px;
         background: #fff; }}
.card .ic {{ margin: 0 auto 6px; }}
.card .ct {{ font-size: 30px; font-weight: 800; text-align: center; margin-bottom: 8px; }}
.card ul {{ list-style: none; font-size: 23px; line-height: 1.5; }}
.card li::before {{ content: "•  "; color: #9aa4b2; }}
.keys {{ display: flex; align-items: center; gap: 50px; }}
.kk {{ display: flex; flex-direction: column; gap: 18px; }}
.k {{ background: #f4f6f8; border-left: 8px solid {NAVY}; border-radius: 12px; padding: 14px 26px; width: 520px; }}
.k b {{ display: block; font-size: 38px; color: {NAVY}; }}
.k span {{ font-size: 26px; color: {MUTED}; }}
.steps {{ list-style: none; display: flex; flex-direction: column; gap: 12px; }}
.steps li {{ font-size: 28px; font-weight: 600; color: #b8c0cc; display: flex; align-items: center; gap: 18px; }}
.steps li span {{ width: 46px; height: 46px; border-radius: 50%; background: #d4dae3; color: #fff; display: flex;
                 align-items: center; justify-content: center; font-weight: 800; }}
.steps li.on {{ color: #1b1f24; }}
.steps li.on span {{ background: {NAVY}; }}
"""


def page(scene: dict, i: int, cap: str) -> str:
    dark = scene.get("dark")
    top = "" if scene["title"] is None else (
        f'<div class="top"><img src="data:image/png;base64,{LOGO}"><div class="b">Analítica Crítica</div>'
        f'<h1>{scene["title"]}</h1></div>')
    return (f'<!doctype html><html lang="es"><head><meta charset="utf-8"><style>{CSS}</style></head>'
            f'<body class="{"dark" if dark else ""}">{top}<div class="stage" style="{"top:0" if scene["title"] is None else ""}">'
            f'{scene["visual"](i)}</div><div class="cap">{cap}</div></body></html>')


def seconds(text: str) -> float:
    return round(max(3.6, len(text) / 14.5 + 1.4), 1)  # ~14 caracteres por segundo, para leer tranquilo


def srt_time(t: float) -> str:
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s - int(s)) * 1000)):03d}"


def main() -> None:
    shots = []  # (png, duración, entra la escena, sale la escena, texto)
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path="/opt/pw-browsers/chromium")
        pg = b.new_page(viewport={"width": 1280, "height": 720}, device_scale_factor=1.5)
        for si, sc in enumerate(SCENES):
            for ci, cap in enumerate(sc["caps"]):
                pg.set_content(page(sc, ci, cap))
                pg.wait_for_timeout(120)
                png = WORK / f"s{si}_{ci}.png"
                pg.screenshot(path=str(png))
                shots.append((png, seconds(cap), ci == 0, ci == len(sc["caps"]) - 1, cap))
        b.close()
    clips, t, srt = [], 0.0, []
    for k, (png, dur, first, last, cap) in enumerate(shots):
        fades = []
        if first:
            fades.append("fade=t=in:st=0:d=0.45")
        if last:
            fades.append(f"fade=t=out:st={dur - 0.45:.2f}:d=0.45")
        vf = ",".join(["scale=1920:1080", *fades, "format=yuv420p"])
        clip = WORK / f"c{k:02d}.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-t", str(dur), "-i", str(png), "-vf", vf,
                        "-r", "25", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-tune", "stillimage",
                        str(clip)], check=True)
        clips.append(clip)
        srt.append(f"{k + 1}\n{srt_time(t)} --> {srt_time(t + dur)}\n{cap}\n")
        t += dur
    lst = WORK / "lista.txt"
    lst.write_text("".join(f"file '{c}'\n" for c in clips))
    video = OUT / "analitica-critica-como-funciona.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
                    "-c:v", "copy", "-c:a", "aac", "-shortest", "-movflags", "+faststart", str(video)], check=True)
    (OUT / "analitica-critica-como-funciona.srt").write_text("\n".join(srt), encoding="utf-8")
    print(video, f"{t:.1f} s", f"{video.stat().st_size / 1e6:.1f} MB", len(shots), "subtítulos")


if __name__ == "__main__":
    main()
