"""Reporte de métricas de un cliente en PDF (web y app de escritorio: el mismo archivo para los dos).

Lo arma `metrics_pdf` con lo que ya calcula el servidor (cubos horarios de `camera_metrics` sumados por día, hora,
cámara y zona) y las alertas del período. Con fpdf2 (puro Python, sin navegador): letras base de PDF, así que los
textos se pasan a latin-1 (`latin`). Colores de la marca, como la app.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from fpdf import FPDF

from core.metrics import WAIT_MIN_SECONDS, post_stats, wait_stats

MARCA = (1, 27, 93)  # azul del logo (#011b5d)
SUAVE = (102, 112, 133)
TEXTO = (29, 36, 51)
BORDE = (216, 222, 233)
FONDO = (243, 245, 249)
CELESTE = (61, 107, 214)
LOGO = Path(__file__).with_name("logo.png")  # el ojo en blanco (sobre la franja azul)
PLAN_NAMES = {"hurto": "Detección de hurtos", "comercio": "Métricas de comercio", "estacion": "Estación de servicio",
              "empleados": "Seguimiento de empleados"}
ZONE_NAMES = {"shelf": "Góndola", "aisle": "Pasillo", "checkout": "Caja", "exit": "Salida", "restricted": "Restringida",
              "entrance": "Entrada", "pump": "Surtidor", "shop": "Tienda", "staff": "Personal"}
LABELS = {"hurto": "Confirmadas como hurto", "sospechoso": "Sospechosas", "normal": "Normales",
          "falso_positivo": "Falsas alarmas", "sin_etiqueta": "Sin revisar"}
DAYS_ES = ("lun", "mar", "mié", "jue", "vie", "sáb", "dom")


def latin(text) -> str:
    """Las letras base del PDF son latin-1: lo que no entra se cambia por algo parecido."""
    t = str(text)
    for a, b in (("—", "-"), ("–", "-"), ("…", "..."), ("→", "->"), ("≈", "~"), ("“", '"'), ("”", '"'), ("’", "'"),
                 ("✔", "v"), ("✘", "x"), ("★", "*")):
        t = t.replace(a, b)
    return t.encode("latin-1", "replace").decode("latin-1")


def dur(seconds: float | None) -> str:
    if not seconds:
        return "-"
    s = int(round(seconds))
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h} h {m:02d} min" if h else f"{m} min {s:02d} s" if m else f"{s} s"


def clock(seconds: float | None) -> str:
    """Tiempos de espera y de los puestos: «1:20» (minutos:segundos) o, de una hora para arriba, «1 h 12 min»."""
    if seconds is None:
        return "-"
    s = round(seconds)
    if s >= 3600:
        h, m = divmod(round(s / 60), 60)
        return f"{h} h {m} min"
    m, s = divmod(s, 60)
    return f"{m}:{s:02d}"


def minutes(seconds: float | None) -> str:
    """«12 min» (o «1 h 12 min»): para los ratos largos que se dicen en una frase."""
    if not seconds:
        return "0 min"
    if seconds < 30:
        return "menos de 1 min"
    m = int(seconds / 60 + 0.5)  # (round() redondea 30 s a «0 min»)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{h} h {m} min"
    return f"{m} min"


def dec(v: float | None, digits: int = 1) -> str:
    """Número con coma decimal («1,3»)."""
    return "-" if v is None else f"{v:.{digits}f}".replace(".", ",")


def wait_zones(bucket: dict) -> list[tuple[str, dict]]:
    """Las zonas de espera con algo medido (`core.metrics.wait_stats` de cada una), la de más esperas primero."""
    out = [(name, wait_stats(w)) for name, w in (bucket.get("waits") or {}).items()]
    return sorted([(n, w) for n, w in out if w["n"] or w["obs_s"]], key=lambda kv: (-kv[1]["n"], kv[0]))


def wait_total(zones: list[tuple[str, dict]]) -> dict:
    """Todas las zonas de espera juntas: esperas y segundos sumados (→ promedio), el máximo el mayor."""
    n = sum(w["n"] for _, w in zones)
    total = sum(w["avg_s"] * w["n"] for _, w in zones if w["n"])
    return {"n": n, "avg_s": total / n if n else None,
            "max_s": max((w["max_s"] for _, w in zones if w["n"]), default=None),
            "over5": sum(w["over"][300] for _, w in zones), "queue_max": max((w["queue_max"] for _, w in zones), default=0)}


def avg(total: float, n: int) -> float | None:
    return total / n if n else None


def fmt_day(iso: str) -> str:
    d = date.fromisoformat(iso)
    return f"{DAYS_ES[d.weekday()]} {d:%d/%m}"


class Reporte(FPDF):
    def __init__(self, title: str):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.title_text = title
        self.set_auto_page_break(auto=True, margin=16)
        self.set_margins(14, 14, 14)
        self.set_title(latin(title))
        self.set_author("Analitica Critica")

    def header(self) -> None:
        self.set_fill_color(*MARCA)
        self.rect(0, 0, 210, 18, style="F")
        x = 14
        if LOGO.exists():
            try:
                self.image(str(LOGO), x=x, y=3.5, h=11)
                x += 14
            except Exception:  # noqa: BLE001 - sin logo sigue igual
                pass
        self.set_xy(x, 5)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 14)
        self.cell(80, 8, latin("Analítica Crítica"))
        self.set_font("Helvetica", "", 10)
        self.set_xy(110, 5)
        self.cell(86, 8, latin(self.title_text), align="R")
        self.set_text_color(*TEXTO)
        self.set_y(24)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*SUAVE)
        self.cell(0, 6, latin(f"Generado el {datetime.now():%d/%m/%Y %H:%M} · página {self.page_no()} de {{nb}}"),
                  align="R")

    # ---------- piezas ----------
    def h2(self, text: str) -> None:
        if self.get_y() > 228:  # que el título no quede solo al pie: va con lo que sigue
            self.add_page()
        self.ln(4)
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(*MARCA)
        self.cell(0, 8, latin(text), new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(*TEXTO)

    def note(self, text: str, size: float = 8.5) -> None:
        self.set_font("Helvetica", "", size)
        self.set_text_color(*SUAVE)
        self.multi_cell(0, 4.3, latin(text), new_x="LMARGIN", new_y="NEXT", align="L")
        self.set_text_color(*TEXTO)

    def tiles(self, items: list[tuple[str, str, str]]) -> None:
        """Números grandes en recuadros: (valor, qué es, aclaración)."""
        if not items:
            return
        gap, w = 4, (182 - 4 * (len(items) - 1)) / len(items)
        y = self.get_y() + 1
        for i, (value, label, hint) in enumerate(items):
            x = 14 + i * (w + gap)
            self.set_fill_color(*FONDO)
            self.set_draw_color(*BORDE)
            self.rect(x, y, w, 25, style="DF")
            self.set_xy(x + 3, y + 2.5)
            self.set_font("Helvetica", "B", 17)
            self.set_text_color(*MARCA)
            self.cell(w - 6, 8, latin(value))
            self.set_xy(x + 3, y + 11.5)
            self.set_font("Helvetica", "B", 8.5)
            self.set_text_color(*TEXTO)
            self.cell(w - 6, 4.5, latin(label))
            self.set_xy(x + 3, y + 16.5)
            self.set_font("Helvetica", "", 7.5)
            self.set_text_color(*SUAVE)
            self.multi_cell(w - 6, 3.5, latin(hint), align="L")
        self.set_text_color(*TEXTO)
        self.set_y(y + 29)

    def bars(self, title: str, rows: list[tuple[str, float]], unit: str) -> None:
        """Barras verticales con el valor arriba de cada una."""
        if self.get_y() > 220:
            self.add_page()
        self.set_font("Helvetica", "B", 10)
        self.cell(0, 6, latin(title), new_x="LMARGIN", new_y="NEXT")
        if not rows or not any(v for _, v in rows):
            self.note(f"Sin {unit} en el período.")
            return
        top = max(v for _, v in rows) or 1
        x0, y0, w, h = 16, self.get_y() + 4, 178, 40
        step = w / len(rows)
        bw = max(1.5, min(14, step * 0.7))
        self.set_draw_color(*BORDE)
        self.line(x0, y0 + h, x0 + w, y0 + h)
        for i, (label, v) in enumerate(rows):
            bh = h * v / top
            x = x0 + i * step + (step - bw) / 2
            self.set_fill_color(*MARCA if v == top else CELESTE)
            if bh > 0:
                self.rect(x, y0 + h - bh, bw, bh, style="F")
            self.set_font("Helvetica", "", 6.5 if len(rows) > 16 else 7.5)
            self.set_text_color(*TEXTO)
            if v:
                self.set_xy(x - 3, y0 + h - bh - 4)
                self.cell(bw + 6, 3.5, f"{v:g}", align="C")
            self.set_text_color(*SUAVE)
            if len(rows) <= 16 or i % 2 == 0:
                self.set_xy(x - 4, y0 + h + 1)
                self.cell(bw + 8, 3.5, latin(label), align="C")
        self.set_text_color(*TEXTO)
        self.set_y(y0 + h + 8)

    def fit(self, text: str, width: float) -> str:
        """El texto recortado con «...» para que no pise la columna de al lado (nombres largos de zona o cámara)."""
        if self.get_string_width(text) <= width:
            return text
        while text and self.get_string_width(text + "...") > width:
            text = text[:-1]
        return text.rstrip() + "..."

    def table(self, head: list[str], rows: list[list[str]], widths: list[float]) -> None:
        if not rows:
            return
        if self.get_y() > 258:
            self.add_page()
        self.set_font("Helvetica", "B", 8.5)
        self.set_fill_color(*FONDO)
        self.set_text_color(*SUAVE)
        self.set_draw_color(*BORDE)
        for h, w in zip(head, widths):
            self.cell(w, 7, latin(h), border="B", fill=True)
        self.ln()
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*TEXTO)
        for r in rows:
            if self.get_y() > 272:
                self.add_page()
            for c, w in zip(r, widths):
                self.cell(w, 6.5, self.fit(latin(c), w - 1.5), border="B")
            self.ln()
        self.ln(2)


def metrics_pdf(client: str, plans: list[str], since: date, until: date, m: dict, alerts: dict | None) -> bytes:
    """`m`: lo de `/api/metrics` (total, by_day, by_hour, by_camera); `alerts`: conteo por etiqueta (o None)."""
    t = m["total"]
    one_day = since == until
    pdf = Reporte("Reporte de métricas")
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(*MARCA)
    pdf.cell(0, 10, latin(client), new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(*TEXTO)
    period = f"Día {since:%d/%m/%Y}" if one_day else f"Del {since:%d/%m/%Y} al {until:%d/%m/%Y}"
    pdf.note(f"{period} · hora del local · planes: "
             + (", ".join(PLAN_NAMES.get(p, p) for p in plans) or "ninguno"), size=9.5)
    if not t.get("frames"):
        pdf.ln(4)
        pdf.note("Todavía no hay métricas en este período: salen de las cámaras que se analizan (llegan cada 5 "
                 "minutos).", size=10)

    zones = sorted((t.get("zones") or {}).items(), key=lambda kv: -kv[1].get("visits", 0))
    limits = m.get("zone_limits") or {}  # «demora estándar» por sector: se marca «(demora)» donde el promedio la supera
    show_people = "comercio" in plans or ("estacion" not in plans and t.get("people"))
    if show_people:
        pdf.h2("Gente")
        if t.get("door"):
            first = (str(t["entries"]), "Entraron",
                     f"{t['exits']} salieron" + (f" · {t['staff']} de personal descontados" if t.get("staff") else ""))
        elif t.get("entries"):
            first = (str(t["entries"]), "Pasaron por la entrada", "entrando o saliendo (falta marcar el sentido)")
        else:
            first = ("-", "Entraron", "falta la zona Entrada (con su sentido) en la cámara de la puerta")
        tiles = [first,
                 (dur(avg(t["people_dwell_s"], t["people"])), "Tiempo promedio", "en cámara, por persona"),
                 (str(t["people_max"]), "Máximo a la vez", "personas al mismo tiempo")]
        if zones:
            tiles.append((str(zones[0][1].get("visits", 0)), "Zona más visitada", zones[0][0]))
        pdf.tiles(tiles)
        # Sin «apariciones» (suma de personas de todas las cámaras): la misma persona cuenta en cada una y de nuevo si
        # se pierde y vuelve. Cuánta gente vino lo dice la puerta.
        if t.get("door") or t.get("entries"):
            title = "Entraron" if t.get("door") else "Pasaron por la entrada"
            what = "ingresos" if t.get("door") else "pasadas"
            if one_day:
                hours = {h["hour"]: h for h in m.get("by_hour") or []}
                lo = min(hours, default=8)
                hi = max(hours, default=20)
                pdf.bars(f"{title} por hora",
                         [(f"{h:02d}", float(hours.get(h, {}).get("entries", 0))) for h in range(lo, hi + 1)], what)
            else:
                pdf.bars(f"{title} por día",
                         [(fmt_day(d["date"]), float(d.get("entries", 0))) for d in m.get("by_day") or []], what)
        if not one_day and m.get("by_day"):
            pdf.table(["Día", "Entraron", "Salieron", "Tiempo promedio", "Máximo a la vez"],
                      [[fmt_day(d["date"]), str(d.get("entries", 0)) if d.get("door") or d.get("entries") else "-",
                        str(d.get("exits", 0)) if d.get("door") else "-",
                        dur(avg(d.get("people_dwell_s", 0), d.get("people", 0))),
                        str(d.get("people_max", 0))] for d in m["by_day"]],
                      [36, 32, 32, 44, 38])
        if zones:
            pdf.h2("A dónde van")

            def _zrow(name, z):
                prom = avg(z.get("dwell_s", 0), z.get("visits", 0))
                lim = limits.get(name)
                txt = dur(prom) + (" (demora)" if lim and z.get("visits") and prom >= lim else "")
                return [name, ZONE_NAMES.get(z.get("type"), z.get("type") or ""), str(z.get("visits", 0)), txt]
            pdf.table(["Zona", "Tipo", "Visitas", "Tiempo promedio"],
                      [_zrow(name, z) for name, z in zones], [70, 40, 30, 42])
    if "estacion" in plans:
        pdf.h2("Estación de servicio")
        pdf.tiles([(str(t["vehicles"]), "Vehículos", f"{t['pump_vehicles']} pasaron por un surtidor"),
                   (dur(avg(t["pump_wait_s"], t["pump_wait_n"])), "Espera hasta el pistero",
                    f"{t['pump_wait_n']} atendidos" if t["pump_wait_n"] else "necesita zona Surtidor"),
                   (str(t["pump_unattended"]), "Sin atención", "vehículos sin que llegara nadie"),
                   (str(t["shop_visits"]), "Bajaron a la tienda", "personas en la zona Tienda")])
        if not one_day:
            pdf.bars("Vehículos por día", [(fmt_day(d["date"]), float(d.get("vehicles", 0)))
                                           for d in m.get("by_day") or []], "vehículos")
    if len(m.get("by_camera") or []) > 1:
        pdf.h2("Por cámara")
        est = "estacion" in plans

        def people(c: dict) -> bool:  # la cámara suma gente (si el cliente mide comercio, solo las que lo miden)
            meas = c.get("measures")
            return meas is None or "comercio" not in plans or "comercio" in meas

        def cars(c: dict) -> bool:
            meas = c.get("measures")
            return meas is None or "estacion" in meas
        pdf.table(["Cámara", "Tiempo promedio", "Máximo a la vez"] + (["Vehículos"] if est else []),
                  [[c["name"], dur(avg(c["people_dwell_s"], c["people"])) if people(c) else "-",
                    str(c["people_max"]) if people(c) else "-"]
                   + ([str(c["vehicles"]) if cars(c) else "-"] if est else []) for c in m["by_camera"]],
                  [90, 44, 38] + ([22] if est else []))
    waits = wait_zones(t) if "comercio" in plans else []
    if waits:
        pdf.h2("Tiempo de espera")
        tot = wait_total(waits)
        pdf.tiles([(clock(tot["avg_s"]), "Espera promedio",
                    f"{tot['n']} espera{'s' if tot['n'] != 1 else ''} en {len(waits)} zona{'s' if len(waits) != 1 else ''}"),
                   (clock(tot["max_s"]), "Espera más larga", "el cliente que más esperó"),
                   (str(tot["over5"]), "Más de 5 min", "clientes que esperaron 5 minutos o más"),
                   (str(tot["queue_max"]), "Fila máxima", "personas esperando a la vez")])
        pdf.table(["Zona", "Esperas", "Promedio", "Máximo", "Más de 5 min", "Más de 10 min", "Fila prom.", "Fila máx."],
                  [[name, str(w["n"]), clock(w["avg_s"]), clock(w["max_s"]), str(w["over"][300]), str(w["over"][600]),
                    dec(w["queue_avg"]), str(w["queue_max"])] for name, w in waits],
                  [40, 17, 20, 20, 22, 24, 19, 20])
        for name, w in waits:
            if w["staffed"]:
                pdf.note(f"{name}: con clientes esperando y nadie en su puesto (la zona de personal más cercana): "
                         f"{minutes(w['unattended_s'])} · se fueron sin que hubiera nadie: {w['left_alone']}", size=9)
        if not any(w["staffed"] for _, w in waits):
            pdf.note("Para saber si había alguien para atender, marcá la zona «Personal (empleados)» (detrás del "
                     "mostrador) en la misma cámara.")
        if not one_day:
            days = [(d["date"], wait_total(wait_zones(d))) for d in m.get("by_day") or []]
            days = [(d, w) for d, w in days if w["n"]]
            if len(days) > 1:
                pdf.set_font("Helvetica", "B", 10)
                pdf.cell(0, 7, latin("Por día"), new_x="LMARGIN", new_y="NEXT")
                pdf.table(["Día", "Esperas", "Promedio", "Máximo", "Más de 5 min", "Fila máx."],
                          [[fmt_day(d), str(w["n"]), clock(w["avg_s"]), clock(w["max_s"]), str(w["over5"]),
                            str(w["queue_max"])] for d, w in days],
                          [36, 24, 28, 28, 32, 34])
    if "empleados" in plans:
        pdf.h2("Empleados")
        posts = [(name, post_stats(p)) for name, p in (t.get("posts") or {}).items()]
        posts = sorted([(n, p) for n, p in posts if p["obs_s"]], key=lambda kv: kv[0])
        watched = t.get("staff_obs_s") or 0.0  # segundos mirados, sumando cámaras (sin esto, máquina sin actualizar)
        seen = (f"Empleados a la vista (por cámara): promedio {dec(t.get('staff_s', 0.0) / watched)}, máximo "
                f"{t.get('staff_max', 0)} a la vez") if watched else ""
        if posts:
            pdf.table(["Puesto", "Cubierto", "Sin nadie", "Gente en el puesto (promedio)", "Máximo"],
                      [[name, f"{dec(p['covered_pct'], 0)} %", clock(p["uncovered_s"]), dec(p["avg_people"]),
                        str(p["max"])] for name, p in posts],
                      [56, 24, 30, 52, 20])
        elif m.get("by_camera") and all(isinstance(c.get("measures"), list) for c in m["by_camera"]) and not any(
                "empleados" in c["measures"] for c in m["by_camera"]):
            pdf.note("Ninguna cámara mide «Seguimiento de empleados» en este período.", size=9)
        else:
            pdf.note("Sin datos de los puestos en este período. Hace falta la zona «Personal (empleados)» en la cámara "
                     "(detrás del mostrador, la caja por dentro) y que la máquina que la analiza esté actualizada (se "
                     "actualiza al reiniciarla). Sin esa zona solo se reconoce como empleado a quien aparece con el "
                     "local cerrado.", size=9)
        if seen:
            pdf.set_font("Helvetica", "", 9.5)
            pdf.multi_cell(0, 5, latin(seen), new_x="LMARGIN", new_y="NEXT")
        dwell = sorted(((n, v) for n, v in (t.get("staff_dwell") or {}).items() if v), key=lambda kv: -kv[1])[:6]
        if dwell:
            pdf.multi_cell(0, 5, latin("Dónde pasan el tiempo: " + " · ".join(f"{n} {clock(v)}" for n, v in dwell)),
                           new_x="LMARGIN", new_y="NEXT")
    if "hurto" in plans and alerts is not None:
        pdf.h2("Alertas de posible hurto")
        total = sum(alerts.values())
        pdf.tiles([(str(total), "Alertas en el período", "indicios para revisar, no pruebas"),
                   (str(alerts.get("hurto", 0)), LABELS["hurto"], "revisadas por una persona"),
                   (str(alerts.get("sospechoso", 0)), LABELS["sospechoso"], ""),
                   (str(alerts.get("sin_etiqueta", 0)), LABELS["sin_etiqueta"], "")])
    pdf.ln(3)
    if pdf.get_y() > 255:
        pdf.add_page()
    pdf.note("Cómo se mide: «Entraron» cuenta a quien cruza la puerta hacia adentro en la cámara de la entrada (una sola "
             "cámara, así nadie se cuenta dos veces). No se suma la gente de todas las cámaras: la misma persona aparece en "
             "varias y, si sale del cuadro y vuelve, cuenta de nuevo. «A dónde van» cuenta visitas: quien vuelve a una "
             "zona suma otra visita. Quien pisa la zona de "
             "personal o aparece con el local cerrado no cuenta como cliente. Cada cámara suma solo lo que se le eligió "
             "medir («-» en lo que no mide). No se reconocen caras.")
    if waits:
        pdf.note(f"Tiempo de espera: cuánto estuvo cada cliente ({WAIT_MIN_SECONDS:g} s o más) en la zona marcada para "
                 "medir la espera; si la zona incluye el lugar donde se atiende, incluye la atención. En una fila "
                 "apretada, quien queda tapado por otro no se ve.")
    if "empleados" in plans:
        pdf.note("Empleados: se reconocen solo por su zona («Personal», pisada 2 s o más) o por estar con el local "
                 "cerrado; no se sabe quién es quién ni se los sigue de una cámara a otra, y quien pisa el puesto "
                 "cuenta como «alguien en el puesto», sea o no empleado.")
    return bytes(pdf.output())
