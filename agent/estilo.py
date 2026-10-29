"""Aspecto de la app de escritorio: colores de la marca, letra y botones, iguales en Windows y en Linux.

Se usa el tema «clam» de Tk (el nativo de Windows no deja cambiar colores de botones) con la paleta de abajo. Los
botones principales («Agregar», «Configurar», «Aprobar») van con `style="Primario.TButton"` (azul de la marca); los
textos de ayuda con `style="Suave.TLabel"`; los títulos con `Titulo.TLabel` / `Subtitulo.TLabel`.
"""

from __future__ import annotations

import logging
import sys

log = logging.getLogger("vigia.estilo")

FONDO = "#f3f5f9"  # fondo de las ventanas
BLANCO = "#ffffff"  # listas, campos y la cabecera
TEXTO = "#1d2433"
SUAVE = "#667085"  # ayudas y datos secundarios
BORDE = "#d8dee9"
MARCA = "#011b5d"  # azul del logo
MARCA_2 = "#1d3f8f"  # azul al pasar el mouse
ELEGIDO = "#dce7fb"  # fila elegida
CABECERA = "#eef2f8"  # encabezados de las listas
OK = "#2f8f46"
AVISO = "#b45309"
ERROR = "#c0392b"


def familia(root) -> str:
    """Segoe UI en Windows; en Linux, la primera que esté."""
    if sys.platform == "win32":
        return "Segoe UI"
    try:
        from tkinter import font

        have = set(font.families(root))
        return next((f for f in ("Segoe UI", "Noto Sans", "DejaVu Sans", "Liberation Sans") if f in have), "TkDefaultFont")
    except Exception:  # noqa: BLE001
        return "TkDefaultFont"


def aplicar(root) -> None:
    """Tema, letras y colores para toda la app (también los diálogos que se abran después)."""
    try:
        from tkinter import font, ttk
    except Exception:  # noqa: BLE001
        return
    fam = familia(root)
    try:
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkCaptionFont", "TkSmallCaptionFont",
                     "TkIconFont", "TkTooltipFont"):
            font.nametofont(name).configure(family=fam, size=10)
        font.nametofont("TkHeadingFont").configure(family=fam, size=9, weight="bold")
    except Exception as exc:  # noqa: BLE001 - cosmético
        log.debug("No se pudieron cambiar las letras: %s", exc)
    root.configure(bg=FONDO)
    root.option_add("*Toplevel.background", FONDO)
    root.option_add("*Menu.background", BLANCO)
    root.option_add("*Menu.foreground", TEXTO)
    root.option_add("*Menu.activeBackground", ELEGIDO)
    root.option_add("*Menu.activeForeground", TEXTO)
    root.option_add("*Menu.relief", "flat")
    root.option_add("*Menu.borderWidth", 1)
    root.option_add("*TCombobox*Listbox.background", BLANCO)
    root.option_add("*TCombobox*Listbox.selectBackground", ELEGIDO)
    root.option_add("*TCombobox*Listbox.selectForeground", TEXTO)
    root.option_add("*Listbox.background", BLANCO)
    root.option_add("*Listbox.selectBackground", ELEGIDO)
    root.option_add("*Listbox.selectForeground", TEXTO)
    root.option_add("*Listbox.relief", "flat")
    root.option_add("*Listbox.highlightThickness", 1)
    root.option_add("*Listbox.highlightColor", BORDE)
    root.option_add("*Listbox.highlightBackground", BORDE)

    st = ttk.Style(root)
    try:
        st.theme_use("clam")
    except Exception:  # noqa: BLE001
        return
    st.configure(".", background=FONDO, foreground=TEXTO, bordercolor=BORDE, lightcolor=FONDO, darkcolor=BORDE,
                 troughcolor="#e4e9f2", focuscolor=MARCA_2, selectbackground=ELEGIDO, selectforeground=TEXTO,
                 fieldbackground=BLANCO, insertcolor=TEXTO, font=(fam, 10))
    # botones: blancos con borde; el principal, azul de la marca
    st.configure("TButton", padding=(12, 5), background=BLANCO, bordercolor=BORDE, lightcolor=BLANCO,
                 darkcolor=BLANCO, relief="flat", focusthickness=0, anchor="center")
    st.map("TButton", background=[("disabled", FONDO), ("pressed", "#e3e8f2"), ("active", "#eef2f8")],
           foreground=[("disabled", "#a0a8b8")], bordercolor=[("active", "#b9c3d6")])
    st.configure("Primario.TButton", background=MARCA, foreground=BLANCO, bordercolor=MARCA, lightcolor=MARCA,
                 darkcolor=MARCA, font=(fam, 10, "bold"))
    st.map("Primario.TButton", background=[("disabled", "#8b97b5"), ("pressed", "#000f38"), ("active", MARCA_2)],
           foreground=[("disabled", "#eef1f7")], bordercolor=[("active", MARCA_2)],
           lightcolor=[("active", MARCA_2)], darkcolor=[("active", MARCA_2)])
    st.configure("Chico.TButton", padding=(8, 2))
    st.configure("Cabecera.TButton", background=BLANCO)
    st.configure("TMenubutton", padding=(12, 5), background=BLANCO, bordercolor=BORDE, lightcolor=BLANCO,
                 darkcolor=BLANCO, arrowcolor=SUAVE, relief="flat")
    st.map("TMenubutton", background=[("active", "#eef2f8")])
    # listas
    st.configure("Treeview", background=BLANCO, fieldbackground=BLANCO, foreground=TEXTO, rowheight=27,
                 bordercolor=BORDE, lightcolor=BORDE, darkcolor=BORDE, borderwidth=1)
    st.map("Treeview", background=[("selected", ELEGIDO)], foreground=[("selected", TEXTO)])
    st.configure("Treeview.Heading", background=CABECERA, foreground=SUAVE, relief="flat", padding=(6, 5),
                 bordercolor=BORDE, lightcolor=CABECERA, darkcolor=BORDE, font=(fam, 9, "bold"))
    st.map("Treeview.Heading", background=[("active", "#e4e9f2")])
    # pestañas
    st.configure("TNotebook", background=FONDO, borderwidth=0, tabmargins=(10, 6, 10, 0), lightcolor=FONDO,
                 darkcolor=FONDO, bordercolor=BORDE)
    st.configure("TNotebook.Tab", padding=(18, 7), background=FONDO, foreground=SUAVE, borderwidth=0,
                 lightcolor=FONDO, darkcolor=FONDO, bordercolor=FONDO, font=(fam, 10, "bold"))
    st.map("TNotebook.Tab", background=[("selected", BLANCO), ("active", "#e9edf5")],
           foreground=[("selected", MARCA)], lightcolor=[("selected", BLANCO)], bordercolor=[("selected", BORDE)])
    # campos
    for name in ("TEntry", "TCombobox", "TSpinbox"):
        st.configure(name, padding=4, fieldbackground=BLANCO, bordercolor=BORDE, lightcolor=BLANCO, darkcolor=BLANCO,
                     arrowcolor=SUAVE)
        st.map(name, bordercolor=[("focus", MARCA_2)], lightcolor=[("focus", MARCA_2)],
               fieldbackground=[("readonly", BLANCO), ("disabled", FONDO)])
    st.configure("TCheckbutton", background=FONDO, indicatorbackground=BLANCO, indicatorforeground=MARCA)
    st.map("TCheckbutton", background=[("active", FONDO)], indicatorbackground=[("pressed", ELEGIDO)])
    st.configure("TRadiobutton", background=FONDO, indicatorbackground=BLANCO, indicatorforeground=MARCA)
    st.configure("TLabelframe", background=FONDO, bordercolor=BORDE, lightcolor=FONDO, darkcolor=FONDO)
    st.configure("TLabelframe.Label", background=FONDO, foreground=SUAVE, font=(fam, 9, "bold"))
    st.configure("Horizontal.TProgressbar", background=MARCA, troughcolor="#e4e9f2", bordercolor=BORDE,
                 lightcolor=MARCA, darkcolor=MARCA)
    st.configure("TScrollbar", background="#e4e9f2", troughcolor=FONDO, bordercolor=FONDO, arrowcolor=SUAVE,
                 lightcolor="#e4e9f2", darkcolor="#e4e9f2")
    st.configure("TSeparator", background=BORDE)
    st.configure("Horizontal.TScale", background=MARCA, troughcolor="#e4e9f2", bordercolor=BORDE,
                 lightcolor=MARCA, darkcolor=MARCA)
    # textos
    st.configure("Titulo.TLabel", font=(fam, 15, "bold"), foreground=MARCA)
    st.configure("Subtitulo.TLabel", font=(fam, 11, "bold"), foreground=TEXTO)
    st.configure("Suave.TLabel", foreground=SUAVE)
    st.configure("Aviso.TLabel", foreground=AVISO)
    # cabecera blanca de la ventana principal
    st.configure("Cabecera.TFrame", background=BLANCO)
    st.configure("Cabecera.TLabel", background=BLANCO)
    st.configure("CabeceraSuave.TLabel", background=BLANCO, foreground=SUAVE)
    st.configure("CabeceraMarca.TLabel", background=BLANCO, foreground=MARCA, font=(fam, 15, "bold"))
    st.configure("Cabecera.TMenubutton", background=BLANCO, bordercolor=BLANCO, lightcolor=BLANCO, darkcolor=BLANCO)
    # barra de abajo (el registro)
    st.configure("Barra.TFrame", background="#e9edf5")
    st.configure("Barra.TLabel", background="#e9edf5", foreground=SUAVE, font=(fam, 9))
    st.configure("Barra.TButton", background="#e9edf5", bordercolor="#e9edf5", lightcolor="#e9edf5",
                 darkcolor="#e9edf5", padding=(8, 1), font=(fam, 9))
