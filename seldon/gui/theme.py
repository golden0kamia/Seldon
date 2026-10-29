"""Thème « cassette futurism » en néomorphisme, accent vert.

Surfaces en plastique mastic (beige chaud) — ou, en variante sombre, en
plastique brun-noir de chaîne hi-fi — en relief doux, l'éditeur en écran
cathodique à phosphore vert, voyants LED et bandes de cassette. Qt ne sait pas
dessiner de ``box-shadow`` en QSS : les ombres néomorphiques (claire en haut à
gauche, sombre en bas à droite) sont peintes par ``NeuCard`` ; les contrôles
imitent le relief par dégradés et bordures claires/sombres.

Variantes « néon », en chasse fixe et coins vifs, peintes par la même mécanique
(``CARD``, ``BACKDROP``, ``SCANLINES``, ``BRAND_GLOW`` de chaque palette) :

- ``retro`` (rétro-futurisme) : synthwave des années 80, nuit bleu-violet, néons
  cyan et magenta, cartes cernées d'un halo, grille en perspective à l'horizon ;
- ``cyber`` (cyberpunk) : terminal noir et vert phosphore, magenta pour les
  repères, coins biseautés, lignes de balayage sur tout le fond, marque « glitchée » ;
- ``hud`` (interface de science-fiction) : traits fins cyan sur bleu nuit,
  équerres aux angles des cartes, quadrillage technique de 50 px.

Les couleurs sont des variables du module : ``apply(app, "dark")`` les remplace
par celles de la variante puis redessine tout (feuille de style, icônes SVG
régénérées, cartes et éditeurs via leur méthode ``theme_changed``).
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QBrush, QColor, QFont, QFontDatabase, QLinearGradient, QPainter, QPainterPath, QPalette, QPen, QPixmap,
    QRadialGradient,
)
from PySide6.QtWidgets import (
    QAbstractButton, QAbstractSpinBox, QApplication, QFrame, QGraphicsDropShadowEffect, QHBoxLayout, QLabel,
    QSizePolicy, QToolButton, QVBoxLayout, QWidget,
)

# --- palettes --------------------------------------------------------------
LIGHT = dict(
    BG="#e2dac9",            # plastique mastic
    BG_HI="#ece6d9",
    BG_LO="#d3cab6",
    SHADOW_DARK="#b3a991",
    SHADOW_LIGHT="#fbf8f0",
    TEXT="#35302a",
    MUTED="#675d4b",         # contraste ≥ 4,5:1 sur BG et BG_HI (texte secondaire)
    ACCENT="#2f9d5c",        # vert d'accent
    ACCENT_HI="#4fd283",
    ACCENT_LO="#1d6e3e",
    ACCENT_TEXT="#176037",   # vert lisible sur la surface (≥ 4,5:1, même creusée)
    SELECT_BG="rgba(47, 157, 92, 0.18)",
    WARN="#a23f28",
    CRT_BG="#141a15",
    CRT_GUTTER="#0f1410",
    SHADOW_STRENGTH=1.0,
)
DARK = dict(
    BG="#2b2824",            # plastique brun-noir
    BG_HI="#34312c",
    BG_LO="#24211e",
    SHADOW_DARK="#141210",
    SHADOW_LIGHT="#3e3a34",
    TEXT="#e8dfcf",
    MUTED="#a89d8a",
    ACCENT="#3cb46c",
    ACCENT_HI="#5ee08f",
    ACCENT_LO="#22804a",
    ACCENT_TEXT="#62dc92",
    SELECT_BG="rgba(94, 224, 143, 0.16)",
    WARN="#e0846e",
    CRT_BG="#0e120f",
    CRT_GUTTER="#0a0d0b",
    SHADOW_STRENGTH=1.6,
)
# communs aux variantes cassette (contrastes WCAG mesurés sur CRT_BG, l'écran le plus clair)
CASSETTE = dict(
    ORANGE="#d8612c",        # bandes de cassette (décor)
    AMBER="#e6a23c",
    INK="#1f1b16",           # encre sur les surfaces colorées (ambre, touche verte) : ≥ 4,9:1
    CRT_FG="#9be8ad",
    CRT_DIM="#5d8f65",       # 4,7:1
    CRT_ERROR="#ec7a64",     # erreurs dans le journal : 6,3:1
    CRT_SELECT="#2b5a3a",
    SYNTAX=dict(keyword="#5ef08a", string="#e8b04a", comment="#6e8e73", number="#7fd9c6",
                key="#b5f2c3", builtin="#6fcbe6", decorator="#e8803a", heading="#5ef08a"),
)
# rétro-futurisme : nuit #1a1a2e, néons cyan (structure) et magenta (action) ; texte ≥ 5,8:1, écran ≥ 5:1
RETRO = dict(
    BG="#1a1a2e",
    BG_HI="#22223f",
    BG_LO="#141426",
    SHADOW_DARK="#0b0b16",
    SHADOW_LIGHT="#34345e",
    TEXT="#e8e6f5",
    MUTED="#9d9cc4",
    ACCENT="#00e5ff",        # néon cyan : bordures, halos
    ACCENT_HI="#7df9ff",
    ACCENT_LO="#6a00d9",     # violet : sélections en texte blanc (8:1)
    ACCENT_TEXT="#22e5f5",
    SELECT_BG="rgba(0, 229, 255, 0.14)",
    WARN="#ff5c8a",
    CRT_BG="#0d0d1a",
    CRT_GUTTER="#09091a",
    SHADOW_STRENGTH=1.6,
    ORANGE="#ff006e",        # bandes : magenta, or, cyan (coucher de soleil synthwave)
    AMBER="#ffd700",
    INK="#0d0d1a",           # sur magenta 5:1, sur or 13,7:1, sur cyan 15:1
    NEON="#ff006e",          # magenta : marque, bouton principal, cases cochées
    MARK="#ff006e",          # repères : onglet actif, sélection, cadre des écrans
    CARD="glow",             # cartes : halo néon
    BACKDROP="horizon",
    SCANLINES=True,
    BRAND_GLOW=(0, 18),      # décalage horizontal, flou du halo de la marque
    CRT_FG="#7df9ff",
    CRT_DIM="#5b86b8",       # 5,1:1
    CRT_ERROR="#ff5c8a",
    CRT_SELECT="#3a1d6e",
    SYNTAX=dict(keyword="#ff4fd8", string="#ffd700", comment="#7480b0", number="#4da6ff",
                key="#7df9ff", builtin="#b388ff", decorator="#ff9e40", heading="#ff4fd8"),
)
# cyberpunk : #0d0d0d, vert terminal pour la structure et l'action, magenta pour les repères
CYBER = dict(
    BG="#0d0d0d",
    BG_HI="#141814",
    BG_LO="#080808",
    SHADOW_DARK="#000000",
    SHADOW_LIGHT="#1f2b22",
    TEXT="#cfe8cf",          # 13,8:1 au pire
    MUTED="#7fa88a",         # 6,7:1
    ACCENT="#00ff00",
    ACCENT_HI="#66ff66",
    ACCENT_LO="#8a008a",     # sélections en texte blanc (8,6:1)
    ACCENT_TEXT="#00ff00",
    SELECT_BG="rgba(0, 255, 0, 0.10)",
    WARN="#ff4545",
    CRT_BG="#050805",
    CRT_GUTTER="#030503",
    SHADOW_STRENGTH=1.6,
    ORANGE="#ff00ff",        # bandes : magenta, or, cyan
    AMBER="#ffd700",
    INK="#0d0d0d",           # sur vert 14:1
    NEON="#00ff00",
    MARK="#ff00ff",
    CARD="chamfer",          # coins biseautés
    BACKDROP="scanlines",
    SCANLINES=True,
    BRAND_GLOW=(-2, 2),      # aberration chromatique magenta façon glitch
    CRT_FG="#33ff66",
    CRT_DIM="#3f9a55",       # 5,7:1
    CRT_ERROR="#ff4545",
    CRT_SELECT="#4a0f4a",
    SYNTAX=dict(keyword="#ff00ff", string="#ffd700", comment="#5f8f6a", number="#00ffff",
                key="#33ff66", builtin="#ff66ff", decorator="#ffa500", heading="#ff00ff"),
)
# HUD / FUI de science-fiction : #0a0f1c, traits cyan fins, rouge d'alerte, vert d'état ; écran ≥ 4,6:1
HUD = dict(
    BG="#0a0f1c",
    BG_HI="#0f1728",
    BG_LO="#070b15",
    SHADOW_DARK="#03060d",
    SHADOW_LIGHT="#1b2c45",
    TEXT="#d8f6ff",
    MUTED="#6fa8bd",         # 6,8:1
    ACCENT="#00ffff",
    ACCENT_HI="#9ff7ff",
    ACCENT_LO="#0058b0",     # sélections en texte blanc (7:1)
    ACCENT_TEXT="#00ffff",
    SELECT_BG="rgba(0, 255, 255, 0.10)",
    WARN="#ff4d4d",
    CRT_BG="#060a14",
    CRT_GUTTER="#04070f",
    SHADOW_STRENGTH=1.6,
    ORANGE="#0080ff",        # bandes : bleu, ambre, cyan
    AMBER="#ffc940",
    INK="#0a0f1c",           # sur cyan 15:1
    NEON="#00ffff",
    MARK="#00ffff",
    CARD="brackets",         # équerres aux angles
    BACKDROP="grid",
    SCANLINES=False,
    BRAND_GLOW=(0, 12),
    CRT_FG="#9ff7ff",
    CRT_DIM="#4a8aa3",       # 5,1:1
    CRT_ERROR="#ff4d4d",
    CRT_SELECT="#0d3a5c",
    SYNTAX=dict(keyword="#00ffff", string="#ffc940", comment="#5c7f99", number="#4da3ff",
                key="#7dffb0", builtin="#7dffb0", decorator="#ff8a5c", heading="#00ffff"),
)
NEON_VARIANTS = {"retro": RETRO, "cyber": CYBER, "hud": HUD}
PALETTES = {"light": {**CASSETTE, **LIGHT}, "dark": {**CASSETTE, **DARK}, **NEON_VARIANTS}
VARIANT = "light"

# valeurs courantes (remplacées par set_variant)
BG = BG_HI = BG_LO = SHADOW_DARK = SHADOW_LIGHT = TEXT = MUTED = ""
ACCENT = ACCENT_HI = ACCENT_LO = ACCENT_TEXT = SELECT_BG = WARN = CRT_BG = CRT_GUTTER = ""
ORANGE = AMBER = INK = CRT_FG = CRT_DIM = CRT_ERROR = CRT_SELECT = ""
NEON = MARK = CARD = BACKDROP = ""   # propres aux variantes néon
SCANLINES = False
BRAND_GLOW = (0, 0)
SYNTAX: dict[str, str] = {}
SHADOW_STRENGTH = 1.0


def set_variant(name: str) -> None:
    global VARIANT
    VARIANT = name if name in PALETTES else "light"
    globals().update(NEON="", MARK="", CARD="", BACKDROP="", SCANLINES=False, BRAND_GLOW=(0, 0))
    globals().update(PALETTES[VARIANT])


def is_neon() -> bool:
    """Rétro-futurisme, cyberpunk ou HUD : chasse fixe, coins vifs, néons."""
    return VARIANT in NEON_VARIANTS


def is_dark() -> bool:
    """Variante sur fond sombre (les variantes néon le sont toutes)."""
    return VARIANT != "light"


set_variant("light")


def system_variant() -> str:
    """Variante correspondant au thème du système (clair par défaut)."""
    from PySide6.QtGui import QGuiApplication

    try:
        scheme = QGuiApplication.styleHints().colorScheme()
    except AttributeError:   # Qt < 6.5
        return "light"
    return "dark" if scheme == Qt.ColorScheme.Dark else "light"


_ICON_DIR = Path(tempfile.mkdtemp(prefix="seldon-gui-icons-"))
atexit.register(shutil.rmtree, _ICON_DIR, True)   # pas de dossier temporaire laissé à chaque lancement

_ICONS = {  # nom -> (tracé SVG, couleur)
    "chevron-down": ('<path d="M4 6l4 4 4-4" {stroke}/>', "MUTED"),
    "chevron-up": ('<path d="M4 10l4-4 4 4" {stroke}/>', "MUTED"),
    "branch-closed": ('<path d="M6 4l4 4-4 4" {stroke}/>', "MUTED"),
    "branch-open": ('<path d="M4 6l4 4 4-4" {stroke}/>', "ACCENT_TEXT"),
    "close": ('<path d="M5 5l6 6M11 5l-6 6" {stroke}/>', "MUTED"),
    "close-hover": ('<circle cx="8" cy="8" r="7" fill="{warn}"/>'
                    '<path d="M5.5 5.5l5 5M10.5 5.5l-5 5" fill="none" stroke="#fff" stroke-width="1.8" stroke-linecap="round"/>',
                    "MUTED"),
}


def _write_icons() -> Path:
    """Icônes SVG aux couleurs de la variante courante (un dossier par variante)."""
    folder = _ICON_DIR / VARIANT
    folder.mkdir(exist_ok=True)
    for name, (body, color) in _ICONS.items():
        stroke = f'fill="none" stroke="{globals()[color]}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"'
        svg = body.format(stroke=stroke, warn=WARN)
        (folder / f"{name}.svg").write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16">{svg}</svg>\n',
            encoding="utf-8")
    return folder

FIELD_PAD = 10   # marge intérieure gauche/droite des champs, en px

# Espacements communs (px) : toute l'interface s'aligne sur ces valeurs.
SPACE = 8        # entre deux éléments voisins (lignes, boutons, champs)
PAD = 12         # marge intérieure des cartes
SECTION = 16     # entre deux blocs d'une même page
SHADOW = 9       # marge réservée autour des cartes pour leurs ombres
GUTTER = SHADOW + PAD   # retrait des éléments posés hors carte, aligné sur le contenu des cartes
TAB_PAD = 14     # marge gauche du texte d'un onglet, reprise entre texte, croix et bord droit
TAB_CLOSE_GAP = 4   # marge droite du texte : Qt ajoute ~10 px avant la croix, soit TAB_PAD au total

# Share Tech pour l'interface, Share Tech Mono pour le code, les valeurs et la signalétique
UI_FAMILIES = ["Share Tech", "IBM Plex Sans Condensed", "DejaVu Sans Condensed", "Sans Serif"]
MONO_FAMILIES = ["Share Tech Mono", "IBM Plex Mono", "DejaVu Sans Mono", "Monospace"]


def _family(candidates: list[str]) -> str:
    available = set(QFontDatabase.families())
    return next((f for f in candidates if f in available), candidates[-1])


def ui_font(size: float = 11.5, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    f = QFont(_family(MONO_FAMILIES if is_neon() else UI_FAMILIES))   # néon : tout en chasse fixe
    f.setPointSizeF(size)
    f.setWeight(weight)
    return f


def mono_font(size: float = 11) -> QFont:
    f = QFont(_family(MONO_FAMILIES))
    f.setPointSizeF(size)
    f.setStyleHint(QFont.StyleHint.Monospace)
    return f


def label_font(size: float = 9.0) -> QFont:
    """Petites capitales espacées, façon sérigraphie d'appareil."""
    f = mono_font(size)
    f.setCapitalization(QFont.Capitalization.AllUppercase)
    f.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 112)
    return f


# --- feuille de style ---------------------------------------------------------
def _raised() -> str:
    a, b = BG_HI, BG_LO   # lus à l'appel : la variante peut avoir changé
    return (f"background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {a}, stop:1 {b});"
            f"border: 1px solid {BG}; border-top-color: {SHADOW_LIGHT}; border-left-color: {SHADOW_LIGHT};"
            f"border-bottom-color: {SHADOW_DARK}; border-right-color: {SHADOW_DARK};")


def _inset() -> str:
    return (f"background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {BG_LO}, stop:0.35 {BG}, stop:1 {BG_HI});"
            f"border: 1px solid {BG}; border-top-color: {SHADOW_DARK}; border-left-color: {SHADOW_DARK};"
            f"border-bottom-color: {SHADOW_LIGHT}; border-right-color: {SHADOW_LIGHT};")


def stylesheet() -> str:
    mono = _family(MONO_FAMILIES)
    icons = _write_icons()

    def icon(name: str) -> str:
        return f'url("{(icons / name).as_posix()}.svg")'

    return f"""
* {{ outline: 0; }}
QMainWindow, QWidget#central, QDialog, QMessageBox {{ background: {BG}; color: {TEXT}; }}
QWidget {{ color: {TEXT}; }}
QToolTip {{ background: {CRT_BG}; color: {CRT_FG}; border: 1px solid {ACCENT_LO}; padding: 6px 8px;
            font-family: "{mono}"; border-radius: 6px; }}

/* boutons : touches d'appareil en relief ; liseré vert = focus clavier (un clic ne le donne pas) */
QPushButton {{ {_raised()} border-radius: 12px; padding: 7px 18px; min-height: 18px; }}
QPushButton:hover {{ color: {ACCENT_TEXT}; }}
QPushButton:focus {{ border: 1px solid {ACCENT_TEXT}; }}
QPushButton:pressed {{ {_inset()} padding-top: 8px; padding-bottom: 6px; }}
QPushButton:disabled {{ color: {MUTED}; background: {BG}; border-color: {BG}; }}
QPushButton#primary {{ color: {INK};
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {ACCENT_HI}, stop:1 {ACCENT});
    border: 1px solid {ACCENT}; border-top-color: #8de6ac; border-left-color: #8de6ac;
    border-bottom-color: {ACCENT_LO}; border-right-color: {ACCENT_LO}; }}
QPushButton#primary:focus {{ border: 1px solid {INK}; }}
QPushButton#primary:pressed {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {ACCENT}, stop:1 {ACCENT_HI}); }}
QPushButton#primary:disabled {{ color: {MUTED}; background: {BG}; border-color: {BG}; }}

QToolButton {{ {_raised()} border-radius: 10px; padding: 3px; min-width: 20px; min-height: 20px; }}
QToolButton:hover {{ color: {ACCENT_TEXT}; }}
QToolButton:focus {{ border: 1px solid {ACCENT_TEXT}; }}
QToolButton:pressed, QToolButton:checked {{ {_inset()} color: {ACCENT_TEXT}; }}
QToolButton#mode {{ font-family: "{mono}"; font-size: 10pt; padding: 6px 16px; border-radius: 13px; color: {MUTED}; }}
QToolButton#mode:hover {{ color: {TEXT}; }}
QToolButton#mode:checked {{ {_inset()} color: {ACCENT_TEXT}; }}
QToolButton#mode:focus {{ border: 1px solid {ACCENT_TEXT}; }}
QToolButton[autoRaise="true"] {{ background: transparent; border: none; color: {MUTED}; }}
QToolButton[autoRaise="true"]:hover {{ color: {WARN}; }}
QToolButton[autoRaise="true"]:focus {{ background: {SELECT_BG}; color: {ACCENT_TEXT}; border-radius: 6px; }}

/* champs : creusés dans la coque */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ {_inset()} border-radius: 10px; padding: 5px {FIELD_PAD}px;
    min-height: 20px; selection-background-color: {ACCENT_LO}; selection-color: white; font-family: "{mono}"; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border: 1px solid {ACCENT}; border-top-color: {ACCENT_LO}; border-left-color: {ACCENT_LO}; }}
QLineEdit[invalid="true"] {{ border: 1px solid {WARN}; }}
QComboBox::drop-down {{ border: none; width: 26px; }}
QComboBox::down-arrow {{ image: {icon("chevron-down")}; width: 14px; height: 14px; margin-right: 8px; }}
QComboBox QAbstractItemView {{ background: {BG_HI}; border: 1px solid {SHADOW_DARK}; border-radius: 8px; padding: 4px;
    selection-background-color: {ACCENT_LO}; selection-color: white; outline: 0; }}
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    border: none; width: 20px; background: transparent; margin-right: 4px; }}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: {icon("chevron-up")}; width: 11px; height: 11px; }}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: {icon("chevron-down")}; width: 11px; height: 11px; }}

/* cases à cocher : voyants LED */
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 9px; {_inset()} }}
QCheckBox::indicator:checked {{ border: 1px solid {ACCENT_LO};
    background: qradialgradient(cx:0.4, cy:0.35, radius:0.7, fx:0.35, fy:0.3,
                stop:0 #d9ffe6, stop:0.35 {ACCENT_HI}, stop:1 {ACCENT_LO}); }}
QCheckBox::indicator:focus {{ border: 1px solid {ACCENT_TEXT}; }}
QCheckBox::indicator:checked:focus {{ border: 1px solid {TEXT}; }}

/* onglets : languettes en relief, l'actif enfoncé et éclairé */
QTabWidget::pane {{ border: none; }}
QTabWidget::tab-bar {{ left: {SHADOW - 4}px; }}
QTabBar {{ qproperty-drawBase: 0; }}
QTabBar::tab {{ {_raised()} border-radius: 11px; padding: 6px {TAB_CLOSE_GAP}px 6px {TAB_PAD}px; margin: {SPACE}px 4px {SPACE}px 4px;
    color: {MUTED}; }}
QTabBar::tab:hover {{ color: {TEXT}; }}
QTabBar::tab:selected {{ {_inset()} color: {ACCENT_TEXT}; }}
QToolButton#tabclose {{ background: transparent; border: none; padding: 0; min-width: 0; min-height: 0;
    image: {icon("close")}; }}
QToolButton#tabclose:hover {{ image: {icon("close-hover")}; }}

/* listes et arborescence posées dans des cartes */
QListWidget, QTreeView {{ background: transparent; border: none; font-size: 11.5pt;
    show-decoration-selected: 0; }}
QListWidget::item {{ padding: {SPACE}px {PAD}px; margin: 2px 2px; border-radius: 10px;
    border: 1px solid transparent; border-left: 3px solid transparent; }}
QListWidget::item:hover {{ color: {ACCENT_TEXT}; }}
QListWidget::item:selected {{ {_inset()} color: {ACCENT_TEXT}; border-left: 3px solid {ACCENT}; }}
QTreeView::item {{ padding: 3px 2px; border-radius: 6px; }}
QTreeView::item:hover {{ background: {BG_HI}; }}
QTreeView::item:selected {{ background: {SELECT_BG}; color: {ACCENT_TEXT}; }}
QTreeView::branch {{ background: transparent; }}
QTreeView::branch:has-children:closed {{ image: {icon("branch-closed")}; }}
QTreeView::branch:has-children:open {{ image: {icon("branch-open")}; }}

QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; border: none; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 4px 2px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px 4px; }}
QScrollBar::handle {{ background: {SHADOW_DARK}; border-radius: 3px; min-height: 30px; min-width: 30px; }}
QScrollBar::handle:hover {{ background: {ACCENT}; }}
QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {{ background: none; border: none; height: 0; width: 0; }}

QSplitter::handle {{ background: transparent; width: 10px; }}

QMenuBar {{ background: {BG}; padding: 4px {SPACE + GUTTER - PAD}px; }}
QMenuBar::item {{ padding: 5px {PAD}px; border-radius: 8px; background: transparent; }}
QMenuBar::item:selected {{ {_inset()} color: {ACCENT_TEXT}; }}
QMenu {{ background: {BG_HI}; border: 1px solid {SHADOW_DARK}; border-radius: 10px; padding: 6px; }}
QMenu::item {{ padding: 6px 22px 6px 14px; border-radius: 7px; }}
QMenu::item:selected {{ background: {ACCENT_LO}; color: white; }}
QMenu::separator {{ height: 1px; background: {SHADOW_DARK}; margin: 5px 8px; }}
QMenu::indicator {{ width: 12px; height: 12px; border-radius: 6px; margin-left: 4px; {_inset()} }}
QMenu::indicator:checked {{ background: {ACCENT_HI}; border: 1px solid {ACCENT_LO}; }}

/* barre d'état : afficheur à phosphore */
QStatusBar {{ background: {CRT_BG}; color: {CRT_FG}; font-family: "{mono}"; font-size: 9pt;
    min-height: 26px; padding: 0 {SPACE + SHADOW}px; border-top: 2px solid {SHADOW_DARK}; }}
QStatusBar QToolButton {{ background: transparent; border: none; color: {CRT_FG}; }}
QStatusBar QLabel {{ color: {CRT_FG}; padding: 0 {PAD}px; }}
QStatusBar::item {{ border: none; }}

/* éditeur : écran cathodique */
QPlainTextEdit#code {{ background: {CRT_BG}; color: {CRT_FG}; border: none; font-family: "{mono}";
    selection-background-color: {CRT_SELECT}; selection-color: #e6ffec; }}
QPlainTextEdit#code QScrollBar::handle {{ background: {CRT_DIM}; }}
QWidget#findbar {{ background: transparent; }}
QLabel#banner {{ background: {AMBER}; color: {INK}; padding: {SPACE}px {PAD}px; border-radius: 9px; }}
QFrame#confirmbar {{ background: {AMBER}; border-radius: 12px; }}
QFrame#confirmbar QLabel {{ color: {INK}; }}
QLabel#hint {{ color: {MUTED}; }}
QLabel#findinfo {{ color: {WARN}; }}
QLabel#pageTitle {{ color: {TEXT}; }}
QLabel#path {{ color: {MUTED}; font-family: "{mono}"; font-size: 9pt; }}
QScrollArea#imageview > QWidget > QWidget {{ background: {CRT_BG}; }}
QToolButton#variant {{ padding-right: 14px; }}
QToolButton#variant::menu-button {{ border: none; width: 14px; }}
QToolButton#variant::menu-arrow {{ image: {icon("chevron-down")}; width: 9px; height: 9px; }}
""" + (_neon_stylesheet() if is_neon() else "")


def _rgba(color: str, alpha: float) -> str:
    c = QColor(color)
    return f"rgba({c.red()}, {c.green()}, {c.blue()}, {alpha})"


def _horizon_svg() -> str:
    """Synthwave : dégradé de nuit, horizon et grille en perspective (SVG étiré sur toute la fenêtre)."""
    w, h, horizon = 1000, 1000, 640
    lines = []
    for i in range(-12, 13):   # fuyantes vers le point de fuite
        lines.append(f'<line x1="500" y1="{horizon}" x2="{500 + i * 160}" y2="{h}"/>')
    y, step = horizon + 6, 6.0
    while y < h:               # horizontales de plus en plus espacées vers le bas
        lines.append(f'<line x1="0" y1="{y:.1f}" x2="{w}" y2="{y:.1f}"/>')
        step *= 1.32
        y += step
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'preserveAspectRatio="none"><defs>'
            f'<linearGradient id="sky" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{BG}"/>'
            f'<stop offset="0.62" stop-color="#241640"/><stop offset="0.64" stop-color="{BG_LO}"/>'
            f'<stop offset="1" stop-color="#0f0f22"/></linearGradient>'
            f'<linearGradient id="fade" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{ACCENT}" stop-opacity="0"/>'
            f'<stop offset="1" stop-color="{ACCENT}" stop-opacity="0.22"/></linearGradient></defs>'
            f'<rect width="{w}" height="{h}" fill="url(#sky)"/>'
            f'<line x1="0" y1="{horizon}" x2="{w}" y2="{horizon}" stroke="{MARK}" stroke-opacity="0.55" stroke-width="2"/>'
            f'<g stroke="url(#fade)" stroke-width="1.4">{"".join(lines)}</g></svg>\n')


def _backdrop() -> str:
    """Fond de la fenêtre propre à la variante néon : horizon étiré, ou motif répété (balayage, quadrillage)."""
    folder = _ICON_DIR / VARIANT
    if BACKDROP == "horizon":
        path = folder / "backdrop.svg"
        path.write_text(_horizon_svg(), encoding="utf-8")
        return f'border-image: url("{path.as_posix()}") 0 0 0 0 stretch stretch;'
    if BACKDROP == "scanlines":   # repeating-linear-gradient 0deg, 2px / 4px
        w, h, body = 4, 4, f'<rect y="2" width="4" height="2" fill="{ACCENT}" fill-opacity="0.035"/>'
    else:                         # quadrillage 50 px
        w = h = 50
        body = f'<path d="M0 0.5H50M0.5 0V50" stroke="{ACCENT}" stroke-opacity="0.07" stroke-width="1"/>'
    path = folder / "backdrop.svg"
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">{body}</svg>\n', encoding="utf-8")
    return f'background-color: {BG}; background-image: url("{path.as_posix()}"); background-repeat: repeat-xy;'


def _neon_stylesheet() -> str:
    """Surcharges néon : coins vifs, liserés lumineux, ``NEON`` pour l'action, ``MARK`` pour les repères."""
    mono = _family(MONO_FAMILIES)
    dim = _rgba(ACCENT, 0.35)     # liseré au repos
    glass = _rgba(ACCENT, 0.06)
    neon = QColor(NEON)
    neon_hi, neon_lo = neon.lighter(125).name(), neon.darker(125).name()
    return f"""
QWidget#central {{ {_backdrop()} }}
QMenuBar {{ background: {BG_LO}; border-bottom: 1px solid {dim}; }}
QMenuBar::item:selected {{ background: {SELECT_BG}; border: none; color: {ACCENT_TEXT}; border-radius: 0; }}
QMenu {{ background: {BG_LO}; border: 1px solid {ACCENT}; border-radius: 0; }}
QMenu::item {{ border-radius: 0; }}
QMenu::item:selected {{ background: {ACCENT_LO}; color: white; }}
QMenu::indicator {{ border-radius: 0; background: {CRT_BG}; border: 1px solid {dim}; }}
QMenu::indicator:checked {{ background: {NEON}; border: 1px solid {NEON}; }}
QToolTip {{ background: {CRT_BG}; color: {CRT_FG}; border: 1px solid {ACCENT}; border-radius: 0; }}

QPushButton {{ background: {glass}; border: 1px solid {dim}; border-radius: 2px; color: {TEXT};
    font-family: "{mono}"; padding: 7px 18px; }}
QPushButton:hover {{ background: {SELECT_BG}; border-color: {ACCENT}; color: {ACCENT_TEXT}; }}
QPushButton:focus {{ border: 1px solid {ACCENT_HI}; }}
QPushButton:pressed {{ background: {ACCENT}; color: {INK}; padding: 7px 18px; }}
QPushButton:disabled {{ background: transparent; border-color: {SHADOW_LIGHT}; color: {MUTED}; }}
QPushButton#primary {{ background: {NEON}; color: {INK}; border: 1px solid {neon_hi}; }}
QPushButton#primary:hover {{ background: {neon_hi}; color: {INK}; }}
QPushButton#primary:focus {{ border: 1px solid {MARK if MARK != NEON else ACCENT_HI}; }}
QPushButton#primary:pressed {{ background: {neon_lo}; }}
QPushButton#primary:disabled {{ background: transparent; border-color: {SHADOW_LIGHT}; color: {MUTED}; }}

QToolButton {{ background: {glass}; border: 1px solid {dim}; border-radius: 2px; }}
QToolButton:hover {{ border-color: {ACCENT}; color: {ACCENT_TEXT}; }}
QToolButton:focus {{ border: 1px solid {ACCENT_HI}; }}
QToolButton:pressed, QToolButton:checked {{ background: {SELECT_BG}; border-color: {ACCENT}; color: {ACCENT_TEXT}; }}
QToolButton#mode {{ border-radius: 0; border: 1px solid transparent; border-bottom: 2px solid transparent;
    background: transparent; color: {MUTED}; }}
QToolButton#mode:hover {{ color: {ACCENT_TEXT}; }}
QToolButton#mode:checked {{ background: {SELECT_BG}; color: {ACCENT_TEXT}; border: 1px solid {dim};
    border-bottom: 2px solid {MARK}; }}
QToolButton#mode:focus {{ border: 1px solid {ACCENT_HI}; }}
QToolButton[autoRaise="true"] {{ background: transparent; border: none; }}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ background: {CRT_BG}; border: 1px solid {dim}; border-radius: 0;
    color: {CRT_FG}; selection-background-color: {ACCENT_LO}; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border: 1px solid {ACCENT}; }}
QLineEdit[invalid="true"] {{ border: 1px solid {WARN}; }}
QComboBox QAbstractItemView {{ background: {CRT_BG}; color: {CRT_FG}; border: 1px solid {ACCENT}; border-radius: 0; }}

QCheckBox::indicator {{ border-radius: 0; background: {CRT_BG}; border: 1px solid {dim}; }}
QCheckBox::indicator:checked {{ background: {NEON}; border: 1px solid {neon_hi}; }}
QCheckBox::indicator:focus {{ border: 1px solid {ACCENT_HI}; }}
QCheckBox::indicator:checked:focus {{ border: 1px solid {MARK if MARK != NEON else TEXT}; }}

QTabBar::tab {{ background: transparent; border: 1px solid transparent; border-bottom: 2px solid {SHADOW_LIGHT};
    border-radius: 0; color: {MUTED}; }}
QTabBar::tab:hover {{ color: {ACCENT_TEXT}; }}
QTabBar::tab:selected {{ background: {SELECT_BG}; border: 1px solid {dim}; border-bottom: 2px solid {MARK};
    color: {ACCENT_TEXT}; }}

QListWidget::item {{ border-radius: 0; }}
QListWidget::item:selected {{ background: {SELECT_BG}; border: 1px solid transparent; border-left: 3px solid {MARK};
    color: {ACCENT_TEXT}; }}
QTreeView::item {{ border-radius: 0; }}
QTreeView::item:hover {{ background: {glass}; }}
QScrollBar::handle {{ background: {SHADOW_LIGHT}; border-radius: 0; }}
QScrollBar::handle:hover {{ background: {ACCENT}; }}

QStatusBar {{ background: {CRT_BG}; border-top: 1px solid {MARK}; }}
QLabel#banner {{ border-radius: 0; }}
QFrame#confirmbar {{ border-radius: 0; }}
QLabel#brand {{ color: {NEON}; }}
"""


class _Polish(QObject):
    """Retouches appliquées à chaque widget quand il est préparé.

    - Le champ interne d'un QSpinBox ignore le ``padding`` de la feuille de style :
      on lui donne la même marge (padding + bordure) que les champs texte.
    - Boutons et cases ne prennent le focus qu'au clavier (Tab) : un clic ne vole
      pas le focus du champ en cours, et le liseré de focus signale la navigation clavier.
    - Le ``min-width`` des QToolButton impose une largeur minimale explicite qui passe
      outre leur politique de taille : les boutons de mode (``#mode``) gardent leur
      largeur naturelle au lieu d'être rognés en « … » quand la fenêtre rétrécit.
    """

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        kind = event.type()
        if kind == QEvent.Type.Polish:
            if isinstance(obj, QAbstractSpinBox):
                obj.lineEdit().setTextMargins(FIELD_PAD + 1, 0, 0, 0)
            elif isinstance(obj, QAbstractButton) and obj.focusPolicy() == Qt.FocusPolicy.StrongFocus:
                obj.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        if kind in (QEvent.Type.Polish, QEvent.Type.StyleChange) and isinstance(obj, QToolButton) \
                and obj.objectName() == "mode":
            obj.setMinimumWidth(obj.sizeHint().width())
        return False


_spin_padding = _Polish()


def apply(app: QApplication, variant: str = "auto") -> str:
    """Applique la variante ``"light"``, ``"dark"`` ou ``"auto"`` (celle du système) ; renvoie la variante retenue."""
    set_variant(system_variant() if variant == "auto" else variant)
    app.removeEventFilter(_spin_padding)
    app.installEventFilter(_spin_padding)
    app.setStyle("Fusion")
    app.setFont(ui_font())
    pal = QPalette()
    for role, color in [
        (QPalette.ColorRole.Window, BG), (QPalette.ColorRole.Base, BG_HI), (QPalette.ColorRole.AlternateBase, BG),
        (QPalette.ColorRole.Button, BG), (QPalette.ColorRole.Text, TEXT), (QPalette.ColorRole.WindowText, TEXT),
        (QPalette.ColorRole.ButtonText, TEXT), (QPalette.ColorRole.Highlight, ACCENT_LO),
        (QPalette.ColorRole.HighlightedText, "#ffffff"), (QPalette.ColorRole.PlaceholderText, MUTED),
        (QPalette.ColorRole.ToolTipBase, CRT_BG), (QPalette.ColorRole.ToolTipText, CRT_FG),
        (QPalette.ColorRole.Link, ACCENT_TEXT),
    ]:
        pal.setColor(role, QColor(color))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(MUTED))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(MUTED))
    app.setPalette(pal)
    app.setStyleSheet(stylesheet())
    for widget in app.allWidgets():   # surfaces peintes à la main : à redessiner
        if hasattr(widget, "theme_changed"):
            widget.theme_changed()
    return VARIANT


# --- surfaces peintes ---------------------------------------------------------
def _rounded(rect: QRectF, radius: float) -> QPainterPath:
    path = QPainterPath()
    path.addRoundedRect(rect, radius, radius)
    return path


class NeuCard(QFrame):
    """Carte néomorphique : ``raised`` (en relief), ``inset`` (creusée) ou ``screen`` (écran cathodique).

    Les ombres douces sont précalculées dans un pixmap à chaque redimensionnement. Une carte
    creusée ne déborde pas : elle n'a pas de marge d'ombre et s'aligne sur ses voisins.
    """

    def __init__(self, kind: str = "raised", radius: float = 18, padding: int = PAD,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.kind, self.radius = kind, radius
        self.margin = 0 if kind == "inset" else SHADOW
        self._cache: QPixmap | None = None
        self._cache_size = QSize()
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        m = self.margin + padding
        self.scanlines = Scanlines(self) if kind == "screen" else None
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(m, m, m, m)
        self.body.setSpacing(SPACE)

    def add(self, widget: QWidget, stretch: int = 0) -> QWidget:
        self.body.addWidget(widget, stretch)
        return widget

    def theme_changed(self) -> None:
        self._cache = None
        if self.scanlines:
            self.scanlines.setVisible(SCANLINES)
            self.scanlines.raise_()
        self.update()

    def resizeEvent(self, event) -> None:  # noqa: N802
        self._cache = None
        if self.scanlines:
            s = self.margin + 10
            self.scanlines.setGeometry(self.rect().adjusted(s, s, -s, -s))
            self.scanlines.raise_()
        super().resizeEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802
        if self._cache is None or self._cache_size != self.size():
            self._cache = self._render()
            self._cache_size = self.size()
        QPainter(self).drawPixmap(0, 0, self._cache)

    def _render(self) -> QPixmap:
        ratio = self.devicePixelRatioF()
        pix = QPixmap(self.size() * ratio)
        pix.setDevicePixelRatio(ratio)
        pix.fill(Qt.GlobalColor.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        s = self.margin
        rect = QRectF(self.rect()).adjusted(s, s, -s, -s)
        r = self.radius
        if is_neon():
            self._paint_neon(p, rect)
        elif self.kind == "inset":
            self._paint_inset(p, rect, r)
        else:
            self._paint_shadows(p, rect, r)
            grad = QLinearGradient(rect.topLeft(), rect.bottomRight())
            grad.setColorAt(0, QColor(BG_HI))
            grad.setColorAt(1, QColor(BG))
            p.fillPath(_rounded(rect, r), grad)
            if self.kind == "screen":   # cadre puis dalle cathodique
                bezel = rect.adjusted(6, 6, -6, -6)
                self._paint_inset(p, bezel, r - 5)
                glass = bezel.adjusted(4, 4, -4, -4)
                p.fillPath(_rounded(glass, r - 8), QColor(CRT_BG))
                glow = QRadialGradient(glass.center(), max(glass.width(), glass.height()) * 0.7)
                tint = QColor(CRT_FG)
                tint.setAlpha(18)
                glow.setColorAt(0, tint)
                glow.setColorAt(1, QColor(0, 0, 0, 0))
                p.fillPath(_rounded(glass, r - 8), glow)
        p.end()
        return pix

    def _paint_neon(self, p: QPainter, rect: QRectF) -> None:
        """Variantes néon : panneau vitré au liseré lumineux (``MARK`` pour les écrans), selon ``CARD`` :
        halo (``glow``), coins biseautés (``chamfer``) ou trait fin et équerres aux angles (``brackets``)."""
        if self.kind == "inset":   # creusée : simple dalle sombre au liseré discret
            p.fillRect(rect, QColor(BG_LO))
            p.setPen(QPen(QColor(SHADOW_LIGHT), 1))
            p.drawRect(rect.adjusted(0.5, 0.5, -0.5, -0.5))
            return
        neon = QColor(MARK if self.kind == "screen" else ACCENT)
        shape = self._neon_shape(rect)
        if CARD == "glow" or self.kind == "screen":   # halo : box-shadow 0 0 20px
            strength = 0.10 if CARD == "glow" else 0.06
            for i in range(SHADOW, 0, -1):
                c = QColor(neon)
                c.setAlphaF(strength * (1 - i / (SHADOW + 1)) ** 1.8)
                p.fillPath(self._neon_shape(rect.adjusted(-i, -i, i, i)), c)
        grad = QLinearGradient(rect.topLeft(), rect.bottomLeft())
        grad.setColorAt(0, QColor(BG_HI))
        grad.setColorAt(1, QColor(BG))
        p.fillPath(shape, grad)
        tint = QColor(neon)
        tint.setAlphaF(0.05)   # lueur intérieure / fond teinté
        p.fillPath(shape, tint)
        if self.kind == "screen":
            glass = rect.adjusted(10, 10, -10, -10)
            p.fillRect(glass, QColor(CRT_BG))
            line = QColor(neon)
            line.setAlphaF(0.35)
            p.setPen(QPen(line, 1))
            p.drawRect(glass.adjusted(0.5, 0.5, -0.5, -0.5))
        edge = QColor(neon)
        if CARD == "glow":
            pen = QPen(edge, 1.5)
        else:   # trait fin : rgba(…, 0.3–0.5), plein sur les écrans
            edge.setAlphaF(0.9 if self.kind == "screen" else 0.45 if CARD == "chamfer" else 0.3)
            pen = QPen(edge, 1)
        p.setPen(pen)
        p.drawPath(self._neon_shape(rect.adjusted(0.5, 0.5, -0.5, -0.5)))
        if CARD == "brackets":   # équerres pleines aux quatre angles
            p.setPen(QPen(neon, 1.5))
            k, b = 16, rect.adjusted(4.5, 4.5, -4.5, -4.5)
            for x, y, dx, dy in ((b.left(), b.top(), 1, 1), (b.right(), b.top(), -1, 1),
                                 (b.left(), b.bottom(), 1, -1), (b.right(), b.bottom(), -1, -1)):
                p.drawLine(QPointF(x, y), QPointF(x + dx * k, y))
                p.drawLine(QPointF(x, y), QPointF(x, y + dy * k))

    @staticmethod
    def _neon_shape(rect: QRectF) -> QPainterPath:
        if CARD == "glow":
            return _rounded(rect, 3)
        path = QPainterPath()
        if CARD == "chamfer":   # coins haut-droit et bas-gauche biseautés
            c = 14
            path.moveTo(rect.left(), rect.top())
            path.lineTo(rect.right() - c, rect.top())
            path.lineTo(rect.right(), rect.top() + c)
            path.lineTo(rect.right(), rect.bottom())
            path.lineTo(rect.left() + c, rect.bottom())
            path.lineTo(rect.left(), rect.bottom() - c)
            path.closeSubpath()
        else:
            path.addRect(rect)
        return path

    def _paint_shadows(self, p: QPainter, rect: QRectF, r: float) -> None:
        steps, offset = SHADOW, 4
        for color, d in ((QColor(SHADOW_DARK), offset), (QColor(SHADOW_LIGHT), -offset)):
            for i in range(steps, 0, -1):
                c = QColor(color)
                c.setAlphaF(0.11 * (1 - i / (steps + 1)) ** 1.6 * (1.25 * SHADOW_STRENGTH if d > 0 else 1.6))
                grown = rect.translated(d * 0.6, d * 0.6).adjusted(-i, -i, i, i)
                p.fillPath(_rounded(grown, r + i), c)

    def _paint_inset(self, p: QPainter, rect: QRectF, r: float) -> None:
        path = _rounded(rect, r)
        p.fillPath(path, QColor(BG))
        p.save()
        p.setClipPath(path)
        steps = 8
        for i in range(steps):
            c_dark, c_light = QColor(SHADOW_DARK), QColor(SHADOW_LIGHT)
            alpha = 0.16 * (1 - i / steps) ** 2 * SHADOW_STRENGTH
            c_dark.setAlphaF(alpha)
            c_light.setAlphaF(alpha * 1.4)
            # ombre portée par le bord supérieur gauche, reflet sur le bord inférieur droit
            p.setPen(Qt.PenStyle.NoPen)
            outer = _rounded(rect.adjusted(-20, -20, 20, 20), r)
            hole = _rounded(rect.adjusted(i + 3, i + 3, i + 3, i + 3), r)
            p.fillPath(outer.subtracted(hole), c_dark)
            hole = _rounded(rect.adjusted(-i - 3, -i - 3, -i - 3, -i - 3), r)
            p.fillPath(outer.subtracted(hole), c_light)
        p.restore()


class Scanlines(QWidget):
    """Lignes de balayage cathodiques posées par-dessus un écran (variantes néon), transparentes aux clics."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        tile = QPixmap(1, 4)
        tile.fill(Qt.GlobalColor.transparent)
        p = QPainter(tile)
        p.fillRect(0, 2, 1, 2, QColor(0, 0, 0, 70))
        p.end()
        self.brush = QBrush(tile)
        self.setVisible(SCANLINES)

    def paintEvent(self, event) -> None:  # noqa: N802
        QPainter(self).fillRect(event.rect(), self.brush)


class Stripes(QWidget):
    """Trois bandes obliques façon étiquette de cassette (orange, ambre, vert)."""

    def __init__(self, width: int = 64, height: int = 18, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(width, height)

    def sizeHint(self) -> QSize:  # noqa: N802
        return self.maximumSize()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        h, w = self.height(), self.width()
        band = w / 4
        for i, color in enumerate((ORANGE, AMBER, ACCENT)):
            x = i * band * 1.05
            path = QPainterPath(QPointF(x + h * 0.5, 0))
            path.lineTo(x + h * 0.5 + band * 0.8, 0)
            path.lineTo(x + band * 0.8, h)
            path.lineTo(x, h)
            path.closeSubpath()
            p.fillPath(path, QColor(color))


class Led(QWidget):
    """Voyant rond : vert allumé, ambre en attente, éteint sinon."""

    def __init__(self, size: int = 12, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(size + 4, size + 4)
        self.color = ACCENT_HI

    def set_color(self, color: str | None) -> None:
        self.color = color
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(2, 2, -2, -2)
        p.setPen(QColor(SHADOW_DARK))
        if self.color:
            glow = QRadialGradient(rect.center() - QPointF(rect.width() * 0.12, rect.height() * 0.15), rect.width() * 0.6)
            glow.setColorAt(0, QColor("#f2fff6"))
            glow.setColorAt(0.35, QColor(self.color))
            glow.setColorAt(1, QColor(self.color).darker(170))
            p.setBrush(glow)
        else:
            p.setBrush(QColor(BG_LO))
        p.drawEllipse(rect)


def caption(text: str, size: float = 8.5) -> QLabel:
    """Étiquette sérigraphiée (capitales espacées, gris chaud)."""
    label = QLabel(text)
    label.setFont(label_font(size))
    label.setObjectName("hint")
    return label


class Masthead(QWidget):
    """Bandeau supérieur : nom de l'appareil, bandes de cassette, voyant d'état et choix du thème."""

    toggle_variant = Signal()
    LONGEST_STATE = "99 modif. non enregistrée(s)"
    mode_changed = Signal(int)   # 0 = paramètres, 1 = éditeur, 2 = journal, 3 = structure

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        name = QLabel("SELDON", objectName="brand")
        f = mono_font(17)
        f.setWeight(QFont.Weight.Bold)
        f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 6)
        name.setFont(f)
        self.name = name
        self.model = caption("Assistant vocal · console de configuration · mk-1")
        self.stripes = Stripes()
        # décor facultatif : n'impose pas sa largeur, masqué si la fenêtre est étroite (resizeEvent)
        self.optional = (self.model, self.stripes)
        for w in self.optional:
            w.setMinimumWidth(1)
        self.led = Led(12)
        self.state = caption(self.LONGEST_STATE)
        self.state.setFixedWidth(self.state.sizeHint().width())   # place du texte le plus long : rien ne bouge
        self.state.setText("prêt")
        self.mode_buttons: list[QToolButton] = []
        modes = QHBoxLayout()
        modes.setSpacing(SPACE)
        for i, (text, tip) in enumerate((("⚙  PARAMÈTRES", "Réglages de config.yaml (Ctrl+1)"),
                                        ("✎  ÉDITEUR", "Fichiers et éditeur de texte (Ctrl+2)"),
                                        ("▤  JOURNAL", "Lancer Seldon et suivre ses logs (Ctrl+3)"),
                                        ("◈  STRUCTURE", "Schéma des sous-systèmes et de leurs échanges (Ctrl+4)"))):
            b = QToolButton(text=text, toolTip=tip, checkable=True, autoExclusive=True, objectName="mode")
            b.setAccessibleName(text.split(maxsplit=1)[1].capitalize())
            b.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)   # jamais rognés en « … »
            b.clicked.connect(lambda _=False, i=i: self.mode_changed.emit(i))
            modes.addWidget(b)
            self.mode_buttons.append(b)
        self.mode_buttons[0].setChecked(True)
        # un clic bascule clair / sombre, la flèche ouvre le choix du thème (menu fourni par la fenêtre)
        self.variant_btn = QToolButton(objectName="variant")
        self.variant_btn.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.variant_btn.setFont(ui_font(11))
        self.variant_btn.setFixedSize(46, 30)   # ☀ et ☾ n'ont pas la même largeur
        self.variant_btn.clicked.connect(self.toggle_variant.emit)
        self.theme_changed()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(GUTTER, SPACE, GUTTER, 0)
        lay.setSpacing(SECTION)
        lay.addWidget(name)
        lay.addWidget(self.stripes)
        lay.addWidget(self.model)
        lay.addStretch(1)
        lay.addLayout(modes)
        lay.addStretch(1)
        state = QHBoxLayout()
        state.setSpacing(SPACE)
        state.addWidget(self.led)
        state.addWidget(self.state)
        lay.addLayout(state)
        lay.addWidget(self.variant_btn)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

    def set_mode(self, index: int) -> None:
        self.mode_buttons[index].setChecked(True)

    def set_state(self, text: str, color: str | None) -> None:
        self.state.setText(text)
        self.state.setAccessibleName(f"État : {text}")
        self.led.set_color(color)

    def resizeEvent(self, event) -> None:  # noqa: N802
        """Fenêtre étroite : la mention « assistant vocal… » puis les bandes s'effacent, les boutons restent entiers."""
        super().resizeEvent(event)
        lay = self.layout()
        need = lay.sizeHint().width() + sum(w.sizeHint().width() + lay.spacing()
                                            for w in self.optional if not w.isVisibleTo(self))
        for w in self.optional:
            w.setVisible(event.size().width() >= need)
            if not w.isVisibleTo(self):
                need -= w.sizeHint().width() + lay.spacing()

    def theme_changed(self) -> None:
        dark = is_dark()
        self.variant_btn.setText("☀" if dark else "☾")
        name = "Passer au thème clair" if dark else "Passer au thème sombre"
        self.variant_btn.setAccessibleName(name)
        self.variant_btn.setToolTip(f"{name} (Ctrl+Maj+L) · flèche : choisir le thème")
        if is_neon():   # text-shadow : halo, ou décalage magenta façon glitch (cyberpunk)
            dx, blur = BRAND_GLOW
            glow = QGraphicsDropShadowEffect(self.name, offset=QPointF(dx, 0), blurRadius=blur,
                                             color=QColor(MARK if dx else NEON))
            self.name.setGraphicsEffect(glow)
        else:
            self.name.setGraphicsEffect(None)
        self.led.update()
        self.stripes.update()
