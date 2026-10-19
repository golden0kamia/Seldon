"""Thème « cassette futurism » en néomorphisme, accent vert.

Surfaces en plastique mastic (beige chaud) — ou, en variante sombre, en
plastique brun-noir de chaîne hi-fi — en relief doux, l'éditeur en écran
cathodique à phosphore vert, voyants LED et bandes de cassette. Qt ne sait pas
dessiner de ``box-shadow`` en QSS : les ombres néomorphiques (claire en haut à
gauche, sombre en bas à droite) sont peintes par ``NeuCard`` ; les contrôles
imitent le relief par dégradés et bordures claires/sombres.

Les couleurs sont des variables du module : ``apply(app, "dark")`` les remplace
par celles de la variante puis redessine tout (feuille de style, icônes SVG
régénérées, cartes et éditeurs via leur méthode ``theme_changed``).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor, QFont, QFontDatabase, QLinearGradient, QPainter, QPainterPath, QPalette, QPixmap,
    QRadialGradient,
)
from PySide6.QtWidgets import (
    QAbstractSpinBox, QApplication, QFrame, QHBoxLayout, QLabel, QSizePolicy, QToolButton, QVBoxLayout,
    QWidget,
)

# --- palettes --------------------------------------------------------------
LIGHT = dict(
    BG="#e2dac9",            # plastique mastic
    BG_HI="#ece6d9",
    BG_LO="#d3cab6",
    SHADOW_DARK="#b3a991",
    SHADOW_LIGHT="#fbf8f0",
    TEXT="#35302a",
    MUTED="#857b69",
    ACCENT="#2f9d5c",        # vert d'accent
    ACCENT_HI="#4fd283",
    ACCENT_LO="#1d6e3e",
    ACCENT_TEXT="#1d6e3e",   # vert lisible sur la surface
    SELECT_BG="rgba(47, 157, 92, 0.18)",
    WARN="#c8553d",
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
    MUTED="#9c917e",
    ACCENT="#3cb46c",
    ACCENT_HI="#5ee08f",
    ACCENT_LO="#22804a",
    ACCENT_TEXT="#62dc92",
    SELECT_BG="rgba(94, 224, 143, 0.16)",
    WARN="#e0705a",
    CRT_BG="#0e120f",
    CRT_GUTTER="#0a0d0b",
    SHADOW_STRENGTH=1.6,
)
PALETTES = {"light": LIGHT, "dark": DARK}
VARIANT = "light"

# communs aux deux variantes
ORANGE = "#d8612c"        # bandes de cassette (décor)
AMBER = "#e6a23c"
CRT_FG = "#9be8ad"
CRT_DIM = "#4f7357"
CRT_SELECT = "#2b5a3a"
SYNTAX = dict(keyword="#5ef08a", string="#e8b04a", comment="#5f7f64", number="#7fd9c6",
              key="#b5f2c3", builtin="#6fcbe6", decorator="#e8803a", heading="#5ef08a")

# valeurs courantes (remplacées par set_variant)
BG = BG_HI = BG_LO = SHADOW_DARK = SHADOW_LIGHT = TEXT = MUTED = ""
ACCENT = ACCENT_HI = ACCENT_LO = ACCENT_TEXT = SELECT_BG = WARN = CRT_BG = CRT_GUTTER = ""
SHADOW_STRENGTH = 1.0


def set_variant(name: str) -> None:
    global VARIANT
    VARIANT = name if name in PALETTES else "light"
    globals().update(PALETTES[VARIANT])


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

# Share Tech pour l'interface, Share Tech Mono pour le code, les valeurs et la signalétique
UI_FAMILIES = ["Share Tech", "IBM Plex Sans Condensed", "DejaVu Sans Condensed", "Sans Serif"]
MONO_FAMILIES = ["Share Tech Mono", "IBM Plex Mono", "DejaVu Sans Mono", "Monospace"]


def _family(candidates: list[str]) -> str:
    available = set(QFontDatabase.families())
    return next((f for f in candidates if f in available), candidates[-1])


def ui_font(size: float = 11.5, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    f = QFont(_family(UI_FAMILIES))
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

/* boutons : touches d'appareil en relief */
QPushButton {{ {_raised()} border-radius: 12px; padding: 7px 18px; min-height: 18px; }}
QPushButton:hover {{ color: {ACCENT_TEXT}; }}
QPushButton:pressed {{ {_inset()} padding-top: 8px; padding-bottom: 6px; }}
QPushButton:disabled {{ color: {MUTED}; background: {BG}; border-color: {BG}; }}
QPushButton#primary {{ color: white;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {ACCENT_HI}, stop:1 {ACCENT});
    border: 1px solid {ACCENT}; border-top-color: #8de6ac; border-left-color: #8de6ac;
    border-bottom-color: {ACCENT_LO}; border-right-color: {ACCENT_LO}; }}
QPushButton#primary:pressed {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {ACCENT_LO}, stop:1 {ACCENT}); }}
QPushButton#primary:disabled {{ color: {MUTED}; background: {BG}; border-color: {BG}; }}

QToolButton {{ {_raised()} border-radius: 10px; padding: 3px; min-width: 20px; min-height: 20px; }}
QToolButton:hover {{ color: {ACCENT_TEXT}; }}
QToolButton:pressed, QToolButton:checked {{ {_inset()} color: {ACCENT_TEXT}; }}
QToolButton#mode {{ font-family: "{mono}"; font-size: 10pt; padding: 6px 16px; border-radius: 13px; color: {MUTED}; }}
QToolButton#mode:hover {{ color: {TEXT}; }}
QToolButton#mode:checked {{ {_inset()} color: {ACCENT_TEXT}; }}
QToolButton[autoRaise="true"] {{ background: transparent; border: none; color: {MUTED}; }}
QToolButton[autoRaise="true"]:hover {{ color: {WARN}; }}

/* champs : creusés dans la coque */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ {_inset()} border-radius: 10px; padding: 5px {FIELD_PAD}px;
    min-height: 20px; selection-background-color: {ACCENT}; selection-color: white; font-family: "{mono}"; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border: 1px solid {ACCENT}; border-top-color: {ACCENT_LO}; border-left-color: {ACCENT_LO}; }}
QLineEdit[invalid="true"] {{ border: 1px solid {WARN}; }}
QComboBox::drop-down {{ border: none; width: 26px; }}
QComboBox::down-arrow {{ image: {icon("chevron-down")}; width: 14px; height: 14px; margin-right: 8px; }}
QComboBox QAbstractItemView {{ background: {BG_HI}; border: 1px solid {SHADOW_DARK}; border-radius: 8px; padding: 4px;
    selection-background-color: {ACCENT}; selection-color: white; outline: 0; }}
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

/* onglets : languettes en relief, l'actif enfoncé et éclairé */
QTabWidget::pane {{ border: none; }}
QTabBar {{ qproperty-drawBase: 0; }}
QTabBar::tab {{ {_raised()} border-radius: 11px; padding: 6px 14px; margin: 6px 4px 8px 4px; color: {MUTED}; }}
QTabBar::tab:hover {{ color: {TEXT}; }}
QTabBar::tab:selected {{ {_inset()} color: {ACCENT_TEXT}; }}
QTabBar::close-button {{ image: {icon("close")}; subcontrol-position: right; margin-left: 6px; }}
QTabBar::close-button:hover {{ image: {icon("close-hover")}; }}

/* listes et arborescence posées dans des cartes */
QListWidget, QTreeView {{ background: transparent; border: none; font-size: 11.5pt;
    show-decoration-selected: 0; }}
QListWidget::item {{ padding: 8px 12px; margin: 3px 2px; border-radius: 10px; border-left: 3px solid transparent; }}
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

QMenuBar {{ background: {BG}; padding: 4px 8px; }}
QMenuBar::item {{ padding: 5px 12px; border-radius: 8px; background: transparent; }}
QMenuBar::item:selected {{ {_inset()} color: {ACCENT_TEXT}; }}
QMenu {{ background: {BG_HI}; border: 1px solid {SHADOW_DARK}; border-radius: 10px; padding: 6px; }}
QMenu::item {{ padding: 6px 22px 6px 14px; border-radius: 7px; }}
QMenu::item:selected {{ background: {ACCENT}; color: white; }}
QMenu::separator {{ height: 1px; background: {SHADOW_DARK}; margin: 5px 8px; }}
QMenu::indicator {{ width: 12px; height: 12px; border-radius: 6px; margin-left: 4px; {_inset()} }}
QMenu::indicator:checked {{ background: {ACCENT_HI}; border: 1px solid {ACCENT_LO}; }}

/* barre d'état : afficheur à phosphore */
QStatusBar {{ background: {CRT_BG}; color: {CRT_FG}; font-family: "{mono}"; font-size: 9pt;
    min-height: 26px; padding: 0 10px; border-top: 2px solid {SHADOW_DARK}; }}
QStatusBar QToolButton {{ background: transparent; border: none; color: {CRT_FG}; }}
QStatusBar QLabel {{ color: {CRT_FG}; padding: 0 8px; }}
QStatusBar::item {{ border: none; }}

/* éditeur : écran cathodique */
QPlainTextEdit#code {{ background: {CRT_BG}; color: {CRT_FG}; border: none; font-family: "{mono}";
    selection-background-color: {CRT_SELECT}; selection-color: #e6ffec; }}
QPlainTextEdit#code QScrollBar::handle {{ background: {CRT_DIM}; }}
QWidget#findbar {{ background: transparent; }}
QLabel#banner {{ background: {AMBER}; color: #2a2520; padding: 6px 10px; border-radius: 9px; }}
QLabel#hint {{ color: {MUTED}; }}
QLabel#findinfo {{ color: {WARN}; }}
QLabel#pageTitle {{ color: {TEXT}; }}
QLabel#path {{ color: {MUTED}; font-family: "{mono}"; font-size: 9pt; }}
QScrollArea#imageview > QWidget > QWidget {{ background: {CRT_BG}; }}
"""


class _SpinBoxPadding(QObject):
    """Aligne le texte des champs numériques sur celui des champs texte.

    Le champ interne d'un QSpinBox ignore le ``padding`` de la feuille de style :
    on lui donne la même marge (padding + bordure) dès qu'il est préparé.
    """

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() == QEvent.Type.Polish and isinstance(obj, QAbstractSpinBox):
            obj.lineEdit().setTextMargins(FIELD_PAD + 1, 0, 0, 0)
        return False


_spin_padding = _SpinBoxPadding()


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
        (QPalette.ColorRole.ButtonText, TEXT), (QPalette.ColorRole.Highlight, ACCENT),
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

    Les ombres douces sont précalculées dans un pixmap à chaque redimensionnement.
    """

    SHADOW = 9      # marge réservée autour de la carte pour les ombres

    def __init__(self, kind: str = "raised", radius: float = 18, padding: int = 12,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.kind, self.radius = kind, radius
        self._cache: QPixmap | None = None
        self._cache_size = QSize()
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        m = self.SHADOW + padding
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(m, m, m, m)
        self.body.setSpacing(8)

    def add(self, widget: QWidget, stretch: int = 0) -> QWidget:
        self.body.addWidget(widget, stretch)
        return widget

    def theme_changed(self) -> None:
        self._cache = None
        self.update()

    def resizeEvent(self, event) -> None:  # noqa: N802
        self._cache = None
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
        s = self.SHADOW
        rect = QRectF(self.rect()).adjusted(s, s, -s, -s)
        r = self.radius
        if self.kind == "inset":
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
                glow.setColorAt(0, QColor(80, 220, 130, 18))
                glow.setColorAt(1, QColor(0, 0, 0, 0))
                p.fillPath(_rounded(glass, r - 8), glow)
        p.end()
        return pix

    def _paint_shadows(self, p: QPainter, rect: QRectF, r: float) -> None:
        steps, offset = self.SHADOW, 4
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


class Stripes(QWidget):
    """Trois bandes obliques façon étiquette de cassette (orange, ambre, vert)."""

    def __init__(self, width: int = 64, height: int = 18, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(width, height)

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
    """Bandeau supérieur : nom de l'appareil, bandes de cassette, voyant d'état et bascule clair/sombre."""

    toggle_variant = Signal()
    mode_changed = Signal(int)   # 0 = paramètres, 1 = éditeur

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        name = QLabel("SELDON")
        f = mono_font(17)
        f.setWeight(QFont.Weight.Bold)
        f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 6)
        name.setFont(f)
        model = caption("Assistant vocal · console de configuration · mk-1")
        self.led = Led(12)
        self.state = caption("prêt")
        self.mode_buttons: list[QToolButton] = []
        modes = QHBoxLayout()
        modes.setSpacing(6)
        for i, (text, tip) in enumerate((("⚙  PARAMÈTRES", "Réglages de config.yaml (Ctrl+1)"),
                                        ("✎  ÉDITEUR", "Fichiers et éditeur de texte (Ctrl+2)"))):
            b = QToolButton(text=text, toolTip=tip, checkable=True, autoExclusive=True, objectName="mode")
            b.clicked.connect(lambda _=False, i=i: self.mode_changed.emit(i))
            modes.addWidget(b)
            self.mode_buttons.append(b)
        self.mode_buttons[0].setChecked(True)
        self.variant_btn = QToolButton(toolTip="Basculer clair / sombre (Ctrl+Maj+L)")
        self.variant_btn.setFont(ui_font(11))
        self.variant_btn.clicked.connect(self.toggle_variant.emit)
        self.theme_changed()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(22, 8, 22, 2)
        lay.setSpacing(14)
        lay.addWidget(name)
        lay.addWidget(Stripes())
        lay.addWidget(model)
        lay.addStretch(1)
        lay.addLayout(modes)
        lay.addStretch(1)
        lay.addWidget(self.led)
        lay.addWidget(self.state)
        lay.addSpacing(10)
        lay.addWidget(self.variant_btn)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

    def set_mode(self, index: int) -> None:
        self.mode_buttons[index].setChecked(True)

    def set_state(self, text: str, color: str | None) -> None:
        self.state.setText(text)
        self.led.set_color(color)

    def theme_changed(self) -> None:
        self.variant_btn.setText("☀" if VARIANT == "dark" else "☾")
        self.led.update()
