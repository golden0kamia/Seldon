"""Page Structure : schéma bloc des sous-systèmes de Seldon et de leurs échanges.

Le schéma est décrit ici (``BLOCKS``, ``LINKS``) d'après le pipeline : une phrase va
du micro à l'écoute, puis ``Seldon`` (``seldon/pipeline.py``) identifie la voix, la
confie au cerveau, qui appelle ses outils (tools, MCP, skills, mémoire) et répond par
la voix. Les moteurs de chaque bloc viennent du registre (``seldon.registry.PLUGINS``).
Un clic sur un bloc éclaire ses liens et le décrit ; un double-clic ouvre son fichier.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (
    QFrame, QGraphicsItem, QGraphicsPathItem, QGraphicsScene, QGraphicsView, QHBoxLayout, QLabel, QVBoxLayout,
    QWidget,
)

from seldon.gui import theme
from seldon.registry import PLUGINS

PACKAGE_DIR = Path(__file__).resolve().parent.parent   # le paquet ``seldon`` lui-même


@dataclass(frozen=True)
class Block:
    key: str
    title: str
    col: int
    row: int
    doc: str
    path: str = ""       # relatif à PACKAGE_DIR ; vide : hors de Seldon (matériel, services)
    category: str = ""   # catégorie de plugins dont on affiche les moteurs


@dataclass(frozen=True)
class Link:
    src: str
    dst: str
    label: str
    both: bool = False   # échange dans les deux sens
    at: float = 0.5      # position de l'étiquette le long du lien


BLOCKS = [
    Block("base", "Contrats", 0, 0, "Utterance, Result, Tool, STT, LLM, TTS : les interfaces que chaque "
          "plugin implémente. À lire en premier.", "base.py"),
    Block("config", "Configuration", 1, 0, "config.yaml et secrets de .env ; le registre (registry.py) "
          "crée les plugins choisis.", "config.py"),
    Block("gui", "Interface", 2, 0, "Édite config.yaml, lance Seldon, suit son journal et répond "
          "à ses confirmations.", "gui/window.py"),
    Block("journal", "Journal", 3, 0, "Conversation, réflexions du cerveau et actions, écrites dans "
          "les logs que lit la page Journal.", "journal.py"),
    Block("mic", "Micro · clavier", 0, 1, "Son capté au micro, ou lignes tapées (--text, Journal)."),
    Block("stt", "Écoute", 1, 1, "Micro et reconnaissance vocale (Whisper), ou clavier : "
          "produit des Utterance.", "ear/whisper.py", "stt"),
    Block("pipeline", "Pipeline", 2, 1, "La boucle (classe Seldon) : écoute → voix → cerveau → parole, "
          "pour chaque phrase.", "pipeline.py"),
    Block("tts", "Voix", 3, 1, "Synthèse vocale : dit les réponses ; l'écoute est en pause pendant ce temps.",
          "voice/piper.py", "tts"),
    Block("speaker_out", "Haut-parleur", 4, 1, "Sortie audio (ou console avec --text)."),
    Block("speakers", "Identification des voix", 1, 2, "Reconnaît qui parle ; son outil « voix » "
          "apprend une voix inconnue.", "ear/speakers.py", "speakers"),
    Block("brain", "Cerveau", 2, 2, "Historique et boucle d'outils autour du modèle de langage, "
          "avec cerveau de secours ; répond par l'outil speak.", "brain/brain.py", "brain"),
    Block("tools", "Tools", 3, 2, "Outils en Python appelés par le cerveau (avec confirmation "
          "si besoin) : speak, terminal, caméra…", "tools/speak.py", "tools"),
    Block("memory", "Mémoire", 4, 2, "Conversations sauvegardées après chaque phrase ; outil "
          "« conversations » pour les retrouver et les reprendre.", "brain/memory.py"),
    Block("skills", "Skills", 2, 3, "Outil « skill » : consignes skills/<nom>/SKILL.md chargées "
          "quand une demande s'y prête.", "skills.py"),
    Block("mcp", "Serveurs MCP", 3, 3, "Un outil par serveur MCP (Home Assistant…), proposé au cerveau.",
          "mcp.py", "mcp"),
    Block("services", "Services externes", 4, 3, "Home Assistant, Nextcloud, Feishin, "
          "bureau du PC, caméra…"),
]

LINKS = [
    Link("gui", "config", "édite"),
    Link("config", "pipeline", "crée les plugins"),
    Link("gui", "pipeline", "lance · confirmations", both=True),
    Link("pipeline", "journal", "logs"),
    Link("journal", "gui", "affiché"),
    Link("mic", "stt", "son · texte"),
    Link("stt", "pipeline", "Utterance"),
    Link("pipeline", "tts", "speak()"),
    Link("tts", "speaker_out", "voix"),
    Link("pipeline", "speakers", "identify()", both=True),
    Link("pipeline", "brain", "phrase ⇄ réponse", both=True),
    Link("pipeline", "memory", "save()", at=0.7),
    Link("brain", "tools", "run()"),
    Link("brain", "mcp", "tools/call", at=0.6),
    Link("brain", "skills", "consignes"),
    Link("tools", "services", "HTTP · D-Bus", at=0.6),
    Link("mcp", "services", "MCP"),
]

BOX_W, BOX_H, COL, ROW = 230, 72, 320, 160
MAX_SCALE = 1.3   # « Ajuster » agrandit le schéma jusque-là


class BlockItem(QGraphicsItem):
    """Un sous-système ; l'état (``sel``, ``near``, ``dim``) règle sa couleur."""

    def __init__(self, block: Block, engines: list[str]) -> None:
        super().__init__()
        self.block, self.engines = block, engines
        self.state = ""
        self.setPos(block.col * COL, block.row * ROW)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        tip = [block.doc] + ([f"Moteurs : {', '.join(engines)}"] if engines else [])
        self.setToolTip("\n".join(tip))

    def boundingRect(self) -> QRectF:  # noqa: N802
        return QRectF(-BOX_W / 2, -BOX_H / 2, BOX_W, BOX_H)

    def paint(self, p: QPainter, _option, _widget=None) -> None:
        inside = bool(self.block.path)
        edge = QColor(theme.ACCENT_HI if self.state in ("sel", "near") else theme.CRT_DIM)
        text, dim = QColor(theme.CRT_FG), QColor(theme.CRT_DIM)
        fill = QColor(edge)
        fill.setAlpha(70 if self.state == "sel" else 30 if self.state == "near" else 12)
        if self.state == "dim":
            for c in (edge, text, dim):
                c.setAlpha(80)
        pen = QPen(edge, 2 if self.state == "sel" else 1.2)
        if not inside:
            pen.setStyle(Qt.PenStyle.DashLine)   # hors de Seldon : matériel, services
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(pen)
        p.setBrush(fill)
        radius = 0 if theme.is_neon() else 8
        r = self.boundingRect().adjusted(0.5, 0.5, -0.5, -0.5)
        p.drawRoundedRect(r, radius, radius)
        inner = r.adjusted(10, 8, -10, -8)
        p.setPen(text)
        p.setFont(theme.ui_font(11, QFont.Weight.DemiBold))
        p.drawText(inner, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, self.block.title)
        p.setPen(dim)
        p.setFont(theme.mono_font(8))
        if self.engines:
            sub = " · ".join(self.engines) if len(self.engines) <= 3 else f"{len(self.engines)} {self.block.category}"
        else:
            sub = self.block.path.split("/")[0].removesuffix(".py") if inside else "externe"
        sub = p.fontMetrics().elidedText(sub, Qt.TextElideMode.ElideRight, int(inner.width()))
        p.drawText(inner, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom, sub)


class LinkItem(QGraphicsPathItem):
    """Flèche droite entre deux blocs, avec son étiquette."""

    def __init__(self, link: Link, src: BlockItem, dst: BlockItem) -> None:
        super().__init__()
        self.link, self.src, self.dst = link, src, dst
        self.state = ""
        a, b = src.pos(), dst.pos()
        self.start, self.end = _border(a, b), _border(b, a)
        path = QPainterPath(self.start)
        path.lineTo(self.end)
        self.setPath(path)
        self.setZValue(-1)

    def paint(self, p: QPainter, _option, _widget=None) -> None:
        color = QColor(theme.ACCENT_HI if self.state == "near" else theme.CRT_DIM)
        color.setAlpha(240 if self.state == "near" else 40 if self.state == "dim" else 170)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(color, 1.8 if self.state == "near" else 1.2))
        p.drawLine(self.start, self.end)
        p.setBrush(color)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPolygon(_head(self.start, self.end))
        if self.link.both:
            p.drawPolygon(_head(self.end, self.start))
        # étiquette sur un fond opaque, lisible même là où deux liens se croisent
        p.setFont(theme.mono_font(8))
        fm = p.fontMetrics()
        mid = self.start + (self.end - self.start) * self.link.at
        w, h = fm.horizontalAdvance(self.link.label) + 10, fm.height() + 4
        box = QRectF(mid.x() - w / 2, mid.y() - h / 2, w, h)
        p.setBrush(QColor(theme.CRT_BG))
        p.drawRoundedRect(box, 3, 3)
        text = QColor(theme.CRT_FG if self.state == "near" else theme.CRT_DIM)
        if self.state == "dim":
            text.setAlpha(90)
        p.setPen(text)
        p.drawText(box, Qt.AlignmentFlag.AlignCenter, self.link.label)

    def boundingRect(self) -> QRectF:  # noqa: N802
        return super().boundingRect().adjusted(-70, -14, 70, 14)   # étiquette et pointes comprises


def _border(a: QPointF, b: QPointF) -> QPointF:
    """Point où le segment a → b sort du bloc centré en ``a``."""
    d = b - a
    scale = min(BOX_W / 2 / abs(d.x()) if d.x() else math.inf, BOX_H / 2 / abs(d.y()) if d.y() else math.inf)
    return a + d * scale + d * (6 / math.hypot(d.x(), d.y()))   # petit écart avant le bord


def _head(start: QPointF, end: QPointF, size: float = 9) -> QPolygonF:
    angle = math.atan2(end.y() - start.y(), end.x() - start.x())
    left = end - QPointF(math.cos(angle - 0.4), math.sin(angle - 0.4)) * size
    right = end - QPointF(math.cos(angle + 0.4), math.sin(angle + 0.4)) * size
    return QPolygonF([end, left, right])


class DiagramView(QGraphicsView):
    block_clicked = Signal(str)
    block_activated = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet("background: transparent;")
        self.setAccessibleName("Schéma des sous-systèmes")
        self.blocks: dict[str, BlockItem] = {}
        self.links: list[LinkItem] = []

    def build(self) -> None:
        scene = self.scene()
        scene.clear()
        self.blocks = {b.key: BlockItem(b, sorted(PLUGINS.get(b.category, {}))) for b in BLOCKS}
        self.links = [LinkItem(link, self.blocks[link.src], self.blocks[link.dst]) for link in LINKS]
        for item in [*self.blocks.values(), *self.links]:
            scene.addItem(item)
        scene.setSceneRect(scene.itemsBoundingRect().adjusted(-20, -20, 20, 20))

    def highlight(self, key: str | None) -> None:
        near = {key} | {x for link in LINKS if key in (link.src, link.dst) for x in (link.src, link.dst)}
        for k, item in self.blocks.items():
            item.state = "" if key is None else "sel" if k == key else "near" if k in near else "dim"
        for item in self.links:
            item.state = "" if key is None else "near" if key in (item.link.src, item.link.dst) else "dim"
        self.scene().update()

    def fit(self) -> None:
        self.resetTransform()
        r = self.sceneRect()
        if r.width() and self.viewport().width() > 50:
            scale = min(MAX_SCALE, self.viewport().width() / r.width(), self.viewport().height() / r.height())
            self.scale(scale, scale)
        self.centerOn(r.center())

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self.fit()

    def _block_at(self, event) -> BlockItem | None:
        item = self.itemAt(event.position().toPoint())
        return item if isinstance(item, BlockItem) else None

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            item = self._block_at(event)
            self.block_clicked.emit(item.block.key if item else "")
            return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if (item := self._block_at(event)):
            self.block_activated.emit(item.block.key)
            return
        super().mouseDoubleClickEvent(event)


class StructurePanel(QWidget):
    """Schéma bloc des sous-systèmes, et description du bloc choisi."""

    open_requested = Signal(Path, int)

    def __init__(self, root: Path = PACKAGE_DIR, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.root = root
        self.selected: str | None = None

        top = QHBoxLayout()
        top.setSpacing(theme.SPACE)
        top.addWidget(theme.caption("structure"))
        top.addStretch(1)

        self.view = DiagramView()
        self.view.build()
        self.view.block_clicked.connect(lambda key: self.select(key or None))
        self.view.block_activated.connect(self._open)
        self.detail = QLabel(wordWrap=True)
        self.detail.setFont(theme.mono_font(10))
        screen = theme.NeuCard("screen", radius=20, padding=16)
        screen.add(self.view, 1)
        screen.add(self.detail)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, theme.SPACE, 0, theme.SPACE)
        head = QVBoxLayout()
        head.setContentsMargins(theme.GUTTER, 0, theme.GUTTER, 0)
        head.addLayout(top)
        lay.addLayout(head)
        lay.addWidget(screen, 1)
        self.theme_changed()

    def select(self, key: str | None) -> None:
        self.selected = key if key in self.view.blocks else None
        self.view.highlight(self.selected)
        self._show_detail()

    def _show_detail(self) -> None:
        item = self.view.blocks.get(self.selected) if self.selected else None
        dim = theme.CRT_DIM
        if item is None:
            self.detail.setText(f'<span style="color:{dim}">Clic : liens et rôle d\'un bloc · '
                                "double-clic : ouvrir son code · pointillés : hors de Seldon</span>")
            return
        b = item.block
        lines = [f'<b style="color:{theme.ACCENT_HI}">{b.title}</b> &nbsp; '
                 f'<span style="color:{theme.CRT_FG}">{b.doc}</span>']
        if item.engines:
            lines.append(f'<span style="color:{dim}">moteurs : {", ".join(item.engines)}</span>')
        if b.path:
            lines.append(f'<span style="color:{dim}">seldon/{b.path} · double-clic : ouvrir</span>')
        self.detail.setText("<br>".join(lines))

    def _open(self, key: str) -> None:
        self.select(key)
        if (path := self.view.blocks[key].block.path) and (self.root / path).is_file():
            self.open_requested.emit(self.root / path, 1)

    def theme_changed(self) -> None:
        self._show_detail()
        self.view.scene().update()
