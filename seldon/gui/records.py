"""Éléments enregistrés : voix des locuteurs et conversations de la mémoire.

Chaque vue lit un dossier de ``data/`` (celui de la configuration) et le surveille :
une liste à gauche, la fiche de l'élément choisi à droite. Rien n'est importé des
plugins (numpy, resemblyzer…) : les empreintes ``.npy`` sont lues par leur en-tête.
"""

from __future__ import annotations

import ast
import html
from datetime import datetime
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QFileSystemWatcher, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton, QScrollArea, QSizePolicy,
    QToolButton, QVBoxLayout, QWidget,
)

from seldon.core.memory import Conversation, load_conversations
from seldon.gui import theme

LIST_WIDTH = 320
TITLE_CHARS = 32
VIEW_HEIGHT = 380
NPY_HEADER_MAX = 64 * 1024   # en-tête .npy plus long : fichier suspect, non lu (numpy s'arrête à 10 000 octets)


def voice_samples(path: Path) -> int | None:
    """Nombre d'échantillons d'une empreinte ``.npy`` (lignes du tableau), lu dans l'en-tête.

    Fichier tronqué, en-tête démesuré ou malformé : ``None``, sans jamais lever d'exception.
    """
    try:
        with path.open("rb") as f:
            magic = f.read(8)
            if len(magic) < 8 or magic[:6] != b"\x93NUMPY":
                return None
            size = int.from_bytes(f.read(2 if magic[6] == 1 else 4), "little")
            if size > NPY_HEADER_MAX:
                return None
            header = ast.literal_eval(f.read(size).decode("latin-1"))
    except (OSError, ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return None
    shape = header.get("shape") if isinstance(header, dict) else None
    if not (isinstance(shape, tuple) and all(isinstance(n, int) and n >= 0 for n in shape)):
        return None
    return shape[0] if len(shape) == 2 else 1 if len(shape) == 1 else None


_cache: dict[Path, tuple[tuple, list[Conversation]]] = {}


def conversations(folder: Path) -> list[Conversation]:
    """``load_conversations`` mise en cache : relue seulement si un fichier a changé (nom, taille, date).

    La liste de la page Mémoire et la fiche de chaque voix s'en servent à chaque clic :
    sans cache, toutes les conversations seraient relues à chaque fois.
    """
    try:
        signature = tuple(sorted((p.name, (s := p.stat()).st_mtime_ns, s.st_size) for p in folder.glob("*.json")))
    except OSError:
        signature = None
    cached = _cache.get(folder)
    if signature is not None and cached and cached[0] == signature:
        return cached[1]
    convs = load_conversations(folder)
    if signature is not None:
        _cache[folder] = (signature, convs)
    return convs


def _when(ts: float) -> str:
    try:
        return datetime.fromtimestamp(ts).strftime("%d/%m/%Y %H:%M")
    except (TypeError, ValueError, OverflowError, OSError):   # date absente ou aberrante
        return "?"


def _home(path: Path, base: Path | None = None) -> str:
    """Chemin court : relatif au dossier de la configuration si possible, sinon avec « ~ »."""
    if base is not None and path.is_relative_to(base):
        return str(path.relative_to(base))
    return str(path).replace(str(Path.home()), "~", 1)


def _paragraphs(text: str) -> str:
    return "<br>".join(html.escape(line) for line in text.splitlines())


class RecordsView(QWidget):
    """Liste des éléments d'un dossier et fiche de l'élément sélectionné."""

    open_requested = Signal(Path)
    title = ""

    def __init__(self, folder: Callable[[], Path], base: Path | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.folder, self.base = folder, base.resolve() if base else None
        self.current_dir: Path | None = None
        self.count = QLabel(objectName="hint")
        self.count.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)   # chemin long : rogné
        refresh = QPushButton("Actualiser")
        refresh.clicked.connect(self.refresh)
        head = QHBoxLayout()
        head.setSpacing(theme.SPACE)
        head.addWidget(theme.caption(self.title))
        head.addWidget(self.count, 1)
        self.tools = QHBoxLayout()
        self.tools.setSpacing(theme.SPACE)
        head.addLayout(self.tools)
        head.addWidget(refresh)

        self.list = QListWidget(textElideMode=Qt.TextElideMode.ElideNone)
        self.list.currentItemChanged.connect(lambda *_: self.show_detail())
        list_card = theme.NeuCard("inset", radius=14, padding=theme.SPACE)
        list_card.setFixedWidth(LIST_WIDTH)
        list_card.add(self.list, 1)
        self.detail = QLabel(wordWrap=True, textFormat=Qt.TextFormat.RichText)
        self.detail.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        scroll = QScrollArea(widgetResizable=True, frameShape=QFrame.Shape.NoFrame)
        scroll.setWidget(self.detail)
        detail_card = theme.NeuCard("inset", radius=14, padding=theme.PAD)
        detail_card.add(scroll, 1)
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(theme.PAD)
        body.addWidget(list_card)
        body.addWidget(detail_card, 1)
        body_box = QWidget()
        body_box.setLayout(body)
        body_box.setFixedHeight(VIEW_HEIGHT)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(theme.SPACE)
        lay.addLayout(head)
        lay.addWidget(body_box)

        self.watcher = QFileSystemWatcher(self)
        self._stale = False   # dossier modifié pendant que la vue était cachée
        self._error = ""      # dernière erreur de lecture du dossier
        self._debounce = QTimer(self, singleShot=True, interval=300)
        self._debounce.timeout.connect(self.refresh)
        self.watcher.directoryChanged.connect(lambda _p: self._folder_changed())
        self.refresh()

    def _folder_changed(self) -> None:
        """Seldon écrit après chaque échange : vue cachée, on relit seulement à son affichage."""
        if self.isVisible():
            self._debounce.start()
        else:
            self._stale = True

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if self._stale:
            self._stale = False
            self.refresh()

    # --- à fournir par les sous-classes ---------------------------------------
    def load(self, folder: Path) -> list[tuple[str, str, str]]:
        """Éléments du dossier : (clé, texte affiché, infobulle)."""
        raise NotImplementedError

    def render(self, key: str) -> str:
        """Fiche HTML de l'élément ``key``."""
        raise NotImplementedError

    def empty_text(self, folder: Path) -> str:
        raise NotImplementedError

    # --- mécanique commune ----------------------------------------------------
    def selected(self) -> str | None:
        item = self.list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def follow_folder(self) -> None:
        """Recharge si le dossier configuré a changé (option modifiée dans le formulaire)."""
        if self.folder().resolve() != self.current_dir:
            self.refresh()

    def refresh(self) -> None:
        folder = self.folder().resolve()
        if folder != self.current_dir and self.watcher.directories():
            self.watcher.removePaths(self.watcher.directories())
        self.current_dir = folder
        if folder.is_dir() and str(folder) not in self.watcher.directories():
            self.watcher.addPath(str(folder))
        keep = self.selected()
        self._stale = False
        try:
            entries, self._error = self.load(folder), ""
        except Exception as e:  # noqa: BLE001 — un fichier abîmé ne doit pas casser l'interface
            entries, self._error = [], f"Lecture impossible de {_home(folder, self.base)} : {e}"
        self.list.blockSignals(True)
        self.list.clear()
        for key, text, tip in entries:
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, key)
            item.setToolTip(tip)
            self.list.addItem(item)
        keys = [k for k, _t, _tip in entries]
        self.list.setCurrentRow(keys.index(keep) if keep in keys else 0 if keys else -1)
        self.list.blockSignals(False)
        self.count.setText(f"{len(entries)} · {_home(folder, self.base)}")
        self.count.setToolTip(str(folder))
        self.show_detail()

    def show_detail(self) -> None:
        key = self.selected()
        if key is None:
            text = _paragraphs(self._error or self.empty_text(self.current_dir))
        else:
            try:
                text = self.render(key)
            except Exception as e:  # noqa: BLE001 — fiche malformée : on le dit au lieu de planter
                text = _paragraphs(f"Élément illisible : {e}")
        self.detail.setText(text)
        self.update_tools()

    def update_tools(self) -> None:
        """Active les boutons propres à la vue selon la sélection."""


class SpeakersView(RecordsView):
    """Voix enregistrées (``<voices_dir>/<nom>.npy``) et conversations où chacun a parlé."""

    title = "voix enregistrées"

    def __init__(self, voices_dir: Callable[[], Path], conversations_dir: Callable[[], Path],
                 base: Path | None = None, parent: QWidget | None = None) -> None:
        self.conversations_dir = conversations_dir
        self.paths: dict[str, Path] = {}
        super().__init__(voices_dir, base, parent)

    def load(self, folder: Path) -> list[tuple[str, str, str]]:
        self.paths = {p.stem: p for p in sorted(folder.glob("*.npy"), key=lambda p: p.stem.lower())} if folder.is_dir() else {}
        return [(name, name, str(path)) for name, path in self.paths.items()]

    def render(self, key: str) -> str:
        path = self.paths[key]
        samples = voice_samples(path)
        try:
            enrolled = _when(path.stat().st_mtime)
        except OSError:
            enrolled = "?"
        try:
            convs = [c for c in conversations(self.conversations_dir().resolve()) if key in (c.speakers or [])]
        except Exception:  # noqa: BLE001 — la fiche de la voix reste utile sans ses conversations
            convs = []
        rows = [f"<p><b>{html.escape(key)}</b></p>",
                f"<p>Échantillons de voix : {samples if samples is not None else '?'}<br>"
                f"Dernier enregistrement : {enrolled}<br>"
                f"Fichier : {html.escape(_home(path, self.base))}</p>"]
        if convs:
            rows.append(f"<p>A participé à {len(convs)} conversation(s) :<br>"
                        + "<br>".join(f"• {_when(c.updated)} — {html.escape(c.title or 'sans titre')}" for c in convs[:8])
                        + ("<br>• …" if len(convs) > 8 else "") + "</p>")
        else:
            rows.append("<p>Aucune conversation enregistrée avec cette voix.</p>")
        rows.append(f"<p>Ajouter des échantillons : <code>python -m seldon.cli.enroll {html.escape(key)}</code></p>")
        return "".join(rows)

    def empty_text(self, folder: Path) -> str:
        return (f"Aucune voix enregistrée dans {_home(folder, self.base)}.\n\n"
                "Pour en ajouter une : python -m seldon.cli.enroll <Nom>")


class ConversationsView(RecordsView):
    """Conversations sauvegardées : résumé (calculé localement) ou transcription complète."""

    title = "conversations enregistrées"

    def __init__(self, folder: Callable[[], Path], base: Path | None = None, parent: QWidget | None = None) -> None:
        self.convs: dict[str, Conversation] = {}
        self.modes: list[QToolButton] = []
        super().__init__(folder, base, parent)
        for i, (text, tip) in enumerate((("Résumé", "Période, participants, thèmes et demandes"),
                                        ("Transcription", "Tous les messages de la conversation"))):
            b = QToolButton(text=text, toolTip=tip, checkable=True, autoExclusive=True, objectName="mode")
            b.setChecked(i == 0)
            b.clicked.connect(self.show_detail)
            self.tools.addWidget(b)
            self.modes.append(b)
        self.open_btn = QPushButton("Ouvrir le fichier")
        self.open_btn.setToolTip("Ouvrir le JSON de la conversation dans l'éditeur")
        self.open_btn.clicked.connect(self._open)
        self.tools.addWidget(self.open_btn)
        self.show_detail()

    def load(self, folder: Path) -> list[tuple[str, str, str]]:
        self.convs = {str(c.id): c for c in conversations(folder)}
        out = []
        for key, c in self.convs.items():
            title = str(c.title or "sans titre")
            short = title if len(title) <= TITLE_CHARS else title[:TITLE_CHARS].rsplit(" ", 1)[0] + "…"
            speakers = [str(s) for s in c.speakers or []]
            who = f"\n{', '.join(speakers)}" if speakers else ""
            count = len(c.transcript) if isinstance(c.transcript, list) else 0
            out.append((key, f"{short}\n{_when(c.updated)} · {count} msg", title + who))
        return out

    def render(self, key: str) -> str:
        conv = self.convs[key]
        head = f"<p><b>{html.escape(str(conv.title or 'sans titre'))}</b></p>"
        if self.modes and self.modes[1].isChecked():
            lines = []
            for m in conv.transcript:
                if not isinstance(m, dict):
                    continue
                who = {"user": m.get("speaker") or "inconnu", "summary": "Résumé"}.get(m.get("role"), "Seldon")
                when = _when(m["ts"])[-5:] if m.get("ts") else ""   # HH:MM
                lines.append(f"<p><b>{html.escape(str(who))}</b> <span style='color:{theme.MUTED}'>{when}</span><br>"
                             f"{_paragraphs(str(m.get('text', '')))}</p>")
            return head + "".join(lines)
        return head + f"<p>{_paragraphs(conv.summary())}</p>"

    def empty_text(self, folder: Path) -> str:
        return (f"Aucune conversation enregistrée dans {_home(folder, self.base)}.\n\n"
                "Elles apparaissent ici dès la première prise de parole avec la mémoire activée.")

    def update_tools(self) -> None:
        on = self.selected() is not None
        for b in [*self.modes, getattr(self, "open_btn", None)]:
            if b is not None:
                b.setEnabled(on)

    def _open(self) -> None:
        if (key := self.selected()) is None or self.current_dir is None:
            return
        # l'identifiant vient du contenu du fichier : « ../ » ne doit pas mener hors du dossier
        path = (self.current_dir / f"{key}.json").resolve()
        if path.parent == self.current_dir and path.is_file():
            self.open_requested.emit(path)
