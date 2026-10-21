"""Éditeur de texte : numéros de ligne, coloration, recherche, onglets."""

from __future__ import annotations

import codecs
import re
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QFileSystemWatcher, QRect, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (
    QAction, QColor, QFont, QKeySequence, QPainter, QPixmap,
    QSyntaxHighlighter, QTextCharFormat, QTextCursor, QTextDocument, QTextFormat,
)
from PySide6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QScrollArea,
    QTabBar, QTabWidget, QTextEdit, QToolButton, QVBoxLayout, QWidget,
)

from seldon.gui import theme
from seldon.gui.fsutil import write_atomic

MAX_BYTES = 5 * 1024 * 1024
HIGHLIGHT_MAX = 1024 * 1024   # au-delà, pas de coloration : elle figerait l'interface (~0,3 s par Mo)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".svg"}

LANGUAGES = {
    ".py": "python", ".pyw": "python",
    ".yaml": "yaml", ".yml": "yaml",
    ".json": "json", ".jsonl": "json",
    ".md": "markdown",
    ".toml": "toml", ".ini": "toml", ".cfg": "toml",
    ".sh": "shell", ".bash": "shell", ".zsh": "shell",
}


def language_of(path: Path) -> str:
    if path.name in (".gitignore", ".env"):
        return "shell"
    return LANGUAGES.get(path.suffix.lower(), "text")


def read_text(path: Path) -> tuple[str, str, str]:
    """Texte d'un fichier (fins de ligne ``\\n``), son encodage et sa fin de ligne d'origine.

    UTF-8 (avec ou sans BOM), sinon Latin-1 : décodage sans perte, le fichier est
    réenregistré octet pour octet au lieu d'être abîmé par des caractères de remplacement.
    """
    data = path.read_bytes()
    if b"\0" in data[:8192]:
        raise ValueError("fichier binaire")
    encoding = "utf-8-sig" if data.startswith(codecs.BOM_UTF8) else "utf-8"
    try:
        text = data.decode(encoding)
    except UnicodeDecodeError:
        encoding = "latin-1"
        text = data.decode(encoding)
    newline = "\r\n" if "\r\n" in text else "\n"
    return text.replace("\r\n", "\n"), encoding, newline


# --- coloration syntaxique ---------------------------------------------------

def _fmt(color: str, bold: bool = False, italic: bool = False) -> QTextCharFormat:
    f = QTextCharFormat()
    f.setForeground(QColor(color))
    if bold:
        f.setFontWeight(QFont.Weight.Bold)
    f.setFontItalic(italic)
    return f


PY_KEYWORDS = (
    "False None True and as assert async await break class continue def del elif else except "
    "finally for from global if import in is lambda nonlocal not or pass raise return try while "
    "with yield match case self"
).split()


class Highlighter(QSyntaxHighlighter):
    def __init__(self, document: QTextDocument, language: str) -> None:
        super().__init__(document)
        self.language = language
        c = {k: _fmt(v) for k, v in theme.SYNTAX.items()}
        c["keyword"].setFontWeight(QFont.Weight.Bold)
        c["heading"].setFontWeight(QFont.Weight.Bold)
        c["comment"].setFontItalic(True)
        self.c = c
        self.rules: list[tuple[re.Pattern, QTextCharFormat, int]] = []
        string = (r'"(?:[^"\\]|\\.)*"|' r"'(?:[^'\\]|\\.)*'")
        number = r"\b-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\b"
        if language == "python":
            self._add(r"\b(?:%s)\b" % "|".join(PY_KEYWORDS), "keyword")
            self._add(r"\b(?:print|len|range|dict|list|set|str|int|float|bool|isinstance|super|open)\b", "builtin")
            self._add(r"\b(?:def|class)\s+(\w+)", "builtin", 1)
            self._add(r"^\s*@[\w.]+", "decorator")
            self._add(number, "number")
            self._add(r'[rbfu]{0,2}' + string, "string")
        elif language == "yaml":
            self._add(r"^\s*(?:-\s+)?([\w.\-/ ]+?)\s*:(?=\s|$)", "key", 1)
            self._add(r"(?<=[:\-]\s)\s*(?:true|false|null|yes|no|~)\b", "keyword")
            self._add(r"(?<=[:\-\[,]\s)\s*" + number, "number")
            self._add(string, "string")
        elif language == "json":
            self._add(r"\b(?:true|false|null)\b", "keyword")
            self._add(number, "number")
            self._add(r'"(?:[^"\\]|\\.)*"', "string")
            self._add(r'("(?:[^"\\]|\\.)*")\s*:', "key", 1)
        elif language == "toml":
            self._add(r"^\s*\[[^\]]*\]+", "heading")
            self._add(r"^\s*([\w.\-]+)\s*=", "key", 1)
            self._add(r"\b(?:true|false)\b", "keyword")
            self._add(number, "number")
            self._add(string, "string")
        elif language == "shell":
            self._add(r"\b(?:if|then|else|elif|fi|for|while|do|done|case|esac|function|return|export|local)\b", "keyword")
            self._add(r"\$\{?\w+\}?", "key")
            self._add(string, "string")
        elif language == "markdown":
            self._add(r"^#{1,6}\s.*", "heading")
            self._add(r"\*\*[^*]+\*\*|__[^_]+__", "keyword")
            self._add(r"`[^`]+`", "string")
            self._add(r"\[[^\]]+\]\([^)]+\)", "builtin")
            self._add(r"^\s*(?:[-*+]|\d+\.)\s", "number")

    HASH_COMMENTS = ("python", "yaml", "toml", "shell")

    def _comment_start(self, text: str) -> int:
        """Position du « # » de commentaire, hors chaînes (-1 si aucun)."""
        quote, i = None, 0
        yaml_like = self.language != "python"
        while i < len(text):
            ch = text[i]
            if quote:
                if ch == "\\" and quote == '"':
                    i += 1
                elif ch == quote:
                    quote = None
            elif ch in "\"'":
                # en YAML / shell, une apostrophe au milieu d'un mot (« l'appel ») n'ouvre pas de chaîne
                before = text[:i].rstrip()
                if not yaml_like or not before or before[-1] in ":-[,{=( ":
                    quote = ch
            elif ch == "#" and (not yaml_like or i == 0 or text[i - 1].isspace()):
                return i
            i += 1
        return -1

    def _add(self, pattern: str, fmt: str, group: int = 0) -> None:
        self.rules.append((re.compile(pattern), self.c[fmt], group))

    def highlightBlock(self, text: str) -> None:  # noqa: N802 (API Qt)
        for pattern, fmt, group in self.rules:
            for m in pattern.finditer(text):
                start, end = m.span(group)
                if start >= 0:
                    self.setFormat(start, end - start, fmt)
        if self.language in self.HASH_COMMENTS and (start := self._comment_start(text)) >= 0:
            self.setFormat(start, len(text) - start, self.c["comment"])
        if self.language == "python":
            self._docstrings(text)
        if self.language == "markdown":   # blocs de code ``` … ```
            fence = text.lstrip().startswith("```")
            inside = self.previousBlockState() == 1
            if inside or fence:
                self.setFormat(0, len(text), self.c["string"])
            self.setCurrentBlockState(int(inside != fence))

    def _docstrings(self, text: str) -> None:
        """Chaînes triples sur plusieurs lignes (état de bloc 2 = à l'intérieur)."""
        fmt, pos = self.c["string"], 0
        self.setCurrentBlockState(0)
        if self.previousBlockState() == 2:
            end = text.find('"""')
            if end < 0:
                self.setFormat(0, len(text), fmt)
                self.setCurrentBlockState(2)
                return
            self.setFormat(0, end + 3, fmt)
            pos = end + 3
        while (start := text.find('"""', pos)) >= 0:
            end = text.find('"""', start + 3)
            if end < 0:
                self.setFormat(start, len(text) - start, fmt)
                self.setCurrentBlockState(2)
                return
            self.setFormat(start, end + 3 - start, fmt)
            pos = end + 3


# --- zone d'édition ----------------------------------------------------------

class LineNumbers(QWidget):
    def __init__(self, editor: "CodeEditor") -> None:
        super().__init__(editor)
        self.editor = editor

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(self.editor.gutter_width(), 0)

    def paintEvent(self, event) -> None:  # noqa: N802
        self.editor.paint_gutter(event)


class CodeEditor(QPlainTextEdit):
    def __init__(self, language: str = "text", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("code")
        self.setFont(theme.mono_font(11.5))
        self.setFrameShape(QPlainTextEdit.Shape.NoFrame)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.highlighter: Highlighter | None = None
        self.set_language(language)
        self.gutter = LineNumbers(self)
        self.blockCountChanged.connect(lambda _: self.setViewportMargins(self.gutter_width(), 0, 0, 0))
        self.updateRequest.connect(self._update_gutter)
        self.cursorPositionChanged.connect(self._highlight_line)
        self.setViewportMargins(self.gutter_width(), 0, 0, 0)
        self._highlight_line()

    def set_language(self, language: str) -> None:
        """Change la coloration et l'indentation (ex. après « Enregistrer sous »)."""
        self.language = language
        self.indent = "  " if language in ("yaml", "json") else "    "
        self.setTabStopDistance(self.fontMetrics().horizontalAdvance(" ") * len(self.indent))
        if self.highlighter is not None:
            self.highlighter.setDocument(None)
            self.highlighter = None
        if language != "text" and self.document().characterCount() <= HIGHLIGHT_MAX:
            self.highlighter = Highlighter(self.document(), language)

    def theme_changed(self) -> None:
        if self.highlighter is not None:   # couleurs de syntaxe propres à la variante
            self.set_language(self.language)
        self._highlight_line()
        self.gutter.update()

    def gutter_width(self) -> int:
        digits = max(3, len(str(self.blockCount())))
        return 12 + self.fontMetrics().horizontalAdvance("9") * digits

    def _update_gutter(self, rect: QRect, dy: int) -> None:
        if dy:
            self.gutter.scroll(0, dy)
        else:
            self.gutter.update(0, rect.y(), self.gutter.width(), rect.height())

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        cr = self.contentsRect()
        self.gutter.setGeometry(QRect(cr.left(), cr.top(), self.gutter_width(), cr.height()))

    def paint_gutter(self, event) -> None:
        painter = QPainter(self.gutter)
        painter.fillRect(event.rect(), QColor(theme.CRT_GUTTER))
        current = self.textCursor().blockNumber()
        block = self.firstVisibleBlock()
        top = round(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        height = self.fontMetrics().height()
        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible():
                n = block.blockNumber()
                painter.setPen(QColor(theme.CRT_FG if n == current else theme.CRT_DIM))
                painter.drawText(0, top, self.gutter.width() - 6, height, Qt.AlignmentFlag.AlignRight, str(n + 1))
            top += round(self.blockBoundingRect(block).height())
            block = block.next()

    def _highlight_line(self) -> None:
        sel = QTextEdit.ExtraSelection()
        color = QColor(theme.ACCENT_HI)
        color.setAlpha(22)
        sel.format.setBackground(color)
        sel.format.setProperty(QTextFormat.Property.FullWidthSelection, True)
        sel.cursor = self.textCursor()
        sel.cursor.clearSelection()
        self.setExtraSelections([sel])

    def keyPressEvent(self, event) -> None:  # noqa: N802
        key = event.key()
        cursor = self.textCursor()
        if key == Qt.Key.Key_Tab and not cursor.hasSelection():
            cursor.insertText(self.indent)
            return
        if key in (Qt.Key.Key_Tab, Qt.Key.Key_Backtab) and cursor.hasSelection():
            self._shift_lines(key == Qt.Key.Key_Backtab)
            return
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            line = cursor.block().text()[: cursor.positionInBlock()]
            indent = line[: len(line) - len(line.lstrip())]
            if line.rstrip().endswith(":") and self.language in ("python", "yaml"):
                indent += self.indent
            elif line.rstrip().endswith(("{", "[")) and self.language == "json":
                indent += self.indent
            cursor.insertText("\n" + indent)
            self.ensureCursorVisible()
            return
        if key == Qt.Key.Key_Slash and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.toggle_comment()
            return
        super().keyPressEvent(event)

    def _selected_blocks(self):
        cursor = self.textCursor()
        doc = self.document()
        first = doc.findBlock(cursor.selectionStart())
        last = doc.findBlock(max(cursor.selectionStart(), cursor.selectionEnd() - (1 if cursor.hasSelection() else 0)))
        block = first
        while block.isValid():
            yield block
            if block == last:
                break
            block = block.next()

    def _edit_blocks(self, transform) -> None:
        cursor = self.textCursor()
        cursor.beginEditBlock()
        for block in list(self._selected_blocks()):
            new = transform(block.text())
            if new != block.text():
                c = QTextCursor(block)
                c.movePosition(QTextCursor.MoveOperation.EndOfBlock, QTextCursor.MoveMode.KeepAnchor)
                c.insertText(new)
        cursor.endEditBlock()

    def _shift_lines(self, back: bool) -> None:
        if back:
            self._edit_blocks(lambda t: t[len(self.indent):] if t.startswith(self.indent) else t.lstrip(" "))
        else:
            self._edit_blocks(lambda t: self.indent + t if t else t)

    def toggle_comment(self) -> None:
        mark = {"python": "#", "yaml": "#", "toml": "#", "shell": "#"}.get(self.language)
        if not mark:
            return
        lines = [b.text() for b in self._selected_blocks()]
        uncomment = all(l.lstrip().startswith(mark) for l in lines if l.strip())

        def tr(t: str) -> str:
            if not t.strip():
                return t
            body = t.lstrip()
            pad = t[: len(t) - len(body)]
            if uncomment:
                body = body[len(mark):]
                return pad + (body[1:] if body.startswith(" ") else body)
            return pad + mark + " " + body

        self._edit_blocks(tr)


# --- barre de recherche ------------------------------------------------------

class FindBar(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("findbar")
        self.editor: CodeEditor | None = None
        self.find = QLineEdit(placeholderText="Rechercher")
        self.replace = QLineEdit(placeholderText="Remplacer par")
        self.case = QToolButton(text="Aa", checkable=True, toolTip="Respecter la casse")
        self.info = QLabel(objectName="findinfo")
        prev_btn = QToolButton(text="↑", toolTip="Précédent (Maj+Entrée)")
        next_btn = QToolButton(text="↓", toolTip="Suivant (Entrée)")
        rep_btn = QPushButton("Remplacer")
        all_btn = QPushButton("Tout remplacer")
        close = QToolButton(text="✕", toolTip="Fermer (Échap)")
        for w, name in ((self.find, "Rechercher"), (self.replace, "Remplacer par"), (self.case, "Respecter la casse"),
                        (prev_btn, "Occurrence précédente"), (next_btn, "Occurrence suivante"),
                        (close, "Fermer la recherche")):
            w.setAccessibleName(name)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(theme.GUTTER, 0, theme.GUTTER, theme.SPACE)
        lay.setSpacing(theme.SPACE)
        lay.addWidget(theme.caption("chercher"))
        for w in (self.find, self.case, prev_btn, next_btn, self.replace, rep_btn, all_btn, self.info):
            lay.addWidget(w)
        lay.addStretch()
        lay.addWidget(close)
        self.find.returnPressed.connect(lambda: self.search(backward=False))
        self.find.textChanged.connect(lambda: self.search(backward=False, from_start=True))
        prev_btn.clicked.connect(lambda: self.search(backward=True))
        next_btn.clicked.connect(lambda: self.search(backward=False))
        rep_btn.clicked.connect(self.replace_one)
        all_btn.clicked.connect(self.replace_all)
        close.clicked.connect(self.close_bar)
        self.hide()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self.close_bar()
        elif event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self.search(backward=True)
        else:
            super().keyPressEvent(event)

    def open_for(self, editor: CodeEditor | None, replace: bool = False) -> None:
        self.editor = editor
        if editor and editor.textCursor().hasSelection() and " " not in editor.textCursor().selectedText():
            self.find.setText(editor.textCursor().selectedText())
        self.replace.setVisible(True)
        self.show()
        (self.replace if replace and self.find.text() else self.find).setFocus()
        self.find.selectAll()

    def close_bar(self) -> None:
        self.hide()
        if self.editor:
            self.editor.setFocus()

    def _flags(self, backward: bool) -> QTextDocument.FindFlag:
        flags = QTextDocument.FindFlag(0)
        if backward:
            flags |= QTextDocument.FindFlag.FindBackward
        if self.case.isChecked():
            flags |= QTextDocument.FindFlag.FindCaseSensitively
        return flags

    def search(self, backward: bool = False, from_start: bool = False) -> bool:
        ed, text = self.editor, self.find.text()
        if not ed or not text:
            self.info.setText("")
            return False
        if from_start:
            c = ed.textCursor()
            c.setPosition(c.selectionStart())
            ed.setTextCursor(c)
        found = ed.find(text, self._flags(backward))
        if not found:  # reboucle
            c = ed.textCursor()
            c.movePosition(QTextCursor.MoveOperation.End if backward else QTextCursor.MoveOperation.Start)
            ed.setTextCursor(c)
            found = ed.find(text, self._flags(backward))
        self.info.setText("" if found else "Introuvable")
        return found

    def _matches(self, selected: str) -> bool:
        a, b = selected, self.find.text()
        return a == b if self.case.isChecked() else a.lower() == b.lower()

    def replace_one(self) -> None:
        ed = self.editor
        if not ed:
            return
        if ed.textCursor().hasSelection() and self._matches(ed.textCursor().selectedText()):
            ed.textCursor().insertText(self.replace.text())
        self.search()

    def replace_all(self) -> None:
        ed, text = self.editor, self.find.text()
        if not ed or not text:
            return
        count = 0
        cursor = ed.textCursor()
        cursor.beginEditBlock()
        c = ed.document().find(text, 0, self._flags(False))
        while not c.isNull():
            c.insertText(self.replace.text())
            count += 1
            c = ed.document().find(text, c.position(), self._flags(False))
        cursor.endEditBlock()
        self.info.setText(f"{count} remplacement(s)")


# --- onglets -----------------------------------------------------------------

UNTITLED = "Sans titre"


class TabBar(QTabBar):
    """Onglets à croix de fermeture espacée : texte, croix et bord droit séparés du même écart.

    La croix native de Qt est collée au bord de l'onglet et ses marges QSS rétrécissent
    l'icône : on pose à la place un bouton dans un support aux marges choisies.
    """

    ICON = 12   # taille de la croix

    def tabInserted(self, index: int) -> None:  # noqa: N802
        super().tabInserted(index)
        button = QToolButton(objectName="tabclose", toolTip="Fermer (Ctrl+W)", cursor=Qt.CursorShape.PointingHandCursor)
        button.setAccessibleName("Fermer l'onglet")
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)   # Ctrl+W au clavier
        button.setFixedSize(self.ICON, self.ICON)
        holder = QWidget()
        lay = QHBoxLayout(holder)
        # croix → bord : TAB_PAD, comme la marge gauche du texte (le rectangle où Qt pose le bouton
        # inclut la marge extérieure de l'onglet, compensée par le blanc autour du tracé de la croix)
        lay.setContentsMargins(0, 0, theme.TAB_PAD + 2, 0)
        lay.addWidget(button)
        button.clicked.connect(lambda: self.tabCloseRequested.emit(self._index_of(holder)))
        self.setTabButton(index, QTabBar.ButtonPosition.RightSide, holder)

    def _index_of(self, holder: QWidget) -> int:
        return next((i for i in range(self.count()) if self.tabButton(i, QTabBar.ButtonPosition.RightSide) is holder), -1)


class FileTab(QWidget):
    """Un fichier ouvert (ou un document « Sans titre », ``path=None``) : éditeur ou aperçu d'image."""

    def __init__(self, path: Path | None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.path = path
        self.editor: CodeEditor | None = None
        self.saved_text = ""
        self.encoding, self.newline = "utf-8", "\n"   # ceux du fichier d'origine, repris à l'enregistrement
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        screen = theme.NeuCard("screen", radius=20, padding=16)
        lay.addWidget(screen)
        if path is not None and path.suffix.lower() in IMAGE_SUFFIXES:
            label = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
            pix = QPixmap(str(path))
            label.setPixmap(pix) if not pix.isNull() else label.setText("Image illisible")
            area = QScrollArea(widgetResizable=True, objectName="imageview")
            area.setWidget(label)
            screen.add(area, 1)
            return
        text = ""
        if path is not None:
            text, self.encoding, self.newline = read_text(path)
        self.editor = CodeEditor("text")
        self.editor.setPlainText(text)
        self.editor.set_language(language_of(path) if path else "text")   # coloration selon la taille du texte
        self.saved_text = self.editor.toPlainText()
        self.editor.document().setModified(False)
        screen.add(self.editor, 1)

    @property
    def title(self) -> str:
        return self.path.name if self.path else UNTITLED

    @property
    def pristine(self) -> bool:
        """Document « Sans titre » vide et intact : remplaçable sans rien perdre."""
        return self.path is None and self.editor is not None and not self.modified and not self.editor.toPlainText()

    def read(self) -> str:
        return read_text(self.path)[0]

    @property
    def modified(self) -> bool:
        return bool(self.editor and self.editor.document().isModified())

    @property
    def format_label(self) -> str:
        """Encodage et fin de ligne, quand ils diffèrent de l'usuel UTF-8 / LF."""
        parts = [self.encoding.upper()] if self.encoding != "utf-8" else []
        return " · ".join(parts + (["CRLF"] if self.newline == "\r\n" else []))

    def save(self) -> None:
        """Enregistre atomiquement, dans l'encodage et avec les fins de ligne d'origine."""
        if self.editor is None or self.path is None:
            return
        text = self.editor.toPlainText()
        out = text.replace("\n", self.newline) if self.newline != "\n" else text
        try:
            data = out.encode(self.encoding)
        except UnicodeEncodeError:   # caractère absent de l'encodage d'origine : passage en UTF-8
            self.encoding = "utf-8"
            data = out.encode(self.encoding)
        write_atomic(self.path, data)
        self.saved_text = text
        self.editor.document().setModified(False)

    def reload(self) -> None:
        if self.editor is None or self.path is None:
            return
        text, self.encoding, self.newline = read_text(self.path)
        if text == self.editor.toPlainText():
            self.editor.document().setModified(False)
            return
        cursor = self.editor.textCursor().position()
        scroll = self.editor.verticalScrollBar().value()
        self.editor.setPlainText(text)
        self.saved_text = text
        c = self.editor.textCursor()
        c.setPosition(min(cursor, len(text)))
        self.editor.setTextCursor(c)
        self.editor.verticalScrollBar().setValue(scroll)
        self.editor.document().setModified(False)


class EditorTabs(QWidget):
    """Onglets de fichiers + barre de recherche ; surveille les modifications externes."""

    saved = Signal(Path)
    current_changed = Signal()
    opened = Signal()           # un document vient d'être ouvert ou créé


    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.tabs = QTabWidget(movable=True, documentMode=True)
        self.tabs.setTabBar(TabBar())
        self.findbar = FindBar()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.tabs)
        lay.addWidget(self.findbar)
        self.tabs.tabBar().tabCloseRequested.connect(self.close_tab)   # QTabWidget ne relaie que si tabsClosable
        self.tabs.currentChanged.connect(lambda _: self.current_changed.emit())
        self.watcher = QFileSystemWatcher(self)
        self.watcher.fileChanged.connect(self._external_change)
        self._changes: set[str] = set()   # changements notifiés, en attente de traitement
        self._asking: set[Path] = set()   # question « recharger ? » déjà affichée pour ces fichiers
        self.default_dir: Callable[[], Path] = Path.cwd   # dossier proposé par « Enregistrer sous »

    # accès
    def file_tabs(self) -> list[FileTab]:
        return [w for i in range(self.tabs.count()) if isinstance(w := self.tabs.widget(i), FileTab)]

    def current(self) -> FileTab | None:
        w = self.tabs.currentWidget()
        return w if isinstance(w, FileTab) else None

    def find_tab(self, path: Path) -> FileTab | None:
        path = path.resolve()
        return next((t for t in self.file_tabs() if t.path is not None and t.path == path), None)

    # ouverture / fermeture
    def open_file(self, path: Path) -> FileTab | None:
        path = path.resolve()
        if (tab := self.find_tab(path)) is not None:
            self.tabs.setCurrentWidget(tab)
            return tab
        if not path.is_file():
            return None
        if path.stat().st_size > MAX_BYTES and path.suffix.lower() not in IMAGE_SUFFIXES:
            QMessageBox.warning(self, "Fichier trop gros", f"{path.name} dépasse {MAX_BYTES // 1024 // 1024} Mo.")
            return None
        try:
            tab = FileTab(path)
        except (OSError, ValueError) as e:
            QMessageBox.warning(self, "Ouverture impossible", f"{path.name} : {e}")
            return None
        # un « Sans titre » vide laisse sa place au fichier ouvert
        scratch = [t for t in self.file_tabs() if t.pristine]
        self._add_tab(tab)
        for t in scratch:
            self._drop(t)
        self.watcher.addPath(str(path))
        return tab

    def new_document(self) -> FileTab:
        tab = FileTab(None)
        self._add_tab(tab)
        return tab

    def ensure_document(self) -> None:
        """L'éditeur n'est jamais vide : sans onglet, on ouvre un « Sans titre »."""
        if self.tabs.count() == 0:
            self.new_document()

    def _add_tab(self, tab: FileTab) -> None:
        i = self.tabs.addTab(tab, tab.title)
        self.tabs.setCurrentIndex(i)
        self._refresh_title(tab)
        if tab.editor:
            tab.editor.modificationChanged.connect(lambda _m, t=tab: self._refresh_title(t))
            tab.editor.cursorPositionChanged.connect(self.current_changed.emit)
            tab.editor.setFocus()
        self.opened.emit()

    def _drop(self, tab: FileTab) -> None:
        if tab.path is not None:
            self.watcher.removePath(str(tab.path))
        self.tabs.removeTab(self.tabs.indexOf(tab))
        tab.deleteLater()

    def _refresh_title(self, tab: FileTab) -> None:
        i = self.tabs.indexOf(tab)
        if i >= 0:
            self.tabs.setTabText(i, ("● " if tab.modified else "") + tab.title)
            self.tabs.setTabToolTip(i, str(tab.path) if tab.path else "Pas encore enregistré (Ctrl+S pour choisir où)")
        self.current_changed.emit()

    def close_tab(self, index: int) -> bool:
        tab = self.tabs.widget(index)
        if not isinstance(tab, FileTab):
            return False
        if tab.modified:
            self.tabs.setCurrentIndex(index)
            answer = QMessageBox.question(
                self, "Modifications non enregistrées", f"Enregistrer {tab.title} avant de fermer ?",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            )
            if answer == QMessageBox.StandardButton.Cancel:
                return False
            if answer == QMessageBox.StandardButton.Save and not self.save(tab):
                return False
        self._drop(tab)
        self.ensure_document()
        return True

    def close_current(self) -> None:
        if (tab := self.current()) is not None:
            self.close_tab(self.tabs.indexOf(tab))

    def confirm_close_all(self) -> bool:
        """Avant de quitter : propose d'enregistrer les fichiers modifiés."""
        dirty = [t for t in self.file_tabs() if t.modified]
        if not dirty:
            return True
        names = "\n".join(f"• {t.title}" for t in dirty)
        answer = QMessageBox.question(
            self, "Modifications non enregistrées", f"Enregistrer les fichiers modifiés ?\n\n{names}",
            QMessageBox.StandardButton.SaveAll | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Cancel:
            return False
        if answer == QMessageBox.StandardButton.SaveAll:
            return all(self.save(t) for t in dirty)
        return True

    # enregistrement
    def save(self, tab: FileTab | None = None) -> bool:
        tab = tab or self.current()
        if tab is None or tab.editor is None:
            return False
        if tab.path is None:
            return self.save_as(tab)
        try:
            tab.save()
        except OSError as e:
            QMessageBox.warning(self, "Enregistrement impossible", f"{tab.path} : {e}")
            return False
        self._refresh_title(tab)
        self.saved.emit(tab.path)
        return True

    def save_as(self, tab: FileTab | None = None) -> bool:
        tab = tab or self.current()
        if tab is None or tab.editor is None:
            return False
        start = tab.path or (self.default_dir() / "nouveau.txt")
        name, _ = QFileDialog.getSaveFileName(self, "Enregistrer sous", str(start))
        if not name:
            return False
        old = tab.path
        if old is not None:
            self.watcher.removePath(str(old))
        tab.path = Path(name).resolve()
        tab.editor.set_language(language_of(tab.path))
        if not self.save(tab):   # échec : l'onglet reste attaché à son fichier d'origine
            tab.path = old
            tab.editor.set_language(language_of(old) if old else "text")
            if old is not None:
                self.watcher.addPath(str(old))
            return False
        self.watcher.addPath(str(tab.path))
        return True

    def save_all(self) -> None:
        for tab in self.file_tabs():
            if tab.modified:
                self.save(tab)

    # fichiers modifiés hors de l'éditeur (paramètres, Seldon, Claude Code…)
    def _external_change(self, path: str) -> None:
        # un enregistrement atomique remplace le fichier : on le resurveille un peu plus tard ;
        # nos propres enregistrements sont ignorés car le disque correspond déjà au texte.
        # Une rafale de notifications pour un même fichier n'est traitée qu'une fois.
        if path in self._changes:
            return
        self._changes.add(path)
        QTimer.singleShot(150, lambda: self._handle_change(path))

    def _handle_change(self, path: str) -> None:
        self._changes.discard(path)
        if Path(path).exists() and path not in self.watcher.files():
            self.watcher.addPath(path)
        self.reload_path(Path(path))

    def reload_path(self, path: Path) -> None:
        tab = self.find_tab(path)
        if tab is None or tab.editor is None or tab.path in self._asking:
            return
        if not tab.path.exists():
            tab.editor.document().setModified(True)   # supprimé : garder le texte comme non enregistré
            return
        try:
            on_disk = tab.read()
        except (OSError, ValueError):
            return
        if on_disk == tab.editor.toPlainText():
            return
        if tab.modified:   # jamais d'écrasement silencieux des modifications en cours
            self._asking.add(tab.path)
            try:
                answer = QMessageBox.question(
                    self, "Fichier modifié sur le disque",
                    f"{tab.path.name} a changé sur le disque mais a aussi des modifications non enregistrées.\n\n"
                    "Recharger et perdre vos modifications ?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No,
                )
            finally:
                self._asking.discard(tab.path)
            if answer != QMessageBox.StandardButton.Yes:
                return
        try:
            tab.reload()
        except (OSError, ValueError):
            return
        self._refresh_title(tab)

    def path_moved(self, old: Path, new: Path) -> None:
        """Renommage depuis l'arborescence : met à jour les onglets concernés."""
        old, new = old.resolve(), new.resolve()
        for tab in self.file_tabs():
            if tab.path is not None and (tab.path == old or old in tab.path.parents):
                self.watcher.removePath(str(tab.path))
                tab.path = new / tab.path.relative_to(old) if tab.path != old else new
                self.watcher.addPath(str(tab.path))
                self._refresh_title(tab)

    def path_deleted(self, path: Path) -> None:
        path = path.resolve()
        for tab in self.file_tabs():
            if tab.path is not None and (tab.path == path or path in tab.path.parents):
                if not tab.modified:
                    self._drop(tab)
                else:
                    tab.editor.document().setModified(True)
        self.ensure_document()

    # recherche
    def show_find(self, replace: bool = False) -> None:
        tab = self.current()
        if tab and tab.editor:
            self.findbar.open_for(tab.editor, replace)

    def make_actions(self) -> dict[str, QAction]:
        def act(text: str, shortcut, slot) -> QAction:
            a = QAction(text, self)
            a.setShortcut(QKeySequence(shortcut))
            a.triggered.connect(slot)
            return a

        return {
            "new": act("&Nouveau document", QKeySequence.StandardKey.New, self.new_document),
            "save": act("&Enregistrer", QKeySequence.StandardKey.Save, lambda: self.save()),
            "save_as": act("Enregistrer &sous…", "Ctrl+Shift+S", lambda: self.save_as()),
            "save_all": act("Enregistrer &tout", "Ctrl+Alt+S", self.save_all),
            "close": act("&Fermer l'onglet", QKeySequence.StandardKey.Close, self.close_current),
            "find": act("&Rechercher", QKeySequence.StandardKey.Find, lambda: self.show_find()),
            "replace": act("Re&mplacer", "Ctrl+H", lambda: self.show_find(True)),
        }
