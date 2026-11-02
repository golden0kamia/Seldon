"""Page Journal : lance le programme principal (``python -m seldon``) et affiche ses logs en direct."""

from __future__ import annotations

import codecs
import os
import re
import secrets
import signal
import sys
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPalette, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QComboBox, QFrame, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton, QSizePolicy, QVBoxLayout,
    QWidget,
)

from seldon import journal, tags
from seldon.gui import theme

MAX_LINES = 5000   # au-delà, les plus anciennes lignes sont oubliées
MAX_PARTIAL = 64 * 1024   # ligne sans fin plus longue : affichée telle quelle
FLUSH_MS = 50      # lignes ajoutées par paquets : un flot de logs ne fige pas l'interface
STOP_GRACE_MS = 3000   # délai laissé à Seldon pour s'arrêter proprement avant l'arrêt forcé
CONFIRM_TAG = tags.tag(tags.CONFIRM, "")   # balise sans jeton, telle qu'affichée (voir seldon.tags)
# Seldon tourne dans sa propre session : Arrêter atteint aussi ses sous-processus (lecteur audio,
# commandes du terminal…), et rien ne reste orphelin à la fermeture.
NEW_SESSION = os.name == "posix" and hasattr(QProcess, "setUnixProcessParameters")

# niveaux des lignes (logging) ; les lignes de l'interface elles-mêmes restent toujours visibles.
# CONVERSATION : ce qui est entendu et ce que Seldon dit, au-dessus des autres INFO.
DEBUG, INFO, CONVERSATION, WARNING, ERROR, ALWAYS = 10, 20, 25, 30, 40, 100
LEVELS = [("Tout", 0), ("Sans DEBUG", INFO), ("Conversation", CONVERSATION), ("Avertissements", WARNING),
          ("Erreurs", ERROR)]

# lignes du journal de Seldon (seldon.journal), affichées sans date ni nom de logger
_JOURNAL_LINE = re.compile(r"\d{4}-\d\d-\d\d (\d\d:\d\d:\d\d)[,.]\d+ INFO (%s): (.*)"
                           % "|".join(re.escape(n) for n in (journal.HEARD, journal.SAID, journal.THOUGHT)))


def line_level(line: str) -> int | None:
    """Niveau d'une ligne de log (format ``date NIVEAU nom: message``), ``None`` si elle n'en porte pas."""
    head = line[:40]
    if " ERROR " in head or " CRITICAL " in head or line.startswith("Traceback"):
        return ERROR
    if " WARNING " in head:
        return WARNING
    if " INFO " in head:
        return INFO
    if " DEBUG " in head:
        return DEBUG
    return None


def level_color(level: int | None) -> str | None:
    return {ERROR: theme.CRT_ERROR, WARNING: theme.AMBER, DEBUG: theme.CRT_DIM}.get(level)


def line_color(line: str) -> str | None:
    """Couleur d'une ligne de log selon son niveau."""
    return level_color(line_level(line))


def journal_line(line: str) -> tuple[str, int, str, str] | None:
    """Ligne de conversation ou de réflexion : (texte affiché, niveau, couleur, style), sinon ``None``."""
    if not (m := _JOURNAL_LINE.fullmatch(line)):
        return None
    time, name, message = m.groups()
    if name == journal.THOUGHT:
        return f"{time}  ⋯ {message}", INFO, theme.CRT_DIM, "italic"
    return f"{time}  {message}", CONVERSATION, theme.ACCENT_HI if name == journal.SAID else theme.CRT_FG, "bold"


class RunnerPanel(QWidget):
    """Démarre / arrête Seldon dans un sous-processus ; stdout et stderr sont fusionnés dans le journal."""

    running_changed = Signal(bool)
    confirmation_requested = Signal(str)
    state_changed = Signal()   # marche / arrêt, confirmation en attente…

    def __init__(self, config_path: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.config_path = config_path
        self.text_mode = False
        self.pending = False   # Seldon attend un oui / non sur son entrée standard
        self.stopping = False  # arrêt demandé, en attente de la fin du processus
        self.failed = False    # dernier lancement terminé sur une erreur
        self.relaunch = False  # redémarrage demandé : relancé dans le même mode une fois arrêté
        self.min_level = 0
        self._pid = 0
        self._token = ""       # jeton du lancement en cours : seules ses balises sont crues (seldon.tags)
        self._group = False    # le processus a sa propre session (groupe = pid)
        self._partial = ""
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._last: tuple[int, str | None, str] = (INFO, None, "")   # niveau, couleur et style de la ligne précédente
        self._buffer: list[tuple[str, str | None, int, str]] = []
        self._flush_timer = QTimer(self, singleShot=True, interval=FLUSH_MS)
        self._flush_timer.timeout.connect(self._flush)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.setWorkingDirectory(str(config_path.parent))
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")    # logs affichés au fil de l'eau, pas par paquets
        env.insert("PYTHONIOENCODING", "utf-8")   # quelle que soit la locale : décodage et réponses en UTF-8
        self.process.setProcessEnvironment(env)
        self.process.readyReadStandardOutput.connect(self._read)
        self.process.started.connect(self._started)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._error)

        self.led = theme.Led(10)
        self.command = QLabel(objectName="path")
        self.command.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.start_btn = QPushButton("▶  Lancer", objectName="primary", toolTip="Mode vocal complet (F5)")
        self.text_btn = QPushButton("▶  Mode texte", toolTip="Clavier + console (Ctrl+F5)")
        self.stop_btn = QPushButton("■  Arrêter", toolTip="Arrêter Seldon (Maj+F5) ; une seconde fois : arrêt immédiat")
        self.restart_btn = QPushButton("↻  Redémarrer", toolTip="Arrêter puis relancer Seldon dans le même mode (Ctrl+Maj+F5)")
        self.filter = QComboBox(toolTip="Niveau des messages affichés")
        self.filter.setAccessibleName("Filtre du journal")
        for text, level in LEVELS:
            self.filter.addItem(text, level)
        self.filter.currentIndexChanged.connect(lambda i: self.set_min_level(self.filter.itemData(i)))
        clear_btn = QPushButton("Effacer", toolTip="Vider le journal")
        self.start_btn.clicked.connect(lambda: self.start(text=False))
        self.text_btn.clicked.connect(lambda: self.start(text=True))
        self.stop_btn.clicked.connect(lambda: self.stop())
        self.restart_btn.clicked.connect(self.restart)
        clear_btn.clicked.connect(self.clear)
        top = QHBoxLayout()
        top.setSpacing(theme.SPACE)
        top.addWidget(self.led)
        top.addWidget(theme.caption("commande"))
        top.addWidget(self.command, 1)
        top.addWidget(self.stop_btn)
        top.addWidget(self.restart_btn)
        top.addWidget(self.text_btn)
        top.addWidget(self.start_btn)

        self.log = QPlainTextEdit(objectName="code", readOnly=True)
        self.log.setMaximumBlockCount(MAX_LINES)
        self.log.setUndoRedoEnabled(False)   # sinon chaque ligne reçue resterait dans l'historique d'annulation
        self.log.setFont(theme.mono_font(10.5))
        self.log.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.log.setFrameShape(QFrame.Shape.NoFrame)
        self.log.setPlaceholderText("Seldon est arrêté.\n\n▶ Lancer (F5) : mode vocal complet\n"
                                    "▶ Mode texte (Ctrl+F5) : clavier et console\n\nSes messages s'afficheront ici.")
        self.theme_changed()
        self.input = QLineEdit()
        self.input.setAccessibleName("Message à Seldon")
        self.input.returnPressed.connect(self.send)

        # demande de confirmation : bandeau ambre bien visible, au-dessus de la saisie
        self.question = QLabel(wordWrap=True, textFormat=Qt.TextFormat.PlainText)   # texte du sous-processus : jamais interprété en HTML
        self.question.setFont(theme.ui_font(12.5))
        self.yes_btn = QPushButton("Oui", objectName="primary", toolTip="Autoriser l'action (« o »)")
        self.no_btn = QPushButton("Non", toolTip="Refuser l'action (« n »)")
        self.yes_btn.clicked.connect(lambda: self.answer(True))
        self.no_btn.clicked.connect(lambda: self.answer(False))
        self.confirm_bar = QFrame(objectName="confirmbar")
        bar = QHBoxLayout(self.confirm_bar)
        bar.setContentsMargins(theme.PAD, theme.SPACE, theme.SPACE, theme.SPACE)
        bar.setSpacing(theme.PAD)
        bar.addWidget(theme.caption("confirmation"))
        bar.addWidget(self.question, 1)
        bar.addWidget(self.no_btn)
        bar.addWidget(self.yes_btn)

        screen = theme.NeuCard("screen", radius=20, padding=16)
        screen.add(self.log, 1)
        entry = QHBoxLayout()   # saisie, puis outils du journal
        entry.setSpacing(theme.SPACE)
        entry.addWidget(self.input, 1)
        entry.addWidget(self.filter)
        entry.addWidget(clear_btn)
        bottom = QVBoxLayout()
        bottom.setContentsMargins(theme.GUTTER, 0, theme.GUTTER, theme.SPACE)
        bottom.setSpacing(theme.SPACE)
        bottom.addWidget(self.confirm_bar)
        bottom.addLayout(entry)
        head = QVBoxLayout()
        head.setContentsMargins(theme.GUTTER, theme.SPACE, theme.GUTTER, 0)
        head.addLayout(top)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addLayout(head)
        lay.addWidget(screen, 1)
        lay.addLayout(bottom)
        self._update()

    @property
    def running(self) -> bool:
        return self.process.state() != QProcess.ProcessState.NotRunning

    def arguments(self, text: bool) -> list[str]:
        args = ["-m", "seldon", "-c", str(self.config_path)]
        return args + ["--text"] if text else args

    # --- processus ----------------------------------------------------------
    def start(self, text: bool = False) -> None:
        if self.running:
            return
        self.text_mode = text
        self.pending = self.stopping = self.failed = False
        args = self.arguments(text)
        self._partial = ""
        self._decoder.reset()
        self._last = (INFO, None, "")
        self._note(f"$ {Path(sys.executable).name} {' '.join(args)}")
        self._token = secrets.token_hex(16)
        env = self.process.processEnvironment()
        env.insert(tags.TOKEN_ENV, self._token)
        self.process.setProcessEnvironment(env)
        self._group = NEW_SESSION
        if NEW_SESSION:
            self.process.setUnixProcessParameters(QProcess.UnixProcessFlag.CreateNewSession)
        self.process.start(sys.executable, args)
        self._update()

    def stop(self, wait: bool = False) -> None:
        """SIGTERM (Seldon s'arrête proprement), puis arrêt forcé s'il ne rend pas la main en 3 s.

        Une seconde demande pendant l'arrêt force aussitôt la fin du processus (et annule un redémarrage).
        """
        self.relaunch = False
        if not self.running:
            return
        if self.stopping:
            self._kill()
            if wait:
                self.process.waitForFinished(1000)
            return
        self.stopping = True
        self._update()
        if not self._signal(getattr(signal, "SIGTERM", None)):
            self.process.terminate()
        if wait:
            if not self.process.waitForFinished(STOP_GRACE_MS):
                self._kill()
                self.process.waitForFinished(1000)
        else:
            pid = self._pid
            QTimer.singleShot(STOP_GRACE_MS, lambda: self.running and self._pid == pid and self._kill())

    def restart(self) -> None:
        """Arrêt habituel (SIGTERM, puis arrêt forcé), puis relance dans le même mode une fois le processus terminé."""
        if not self.running or self.stopping:
            return
        self.stop()
        self.relaunch = True
        self._update()

    def _signal(self, sig: int | None) -> bool:
        """Envoie ``sig`` au processus (et à son groupe) ; ``False`` si impossible."""
        if sig is None or not self._pid or os.name != "posix":
            return False
        try:
            if self._group:
                os.killpg(self._pid, sig)
            else:
                os.kill(self._pid, sig)
        except OSError:   # déjà terminé entre-temps
            return False
        return True

    def _kill(self) -> None:
        self._signal(getattr(signal, "SIGKILL", None))
        self.process.kill()

    def send(self) -> None:
        # une seule ligne : un retour à la ligne collé ne doit pas répondre d'avance à une autre question
        line = " ".join(self.input.text().splitlines())
        if not (self.running and (self.text_mode or self.pending) and line.strip()):
            return
        self._write(line)
        self.input.clear()
        if self.pending:
            self.pending = False
            self._update()

    def answer(self, yes: bool) -> None:
        """Répond à la demande de confirmation en cours."""
        if not (self.running and self.pending):
            return
        self._write("oui" if yes else "non")
        self.pending = False
        self._update()

    def _write(self, line: str) -> None:
        self.process.write((line + "\n").encode("utf-8"))
        self._push(f"> {line}", theme.ACCENT_HI, ALWAYS)

    def _read(self) -> None:
        # décodeur incrémental : un caractère UTF-8 coupé entre deux lectures n'est pas abîmé
        self.feed(self._decoder.decode(bytes(self.process.readAllStandardOutput())))

    def feed(self, text: str) -> None:
        """Traite un morceau de la sortie de Seldon : lignes complètes affichées, reste gardé."""
        data = self._partial + text
        *lines, partial = data.replace("\r\n", "\n").split("\n")
        # « \r » seul : la ligne est réécrite sur place (barres de progression) ; seul le dernier état compte
        # (un « \r » final est gardé : la suite peut être le « \n » d'un « \r\n » coupé en deux)
        if "\r" in partial[:-1]:
            partial = partial[partial.rindex("\r", 0, len(partial) - 1) + 1:]
        if len(partial) > MAX_PARTIAL:
            lines.append(partial)
            partial = ""
        self._partial = partial
        for raw in lines:
            line = next((s for s in reversed(raw.split("\r")) if s), "") if "\r" in raw else raw
            level, font = line_level(line), ""
            if level is None and line[:1].isspace() and line.strip():
                level, color, font = self._last   # suite indentée (trace, « → piste ») : même niveau, même couleur
                if font:   # suite d'une ligne du journal : alignée sous son texte, après l'heure
                    line = " " * 8 + line
            elif found := journal_line(line):
                line, level, color, font = found
            else:
                level = level or INFO
                color = level_color(level)
            self._last = (level, color, font)
            # balise avec le jeton de ce lancement seulement : une sortie qui l'imite est un simple texte
            confirm = tags.tag(tags.CONFIRM, self._token)
            if line.startswith(confirm):   # en début de ligne seulement : pas une citation dans un log
                self._ask(line[len(confirm):].removesuffix("[o/N]").strip())
                color, level, font = theme.AMBER, ALWAYS, ""
                line = CONFIRM_TAG + line[len(confirm):]   # le jeton n'est pas affiché
            self._push(line, color, level, font)

    def _ask(self, question: str) -> None:
        self.pending = True
        self.question.setText(question or "Autoriser l'action ?")
        self._update()
        self.input.setFocus()
        self.confirmation_requested.emit(question)

    def _started(self) -> None:
        self._pid = self.process.processId()
        self._update()
        self.running_changed.emit(True)

    def _finished(self, code: int, status: QProcess.ExitStatus) -> None:
        rest = self._partial + self._decoder.decode(b"", final=True)
        if rest.strip():
            self._push(rest, line_color(rest))
        self._partial = ""
        if self._group:   # sous-processus restés orphelins (le groupe survit à son meneur)
            self._signal(getattr(signal, "SIGKILL", None))
        self._pid = 0
        crashed = status == QProcess.ExitStatus.CrashExit
        if self.stopping:
            self._note("— Seldon arrêté" + (" (arrêt forcé)" if crashed else f" (code {code})"))
        elif crashed or code:
            self.failed = True
            self._note("— Seldon s'est arrêté sur une erreur " + ("(plantage)" if crashed else f"(code {code})")
                       + " : voir les messages en rouge ci-dessus", theme.CRT_ERROR)
        else:
            self._note(f"— Seldon arrêté (code {code})")
        self._flush()
        self.pending = self.stopping = False
        self._update()
        self.running_changed.emit(False)
        if self.relaunch:
            self.relaunch = False
            self.start(text=self.text_mode)

    def _error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self.failed = True
            self._note(f"— impossible de lancer : {self.process.errorString()}", theme.CRT_ERROR)
            self._flush()
            self._update()

    # --- affichage ----------------------------------------------------------
    def _push(self, line: str, color: str | None = None, level: int = INFO, font: str = "") -> None:
        """Ajoute une ligne au prochain paquet affiché ; ``font`` : "", "bold" ou "italic"."""
        self._buffer.append((line, color, level, font))
        if not self._flush_timer.isActive():
            self._flush_timer.start()

    def _flush(self) -> None:
        """Écrit d'un coup les lignes en attente ; ne suit le bas que si l'on y est déjà."""
        self._flush_timer.stop()
        if not self._buffer:
            return
        lines, self._buffer = self._buffer, []
        bar = self.log.verticalScrollBar()
        follow = bar.value() >= bar.maximum() - 4
        doc = self.log.document()
        cursor = QTextCursor(doc)
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.beginEditBlock()
        for text, color, level, font in lines:
            if not doc.isEmpty() or cursor.block().userState() >= 0:
                cursor.insertBlock()
            fmt = QTextCharFormat()
            if color:
                fmt.setForeground(QColor(color))
            if font == "bold":
                fmt.setFontWeight(QFont.Weight.Bold)
            fmt.setFontItalic(font == "italic")
            cursor.insertText(text, fmt)
            block = cursor.block()
            block.setUserState(level)
            block.setVisible(level >= self.min_level)
        cursor.endEditBlock()
        if follow:
            bar.setValue(bar.maximum())

    def set_min_level(self, level: int) -> None:
        """N'affiche que les lignes de niveau ``level`` ou plus (les lignes de l'interface restent)."""
        self.min_level = level
        doc = self.log.document()
        block = doc.firstBlock()
        while block.isValid():
            block.setVisible(block.userState() < 0 or block.userState() >= level)
            block = block.next()
        doc.markContentsDirty(0, doc.characterCount())
        self.log.viewport().update()
        bar = self.log.verticalScrollBar()
        bar.setValue(bar.maximum())

    def theme_changed(self) -> None:
        pal = self.log.palette()
        pal.setColor(QPalette.ColorRole.PlaceholderText, QColor(theme.CRT_DIM))   # lisible sur l'écran
        self.log.setPalette(pal)

    def clear(self) -> None:
        self._buffer.clear()
        self.log.clear()

    def _note(self, text: str, color: str | None = None) -> None:
        self._push(text, color or theme.ACCENT_HI, ALWAYS)

    def _update(self) -> None:
        on = self.running
        self.start_btn.setEnabled(not on)
        self.text_btn.setEnabled(not on)
        self.stop_btn.setEnabled(on)
        self.restart_btn.setEnabled(on and not self.stopping)
        self.input.setEnabled(on and (self.text_mode or self.pending))
        if self.pending:
            hint = "Répondre : « o » / « n », ou boutons Oui / Non"
        elif not on:
            hint = "Seldon est arrêté : ▶ Lancer (F5) ou ▶ Mode texte (Ctrl+F5)"
        elif self.text_mode:
            hint = "Message à Seldon (Entrée) — « Nom: message » simule un locuteur"
        else:
            hint = "Mode vocal : parlez à Seldon ; ce champ sert aux confirmations"
        self.input.setPlaceholderText(hint)
        self.yes_btn.setVisible(self.pending)
        self.no_btn.setVisible(self.pending)
        self.confirm_bar.setVisible(self.pending)
        mode = "mode texte" if self.text_mode else "mode vocal"
        if self.stopping:
            state, color = "redémarrage…" if self.relaunch else "arrêt en cours…", theme.AMBER
        elif self.pending:
            state, color = f"{mode} · confirmation demandée", theme.AMBER
        elif on:
            state, color = f"en marche · {mode}", theme.ACCENT_HI
        elif self.failed:
            state, color = "arrêté sur une erreur", theme.WARN
        else:
            state, color = "arrêté", None
        self.led.set_color(color)
        self.command.setText(f"{state}  ·  python {' '.join(self.arguments(self.text_mode))}")   # l'état d'abord : jamais rogné
        self.command.setToolTip(f"{state}\npython {' '.join(self.arguments(self.text_mode))}")
        self.state_changed.emit()
