"""Page Journal : lance le programme principal (``python -m seldon``) et affiche ses logs en direct."""

from __future__ import annotations

import html
import os
import signal
import sys
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, QTimer, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from seldon.gui import theme

MAX_LINES = 5000   # au-delà, les plus anciennes lignes sont oubliées
CONFIRM_TAG = "[confirmation]"   # question posée par seldon.core.pipeline.console_confirm
MIC_TAG = "[micro]"              # état du micro annoncé par seldon.stt.whisper.WhisperSTT.mute
MUTE_SIGNALS = {True: getattr(signal, "SIGUSR1", None), False: getattr(signal, "SIGUSR2", None)}   # cf. seldon.core.pipeline


def line_color(line: str) -> str | None:
    """Couleur d'une ligne de log selon son niveau (format ``date NIVEAU nom: message``)."""
    head = line[:40]
    if " ERROR " in head or " CRITICAL " in head or line.startswith("Traceback"):
        return theme.WARN
    if " WARNING " in head:
        return theme.AMBER
    if " DEBUG " in head:
        return theme.CRT_DIM
    return None


class RunnerPanel(QWidget):
    """Démarre / arrête Seldon dans un sous-processus ; stdout et stderr sont fusionnés dans le journal."""

    running_changed = Signal(bool)
    confirmation_requested = Signal(str)

    def __init__(self, config_path: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.config_path = config_path
        self.text_mode = False
        self.pending = False   # Seldon attend un oui / non sur son entrée standard
        self.ready = False     # a déjà écrit : ses signaux sont en place, le bouton micro peut servir
        self._partial = ""
        self._last_color: str | None = None
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.setWorkingDirectory(str(config_path.parent))
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")   # logs affichés au fil de l'eau, pas par paquets
        self.process.setProcessEnvironment(env)
        self.process.readyReadStandardOutput.connect(self._read)
        self.process.started.connect(self._started)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._error)

        self.command = QLabel(objectName="path")
        self.command.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.start_btn = QPushButton("▶  Lancer", objectName="primary", toolTip="Mode vocal complet (F5)")
        self.text_btn = QPushButton("▶  Mode texte", toolTip="Clavier + console (Ctrl+F5)")
        self.stop_btn = QPushButton("■  Arrêter", toolTip="Arrêter Seldon (Maj+F5)")
        self.mic_btn = QPushButton("🎙  Micro", checkable=True, toolTip="Couper / rétablir le micro (F6)")
        self.mic_btn.toggled.connect(self.mute)
        clear_btn = QPushButton("Effacer")
        self.start_btn.clicked.connect(lambda: self.start(text=False))
        self.text_btn.clicked.connect(lambda: self.start(text=True))
        self.stop_btn.clicked.connect(self.stop)
        clear_btn.clicked.connect(lambda: self.log.clear())
        top = QHBoxLayout()
        top.setSpacing(theme.SPACE)
        top.addWidget(theme.caption("commande"))
        top.addWidget(self.command, 1)
        top.addWidget(clear_btn)
        top.addWidget(self.mic_btn)
        top.addWidget(self.stop_btn)
        top.addWidget(self.text_btn)
        top.addWidget(self.start_btn)

        self.log = QPlainTextEdit(objectName="code", readOnly=True)
        self.log.setMaximumBlockCount(MAX_LINES)
        self.log.setFont(theme.mono_font(10.5))
        self.log.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.input = QLineEdit(placeholderText="Message à Seldon (mode texte) — « Nom: message » simule un locuteur")
        self.input.returnPressed.connect(self.send)
        self.yes_btn = QPushButton("Oui", objectName="primary", toolTip="Autoriser l'action")
        self.no_btn = QPushButton("Non", toolTip="Refuser l'action")
        self.yes_btn.clicked.connect(lambda: self.answer(True))
        self.no_btn.clicked.connect(lambda: self.answer(False))
        bottom_row = QWidget()
        bottom = QHBoxLayout(bottom_row)
        bottom.setContentsMargins(0, 0, 0, 0)
        bottom.setSpacing(theme.SPACE)
        bottom.addWidget(self.input, 1)
        bottom.addWidget(self.no_btn)
        bottom.addWidget(self.yes_btn)
        card = theme.NeuCard("raised")
        card.add(theme.caption("journal"))
        card.add(self.log, 1)
        card.add(bottom_row)

        head = QVBoxLayout()
        head.setContentsMargins(theme.GUTTER, theme.SPACE, theme.GUTTER, 0)
        head.addLayout(top)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addLayout(head)
        lay.addWidget(card, 1)
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
        self.pending = False
        self.ready = False
        self._set_muted(False)
        args = self.arguments(text)
        self._partial = ""
        self._note(f"$ {Path(sys.executable).name} {' '.join(args)}")
        self.process.start(sys.executable, args)
        self._update()

    def stop(self, wait: bool = False) -> None:
        """Ctrl+C (SIGINT, comme au terminal), puis arrêt forcé si Seldon ne rend pas la main en 3 s."""
        if not self.running:
            return
        pid = self.process.processId()
        if pid and hasattr(signal, "SIGINT") and os.name == "posix":
            os.kill(pid, signal.SIGINT)
        else:
            self.process.terminate()
        if wait:
            if not self.process.waitForFinished(3000):
                self.process.kill()
                self.process.waitForFinished(1000)
        else:
            QTimer.singleShot(3000, lambda: self.running and self.process.processId() == pid and self.process.kill())

    def mute(self, on: bool) -> None:
        """Coupe (``on``) ou rétablit le micro de Seldon en mode vocal."""
        sig = MUTE_SIGNALS[on]
        if not (self.can_mute and sig):
            self._set_muted(False)
            return
        os.kill(self.process.processId(), sig)
        self._update()

    @property
    def can_mute(self) -> bool:
        return self.running and self.ready and not self.text_mode and os.name == "posix"

    def _set_muted(self, on: bool) -> None:
        """Met le bouton dans l'état ``on`` sans envoyer de signal."""
        self.mic_btn.blockSignals(True)
        self.mic_btn.setChecked(on)
        self.mic_btn.blockSignals(False)
        self._update()

    def send(self) -> None:
        line = self.input.text()
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
        self._append(f"> {line}", theme.ACCENT_HI)

    def _read(self) -> None:
        data = self._partial + bytes(self.process.readAllStandardOutput()).decode("utf-8", "replace")
        *lines, self._partial = data.replace("\r\n", "\n").split("\n")
        if lines and not self.ready:
            self.ready = True
            self._update()
        for line in lines:
            # « → piste » : suite du message d'erreur précédent, même couleur
            color = self._last_color if line.startswith("  → ") else line_color(line)
            self._last_color = color
            if (at := line.find(CONFIRM_TAG)) >= 0:
                self._ask(line[at + len(CONFIRM_TAG):].removesuffix("[o/N]").strip())
                color = theme.AMBER
            if (at := line.find(MIC_TAG)) >= 0:   # état réel du micro, au cas où un signal serait perdu
                self._set_muted(line[at + len(MIC_TAG):].strip().startswith("coupé"))
            self._append(line, color)

    def _ask(self, question: str) -> None:
        self.pending = True
        self._update()
        self.input.setFocus()
        self.confirmation_requested.emit(question)

    def _started(self) -> None:
        self._update()
        self.running_changed.emit(True)

    def _finished(self, code: int, status: QProcess.ExitStatus) -> None:
        if self._partial.strip():
            self._append(self._partial, line_color(self._partial))
        self._partial = ""
        self.pending = False
        self.ready = False
        self._set_muted(False)
        crashed = status == QProcess.ExitStatus.CrashExit
        self._note("— Seldon arrêté" + (" (interrompu)" if crashed else f" (code {code})"))
        self._update()
        self.running_changed.emit(False)

    def _error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self._note(f"— impossible de lancer : {self.process.errorString()}", theme.WARN)
            self._update()

    # --- affichage ----------------------------------------------------------
    def _append(self, line: str, color: str | None = None) -> None:
        bar = self.log.verticalScrollBar()
        follow = bar.value() >= bar.maximum() - 4   # on ne suit le bas que si l'on y est déjà
        text = html.escape(line).replace(" ", "&nbsp;") or "&nbsp;"
        self.log.appendHtml(f'<span style="color:{color}">{text}</span>' if color else text)
        if follow:
            bar.setValue(bar.maximum())

    def _note(self, text: str, color: str | None = None) -> None:
        self._append(text, color or theme.ACCENT_HI)

    def _update(self) -> None:
        on = self.running
        self.start_btn.setEnabled(not on)
        self.text_btn.setEnabled(not on)
        self.stop_btn.setEnabled(on)
        self.input.setEnabled(on and (self.text_mode or self.pending))
        self.input.setPlaceholderText(
            "Confirmer ? « o » / « n », ou boutons Oui / Non" if self.pending
            else "Message à Seldon (mode texte) — « Nom: message » simule un locuteur"
        )
        self.mic_btn.setVisible(not self.text_mode)
        self.mic_btn.setEnabled(self.can_mute)
        self.mic_btn.setText("🔇  Micro coupé" if self.mic_btn.isChecked() else "🎙  Micro")
        self.yes_btn.setVisible(self.pending)
        self.no_btn.setVisible(self.pending)
        mode = "mode texte" if self.text_mode else "mode vocal"
        self.command.setText(f"python {' '.join(self.arguments(self.text_mode))}  ·  {mode if on else 'arrêté'}")
