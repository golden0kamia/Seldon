"""Fenêtre principale : quatre pages, Paramètres, Éditeur (arborescence + fichiers en onglets), Journal et Structure."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QByteArray, QSettings, Qt
from PySide6.QtGui import QAction, QActionGroup, QGuiApplication, QKeySequence, QTextCursor
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QLabel, QMainWindow, QSplitter, QStackedWidget, QVBoxLayout, QWidget,
)

from seldon.gui import theme
from seldon.gui.editor import EditorTabs, FileTab
from seldon.gui.files import FileBrowser
from seldon.gui.runner import RunnerPanel
from seldon.gui.settings import SettingsPanel
from seldon.gui.structure import StructurePanel

THEMES = [("auto", "Automatique (système)"), ("light", "Clair"), ("dark", "Sombre"),
          ("retro", "Rétro-futurisme (néon)"), ("cyber", "Cyberpunk (terminal)"), ("hud", "HUD de science-fiction")]
SETTINGS_PAGE, EDITOR_PAGE, LOGS_PAGE, STRUCTURE_PAGE = 0, 1, 2, 3


def project_roots(project: Path, doc) -> list[tuple[str, Path]]:
    """Dossiers proposés d'office : le projet Seldon, ses données, et les ``cwd`` des actions."""
    roots = [("Seldon", project)]
    data = project / "data"
    if data.is_dir():
        roots.append(("Données", data))
    for item in (doc or {}).get("actions", []) or []:
        if not hasattr(item, "get"):
            continue
        cwd = (item.get("options") or {}).get("cwd")
        if isinstance(cwd, str) and cwd:
            p = Path(cwd).expanduser()
            roots.append((f"Projets ({item.get('engine')})", p if p.is_absolute() else project / p))
    return roots


class MainWindow(QMainWindow):
    def __init__(self, config_path: Path, root: Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Seldon")
        self.store = QSettings("Seldon", "seldon-gui")
        self.config_path = config_path.resolve()
        self.project = self.config_path.parent

        self.files = FileBrowser()
        self.editors = EditorTabs()
        self.editors.default_dir = lambda: self.files.root or self.project
        self.settings = SettingsPanel(self.config_path)
        self.runner = RunnerPanel(self.config_path)
        self.structure = StructurePanel()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.files)
        splitter.addWidget(self.editors)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 1000])
        self.splitter = splitter
        self.pages = QStackedWidget()
        self.pages.addWidget(self.settings)   # SETTINGS_PAGE
        self.pages.addWidget(splitter)        # EDITOR_PAGE
        self.pages.addWidget(self.runner)     # LOGS_PAGE
        self.pages.addWidget(self.structure)  # STRUCTURE_PAGE
        self.masthead = theme.Masthead()
        central = QWidget(objectName="central")
        lay = QVBoxLayout(central)
        lay.setContentsMargins(theme.SPACE, 0, theme.SPACE, 0)
        lay.setSpacing(0)
        lay.addWidget(self.masthead)
        lay.addWidget(self.pages, 1)
        self.setCentralWidget(central)

        self.position = QLabel()
        self.statusBar().addPermanentWidget(self.position)

        self.masthead.mode_changed.connect(self.show_page)
        self.files.open_requested.connect(self.open_path)
        self.files.moved.connect(self.editors.path_moved)
        self.files.deleted.connect(self.editors.path_deleted)
        self.files.root_changed.connect(lambda p: self.statusBar().showMessage(str(p), 3000))
        self.settings.open_requested.connect(self.open_path)
        self.structure.open_requested.connect(self.open_location)
        self.settings.saved.connect(self._settings_saved)
        self.settings.dirty_changed.connect(lambda _d: self._update_status())
        self.settings.reloaded.connect(self._update_status)
        self.editors.opened.connect(lambda: self.show_page(EDITOR_PAGE))
        self.editors.current_changed.connect(self._update_status)
        self.editors.saved.connect(lambda p: self.statusBar().showMessage(f"Enregistré : {p}", 3000))
        self.runner.running_changed.connect(self._running_changed)
        self.runner.state_changed.connect(self._runner_state)
        self.runner.confirmation_requested.connect(self._confirmation_requested)

        self._build_menus()
        self._restore(root)

    # --- menus --------------------------------------------------------------
    def _build_menus(self) -> None:
        ed = self.editors.make_actions()
        ed["save"].triggered.disconnect()
        ed["save"].triggered.connect(self.save_current)

        def act(text: str, slot, shortcut=None) -> QAction:
            a = QAction(text, self)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            a.triggered.connect(slot)
            return a

        file_menu = self.menuBar().addMenu("&Fichier")
        file_menu.addAction(ed["new"])
        file_menu.addAction(act("&Ouvrir un fichier…", self.choose_file, QKeySequence.StandardKey.Open))
        file_menu.addAction(act("Ouvrir un &dossier…", self.files.choose_root, "Ctrl+Shift+O"))
        file_menu.addSeparator()
        for key in ("save", "save_as", "save_all", "close"):
            file_menu.addAction(ed[key])
        file_menu.addSeparator()
        file_menu.addAction(act("&Quitter", self.close, QKeySequence.StandardKey.Quit))

        edit_menu = self.menuBar().addMenu("&Édition")
        edit_menu.addAction(ed["find"])
        edit_menu.addAction(ed["replace"])
        edit_menu.addAction(act("Commenter / décommenter", self._toggle_comment, "Ctrl+/"))

        run = self.menuBar().addMenu("&Seldon")
        self.run_actions = {
            "start": act("▶ &Lancer (mode vocal)", lambda: self.launch(text=False), "F5"),
            "text": act("▶ Lancer en mode &texte", lambda: self.launch(text=True), "Ctrl+F5"),
            "stop": act("■ &Arrêter", lambda: self.runner.stop(), "Shift+F5"),
            "restart": act("↻ &Redémarrer (même mode)", self.runner.restart, "Ctrl+Shift+F5"),
            "mute": act("🔇 &Couper / rétablir le micro", self.runner.mic_btn.toggle, "F6"),
        }
        for a in self.run_actions.values():
            run.addAction(a)
        run.addSeparator()
        run.addAction(act("▤ &Journal en direct", lambda: self.show_page(LOGS_PAGE), "Ctrl+3"))
        run.addAction(act("&Effacer le journal", self.runner.clear))
        self._running_changed(False)

        view = self.menuBar().addMenu("&Affichage")
        view.addAction(act("⚙ Paramètres", lambda: self.show_page(SETTINGS_PAGE), "Ctrl+1"))
        view.addAction(act("✎ Éditeur", lambda: self.show_page(EDITOR_PAGE), "Ctrl+2"))
        view.addAction(act("▤ Journal", lambda: self.show_page(LOGS_PAGE), "Ctrl+3"))
        view.addAction(act("◈ Structure du code", lambda: self.show_page(STRUCTURE_PAGE), "Ctrl+4"))
        view.addSeparator()
        self.show_hidden = act("Afficher les fichiers cachés", self.files.set_show_hidden)
        self.show_hidden.setCheckable(True)
        view.addAction(self.show_hidden)
        toggle_tree = act("Panneau des fichiers", lambda on: self.files.setVisible(on), "Ctrl+B")
        toggle_tree.setCheckable(True)
        toggle_tree.setChecked(True)
        view.addAction(toggle_tree)
        view.addAction(act("Afficher le fichier courant dans l'arborescence", self._reveal_current))
        view.addSeparator()
        themes = view.addMenu("&Thème")
        group = QActionGroup(self)
        self.theme_actions: dict[str, QAction] = {}
        for key, label in THEMES:
            a = act(label, lambda _=False, k=key: self.set_theme(k))
            a.setCheckable(True)
            group.addAction(a)
            themes.addAction(a)
            self.theme_actions[key] = a
        view.addAction(act("Basculer clair / sombre", self.toggle_theme, "Ctrl+Shift+L"))
        self.masthead.toggle_variant.connect(self.toggle_theme)
        self.masthead.variant_btn.setMenu(themes)
        hints = QGuiApplication.styleHints()
        if hasattr(hints, "colorSchemeChanged"):
            hints.colorSchemeChanged.connect(lambda _s: self.theme_mode == "auto" and self.set_theme("auto"))

    # --- pages --------------------------------------------------------------
    def show_page(self, index: int) -> None:
        self.pages.setCurrentIndex(index)
        self.masthead.set_mode(index)
        # focus sur l'élément principal de la page (sinon le premier bouton, liseré compris)
        if index == EDITOR_PAGE:
            self.editors.ensure_document()
            if (tab := self.editors.current()) and tab.editor:
                tab.editor.setFocus()
        elif index == LOGS_PAGE:
            (self.runner.input if self.runner.input.isEnabled() else self.runner.log).setFocus()
        elif index == STRUCTURE_PAGE:
            self.structure.ensure_loaded()   # analyse du code au premier affichage seulement
            self.structure.search.setFocus()
        else:
            self.settings.nav.setFocus()
        self._update_status()

    # --- programme principal -------------------------------------------------
    def launch(self, text: bool = False) -> None:
        """Lance Seldon sur ``config.yaml`` ; des paramètres non enregistrés sont proposés à l'enregistrement."""
        if self.runner.running or not self.settings.confirm_close():
            return
        self.show_page(LOGS_PAGE)
        self.runner.start(text=text)

    def _running_changed(self, running: bool) -> None:
        self.run_actions["start"].setEnabled(not running)
        self.run_actions["text"].setEnabled(not running)
        self.run_actions["stop"].setEnabled(running)
        self._runner_state()

    def _runner_state(self) -> None:
        if hasattr(self, "run_actions"):
            self.run_actions["mute"].setEnabled(self.runner.can_mute)   # mode vocal, une fois Seldon prêt
            self.run_actions["restart"].setEnabled(self.runner.restart_btn.isEnabled())
        self._update_status()

    def _confirmation_requested(self, question: str) -> None:
        """Seldon attend un oui / non : on affiche le Journal et on signale la fenêtre."""
        self.show_page(LOGS_PAGE)
        self.statusBar().showMessage(f"Confirmation demandée : {question}", 10000)
        QApplication.alert(self)

    # --- thème ----------------------------------------------------------------
    def set_theme(self, mode: str) -> None:
        """``auto`` suit le système ; les autres clés de ``THEMES`` forcent la variante."""
        self.theme_mode = mode if mode in dict(THEMES) else "auto"
        self.theme_actions[self.theme_mode].setChecked(True)
        app = QApplication.instance()
        if app is not None:
            theme.apply(app, self.theme_mode)
        self.store.setValue("theme", self.theme_mode)

    def toggle_theme(self) -> None:
        self.set_theme("light" if theme.is_dark() else "dark")

    def _toggle_comment(self) -> None:
        tab = self.editors.current()
        if tab and tab.editor:
            tab.editor.toggle_comment()

    def _reveal_current(self) -> None:
        tab = self.editors.current()
        if tab and tab.path:
            self.show_page(EDITOR_PAGE)
            if self.files.root is None or self.files.root not in tab.path.parents:
                self.files.set_root(tab.path.parent)
            self.files.reveal(tab.path)

    # --- actions ------------------------------------------------------------
    def open_path(self, path: Path) -> None:
        self.editors.open_file(path)
        self.show_page(EDITOR_PAGE)

    def open_location(self, path: Path, line: int) -> None:
        """Ouvre ``path`` dans l'éditeur, curseur au début de la ligne ``line`` (1 = première)."""
        tab = self.editors.open_file(path)
        self.show_page(EDITOR_PAGE)
        if tab and tab.editor:
            block = tab.editor.document().findBlockByNumber(max(line - 1, 0))
            cursor = QTextCursor(block)
            tab.editor.setTextCursor(cursor)
            tab.editor.centerCursor()

    def choose_file(self) -> None:
        start = str(self.files.root or self.project)
        paths, _ = QFileDialog.getOpenFileNames(self, "Ouvrir des fichiers", start)
        for p in paths:
            self.open_path(Path(p))

    def save_current(self) -> None:
        if self.pages.currentIndex() == SETTINGS_PAGE:
            self.settings.save()
        else:
            self.editors.save()

    def _settings_saved(self, path: Path) -> None:
        self.statusBar().showMessage(f"Paramètres enregistrés : {path}", 3000)
        self.editors.reload_path(path)
        self.files.set_roots(project_roots(self.project, self.settings.doc), self.files.recent)

    def _update_status(self) -> None:
        pending =self.settings.dirty + sum(t.modified for t in self.editors.file_tabs())
        if self.runner.pending:
            self.masthead.set_state("confirmation demandée", theme.AMBER)
        elif pending:
            self.masthead.set_state(f"{pending} modif. non enregistrée(s)", theme.AMBER)
        elif self.runner.running:
            self.masthead.set_state("seldon en marche", theme.ACCENT_HI)
        elif self.runner.failed:
            self.masthead.set_state("arrêt sur erreur", theme.WARN)
        elif self.settings.doc is None:
            self.masthead.set_state("config. illisible", theme.WARN)
        else:
            self.masthead.set_state("prêt", theme.ACCENT_HI)
        tab = self.editors.current()
        if self.pages.currentIndex() == EDITOR_PAGE and isinstance(tab, FileTab) and tab.editor:
            c = tab.editor.textCursor()
            fmt = f"  ·  {tab.format_label}" if tab.format_label else ""   # ex. LATIN-1 · CRLF
            self.position.setText(f"{tab.editor.language.upper()}{fmt}  ·  LN {c.blockNumber() + 1:>4}  COL {c.positionInBlock() + 1:>3}")
            self.setWindowTitle(f"{tab.title} — Seldon")
        else:
            self.position.setText("")
            titles = {SETTINGS_PAGE: "Paramètres — Seldon", LOGS_PAGE: "Journal — Seldon",
                      STRUCTURE_PAGE: "Structure — Seldon"}
            self.setWindowTitle(titles.get(self.pages.currentIndex(), "Seldon"))

    # --- persistance --------------------------------------------------------
    def _restore(self, root: Path | None) -> None:
        geometry = self.store.value("geometry")
        if isinstance(geometry, QByteArray):
            self.restoreGeometry(geometry)
        else:
            self.resize(1400, 850)
        splitter = self.store.value("splitter")
        if isinstance(splitter, QByteArray):
            self.splitter.restoreState(splitter)
        self.theme_mode = str(self.store.value("theme", "auto"))
        self.theme_actions.get(self.theme_mode, self.theme_actions["auto"]).setChecked(True)
        hidden = self.store.value("show_hidden", False, type=bool)
        self.show_hidden.setChecked(hidden)
        self.files.set_show_hidden(hidden)
        recent = [Path(p) for p in (self.store.value("recent_roots") or []) if p]
        self.files.set_roots(project_roots(self.project, self.settings.doc), recent)
        last = self.store.value("root")
        self.files.set_root(root or (Path(last) if last and Path(last).is_dir() else self.project))
        for p in self.store.value("open_files") or []:
            if p and Path(p).is_file():
                self.editors.open_file(Path(p))
        self.editors.ensure_document()
        self.show_page(int(self.store.value("page", SETTINGS_PAGE, type=int)) % self.pages.count())

    def closeEvent(self, event) -> None:  # noqa: N802
        if not self.settings.confirm_close() or not self.editors.confirm_close_all():
            event.ignore()
            return
        self.runner.stop(wait=True)
        self.store.setValue("geometry", self.saveGeometry())
        self.store.setValue("splitter", self.splitter.saveState())
        self.store.setValue("show_hidden", self.show_hidden.isChecked())
        self.store.setValue("root", str(self.files.root or ""))
        self.store.setValue("recent_roots", [str(p) for p in self.files.recent])
        self.store.setValue("open_files", [str(t.path) for t in self.editors.file_tabs() if t.path])
        self.store.setValue("page", self.pages.currentIndex())
        event.accept()
