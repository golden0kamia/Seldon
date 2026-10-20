"""Arborescence de fichiers : choix du dossier racine, création, renommage, corbeille."""

from __future__ import annotations

import shutil
from pathlib import Path

from PySide6.QtCore import QDir, QFile, QModelIndex, QSortFilterProxyModel, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QGuiApplication
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QFileSystemModel, QHBoxLayout, QInputDialog, QMenu, QMessageBox,
    QToolButton, QTreeView, QVBoxLayout, QWidget,
)

from seldon.gui import theme

HIDDEN = {"__pycache__", ".git", ".venv", "venv", ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules"}


class HideFilter(QSortFilterProxyModel):
    """Masque les dossiers techniques (cache, venv, .git…) et les fichiers cachés."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.show_all = False

    def filterAcceptsRow(self, row: int, parent: QModelIndex) -> bool:  # noqa: N802
        if self.show_all:
            return True
        name = self.sourceModel().index(row, 0, parent).data()
        if not name:
            return True
        return name not in HIDDEN and not name.endswith(".egg-info") and not (name.startswith(".") and name not in (".gitignore", ".env"))


class FileBrowser(QWidget):
    open_requested = Signal(Path)
    moved = Signal(Path, Path)
    deleted = Signal(Path)
    root_changed = Signal(Path)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.model = QFileSystemModel(self)
        self.model.setFilter(QDir.Filter.AllEntries | QDir.Filter.NoDotAndDotDot | QDir.Filter.Hidden)
        self.proxy = HideFilter(self)
        self.proxy.setSourceModel(self.model)
        self.proxy.setSortCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)

        self.roots = QComboBox(toolTip="Dossier affiché (projets récents)")
        self.roots.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.roots.setMinimumContentsLength(12)
        self.roots.activated.connect(lambda i: self.set_root(Path(self.roots.itemData(i))))
        browse = QToolButton(text="…", toolTip="Ouvrir un autre dossier")
        browse.clicked.connect(self.choose_root)
        up = QToolButton(text="↑", toolTip="Dossier parent")
        up.clicked.connect(lambda: self.root and self.set_root(self.root.parent))

        self.tree = QTreeView(headerHidden=True, uniformRowHeights=True)
        self.tree.setModel(self.proxy)
        for col in (1, 2, 3):
            self.tree.hideColumn(col)
        self.tree.setSortingEnabled(True)
        self.tree.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._menu)
        self.tree.activated.connect(self._activated)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(theme.SPACE)
        top.addWidget(self.roots, 1)
        top.addWidget(up)
        top.addWidget(browse)
        card = theme.NeuCard("raised")
        card.add(theme.caption("fichiers"))
        card.body.addLayout(top)
        well = card.add(theme.NeuCard("inset", radius=14, padding=4), 1)
        well.add(self.tree, 1)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(card)
        self.root: Path | None = None
        self.fixed_roots: list[tuple[str, Path]] = []
        self.recent: list[Path] = []

    # --- racine ---------------------------------------------------------------
    def set_roots(self, fixed: list[tuple[str, Path]], recent: list[Path]) -> None:
        self.fixed_roots = [(label, p.resolve()) for label, p in fixed]
        self.recent = [p for p in recent if p.is_dir()]
        self._fill_roots()

    def _fill_roots(self) -> None:
        self.roots.clear()
        seen = set()
        for label, p in self.fixed_roots:
            if p.is_dir() and p not in seen:
                self.roots.addItem(f"{label} — {p.name}", str(p))
                self.roots.setItemData(self.roots.count() - 1, str(p), Qt.ItemDataRole.ToolTipRole)
                seen.add(p)
        if self.recent and any(p not in seen for p in self.recent):
            self.roots.insertSeparator(self.roots.count())
        for p in self.recent:
            if p not in seen:
                self.roots.addItem(str(p).replace(str(Path.home()), "~", 1), str(p))
                seen.add(p)
        if self.root is not None:
            i = self.roots.findData(str(self.root))
            if i < 0:
                self.roots.insertItem(0, str(self.root).replace(str(Path.home()), "~", 1), str(self.root))
                i = 0
            self.roots.setCurrentIndex(i)

    def set_root(self, path: Path) -> None:
        path = path.expanduser().resolve()
        if not path.is_dir():
            return
        self.root = path
        src = self.model.setRootPath(str(path))
        self.tree.setRootIndex(self.proxy.mapFromSource(src))
        if path not in [p for _, p in self.fixed_roots]:
            self.recent = [path] + [p for p in self.recent if p != path][:9]
        self._fill_roots()
        self.root_changed.emit(path)

    def choose_root(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Ouvrir un dossier", str(self.root or Path.home()))
        if d:
            self.set_root(Path(d))

    def set_show_hidden(self, show: bool) -> None:
        if hasattr(self.proxy, "beginFilterChange"):   # Qt ≥ 6.9
            self.proxy.beginFilterChange()
            self.proxy.show_all = show
            self.proxy.endFilterChange()
        else:
            self.proxy.show_all = show
            self.proxy.invalidateFilter()

    def reveal(self, path: Path) -> None:
        idx = self.proxy.mapFromSource(self.model.index(str(path)))
        if idx.isValid():
            self.tree.setCurrentIndex(idx)
            self.tree.scrollTo(idx)

    # --- actions sur les fichiers --------------------------------------------
    def path_at(self, index: QModelIndex) -> Path | None:
        if not index.isValid():
            return None
        return Path(self.model.filePath(self.proxy.mapToSource(index)))

    def _activated(self, index: QModelIndex) -> None:
        path = self.path_at(index)
        if path and path.is_file():
            self.open_requested.emit(path)

    def _menu(self, pos) -> None:
        path = self.path_at(self.tree.indexAt(pos))
        folder = path if path and path.is_dir() else (path.parent if path else self.root)
        if folder is None:
            return
        menu = QMenu(self)
        if path and path.is_file():
            menu.addAction("Ouvrir", lambda: self.open_requested.emit(path))
        menu.addAction("Nouveau fichier…", lambda: self.new_file(folder))
        menu.addAction("Nouveau dossier…", lambda: self.new_folder(folder))
        if path:
            menu.addSeparator()
            menu.addAction("Renommer…", lambda: self.rename(path))
            menu.addAction("Mettre à la corbeille", lambda: self.delete(path))
            menu.addSeparator()
            menu.addAction("Copier le chemin", lambda: QGuiApplication.clipboard().setText(str(path)))
            if path.is_dir():
                menu.addAction("Afficher ce dossier ici", lambda: self.set_root(path))
        menu.addAction("Ouvrir dans le gestionnaire de fichiers",
                       lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder))))
        menu.exec(self.tree.viewport().mapToGlobal(pos))

    def _ask_name(self, title: str, label: str, text: str = "") -> str | None:
        name, ok = QInputDialog.getText(self, title, label, text=text)
        name = name.strip()
        if not ok or not name:
            return None
        if "/" in name and title == "Renommer":
            QMessageBox.warning(self, title, "Le nom ne doit pas contenir « / ».")
            return None
        return name

    def new_file(self, folder: Path) -> None:
        name = self._ask_name("Nouveau fichier", f"Nom du fichier dans {folder.name}/ :")
        if not name:
            return
        path = folder / name
        if path.exists():
            QMessageBox.warning(self, "Nouveau fichier", f"{name} existe déjà.")
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        except OSError as e:
            QMessageBox.warning(self, "Nouveau fichier", str(e))
            return
        self.open_requested.emit(path)

    def new_folder(self, folder: Path) -> None:
        name = self._ask_name("Nouveau dossier", f"Nom du dossier dans {folder.name}/ :")
        if not name:
            return
        try:
            (folder / name).mkdir(parents=True)
        except OSError as e:
            QMessageBox.warning(self, "Nouveau dossier", str(e))

    def rename(self, path: Path) -> None:
        name = self._ask_name("Renommer", "Nouveau nom :", path.name)
        if not name or name == path.name:
            return
        target = path.with_name(name)
        if target.exists():
            QMessageBox.warning(self, "Renommer", f"{name} existe déjà.")
            return
        try:
            path.rename(target)
        except OSError as e:
            QMessageBox.warning(self, "Renommer", str(e))
            return
        self.moved.emit(path, target)

    def delete(self, path: Path) -> None:
        kind = "le dossier" if path.is_dir() else "le fichier"
        if QMessageBox.question(self, "Corbeille", f"Mettre {kind} « {path.name} » à la corbeille ?") \
                != QMessageBox.StandardButton.Yes:
            return
        result = QFile.moveToTrash(str(path))   # bool, ou (bool, chemin) selon la version de PySide6
        ok = result[0] if isinstance(result, tuple) else bool(result)
        if not ok and path.exists():
            if QMessageBox.question(self, "Corbeille indisponible",
                                    f"Impossible d'utiliser la corbeille. Supprimer définitivement « {path.name} » ?") \
                    != QMessageBox.StandardButton.Yes:
                return
            try:
                shutil.rmtree(path) if path.is_dir() else path.unlink()
            except OSError as e:
                QMessageBox.warning(self, "Suppression", str(e))
                return
        self.deleted.emit(path)
