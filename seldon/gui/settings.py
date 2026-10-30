"""Panneau de paramètres : formulaire généré depuis config.yaml et les signatures des plugins.

Chaque champ modifie directement le document ``ruamel`` (commentaires conservés) ;
« Enregistrer » le réécrit. Les moteurs proposés et leurs options viennent de
``plugins.discover()`` : un nouveau plugin apparaît sans toucher à l'interface.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QFileSystemWatcher, QRect, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QMessageBox, QPushButton, QScrollArea, QSizePolicy, QStackedWidget,
    QToolButton, QVBoxLayout, QWidget,
)
from ruamel.yaml.comments import CommentedMap, CommentedSeq

from seldon.config import Config
from seldon.skills import base as skills
from seldon.core import registry
from seldon.core.errors import describe
from seldon.gui import configdoc, theme
from seldon.gui.brainform import BackupEditor, BrainEditor
from seldon.gui.fsutil import write_atomic
from seldon.gui.plugins import Param, PluginInfo, choices_from_comment, class_params, discover
from seldon.gui.records import ConversationsView, RecordsView, SpeakersView
from seldon.gui.widgets import FIXED_CHOICES, guard_wheel, hint_label, make_editor  # noqa: F401 (réexportés)

PACKAGE = Path(__file__).resolve().parent.parent
SKILL_NAME = re.compile(r"[a-z0-9][a-z0-9_-]*")
SKILL_TEMPLATE = """---
name: {name}
description: À compléter — quand le cerveau doit utiliser ce skill (« Quand on demande… »).
---
Consignes à suivre, étape par étape, avec les tools et serveurs MCP activés.
"""

SECTIONS = [  # (clé, titre, catégorie de plugins)
    ("stt", "Écoute (STT)", "stt"),
    ("speakers", "Locuteurs", "speakers"),
    ("brain", "Cerveau", "brain"),
    ("tts", "Voix (TTS)", "tts"),
]

class OptionsForm(QWidget):
    """Formulaire clé → valeur sur un ``CommentedMap`` (modifié en place)."""

    WARNING = "⚠"

    def __init__(self, mapping: CommentedMap, known: dict[str, Param], on_change: Callable[[], None],
                 keys: Callable[[str], bool] | None = None, extensible: bool = True) -> None:
        super().__init__()
        self.mapping, self.known, self.on_change = mapping, known, on_change
        self.keys = keys or (lambda _k: True)
        self.extensible = extensible
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(theme.PAD)
        self.grid.setVerticalSpacing(theme.SPACE)
        self.grid.setColumnStretch(1, 3)
        self.grid.setColumnStretch(2, 4)
        self.rebuild()

    def set_known(self, known: dict[str, Param]) -> None:
        self.known = known
        self.rebuild()

    def rebuild(self) -> None:
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        row = 0
        for key in list(self.mapping):
            if not self.keys(str(key)) or isinstance(self.mapping[key], CommentedMap) and not self.extensible:
                continue
            self._add_row(row, str(key))
            row += 1
        if row == 0:
            self.grid.addWidget(hint_label("Aucune option : les valeurs par défaut du moteur s'appliquent."), row, 0, 1, 3)

    def _set(self, key: str, value: Any) -> None:
        self.mapping[key] = value
        self.on_change()

    def _add_row(self, row: int, key: str) -> None:
        value = self.mapping[key]
        param = self.known.get(key)
        comment = configdoc.comment(self.mapping, key)
        hint = comment or (param.comment if param else "")
        choices = FIXED_CHOICES.get(key) or choices_from_comment(comment) or (choices_from_comment(param.comment) if param else [])
        if isinstance(value, str) and choices and value not in choices and not (param and param.default in choices):
            choices = []   # le commentaire ne décrivait pas des valeurs possibles
        label = QLabel(key)
        label.setFont(theme.mono_font(10))
        if self.extensible:   # place du « ⚠ » réservée : changer de moteur ne décale pas les champs
            label.setMinimumWidth(label.fontMetrics().horizontalAdvance(f"{key} {self.WARNING}"))
        tips = [param.hint()] if param else []
        if self.known and param is None and self.extensible:
            label.setText(f"{key} {self.WARNING}")
            tips.append("Option inconnue pour ce moteur (ignorée ou erreur au démarrage).")
        label.setToolTip("\n".join(t for t in tips if t))
        editor = make_editor(value, choices, lambda v, k=key: self._set(k, v))
        editor.setToolTip(label.toolTip() or editor.toolTip())
        self.grid.addWidget(label, row, 0)
        self.grid.addWidget(editor, row, 1)
        self.grid.addWidget(hint_label(hint), row, 2)


def _engine_combo(plugins: dict[str, PluginInfo], current: str) -> QComboBox:
    combo = guard_wheel(QComboBox(editable=True))
    combo.setAccessibleName("Moteur")
    names = sorted(plugins)
    combo.addItems(names if current in names else [current, *names])
    for i in range(combo.count()):
        info = plugins.get(combo.itemText(i))
        if info:
            combo.setItemData(i, info.doc, Qt.ItemDataRole.ToolTipRole)
    combo.setCurrentText(current)
    combo.setToolTip("Nom du plugin (ou « module:Classe » pour un plugin externe)")
    return combo


class StageEditor(QWidget):
    """Moteur + options d'un étage (``{engine, options}``).

    ``choose=False`` : moteur fixe, sans sélecteur (tools et serveurs MCP, où le
    moteur est l'élément lui-même).
    """

    def __init__(self, node: CommentedMap, plugins: dict[str, PluginInfo], on_change: Callable[[], None],
                 reserve: bool = True, choose: bool = True) -> None:
        super().__init__()
        self.node, self.plugins, self.on_change = node, plugins, on_change
        self.reserve = reserve   # hauteur de la plus longue description : les champs restent en place
        engine = str(node.get("engine", ""))
        self.combo = _engine_combo(plugins, engine)
        self.doc = hint_label("")
        self.form = OptionsForm(node["options"], self._known(engine), on_change)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(theme.SPACE)
        if choose:
            head = QHBoxLayout()
            head.setSpacing(theme.PAD)
            head.addWidget(theme.caption("moteur"))
            head.addWidget(self.combo, 1)
            lay.addLayout(head)
        else:
            self.combo.hide()
        lay.addWidget(self.doc)
        lay.addWidget(self.form)
        self._show_doc(engine)
        self.combo.currentTextChanged.connect(self._engine_changed)

    def _known(self, engine: str) -> dict[str, Param]:
        info = self.plugins.get(engine)
        return info.params if info else {}

    def _show_doc(self, engine: str) -> None:
        info = self.plugins.get(engine)
        text = info.doc if info else ("Plugin externe" if ":" in engine else "Plugin inconnu")
        self.doc.setText(self._doc_text(text))
        if not self.reserve:
            self.doc.setVisible(bool(self.doc.text()))   # pas de ligne vide (et d'écart en trop) sans description

    def _doc_text(self, text: str) -> str:
        comment = configdoc.comment(self.node, "engine")
        return text + (f"\nNote : {comment}" if comment else "")

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self.reserve:
            fm, width = self.doc.fontMetrics(), max(self.doc.width(), 1)
            heights = [fm.boundingRect(QRect(0, 0, width, 10_000), Qt.TextFlag.TextWordWrap, self._doc_text(info.doc)).height()
                       for info in self.plugins.values()]
            self.doc.setMinimumHeight(max(heights, default=0))

    def _engine_changed(self, engine: str) -> None:
        engine = engine.strip()
        if not engine:
            return
        self.node["engine"] = engine
        self._show_doc(engine)
        self.form.set_known(self._known(engine))
        self.on_change()


class ActionsPage(QWidget):
    """Liste de tools ou de serveurs MCP (``{engine, options}``), avec leurs options et un bouton pour éditer leur code."""

    open_requested = Signal(Path)

    def __init__(self, actions: CommentedSeq, plugins: dict[str, PluginInfo], on_change: Callable[[], None],
                 hint: str = "", noun: str = "l'action") -> None:
        super().__init__()
        self.actions, self.plugins, self.on_change = actions, plugins, on_change
        self.hint, self.noun = hint, noun
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self.lay.setSpacing(theme.SPACE)
        self.rebuild()

    def rebuild(self) -> None:
        while self.lay.count():
            item = self.lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                while item.layout().count():
                    sub = item.layout().takeAt(0)
                    if sub.widget():
                        sub.widget().deleteLater()
        self.lay.addWidget(hint_label(self.hint))
        for i in range(len(self.actions)):
            node = configdoc.action_entry(self.actions, i)
            box = theme.NeuCard("inset", radius=16)
            v = box.body
            tools = QHBoxLayout()
            tools.setSpacing(theme.SPACE)
            title = QLabel(str(node.get("engine", "")))
            title.setFont(theme.label_font(10.5))
            tools.addWidget(theme.Led(10))
            tools.addWidget(title)
            if note := configdoc.comment(self.actions, i):
                tools.addWidget(hint_label(note), 1)
            else:
                tools.addStretch(1)
            for text, tip, slot in (("↑", "Monter", lambda _=False, i=i: self._move(i, -1)),
                                    ("↓", "Descendre", lambda _=False, i=i: self._move(i, 1)),
                                    ("✕", f"Retirer {self.noun}", lambda _=False, i=i: self._remove(i))):
                b = QToolButton(text=text, toolTip=tip, autoRaise=True)
                b.setAccessibleName(f"{tip} : {node.get('engine', '')}")
                b.clicked.connect(slot)
                tools.addWidget(b)
            info = self.plugins.get(str(node.get("engine", "")))
            file = info.path if info else None
            edit = QToolButton(text="✎", toolTip=f"Ouvrir {file.name}" if file else "Code introuvable",
                               autoRaise=True, enabled=bool(file and file.is_file()))
            edit.setAccessibleName(f"Ouvrir le code : {node.get('engine', '')}")
            edit.clicked.connect(lambda _=False, f=file: self.open_requested.emit(f))
            tools.insertWidget(tools.count() - 1, edit)   # avant « Retirer », comme pour les skills
            v.addLayout(tools)
            v.addWidget(StageEditor(node, self.plugins, self.on_change, reserve=False, choose=False))
            self.lay.addWidget(box)
        used = {str(self.actions[i].get("engine")) for i in range(len(self.actions))}
        add_box = guard_wheel(QComboBox(editable=True))
        add_box.setAccessibleName(f"Ajouter {self.noun}")
        add_box.addItems([n for n in sorted(self.plugins) if n not in used])
        for i in range(add_box.count()):
            add_box.setItemData(i, self.plugins[add_box.itemText(i)].doc, Qt.ItemDataRole.ToolTipRole)
        add = QPushButton(f"Ajouter {self.noun}")
        add.clicked.connect(lambda: self._add(add_box.currentText().strip()))
        line = QHBoxLayout()
        line.setSpacing(theme.SPACE)
        line.addWidget(add_box, 1)
        line.addWidget(add)
        line.addStretch(1)
        self.lay.addLayout(line)
        self.lay.addStretch(1)

    def _add(self, name: str) -> None:
        if not name:
            return
        self.actions.append(name)
        self.on_change()
        self.rebuild()

    def _remove(self, i: int) -> None:
        del self.actions[i]
        self.on_change()
        self.rebuild()

    def _move(self, i: int, delta: int) -> None:
        j = i + delta
        if 0 <= j < len(self.actions):
            self.actions[i], self.actions[j] = self.actions[j], self.actions[i]
            self.on_change()
            self.rebuild()


class SkillsPage(QWidget):
    """Skills du dossier ``skills/`` : une case pour l'activer, un bouton pour l'éditer."""

    open_requested = Signal(Path)

    def __init__(self, doc: CommentedMap, folder: Path, on_change: Callable[[], None]) -> None:
        super().__init__()
        self.doc, self.folder, self.on_change = doc, folder, on_change
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self.lay.setSpacing(theme.SPACE)
        self.rebuild()

    def _enabled(self) -> list[str]:
        value = self.doc.get("skills")
        return [str(v) for v in value] if isinstance(value, list) else []

    def rebuild(self) -> None:
        while self.lay.count():
            item = self.lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                while item.layout().count():
                    sub = item.layout().takeAt(0)
                    if sub.widget():
                        sub.widget().deleteLater()
        self.lay.addWidget(hint_label(f"Dossiers {self.folder.name}/<nom>/SKILL.md : la description dit au "
                                      "cerveau quand charger les consignes, qu'il suit avec les tools et MCP."))
        enabled = self._enabled()
        for name in sorted(set(skills.available(self.folder)) | set(enabled)):
            row = QHBoxLayout()
            row.setSpacing(theme.SPACE)
            box = QCheckBox(name, checked=name in enabled)
            box.toggled.connect(lambda on, n=name: self._toggle(n, on))
            row.addWidget(box)
            try:
                about = skills.load(name, self.folder).description
            except Exception as e:  # noqa: BLE001 — message montré tel quel
                about = f"{OptionsForm.WARNING} {describe(e)}"
            row.addWidget(hint_label(about), 1)
            file = skills.skill_dir(name, self.folder) / skills.SKILL_FILE
            edit = QToolButton(text="✎", toolTip=f"Ouvrir {file.name}", autoRaise=True, enabled=file.is_file())
            edit.setAccessibleName(f"Ouvrir le skill {name}")
            edit.clicked.connect(lambda _=False, f=file: self.open_requested.emit(f))
            row.addWidget(edit)
            self.lay.addLayout(row)
        self.new_name = QLineEdit(placeholderText="nom-du-skill")
        self.new_name.setAccessibleName("Nom du nouveau skill")
        create = QPushButton("Nouveau skill")
        create.clicked.connect(lambda: self.create(self.new_name.text().strip()))
        line = QHBoxLayout()
        line.setSpacing(theme.SPACE)
        line.addWidget(self.new_name, 1)
        line.addWidget(create)
        line.addStretch(1)
        self.lay.addLayout(line)
        self.lay.addStretch(1)

    def _toggle(self, name: str, on: bool) -> None:
        value = self.doc.get("skills")
        if not isinstance(value, CommentedSeq):
            value = self.doc["skills"] = CommentedSeq(value or [])
        if on and name not in value:
            value.append(name)
        elif not on and name in value:
            value.remove(name)
        self.on_change()

    def create(self, name: str) -> Path | None:
        """Crée ``skills/<name>/SKILL.md`` (modèle à compléter), l'active et l'ouvre."""
        if not SKILL_NAME.fullmatch(name):
            QMessageBox.warning(self, "Nouveau skill", "Nom attendu : lettres minuscules, chiffres et tirets.")
            return None
        file = self.folder / name / skills.SKILL_FILE
        if not file.exists():
            file.parent.mkdir(parents=True, exist_ok=True)
            write_atomic(file, SKILL_TEMPLATE.format(name=name).encode("utf-8"))
        self._toggle(name, True)
        self.rebuild()
        self.open_requested.emit(file)
        return file


def validate(text: str) -> str | None:
    """Vérifie que le YAML produit se charge comme une config Seldon ; renvoie l'erreur éventuelle."""
    try:
        Config.parse(text)
    except Exception as e:  # noqa: BLE001 — message montré tel quel
        return describe(e)
    return None


class SettingsPanel(QWidget):
    saved = Signal(Path)
    open_requested = Signal(Path)
    dirty_changed = Signal(bool)
    reloaded = Signal()   # formulaire reconstruit (config relue, valide ou non)

    def __init__(self, config_path: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.path = config_path.resolve()
        self.plugins = discover(PACKAGE)
        self.memory_params = class_params(PACKAGE / "core" / "memory.py", "ConversationMemory")
        self.memory_params.pop("brain", None)
        self.memory_params.pop("brain_kind", None)
        self.doc = CommentedMap()
        self.disk_text = ""
        self.dirty = False
        self.records: list[RecordsView] = []

        self.title = QLabel(objectName="path")
        self.title.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)   # chemin long : rogné
        self.banner = QLabel(wordWrap=True, objectName="banner")
        self.banner.hide()
        self.save_btn = QPushButton("Enregistrer", objectName="primary")
        self.revert_btn = QPushButton("Annuler les modifications")
        yaml_btn = QPushButton("Ouvrir le YAML")
        self.save_btn.clicked.connect(self.save)
        self.revert_btn.clicked.connect(self.revert)
        yaml_btn.clicked.connect(lambda: self.open_requested.emit(self.path))
        top = QHBoxLayout()
        top.setSpacing(theme.SPACE)
        top.addWidget(theme.caption("fichier"))
        top.addWidget(self.title, 1)
        top.addWidget(yaml_btn)
        top.addWidget(self.revert_btn)
        top.addWidget(self.save_btn)

        self.nav = QListWidget()
        self.pages = QStackedWidget()
        self.nav.currentRowChanged.connect(self.pages.setCurrentIndex)
        nav_card = theme.NeuCard("raised")
        nav_card.setFixedWidth(220)
        nav_card.add(theme.caption("sections"))
        nav_card.add(self.nav, 1)
        pages_card = theme.NeuCard("raised")
        pages_card.add(self.pages, 1)
        body = QHBoxLayout()
        body.setSpacing(0)   # les cartes portent déjà leur marge d'ombre
        body.addWidget(nav_card)
        body.addWidget(pages_card, 1)

        head = QVBoxLayout()
        head.setContentsMargins(theme.GUTTER, theme.SPACE, theme.GUTTER, 0)
        head.setSpacing(theme.SPACE)
        head.addLayout(top)
        head.addWidget(self.banner)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addLayout(head)
        lay.addLayout(body, 1)

        self.watcher = QFileSystemWatcher(self)
        self._disk_timer = QTimer(self, singleShot=True, interval=150)   # une rafale de notifications = une relecture
        self._disk_timer.timeout.connect(self._disk_changed)
        self.watcher.fileChanged.connect(lambda _p: self._disk_timer.start())
        self.reload()

    # --- construction -------------------------------------------------------
    def _page(self, title: str, widget: QWidget, intro: str = "", extra: tuple[QWidget, ...] = ()) -> None:
        inner = QWidget()
        v = QVBoxLayout(inner)
        v.setContentsMargins(0, 0, theme.PAD, 0)   # place de la barre de défilement
        v.setSpacing(theme.SPACE)
        head = QLabel(title.upper(), objectName="pageTitle")
        f = theme.mono_font(14)
        f.setWeight(f.Weight.Bold)
        f.setLetterSpacing(f.SpacingType.AbsoluteSpacing, 3)
        head.setFont(f)
        row = QHBoxLayout()
        row.setSpacing(theme.PAD)
        row.addWidget(head)
        row.addWidget(theme.Stripes(48, 14))
        row.addStretch(1)
        v.addLayout(row)
        if intro:
            v.addWidget(hint_label(intro))
        v.addSpacing(theme.SECTION - theme.SPACE)
        v.addWidget(widget)
        for view in extra:
            v.addSpacing(theme.SECTION - theme.SPACE)
            v.addWidget(view)
        v.addStretch(1)
        area = QScrollArea(widgetResizable=True, frameShape=QFrame.Shape.NoFrame)
        area.setWidget(inner)
        self.pages.addWidget(area)
        self.nav.addItem(title)

    def reload(self) -> None:
        row = max(self.nav.currentRow(), 0)
        self.banner.hide()
        text = None
        try:
            text = self.path.read_text(encoding="utf-8") if self.path.exists() else ""
            self.disk_text = text
            self.doc = configdoc.loads(text)
        except Exception as e:  # noqa: BLE001 — YAML invalide : on le dit et on propose l'éditeur
            self.doc = None
            error = (validate(text) if text else None) or str(e)   # message en français, avec ligne et colonne
            self.banner.setText(f"{self.path.name} est illisible, le formulaire est indisponible.\n{error}\n"
                                "Corrigez-le dans l'éditeur (« Ouvrir le YAML ») : il sera relu dès son enregistrement.")
            self.banner.show()
        else:
            if not self.path.exists():
                self.banner.setText(f"{self.path.name} n'existe pas encore : il sera créé à l'enregistrement.")
                self.banner.show()
        if str(self.path) not in self.watcher.files() and self.path.exists():
            self.watcher.addPath(str(self.path))
        self.nav.clear()
        self.records = []
        while self.pages.count():
            w = self.pages.widget(0)
            self.pages.removeWidget(w)
            w.deleteLater()
        self.title.setText(str(self.path).replace(str(Path.home()), "~", 1))
        self.title.setToolTip(str(self.path))
        if self.doc is None:
            empty = hint_label("Formulaire indisponible tant que la configuration est illisible.")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.pages.addWidget(empty)
            self._set_dirty(False)
            self.reloaded.emit()
            return

        reserved = {k for k, _t, _c in SECTIONS} | {"tools", "mcp", "skills", "actions", "memory"}
        general = OptionsForm(self.doc, {}, self._changed,
                              keys=lambda k: k not in reserved, extensible=False)
        self._page("Général", general, "Réglages globaux de l'assistant.")
        memory = self.doc.get("memory")
        if not isinstance(memory, CommentedMap):
            memory = self.doc["memory"] = CommentedMap(enabled=True)
        self._page("Mémoire", OptionsForm(memory, {"enabled": Param("enabled", True, "True", "bool"), **self.memory_params},
                                          self._changed), "Conversations sauvegardées et reprise.",
                   (self._records(ConversationsView(self.conversations_dir, self.path.parent)),))
        for key, title, cat in SECTIONS:
            node = configdoc.stage(self.doc, key)
            extra = (self._records(SpeakersView(self.voices_dir, self.conversations_dir, self.path.parent)),) if key == "speakers" else ()
            if key == "brain":
                env = self.path.parent / ".env"
                extra = (BackupEditor(node, self.plugins.get(cat, {}), self._changed, env),)
                editor = BrainEditor(node, self.plugins.get(cat, {}), self._changed, env)
            else:
                editor = StageEditor(node, self.plugins.get(cat, {}), self._changed)
            self._page(title, editor, extra=extra)
        migrated = configdoc.split_actions(self.doc, registry.category_of)   # ancien format « actions: »
        # conversations et voix : outils de la mémoire et des locuteurs, plus des tools à activer
        migrated = configdoc.drop_actions(self.doc.get("tools"), registry.PLUGIN_TOOLS) or migrated
        for key, title, intro, noun, default in (
                ("tools", "Tools", "Actions en Python que le cerveau peut appeler. L'ordre n'a pas d'importance.",
                 "le tool", ["speak"]),
                ("mcp", "MCP", "Serveurs MCP : leurs outils sont proposés au cerveau.", "le serveur MCP", [])):
            seq = self.doc.get(key)
            if not isinstance(seq, CommentedSeq):
                seq = self.doc[key] = CommentedSeq(seq or default)
            page = ActionsPage(seq, self.plugins.get(key, {}), self._changed, intro, noun)
            page.open_requested.connect(self.open_requested.emit)
            self._page(title, page)
        skill_page = SkillsPage(self.doc, self._data_dir(None, skills.SKILLS_DIR), self._changed)
        skill_page.open_requested.connect(self.open_requested.emit)
        self._page("Skills", skill_page)
        self.nav.setCurrentRow(min(row, self.nav.count() - 1))
        self._set_dirty(migrated)   # à enregistrer pour adopter tools / mcp
        self.reloaded.emit()

    def _records(self, view: RecordsView) -> RecordsView:
        view.open_requested.connect(self.open_requested.emit)
        self.records.append(view)
        return view

    # --- dossiers de données (relatifs au dossier de config.yaml) -------------
    def _data_dir(self, value: Any, default: Any) -> Path:
        path = Path(str(value or default)).expanduser()
        return path if path.is_absolute() else self.path.parent / path

    def conversations_dir(self) -> Path:
        memory = self.doc.get("memory") if self.doc is not None else None
        default = self.memory_params.get("dir")
        return self._data_dir(memory.get("dir") if hasattr(memory, "get") else None,
                              default.default if default else "data/conversations")

    def voices_dir(self) -> Path:
        stage = self.doc.get("speakers") if self.doc is not None else None
        options = (stage.get("options") if hasattr(stage, "get") else None) or {}
        param = self.plugins.get("speakers", {}).get("resemblyzer")
        default = param.params.get("voices_dir") if param else None
        return self._data_dir(options.get("voices_dir"), default.default if default else "data/voices")

    # --- état ---------------------------------------------------------------
    def _changed(self) -> None:
        self._set_dirty(configdoc.to_text(self.doc) != self.disk_text)
        for view in self.records:
            view.follow_folder()

    def _set_dirty(self, dirty: bool) -> None:
        self.save_btn.setEnabled(dirty)
        self.revert_btn.setEnabled(dirty)
        if dirty != self.dirty:
            self.dirty = dirty
            self.dirty_changed.emit(dirty)

    def save(self) -> bool:
        if self.doc is None:
            return False
        text = configdoc.to_text(self.doc)
        if (error := validate(text)) is not None:
            QMessageBox.warning(self, "Configuration invalide", error)
            return False
        try:   # atomique, droits conservés : jamais de config.yaml à moitié écrit
            write_atomic(self.path, text.encode("utf-8"))
        except OSError as e:
            QMessageBox.warning(self, "Enregistrement impossible", str(e))
            return False
        self.disk_text = text
        self.banner.hide()
        self._set_dirty(False)
        self.saved.emit(self.path)
        return True

    def revert(self) -> None:
        self.reload()

    def confirm_close(self) -> bool:
        if not self.dirty:
            return True
        answer = QMessageBox.question(
            self, "Paramètres non enregistrés", "Enregistrer les modifications des paramètres ?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Save:
            return self.save()
        return answer == QMessageBox.StandardButton.Discard

    def _disk_changed(self) -> None:
        if self.path.exists() and str(self.path) not in self.watcher.files():
            self.watcher.addPath(str(self.path))
        try:
            text = self.path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return
        if text == self.disk_text:
            return
        if not self.dirty:
            self.reload()
            return
        self.banner.setText("config.yaml a été modifié ailleurs. « Annuler les modifications » pour recharger, "
                            "ou « Enregistrer » pour écraser avec ce formulaire.")
        self.banner.show()
