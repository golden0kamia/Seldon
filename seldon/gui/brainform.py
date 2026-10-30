"""Page « Cerveau » : formulaire propre à chaque moteur.

Chaque moteur montre toutes ses options, rangées en trois blocs :
connexion (adresse, clé API, modèle avec la liste de ceux du serveur),
réglages courants, et options avancées repliées. Une option absente de
config.yaml affiche sa valeur par défaut et n'est écrite que si on la change.

Changer de moteur met de côté les options du précédent (retrouvées si on y
revient) : on peut comparer plusieurs moteurs sans tout ressaisir.

Les clés API (Anthropic, OpenAI) ne vont jamais dans config.yaml : elles sont
écrites dans le fichier ``.env`` voisin (droits 600). Claude avec
``auth: abonnement`` n'a pas de clé : le compte est celui de Claude Code
(``claude auth login``), vérifié par le bouton « Lister ».
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QToolButton, QVBoxLayout,
    QWidget,
)
from ruamel.yaml.comments import CommentedMap

from seldon.config import _ENV_LINE
from seldon.gui import configdoc, theme
from seldon.gui.fsutil import write_atomic
from seldon.gui.plugins import MISSING, Param, PluginInfo, choices_from_comment
from seldon.gui.widgets import FIXED_CHOICES, guard_wheel, hint_label, make_editor

ENGINE_LABELS = {
    "llamacpp": "llama.cpp — serveur local",
    "ollama": "Ollama — local",
    "claude": "Claude — Anthropic, en ligne",
    "openai": "OpenAI — GPT, en ligne",
}
ENV_KEYS = {"claude": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}   # clé lue dans .env
CONNECTION = {"llamacpp": ("base_url", "api_key"), "ollama": ("host",), "openai": ("base_url",),
              "claude": ("auth", "claude_bin")}
# auth: abonnement — pas de liste de modèles sans clé API : ceux de la famille actuelle
ACCOUNT_MODELS = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5", "claude-fable-5-1"]
MAIN = ("effort", "reasoning_effort", "thinking", "think", "vision", "temperature", "stream", "extra_instructions")
MODEL_PLACEHOLDER = {"llamacpp": "modèle chargé sur le serveur"}
WARNING = "⚠"


# --- fichier .env ---------------------------------------------------------------

def read_env(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return {}
    out = {}
    for line in lines:
        if line.lstrip().startswith("#") or not (m := _ENV_LINE.match(line)):
            continue
        key, value = m.groups()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[key] = value
    return out


def write_env(path: Path, key: str, value: str) -> None:
    """Définit ``key`` dans le fichier .env (ligne remplacée ou ajoutée), droits 600."""
    if not re.fullmatch(r"[^\s#\"']+", value):
        raise ValueError("la clé ne doit contenir ni espace, ni guillemet, ni « # »")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        lines = []
    pattern = re.compile(rf"^\s*(?:export\s+)?{re.escape(key)}\s*=")
    for i, line in enumerate(lines):
        if pattern.match(line):
            lines[i] = f"{key}={value}"
            break
    else:
        lines.append(f"{key}={value}")
    write_atomic(path, ("\n".join(lines) + "\n").encode("utf-8"))
    os.chmod(path, 0o600)


def api_key(engine: str, options: dict[str, Any], env_path: Path) -> str | None:
    if options.get("api_key"):
        return str(options["api_key"])
    name = ENV_KEYS.get(engine)
    return (os.environ.get(name) or read_env(env_path).get(name)) if name else None


def uses_account(engine: str, options: dict[str, Any]) -> bool:
    return engine == "claude" and options.get("auth") == "abonnement"


def account_status(claude_bin: str | None) -> str:
    """Compte claude.ai auquel Claude Code est connecté (``claude auth status``) ; ValueError sinon."""
    path = shutil.which(claude_bin or "claude")
    if path is None:
        raise ValueError("Claude Code (commande claude) introuvable")
    out = subprocess.run([path, "auth", "status"], capture_output=True, text=True, timeout=20,
                         env={**os.environ, "ANTHROPIC_API_KEY": "", "ANTHROPIC_AUTH_TOKEN": ""})
    try:
        status = json.loads(out.stdout)
    except ValueError:
        status = {}
    if not status.get("loggedIn") or status.get("authMethod") != "claude.ai":
        raise ValueError("compte non connecté : lancez « claude auth login » dans un terminal")
    return str(status.get("email") or "compte claude.ai")


# --- liste des modèles ----------------------------------------------------------

def list_models(engine: str, options: dict[str, Any], defaults: dict[str, Any], env_path: Path) -> list[str]:
    """Modèles proposés par le serveur du moteur (requête réseau : à lancer hors du fil de l'interface)."""
    import httpx

    opt = {**defaults, **{k: v for k, v in options.items() if v is not None}}
    key = api_key(engine, options, env_path)
    if uses_account(engine, opt):
        account_status(opt.get("claude_bin"))
        return list(ACCOUNT_MODELS)
    if engine == "claude":
        if not key:
            raise ValueError("clé API Anthropic absente")
        r = httpx.get("https://api.anthropic.com/v1/models", params={"limit": 100}, timeout=10,
                      headers={"x-api-key": key, "anthropic-version": "2023-06-01"})
        r.raise_for_status()
        return [m["id"] for m in r.json()["data"]]
    if engine == "ollama":
        host = opt.get("host") or os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
        host = host if "://" in host else f"http://{host}"
        r = httpx.get(f"{host.rstrip('/')}/api/tags", timeout=10)
        r.raise_for_status()
        return sorted(m["name"] for m in r.json()["models"])
    if "base_url" not in opt:
        raise ValueError("ce moteur ne sait pas lister ses modèles")
    if engine == "openai" and not key:
        raise ValueError("clé API OpenAI absente")
    r = httpx.get(f"{str(opt['base_url']).rstrip('/')}/models", timeout=10,
                  headers={"Authorization": f"Bearer {key}"} if key else {})
    r.raise_for_status()
    ids = [m["id"] for m in r.json()["data"]]
    return ids if engine != "openai" else sorted(i for i in ids if i.startswith(("gpt", "o", "chatgpt")))


def _error_text(e: BaseException) -> str:
    response = getattr(e, "response", None)
    if response is not None and response.status_code in (401, 403):
        return "clé API refusée"
    if type(e).__name__ in ("ConnectError", "ConnectTimeout"):
        return "serveur injoignable"
    if isinstance(e, ImportError):
        return "module httpx non installé"
    return str(e) or type(e).__name__


class _Fetch(QObject):
    """Résultat d'une requête lancée dans un fil à part (signal reçu dans le fil de l'interface)."""

    done = Signal(object, str)   # liste de modèles ou None, message d'erreur


# --- formulaire ------------------------------------------------------------------

class _Section(QWidget):
    """Grille libellé / champ / aide pour une liste d'options."""

    def __init__(self) -> None:
        super().__init__()
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(theme.PAD)
        self.grid.setVerticalSpacing(theme.SPACE)
        self.grid.setColumnStretch(1, 3)
        self.grid.setColumnStretch(2, 4)
        self.rows = 0

    def add(self, label: QLabel, editor: QWidget, hint: str) -> None:
        self.grid.addWidget(label, self.rows, 0)
        self.grid.addWidget(editor, self.rows, 1)
        self.grid.addWidget(hint_label(hint), self.rows, 2)
        self.rows += 1


class BrainEditor(QWidget):
    """Moteur du cerveau et ses options (``{engine, options}``), formulaire selon le moteur."""

    def __init__(self, node: CommentedMap, plugins: dict[str, PluginInfo], on_change: Callable[[], None],
                 env_path: Path) -> None:
        super().__init__()
        self.node, self.plugins, self.on_change, self.env_path = node, plugins, on_change, env_path
        self.stash: dict[str, CommentedMap] = {}    # options des moteurs quittés
        self.models: dict[str, list[str]] = {}      # modèles listés, par moteur
        self.fetch = _Fetch(self)
        self.fetch.done.connect(self._models_fetched)
        self.advanced_open = False

        self.combo = guard_wheel(QComboBox())
        self.combo.setAccessibleName("Moteur")
        engine = self.engine
        names = [n for n in ENGINE_LABELS if n in plugins] + sorted(n for n in plugins if n not in ENGINE_LABELS)
        for name in names if engine in names else [engine, *names]:
            self.combo.addItem(ENGINE_LABELS.get(name, name), name)
            if name in plugins:
                self.combo.setItemData(self.combo.count() - 1, plugins[name].doc, Qt.ItemDataRole.ToolTipRole)
        self.combo.setCurrentIndex(self.combo.findData(engine))
        self.combo.currentIndexChanged.connect(lambda _i: self._engine_changed(self.combo.currentData()))
        head = QHBoxLayout()
        head.setSpacing(theme.PAD)
        head.addWidget(theme.caption("moteur"))
        head.addWidget(self.combo, 1)

        self.doc = hint_label("")
        self.body = QVBoxLayout()
        self.body.setSpacing(theme.SPACE)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(theme.SPACE)
        lay.addLayout(head)
        lay.addWidget(self.doc)
        lay.addLayout(self.body)
        self.rebuild()

    # --- état ---------------------------------------------------------------
    @property
    def engine(self) -> str:
        return str(self.node.get("engine", ""))

    @property
    def options(self) -> CommentedMap:
        return self.node["options"]

    @property
    def params(self) -> dict[str, Param]:
        info = self.plugins.get(self.engine)
        return info.params if info else {}

    def _default(self, key: str) -> Any:
        param = self.params.get(key)
        return None if param is None or param.default is MISSING else param.default

    def _set(self, key: str, value: Any) -> None:
        self.options[key] = value
        self.on_change()
        if key == "auth":   # clé API ou compte : la ligne de connexion change
            self.rebuild()

    def _engine_changed(self, engine: str | None) -> None:
        if not engine or engine == self.engine:
            return
        self.stash[self.engine] = self.options
        configdoc.drop_comment(self.node, "engine")   # il décrivait le moteur quitté
        self.node["engine"] = engine
        self.node["options"] = self.stash.pop(engine, None) or CommentedMap()
        self.on_change()
        self.rebuild()

    # --- construction -------------------------------------------------------
    def rebuild(self) -> None:
        while self.body.count():
            item = self.body.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        info = self.plugins.get(self.engine)
        comment = configdoc.comment(self.node, "engine")
        self.doc.setText((info.doc if info else "Plugin externe : options libres (YAML).")
                         + (f"\nNote : {comment}" if comment else ""))

        params = self.params
        connection = [k for k in CONNECTION.get(self.engine, ()) if k in params]
        main = [k for k in MAIN if k in params]
        shown = {"model", *connection, *main}
        advanced = [k for k in params if k not in shown]
        unknown = [str(k) for k in self.options if str(k) not in params]

        self.body.addWidget(theme.caption("connexion"))
        conn = _Section()
        if "auth" in connection:   # clé API ou abonnement : choix en tête, il décide des lignes suivantes
            connection.remove("auth")
            self._add_row(conn, "auth")
        if uses_account(self.engine, self.options):
            label = QLabel("compte")
            label.setFont(theme.mono_font(10))
            conn.add(label, QLabel("abonnement claude.ai, via Claude Code"),
                     "connexion : claude auth login (terminal) ; « Lister » la vérifie")
        elif self.engine in ENV_KEYS:
            self._add_key_row(conn)
        if "model" in params or "model" in self.options:
            self._add_model_row(conn)
        for key in connection:
            self._add_row(conn, key)
        self.body.addWidget(conn)
        self.status = hint_label("")
        self.status.setVisible(False)
        self.body.addWidget(self.status)

        if main:
            self.body.addWidget(theme.caption("réglages"))
            section = _Section()
            for key in main:
                self._add_row(section, key)
            self.body.addWidget(section)

        if advanced or unknown:
            toggle = QToolButton(checkable=True, autoRaise=True, checked=self.advanced_open,
                                 text=f"Options avancées ({len(advanced) + len(unknown)})")
            toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            toggle.setArrowType(Qt.ArrowType.DownArrow if self.advanced_open else Qt.ArrowType.RightArrow)
            section = _Section()
            for key in [*advanced, *unknown]:
                self._add_row(section, key)
            section.setVisible(self.advanced_open)

            def toggled(on: bool) -> None:
                self.advanced_open = on
                toggle.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)
                section.setVisible(on)

            toggle.toggled.connect(toggled)
            self.body.addWidget(toggle)
            self.body.addWidget(section)

    def _label(self, key: str, known: bool = True) -> QLabel:
        label = QLabel(key if known else f"{key} {WARNING}")
        label.setFont(theme.mono_font(10))
        param = self.params.get(key)
        label.setToolTip(param.hint() if param else "Option inconnue pour ce moteur (erreur au démarrage).")
        return label

    def _add_row(self, section: _Section, key: str) -> None:
        param = self.params.get(key)
        value = self.options[key] if key in self.options else self._default(key)
        # option connue : aide et choix du moteur, pas du commentaire de config.yaml, pour que le
        # cerveau principal et celui de secours aient le même formulaire pour un même moteur
        comment = param.comment if param else configdoc.comment(self.options, key)
        choices = FIXED_CHOICES.get(key) or choices_from_comment(comment)
        if isinstance(value, str) and choices and value not in choices and not (param and param.default in choices):
            choices = []
        editor = make_editor(value, choices, lambda v, k=key: self._set(k, v))
        label = self._label(key, param is not None or not self.params)
        editor.setToolTip(label.toolTip())
        section.add(label, editor, comment)

    def _add_model_row(self, section: _Section) -> None:
        value = self.options.get("model", self._default("model"))
        box = guard_wheel(QComboBox(editable=True))
        box.setAccessibleName("Modèle")
        box.addItems(self.models.get(self.engine, []))
        box.setCurrentText("" if value is None else str(value))
        box.lineEdit().setPlaceholderText(MODEL_PLACEHOLDER.get(self.engine, "nom du modèle"))
        box.currentTextChanged.connect(lambda t: self._set("model", t.strip() or None))
        fetch = QPushButton("Lister")
        fetch.setToolTip("Interroge le serveur : vérifie la connexion et propose ses modèles")
        fetch.clicked.connect(self._fetch_models)
        line = QWidget()
        h = QHBoxLayout(line)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(theme.SPACE)
        h.addWidget(box, 1)
        h.addWidget(fetch)
        self.model_box = box
        param = self.params.get("model")
        section.add(self._label("model"), line, param.comment if param else "")

    def _add_key_row(self, section: _Section) -> None:
        name = ENV_KEYS[self.engine]
        present = bool(os.environ.get(name) or read_env(self.env_path).get(name))
        field = QLineEdit(echoMode=QLineEdit.EchoMode.Password)
        field.setAccessibleName("Clé API")
        field.setPlaceholderText("définie dans .env — coller pour la remplacer" if present else "collez la clé API")
        save = QPushButton("Enregistrer")
        save.setToolTip(f"Écrit {name} dans {self.env_path} (jamais dans config.yaml)")
        save.setEnabled(False)
        field.textChanged.connect(lambda t: save.setEnabled(bool(t.strip())))

        def store() -> None:
            try:
                write_env(self.env_path, name, field.text().strip())
            except (OSError, ValueError) as e:
                self._show_status(f"✗ Clé non enregistrée : {e}", error=True)
                return
            field.clear()
            field.setPlaceholderText("définie dans .env — coller pour la remplacer")
            self._show_status(f"✓ {name} enregistrée dans {self.env_path.name}")

        save.clicked.connect(store)
        field.returnPressed.connect(store)
        line = QWidget()
        h = QHBoxLayout(line)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(theme.SPACE)
        h.addWidget(field, 1)
        h.addWidget(save)
        label = QLabel("clé API")
        label.setFont(theme.mono_font(10))
        section.add(label, line, f"{name} dans .env" + ("" if present else " — absente"))

    def _show_status(self, text: str, error: bool = False) -> None:
        self.status.setText(text)
        self.status.setObjectName("findinfo" if error else "hint")
        self.status.style().polish(self.status)
        self.status.setVisible(True)

    # --- modèles ------------------------------------------------------------
    def _fetch_models(self) -> None:
        engine = self.engine
        options = dict(self.options)
        defaults = {k: p.default for k, p in self.params.items() if p.default is not MISSING}
        self._show_status("Connexion au serveur…")
        fetch = self.fetch

        def work() -> None:
            try:
                models, error = list_models(engine, options, defaults, self.env_path), ""
            except Exception as e:  # noqa: BLE001 — affiché tel quel
                models, error = None, _error_text(e)
            try:
                fetch.done.emit((engine, models), error)
            except RuntimeError:   # formulaire fermé entre-temps
                pass

        threading.Thread(target=work, daemon=True).start()

    def _models_fetched(self, result: tuple[str, list[str] | None], error: str) -> None:
        engine, models = result
        if models is None:
            self._show_status(f"✗ {error}", error=True)
            return
        self.models[engine] = models
        if engine != self.engine:
            return
        current = self.model_box.currentText()
        self.model_box.blockSignals(True)
        self.model_box.clear()
        self.model_box.addItems(models)
        self.model_box.setCurrentText(current)
        self.model_box.blockSignals(False)
        missing = current and current not in models
        if uses_account(engine, self.options):
            self._show_status(f"✓ Compte claude.ai connecté — modèles {', '.join(models)}")
            return
        self._show_status(f"✓ Connexion réussie : {len(models)} modèle(s) disponible(s)"
                          + (f" — « {current} » n'en fait pas partie" if missing else ""), error=bool(missing))


class BackupEditor(QWidget):
    """Cerveau de secours (``brain.backup``) : utilisé quand le principal ne répond pas."""

    def __init__(self, node: CommentedMap, plugins: dict[str, PluginInfo], on_change: Callable[[], None],
                 env_path: Path) -> None:
        super().__init__()
        self.node, self.plugins, self.on_change, self.env_path = node, plugins, on_change, env_path
        self.saved: CommentedMap | None = None   # réglages gardés si on décoche puis recoche
        self.check = QCheckBox("Cerveau de secours")
        self.check.setChecked(isinstance(node.get("backup"), CommentedMap))
        self.check.toggled.connect(self._toggled)
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self.lay.setSpacing(theme.SPACE)
        self.lay.addWidget(self.check)
        self.lay.addWidget(hint_label("Prend le relais si le cerveau principal ne démarre pas ou échoue "
                                      "(serveur arrêté, clé refusée, quota…) ; le principal est réessayé après 5 min."))
        self.editor: BrainEditor | None = None
        if self.check.isChecked():
            self._show()

    def _show(self) -> None:
        self.editor = BrainEditor(configdoc.stage(self.node, "backup"), self.plugins, self.on_change, self.env_path)
        self.lay.addWidget(self.editor)

    def _toggled(self, on: bool) -> None:
        if on:
            if self.saved is not None:
                self.node["backup"] = self.saved
            else:
                current = str(self.node.get("engine", ""))
                engine = next((e for e in ("openai", "claude", *sorted(self.plugins)) if e != current), current)
                self.node["backup"] = CommentedMap(engine=engine, options=CommentedMap())
            self._show()
        else:
            self.saved = self.node.pop("backup", None)
            if self.editor is not None:
                self.editor.deleteLater()
                self.editor = None
        self.on_change()
