"""Tests de l'interface (Qt hors écran) : découverte des plugins, aller-retour YAML, fenêtre."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

pytest.importorskip("ruamel.yaml")

from seldon.gui import configdoc  # noqa: E402
from seldon.gui.plugins import choices_from_comment, discover  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def test_discover_reads_plugins_without_importing():
    plugins = discover()
    assert {"stt", "speakers", "brain", "tts", "actions"} <= set(plugins)
    thinking = plugins["brain"]["llamacpp"].params["thinking"]
    assert thinking.default is False and "raisonnement" in thinking.comment
    assert "claude_code" in plugins["actions"]   # trouvé même hors de _BUILTINS


def test_choices_from_comment():
    assert choices_from_comment("low | medium | high") == ["low", "medium", "high"]
    assert choices_from_comment('"silero" ou "energy"') == ["silero", "energy"]
    assert choices_from_comment("tiny / base / small") == ["tiny", "base", "small"]
    assert choices_from_comment("rien à proposer") == []


def test_config_roundtrip_keeps_comments():
    text = (ROOT / "config.yaml").read_text(encoding="utf-8")
    doc = configdoc.loads(text)
    for key in configdoc.STAGES:
        configdoc.stage(doc, key)
    for i in range(len(doc["actions"])):
        configdoc.action_entry(doc["actions"], i)
    assert configdoc.to_text(doc) == text
    doc["brain"]["options"]["max_steps"] = 12
    out = configdoc.to_text(doc)
    assert "max_steps: 12" in out and "# true si le modèle lit les images" in out
    assert "model: null" in out


# --- fenêtre -----------------------------------------------------------------

@pytest.fixture(scope="module")
def qapp(tmp_path_factory):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ["XDG_CONFIG_HOME"] = str(tmp_path_factory.mktemp("xdg"))   # QSettings isolés
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from PySide6.QtCore import QSettings

    from seldon.gui.window import MainWindow

    QSettings("Seldon", "seldon-gui").clear()   # chaque test part d'une fenêtre neuve
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Discard)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: pytest.fail(f"avertissement : {a[2]}"))
    shutil.copy(ROOT / "config.yaml", tmp_path / "config.yaml")
    (tmp_path / "projet").mkdir()
    (tmp_path / "projet" / "main.py").write_text("print('bonjour')\n", encoding="utf-8")
    w = MainWindow(tmp_path / "config.yaml", tmp_path / "projet")
    yield w
    w.close()


def test_settings_edit_and_save(window, tmp_path):
    from PySide6.QtWidgets import QCheckBox

    s = window.settings
    assert s.nav.count() == 7 and not s.dirty
    brain_page = s.pages.widget(4)   # Général, Mémoire, STT, Locuteurs, Cerveau…
    checks = {c.parent().layout().itemAt(c.parent().layout().indexOf(c) - 1).widget().text(): c
              for c in brain_page.findChildren(QCheckBox)}
    checks["thinking"].setChecked(True)
    assert s.dirty
    assert s.save()
    text = (tmp_path / "config.yaml").read_text(encoding="utf-8")
    assert "thinking: true        # raisonnement du modèle" in text
    assert "# engine: ollama" in text   # blocs commentés conservés


def test_editor_open_edit_save(window, tmp_path):
    path = tmp_path / "projet" / "main.py"
    tab = window.editors.open_file(path)
    assert tab is not None and tab.editor.language == "python"
    tab.editor.appendPlainText("x = 1")
    assert tab.modified
    window.save_current()
    assert path.read_text(encoding="utf-8").endswith("x = 1")
    assert not tab.modified


def test_config_opened_in_editor_follows_settings(window, tmp_path):
    tab = window.editors.open_file(tmp_path / "config.yaml")
    window.settings.doc["log_level"] = "DEBUG"
    window.settings._changed()
    window.settings.save()
    assert "log_level: DEBUG" in tab.editor.toPlainText()


def test_file_browser_root_and_rename(window, tmp_path):
    assert window.files.root == (tmp_path / "projet").resolve()
    labels = [window.files.roots.itemText(i) for i in range(window.files.roots.count())]
    assert any(l.startswith("Seldon") for l in labels)
    old = tmp_path / "projet" / "main.py"
    tab = window.editors.open_file(old)
    new = old.with_name("app.py")
    old.rename(new)
    window.files.moved.emit(old, new)
    assert tab.path == new.resolve()


def test_theme_variants_switch_and_persist(window, qapp):
    from seldon.gui import theme

    window.set_theme("dark")
    assert theme.VARIANT == "dark" and theme.DARK["BG"] in qapp.styleSheet()
    assert window.store.value("theme") == "dark"
    window.toggle_theme()
    assert theme.VARIANT == "light" and theme.LIGHT["BG"] in qapp.styleSheet()
    assert theme.DARK["BG"] not in theme.stylesheet()


def test_editor_page_always_has_a_document(window):
    from seldon.gui.window import EDITOR_PAGE, SETTINGS_PAGE

    window.show_page(EDITOR_PAGE)
    tabs = window.editors.file_tabs()
    assert len(tabs) == 1 and tabs[0].path is None and tabs[0].editor is not None
    window.editors.close_current()          # fermer le dernier onglet en rouvre un vierge
    assert len(window.editors.file_tabs()) == 1
    window.show_page(SETTINGS_PAGE)
    assert window.masthead.mode_buttons[SETTINGS_PAGE].isChecked()


def test_opening_file_replaces_blank_document_and_shows_editor(window, tmp_path):
    from seldon.gui.window import EDITOR_PAGE

    tab = window.editors.open_file(tmp_path / "projet" / "main.py")
    assert window.pages.currentIndex() == EDITOR_PAGE
    assert [t.path for t in window.editors.file_tabs()] == [tab.path]


def test_untitled_document_save_as(window, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    from seldon.gui.window import EDITOR_PAGE

    target = tmp_path / "projet" / "notes.md"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), ""))
    window.show_page(EDITOR_PAGE)
    tab = window.editors.new_document()
    tab.editor.setPlainText("# Notes")
    window.save_current()
    assert target.read_text(encoding="utf-8") == "# Notes"
    assert tab.path == target.resolve() and tab.editor.language == "markdown"
