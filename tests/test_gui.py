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


def test_voice_samples_reads_npy_header(tmp_path):
    np = pytest.importorskip("numpy")
    from seldon.gui.records import voice_samples

    np.save(tmp_path / "Bastien.npy", np.zeros((5, 256), np.float32))
    (tmp_path / "faux.npy").write_bytes(b"rien")
    assert voice_samples(tmp_path / "Bastien.npy") == 5
    assert voice_samples(tmp_path / "faux.npy") is None


def test_speakers_and_memory_pages_list_records(window, tmp_path):
    import json

    from seldon.gui.records import ConversationsView, SpeakersView

    voices, convs = tmp_path / "data" / "voices", tmp_path / "data" / "conversations"
    voices.mkdir(parents=True)
    convs.mkdir(parents=True)
    (voices / "Claire.npy").write_bytes(b"")
    (convs / "c1.json").write_text(json.dumps(dict(
        id="c1", created=0, updated=60, brain="llamacpp", title="Lance les tests", speakers=["Claire"],
        transcript=[{"role": "user", "text": "Lance les tests", "speaker": "Claire", "ts": 60}])), encoding="utf-8")
    s = window.settings
    speakers = s.findChild(SpeakersView)
    memory = s.findChild(ConversationsView)
    for view in (speakers, memory):
        view.refresh()
    assert speakers.list.item(0).text() == "Claire"
    assert "Lance les tests" in speakers.detail.text()          # conversations de ce locuteur
    assert memory.list.item(0).text().startswith("Lance les tests")
    assert "Participants : Claire." in memory.detail.text()      # résumé par défaut
    memory.modes[1].click()
    assert "<b>Claire</b>" in memory.detail.text()               # transcription
    opened = []
    s.open_requested.connect(opened.append)
    memory.open_btn.click()
    assert opened == [convs.resolve() / "c1.json"]
    s.doc["memory"]["dir"] = "ailleurs"                          # le dossier suit le formulaire
    s._changed()
    assert memory.list.count() == 0 and "ailleurs" in memory.detail.text()


def test_tab_close_button_is_spaced_from_text_and_edge(window, tmp_path):
    from PySide6.QtWidgets import QTabBar

    from seldon.gui import theme

    tab = window.editors.open_file(tmp_path / "projet" / "main.py")
    bar = window.editors.tabs.tabBar()
    holder = bar.tabButton(window.editors.tabs.indexOf(tab), QTabBar.ButtonPosition.RightSide)
    assert holder.layout().contentsMargins().right() > theme.TAB_PAD   # la croix n'est plus collée au bord
    holder.findChild(type(window.masthead.variant_btn)).click()       # la croix ferme bien l'onglet
    assert tab not in window.editors.file_tabs()


def test_layout_does_not_move_with_state_or_engine(window, qapp):
    from PySide6.QtCore import QPoint
    from PySide6.QtWidgets import QLineEdit

    from seldon.gui import theme
    from seldon.gui.settings import StageEditor

    window.resize(1400, 850)
    window.show()
    qapp.processEvents()
    m = window.masthead
    where = lambda: [b.mapTo(window, QPoint(0, 0)) for b in (*m.mode_buttons, m.led, m.variant_btn)]
    before = where()
    m.set_state("12 modif. non enregistrée(s)", theme.AMBER)   # texte d'état plus long
    qapp.processEvents()
    assert where() == before

    s = window.settings
    s.nav.setCurrentRow(4)   # Cerveau
    stage = s.pages.currentWidget().findChild(StageEditor)
    fields = lambda: sorted((f.mapTo(window, QPoint(0, 0)).x(), f.mapTo(window, QPoint(0, 0)).y())
                            for i in range(stage.form.grid.count())
                            if isinstance(f := stage.form.grid.itemAt(i).widget(), QLineEdit))
    qapp.processEvents()
    ref = fields()
    for engine in ("ollama", "claude", "llamacpp"):   # options inconnues de certains moteurs : « ⚠ »
        stage.combo.setCurrentText(engine)
        qapp.processEvents()
        assert fields() == ref, engine


def test_logs_page_runs_program_and_streams_output(window, qapp, monkeypatch):
    from PySide6.QtCore import QDeadlineTimer

    from seldon.gui.window import LOGS_PAGE

    r = window.runner
    script = ("import sys; print('2026 WARNING seldon: attention', flush=True); "
              "print('écho', sys.stdin.readline().strip(), flush=True)")
    monkeypatch.setattr(r, "arguments", lambda text: ["-c", script])
    window.launch(text=True)
    assert window.pages.currentIndex() == LOGS_PAGE
    assert r.process.waitForStarted(5000) and r.running
    assert not window.run_actions["start"].isEnabled() and window.run_actions["stop"].isEnabled()
    assert r.input.isEnabled()                       # stdin ouvert en mode texte
    r.input.setText("bonjour")
    r.send()
    deadline = QDeadlineTimer(5000)
    while r.running and not deadline.hasExpired():
        r.process.waitForFinished(50)
        qapp.processEvents()
    qapp.processEvents()
    log = r.log.toPlainText()
    assert "WARNING seldon: attention" in log and "écho bonjour" in log and "code 0" in log
    assert window.run_actions["start"].isEnabled() and not r.input.isEnabled()
    assert window.masthead.state.text() == "prêt"


def test_logs_page_answers_confirmation_in_voice_mode(window, qapp, monkeypatch):
    from PySide6.QtCore import QDeadlineTimer

    from seldon.gui.window import LOGS_PAGE

    r = window.runner
    script = ("import sys; print('\\n[confirmation] Lancer ls ? [o/N]', flush=True); "
              "print('réponse', sys.stdin.readline().strip(), flush=True)")
    monkeypatch.setattr(r, "arguments", lambda text: ["-c", script])
    window.launch(text=False)
    window.show_page(0)
    assert r.process.waitForStarted(5000)
    assert not r.input.isEnabled() and r.yes_btn.isHidden()   # mode vocal : pas de saisie libre
    deadline = QDeadlineTimer(5000)
    while not r.pending and not deadline.hasExpired():
        r.process.waitForReadyRead(50)
        qapp.processEvents()
    assert r.pending and r.input.isEnabled() and not r.yes_btn.isHidden()
    assert window.pages.currentIndex() == LOGS_PAGE
    r.yes_btn.click()
    assert not r.pending and r.yes_btn.isHidden()
    while r.running and not deadline.hasExpired():
        r.process.waitForFinished(50)
        qapp.processEvents()
    qapp.processEvents()
    assert "réponse oui" in r.log.toPlainText()


def test_logs_page_mutes_microphone_with_signals(window, qapp, monkeypatch):
    import os

    if os.name != "posix":
        pytest.skip("signaux POSIX")
    from PySide6.QtCore import QDeadlineTimer

    r = window.runner
    script = ("import signal, sys, time\n"
              "say = lambda m: print('2026 INFO seldon.stt.whisper: [micro]', m, flush=True)\n"
              "signal.signal(signal.SIGUSR1, lambda *a: say('coupé'))\n"
              "signal.signal(signal.SIGUSR2, lambda *a: (say('rétabli'), sys.exit(0)))\n"
              "print('prêt', flush=True)\n"
              "while True: time.sleep(0.01)\n")
    monkeypatch.setattr(r, "arguments", lambda text: ["-c", script])
    window.launch(text=False)
    assert r.process.waitForStarted(5000)
    assert not r.mic_btn.isHidden() and not r.mic_btn.isEnabled()   # pas encore de signaux en place
    deadline = QDeadlineTimer(5000)

    def wait(cond):
        while not cond() and not deadline.hasExpired():
            r.process.waitForReadyRead(50)
            qapp.processEvents()
        assert cond()

    wait(lambda: r.ready)
    assert r.mic_btn.isEnabled()
    window.run_actions["mute"].trigger()               # F6
    wait(lambda: "[micro] coupé" in r.log.toPlainText())
    assert r.mic_btn.isChecked() and "coupé" in r.mic_btn.text()
    r.mic_btn.click()
    wait(lambda: not r.running)
    assert "[micro] rétabli" in r.log.toPlainText()
    assert not r.mic_btn.isChecked() and not r.mic_btn.isEnabled()
