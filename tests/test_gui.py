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


def test_retro_theme_from_menu_and_back(window, qapp):
    from seldon.gui import theme

    window.masthead.variant_btn.menu().actions()[3].trigger()    # flèche du bandeau : choix du thème
    assert theme.VARIANT == "retro" and window.store.value("theme") == "retro"
    css = qapp.styleSheet()
    assert theme.RETRO["NEON"] in css and "backdrop.svg" in css
    assert window.masthead.name.graphicsEffect() is not None   # halo néon sur la marque
    screens = [c for c in window.findChildren(theme.NeuCard) if c.kind == "screen"]
    assert screens and all(not c.scanlines.isHidden() for c in screens)
    window.toggle_theme()                                         # rétro est sombre : on passe au clair
    assert theme.VARIANT == "light" and "backdrop.svg" not in qapp.styleSheet()
    assert window.masthead.name.graphicsEffect() is None and all(c.scanlines.isHidden() for c in screens)


@pytest.mark.parametrize("variant", ["cyber", "hud"])
def test_neon_variants(window, qapp, variant):
    from seldon.gui import theme

    window.set_theme(variant)
    assert theme.VARIANT == variant and theme.is_neon() and theme.is_dark()
    css = qapp.styleSheet()
    assert theme.PALETTES[variant]["BG"] in css and "background-repeat: repeat-xy" in css
    screens = [c for c in window.findChildren(theme.NeuCard) if c.kind == "screen"]
    assert all(c.scanlines.isHidden() != theme.SCANLINES for c in screens)
    for card in window.findChildren(theme.NeuCard):
        card.resize(200, 120)
        assert not card._render().isNull()
    window.set_theme("light")


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
    script = ("import os, sys; print(f'\\n[confirmation {os.environ[\"SELDON_GUI_TOKEN\"]}] Lancer ls ? [o/N]', "
              "flush=True); "
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


def test_logs_page_restarts_in_same_mode(window, qapp, monkeypatch):
    from PySide6.QtCore import QDeadlineTimer

    r = window.runner
    modes = []
    script = "import os, time; print('pid', os.getpid(), flush=True); time.sleep(60)"
    monkeypatch.setattr(r, "arguments", lambda text: modes.append(text) or ["-c", script])
    assert not r.restart_btn.isEnabled()   # rien à redémarrer
    window.launch(text=True)
    deadline = QDeadlineTimer(10000)

    def wait(cond):
        while not cond() and not deadline.hasExpired():
            r.process.waitForReadyRead(50)
            qapp.processEvents()
        assert cond()

    wait(lambda: r.ready)
    first = r._pid
    modes.clear()
    r.restart_btn.click()
    assert r.stopping and not r.restart_btn.isEnabled() and "redémarrage" in r.command.text()
    wait(lambda: r.ready and r._pid not in (0, first))   # arrêté (Ctrl+C) puis relancé
    assert r.running and r.text_mode and modes and all(modes)   # même mode : texte
    r._flush()
    assert "Seldon arrêté" in r.log.toPlainText()
    assert window.run_actions["restart"].isEnabled()
    r.stop(wait=True)
    assert not r.running and not r.relaunch


# --- robustesse ----------------------------------------------------------------

def test_plugin_scan_survives_odd_sources(tmp_path):
    from seldon.gui.plugins import scan_file

    (tmp_path / "a.py").write_text('@register("tts", "x")\nclass X:\n    def __init__(self, s={[]}, n=-1): ...\n',
                                   encoding="utf-8")
    (tmp_path / "b.py").write_bytes(b'@register("tts", "y")\nclass Y: pass\n\0')   # octet nul : ValueError
    (tmp_path / "c.py").write_text("@register(\n", encoding="utf-8")
    [info] = scan_file(tmp_path / "a.py")
    assert info.params["s"].default_src == "{[]}" and info.params["n"].default == -1
    assert scan_file(tmp_path / "b.py") == [] and scan_file(tmp_path / "c.py") == []


def test_voice_samples_rejects_malformed_headers(tmp_path):
    from seldon.gui.records import voice_samples

    def npy(name: str, header: bytes, size: int | None = None) -> Path:
        p = tmp_path / name
        size = len(header) if size is None else size
        p.write_bytes(b"\x93NUMPY\x01\x00" + size.to_bytes(2, "little") + header)
        return p

    assert voice_samples(npy("ok.npy", b"{'descr': '<f4', 'fortran_order': False, 'shape': (3, 256), }")) == 3
    assert voice_samples(npy("plat.npy", b"{'shape': 5}")) is None          # forme non tuple
    assert voice_samples(npy("texte.npy", b"{'shape': ('a', 1)}")) is None
    assert voice_samples(npy("code.npy", b"__import__('os')")) is None      # jamais exécuté
    assert voice_samples(npy("court.npy", b"{", size=60_000)) is None       # tronqué
    assert voice_samples(npy("rec.npy", b"(" * 5000)) is None               # récursion
    (tmp_path / "six.npy").write_bytes(b"\x93NUMPY")
    assert voice_samples(tmp_path / "six.npy") is None


def test_check_name_refuses_paths_outside_folder():
    from seldon.gui.files import check_name

    assert check_name("notes.md") is None and check_name("src/notes.md", nested=True) is None
    for bad in ("../x", "/etc/passwd", "a/../../b", ".", "..", "a//b"):
        assert check_name(bad, nested=True) is not None, bad
    assert check_name("a/b") is not None   # renommage : pas de « / »


def test_new_file_never_overwrites_or_follows_links(window, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QInputDialog, QMessageBox

    folder = tmp_path / "projet"
    target = tmp_path / "ailleurs.txt"
    (folder / "lien").symlink_to(target)          # lien cassé vers un fichier hors du dossier
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warnings.append(a[2]))
    for name in ("lien", "main.py", "../evasion.txt"):
        monkeypatch.setattr(QInputDialog, "getText", lambda *a, n=name, **k: (n, True))
        window.files.new_file(folder)
    assert not target.exists() and not (tmp_path / "evasion.txt").exists()
    assert (folder / "main.py").read_text(encoding="utf-8") == "print('bonjour')\n"
    assert len(warnings) == 3


def test_file_tree_shows_root_inside_hidden_folder(window, tmp_path):
    hidden = tmp_path / ".cache" / "projet"
    (hidden / "sous").mkdir(parents=True)
    window.files.set_root(hidden)
    root = window.files.tree.rootIndex()
    assert root.isValid() and window.files.path_at(root) == hidden.resolve()


def test_editor_keeps_encoding_newlines_and_permissions(window, tmp_path):
    path = tmp_path / "projet" / "vieux.txt"
    path.write_bytes("caf\xe9\r\nligne 2\r\n".encode("latin-1"))
    path.chmod(0o640)
    tab = window.editors.open_file(path)
    assert tab.editor.toPlainText() == "café\nligne 2\n" and tab.format_label == "LATIN-1 · CRLF"
    tab.editor.appendPlainText("fin")
    assert window.editors.save(tab)
    assert path.read_bytes() == "caf\xe9\r\nligne 2\r\n\r\nfin".encode("latin-1")
    assert path.stat().st_mode & 0o777 == 0o640
    assert not list(path.parent.glob(".vieux.txt.*"))   # pas de fichier temporaire oublié


def test_big_file_is_not_highlighted(window, tmp_path):
    from seldon.gui.editor import HIGHLIGHT_MAX

    path = tmp_path / "projet" / "gros.py"
    path.write_text("x = 1\n" * (HIGHLIGHT_MAX // 6 + 10), encoding="utf-8")
    tab = window.editors.open_file(path)
    assert tab.editor.language == "python" and tab.editor.highlighter is None
    small = window.editors.open_file(tmp_path / "projet" / "main.py")
    assert small.editor.highlighter is not None


def test_settings_keep_out_of_range_numbers(window):
    from PySide6.QtWidgets import QLineEdit, QSpinBox

    from seldon.gui.settings import make_editor

    assert isinstance(make_editor(5, [], lambda v: None), QSpinBox)
    big = make_editor(4_294_967_296, [], lambda v: None)
    fine = make_editor(0.0001, [], lambda v: None)
    assert isinstance(big, QLineEdit) and big.text() == "4294967296"   # pas d'écrêtage à 1e9
    assert isinstance(fine, QLineEdit) and fine.text() == "0.0001"     # pas d'arrondi à 0,000


def test_settings_save_is_atomic_and_keeps_permissions(window, tmp_path):
    path = tmp_path / "config.yaml"
    path.chmod(0o600)
    window.settings.doc["log_level"] = "DEBUG"
    window.settings._changed()
    assert window.settings.save()
    assert "log_level: DEBUG" in path.read_text(encoding="utf-8")
    assert path.stat().st_mode & 0o777 == 0o600 and not list(tmp_path.glob(".config.yaml.*"))


def test_invalid_config_is_explained_in_french(window, tmp_path):
    (tmp_path / "config.yaml").write_text("log_level: [\n", encoding="utf-8")
    window.settings.reload()
    s = window.settings
    assert s.doc is None and not s.banner.isHidden()
    assert "ligne 2" in s.banner.text() and "Ouvrir le YAML" in s.banner.text()
    assert window.masthead.state.text() == "config. illisible"


def test_conversation_records_are_cached_and_robust(tmp_path, qapp):
    import json

    from seldon.gui import records

    (tmp_path / "c1.json").write_text(json.dumps(dict(created=0, updated="hier", brain="x")), encoding="utf-8")
    (tmp_path / "c2.json").write_text(json.dumps(dict(
        id="../../secret", created=0, updated=0, brain="x", title="Bizarre", inconnu=1,
        transcript=[{"role": "user", "text": "sans date"}])), encoding="utf-8")
    (tmp_path / "c3.json").write_bytes(b"\xff\xfe pas du JSON")
    first = records.conversations(tmp_path)
    assert records.conversations(tmp_path) is first          # rien n'a changé : pas de relecture
    view = records.ConversationsView(lambda: tmp_path)
    assert view.list.count() == 1                              # c1 (date invalide) et c3 ignorés, sans exception
    view.modes[1].click()
    assert "sans date" in view.detail.text()
    opened = []
    view.open_requested.connect(opened.append)
    view.open_btn.click()
    assert opened == [(tmp_path / "c2.json").resolve()]       # l'id vient du nom du fichier, pas du JSON


def test_journal_lines_levels_progress_and_confirmation(window):
    from seldon.gui import runner as rmod

    r = window.runner
    r.clear()
    r.feed("2026 ERROR seldon: échec\n  → piste\n")
    r.feed("téléchargement 10 %\rtéléchargement 50 %\r")        # barre de progression
    r.feed("\ntéléchargement 100 %\n")
    r.feed("2026 DEBUG seldon: détail\nlog cité : [confirmation] faux ? [o/N]\n")
    r.feed("2026 INFO x: début\r")                              # « \r\n » coupé entre deux lectures
    r.feed("\n")
    r._flush()
    lines = r.log.toPlainText().split("\n")
    assert lines[:2] == ["2026 ERROR seldon: échec", "  → piste"]
    assert "téléchargement 50 %" in lines and "téléchargement 10 %" not in lines
    assert "téléchargement 100 %" in lines and "2026 INFO x: début" in lines
    assert not r.pending                                        # tag pas en début de ligne : ignoré
    doc = r.log.document()
    assert doc.findBlockByNumber(1).userState() == rmod.ERROR   # la suite indentée hérite du niveau
    r.set_min_level(rmod.WARNING)
    blocks = [doc.findBlockByNumber(i) for i in range(doc.blockCount())]
    assert [b.text() for b in blocks if b.isVisible()] == ["2026 ERROR seldon: échec", "  → piste"]
    r.set_min_level(0)


def test_journal_decodes_utf8_split_across_reads(window):
    r = window.runner
    r.clear()
    data = "é ✓\n".encode("utf-8")
    for i in range(len(data)):
        r.feed(r._decoder.decode(data[i:i + 1]))
    r._flush()
    assert r.log.toPlainText() == "é ✓"


def test_stop_also_ends_child_processes(window, qapp, monkeypatch):
    import os
    import signal
    import time

    from PySide6.QtCore import QDeadlineTimer

    from seldon.gui import runner as rmod

    if not rmod.NEW_SESSION:
        pytest.skip("groupes de processus POSIX")
    r = window.runner
    script = ("import subprocess, sys, time\n"
              "p = subprocess.Popen([sys.executable, '-c', 'import signal, time; "
              "signal.signal(signal.SIGINT, signal.SIG_IGN); time.sleep(60)'])\n"
              "print('enfant', p.pid, flush=True)\n"
              "time.sleep(60)\n")
    monkeypatch.setattr(r, "arguments", lambda text: ["-c", script])
    r.start()
    deadline = QDeadlineTimer(10000)
    while "enfant" not in r.log.toPlainText() and not deadline.hasExpired():
        r.process.waitForReadyRead(50)
        qapp.processEvents()
        r._flush()
    child = int(r.log.toPlainText().split("enfant ")[1].split()[0])
    r.stop(wait=True)   # Ctrl+C au groupe (que l'enfant ignore), puis SIGTERM aux restes
    assert not r.running
    for _ in range(100):
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        os.kill(child, signal.SIGKILL)
        pytest.fail("sous-processus orphelin")


def test_masthead_drops_caption_before_squeezing_buttons(window, qapp):
    window.show()
    window.resize(900, 600)
    qapp.processEvents()
    m = window.masthead
    assert m.model.isHidden()
    assert all(b.width() >= b.sizeHint().width() for b in m.mode_buttons)
    window.resize(1700, 850)
    qapp.processEvents()
    assert not m.model.isHidden()


def test_journal_ignores_tags_without_launch_token(window):
    r = window.runner
    r.clear()
    r._token = "jeton"   # comme après un lancement
    r.feed("[confirmation] Exécuter rm -rf ~ ? [o/N]\n[micro] coupé\n")   # imitées : sortie d'une commande…
    assert not r.pending and not r.mic_btn.isChecked()
    r.feed("[confirmation jeton] Lancer ls ? [o/N]\n")
    r._flush()
    assert r.pending and "jeton" not in r.log.toPlainText() and "[confirmation] Lancer ls" in r.log.toPlainText()


def test_structure_analysis_follows_imports_calls_and_inheritance(tmp_path):
    from seldon.gui.structure import analyze

    pkg = tmp_path / "pkg"
    (pkg / "core").mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "core" / "__init__.py").write_text("from .base import Base\n", encoding="utf-8")
    (pkg / "core" / "base.py").write_text(
        '"""Fondations."""\nclass Base:\n    def run(self):\n        return helper()\n\n'
        "def helper():\n    return 1\n", encoding="utf-8")
    (pkg / "app.py").write_text(
        "from pkg.core import Base\nfrom pkg.core import base as b\n\n"
        "@register(\"brain\", \"demo\")\nclass App(Base):\n    def start(self):\n"
        "        self.run()\n        super().run()\n        b.helper()\n", encoding="utf-8")
    (pkg / "broken.py").write_text("def (:\n", encoding="utf-8")

    st = analyze(pkg)
    app, base = st.modules["pkg.app"], st.modules["pkg.core.base"]
    assert app.imports == {"pkg.core.base"} and "pkg.app" in base.imported_by   # ré-export suivi jusqu'à la définition
    assert base.doc == "Fondations." and st.modules["pkg.broken"].error
    cls = st.symbols["pkg.app.App"]
    assert cls.plugin == "brain/demo" and cls.bases == ["pkg.core.base.Base"]
    assert st.symbols["pkg.app.App.start"].calls == {"pkg.core.base.Base.run", "pkg.core.base.helper"}
    assert "pkg.app.App.start" in st.symbols["pkg.core.base.helper"].called_by
    assert st.packages()[0] == "pkg" and st.search("helper") == {"pkg.core.base"}


def test_structure_page_maps_package_and_opens_definitions(window):
    from seldon.gui.window import EDITOR_PAGE, STRUCTURE_PAGE

    panel = window.structure
    assert panel.structure is None   # analyse seulement à l'affichage
    window.show_page(STRUCTURE_PAGE)
    st = panel.structure
    assert "seldon.core.pipeline" in st.modules and "seldon.core.pipeline" in panel.map.boxes
    assert window.masthead.mode_buttons[STRUCTURE_PAGE].isChecked()

    panel.search.setText("from_config")
    panel.search.returnPressed.emit()
    assert panel.selected == "seldon.core.pipeline"
    assert panel.map.boxes["seldon.core.events"].state == "out"
    assert panel.map.boxes["seldon.__main__"].state == "in"
    item = panel.tree.currentItem()
    assert item is not None and "from_config" in item.text(0)
    panel._item_activated(item)
    assert window.pages.currentIndex() == EDITOR_PAGE
    tab = window.editors.current()
    assert tab.path == st.modules["seldon.core.pipeline"].path
    assert "def from_config" in tab.editor.textCursor().block().text()
