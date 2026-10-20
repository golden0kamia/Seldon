"""Messages d'erreur compréhensibles (configuration, plugins, démarrage)."""

from __future__ import annotations

import logging

import pytest

from seldon.config import Config
from seldon.core import registry
from seldon.core.errors import SeldonError, describe, log_error, spoken


def test_missing_config_file(tmp_path):
    with pytest.raises(SeldonError, match="introuvable") as e:
        Config.load(tmp_path / "absent.yaml")
    assert "-c" in str(e.value)


def test_yaml_syntax_error_gives_line():
    with pytest.raises(SeldonError, match="ligne 2"):
        Config.parse("stt: text\n  brain: x\n")


@pytest.mark.parametrize("text, match", [
    ("brain: {options: {model: x}}", "engine"),
    ("tts: 3", "tts"),
    ("tts: {engine: piper, voice: x}", "options:"),
    ("actions: speak", "liste"),
    ("confirm: peut-être", "console, always, never"),
    ("log_level: bavard", "DEBUG"),
    ("follow_up_s: longtemps", "follow_up_s"),
])
def test_invalid_config(text, match):
    with pytest.raises(SeldonError, match=match):
        Config.parse(text)


def test_unknown_engine_suggests_close_name():
    with pytest.raises(SeldonError, match="pipr|piper") as e:
        registry.resolve("tts", "pipr")
    assert "vouliez-vous dire « piper »" in str(e.value)


def test_unknown_option_suggests_close_name():
    with pytest.raises(SeldonError) as e:
        registry.create("tts", "console", prefx="> ")
    assert "prefx" in str(e.value) and "prefix" in str(e.value)


def test_missing_dependency_names_extra():
    registry._BUILTINS["tts"]["fantome"] = "tests._module_fantome"
    try:
        with pytest.raises(SeldonError, match="introuvable"):
            registry.resolve("tts", "fantome")
    finally:
        del registry._BUILTINS["tts"]["fantome"]
    err = ModuleNotFoundError("No module named 'piper'", name="piper")
    assert ".[tts]" in describe(err)


def test_connection_errors_are_explained():
    assert "connexion impossible" in describe(ConnectionRefusedError("refusé"))
    assert "joindre" in spoken(ConnectionRefusedError())
    assert spoken(ValueError()) == "Désolé, j'ai rencontré une erreur."


def test_log_error_hides_traceback(caplog):
    caplog.set_level(logging.INFO)
    try:
        raise ValueError("valeur bizarre")
    except ValueError as e:
        log_error(logging.getLogger("t"), "Échec", e)
    (record,) = caplog.records
    assert record.levelno == logging.ERROR and record.exc_info is None
    assert "valeur bizarre" in record.getMessage() and "DEBUG" in record.getMessage()


async def test_start_failure_names_the_stage():
    from seldon.core.pipeline import Assistant

    cfg = Config.parse("stt: text\ntts: console\nbrain: {engine: llamacpp, options: {base_url: 'http://127.0.0.1:9'}}\n"
                       "memory: {enabled: false}\nactions: [speak]\n")
    a = Assistant.from_config(cfg)
    with pytest.raises(SeldonError, match="cerveau") as e:
        await a.run()
    assert "llama serve" in str(e.value)


def test_main_prints_message_without_traceback(tmp_path, caplog, monkeypatch):
    import sys

    from seldon.__main__ import main

    monkeypatch.setattr(sys, "argv", ["seldon", "-c", str(tmp_path / "absent.yaml")])
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 1
    (record,) = caplog.records
    assert "introuvable" in record.getMessage() and record.exc_info is None
