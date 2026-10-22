"""Tests du chargement des identifiants depuis .env."""

from __future__ import annotations

import os

from seldon.config import load_secrets


def test_load_secrets(tmp_path, monkeypatch):
    for key in ("HA_TOKEN", "FEISHIN_USERNAME", "FEISHIN_PASSWORD", "QUOTED", "EMPTY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("FEISHIN_USERNAME", "déjà-exporté")
    env = tmp_path / ".env"
    env.write_text("# commentaire\n\nHA_TOKEN=abc.def  # fin de ligne\nexport FEISHIN_PASSWORD='p a#ss'\n"
                   "FEISHIN_USERNAME=fichier\nQUOTED=\"x\"\nEMPTY=\nligne invalide\n", encoding="utf-8")
    env.chmod(0o600)
    assert load_secrets(env) == ["HA_TOKEN", "FEISHIN_PASSWORD", "QUOTED", "EMPTY"]
    assert os.environ["HA_TOKEN"] == "abc.def"
    assert os.environ["FEISHIN_PASSWORD"] == "p a#ss"
    assert os.environ["FEISHIN_USERNAME"] == "déjà-exporté"   # l'environnement l'emporte
    assert os.environ["QUOTED"] == "x"
    assert os.environ["EMPTY"] == ""


def test_load_secrets_missing_and_permissions(tmp_path, caplog):
    assert load_secrets(tmp_path / ".env") == []
    env = tmp_path / ".env"
    env.write_text("", encoding="utf-8")
    env.chmod(0o644)
    load_secrets(env)
    assert "chmod 600" in caplog.text
