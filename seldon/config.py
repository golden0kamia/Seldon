"""Lecture de ``config.yaml`` et des secrets de ``.env``.

Chaque plugin s'écrit ``{engine: <nom>, options: {...}}`` (ou juste ``<nom>``) ;
les options sont passées telles quelles à son constructeur. Les secrets
(jetons, mots de passe) ne vont jamais dans config.yaml mais dans ``.env``.
"""

from __future__ import annotations

import logging
import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from seldon.errors import SeldonError

log = logging.getLogger(__name__)

SECTIONS = ("log_level", "confirm", "stt", "speakers", "brain", "tts", "memory", "tools", "mcp", "skills")
CONFIRM_MODES = ("console", "always", "never")

_ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def load_secrets(path: str | Path) -> list[str]:
    """Charge un fichier ``.env`` (``CLÉ=valeur``) dans l'environnement ; renvoie les clés lues.

    Une variable déjà définie garde sa valeur. Fichier absent : rien.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    if path.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        log.warning("%s est lisible par d'autres utilisateurs : chmod 600 %s", path, path)
    loaded = []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _ENV_LINE.match(line)
        if not m:   # la ligne n'est pas citée : ce pourrait être un secret mal formé
            log.warning("%s, ligne %d ignorée : attendu CLÉ=valeur", path, n)
            continue
        key, value = m.groups()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


@dataclass
class Plugin:
    """Un plugin à construire : ``engine`` (son nom dans le registre) et ses ``options``."""

    engine: str
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, raw: Any, where: str) -> Plugin:
        if isinstance(raw, str) and raw.strip():
            return cls(raw.strip())
        if not isinstance(raw, dict) or not isinstance(raw.get("engine"), str):
            raise SeldonError(f"Section « {where} » invalide : {raw!r}",
                              "attendu un nom (ex. piper) ou {engine: <nom>, options: {...}}")
        options = raw.get("options") or {}
        if not isinstance(options, dict):
            raise SeldonError(f"Section « {where} » : « options » doit contenir des clés: valeurs")
        return cls(raw["engine"].strip(), dict(options))


@dataclass
class Config:
    stt: Plugin
    speakers: Plugin | None   # None : pas d'identification des voix
    brain: Plugin
    backup: Plugin | None     # cerveau de secours (brain.backup)
    tts: Plugin
    memory: dict[str, Any]    # options de la mémoire, None si désactivée
    tools: list[Plugin]
    mcp: list[Plugin]
    skills: list[str]
    confirm: str = "console"
    log_level: str = "INFO"

    @classmethod
    def load(cls, path: str | Path) -> Config:
        try:
            text = Path(path).read_text(encoding="utf-8")
        except FileNotFoundError:
            raise SeldonError(f"Configuration introuvable : {Path(path).resolve()}",
                              "lancez Seldon depuis son dossier ou passez -c chemin/config.yaml") from None
        return cls.parse(text, str(path))

    @classmethod
    def parse(cls, text: str, source: str = "config.yaml") -> Config:
        try:
            raw = yaml.safe_load(text) or {}
        except yaml.YAMLError as e:
            raise SeldonError(f"{source} : YAML invalide ({e})", "vérifiez l'indentation et les « : »") from None
        if not isinstance(raw, dict):
            raise SeldonError(f"{source} : attendu une liste de clés: valeurs")
        for key in set(map(str, raw)) - set(SECTIONS):
            log.warning("%s : clé « %s » inconnue, ignorée", source, key)

        def listed(key: str) -> list[Any]:
            value = raw.get(key) or []
            if not isinstance(value, list):
                raise SeldonError(f"{source} : « {key} » doit être une liste (lignes « - … »)")
            return value

        brain = raw.get("brain", "llamacpp")
        backup = brain.pop("backup", None) if isinstance(brain, dict) else None
        speakers = raw.get("speakers")
        memory = raw.get("memory") or {}
        if not isinstance(memory, dict):
            raise SeldonError(f"{source} : « memory » doit contenir des clés: valeurs")
        memory = dict(memory)
        if memory.pop("enabled", True) is False:
            memory = None
        confirm = raw.get("confirm", "console")
        if confirm not in CONFIRM_MODES:
            raise SeldonError(f"{source} : confirm « {confirm} » inconnu", f"valeurs : {', '.join(CONFIRM_MODES)}")
        skills = listed("skills")
        if not all(isinstance(s, str) for s in skills):
            raise SeldonError(f"{source} : « skills » liste des noms de dossiers (skills/<nom>/SKILL.md)")
        return cls(
            stt=Plugin.parse(raw.get("stt", "keyboard"), "stt"),
            speakers=None if speakers in (None, "none") or (isinstance(speakers, dict) and speakers.get("engine") == "none")
            else Plugin.parse(speakers, "speakers"),
            brain=Plugin.parse(brain, "brain"),
            backup=Plugin.parse(backup, "brain.backup") if backup else None,
            tts=Plugin.parse(raw.get("tts", "console"), "tts"),
            memory=memory,
            tools=[Plugin.parse(t, f"tools[{i + 1}]") for i, t in enumerate(listed("tools"))],
            mcp=[Plugin.parse(m, f"mcp[{i + 1}]") for i, m in enumerate(listed("mcp"))],
            skills=[s.strip() for s in skills],
            confirm=confirm,
            log_level=str(raw.get("log_level", "INFO")).upper(),
        )
