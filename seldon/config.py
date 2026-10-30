"""Chargement de la configuration YAML.

Chaque étage est décrit par ``{engine: <nom>, options: {...}}`` ; les options
sont passées telles quelles au constructeur du plugin. Les actions du cerveau
se répartissent en trois listes : ``tools`` et ``mcp`` (même format, ``engine`` =
nom dans le registre) et ``skills`` (noms ou chemins de dossiers de skills).
L'ancienne liste unique ``actions`` reste lue et répartie entre tools et mcp.
"""

from __future__ import annotations

import difflib
import logging
import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from seldon.core import registry
from seldon.core.errors import SeldonError

log = logging.getLogger(__name__)

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
CONFIRM_MODES = ("console", "always", "never")
TOP_LEVEL = ("stt", "speakers", "brain", "tts", "tools", "skills", "mcp", "actions", "memory", "commands", "confirm", "follow_up_s",
             "acknowledge", "log_level")


_ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def load_secrets(path: str | Path) -> list[str]:
    """Charge un fichier ``.env`` (``CLÉ=valeur``) dans l'environnement ; renvoie les clés définies.

    Les identifiants (HA_TOKEN, FEISHIN_PASSWORD, ANTHROPIC_API_KEY…) restent ainsi
    hors de config.yaml. Une variable déjà exportée garde sa valeur. Fichier absent : rien.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
        mode = path.stat().st_mode
    except FileNotFoundError:
        return []
    except (OSError, UnicodeDecodeError) as e:
        raise SeldonError(f"Impossible de lire {path} : {e}") from None
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        log.warning("%s est lisible par d'autres utilisateurs : chmod 600 %s", path, path)
    loaded = []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _ENV_LINE.match(line)
        if not m:   # pas la ligne dans le message : ce pourrait être un secret mal formé
            log.warning("%s, ligne %d ignorée : attendu CLÉ=valeur", path, n)
            continue
        key, value = m.groups()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:   # commentaire en fin de ligne, hors guillemets
            value = value.split(" #", 1)[0].rstrip()
        if key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


@dataclass
class StageConfig:
    engine: str
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, raw: Any, where: str = "étage") -> "StageConfig":
        if isinstance(raw, str) and raw.strip():
            return cls(engine=raw.strip())
        if not isinstance(raw, dict):
            raise SeldonError(
                f"Section « {where} » invalide : {raw!r}",
                "attendu un nom de moteur (ex. « piper ») ou {engine: <nom>, options: {...}}",
            )
        engine = raw.get("engine")
        if not isinstance(engine, str) or not engine.strip():
            raise SeldonError(f"Section « {where} » : clé « engine » manquante ou vide",
                              "ex. engine: piper")
        options = raw.get("options") or {}
        if not isinstance(options, dict):
            raise SeldonError(f"Section « {where} » : « options » doit être une liste de clés: valeurs, pas {options!r}")
        unknown = set(raw) - {"engine", "options"}
        if unknown:
            raise SeldonError(f"Section « {where} » : clé(s) inattendue(s) {', '.join(sorted(map(str, unknown)))}",
                              "les réglages du moteur vont sous « options: »")
        return cls(engine=engine.strip(), options=dict(options))


@dataclass
class Config:
    stt: StageConfig
    speakers: StageConfig
    brain: StageConfig
    tts: StageConfig
    tools: list[StageConfig] = field(default_factory=list)   # actions en Python (seldon/tools)
    mcp: list[StageConfig] = field(default_factory=list)     # serveurs MCP (seldon/mcp)
    skills: list[str] = field(default_factory=list)          # dossiers skills/<nom>/SKILL.md
    brain_backup: StageConfig | None = None   # cerveau de secours (brain.backup)
    memory: dict[str, Any] = field(default_factory=dict)   # options de ConversationMemory + enabled
    commands: bool = True       # commandes locales : exit, reset, aide…
    confirm: str = "console"   # "console" (demande au clavier), "always" ou "never"
    follow_up_s: float = 20     # avec mots d'éveil : écoute sans mot d'éveil après un échange
    acknowledge: bool = True    # annonce qu'une action demandée est comprise et en cours
    log_level: str = "INFO"

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        path = Path(path)
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise SeldonError(f"Fichier de configuration introuvable : {path.resolve()}",
                              "lancez Seldon depuis son dossier ou indiquez le fichier avec -c chemin/config.yaml") from None
        except (OSError, UnicodeDecodeError) as e:
            raise SeldonError(f"Impossible de lire {path} : {e}") from None
        return cls.parse(text, str(path))

    @classmethod
    def parse(cls, text: str, source: str = "config.yaml") -> "Config":
        try:
            raw = yaml.safe_load(text) or {}
        except yaml.YAMLError as e:
            mark = getattr(e, "problem_mark", None)
            where = f" (ligne {mark.line + 1}, colonne {mark.column + 1})" if mark else ""
            problem = getattr(e, "problem", None) or str(e)
            raise SeldonError(f"{source} : YAML invalide{where} : {problem}",
                              "vérifiez l'indentation (espaces, pas de tabulations) et les « : »") from None
        if not isinstance(raw, dict):
            raise SeldonError(f"{source} : la configuration doit être une liste de clés: valeurs")

        lists: dict[str, list[Any]] = {}
        for key in ("tools", "mcp", "skills", "actions"):
            value = raw.get(key) or []
            if not isinstance(value, list):
                raise SeldonError(f"{source} : « {key} » doit être une liste (lignes commençant par « - »)")
            lists[key] = value
        tools = [StageConfig.parse(a, f"tools[{i + 1}]") for i, a in enumerate(lists["tools"])]
        mcp = [StageConfig.parse(a, f"mcp[{i + 1}]") for i, a in enumerate(lists["mcp"])]
        for i, a in enumerate(lists["actions"]):   # ancien format : une seule liste
            stage = StageConfig.parse(a, f"actions[{i + 1}]")
            (mcp if registry.category_of(stage.engine) == "mcp" else tools).append(stage)
        if provided := [s.engine for s in tools if s.engine in registry.PLUGIN_TOOLS]:   # anciennes configs
            tools = [s for s in tools if s.engine not in registry.PLUGIN_TOOLS]
            log.info("%s : %s ignoré(s) dans « tools » : fourni(s) par %s", source, ", ".join(provided),
                     ", ".join(sorted({registry.PLUGIN_TOOLS[e] for e in provided})))
        if lists["actions"]:
            log.info("%s : « actions » est lu, mais remplacé par « tools » et « mcp »", source)
        if "tools" not in raw and "actions" not in raw:
            tools = [StageConfig("speak")]
        if not all(isinstance(s, str) and s.strip() for s in lists["skills"]):
            raise SeldonError(f"{source} : « skills » doit lister des noms de dossiers de skills",
                              "ex. « - bilan-du-matin » pour skills/bilan-du-matin/SKILL.md")
        memory = raw.get("memory") or {}
        if not isinstance(memory, dict):
            raise SeldonError(f"{source} : « memory » doit contenir des clés: valeurs (enabled, dir…)")
        confirm = raw.get("confirm", "console")
        if confirm not in CONFIRM_MODES:
            raise SeldonError(f"{source} : confirm « {confirm} » inconnu", f"valeurs possibles : {', '.join(CONFIRM_MODES)}")
        log_level = str(raw.get("log_level", "INFO")).upper()
        if log_level not in LOG_LEVELS:
            raise SeldonError(f"{source} : log_level « {log_level} » inconnu", f"valeurs possibles : {', '.join(LOG_LEVELS)}")
        follow_up_s = raw.get("follow_up_s", 20)
        if isinstance(follow_up_s, bool) or not isinstance(follow_up_s, (int, float)) or follow_up_s < 0:
            raise SeldonError(f"{source} : follow_up_s doit être un nombre de secondes, pas {follow_up_s!r}")
        for key in ("commands", "acknowledge"):   # « non » ou "false" entre guillemets vaudraient vrai
            if not isinstance(raw.get(key, True), bool):
                raise SeldonError(f"{source} : {key} doit valoir true ou false, pas {raw[key]!r}")
        brain = raw.get("brain", "llamacpp")
        backup = None
        if isinstance(brain, dict) and "backup" in brain:
            brain = dict(brain)
            backup = brain.pop("backup")
            backup = StageConfig.parse(backup, "brain.backup") if backup else None
        for key in sorted(set(map(str, raw)) - set(TOP_LEVEL)):   # faute de frappe : réglage ignoré sinon
            close = difflib.get_close_matches(key, TOP_LEVEL, n=1)
            log.warning("%s : clé « %s » inconnue, ignorée%s", source, key,
                        f" (vouliez-vous dire « {close[0]} » ?)" if close else "")
        return cls(
            stt=StageConfig.parse(raw.get("stt", "text"), "stt"),
            speakers=StageConfig.parse(raw.get("speakers", "none"), "speakers"),
            brain=StageConfig.parse(brain, "brain"),
            brain_backup=backup,
            tts=StageConfig.parse(raw.get("tts", "console"), "tts"),
            tools=tools,
            mcp=mcp,
            skills=[s.strip() for s in lists["skills"]],
            memory=dict(memory),
            commands=bool(raw.get("commands", True)),
            confirm=confirm,
            follow_up_s=follow_up_s,
            acknowledge=bool(raw.get("acknowledge", True)),
            log_level=log_level,
        )
