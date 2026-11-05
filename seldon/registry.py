"""La liste de tous les plugins de Seldon : nom (dans config.yaml) -> classe.

Ajouter un plugin = écrire sa classe puis ajouter une ligne ici. Les modules ne
sont importés que si le plugin est utilisé (pas de torch ni d'opencv chargés
pour rien). Un plugin externe s'écrit directement ``paquet.module:Classe``
dans config.yaml.

Les options d'un plugin sont les paramètres de son ``__init__`` : l'interface
les lit sans importer le module, elles doivent donc avoir des valeurs par
défaut littérales (``timeout: int = 30``).
"""

from __future__ import annotations

import difflib
import importlib
import inspect
from typing import Any

from seldon.errors import SeldonError, missing_module

PLUGINS: dict[str, dict[str, str]] = {
    "stt": {
        "whisper": "seldon.ear.whisper:Whisper",
        "keyboard": "seldon.ear.keyboard:Keyboard",
    },
    "speakers": {
        "resemblyzer": "seldon.ear.speakers:Resemblyzer",
    },
    "brain": {
        "llamacpp": "seldon.brain.llamacpp:LlamaCpp",
        "claude": "seldon.brain.claude:Claude",
    },
    "tts": {
        "piper": "seldon.voice.piper:Piper",
        "console": "seldon.voice.console:Console",
    },
    "tools": {
        "speak": "seldon.tools.speak:Speak",
        "terminal": "seldon.tools.terminal:Terminal",
        "camera": "seldon.tools.camera:Camera",
        "feishin": "seldon.tools.feishin:Feishin",
        "desktop": "seldon.tools.desktop:Desktop",
        "nextcloud": "seldon.tools.nextcloud:Nextcloud",
        "web": "seldon.tools.web:Web",
    },
    "mcp": {
        "http": "seldon.mcp:MCPServer",
        "home_assistant": "seldon.mcp:HomeAssistant",
    },
}


def resolve(category: str, engine: str) -> type:
    """Classe du plugin ``engine`` de ``category`` (importe son module)."""
    target = engine if ":" in engine else PLUGINS.get(category, {}).get(engine)
    if target is None:
        known = sorted(PLUGINS.get(category, {}))
        close = difflib.get_close_matches(engine, known, n=1)
        raise SeldonError(f"Aucun plugin {category} « {engine} »" + (f" (vouliez-vous dire « {close[0]} » ?)" if close else ""),
                          f"disponibles : {', '.join(known)}")
    module, _, cls = target.partition(":")
    try:
        return getattr(importlib.import_module(module), cls)
    except ImportError as e:
        raise missing_module(e, f"Le plugin {category} « {engine} »") from e
    except AttributeError as e:
        raise SeldonError(f"Classe « {cls} » absente de {module}") from e


def create(category: str, engine: str, options: dict[str, Any]) -> Any:
    """Construit le plugin avec ses options ; une option inconnue donne un message clair."""
    cls = resolve(category, engine)
    accepted = [n for n in inspect.signature(cls).parameters if n not in ("self", "args", "kwargs")]
    for option in options:
        if option not in accepted:
            close = difflib.get_close_matches(option, accepted, n=1)
            raise SeldonError(f"Option inconnue « {option} » pour {category} « {engine} »"
                              + (f" (vouliez-vous dire « {close[0]} » ?)" if close else ""),
                              f"options acceptées : {', '.join(accepted) or 'aucune'}")
    try:
        return cls(**options)
    except (TypeError, ValueError) as e:
        raise SeldonError(f"Option invalide pour {category} « {engine} » : {e}") from e
