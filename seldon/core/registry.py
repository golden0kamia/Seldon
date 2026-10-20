"""Registre de plugins générique.

Chaque étage (stt, speakers, brain, tts, actions) a sa propre catégorie.
Une implémentation s'enregistre avec un décorateur :

    @register("tts", "piper")
    class PiperTTS(TTSEngine): ...

et se construit depuis la config avec ``create("tts", "piper", **options)``.
Les modules sont importés paresseusement (voir ``_BUILTINS``) pour ne pas
charger torch/whisper/opencv quand ils ne sont pas utilisés.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, TypeVar

T = TypeVar("T")

_REGISTRY: dict[str, dict[str, type]] = {}

# catégorie -> nom -> module à importer pour déclencher l'enregistrement.
# Ajouter une implémentation = ajouter une ligne ici (ou passer "module:Classe" dans la config).
_BUILTINS: dict[str, dict[str, str]] = {
    "stt": {
        "text": "seldon.stt.text",
        "whisper": "seldon.stt.whisper",
    },
    "speakers": {
        "none": "seldon.speakers.base",
        "resemblyzer": "seldon.speakers.resemblyzer_id",
    },
    "brain": {
        "llamacpp": "seldon.brain.llamacpp",
        "ollama": "seldon.brain.ollama",
        "claude": "seldon.brain.claude",
    },
    "tts": {
        "console": "seldon.tts.console",
        "piper": "seldon.tts.piper",
        "pyttsx3": "seldon.tts.pyttsx3_tts",
    },
    "actions": {
        "speak": "seldon.actions.speak",
        "terminal": "seldon.actions.terminal",
        "camera": "seldon.actions.camera",
        "conversations": "seldon.actions.conversations",
        "voix": "seldon.actions.voices",
    },
}


def register(category: str, name: str) -> Callable[[type[T]], type[T]]:
    def deco(cls: type[T]) -> type[T]:
        _REGISTRY.setdefault(category, {})[name] = cls
        return cls

    return deco


def resolve(category: str, name: str) -> type:
    if ":" in name:  # plugin externe : "mon_paquet.module:MaClasse"
        module, attr = name.split(":", 1)
        return getattr(importlib.import_module(module), attr)
    if name not in _REGISTRY.get(category, {}):
        module = _BUILTINS.get(category, {}).get(name)
        if module is None:
            raise KeyError(f"Aucun plugin '{name}' dans la catégorie '{category}'")
        importlib.import_module(module)
    return _REGISTRY[category][name]


def create(category: str, name: str, **options: Any) -> Any:
    return resolve(category, name)(**options)


def available(category: str) -> list[str]:
    return sorted(set(_BUILTINS.get(category, {})) | set(_REGISTRY.get(category, {})))
