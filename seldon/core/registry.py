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

import difflib
import importlib
import inspect
from typing import Any, Callable, TypeVar

from seldon.core.errors import SeldonError, missing_module

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
        "feishin": "seldon.actions.feishin",
        "home_assistant": "seldon.actions.home_assistant",
        "desktop": "seldon.actions.desktop",
        "controle": "seldon.actions.control",
    },
}


def register(category: str, name: str) -> Callable[[type[T]], type[T]]:
    def deco(cls: type[T]) -> type[T]:
        _REGISTRY.setdefault(category, {})[name] = cls
        return cls

    return deco


def _import(module: str, what: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as e:
        if e.name and e.name != module and not module.startswith(e.name + "."):
            raise missing_module(e, what) from e   # dépendance du plugin absente
        raise SeldonError(f"{what} : module « {module} » introuvable") from e


def resolve(category: str, name: str) -> type:
    what = f"Le moteur {category} « {name} »"
    if not isinstance(name, str) or not name:
        raise SeldonError(f"Moteur {category} non précisé", f"choisissez parmi : {', '.join(available(category))}")
    if ":" in name:  # plugin externe : "mon_paquet.module:MaClasse"
        module, attr = name.split(":", 1)
        try:
            return getattr(_import(module, what), attr)
        except AttributeError as e:
            raise SeldonError(f"{what} : « {attr} » n'existe pas dans le module « {module} »") from e
    if name not in _REGISTRY.get(category, {}):
        module = _BUILTINS.get(category, {}).get(name)
        if module is None:
            known = available(category)
            close = difflib.get_close_matches(name, known, n=1)
            raise SeldonError(
                f"Aucun moteur « {name} » dans la catégorie « {category} »"
                + (f" (vouliez-vous dire « {close[0]} » ?)" if close else ""),
                f"moteurs disponibles : {', '.join(known)}",
            )
        _import(module, what)
    return _REGISTRY[category][name]


def check_options(cls: type, category: str, name: str, options: dict[str, Any]) -> None:
    """Vérifie les options avant construction : une faute de frappe donne un message clair."""
    try:
        params = inspect.signature(cls).parameters
    except (TypeError, ValueError):
        return
    if any(p.kind is p.VAR_KEYWORD for p in params.values()):
        return
    accepted = [n for n, p in params.items() if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
    for option in options:
        if option not in accepted:
            close = difflib.get_close_matches(option, accepted, n=1)
            raise SeldonError(
                f"Option inconnue « {option} » pour {category} « {name} »"
                + (f" (vouliez-vous dire « {close[0]} » ?)" if close else ""),
                f"options acceptées : {', '.join(accepted) or 'aucune'}",
            )
    missing = [n for n, p in params.items()
               if p.default is p.empty and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) and n not in options]
    if missing:
        raise SeldonError(f"Option obligatoire manquante pour {category} « {name} » : {', '.join(missing)}")
    for option, value in options.items():
        expected = _expected(params[option])
        if expected is None or (value is None and expected[1]):
            continue
        kind, _ = expected
        ok = isinstance(value, bool) if kind == "bool" else isinstance(value, (int, float)) and not isinstance(value, bool)
        if not ok:
            raise SeldonError(
                f"Option « {option} » de {category} « {name} » : "
                f"{'true ou false' if kind == 'bool' else 'un nombre'} attendu, pas {value!r}",
                "sans guillemets dans config.yaml (ex. 700, 0.5, true)",
            )


def _expected(param: inspect.Parameter) -> tuple[str, bool] | None:
    """(« bool » ou « number », None accepté ?) attendu pour un paramètre, d'après sa valeur par
    défaut et son annotation ; None si le type n'est pas simple (chaîne, liste, union mixte…)."""
    ann = param.annotation
    ann = "" if ann is param.empty else ann if isinstance(ann, str) else getattr(ann, "__name__", str(ann))
    types = {t.strip() for t in ann.split("|")} - {""}
    nullable = "None" in types or param.default is None
    types.discard("None")
    default = param.default
    if types == {"bool"} or (not types and isinstance(default, bool)):
        return "bool", nullable
    if (types and types <= {"int", "float"}) or (not types and isinstance(default, (int, float))
                                                 and not isinstance(default, bool)):
        return "number", nullable
    return None


def create(category: str, name: str, **options: Any) -> Any:
    cls = resolve(category, name)
    check_options(cls, category, name, options)
    try:
        return cls(**options)
    except SeldonError:
        raise
    except (TypeError, ValueError) as e:
        raise SeldonError(f"Option invalide pour {category} « {name} » : {e}") from e


def available(category: str) -> list[str]:
    return sorted(set(_BUILTINS.get(category, {})) | set(_REGISTRY.get(category, {})))
