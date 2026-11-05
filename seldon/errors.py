"""Erreurs lisibles : un message en français plutôt qu'une trace Python.

``SeldonError`` porte un message et, si possible, une piste pour corriger
(``hint``). ``describe`` traduit n'importe quelle exception en une phrase ; la
trace complète n'est journalisée qu'avec ``log_level: DEBUG``.
"""

from __future__ import annotations

import logging
import re

# module Python manquant -> extra de pyproject.toml à installer
EXTRAS = {
    "faster_whisper": "stt", "sounddevice": "stt", "numpy": "stt",
    "resemblyzer": "speakers", "torch": "speakers",
    "piper": "tts",
    "anthropic": "llm-claude", "claude_agent_sdk": "llm-claude", "httpx": "llm-local",
    "cv2": "camera", "websockets": "music", "playwright": "web",
    "PySide6": "gui", "ruamel": "gui",
}

# Secrets qu'un message d'erreur peut contenir (URL, en-têtes, réponses) : masqués.
_SECRETS = [
    (re.compile(r"\b(sk-(?:[a-z]+\d*-)?)[A-Za-z0-9_-]{8,}"), r"\1***"),
    (re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}"), r"\1 ***"),
    (re.compile(r"(?i)\b((?:x-)?api[_-]?key|access[_-]?token|token|secret|password|passwd|authorization)"
                r"(\"?'?\s*[:=]\s*\"?'?)(?!\*\*\*|bearer\b|basic\b)[^\s\"'&,;}]+"), r"\1\2***"),
    (re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^/\s:@]+:[^/\s@]+@"), r"\1***@"),
]


def redact(text: str) -> str:
    """Masque les clés, jetons et mots de passe présents dans ``text``."""
    for pattern, replacement in _SECRETS:
        text = pattern.sub(replacement, text)
    return text


class SeldonError(RuntimeError):
    """Erreur attendue (configuration, dépendance, service absent) : affichée sans trace."""

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        return f"{self.message}\n  → {self.hint}" if self.hint else self.message


def missing_module(exc: ImportError, what: str) -> SeldonError:
    extra = EXTRAS.get((exc.name or "").split(".")[0])
    hint = f".venv/bin/pip install -e '.[{extra}]'" if extra else f"pip install {exc.name}"
    return SeldonError(f"{what} nécessite le module Python « {exc.name} »", hint)


def describe(exc: BaseException) -> str:
    """Message lisible pour n'importe quelle exception."""
    if isinstance(exc, SeldonError):
        return redact(str(exc))
    names = {c.__name__ for c in type(exc).__mro__}
    detail = redact(" ".join(str(exc).split()))
    if isinstance(exc, ImportError) and exc.name:
        return str(missing_module(exc, "Seldon"))
    if isinstance(exc, TimeoutError) or names & {"TimeoutException", "APITimeoutError"}:
        return "délai d'attente dépassé (le service ne répond pas assez vite)"
    if names & {"ConnectError", "APIConnectionError", "ConnectionError"}:
        return f"connexion impossible ({detail or 'service injoignable'})"
    if "PortAudioError" in names:
        return f"micro ou haut-parleur indisponible ({detail})"
    if isinstance(exc, FileNotFoundError) and exc.filename:
        return f"fichier introuvable : {exc.filename}"
    return f"{detail or 'erreur inattendue'} ({type(exc).__name__})"


def log_error(log: logging.Logger, what: str, exc: BaseException) -> None:
    """Journalise ``what : message lisible`` ; la trace seulement en DEBUG."""
    log.error("%s : %s", what, describe(exc))
    log.debug("Trace :", exc_info=exc)
