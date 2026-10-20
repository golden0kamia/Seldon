"""Erreurs compréhensibles : messages en français plutôt que des traces Python.

``SeldonError`` porte un message destiné à l'utilisateur et, éventuellement,
une piste pour corriger (``hint``). ``describe`` traduit n'importe quelle
exception (module manquant, serveur injoignable, micro absent…) en une phrase
lisible ; la trace complète n'est affichée qu'avec ``log_level: DEBUG``.
"""

from __future__ import annotations

import logging

# module Python manquant -> extra à installer (pyproject.toml)
EXTRAS = {
    "faster_whisper": "stt",
    "ctranslate2": "stt",
    "onnxruntime": "stt",
    "sounddevice": "stt",
    "numpy": "stt",
    "resemblyzer": "speakers",
    "torch": "speakers",
    "piper": "tts",
    "pyttsx3": "tts",
    "cv2": "camera",
    "httpx": "llm-local",
    "ollama": "llm-local",
    "anthropic": "llm-claude",
    "PySide6": "gui",
    "ruamel": "gui",
}

DEBUG_HINT = "relancer avec « log_level: DEBUG » dans config.yaml pour la trace complète"


class SeldonError(RuntimeError):
    """Erreur attendue (configuration, dépendance, service absent) : affichée sans trace."""

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        return f"{self.message}\n  → {self.hint}" if self.hint else self.message


def install_hint(module: str | None) -> str:
    extra = EXTRAS.get((module or "").split(".")[0])
    return f"installez-le : .venv/bin/pip install -e '.[{extra}]'" if extra else f"installez-le : pip install {module}"


def missing_module(exc: ImportError, what: str) -> SeldonError:
    return SeldonError(f"{what} nécessite le module Python « {exc.name} », non installé", install_hint(exc.name))


def _names(exc: BaseException) -> set[str]:
    return {c.__name__ for c in type(exc).__mro__}


def kind(exc: BaseException) -> str:
    """Catégorie d'une erreur : connection, timeout, auth, quota, audio, missing, file, other."""
    names = _names(exc)
    text = str(exc).lower()
    if "TimeoutException" in names or "APITimeoutError" in names or isinstance(exc, TimeoutError):
        return "timeout"
    if names & {"ConnectError", "APIConnectionError", "ConnectionError", "NetworkError", "URLError"}:
        return "connection"
    if names & {"AuthenticationError", "PermissionDeniedError"} or "api_key" in text:
        return "auth"
    if names & {"RateLimitError", "OverloadedError"}:
        return "quota"
    if "PortAudioError" in names or "portaudio" in text:
        return "audio"
    if isinstance(exc, ImportError):
        return "missing"
    if isinstance(exc, OSError) and getattr(exc, "filename", None):
        return "file"
    return "other"


def describe(exc: BaseException) -> str:
    """Message lisible pour une exception quelconque (sans trace)."""
    if isinstance(exc, SeldonError):
        return str(exc)
    k = kind(exc)
    detail = str(exc).strip()
    if k == "missing" and isinstance(exc, ImportError) and exc.name:
        return f"module Python « {exc.name} » non installé\n  → {install_hint(exc.name)}"
    if k == "timeout":
        return "délai d'attente dépassé (le service ne répond pas assez vite)"
    if k == "connection":
        return f"connexion impossible ({detail or 'service injoignable'})"
    if k == "auth":
        return "authentification refusée : clé API absente ou invalide\n  → définissez ANTHROPIC_API_KEY"
    if k == "quota":
        return "service surchargé ou quota atteint, réessayez dans un instant"
    if k == "audio":
        return (f"périphérique audio indisponible ({detail})\n"
                "  → vérifiez le micro / haut-parleur (python -m sounddevice les liste) et libportaudio2")
    if isinstance(exc, FileNotFoundError) and exc.filename:
        return f"fichier introuvable : {exc.filename}"
    if isinstance(exc, PermissionError) and exc.filename:
        return f"accès refusé : {exc.filename}"
    if isinstance(exc, MemoryError):
        return "mémoire insuffisante (essayez un modèle plus petit)"
    return f"{detail or 'erreur inattendue'} ({type(exc).__name__})"


def spoken(exc: BaseException, stage: str = "brain") -> str:
    """Phrase courte à prononcer quand une prise de parole n'a pas pu être traitée.

    ``stage`` : ``brain`` (le modèle de langage est en cause) ou autre (action,
    commande… : on parle alors d'un service en général).
    """
    if stage == "brain":
        return {
            "connection": "Désolé, je n'arrive pas à joindre mon modèle de langage.",
            "timeout": "Désolé, mon modèle de langage met trop de temps à répondre.",
            "auth": "Désolé, l'accès à mon modèle de langage est refusé : vérifiez la clé API.",
            "quota": "Désolé, mon modèle de langage est surchargé, réessayez dans un instant.",
        }.get(kind(exc), "Désolé, j'ai rencontré une erreur.")
    return {
        "connection": "Désolé, je n'arrive pas à joindre un service dont j'ai besoin.",
        "timeout": "Désolé, un service met trop de temps à répondre.",
        "auth": "Désolé, l'accès à un service m'est refusé.",
        "quota": "Désolé, un service est surchargé, réessayez dans un instant.",
        "audio": "Désolé, j'ai un problème avec le micro ou le haut-parleur.",
        "missing": "Désolé, il me manque un module pour faire cela.",
        "file": "Désolé, je n'ai pas pu accéder à un fichier.",
    }.get(kind(exc), "Désolé, j'ai rencontré une erreur.")


def log_error(logger: logging.Logger, what: str, exc: BaseException) -> None:
    """Journalise ``what : message lisible`` ; la trace Python seulement en DEBUG."""
    expected = isinstance(exc, SeldonError) or kind(exc) != "other"
    hint = "" if expected or logger.isEnabledFor(logging.DEBUG) else f"\n  → {DEBUG_HINT}"
    logger.error("%s : %s%s", what, describe(exc), hint)
    logger.debug("Trace de l'erreur :", exc_info=exc)
