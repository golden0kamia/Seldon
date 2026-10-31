"""Journal de Seldon : la conversation, les réflexions du cerveau et ses actions.

Chaque catégorie a son logger, que la page Journal de l'interface reconnaît à son nom :

- ``seldon.conversation.locuteur`` / ``seldon.conversation.seldon`` (INFO) : ce qui est
  entendu et ce que Seldon dit. Toujours journalisé, quel que soit ``log_level`` ;
- ``seldon.reflexion`` (INFO) : raisonnement du modèle et texte écrit entre deux outils ;
- ``seldon.action`` (DEBUG) : appels d'outils et leurs résultats.

Un texte sur plusieurs lignes est écrit en une seule entrée, lignes suivantes indentées :
la page Journal les rattache à la première, et un texte ne peut pas y imiter une autre
ligne de log (niveau, balise).
"""

from __future__ import annotations

import json
import logging
from typing import Any

HEARD = "seldon.conversation.locuteur"
SAID = "seldon.conversation.seldon"
THOUGHT = "seldon.reflexion"
ACTION = "seldon.action"

MAX_ACTION_CHARS = 300   # paramètres et résultats d'action tronqués au-delà

logging.getLogger("seldon.conversation").setLevel(logging.INFO)   # la conversation reste visible
_heard = logging.getLogger(HEARD)
_said = logging.getLogger(SAID)
_thought = logging.getLogger(THOUGHT)
_action = logging.getLogger(ACTION)


def _lines(text: str) -> str:
    """Texte sur une entrée : lignes vides retirées, suivantes indentées."""
    return "\n  ".join(line.rstrip() for line in str(text).strip().splitlines() if line.strip())


def _short(text: str) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= MAX_ACTION_CHARS else text[:MAX_ACTION_CHARS - 1] + "…"


def heard(speaker: str | None, text: str) -> None:
    """Phrase entendue (ou tapée) : « Nom › texte »."""
    if text := _lines(text):
        _heard.info("%s › %s", speaker or "inconnu", text)


def said(text: str) -> None:
    """Ce que Seldon dit (réponse, accusé de réception, annonce, message d'erreur)."""
    if text := _lines(text):
        _said.info("Seldon › %s", text)


def thought(text: str | None) -> None:
    """Réflexion du cerveau, jamais prononcée."""
    if text and (text := _lines(text)):
        _thought.info("%s", text)


def action(name: str, params: dict[str, Any]) -> None:
    if _action.isEnabledFor(logging.DEBUG):
        _action.debug("%s %s", name, _short(json.dumps(params, ensure_ascii=False, default=str)))


def action_result(name: str, is_error: bool, text: str) -> None:
    if _action.isEnabledFor(logging.DEBUG):
        _action.debug("%s → %s%s", name, "erreur : " if is_error else "", _short(text) or "ok")
