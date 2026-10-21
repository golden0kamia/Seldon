"""Balises que la page Journal de l'interface repère dans la sortie de Seldon.

L'interface lance Seldon avec un jeton aléatoire (``SELDON_GUI_TOKEN``, nouveau à
chaque lancement) ; Seldon le retire aussitôt de son environnement (aucune
commande, aucun sous-processus ne le voit) et l'ajoute à ses balises :
« [confirmation 3f…] question ». Une ligne qui en imite une (réponse du modèle,
sortie d'une commande, log qui cite une phrase) n'a pas le jeton : l'interface
l'ignore, et ne peut donc ni afficher une fausse question, ni faire croire le
micro coupé.
"""

from __future__ import annotations

import os

TOKEN_ENV = "SELDON_GUI_TOKEN"
CONFIRM = "confirmation"   # question posée par seldon.core.pipeline.console_confirm
MIC = "micro"              # état du micro annoncé par seldon.stt.whisper

_token = ""


def take_token() -> None:
    """Lit puis retire le jeton de l'environnement (au démarrage de Seldon)."""
    global _token
    _token = os.environ.pop(TOKEN_ENV, "")


def tag(name: str, token: str | None = None) -> str:
    """Balise ``[name jeton]`` ; ``[name]`` sans jeton (Seldon lancé hors de l'interface)."""
    token = _token if token is None else token
    return f"[{name} {token}]" if token else f"[{name}]"
