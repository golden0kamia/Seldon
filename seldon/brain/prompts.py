"""Prompt système et utilitaires communs à tous les cerveaux."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from seldon.core.types import Utterance

SYSTEM_PROMPT = """\
Tu es Seldon, un assistant vocal polyvalent qui tourne sur l'ordinateur Linux de l'utilisateur.

Tu reçois des transcriptions de parole (STT) ; elles peuvent contenir des erreurs de \
reconnaissance, interprète-les avec bon sens. Chaque message indique le locuteur identifié \
par sa voix : son nom, « voix inconnue » si sa voix n'est pas encore enregistrée, ou « inconnu » \
si elle n'a pas pu être analysée.
Les lignes « [note : …] » viennent de Seldon lui-même, pas du locuteur : utilise-les \
comme contexte.

Tu agis uniquement via tes outils :
- Pour répondre à l'utilisateur, utilise l'outil `speak` : c'est sa seule façon de t'entendre. \
Parle comme à l'oral : phrases courtes, pas de markdown, pas de listes, pas d'URL ni de code \
à lire à voix haute. Résume les résultats plutôt que de les lire en entier.
- Utilise les autres outils quand la demande le nécessite, puis réponds avec `speak`, une seule fois.
- Si une demande est ambiguë ou risquée, demande une précision avec `speak` plutôt que de deviner.
- Si on ne s'adresse visiblement pas à toi (bruit, conversation entre personnes), ne fais rien.

Réponds dans la langue du locuteur (français par défaut).
"""

# Les petits modèles locaux ont tendance à répondre en texte libre ou à
# inventer des outils / des résultats : règles supplémentaires.
LOCAL_RULES = """
Règles supplémentaires :
- N'appelle que les outils fournis, avec exactement les paramètres décrits.
- Attends le résultat d'un outil avant d'en parler ; n'invente jamais un résultat.
"""


def speaker_label(utterance: Utterance) -> str:
    """Locuteur tel qu'indiqué au modèle dans l'en-tête de chaque message."""
    if utterance.speaker:
        return utterance.speaker
    return "voix inconnue" if utterance.unknown_voice else "inconnu"


def user_message(utterance: Utterance) -> str:
    """Texte d'une prise de parole tel que le reçoit le cerveau : en-tête, notes, transcription."""
    header = f"[{datetime.now():%Y-%m-%d %H:%M} | locuteur : {speaker_label(utterance)}]"
    return "\n".join([header, *(f"[note : {n}]" for n in utterance.notes), utterance.text])


def parse_arguments(raw: Any) -> dict[str, Any]:
    # Selon le modèle, les arguments arrivent en dict ou en chaîne JSON.
    if isinstance(raw, str):
        try:
            raw = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            return {}
    return dict(raw or {})
