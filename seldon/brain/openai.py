"""Cerveau OpenAI (GPT) via l'API Chat Completions.

Même boucle que le cerveau llama.cpp (API compatible OpenAI, appels d'outils,
réponse lue en flux) : seuls l'adresse, la clé et quelques paramètres changent.
La clé vient de ``OPENAI_API_KEY`` (fichier .env) ; ``api_key`` dans la config
reste possible mais déconseillé.

``base_url`` permet aussi de viser un autre fournisseur compatible OpenAI
(OpenRouter, Mistral, Groq…) avec sa propre clé.
"""

from __future__ import annotations

import os
from typing import Any

from seldon.actions.base import Action
from seldon.brain.llamacpp import LlamaCppBrain
from seldon.core.errors import SeldonError
from seldon.core.registry import register


@register("brain", "openai")
class OpenAIBrain(LlamaCppBrain):
    server = "serveur OpenAI"
    key_hint = "vérifiez OPENAI_API_KEY dans le fichier .env"
    max_tokens_field = "max_completion_tokens"

    def __init__(
        self,
        model: str = "gpt-5.6-luna",        # liste : platform.openai.com/docs/models
        reasoning_effort: str | None = "low",   # minimal | low | medium | high ; null = défaut du modèle
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,         # défaut : $OPENAI_API_KEY
        vision: bool = True,                # le modèle lit les images (caméra)
        temperature: float | None = None,   # None : défaut du modèle (certains n'acceptent que celui-là)
        max_tokens: int = 4096,
        max_steps: int = 10,
        max_history_messages: int = 200,    # au-delà, les plus anciens échanges sont oubliés
        max_history_chars: int = 60000,     # idem en taille ; les anciens résultats d'outils sont d'abord abrégés
        keep_images: int = 1,               # captures gardées dans l'historique (les plus récentes)
        stream: bool = True,                # dit chaque phrase de la réponse dès qu'elle est écrite
        extra_instructions: str = "",
        timeout_s: float = 120,
    ) -> None:
        super().__init__(
            base_url=base_url, model=model, api_key=api_key or os.environ.get("OPENAI_API_KEY") or None,
            vision=vision, thinking=None, temperature=temperature, max_tokens=max_tokens, max_steps=max_steps,
            max_history_messages=max_history_messages, max_history_chars=max_history_chars,
            keep_images=keep_images, stream=stream, extra_instructions=extra_instructions, timeout_s=timeout_s,
            server_command=None,
        )
        self.reasoning_effort = reasoning_effort

    async def start(self) -> None:
        if not self.api_key:
            raise SeldonError("Clé API OpenAI absente", "définissez OPENAI_API_KEY dans le fichier .env")
        if not self.model:
            raise SeldonError("Modèle OpenAI non précisé", "ex. brain.options.model: gpt-5.6-luna")
        await super().start()

    def _body(self, tools: list[dict[str, Any]], messages: list[dict[str, Any]] | None = None,
              max_tokens: int | None = None) -> dict[str, Any]:
        body = super()._body(tools, messages, max_tokens)
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        return body

    async def warmup(self, actions: list[Action]) -> None:
        """Rien : le cache du prompt d'OpenAI est automatique, une requête de préparation serait facturée."""
