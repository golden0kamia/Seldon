"""Cerveau local via Ollama (modèles llama.cpp/GGUF : Qwen, Llama, Mistral…).

Même boucle que le cerveau Claude : chaque action est exposée comme un outil
(format « function » d'Ollama), le modèle choisit des outils, on les exécute,
les résultats lui sont renvoyés (messages ``role: tool``) jusqu'à ce qu'il
réponde sans appeler d'outil.

Prérequis : ``ollama serve`` lancé et le modèle téléchargé (``ollama pull <modèle>``).
Le modèle doit annoncer la capacité « tools » (``ollama show <modèle>``).
"""

from __future__ import annotations

import base64
import logging
from typing import Any

import ollama

from seldon.brain import history
from seldon.brain.base import ActionExecutor, Brain, ends_turn
from seldon.core.action import Action
from seldon.brain.prompts import (
    LOCAL_RULES, STEP_LIMIT_REPLY, SYSTEM_PROMPT, bad_arguments, strip_thinking, tool_arguments, user_message,
)
from seldon.core.errors import SeldonError
from seldon.core.registry import register
from seldon.core.types import ActionResult, Utterance

log = logging.getLogger(__name__)

def _tool(action: Action) -> dict[str, Any]:
    d = action.tool_definition()
    return {
        "type": "function",
        "function": {"name": d["name"], "description": d["description"], "parameters": d["input_schema"]},
    }


@register("brain", "ollama")
class OllamaBrain(Brain):
    def __init__(
        self,
        model: str = "llama3.1:8b",
        host: str | None = None,           # défaut : $OLLAMA_HOST ou http://localhost:11434
        vision: bool = False,              # le modèle accepte-t-il des images ? (qwen2.5vl, gemma3…)
        think: bool | None = None,         # raisonnement des modèles qui le supportent (qwen3…)
        temperature: float = 0.3,
        num_ctx: int = 16384,              # fenêtre de contexte allouée par Ollama
        keep_alive: str = "30m",           # garde le modèle chargé en mémoire entre deux phrases
        max_steps: int = 10,               # appels au modèle max par prise de parole
        max_history_messages: int = 200,   # au-delà, les plus anciens échanges sont oubliés
        max_history_chars: int = 30000,    # idem en taille (à ajuster avec num_ctx) ; anciens résultats d'outils abrégés d'abord
        keep_images: int = 1,              # captures gardées dans l'historique (les plus récentes)
        extra_instructions: str = "",
        timeout_s: float = 300,
    ) -> None:
        self.model = model
        self.host = host
        self.vision = vision
        self.think = think
        self.options = {"temperature": temperature, "num_ctx": num_ctx}
        self.keep_alive = keep_alive
        self.max_steps = max_steps
        self.max_history_messages = max_history_messages
        self.max_history_chars = max_history_chars
        self.keep_images = keep_images
        self.system = SYSTEM_PROMPT + LOCAL_RULES + (f"\n{extra_instructions.strip()}\n" if extra_instructions else "")
        self.timeout_s = timeout_s
        self.client: ollama.AsyncClient | None = None
        self.messages: list[dict[str, Any]] = []

    async def start(self) -> None:
        self.client = ollama.AsyncClient(host=self.host, timeout=self.timeout_s)
        try:
            await self.client.show(self.model)
        except ollama.ResponseError as e:
            raise SeldonError(f"Modèle Ollama « {self.model} » introuvable", f"téléchargez-le : ollama pull {self.model}") from e
        except Exception as e:  # serveur injoignable (httpx.ConnectError…)
            raise SeldonError(f"Ollama injoignable ({self.host or 'localhost:11434'})", "lancez `ollama serve`") from e

    def reset(self) -> None:
        self.messages.clear()

    def _tool_message(self, name: str, result: ActionResult) -> dict[str, Any]:
        text = result.text or ("Erreur." if result.is_error else "OK.")
        if result.is_error:
            text = f"ERREUR : {text}"
        msg: dict[str, Any] = {"role": "tool", "tool_name": name, "content": text}
        if result.images:
            if self.vision:
                msg["images"] = [base64.standard_b64encode(img.data).decode("ascii") for img in result.images]
            else:
                msg["content"] += f"\n({len(result.images)} image(s) obtenue(s), mais ce modèle ne peut pas les voir.)"
        return msg

    async def warmup(self, actions: list[Action]) -> None:
        """Charge le modèle et met en cache le prompt système et les outils dès le démarrage."""
        if self.client is None:
            return
        await self.client.chat(
            model=self.model,
            messages=[{"role": "system", "content": self.system}, {"role": "user", "content": "."}],
            tools=[_tool(a) for a in actions],
            think=self.think,
            options={**self.options, "num_predict": 1},
            keep_alive=self.keep_alive,
        )

    async def summarize(self, instructions: str, text: str) -> str | None:
        if self.client is None:
            return None
        response = await self.client.chat(
            model=self.model,
            messages=[{"role": "system", "content": instructions}, {"role": "user", "content": text}],
            think=self.think,
            options=self.options,
            keep_alive=self.keep_alive,
        )
        return strip_thinking(response.message.content or "") or None

    async def handle(self, utterance: Utterance, actions: list[Action], execute: ActionExecutor) -> str | None:
        assert self.client is not None, "start() n'a pas été appelé"
        history.compact(self.messages, history.OLLAMA, self.max_history_chars, self.max_history_messages)

        snapshot = len(self.messages)
        self.messages.append({"role": "user", "content": user_message(utterance)})
        tools = [_tool(a) for a in actions]
        known = {a.name for a in actions}

        try:
            for _ in range(self.max_steps):
                response = await self.client.chat(
                    model=self.model,
                    messages=[{"role": "system", "content": self.system}, *self.messages],
                    tools=tools,
                    think=self.think,
                    options=self.options,
                    keep_alive=self.keep_alive,
                )
                msg = response.message
                calls = list(msg.tool_calls or [])
                content = strip_thinking(msg.content or "")
                self.messages.append({
                    "role": "assistant",
                    "content": content,
                    **({"tool_calls": [c.model_dump() for c in calls]} if calls else {}),
                })
                if not calls:
                    return content or None

                done = []
                for call in calls:
                    name = call.function.name
                    params = tool_arguments(call.function.arguments)
                    if name not in known:
                        result = ActionResult.error(f"Outil inconnu « {name} ». Outils disponibles : {', '.join(sorted(known))}.")
                    elif params is None:
                        result = ActionResult.error(bad_arguments(call.function.arguments))
                    else:
                        result = await execute(name, params)
                    self.messages.append(self._tool_message(name, result))
                    done.append((name, result))
                if any(r.images for _, r in done):
                    history.drop_old_images(self.messages, history.OLLAMA, self.keep_images)
                if ends_turn(actions, done):
                    return None
            log.warning("Nombre maximal d'étapes atteint (%d)", self.max_steps)
            return STEP_LIMIT_REPLY
        except BaseException:
            del self.messages[snapshot:]
            raise
