"""Cerveau local via un serveur llama.cpp (``llama serve`` / ``llama-server``).

Utilise l'API compatible OpenAI (``/v1/chat/completions``) avec appels d'outils.
Le serveur doit déjà tourner, par exemple :

    llama serve --hf-repo unsloth/qwen3.8-27B-GGUF:Q3_K_XL -c 65000

Fonctionne aussi avec tout autre serveur compatible OpenAI (vLLM, LM Studio…).
"""

from __future__ import annotations

import base64
import logging
import time
from typing import Any

import httpx

from seldon.actions.base import Action
from seldon.brain.base import ActionExecutor, Brain, ends_turn
from seldon.brain.prompts import LOCAL_RULES, SYSTEM_PROMPT, parse_arguments, user_message
from seldon.core.errors import SeldonError
from seldon.core.registry import register
from seldon.core.types import ActionResult, Utterance

log = logging.getLogger(__name__)


def _http_error(response: httpx.Response) -> tuple[str, str | None]:
    """Message (et piste) pour une erreur HTTP du serveur."""
    try:
        detail = response.json().get("error", {})
        detail = detail.get("message", "") if isinstance(detail, dict) else str(detail)
    except ValueError:
        detail = response.text
    detail = " ".join(detail.split())[:300]
    message = f"Le serveur llama a refusé la requête (HTTP {response.status_code}) : {detail}"
    if "context" in detail.lower() and ("exceed" in detail.lower() or "size" in detail.lower()):
        return message, "conversation trop longue pour le contexte du modèle : dites « nouvelle », ou lancez le serveur avec un -c plus grand"
    if response.status_code in (401, 403):
        return message, "vérifiez brain.options.api_key"
    if "tool" in detail.lower() or "jinja" in detail.lower():
        return message, "le modèle doit gérer les appels d'outils : lancez `llama serve` avec --jinja"
    return message, None


@register("brain", "llamacpp")
class LlamaCppBrain(Brain):
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:9931/v1",
        model: str | None = None,           # None : premier modèle annoncé par /v1/models
        api_key: str | None = None,
        vision: bool = False,               # le modèle accepte-t-il des images ?
        thinking: bool | None = False,      # raisonnement (Qwen3…) : False = réponses bien plus rapides
        temperature: float | None = None,   # None : réglage du serveur
        max_tokens: int = 4096,
        max_steps: int = 10,
        max_history_messages: int = 200,
        extra_instructions: str = "",
        timeout_s: float = 300,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.vision = vision
        self.thinking = thinking
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_steps = max_steps
        self.max_history_messages = max_history_messages
        self.system = SYSTEM_PROMPT + LOCAL_RULES + (f"\n{extra_instructions.strip()}\n" if extra_instructions else "")
        self.timeout_s = timeout_s
        self.client: httpx.AsyncClient | None = None
        self.messages: list[dict[str, Any]] = []

    async def start(self) -> None:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        self.client = httpx.AsyncClient(base_url=self.base_url, headers=headers, timeout=self.timeout_s)
        try:
            models = (await self.client.get("/models")).raise_for_status().json()["data"]
        except httpx.HTTPStatusError as e:
            raise SeldonError(f"Le serveur {self.base_url} a répondu HTTP {e.response.status_code} à /models",
                              "base_url doit pointer vers une API compatible OpenAI (se termine souvent par /v1)") from e
        except httpx.HTTPError as e:
            raise SeldonError(f"Serveur llama injoignable ({self.base_url})",
                              "lancez-le (`llama serve …`) ou corrigez brain.options.base_url") from e
        except (ValueError, KeyError, TypeError) as e:
            raise SeldonError(f"Réponse inattendue de {self.base_url}/models : ce n'est pas un serveur compatible OpenAI") from e
        ids = [m["id"] for m in models]
        if not ids:
            raise SeldonError("Aucun modèle chargé sur le serveur llama", "lancez `llama serve` avec un modèle")
        if self.model is None:
            loaded = [m["id"] for m in models if m.get("status", {}).get("value") == "loaded"]
            self.model = (loaded or ids or [None])[0]
        if self.model not in ids:
            raise SeldonError(f"Modèle « {self.model} » absent du serveur llama",
                              f"disponibles : {', '.join(ids)} (ou model: null pour le modèle chargé)")
        log.info("Cerveau llama.cpp : %s", self.model)

    async def stop(self) -> None:
        if self.client is not None:
            await self.client.aclose()

    def reset(self) -> None:
        self.messages.clear()

    def _tool_messages(self, call_id: str, result: ActionResult) -> list[dict[str, Any]]:
        text = result.text or ("Erreur." if result.is_error else "OK.")
        if result.is_error:
            text = f"ERREUR : {text}"
        if result.images and not self.vision:
            text += f"\n({len(result.images)} image(s) obtenue(s), mais ce modèle ne peut pas les voir.)"
        msgs = [{"role": "tool", "tool_call_id": call_id, "content": text}]
        if result.images and self.vision:
            # Les messages "tool" ne portent que du texte : les images suivent dans un message utilisateur.
            msgs.append({"role": "user", "content": [
                {"type": "text", "text": "Image(s) renvoyée(s) par l'outil :"},
                *({"type": "image_url", "image_url": {
                    "url": f"data:{img.media_type};base64,{base64.b64encode(img.data).decode('ascii')}"}}
                  for img in result.images),
            ]})
        return msgs

    async def _chat(self, tools: list[dict[str, Any]], messages: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages or [{"role": "system", "content": self.system}, *self.messages],
            "max_tokens": self.max_tokens,
        }
        if tools:
            body["tools"] = tools
        if self.thinking is not None:
            body["chat_template_kwargs"] = {"enable_thinking": self.thinking}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        response = await self.client.post("/chat/completions", json=body)
        if response.is_error:
            raise SeldonError(*_http_error(response))
        try:
            return response.json()["choices"][0]["message"]
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise SeldonError(f"Réponse illisible du serveur llama : {response.text[:200]}") from e

    async def summarize(self, instructions: str, text: str) -> str | None:
        if self.client is None:
            return None
        msg = await self._chat([], [{"role": "system", "content": instructions}, {"role": "user", "content": text}])
        return (msg.get("content") or "").strip() or None

    async def handle(self, utterance: Utterance, actions: list[Action], execute: ActionExecutor) -> str | None:
        assert self.client is not None, "start() n'a pas été appelé"
        if len(self.messages) > self.max_history_messages:
            self.reset()

        snapshot = len(self.messages)
        self.messages.append({"role": "user", "content": user_message(utterance)})
        tools = [
            {"type": "function", "function": {
                "name": d["name"], "description": d["description"], "parameters": d["input_schema"]}}
            for d in (a.tool_definition() for a in actions)
        ]
        known = {a.name for a in actions}

        try:
            for _ in range(self.max_steps):
                msg = await self._chat(tools)
                calls = msg.get("tool_calls") or []
                self.messages.append({
                    "role": "assistant",
                    "content": msg.get("content") or "",
                    **({"tool_calls": calls} if calls else {}),
                })
                if not calls:
                    return (msg.get("content") or "").strip() or None

                done = []
                for call in calls:
                    name = call["function"]["name"]
                    if name in known:
                        result = await execute(name, parse_arguments(call["function"].get("arguments")))
                    else:
                        result = ActionResult.error(f"Outil inconnu « {name} ». Outils disponibles : {', '.join(sorted(known))}.")
                    self.messages.extend(self._tool_messages(call["id"], result))
                    done.append((name, result))
                if ends_turn(actions, done):
                    return None
            log.warning("Nombre maximal d'étapes atteint (%d)", self.max_steps)
            return None
        except Exception:
            del self.messages[snapshot:]
            raise
