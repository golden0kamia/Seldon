"""Cerveau local via un serveur llama.cpp (``llama serve`` / ``llama-server``).

Utilise l'API compatible OpenAI (``/v1/chat/completions``) avec appels d'outils.
Le serveur doit déjà tourner, par exemple :

    llama serve --hf-repo unsloth/qwen3.8-27B-GGUF:Q3_K_XL -c 65000

Fonctionne aussi avec tout autre serveur compatible OpenAI (vLLM, LM Studio…).

Latence :
- la réponse est lue en flux (``stream``) : chaque phrase de l'appel ``speak``
  est dite dès qu'elle est écrite, pendant que le modèle continue (voir
  ``seldon.brain.speech``) ;
- le prompt système et les outils ne changent pas d'un tour à l'autre et
  l'historique n'est retouché que rarement (``seldon.brain.history``) : le
  serveur réutilise le cache du préfixe et ne calcule que le nouveau message ;
  ``warmup`` remplit ce cache dès le démarrage.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Any

import httpx

from seldon.actions.base import Action
from seldon.brain import history
from seldon.brain.base import ActionExecutor, Brain, ends_turn
from seldon.brain.prompts import (
    LOCAL_RULES, STEP_LIMIT_REPLY, SYSTEM_PROMPT, bad_arguments, strip_thinking, tool_arguments, user_message,
)
from seldon.brain.speech import SpeechStream
from seldon.core.errors import SeldonError, redact
from seldon.core.registry import register
from seldon.core.types import ActionResult, Utterance

log = logging.getLogger(__name__)

CONNECT_TIMEOUT_S = 10   # serveur local : s'il ne répond pas vite, il est arrêté ou injoignable


def _http_error(response: httpx.Response, server: str = "serveur llama",
                key_hint: str = "vérifiez brain.options.api_key") -> tuple[str, str | None]:
    """Message (et piste) pour une erreur HTTP du serveur."""
    try:
        detail = response.json().get("error", {})
        detail = detail.get("message", "") if isinstance(detail, dict) else str(detail)
    except ValueError:
        detail = response.text
    detail = redact(" ".join(detail.split())[:300])
    message = f"Le {server} a refusé la requête (HTTP {response.status_code}) : {detail}"
    if "context" in detail.lower() and ("exceed" in detail.lower() or "size" in detail.lower()):
        return message, "conversation trop longue pour le contexte du modèle : dites « nouvelle », ou lancez le serveur avec un -c plus grand"
    if response.status_code in (401, 403):
        return message, key_hint
    if "tool" in detail.lower() or "jinja" in detail.lower():
        return message, "le modèle doit gérer les appels d'outils : lancez `llama serve` avec --jinja"
    return message, None


class _NoStreaming(Exception):
    """Le serveur refuse le flux avec des outils (ancienne version)."""


@register("brain", "llamacpp")
class LlamaCppBrain(Brain):
    server = "serveur llama"                     # nom dans les messages d'erreur
    key_hint = "vérifiez brain.options.api_key"   # piste pour une erreur 401/403
    max_tokens_field = "max_tokens"             # OpenAI : max_completion_tokens

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
        max_history_messages: int = 200,    # au-delà, les plus anciens échanges sont oubliés
        max_history_chars: int = 40000,     # idem en taille ; les anciens résultats d'outils sont d'abord abrégés
        keep_images: int = 1,               # captures gardées dans l'historique (les plus récentes)
        stream: bool = True,                # dit chaque phrase de la réponse dès qu'elle est écrite
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
        self.max_history_chars = max_history_chars
        self.keep_images = keep_images
        self.stream = stream
        self.system = SYSTEM_PROMPT + LOCAL_RULES + (f"\n{extra_instructions.strip()}\n" if extra_instructions else "")
        self.timeout_s = timeout_s
        self.client: httpx.AsyncClient | None = None
        self.messages: list[dict[str, Any]] = []

    async def start(self) -> None:
        if self.client is not None:
            await self.client.aclose()
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        # Un seul client (connexion gardée ouverte) pour toutes les requêtes.
        self.client = httpx.AsyncClient(base_url=self.base_url, headers=headers,
                                        timeout=httpx.Timeout(self.timeout_s, connect=min(CONNECT_TIMEOUT_S, self.timeout_s)))
        try:
            models = (await self.client.get("/models")).raise_for_status().json()["data"]
        except httpx.HTTPStatusError as e:
            raise SeldonError(f"Le serveur {self.base_url} a répondu HTTP {e.response.status_code} à /models",
                              "base_url doit pointer vers une API compatible OpenAI (se termine souvent par /v1)") from e
        except httpx.HTTPError as e:
            raise SeldonError(f"{self.server.capitalize()} injoignable ({self.base_url})",
                              "lancez-le (`llama serve …`) ou corrigez brain.options.base_url") from e
        except (ValueError, KeyError, TypeError) as e:
            raise SeldonError(f"Réponse inattendue de {self.base_url}/models : ce n'est pas un serveur compatible OpenAI") from e
        ids = [m["id"] for m in models]
        if not ids:
            raise SeldonError(f"Aucun modèle disponible sur le {self.server}", "lancez `llama serve` avec un modèle")
        if self.model is None:
            loaded = [m["id"] for m in models if m.get("status", {}).get("value") == "loaded"]
            self.model = (loaded or ids or [None])[0]
        if self.model not in ids:
            raise SeldonError(f"Modèle « {self.model} » absent du {self.server}",
                              f"disponibles : {', '.join(ids)} (ou model: null pour le modèle chargé)")
        log.info("Cerveau %s : %s", self.server, self.model)

    async def stop(self) -> None:
        if self.client is not None:
            await self.client.aclose()

    def reset(self) -> None:
        self.messages.clear()

    @staticmethod
    def _tools(actions: list[Action]) -> list[dict[str, Any]]:
        # Même ordre et même contenu à chaque tour : le préfixe du prompt reste en cache côté serveur.
        return [
            {"type": "function", "function": {
                "name": d["name"], "description": d["description"], "parameters": d["input_schema"]}}
            for d in (a.tool_definition() for a in actions)
        ]

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

    def _body(self, tools: list[dict[str, Any]], messages: list[dict[str, Any]] | None = None,
              max_tokens: int | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages or [{"role": "system", "content": self.system}, *self.messages],
            self.max_tokens_field: max_tokens or self.max_tokens,
        }
        if tools:
            body["tools"] = tools
        if self.thinking is not None:
            body["chat_template_kwargs"] = {"enable_thinking": self.thinking}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        return body

    async def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        response = await self.client.post("/chat/completions", json=body)
        if response.is_error:
            raise SeldonError(*_http_error(response, self.server, self.key_hint))
        try:
            return response.json()["choices"][0]["message"]
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise SeldonError(f"Réponse illisible du {self.server} : {response.text[:200]}") from e

    async def _stream(self, body: dict[str, Any], speech: SpeechStream) -> dict[str, Any]:
        """Lit la réponse en flux (SSE) ; les phrases du premier appel ``speak`` sont dites au fil de l'eau."""
        content: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        async with self.client.stream("POST", "/chat/completions", json={**body, "stream": True}) as response:
            if response.is_error:
                await response.aread()
                message, hint = _http_error(response, self.server, self.key_hint)
                if "stream" in message.lower():
                    raise _NoStreaming(message)
                raise SeldonError(message, hint)
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except ValueError as e:
                    raise SeldonError(f"Réponse illisible du {self.server} : {data[:200]}") from e
                if chunk.get("error"):
                    error = chunk["error"]
                    detail = error.get("message", "") if isinstance(error, dict) else str(error)
                    raise SeldonError(f"Le {self.server} a signalé une erreur : {redact(detail[:300])}")
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        content.append(delta["content"])
                    for tc in delta.get("tool_calls") or []:
                        call = calls.setdefault(tc.get("index", 0), {
                            "id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                        call["id"] = tc.get("id") or call["id"]
                        fn = tc.get("function") or {}
                        call["function"]["name"] += fn.get("name") or ""
                        call["function"]["arguments"] += fn.get("arguments") or ""
                first = calls.get(0)
                if first and first["function"]["name"] == speech.name and first["function"]["arguments"]:
                    speech.feed(first["function"]["arguments"])
                    if speech.stopped:
                        break   # coupé : inutile de laisser le modèle finir (la connexion fermée l'arrête)
        return {"role": "assistant", "content": "".join(content), "tool_calls": [calls[i] for i in sorted(calls)]}

    async def _chat(self, tools: list[dict[str, Any]], messages: list[dict[str, Any]] | None = None,
                    speech: SpeechStream | None = None) -> dict[str, Any]:
        body = self._body(tools, messages)
        if speech is not None and self.stream:
            try:
                return await self._stream(body, speech)
            except _NoStreaming as e:
                log.warning("Réponse en flux refusée par le serveur (%s) : désactivée", e)
                self.stream = False
        return await self._post(body)

    async def warmup(self, actions: list[Action]) -> None:
        """Fait calculer au serveur le prompt système et les outils (gardés en cache) dès le démarrage."""
        if self.client is None:
            return
        messages = [{"role": "system", "content": self.system}, {"role": "user", "content": "."}]
        await self._post(self._body(self._tools(actions), messages, max_tokens=1))
        log.debug("Cache du prompt préparé")

    async def summarize(self, instructions: str, text: str) -> str | None:
        if self.client is None:
            return None
        msg = await self._chat([], [{"role": "system", "content": instructions}, {"role": "user", "content": text}])
        return strip_thinking(msg.get("content") or "") or None

    async def handle(self, utterance: Utterance, actions: list[Action], execute: ActionExecutor) -> str | None:
        assert self.client is not None, "start() n'a pas été appelé"
        history.compact(self.messages, history.OPENAI, self.max_history_chars, self.max_history_messages)

        snapshot = len(self.messages)
        self.messages.append({"role": "user", "content": user_message(utterance)})
        tools = self._tools(actions)
        known = {a.name for a in actions}

        try:
            for step in range(self.max_steps):
                speech = SpeechStream(execute) if self.stream and "speak" in known else None
                try:
                    msg = await self._chat(tools, speech=speech)
                except asyncio.CancelledError:
                    if speech is not None:
                        speech.cancel()
                    raise
                except Exception:
                    if speech is not None:   # la phrase en cours se termine, la suite est abandonnée
                        speech.abort()
                        await speech.close()
                    raise
                calls = [c for c in msg.get("tool_calls") or [] if isinstance(c, dict)]
                if speech is not None and speech.stopped:
                    calls = calls[:1]   # interrompu pendant la génération : les appels suivants sont incomplets
                content = strip_thinking(msg.get("content") or "")
                self.messages.append({"role": "assistant", "content": content, **({"tool_calls": calls} if calls else {})})
                if not calls:
                    return content or None

                done, replies = [], []
                for i, call in enumerate(calls):
                    fn = call.setdefault("function", {})
                    name = fn.get("name") or ""
                    call["id"] = call.get("id") or f"call_{step}_{i}"
                    params = tool_arguments(fn.get("arguments"))
                    if i == 0 and speech is not None and speech.started:
                        result = await speech.finish(params.get("text") if params else None)
                        if speech.stopped or params is None:   # l'historique garde ce qui a été dit
                            fn["arguments"] = json.dumps({"text": speech.spoken_text}, ensure_ascii=False)
                        if speech.stopped:   # coupé : les appels suivants ignoraient l'interruption
                            del calls[1:]
                    elif name not in known:
                        result = ActionResult.error(f"Outil inconnu « {name} ». Outils disponibles : {', '.join(sorted(known))}.")
                    elif params is None:
                        result = ActionResult.error(bad_arguments(fn.get("arguments")))
                    else:
                        result = await execute(name, params)
                    replies.extend(self._tool_messages(call["id"], result))
                    done.append((name, result))
                # Réponses « tool » d'abord, images ensuite : OpenAI exige qu'elles suivent l'appel sans interruption.
                self.messages.extend(sorted(replies, key=lambda m: m["role"] != "tool"))
                if any(r.images for _, r in done):
                    history.drop_old_images(self.messages, history.OPENAI, self.keep_images)
                if ends_turn(actions, done):
                    return None
            log.warning("Nombre maximal d'étapes atteint (%d)", self.max_steps)
            return STEP_LIMIT_REPLY
        except BaseException:
            del self.messages[snapshot:]
            raise
