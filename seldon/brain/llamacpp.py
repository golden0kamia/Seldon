"""LLM local via un serveur llama.cpp (``llama serve``), API compatible OpenAI.

Si le serveur ne répond pas au démarrage, ``server_command`` le lance (par
défaut ``./llama-serve.sh``) et Seldon attend que le modèle soit chargé. Le
serveur continue de tourner après l'arrêt de Seldon.

Marche aussi avec tout serveur compatible OpenAI (Ollama, vLLM, LM Studio…).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import shlex
import subprocess
from typing import Any

import httpx

from seldon import journal
from seldon.base import LLM, Result, Tool, ToolCall, turn_over
from seldon.errors import SeldonError, redact

log = logging.getLogger(__name__)

_THINK = re.compile(r"<think>(.*?)(</think>|$)", re.S)
STEP_LIMIT = "Désolé, je n'ai pas réussi à terminer cette demande."


class LlamaCpp(LLM):
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:9931/v1",
        model: str | None = None,          # None : le modèle chargé sur le serveur
        api_key_env: str | None = None,    # variable de .env contenant la clé (serveur distant)
        vision: bool = False,              # le modèle lit-il les images (caméra) ?
        thinking: bool = False,            # raisonnement du modèle : plus lent, parfois plus juste
        temperature: float | None = None,  # None : réglage du serveur
        max_tokens: int = 4096,
        max_steps: int = 10,               # appels au modèle au plus par phrase
        timeout_s: float = 300,
        server_command: str | None = "./llama-serve.sh",   # lancé si le serveur ne répond pas ; null : jamais
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = os.environ.get(api_key_env) if api_key_env else None
        self.vision = vision
        self.thinking = thinking
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_steps = max_steps
        self.timeout_s = timeout_s
        self.server_command = server_command
        self.client: httpx.AsyncClient | None = None

    # --- démarrage ----------------------------------------------------------------

    async def start(self) -> None:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        self.client = httpx.AsyncClient(base_url=self.base_url, headers=headers,
                                        timeout=httpx.Timeout(self.timeout_s, connect=10))
        try:
            models = await self._models()
        except httpx.ConnectError:
            if not self.server_command:
                raise SeldonError(f"Serveur llama injoignable ({self.base_url})", "lancez `llama serve`") from None
            models = await self._launch_server()
        if self.model is None:
            loaded = [m["id"] for m in models if m.get("status", {}).get("value") == "loaded"]
            self.model = (loaded or [m["id"] for m in models])[0]
        log.info("Cerveau llama.cpp : %s", self.model)

    async def _models(self) -> list[dict[str, Any]]:
        response = await self.client.get("/models")
        response.raise_for_status()
        models = response.json()["data"]
        if not models:
            raise SeldonError("Aucun modèle sur le serveur llama", "lancez `llama serve` avec un modèle")
        return models

    async def _launch_server(self) -> list[dict[str, Any]]:
        log.info("Serveur llama arrêté : lancement de %s (sortie dans llama-serve.log)", self.server_command)
        with open("llama-serve.log", "ab") as out:
            # Nouvelle session : le serveur survit à l'arrêt de Seldon (pas de rechargement du modèle).
            proc = subprocess.Popen(shlex.split(self.server_command), stdin=subprocess.DEVNULL,
                                    stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(600):   # chargement du modèle : jusqu'à 10 minutes
            await asyncio.sleep(1)
            if proc.poll() is not None:
                raise SeldonError(f"Le serveur llama s'est arrêté (code {proc.returncode})", "voir llama-serve.log")
            try:
                return await self._models()
            except httpx.HTTPError:   # pas encore prêt (connexion refusée, 503 pendant le chargement)
                continue
        raise SeldonError("Le serveur llama n'est pas prêt après 10 minutes", "voir llama-serve.log")

    async def stop(self) -> None:
        if self.client is not None:
            await self.client.aclose()

    # --- une phrase ------------------------------------------------------------------

    async def run(self, system: str, messages: list[dict[str, str]], tools: list[Tool],
                  call: ToolCall) -> str | None:
        convo: list[dict[str, Any]] = [{"role": "system", "content": system}, *messages]
        schemas = [{"type": "function", "function": t.schema()} for t in tools]
        for step in range(self.max_steps):
            msg = await self._chat(convo, schemas)
            text, thought = _split_thinking(msg.get("content") or "")
            journal.thought(msg.get("reasoning_content") or thought)
            calls = msg.get("tool_calls") or []
            if not calls:
                return text or None
            journal.thought(text)   # texte écrit avec des appels d'outils : jamais prononcé
            convo.append({"role": "assistant", "content": text, "tool_calls": calls})
            done, replies = [], []
            for i, c in enumerate(calls):
                c["id"] = c.get("id") or f"appel_{step}_{i}"
                name = c["function"]["name"]
                try:
                    args = json.loads(c["function"].get("arguments") or "{}")
                    result = await call(name, args) if isinstance(args, dict) else Result.fail("Arguments : objet JSON attendu.")
                except json.JSONDecodeError:
                    result = Result.fail("Arguments illisibles : JSON valide attendu.")
                done.append((name, result))
                replies.extend(self._tool_reply(c["id"], result))
            # Les réponses « tool » doivent suivre l'appel sans interruption : les images après.
            convo.extend(sorted(replies, key=lambda m: m["role"] != "tool"))
            if turn_over(tools, done):
                return None
        log.warning("Nombre maximal d'étapes atteint (%d)", self.max_steps)
        return STEP_LIMIT

    async def _chat(self, convo: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self.model, "messages": convo, "tools": tools, "max_tokens": self.max_tokens,
                                "chat_template_kwargs": {"enable_thinking": self.thinking}}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        response = await self.client.post("/chat/completions", json=body)
        if response.is_error:
            raise SeldonError(f"Le serveur llama a refusé la requête (HTTP {response.status_code}) : "
                              f"{redact(response.text[:300])}",
                              "le modèle doit gérer les outils : lancez `llama serve` avec --jinja")
        return response.json()["choices"][0]["message"]

    def _tool_reply(self, call_id: str, result: Result) -> list[dict[str, Any]]:
        text = ("ERREUR : " if result.error else "") + (result.text or "OK.")
        if result.images and not self.vision:
            text += "\n(Image obtenue, mais ce modèle ne peut pas la voir.)"
        reply: list[dict[str, Any]] = [{"role": "tool", "tool_call_id": call_id, "content": text}]
        if result.images and self.vision:   # un message « tool » ne porte que du texte : l'image suit
            reply.append({"role": "user", "content": [{"type": "text", "text": "Image renvoyée par l'outil :"}, *(
                {"type": "image_url", "image_url": {
                    "url": f"data:{img.media_type};base64,{base64.b64encode(img.data).decode()}"}}
                for img in result.images)]})
        return reply


def _split_thinking(content: str) -> tuple[str, str]:
    """Sépare la réponse du raisonnement ``<think>…</think>`` que certains modèles écrivent."""
    thought = " ".join(m.group(1).strip() for m in _THINK.finditer(content))
    return _THINK.sub("", content).strip(), thought
