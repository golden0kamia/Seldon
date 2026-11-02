"""LLM Claude, de deux façons (option ``auth``) :

- ``api`` : clé ``ANTHROPIC_API_KEY`` (fichier .env), facturée au token ;
- ``abonnement`` : compte claude.ai (Pro / Max) via Claude Code, connecté une
  fois avec ``claude auth login``. Claude Code ne reçoit aucun de ses propres
  outils ni réglages : seulement les outils de Seldon, par un serveur MCP interne.
"""

from __future__ import annotations

import base64
import logging
import shutil
from typing import Any

from seldon import journal
from seldon.base import LLM, Result, Tool, ToolCall, turn_over
from seldon.errors import SeldonError

log = logging.getLogger(__name__)

STEP_LIMIT = "Désolé, je n'ai pas réussi à terminer cette demande."
LOGIN_HINT = "connectez votre compte une fois : claude auth login (dans un terminal)"


class Claude(LLM):
    def __init__(
        self,
        auth: str = "api",              # api | abonnement
        model: str = "claude-opus-5-5",
        effort: str = "low",            # low | medium | high | xhigh | max
        max_tokens: int = 16000,
        max_steps: int = 10,            # appels au modèle au plus par phrase
    ) -> None:
        if auth not in ("api", "abonnement"):
            raise ValueError(f"auth inconnu : {auth!r} (api ou abonnement)")
        self.auth = auth
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.max_steps = max_steps
        self.client: Any = None

    async def start(self) -> None:
        if self.auth == "abonnement":
            import claude_agent_sdk  # noqa: F401  (module absent : message d'installation)
            if not shutil.which("claude"):
                log.debug("CLI claude absent du PATH : celui du SDK sera utilisé")
            return
        import anthropic

        try:
            self.client = anthropic.AsyncAnthropic()
        except anthropic.AnthropicError as e:
            raise SeldonError("Clé API Anthropic absente", "ajoutez ANTHROPIC_API_KEY dans .env") from e

    async def run(self, system: str, messages: list[dict[str, str]], tools: list[Tool],
                  call: ToolCall) -> str | None:
        if self.auth == "abonnement":
            return await self._run_account(system, messages, tools, call)
        return await self._run_api(system, messages, tools, call)

    # --- clé API : la boucle outils <-> modèle est ici ----------------------------------

    async def _run_api(self, system: str, messages: list[dict[str, str]], tools: list[Tool],
                       call: ToolCall) -> str | None:
        convo: list[dict[str, Any]] = list(messages)
        schemas = [{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools]
        for _ in range(self.max_steps):
            response = await self.client.messages.create(
                model=self.model, max_tokens=self.max_tokens, system=system, tools=schemas, messages=convo,
                thinking={"type": "adaptive", "display": "summarized"}, output_config={"effort": self.effort},
                cache_control={"type": "ephemeral"},
            )
            for block in response.content:
                if block.type == "thinking":
                    journal.thought(block.thinking)
            text = "".join(b.text for b in response.content if b.type == "text").strip()
            calls = [b for b in response.content if b.type == "tool_use"]
            if response.stop_reason == "refusal":
                return "Désolé, je ne peux pas faire ça."
            if not calls or response.stop_reason != "tool_use":
                return text or None
            journal.thought(text)
            convo.append({"role": "assistant", "content": [b.model_dump(mode="json", exclude_none=True)
                                                           for b in response.content]})
            done, results = [], []
            for c in calls:
                result = await call(c.name, dict(c.input))
                done.append((c.name, result))
                results.append({"type": "tool_result", "tool_use_id": c.id, "is_error": result.error,
                                "content": _content(result, "source")})
            if turn_over(tools, done):
                return None
            convo.append({"role": "user", "content": results})
        return STEP_LIMIT

    # --- abonnement : Claude Code fait la boucle, nos outils lui sont prêtés -------------

    async def _run_account(self, system: str, messages: list[dict[str, str]], tools: list[Tool],
                           call: ToolCall) -> str | None:
        from claude_agent_sdk import (AssistantMessage, ClaudeAgentOptions, ResultMessage, SdkMcpTool, TextBlock,
                                      ThinkingBlock, create_sdk_mcp_server, query)

        def handler(tool: Tool):
            async def run(args: dict[str, Any]) -> dict[str, Any]:
                result = await call(tool.name, dict(args))
                content = _content(result, "mcp")
                if tool.ends_turn and not result.error:
                    content.append({"type": "text", "text": "(Réponse prononcée : termine ton tour sans rien ajouter.)"})
                return {"content": content, "is_error": result.error}
            return run

        server = create_sdk_mcp_server("seldon", tools=[
            SdkMcpTool(t.name, t.description, t.parameters, handler(t)) for t in tools])
        options = ClaudeAgentOptions(
            system_prompt=system + "\nTes outils portent le préfixe mcp__seldon__ (ex. mcp__seldon__speak).\n",
            tools=[], strict_mcp_config=True, setting_sources=[],   # rien de la config Claude Code de l'utilisateur
            mcp_servers={"seldon": server}, allowed_tools=[f"mcp__seldon__{t.name}" for t in tools],
            model=self.model, effort=self.effort, max_turns=self.max_steps,
            env={"ANTHROPIC_API_KEY": "", "ANTHROPIC_AUTH_TOKEN": ""},   # le compte, pas la clé API
        )
        # Une session neuve par phrase : la conversation lui est donnée en texte.
        earlier = "\n".join(f"{'Utilisateur' if m['role'] == 'user' else 'Seldon'} : {m['content']}"
                            for m in messages[:-1])
        prompt = (f"(Échanges précédents :\n{earlier})\n\n" if earlier else "") + messages[-1]["content"]

        text, final = "", None
        async for msg in query(prompt=prompt, options=options):
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, ThinkingBlock):
                        journal.thought(block.thinking)
                text = "".join(b.text for b in msg.content if isinstance(b, TextBlock)).strip() or text
            elif isinstance(msg, ResultMessage):
                final = msg
        if final is None:
            raise SeldonError("Claude (abonnement) : réponse interrompue")
        if final.subtype == "error_max_turns":
            return STEP_LIMIT
        if final.is_error:
            detail = str(final.result or final.subtype)
            hint = LOGIN_HINT if "login" in detail.lower() or "auth" in detail.lower() else None
            raise SeldonError(f"Claude (abonnement) : {detail}", hint)
        return text or None


def _content(result: Result, style: str) -> list[dict[str, Any]]:
    """Résultat d'outil au format Anthropic (``source``) ou MCP (``mcp``)."""
    blocks: list[dict[str, Any]] = []
    for img in result.images:
        data = base64.standard_b64encode(img.data).decode("ascii")
        blocks.append({"type": "image", "source": {"type": "base64", "media_type": img.media_type, "data": data}}
                      if style == "source" else {"type": "image", "data": data, "mimeType": img.media_type})
    blocks.append({"type": "text", "text": result.text or ("Erreur." if result.error else "OK.")})
    return blocks
