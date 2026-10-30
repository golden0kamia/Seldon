"""Claude via le compte claude.ai (abonnement Pro / Max) plutôt qu'une clé API.

Utilisé par le cerveau ``claude`` avec ``auth: abonnement``. Les requêtes
passent par Claude Code (SDK Claude Agent), connecté une fois pour toutes avec
``claude auth login`` : elles sont décomptées des limites de l'abonnement, pas
facturées au token.

Claude Code ne reçoit aucun de ses outils intégrés (fichiers, terminal…), ni
les réglages, hooks ou serveurs MCP de l'utilisateur : seulement les actions
de Seldon, exposées par un serveur MCP interne et exécutées par le pipeline
(confirmations comprises). ``ANTHROPIC_API_KEY`` est masquée au processus, qui
utiliserait sinon la clé (et la facturation au token).

La conversation vit dans une session Claude Code gardée ouverte entre les
tours (pas de redémarrage à chaque phrase). ``messages`` n'en garde qu'une
copie en texte simple (pour la mémoire et le cerveau de secours) ; une session
neuve (reprise d'une conversation, échanges faits par un autre cerveau) reçoit
ces échanges en tête de sa première phrase.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import shutil
from typing import Any

from seldon.actions.base import Action
from seldon.brain.base import ActionExecutor, ends_turn
from seldon.brain.prompts import STEP_LIMIT_REPLY, user_message
from seldon.core.errors import SeldonError
from seldon.core.types import ActionResult, Utterance

log = logging.getLogger(__name__)

SERVER = "seldon"
PREFIX = f"mcp__{SERVER}__"
NO_KEY = {"ANTHROPIC_API_KEY": "", "ANTHROPIC_AUTH_TOKEN": ""}   # vide = absente pour Claude Code
LOGIN_HINT = "connectez votre compte une fois : claude auth login (dans un terminal)"
TOOLS_NOTE = (
    f"\nTes outils portent le préfixe {PREFIX} (ex. {PREFIX}speak) : ce sont les seuls disponibles.\n"
)


def _text(content: Any) -> str:
    """Texte d'un message, quel que soit son format (chaîne ou blocs Anthropic)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text").strip()
    return ""


def transcript(messages: list[dict[str, Any]], max_chars: int) -> str:
    """Échanges passés en texte (les plus récents, ``max_chars`` au plus), pour une session neuve."""
    lines = []
    for m in messages:
        text = " ".join(_text(m.get("content")).split())
        if text:
            lines.append(f"{'Utilisateur' if m.get('role') == 'user' else 'Seldon'} : {text}")
    out, size = [], 0
    for line in reversed(lines):
        size += len(line) + 1
        if size > max_chars:
            break
        out.append(line)
    return "\n".join(reversed(out))


def _result_content(result: ActionResult, last: bool) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {"type": "image", "data": base64.standard_b64encode(img.data).decode("ascii"), "mimeType": img.media_type}
        for img in result.images
    ]
    text = result.text or ("Erreur." if result.is_error else "OK.")
    if last:   # équivalent de la fin de tour du cerveau API : rien d'autre ne serait prononcé
        text += "\n(Réponse prononcée : termine ton tour sans rien ajouter.)"
    content.append({"type": "text", "text": text})
    return {"content": content, "is_error": result.is_error}


def _error(result: Any) -> SeldonError:
    detail = " ".join(str(x) for x in [result.result, *(result.errors or [])] if x).strip()
    low = detail.lower()
    if any(w in low for w in ("login", "not logged", "authentication", "oauth", "invalid api key")):
        return SeldonError("Compte Claude non connecté", LOGIN_HINT)
    if result.api_error_status == 429 or "limit" in low:
        return SeldonError(f"Limite d'utilisation de l'abonnement Claude atteinte : {detail or 'réessayez plus tard'}")
    return SeldonError(f"Claude (abonnement) : {detail or result.subtype}")


class AccountSession:
    """Moitié « abonnement » de ``ClaudeBrain`` : une session Claude Code gardée ouverte."""

    def __init__(self, brain: Any) -> None:
        self.brain = brain              # options (modèle, effort…) et ``messages`` partagés
        self.client: Any = None
        self.tools: tuple[str, ...] = ()
        self.session_id: str | None = None
        self.backlog = 0                # derniers ``messages`` que la session ne connaît pas encore
        self.execute: ActionExecutor | None = None
        self.actions: list[Action] = []
        self.done: list[tuple[str, ActionResult]] = []
        self.said: list[str] = []

    # --- processus Claude Code ------------------------------------------------
    def check(self) -> None:
        try:
            import claude_agent_sdk  # noqa: F401
        except ImportError as e:
            raise SeldonError("Module claude-agent-sdk absent (auth: abonnement)",
                              "pip install 'seldon[llm-claude]'") from e
        if self.brain.claude_bin and not shutil.which(self.brain.claude_bin):
            raise SeldonError(f"Claude Code introuvable : {self.brain.claude_bin}", LOGIN_HINT)

    def options(self, **extra: Any) -> Any:
        from claude_agent_sdk import ClaudeAgentOptions

        return ClaudeAgentOptions(
            tools=[],                 # aucun outil intégré de Claude Code
            strict_mcp_config=True,   # ni les serveurs MCP de l'utilisateur…
            setting_sources=[],       # …ni ses réglages, hooks, CLAUDE.md
            model=self.brain.model,
            env=dict(NO_KEY),
            cli_path=shutil.which(self.brain.claude_bin) if self.brain.claude_bin else None,
            stderr=lambda line: log.debug("claude : %s", line),
            **extra,
        )

    def _server(self, actions: list[Action]) -> Any:
        from claude_agent_sdk import SdkMcpTool, create_sdk_mcp_server

        def handler(name: str):
            async def run(args: dict[str, Any]) -> dict[str, Any]:
                if self.execute is None:   # appel tardif, tour déjà terminé
                    return {"content": [{"type": "text", "text": "Tour terminé."}], "is_error": True}
                result = await self.execute(name, dict(args))
                self.done.append((name, result))
                if name == "speak" and not result.is_error and isinstance(args.get("text"), str):
                    self.said.append(args["text"])
                return _result_content(result, ends_turn(self.actions, [(name, result)]))
            return run

        tools = [SdkMcpTool(a.name, a.description, a.input_schema, handler(a.name)) for a in actions]
        return create_sdk_mcp_server(SERVER, tools=tools)

    async def connect(self, actions: list[Action]) -> None:
        from claude_agent_sdk import ClaudeSDKClient

        names = tuple(a.name for a in actions)
        if self.client is not None and names == self.tools:
            return
        await self.close()
        options = self.options(
            system_prompt=self.brain.system + TOOLS_NOTE,
            mcp_servers={SERVER: self._server(actions)},
            allowed_tools=[PREFIX + n for n in names],
            effort=self.brain.effort,
            max_turns=self.brain.max_steps,
            resume=self.session_id,   # session rouverte : même conversation
        )
        client = ClaudeSDKClient(options)
        try:
            await client.connect()
        except Exception as e:
            raise SeldonError(f"Claude Code n'a pas pu démarrer : {e}", LOGIN_HINT) from e
        self.client, self.tools = client, names
        log.debug("Session Claude Code ouverte (abonnement)")

    async def close(self) -> None:
        client, self.client = self.client, None
        if client is not None:
            try:
                await client.disconnect()
            except Exception as e:  # noqa: BLE001 — processus déjà arrêté
                log.debug("Fermeture de Claude Code : %s", e)

    def _drop_client(self) -> None:
        """Abandonne le processus en cours (fermé en arrière-plan) ; la session sera reprise."""
        client, self.client = self.client, None
        if client is not None:
            try:
                asyncio.get_running_loop().create_task(client.disconnect())
            except RuntimeError:   # pas de boucle : le processus s'arrêtera avec Seldon
                pass

    def forget(self) -> None:
        """La prochaine phrase ouvre une session neuve, qui reçoit ``messages`` en tête."""
        self._drop_client()
        self.session_id = None
        self.backlog = len(self.brain.messages)

    def added(self, count: int) -> None:
        """``count`` messages ajoutés à ``messages`` par un autre cerveau : transmis à la prochaine phrase."""
        self.backlog += count

    # --- tours ----------------------------------------------------------------
    async def summarize(self, instructions: str, text: str) -> str | None:
        from claude_agent_sdk import ResultMessage, query

        out = None
        async for msg in query(prompt=text, options=self.options(system_prompt=instructions, effort="low",
                                                                  max_turns=1)):
            if isinstance(msg, ResultMessage):
                if msg.is_error:
                    raise _error(msg)
                out = (msg.result or "").strip() or None
        return out

    async def handle(self, utterance: Utterance, actions: list[Action], execute: ActionExecutor) -> str | None:
        from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

        await self.connect(actions)
        messages = self.brain.messages
        message = user_message(utterance)
        earlier = transcript(messages[len(messages) - self.backlog:], self.brain.max_history_chars) if self.backlog else ""
        prompt = f"[note : échanges précédents de cette conversation]\n{earlier}\n\n{message}" if earlier else message

        self.execute, self.actions, self.done, self.said = execute, actions, [], []
        text, result = "", None
        try:
            await self.client.query(prompt)
            async for msg in self.client.receive_response():
                if isinstance(msg, AssistantMessage):
                    text = "".join(b.text for b in msg.content if isinstance(b, TextBlock)).strip() or text
                elif isinstance(msg, ResultMessage):
                    result = msg
        except BaseException:
            self._drop_client()   # flux interrompu (annulation, processus tombé)
            raise
        finally:
            self.execute = None

        if result is None or (result.is_error and result.subtype != "error_max_turns"):
            self._drop_client()
            raise _error(result) if result is not None else SeldonError("Claude (abonnement) : réponse interrompue")
        self.session_id, self.backlog = result.session_id, 0

        if result.subtype == "error_max_turns":
            log.warning("Nombre maximal d'étapes atteint (%d)", self.brain.max_steps)
            reply = STEP_LIMIT_REPLY
        else:
            reply = None if self.said else text or None
        messages.append({"role": "user", "content": message})
        messages.append({"role": "assistant", "content": " ".join(self.said) or reply or "…"})
        excess = len(messages) - self.brain.max_history_messages
        if excess > 0:
            del messages[:excess + excess % 2]
        return reply
