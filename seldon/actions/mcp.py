"""Base des actions branchées sur un serveur MCP (Model Context Protocol).

Contient un petit client MCP en Streamable HTTP (POST JSON-RPC, réponses JSON ou
SSE ; initialize, tools/list, tools/call) basé sur httpx, et une action qui
expose tous les outils du serveur au modèle par un seul outil dont les
paramètres sont ``tool`` (nom de l'outil MCP) et ``arguments`` (ses paramètres) ;
la description détaille chaque outil. Le pipeline associe en effet un nom
d'outil à une action.

La liste des outils est chargée au démarrage — si le serveur est injoignable,
Seldon démarre quand même et réessaie au premier appel. Une sous-classe précise
le nom du service (``label``), l'en-tête d'authentification (``_auth_header``)
et, si besoin, les outils à confirmer (``_needs_confirmation``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, ClassVar

from seldon.actions.base import Action
from seldon.core.errors import SeldonError, describe, redact
from seldon.core.types import ActionContext, ActionResult

log = logging.getLogger(__name__)

PROTOCOL_VERSION = "2025-06-18"     # version proposée ; celle du serveur est ensuite utilisée
MAX_BODY = 2_000_000                # octets lus au plus par réponse HTTP
MAX_PAGES = 20                      # pages de tools/list
REFRESH_DELAY = 30.0                # s entre deux rechargements de la liste d'outils


class MCPError(SeldonError):
    """Échec d'un échange MCP (message lisible, sans secret)."""


def _sse_messages(lines: list[str]) -> list[dict[str, Any]]:
    """Messages JSON d'un flux ``text/event-stream`` (champs ``data:`` regroupés par événement)."""
    out: list[dict[str, Any]] = []
    data: list[str] = []
    for line in [*lines, ""]:
        line = line.rstrip("\r")
        if not line:
            if data:
                try:
                    msg = json.loads("\n".join(data))
                except ValueError:
                    msg = None
                if isinstance(msg, dict):
                    out.append(msg)
                data = []
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
    return out


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: max(0, limit - 30)].rstrip() + f"… [tronqué, {len(text)} caractères]"


def _first_sentence(text: str, limit: int = 160) -> str:
    text = " ".join((text or "").split())
    m = re.match(r"(.+?[.!?])(\s|$)", text)
    return _clip(m.group(1) if m else text, limit)


def _param_summary(schema: dict[str, Any]) -> str:
    """« name, area, domain (light|switch…), brightness* » : paramètres d'un outil (* = obligatoire)."""
    props = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    parts = []
    for key, spec in props.items():
        spec = spec if isinstance(spec, dict) else {}
        items = spec.get("items")
        enum = spec.get("enum") or (items.get("enum") if isinstance(items, dict) else None)
        label = key + ("*" if key in required else "")
        if enum:
            shown = "|".join(map(str, enum[:6])) + ("|…" if len(enum) > 6 else "")
            label += f" ({shown})"
        elif spec.get("type") in ("integer", "number"):
            lo, hi = spec.get("minimum"), spec.get("maximum")
            label += f" (nombre{f' {lo}-{hi}' if lo is not None and hi is not None else ''})"
        parts.append(label)
    return ", ".join(parts) if parts else "aucun"


def _args_summary(args: dict[str, Any]) -> str:
    """« title « Courses », category « Maison » » : premiers paramètres d'un appel, pour une confirmation."""
    return ", ".join(f"{k} « {_clip(str(v), 60)} »" for k, v in list(args.items())[:3]) or "sans paramètre"


class MCPAction(Action):
    """Action dont les outils sont ceux d'un serveur MCP (voir le docstring du module)."""

    label: ClassVar[str] = "Serveur MCP"   # nom du service dans les messages
    intro: ClassVar[str] = ""   # début de la description donnée au modèle
    example_tool: ClassVar[str] = "nom_outil"
    max_description_chars: ClassVar[int] = 6000
    auth_hint: ClassVar[str] = "identifiants invalides"   # précision d'un refus HTTP 401/403
    not_found_hint: ClassVar[str] = "URL erronée"   # précision d'un HTTP 404

    def __init__(
        self,
        url: str,
        timeout: float = 10.0,
        confirm_tools: list[str] | None = None,
        allowed_tools: list[str] | None = None,
        denied_tools: list[str] | None = None,
        max_result_chars: int = 4000,
        verify_ssl: bool = True,
    ) -> None:
        self.url = url
        self.timeout = float(timeout)
        self.confirm_tools = set(confirm_tools or ())
        self.allowed_tools = set(allowed_tools) if allowed_tools else None
        self.denied_tools = set(denied_tools or ())
        self.max_result_chars = max(200, int(max_result_chars))
        self.verify_ssl = verify_ssl
        self.client: Any = None   # httpx.AsyncClient (créé au premier usage)
        self.tools: dict[str, dict[str, Any]] | None = None   # nom -> définition MCP ; None : pas encore chargée
        self._session_id: str | None = None
        self._protocol: str | None = None   # version négociée ; None : session à initialiser
        self._next_id = 0
        self._lock = asyncio.Lock()   # chargement des outils
        self._session_lock = asyncio.Lock()   # ouverture de la session MCP
        self._last_refresh = 0.0

    # --- à préciser par les sous-classes ---

    def _auth_header(self) -> str | None:
        """Valeur de l'en-tête ``Authorization`` ; None : aucun."""
        return None

    def _missing_config(self) -> str | None:
        """Pourquoi l'action est inutilisable (identifiant manquant…) ; None : prête."""
        return None

    async def _needs_confirmation(self, tool: str, args: dict[str, Any]) -> str | None:
        """Description de la cible si l'appel doit être confirmé, sinon None."""
        if tool in self.confirm_tools:
            return _args_summary(args)
        return None

    def _item_text(self, text: str) -> str:
        """Texte d'un élément de résultat, éventuellement simplifié."""
        return text

    # --- définition de l'outil pour le modèle ---

    @property
    def description(self) -> str:  # type: ignore[override]
        head = self.intro
        if not self.tools:
            return head + " (Liste des outils indisponible pour l'instant : elle sera chargée au premier appel.)"
        lines = [f"- {name} : {_first_sentence(t.get('description', ''))} Paramètres : "
                 f"{_param_summary(t.get('inputSchema') or {})}" for name, t in self.tools.items()]
        return _clip(head + "\nOutils disponibles (* = obligatoire) :\n" + "\n".join(lines), self.max_description_chars)

    @property
    def input_schema(self) -> dict[str, Any]:  # type: ignore[override]
        tool: dict[str, Any] = {"type": "string", "description": f"Nom de l'outil {self.label} (ex. {self.example_tool})."}
        if self.tools:
            tool["enum"] = list(self.tools)
        return {
            "type": "object",
            "properties": {
                "tool": tool,
                "arguments": {"type": "object", "description": "Paramètres de l'outil ({} s'il n'en a pas)."},
            },
            "required": ["tool"],
            "additionalProperties": False,
        }

    # --- cycle de vie ---

    async def start(self) -> None:
        if missing := self._missing_config():
            log.warning("%s : %s — action inutilisable.", self.label, missing)
            return
        try:
            await self._load_tools()
            log.info("%s : %d outil(s) MCP disponible(s).", self.label, len(self.tools or {}))
        except Exception as e:   # serveur éteint au démarrage : on réessaiera au premier appel
            log.warning("%s injoignable au démarrage (%s) ; nouvel essai au premier appel.", self.label, describe(e))

    async def stop(self) -> None:
        client, self.client = self.client, None
        if client is None:
            return
        if self._session_id:
            try:   # fin de session polie ; sans importance si le serveur ne la gère pas
                await client.delete(self.url, headers=self._headers(), timeout=min(self.timeout, 3.0))
            except Exception:
                pass
        self._session_id = self._protocol = None
        await client.aclose()

    # --- exécution ---

    async def run(self, params: dict[str, Any], ctx: ActionContext) -> ActionResult:
        if missing := self._missing_config():
            return ActionResult.error(f"{self.label} n'est pas configuré : {missing}.")
        tool = params.get("tool")
        args = params.get("arguments")
        if args is None:   # petits modèles : paramètres parfois mis à plat à côté de « tool »
            args = {k: v for k, v in params.items() if k not in ("tool", "arguments")}
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except ValueError:
                return ActionResult.error("« arguments » doit être un objet JSON.")
        if not isinstance(tool, str) or not tool.strip() or not isinstance(args, dict):
            return ActionResult.error(f"Indique « tool » (nom de l'outil {self.label}) et « arguments » (objet).")
        tool = tool.strip()
        args = {k: v for k, v in args.items() if v is not None}

        try:
            if self.tools is None or (tool not in self.tools and time.monotonic() - self._last_refresh > REFRESH_DELAY):
                await self._load_tools()
        except Exception as e:
            return ActionResult.error(f"{self.label} indisponible : {describe(e)}")
        if not self._permitted(tool):
            return ActionResult.error(f"Outil {self.label} non autorisé : {tool[:60]}.")
        if tool not in (self.tools or {}):
            return ActionResult.error(f"Outil {self.label} inconnu : {tool[:60]}. Disponibles : {', '.join(self.tools or {}) or 'aucun'}.")

        if reason := await self._needs_confirmation(tool, args):
            if not await ctx.confirm(f"{self.label} : {tool} sur {reason}. Je confirme ?"):
                return ActionResult.error("Action annulée : non confirmée par l'utilisateur.")

        try:
            result = await self._request("tools/call", {"name": tool, "arguments": args})
        except Exception as e:
            return ActionResult.error(f"{self.label} : {describe(e)}")
        text = self._result_text(result)
        if result.get("isError"):
            hint = f" Paramètres de {tool} : {_param_summary(self.tools[tool].get('inputSchema') or {})}."
            return ActionResult.error(_clip(f"{text or 'Erreur.'}{hint}", self.max_result_chars))
        return ActionResult(text=_clip(text or "OK.", self.max_result_chars))

    def _permitted(self, tool: str) -> bool:
        return tool not in self.denied_tools and (self.allowed_tools is None or tool in self.allowed_tools)

    def _result_text(self, result: dict[str, Any]) -> str:
        parts = []
        for item in result.get("content") or []:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(self._item_text(str(item.get("text", ""))))
            elif isinstance(item, dict):
                parts.append(f"[contenu {item.get('type', '?')} non affiché]")
        if not parts and result.get("structuredContent") is not None:
            parts.append(json.dumps(result["structuredContent"], ensure_ascii=False))
        return redact("\n".join(parts).strip())

    # --- client MCP (Streamable HTTP) ---

    async def _load_tools(self) -> None:
        async with self._lock:
            self._last_refresh = time.monotonic()
            tools: dict[str, dict[str, Any]] = {}
            cursor = None
            for _ in range(MAX_PAGES):
                result = await self._request("tools/list", {"cursor": cursor} if cursor else {})
                for t in result.get("tools") or []:
                    if isinstance(t, dict) and isinstance(t.get("name"), str) and self._permitted(t["name"]):
                        tools[t["name"]] = t
                cursor = result.get("nextCursor")
                if not cursor:
                    break
            self.tools = tools

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        if auth := self._auth_header():
            h["Authorization"] = auth
        if self._session_id:
            h["Mcp-Session-Id"] = self._session_id
        if self._protocol:
            h["MCP-Protocol-Version"] = self._protocol
        return h

    def _client(self) -> Any:
        if self.client is None:
            import httpx

            self.client = httpx.AsyncClient(timeout=self.timeout, verify=self.verify_ssl)
        return self.client

    async def _ensure_session(self) -> None:
        async with self._session_lock:
            if self._protocol is None:
                await self._initialize()

    async def _initialize(self) -> None:
        self._session_id = None
        result = await self._post({
            "jsonrpc": "2.0", "id": self._new_id(), "method": "initialize",
            "params": {"protocolVersion": PROTOCOL_VERSION, "capabilities": {},
                       "clientInfo": {"name": "seldon", "version": "0.1.0"}},
        }, initializing=True)
        self._protocol = str(result.get("protocolVersion") or PROTOCOL_VERSION)
        await self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    async def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Appel JSON-RPC ; une session expirée (404) est rouverte une fois."""
        for attempt in (1, 2):
            await self._ensure_session()
            try:
                return await self._post({"jsonrpc": "2.0", "id": self._new_id(), "method": method, "params": params})
            except _SessionExpired:
                self._session_id = self._protocol = None
                if attempt == 2:
                    raise MCPError(f"session MCP refusée par {self.label}") from None
        raise AssertionError("inaccessible")

    async def _post(self, message: dict[str, Any], initializing: bool = False) -> dict[str, Any]:
        """Envoie un message ; renvoie ``result`` (requête) ou {} (notification)."""
        import httpx

        client = self._client()
        try:
            async with client.stream("POST", self.url, json=message, headers=self._headers()) as resp:
                if resp.status_code in (401, 403):
                    raise MCPError(f"accès refusé (HTTP {resp.status_code}) : {self.auth_hint}")
                if resp.status_code == 404 and self._session_id and not initializing:
                    raise _SessionExpired
                if resp.status_code == 404:
                    raise MCPError(f"serveur MCP introuvable (HTTP 404) : {self.not_found_hint}")
                if resp.status_code >= 400:
                    body = (await resp.aread())[:300].decode("utf-8", "replace")
                    raise MCPError(f"HTTP {resp.status_code} : {redact(body.strip()) or resp.reason_phrase}")
                if initializing and (sid := resp.headers.get("mcp-session-id")):
                    self._session_id = sid
                if "id" not in message:   # notification : 202 sans corps attendu
                    return {}
                reply = await self._read_reply(resp, message["id"])
        except httpx.TimeoutException:
            raise MCPError(f"{self.label} ne répond pas (plus de {self.timeout:g} s)") from None
        except httpx.HTTPError as e:
            raise MCPError(f"connexion impossible à {self.url} ({type(e).__name__})") from None
        if "error" in reply:
            err = reply["error"] if isinstance(reply["error"], dict) else {"message": str(reply["error"])}
            raise MCPError(f"erreur MCP {err.get('code', '')} : {redact(str(err.get('message', '')))[:300]}".replace("  ", " "))
        result = reply.get("result")
        if not isinstance(result, dict):
            raise MCPError("réponse MCP sans résultat")
        return result

    @staticmethod
    async def _read_reply(resp: Any, msg_id: int) -> dict[str, Any]:
        """Réponse JSON-RPC ``msg_id`` d'un corps JSON ou d'un flux SSE (lu au plus ``MAX_BODY`` octets)."""
        ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
        if ctype == "text/event-stream":
            lines: list[str] = []
            size = 0
            async for line in resp.aiter_lines():
                size += len(line) + 1
                if size > MAX_BODY:
                    raise MCPError("réponse MCP trop volumineuse")
                lines.append(line)
                if not line.strip():   # fin d'un événement : la réponse est peut-être arrivée
                    for msg in _sse_messages(lines):
                        if msg.get("id") == msg_id:
                            return msg
                    lines = []
            for msg in _sse_messages(lines):
                if msg.get("id") == msg_id:
                    return msg
            raise MCPError("flux MCP terminé sans réponse")
        body = b""
        async for chunk in resp.aiter_bytes():
            body += chunk
            if len(body) > MAX_BODY:
                raise MCPError("réponse MCP trop volumineuse")
        try:
            data = json.loads(body)
        except ValueError:
            raise MCPError(f"réponse MCP illisible (type {ctype or 'inconnu'})") from None
        for msg in data if isinstance(data, list) else [data]:
            if isinstance(msg, dict) and msg.get("id") == msg_id:
                return msg
        raise MCPError("réponse MCP sans identifiant correspondant")


class _SessionExpired(Exception):
    pass
