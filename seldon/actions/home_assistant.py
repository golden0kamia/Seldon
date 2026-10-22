"""Action « home_assistant » : pilote la maison via le serveur MCP de Home Assistant.

Home Assistant expose ses outils d'Assist (HassTurnOn, HassTurnOff, HassLightSet,
GetLiveContext…, plus les scripts exposés) par l'intégration « Model Context
Protocol Server », en Streamable HTTP sur ``/api/mcp`` (POST JSON-RPC,
authentification ``Authorization: Bearer <jeton longue durée>``). Ce module
contient un petit client MCP (initialize, tools/list, tools/call) basé sur httpx.

Le pipeline associe un nom d'outil à une action : tous les outils HA passent donc
par un seul outil ``home_assistant`` dont les paramètres sont ``tool`` (nom de
l'outil HA) et ``arguments`` (ses paramètres) ; la description détaille chaque
outil HA. La liste est chargée au démarrage — si Home Assistant est injoignable,
Seldon démarre quand même et réessaie au premier appel.

Le jeton n'est jamais à écrire dans le code : option ``token`` ou, à défaut,
variable d'environnement ``HA_TOKEN`` (ou ``HOME_ASSISTANT_TOKEN``).

Sécurité : une confirmation vocale est demandée pour les outils de
``confirm_tools``, et pour les commandes (allumer/éteindre, ouvrir/fermer…)
visant un domaine de ``confirm_domains`` (serrure, alarme…) ou un appareil dont
le nom contient un mot de ``confirm_keywords`` (portail, garage…). Le domaine
d'un appareil désigné par son seul nom est retrouvé grâce à GetLiveContext.
Seules les entités exposées à Assist dans Home Assistant sont accessibles.

Exemple ``config.yaml`` :

    actions:
      - engine: home_assistant
        options:
          url: http://192.168.1.105:8123/api/mcp
          # token : variable d'environnement HA_TOKEN (ne pas l'écrire ici)
          timeout: 10
          confirm_domains: [lock, alarm_control_panel]
          denied_tools: [HassBroadcast]
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from typing import Any

from seldon.actions.base import Action
from seldon.core.errors import SeldonError, describe, redact
from seldon.core.registry import register
from seldon.core.types import ActionContext, ActionResult

log = logging.getLogger(__name__)

PROTOCOL_VERSION = "2025-06-18"     # version proposée ; celle du serveur est ensuite utilisée
TOKEN_ENV = ("HA_TOKEN", "HOME_ASSISTANT_TOKEN")
MAX_BODY = 2_000_000                # octets lus au plus par réponse HTTP
MAX_PAGES = 20                      # pages de tools/list
REFRESH_DELAY = 30.0                # s entre deux rechargements de la liste d'outils
CONTEXT_TTL = 60.0                  # s de validité du contexte (noms -> domaines)

# Outils de commande dont la cible peut être une serrure, un portail, une alarme…
CONTROL_TOOLS = {"HassTurnOn", "HassTurnOff", "HassToggle", "HassSetPosition"}
SENSITIVE_DEVICE_CLASSES = {"garage", "gate", "door", "garage_door", "lock"}


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
    """« name, area, domain (light|switch…), brightness* » : paramètres d'un outil HA (* = obligatoire)."""
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


def _unwrap(text: str) -> str:
    """Le texte des réponses HA est souvent du JSON ``{"success": …, "result": "…"}`` : garde ``result``."""
    try:
        data = json.loads(text)
    except ValueError:
        return text
    if isinstance(data, dict) and isinstance(data.get("result"), str) and set(data) <= {"success", "result"}:
        return data["result"]
    return text


def _domains_by_name(context: str) -> dict[str, str]:
    """Nom d'appareil (minuscules) -> domaine, d'après le texte de GetLiveContext."""
    out: dict[str, str] = {}
    names: list[str] = []
    for line in context.splitlines():
        if m := re.match(r"\s*-\s*names:\s*(.+)", line):
            names = [n.strip().strip("'\"").lower() for n in m.group(1).split(",") if n.strip()]
        elif (m := re.match(r"\s+domain:\s*(\S+)", line)) and names:
            for n in names:
                out.setdefault(n, m.group(1).strip("'\""))
            names = []
    return out


@register("actions", "home_assistant")
class HomeAssistant(Action):
    name = "home_assistant"

    def __init__(
        self,
        url: str = "http://192.168.1.105:8123/api/mcp",   # serveur MCP de Home Assistant (Streamable HTTP)
        token: str | None = None,   # jeton longue durée ; None : variable d'environnement HA_TOKEN
        timeout: float = 10.0,   # délai max d'une requête (s)
        confirm_tools: list[str] | None = None,   # outils HA toujours soumis à confirmation
        confirm_domains: list[str] | None = ("lock", "alarm_control_panel"),   # domaines dont les commandes sont à confirmer (serrure, alarme)
        confirm_keywords: list[str] | None = ("portail", "garage", "serrure", "verrou", "alarme"),   # mots du nom d'appareil à confirmer
        allowed_tools: list[str] | None = None,   # si renseigné : seuls ces outils HA sont proposés
        denied_tools: list[str] | None = None,   # outils HA jamais proposés
        max_result_chars: int = 4000,   # longueur max du résultat renvoyé au modèle
        verify_ssl: bool = True,   # vérifier le certificat (https)
    ) -> None:
        self.url = url
        self._token = token or next((v for k in TOKEN_ENV if (v := os.environ.get(k))), None)
        self.timeout = float(timeout)
        self.confirm_tools = set(confirm_tools or ())
        self.confirm_domains = {d.lower() for d in confirm_domains or ()}
        self.confirm_keywords = [k.lower() for k in confirm_keywords or ()]
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
        self._context: tuple[float, dict[str, str]] | None = None

    # --- définition de l'outil pour le modèle ---

    @property
    def description(self) -> str:  # type: ignore[override]
        head = (
            "Pilote la maison connectée via Home Assistant : lumières, prises, volets, chauffage, "
            "médias, minuteurs, listes… Choisis l'outil Home Assistant dans « tool » et mets ses "
            "paramètres dans « arguments », par exemple {\"tool\": \"HassTurnOn\", \"arguments\": "
            "{\"name\": \"lampe du salon\"}}. Désigne un appareil par « name » ou une pièce par "
            "« area » (noms tels qu'ils apparaissent dans Home Assistant). Pour connaître les "
            "appareils, leurs noms ou leur état, appelle d'abord GetLiveContext."
        )
        if not self.tools:
            return head + " (Liste des outils indisponible pour l'instant : elle sera chargée au premier appel.)"
        lines = [f"- {name} : {_first_sentence(t.get('description', ''))} Paramètres : "
                 f"{_param_summary(t.get('inputSchema') or {})}" for name, t in self.tools.items()]
        return _clip(head + "\nOutils disponibles (* = obligatoire) :\n" + "\n".join(lines), 6000)

    @property
    def input_schema(self) -> dict[str, Any]:  # type: ignore[override]
        tool: dict[str, Any] = {"type": "string", "description": "Nom de l'outil Home Assistant (ex. HassTurnOn)."}
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
        if not self._token:
            log.warning("Home Assistant : aucun jeton (option token ou variable %s) — action inutilisable.", TOKEN_ENV[0])
            return
        try:
            await self._load_tools()
            log.info("Home Assistant : %d outil(s) MCP disponible(s).", len(self.tools or {}))
        except Exception as e:   # HA éteint au démarrage : on réessaiera au premier appel
            log.warning("Home Assistant injoignable au démarrage (%s) ; nouvel essai au premier appel.", describe(e))

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
        if not self._token:
            return ActionResult.error("Home Assistant n'est pas configuré : jeton d'accès manquant (variable HA_TOKEN).")
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
            return ActionResult.error("Indique « tool » (nom de l'outil Home Assistant) et « arguments » (objet).")
        tool = tool.strip()
        args = {k: v for k, v in args.items() if v is not None}

        try:
            if self.tools is None or (tool not in self.tools and time.monotonic() - self._last_refresh > REFRESH_DELAY):
                await self._load_tools()
        except Exception as e:
            return ActionResult.error(f"Home Assistant indisponible : {describe(e)}")
        if not self._permitted(tool):
            return ActionResult.error(f"Outil Home Assistant non autorisé : {tool[:60]}.")
        if tool not in (self.tools or {}):
            return ActionResult.error(f"Outil Home Assistant inconnu : {tool[:60]}. Disponibles : {', '.join(self.tools or {}) or 'aucun'}.")

        if reason := await self._needs_confirmation(tool, args):
            if not await ctx.confirm(f"Home Assistant : {tool} sur {reason}. Je confirme ?"):
                return ActionResult.error("Action annulée : non confirmée par l'utilisateur.")

        try:
            result = await self._request("tools/call", {"name": tool, "arguments": args})
        except Exception as e:
            return ActionResult.error(f"Home Assistant : {describe(e)}")
        text = self._result_text(result)
        if result.get("isError"):
            hint = f" Paramètres de {tool} : {_param_summary(self.tools[tool].get('inputSchema') or {})}."
            return ActionResult.error(_clip(f"{text or 'Erreur.'}{hint}", self.max_result_chars))
        return ActionResult(text=_clip(text or "OK.", self.max_result_chars))

    def _permitted(self, tool: str) -> bool:
        return tool not in self.denied_tools and (self.allowed_tools is None or tool in self.allowed_tools)

    @staticmethod
    def _result_text(result: dict[str, Any]) -> str:
        parts = []
        for item in result.get("content") or []:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(_unwrap(str(item.get("text", ""))))
            elif isinstance(item, dict):
                parts.append(f"[contenu {item.get('type', '?')} non affiché]")
        if not parts and result.get("structuredContent") is not None:
            parts.append(json.dumps(result["structuredContent"], ensure_ascii=False))
        return redact("\n".join(parts).strip())

    # --- confirmation ---

    async def _needs_confirmation(self, tool: str, args: dict[str, Any]) -> str | None:
        """Description de la cible si l'appel doit être confirmé, sinon None."""
        target = ", ".join(f"{k} « {v} »" for k, v in args.items()
                           if k in ("name", "area", "floor", "domain", "device_class")) or "toute la maison"
        if tool in self.confirm_tools:
            return target
        if tool not in CONTROL_TOOLS:
            return None

        def as_set(value: Any) -> set[str]:
            values = value if isinstance(value, list) else [value]
            return {str(v).lower() for v in values if v is not None}

        if as_set(args.get("domain")) & self.confirm_domains:
            return target
        # portes, portails, garages : confirmés tant que la confirmation par domaine est active
        if as_set(args.get("device_class")) & SENSITIVE_DEVICE_CLASSES and self.confirm_domains:
            return target
        text = " ".join(str(args.get(k, "")) for k in ("name", "area")).lower()
        if any(k in text for k in self.confirm_keywords):
            return target
        name = args.get("name")
        if isinstance(name, str) and self.confirm_domains and "domain" not in args:
            domain = (await self._domains()).get(name.strip().lower())
            if domain in self.confirm_domains:
                return target
        return None

    async def _domains(self) -> dict[str, str]:
        """Nom -> domaine des appareils exposés (GetLiveContext, en cache) ; {} si indisponible."""
        now = time.monotonic()
        if self._context and now - self._context[0] < CONTEXT_TTL:
            return self._context[1]
        mapping: dict[str, str] = {}
        if "GetLiveContext" in (self.tools or {}):
            try:
                result = await self._request("tools/call", {"name": "GetLiveContext", "arguments": {}})
                if not result.get("isError"):
                    mapping = _domains_by_name(self._result_text(result))
            except Exception as e:
                log.debug("GetLiveContext indisponible : %s", describe(e))
        self._context = (now, mapping)
        return mapping

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
        h = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
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
                    raise MCPError("session MCP refusée par Home Assistant") from None
        raise AssertionError("inaccessible")

    async def _post(self, message: dict[str, Any], initializing: bool = False) -> dict[str, Any]:
        """Envoie un message ; renvoie ``result`` (requête) ou {} (notification)."""
        import httpx

        client = self._client()
        try:
            async with client.stream("POST", self.url, json=message, headers=self._headers()) as resp:
                if resp.status_code in (401, 403):
                    raise MCPError(f"accès refusé (HTTP {resp.status_code}) : jeton invalide, expiré ou "
                                   "compte non administrateur")
                if resp.status_code == 404 and self._session_id and not initializing:
                    raise _SessionExpired
                if resp.status_code == 404:
                    raise MCPError("serveur MCP introuvable (HTTP 404) : intégration « Model Context Protocol "
                                   "Server » non installée ou URL erronée")
                if resp.status_code >= 400:
                    body = (await resp.aread())[:300].decode("utf-8", "replace")
                    raise MCPError(f"HTTP {resp.status_code} : {redact(body.strip()) or resp.reason_phrase}")
                if initializing and (sid := resp.headers.get("mcp-session-id")):
                    self._session_id = sid
                if "id" not in message:   # notification : 202 sans corps attendu
                    return {}
                reply = await self._read_reply(resp, message["id"])
        except httpx.TimeoutException:
            raise MCPError(f"Home Assistant ne répond pas (plus de {self.timeout:g} s)") from None
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
