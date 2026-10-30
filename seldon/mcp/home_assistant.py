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

    mcp:
      - engine: home_assistant
        options:
          url: http://192.168.1.105:8123/api/mcp
          # token : variable d'environnement HA_TOKEN (ne pas l'écrire ici)
          timeout: 10
          confirm_domains: [lock, alarm_control_panel]
          denied_tools: [HassBroadcast]
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

from seldon.core.errors import describe
from seldon.core.registry import register
from seldon.mcp.base import MCPAction, MCPError, _sse_messages  # noqa: F401 (réexportés)

log = logging.getLogger(__name__)

TOKEN_ENV = ("HA_TOKEN", "HOME_ASSISTANT_TOKEN")
CONTEXT_TTL = 60.0                  # s de validité du contexte (noms -> domaines)

# Outils de commande dont la cible peut être une serrure, un portail, une alarme…
CONTROL_TOOLS = {"HassTurnOn", "HassTurnOff", "HassToggle", "HassSetPosition"}
SENSITIVE_DEVICE_CLASSES = {"garage", "gate", "door", "garage_door", "lock"}


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


@register("mcp", "home_assistant")
class HomeAssistant(MCPAction):
    name = "home_assistant"
    label = "Home Assistant"
    example_tool = "HassTurnOn"
    intro = (
        "Pilote la maison connectée via Home Assistant : lumières, prises, volets, chauffage, "
        "médias, minuteurs, listes… Choisis l'outil Home Assistant dans « tool » et mets ses "
        "paramètres dans « arguments », par exemple {\"tool\": \"HassTurnOn\", \"arguments\": "
        "{\"name\": \"lampe du salon\"}}. Désigne un appareil par « name » ou une pièce par "
        "« area » (noms tels qu'ils apparaissent dans Home Assistant). Pour connaître les "
        "appareils, leurs noms ou leur état, appelle d'abord GetLiveContext."
    )
    auth_hint = "jeton invalide, expiré ou compte non administrateur"
    not_found_hint = "intégration « Model Context Protocol Server » non installée ou URL erronée"

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
        super().__init__(url, timeout=timeout, confirm_tools=confirm_tools, allowed_tools=allowed_tools,
                         denied_tools=denied_tools, max_result_chars=max_result_chars, verify_ssl=verify_ssl)
        self._token = token or next((v for k in TOKEN_ENV if (v := os.environ.get(k))), None)
        self.confirm_domains = {d.lower() for d in confirm_domains or ()}
        self.confirm_keywords = [k.lower() for k in confirm_keywords or ()]
        self._context: tuple[float, dict[str, str]] | None = None

    def _auth_header(self) -> str | None:
        return f"Bearer {self._token}" if self._token else None

    def _missing_config(self) -> str | None:
        return None if self._token else "jeton d'accès manquant (variable HA_TOKEN)"

    def _item_text(self, text: str) -> str:
        return _unwrap(text)

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
