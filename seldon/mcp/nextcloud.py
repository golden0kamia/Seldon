"""Action « nextcloud » : notes, agenda, tâches, contacts, fichiers… via un serveur MCP Nextcloud.

Prévue pour nextcloud-mcp-server (https://github.com/cbcoutinho/nextcloud-mcp-server),
lancé à côté de Seldon (Docker ou ``uvx``) ; il répond en Streamable HTTP sur
``http://127.0.0.1:8000/mcp`` et ses outils s'appellent en général ``nc_<app>_<action>``
(nc_notes_create_note, nc_calendar_list_events, nc_webdav_read_file…, mais deck_…, talk_…).

Le serveur expose plus d'une centaine d'outils, trop pour un petit modèle : seuls
ceux des applications de ``apps`` sont proposés (``null`` : toutes). Tous passent
par un seul outil ``nextcloud`` (``tool`` + ``arguments``, voir ``MCPAction``).

Authentification (``auth``) :
- ``none`` (défaut) : mode mono-utilisateur, le serveur MCP détient lui-même le
  mot de passe d'application Nextcloud ; Seldon n'envoie rien ;
- ``basic`` : mode ``multi_user_basic`` du serveur, Seldon envoie l'identifiant
  ``NEXTCLOUD_USERNAME`` et le mot de passe d'application ``NEXTCLOUD_PASSWORD`` ;
- ``bearer`` : jeton ``NEXTCLOUD_MCP_TOKEN`` (serveur derrière un proxy ou OAuth).
Les secrets ne s'écrivent jamais dans le code ni dans config.yaml (fichier .env).

Sécurité : une confirmation vocale est demandée pour les outils dont le nom
contient un mot de ``confirm_words`` (suppression, déplacement, envoi, partage…).

Exemple ``config.yaml`` :

    mcp:
      - engine: nextcloud
        options:
          url: http://127.0.0.1:8000/mcp
          apps: [notes, calendar, contacts]
"""

from __future__ import annotations

import base64
import os
from typing import Any

from seldon.core.registry import register
from seldon.mcp.base import MCPAction, _args_summary

AUTH_MODES = ("none", "basic", "bearer")
# application -> préfixe de ses outils, quand il diffère de « nc_<app>_ » (deck_…, talk_…)
TOOL_PREFIXES = {"sharing": "nc_share_", "shopping_list": "nc_shopping_", "deck": "deck_",
                 "talk": "talk_", "collectives": "collectives_"}


@register("mcp", "nextcloud")
class Nextcloud(MCPAction):
    name = "nextcloud"
    label = "Nextcloud"
    example_tool = "nc_notes_create_note"
    intro = (
        "Accède au Nextcloud de l'utilisateur : notes, agenda et tâches, contacts, fichiers… "
        "Choisis l'outil Nextcloud dans « tool » et mets ses paramètres dans « arguments », par "
        "exemple {\"tool\": \"nc_notes_search_notes\", \"arguments\": {\"query\": \"courses\"}}. "
        "Pour les prochains rendez-vous, utilise nc_calendar_get_upcoming_events ; pour retrouver "
        "un élément, cherche-le avant de le modifier ou de le supprimer."
    )
    max_description_chars = 9000
    auth_hint = "identifiants refusés (auth, NEXTCLOUD_USERNAME / NEXTCLOUD_PASSWORD ou NEXTCLOUD_MCP_TOKEN)"
    not_found_hint = "URL erronée (nextcloud-mcp-server répond sur /mcp)"

    def __init__(
        self,
        url: str = "http://127.0.0.1:8000/mcp",   # serveur MCP Nextcloud (Streamable HTTP)
        apps: list[str] | None = ("notes", "calendar", "contacts"),   # applications proposées (notes, calendar, contacts, webdav, deck, tables, cookbook, news, mail, sharing, talk…) ; null : toutes
        auth: str = "none",   # none (le serveur détient les identifiants) | basic | bearer
        timeout: float = 20.0,   # délai max d'une requête (s)
        confirm_words: list[str] | None = ("delete", "move", "restore", "bulk", "manage", "send", "share"),   # outil dont le nom contient l'un de ces mots : confirmation
        confirm_tools: list[str] | None = None,   # outils toujours soumis à confirmation
        allowed_tools: list[str] | None = None,   # si renseigné : seuls ces outils sont proposés
        denied_tools: list[str] | None = None,   # outils jamais proposés
        max_result_chars: int = 4000,   # longueur max du résultat renvoyé au modèle
        verify_ssl: bool = True,   # vérifier le certificat (https)
    ) -> None:
        super().__init__(url, timeout=timeout, confirm_tools=confirm_tools, allowed_tools=allowed_tools,
                         denied_tools=denied_tools, max_result_chars=max_result_chars, verify_ssl=verify_ssl)
        if auth not in AUTH_MODES:
            raise ValueError(f"auth doit valoir {', '.join(AUTH_MODES)}, pas {auth!r}")
        self.prefixes = tuple(TOOL_PREFIXES.get(a, f"nc_{a}_") for a in (x.lower().strip() for x in apps)) if apps else None
        self.auth = auth
        self.confirm_words = [w.lower() for w in confirm_words or ()]
        self._username = os.environ.get("NEXTCLOUD_USERNAME")
        self._password = os.environ.get("NEXTCLOUD_PASSWORD")
        self._token = os.environ.get("NEXTCLOUD_MCP_TOKEN")

    def _permitted(self, tool: str) -> bool:
        if self.prefixes is not None and not tool.startswith(self.prefixes):
            return False
        return super()._permitted(tool)

    def _auth_header(self) -> str | None:
        if self.auth == "basic" and self._username and self._password:
            return "Basic " + base64.b64encode(f"{self._username}:{self._password}".encode()).decode()
        if self.auth == "bearer" and self._token:
            return f"Bearer {self._token}"
        return None

    def _missing_config(self) -> str | None:
        if self.auth == "basic" and not (self._username and self._password):
            return "identifiants manquants (variables NEXTCLOUD_USERNAME et NEXTCLOUD_PASSWORD)"
        if self.auth == "bearer" and not self._token:
            return "jeton manquant (variable NEXTCLOUD_MCP_TOKEN)"
        return None

    async def _needs_confirmation(self, tool: str, args: dict[str, Any]) -> str | None:
        if reason := await super()._needs_confirmation(tool, args):
            return reason
        if any(w in tool.lower() for w in self.confirm_words):
            return _args_summary(args)
        return None
