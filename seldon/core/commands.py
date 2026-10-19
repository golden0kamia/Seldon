"""Commandes locales traitées avant le cerveau (aucun appel au modèle).

Une prise de parole déclenche une commande si son texte normalisé (minuscules,
sans ponctuation, « / » initial facultatif) correspond exactement à un alias :
« exit », « /exit », « Quitter. »… Pour en ajouter une :

    @command("heure", "quelle heure", help="Donne l'heure")
    async def _heure(assistant, args):
        return datetime.now().strftime("%H:%M")
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Awaitable, Callable

if TYPE_CHECKING:
    from seldon.core.pipeline import Assistant

Handler = Callable[["Assistant", str], Awaitable[str | None]]


@dataclass
class Command:
    aliases: tuple[str, ...]
    help: str
    handler: Handler


COMMANDS: list[Command] = []


def command(*aliases: str, help: str = "") -> Callable[[Handler], Handler]:
    def deco(fn: Handler) -> Handler:
        COMMANDS.append(Command(tuple(aliases), help, fn))
        return fn

    return deco


def normalize(text: str) -> str:
    text = text.strip().lower().lstrip("/")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s'-]", " ", text)).strip()


def match(text: str) -> tuple[Command, str] | None:
    """Commande correspondant au texte, avec les arguments éventuels."""
    norm = normalize(text)
    for cmd in COMMANDS:
        for alias in cmd.aliases:
            if norm == alias or (norm.startswith(alias + " ") and text.lstrip().startswith("/")):
                return cmd, norm[len(alias):].strip()
    return None


@command("exit", "quit", "quitter", "stop seldon", help="Quitte l'assistant")
async def _exit(assistant: Assistant, args: str) -> str | None:
    assistant.request_stop()
    return "Au revoir."


@command("nouvelle", "reset", "oublie tout", "nouvelle conversation",
         help="Commence une nouvelle conversation (l'actuelle reste en mémoire)")
async def _new(assistant: Assistant, args: str) -> str | None:
    if assistant.memory is None:
        assistant.brain.reset()
        return "Conversation effacée."
    assistant.memory.new()
    return "Nouvelle conversation."


_listing: list[str] = []   # identifiants du dernier affichage de /conversations


def _pick(assistant: Assistant, args: str):
    """Conversation désignée par un numéro de la dernière liste ou par des mots-clés."""
    memory = assistant.memory
    if args.isdigit() and 0 < int(args) <= len(_listing):
        return memory.load(_listing[int(args) - 1])
    found = memory.search(args, limit=1) if args else []
    return memory.load(found[0].id) if found else None


@command("conversations", "historique", help="Liste les conversations (/conversations [mots-clés])")
async def _conversations(assistant: Assistant, args: str) -> str | None:
    if assistant.memory is None:
        return "La mémoire des conversations est désactivée."
    found = assistant.memory.search(args, limit=15)
    _listing[:] = [c.id for c in found]
    current = assistant.memory.current.id if assistant.memory.current else None
    if not found:
        return "Aucune conversation trouvée."
    lines = [f"  {i}. {'*' if c.id == current else ' '} {c.describe()}" for i, c in enumerate(found, 1)]
    print("Conversations (/reprendre N) :\n" + "\n".join(lines), flush=True)
    return None


@command("reprendre", help="Reprend une conversation (/reprendre N ou /reprendre mots-clés)")
async def _resume(assistant: Assistant, args: str) -> str | None:
    if assistant.memory is None:
        return "La mémoire des conversations est désactivée."
    conv = _pick(assistant, args)
    if conv is None:
        return "Je n'ai pas trouvé cette conversation. Tape /conversations pour la liste."
    assistant.memory.resume(conv.id)
    last = next((m["text"] for m in reversed(conv.transcript) if m["role"] == "user"), "")
    return f"On reprend « {conv.title} ». Ta dernière question était : {last}"


@command("supprimer", help="Supprime une conversation (/supprimer N ou mots-clés)")
async def _delete(assistant: Assistant, args: str) -> str | None:
    if assistant.memory is None or not args:
        return "Précise la conversation : /supprimer N."
    conv = _pick(assistant, args)
    if conv is None or not await assistant.confirm(f"Supprimer « {conv.title} » ?"):
        return "Rien n'a été supprimé."
    assistant.memory.delete(conv.id)
    return "Conversation supprimée."


@command("actions", "outils", help="Liste les actions disponibles pour le cerveau")
async def _actions(assistant: Assistant, args: str) -> str | None:
    return "Actions : " + ", ".join(sorted(assistant.actions)) + "."


@command("help", "aide", help="Affiche cette aide")
async def _help(assistant: Assistant, args: str) -> str | None:
    lines = [f"  /{c.aliases[0]:<10} {c.help}  ({', '.join(c.aliases[1:])})" for c in COMMANDS]
    print("Commandes :\n" + "\n".join(lines), flush=True)
    return None
