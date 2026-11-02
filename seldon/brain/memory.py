"""Mémoire des conversations (plugin ``memory:``) et son outil ``conversations``.

Chaque conversation est un fichier JSON privé dans ``dir`` :

    {"id": "20261031-142500-ab12", "title": "…", "created": …, "updated": …,
     "speakers": ["Bastien"],
     "messages": [{"role": "user" | "assistant", "text": "…", "speaker": "Bastien", "time": …}]}

``messages`` est l'historique du cerveau (``Brain.history``) : il est
sauvegardé après chaque phrase. Après ``new_after_idle_s`` secondes de silence,
une nouvelle conversation commence (l'ancienne reste disponible). Le cerveau
retrouve, reprend ou supprime une conversation avec l'outil ``conversations``.

Les identifiants viennent souvent du cerveau : seuls ceux de la forme
``[\\w-]+`` désignent un fichier, toujours dans ``dir``.
"""

from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from seldon.base import Context, Result, Tool
from seldon.storage import write_private

if TYPE_CHECKING:
    from seldon.brain.brain import Brain

log = logging.getLogger(__name__)

_ID = re.compile(r"[\w-]{1,100}")


def valid_id(conv_id: Any) -> bool:
    """Vrai si ``conv_id`` peut désigner un fichier de conversation (sans sortir du dossier)."""
    return isinstance(conv_id, str) and _ID.fullmatch(conv_id) is not None


def _fold(text: str) -> str:
    """Texte sans accents ni majuscules, pour comparer."""
    return unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()


@dataclass
class Conversation:
    id: str
    title: str = ""
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    speakers: list[str] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)

    def describe(self) -> str:
        when = datetime.fromtimestamp(self.updated).strftime("%d/%m %H:%M")
        who = f" ({', '.join(self.speakers)})" if self.speakers else ""
        return f"{when} « {self.title or 'sans titre'} »{who}, {len(self.messages)} messages"

    def tail(self, n: int = 10) -> str:
        return "\n".join(f"{(m.get('speaker') or 'Utilisateur') if m['role'] == 'user' else 'Seldon'} : {m['text']}"
                         for m in self.messages[-n:])


def read(path: Path) -> Conversation | None:
    """Conversation lue depuis ``path`` ; None si le fichier est illisible."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        messages = [m for m in data.get("messages", [])
                    if isinstance(m, dict) and m.get("role") in ("user", "assistant") and isinstance(m.get("text"), str)]
        return Conversation(id=path.stem, title=str(data.get("title", "")), created=float(data.get("created", 0)),
                            updated=float(data.get("updated", 0)), speakers=[str(s) for s in data.get("speakers", [])],
                            messages=messages)
    except (OSError, ValueError, TypeError, AttributeError) as e:
        log.warning("Conversation illisible : %s (%s)", path, e)
        return None


def load_all(folder: str | Path) -> list[Conversation]:
    """Toutes les conversations du dossier, la plus récente d'abord."""
    found = (read(p) for p in Path(folder).glob("*.json"))
    return sorted((c for c in found if c), key=lambda c: c.updated, reverse=True)


class Memory:
    def __init__(self, brain: Brain, dir: str = "data/conversations", new_after_idle_s: float = 600) -> None:
        self.brain = brain
        self.dir = Path(dir).expanduser()
        self.new_after_idle_s = new_after_idle_s
        self.current = self._new()

    def tools(self) -> list[Tool]:
        return [Conversations()]

    @staticmethod
    def _new() -> Conversation:
        return Conversation(id=f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}")

    def _path(self, conv_id: str) -> Path:
        if not valid_id(conv_id):
            raise ValueError(f"identifiant de conversation invalide : {conv_id[:40]!r}")
        return self.dir / f"{conv_id}.json"

    # --- appelé par le pipeline ---
    def before_turn(self, now: float) -> None:
        """Après un long silence, la phrase suivante ouvre une nouvelle conversation."""
        if self.current.messages and now - self.current.updated > self.new_after_idle_s:
            self.new()

    def save(self) -> None:
        """Écrit la conversation en cours (après chaque phrase)."""
        conv = self.current
        conv.messages = list(self.brain.history)
        if not conv.messages:
            return
        conv.updated = time.time()
        conv.title = conv.title or next((m["text"][:60] for m in conv.messages if m["role"] == "user"), "")
        conv.speakers = sorted({m["speaker"] for m in conv.messages if m.get("speaker")})
        data = json.dumps(asdict(conv), ensure_ascii=False, indent=1)
        write_private(self._path(conv.id), data.encode("utf-8"))

    # --- appelé par l'outil « conversations » ---
    def new(self) -> None:
        self.current = self._new()
        self.brain.history = []

    def load(self, conv_id: str) -> Conversation | None:
        try:
            path = self._path(conv_id)
        except ValueError:
            return None
        # Un lien symbolique pourrait mener hors du dossier : refusé.
        return read(path) if path.is_file() and not path.is_symlink() else None

    def resume(self, conv: Conversation) -> None:
        self.current = conv
        self.brain.history = list(conv.messages)

    def search(self, query: str, limit: int = 8) -> list[Conversation]:
        words = [w for w in _fold(query).split() if len(w) > 2]
        found = []
        for conv in load_all(self.dir):
            if conv.id == self.current.id:
                continue
            text = _fold(conv.title + " " + " ".join(m["text"] for m in conv.messages))
            if all(w in text for w in words):
                found.append(conv)
        return found[:limit]

    def delete(self, conv_id: str) -> bool:
        conv = self.load(conv_id)
        if conv is None:
            return False
        self._path(conv.id).unlink(missing_ok=True)
        if conv.id == self.current.id:
            self.new()
        return True


class Conversations(Tool):
    name = "conversations"
    description = (
        "Mémoire des conversations passées. "
        "operation='chercher' : liste les conversations qui parlent de `query` (les plus récentes si vide). "
        "operation='reprendre' : reprend la conversation `id` et renvoie ses derniers échanges. "
        "operation='nouvelle' : commence une nouvelle conversation. "
        "operation='supprimer' : supprime la conversation `id` (l'utilisateur doit confirmer ; "
        "ne le fais que s'il le demande)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["chercher", "reprendre", "nouvelle", "supprimer"]},
            "query": {"type": "string", "description": "Mots-clés du sujet (pour 'chercher')."},
            "id": {"type": "string", "description": "Identifiant de la conversation."},
        },
        "required": ["operation"],
    }

    async def run(self, args: dict[str, Any], ctx: Context) -> Result:
        memory: Memory = ctx.app.memory
        op, conv_id = args.get("operation"), str(args.get("id") or "")
        if op == "chercher":
            found = memory.search(str(args.get("query") or ""))
            return Result("\n".join(f"- id={c.id} : {c.describe()}" for c in found) or "Aucune conversation trouvée.")
        if op == "nouvelle":
            memory.new()
            return Result("Nouvelle conversation commencée.")
        conv = memory.load(conv_id)
        if conv is None:
            return Result.fail("Conversation introuvable : utilise d'abord operation='chercher'.")
        if op == "reprendre":
            memory.resume(conv)
            return Result(f"Conversation « {conv.title} » reprise. Derniers échanges :\n{conv.tail()}")
        if op == "supprimer":
            if not await ctx.confirm(f"Supprimer la conversation « {conv.title or 'sans titre'} » ?"):
                return Result("L'utilisateur a refusé : rien n'a été supprimé.")
            memory.delete(conv.id)
            return Result("Conversation supprimée.")
        return Result.fail(f"Opération inconnue : {op}")
