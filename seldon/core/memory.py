"""Mémoire des conversations : sauvegarde, liste et reprise.

Chaque conversation est un fichier JSON dans ``data/conversations/`` :

    {id, title, created, updated, brain, speakers,
     transcript: [{role, text, speaker, ts}],   # ce qui a été dit (neutre)
     state: {...}}                               # historique complet du cerveau

Une conversation est sauvegardée après chaque prise de parole. Après une
période d'inactivité, la suivante démarre une nouvelle conversation (l'ancienne
reste disponible). Une conversation se reprend avec son historique complet si
le cerveau est du même type, sinon à partir de la transcription.

Un changement demandé par le cerveau (outil ``conversations``) pendant un tour
n'est appliqué qu'à la fin de ce tour, pour ne pas modifier son historique en
cours de route. Le cerveau peut aussi renommer une conversation, en retirer des
messages ou la supprimer (après confirmation, voir l'action ``conversations``).

Au premier message d'une nouvelle conversation, ``related_note()`` cherche si le
sujet a déjà été abordé et invite le cerveau à proposer de le poursuivre.

``compress_all()``, appelé périodiquement par le pipeline, résume (avec le
cerveau, ou localement à défaut) le début des conversations trop volumineuses :
les anciens messages sont remplacés par une entrée ``role: "summary"``.

La mémoire sait aussi quand chaque personne a parlé pour la dernière fois :
``absence_note()`` signale au cerveau quelqu'un qui revient après une longue
absence (ou qui lui parle pour la première fois), pour qu'il l'accueille.

Sécurité : les identifiants viennent souvent du cerveau (donc de ce qu'on lui
dit) ; seuls ceux de la forme ``[\\w-]+`` désignent un fichier, toujours dans
``dir``, et l'identifiant d'une conversation lue est le nom de son fichier. Les
fichiers sont privés et écrits de façon atomique (``seldon.core.storage``).

Performance : les conversations lues sur disque (sans l'état du cerveau) et leurs
mots sont gardés en cache et relus seulement si leur fichier change (inode, date,
taille) : chercher un sujet ne relit pas tout le dossier à chaque phrase.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
import unicodedata
import uuid
from collections import Counter
from contextlib import AbstractAsyncContextManager, nullcontext
from dataclasses import dataclass, field, fields
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from seldon.brain.base import Brain
from seldon.brain.prompts import neutralize
from seldon.core.action import Action
from seldon.core.conversations import Conversations
from seldon.core.storage import write_private

log = logging.getLogger(__name__)

# Identifiant de conversation = nom de son fichier. Format de _new() : AAAAMMJJ-HHMMSS-xxxx ; les
# anciens noms (lettres, chiffres, _ et -) restent acceptés. Ni « / » ni « . » : jamais hors du dossier.
_ID = re.compile(r"[\w-]{1,100}")

_STOPWORDS = set(
    "le la les un une des de du et ou en a au aux sur pour par avec dans que qui quoi est "
    "on je tu il elle nous vous ils ce cette ces mon ma mes ton ta tes son sa ses pas ne "
    "conversation discussion parlait parle reprendre reprends sujet".split()
)


# mots trop courants pour caractériser un sujet dans un résumé
_FILLER = set(
    "peux peut pouvez faire fais fait bien oui non merci bonjour salut alors aussi plus tout tous "
    "comme mais donc sont suis avez veux voudrais quel quelle quels est-ce cela ceci voila voici "
    "tres trop encore deja juste etre avoir dit dis dire seldon".split()
)


def valid_id(conv_id: Any) -> bool:
    """Vrai si ``conv_id`` peut désigner un fichier de conversation (sans sortir du dossier)."""
    return isinstance(conv_id, str) and _ID.fullmatch(conv_id) is not None


_WORD = re.compile(r"[^\W_]+")   # suite de caractères alphanumériques (str.isalnum)


@lru_cache(maxsize=1 << 16)   # le vocabulaire se répète beaucoup d'une conversation à l'autre
def _fold(word: str) -> str:
    return unicodedata.normalize("NFKD", word.lower()).encode("ascii", "ignore").decode()


def _tokens(text: str) -> dict[str, str]:
    """Mots significatifs : forme sans accents ni majuscules -> forme d'origine (en minuscules)."""
    out: dict[str, str] = {}
    for word in _WORD.findall(text):
        key = _fold(word)
        if len(key) > 2 and key not in _STOPWORDS:
            out.setdefault(key, word.lower())
    return out


def _words(text: str) -> set[str]:
    return {key for word in set(_WORD.findall(text)) if len(key := _fold(word)) > 2 and key not in _STOPWORDS}


def _duration(seconds: float) -> str:
    """Durée approximative en toutes lettres : « 3 heures », « 2 semaines »…"""
    for unit, size, until in (("heure", 3600, 86400), ("jour", 86400, 14 * 86400), ("semaine", 7 * 86400, 61 * 86400)):
        if seconds < until:
            n = max(1, round(seconds / size))
            return f"{n} {unit}{'s' if n > 1 else ''}"
    return f"{round(seconds / (30 * 86400))} mois"


@dataclass
class Conversation:
    id: str
    created: float
    updated: float
    brain: str
    title: str = ""
    speakers: list[str] = field(default_factory=list)
    transcript: list[dict[str, Any]] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        when = datetime.fromtimestamp(self.updated).strftime("%d/%m %H:%M")
        who = f" ({', '.join(self.speakers)})" if self.speakers else ""
        return f"{when} « {self.title or 'sans titre'} »{who}, {len(self.transcript)} messages"

    def topics(self, n: int = 6) -> list[str]:
        """Mots les plus présents dans les messages (le titre compte double)."""
        counts: Counter[str] = Counter()
        shown: dict[str, str] = {}
        for text in [self.title, *(m["text"] for m in self.transcript)]:
            tokens = {k: w for k, w in _tokens(text).items() if k not in _FILLER}
            counts.update(tokens.keys())
            for k, w in tokens.items():
                shown.setdefault(k, w)
        best = [k for k, c in counts.most_common(n) if c > 1] or [k for k, _ in counts.most_common(min(n, 3))]
        return [shown[k] for k in best]

    def summary(self, requests: int = 3, chars: int = 140) -> str:
        """Résumé calculé localement (sans le cerveau) : période, participants, thèmes, demandes."""
        def short(text: str) -> str:
            text = " ".join(text.split())
            return text if len(text) <= chars else text[:chars].rsplit(" ", 1)[0] + "…"

        start, end = datetime.fromtimestamp(self.created), datetime.fromtimestamp(self.updated)
        minutes = round((self.updated - self.created) / 60)
        lines = [f"Le {start:%d/%m/%Y} de {start:%H:%M} à {end:%H:%M}"
                 + (f" ({minutes} min)" if minutes else "") + f", {len(self.transcript)} messages."]
        lines.append(f"Participants : {', '.join(self.speakers) if self.speakers else 'locuteur inconnu'}.")
        if topics := self.topics():
            lines.append(f"Thèmes : {', '.join(topics)}.")
        asked = [m["text"] for m in self.transcript if m["role"] == "user"]
        if asked:
            lines.append("Demandes :")
            lines += [f"• {short(t)}" for t in asked[:requests]]
            if len(asked) > requests:
                lines.append(f"• … et {len(asked) - requests} autre(s)")
        if replies := [m["text"] for m in self.transcript if m["role"] == "assistant"]:
            lines.append(f"Dernière réponse : {short(replies[-1])}")
        return "\n".join(lines)

    def tail(self, n: int = 10) -> str:
        return "\n".join(f"{_ROLE_LABELS.get(m['role'], 'Seldon')} : {m['text']}" for m in self.transcript[-n:])

    def size(self) -> int:
        """Taille de la transcription, en caractères."""
        return sum(len(m["text"]) for m in self.transcript)

    def matching(self, text: str) -> list[dict[str, Any]]:
        """Messages (hors résumés) qui contiennent ``text``, sans tenir compte des accents ni de la casse."""
        needle = _fold(" ".join(text.split()))
        return [m for m in self.transcript if needle and m["role"] != "summary" and needle in _fold(" ".join(m["text"].split()))]


_ROLE_LABELS = {"user": "Utilisateur", "assistant": "Seldon", "summary": "Résumé des échanges précédents"}

SUMMARY_INSTRUCTIONS = (
    "Résume en français, en quelques phrases, cet extrait de conversation entre un utilisateur "
    "et Seldon, son assistant vocal. Garde les faits, décisions, préférences, noms, chiffres et "
    "demandes encore en suspens ; omets les politesses. Réponds uniquement par le résumé."
)


def _number(value: Any) -> bool:
    """Horodatage utilisable : nombre fini (ni booléen, ni NaN), dans les limites de datetime."""
    return type(value) in (int, float) and -1e11 < value < 1e11


def _conversation(data: Any, conv_id: str, with_state: bool = False) -> Conversation:
    """Conversation lue dans ``<conv_id>.json`` (son identifiant est toujours le nom du fichier).

    Les clés inconnues sont ignorées ; ValueError si une donnée utile est invalide.
    """
    if not isinstance(data, dict):
        raise ValueError("objet JSON attendu")
    created, updated, brain = data.get("created"), data.get("updated"), data.get("brain")
    if not (_number(created) and _number(updated) and isinstance(brain, str)):
        raise ValueError("champs created / updated / brain invalides")
    title, speakers, transcript = data.get("title") or "", data.get("speakers") or [], data.get("transcript") or []
    if not (isinstance(title, str) and isinstance(speakers, list) and all(isinstance(s, str) for s in speakers)
            and isinstance(transcript, list)):
        raise ValueError("champs title / speakers / transcript invalides")
    for m in transcript:
        if not (isinstance(m, dict) and isinstance(m.get("role"), str) and isinstance(m.get("text"), str)
                and isinstance(m.get("speaker"), (str, type(None)))):
            raise ValueError("message invalide dans la transcription")
        if not _number(m.get("ts")):
            m["ts"] = updated
    state = data.get("state") if with_state else None
    return Conversation(id=conv_id, created=created, updated=updated, brain=brain, title=title,
                        speakers=speakers, transcript=transcript, state=state if isinstance(state, dict) else {})


def _read(path: Path, conv_id: str, with_state: bool = False) -> Conversation | None:
    """Conversation du fichier ``path`` ; None (avec un avertissement) s'il est illisible."""
    try:
        return _conversation(json.loads(path.read_text(encoding="utf-8")), conv_id, with_state)
    except FileNotFoundError:   # supprimée entre-temps
        return None
    except (OSError, ValueError, TypeError, RecursionError) as e:
        log.warning("Conversation illisible : %s (%s)", path, e)
        return None


def _valid_state(state: dict[str, Any]) -> bool:
    """État du cerveau restaurable tel quel (sinon : reconstruit à partir de la transcription)."""
    messages = state.get("messages")
    return bool(messages) and isinstance(messages, list) and all(isinstance(m, dict) for m in messages)


def _same(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return (a["role"], a["text"], a["ts"]) == (b["role"], b["text"], b["ts"])


@dataclass
class _Indexed:
    """Conversation lue sur disque (sans l'état du cerveau) et ses mots, gardés en cache."""

    stamp: tuple[int, int, int]   # (inode, date de modification, taille) du fichier lu
    conv: Conversation
    title_words: set[str]
    body_words: set[str]

    @classmethod
    def of(cls, conv: Conversation, stamp: tuple[int, int, int] = (0, 0, 0)) -> _Indexed:
        return cls(stamp, conv, _words(conv.title), _words(" ".join(m["text"] for m in conv.transcript)))


class ConversationMemory:
    def __init__(
        self,
        brain: Brain,
        brain_kind: str,
        dir: str = "data/conversations",
        new_after_idle_s: float = 600,
        title_chars: int = 70,
        greet_after_s: float | None = 6 * 3600,   # absence au-delà de laquelle Seldon accueille ; None = jamais
        suggest_related: bool = True,   # nouvelle conversation sur un sujet déjà abordé : propose de le poursuivre
        related_min_words: int = 2,   # mots du sujet en commun pour juger qu'il a déjà été abordé
        compress_every_s: float | None = 1800,   # vérification des conversations trop longues ; None = jamais
        compress_max_chars: int = 20000,   # au-delà, le début de la conversation est résumé
        compress_keep: int = 12,   # derniers messages gardés tels quels lors d'un résumé
    ) -> None:
        self.brain = brain
        self.brain_kind = brain_kind
        self.dir = Path(dir).expanduser()
        self.new_after_idle_s = new_after_idle_s
        self.title_chars = title_chars
        self.greet_after_s = greet_after_s
        self.suggest_related = suggest_related
        self.related_min_words = related_min_words
        self.compress_every_s = compress_every_s
        self.compress_max_chars = compress_max_chars
        self.compress_keep = compress_keep
        self._last_seen: dict[str, tuple[float, str, str]] | None = None   # locuteur -> (instant, titre, id)
        self.current: Conversation | None = None
        self._pending: list[tuple[str, Any]] = []   # changements à appliquer en fin de tour
        self._suggested: tuple[str, str] | None = None   # (conversation en cours, conversation proposée)
        self._turn_start = 0   # début du tour en cours dans la transcription
        self._turn_state: dict[str, Any] = {}   # état du cerveau au début du tour
        self._index: dict[str, _Indexed] = {}   # cache des conversations sur disque, par identifiant
        self._index_lock = threading.Lock()   # _refresh() tourne aussi dans un thread
        self.listing: list[str] = []   # identifiants du dernier affichage de /conversations
        # Lecture initiale en arrière-plan (pendant le chargement des modèles), pas à la première phrase.
        threading.Thread(target=self._warm_up, name="seldon-memory", daemon=True).start()

    def actions(self) -> list[Action]:
        """Outils que la mémoire propose au cerveau (retrouver, reprendre, modifier une conversation)."""
        return [Conversations()]

    def _warm_up(self) -> None:
        try:
            self._refresh()
        except Exception as e:   # le cache sera rempli au premier besoin
            log.warning("Lecture des conversations échouée : %s", e)

    # --- stockage -----------------------------------------------------------

    def _path(self, conv_id: str) -> Path:
        """Fichier de ``conv_id`` ; ValueError si l'identifiant (souvent fourni par le cerveau) est invalide."""
        if not valid_id(conv_id):
            raise ValueError(f"Identifiant de conversation invalide : {str(conv_id)[:80]!r}")
        path = self.dir / f"{conv_id}.json"
        if path.resolve().parent != self.dir.resolve():   # lien symbolique vers l'extérieur
            raise ValueError(f"Conversation hors du dossier : {conv_id}")
        return path

    def save(self, state: dict[str, Any] | None = None) -> None:
        conv = self.current
        if conv is None or not conv.transcript:
            return
        conv.state = self.brain.get_state() if state is None else state
        self._write(conv)

    def _write(self, conv: Conversation) -> None:
        path = self._path(conv.id)
        data = json.dumps({f.name: getattr(conv, f.name) for f in fields(conv)}, ensure_ascii=False)
        write_private(path, data.encode("utf-8"))
        self._uncache(conv.id)

    def _uncache(self, conv_id: str) -> None:
        with self._index_lock:
            self._index.pop(conv_id, None)

    def load(self, conv_id: str) -> Conversation | None:
        """Conversation ``conv_id`` avec l'état du cerveau ; None si absente, illisible ou id invalide."""
        try:
            path = self._path(conv_id)
        except ValueError as e:
            if conv_id:
                log.warning("%s", e)
            return None
        return _read(path, conv_id, with_state=True)

    def _refresh(self) -> dict[str, _Indexed]:
        """Met le cache à jour : seuls les fichiers nouveaux ou modifiés sont relus (bloquant)."""
        with self._index_lock:
            try:
                entries = list(os.scandir(self.dir))
            except FileNotFoundError:
                entries = []
            except OSError as e:
                log.warning("Dossier des conversations illisible : %s (%s)", self.dir, e)
                entries = []
            current = self.current.id if self.current and self.current.transcript else None
            old, new = self._index, {}
            for entry in entries:
                conv_id = entry.name[:-5]
                if not entry.name.endswith(".json") or not valid_id(conv_id) or entry.is_symlink():
                    continue   # comme load() : jamais un fichier hors du dossier
                cached = old.get(conv_id)
                if conv_id == current and cached is None:   # remplacée par self.current : inutile de la lire
                    continue
                try:
                    st = entry.stat()
                except OSError:
                    continue
                stamp = (st.st_ino, st.st_mtime_ns, st.st_size)
                if cached is not None and cached.stamp == stamp:
                    new[conv_id] = cached
                elif (conv := _read(Path(entry.path), conv_id)) is not None:
                    new[conv_id] = _Indexed.of(conv, stamp)
            self._index = new
            return new

    def _indexed(self) -> list[_Indexed]:
        """Conversations (celle en cours comprise) et leurs mots, la plus récente d'abord."""
        index = self._refresh()
        current = self.current if self.current and self.current.transcript else None
        out = [e for conv_id, e in index.items() if current is None or conv_id != current.id]
        if current is not None:
            out.append(_Indexed.of(current))
        return sorted(out, key=lambda e: e.conv.updated, reverse=True)

    def all(self) -> list[Conversation]:
        """Toutes les conversations, la plus récente d'abord (sans l'état du cerveau).

        Elles sont partagées avec le cache : les lire seulement (``load()`` pour en modifier une).
        """
        return [e.conv for e in self._indexed()]

    def search(self, query: str = "", limit: int = 10) -> list[Conversation]:
        """Conversations correspondant le mieux aux mots de ``query`` (récentes si vide)."""
        terms = _words(str(query or ""))
        if not terms:
            return self.all()[:limit]
        return [c for _, c in self._scored(terms)[:limit]]

    def _scored(self, terms: set[str]) -> list[tuple[int, Conversation]]:
        """(nombre de mots communs, conversation), la plus pertinente d'abord (titre compté triple)."""
        if not terms:
            return []
        scored = []
        for e in self._indexed():
            score = 3 * len(terms & e.title_words) + len(terms & e.body_words)
            if score:
                scored.append((score, e.conv.updated, len(terms & (e.title_words | e.body_words)), e.conv))
        scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
        return [(common, c) for _, _, common, c in scored]

    def related_note(self, text: str) -> str | None:
        """Au premier message d'une conversation : note pour le cerveau si ce sujet a déjà été abordé."""
        conv = self.current
        if not self.suggest_related or conv is None or len(conv.transcript) != 1:
            return None
        terms = {k for k in _words(text) if k not in _FILLER}
        found = next((c for common, c in self._scored(terms)
                      if c.id != conv.id and common >= self.related_min_words), None)
        if found is None:
            return None
        self._suggested = (conv.id, found.id)
        return (
            f"ce sujet a peut-être déjà été abordé dans la conversation « {found.title or 'sans titre'} » "
            f"du {datetime.fromtimestamp(found.updated):%d/%m/%Y} (id={found.id}). Si c'est bien le même sujet, "
            "demande d'abord à l'utilisateur s'il veut la poursuivre ; s'il accepte, reprends-la avec l'outil "
            "conversations (operation='reprendre')."
        )

    def last_seen(self, speaker: str) -> tuple[float, str, str] | None:
        """(instant, titre, id de la conversation) de la dernière prise de parole de ``speaker``."""
        if self._last_seen is None:
            seen: dict[str, tuple[float, str, str]] = {}
            for conv in self.all():
                for m in conv.transcript:
                    who = m.get("speaker")
                    if m["role"] == "user" and who and (who not in seen or m["ts"] > seen[who][0]):
                        seen[who] = (m["ts"], conv.title, conv.id)
            self._last_seen = seen
        return self._last_seen.get(speaker)

    def absence_note(self, speaker: str | None, now: float | None = None) -> str | None:
        """Note pour le cerveau si ``speaker`` revient après une longue absence (ou n'a jamais parlé)."""
        if not speaker or self.greet_after_s is None:
            return None
        seen = self.last_seen(speaker)
        if seen is None:
            return f"c'est la première conversation avec {speaker} : accueille cette personne."
        ts, title, conv_id = seen
        away = (now or time.time()) - ts
        if away < self.greet_after_s:
            return None
        return (
            f"{speaker} te reparle après {_duration(away)} d'absence ; votre dernier échange portait sur "
            f"« {title or 'sans titre'} » (conversation id={conv_id}). Salue cette personne naturellement ; "
            "propose de reprendre ce sujet seulement si c'est à propos."
        )

    def is_current(self, conv_id: str) -> bool:
        return self.current is not None and self.current.id == conv_id

    def rename(self, conv_id: str, title: str) -> bool:
        if self.is_current(conv_id):
            self.current.title = title   # sauvegardé en fin de tour
            return True
        conv = self.load(conv_id)
        if conv is None:
            return False
        conv.title = title
        self._write(conv)
        return True

    def forget(self, conv_id: str, text: str) -> int:
        """Retire les messages contenant ``text`` (en fin de tour pour la conversation en cours)."""
        if self.is_current(conv_id):
            count = len(self.current.matching(text))
            if count:
                self._pending.append(("forget", (conv_id, text)))
            return count
        return self._forget_now(conv_id, text)

    def _forget_now(self, conv_id: str, text: str) -> int:
        current = self.is_current(conv_id)
        conv = self.current if current else self.load(conv_id)
        if conv is None:
            return 0
        removed = conv.matching(text)
        if not removed:
            return 0
        conv.transcript = [m for m in conv.transcript if all(m is not r for r in removed)]
        self._last_seen = None
        if not conv.transcript:
            self.delete(conv_id)
        elif current:   # l'historique du cerveau est reconstruit à partir de la transcription
            self.brain.set_state({"messages": transcript_messages(conv.transcript)})
            self.save()
        else:
            conv.state = {}   # reconstruit à partir de la transcription à la reprise
            self._write(conv)
        return len(removed)

    def request_delete(self, conv_id: str) -> None:
        """Supprime ``conv_id`` ; en fin de tour si c'est la conversation en cours."""
        if self.is_current(conv_id):
            self._pending.append(("delete", conv_id))
        else:
            self.delete(conv_id)

    def delete(self, conv_id: str) -> bool:
        self._last_seen = None
        if self.current and self.current.id == conv_id:
            self.current = None
            self.brain.reset()
        return self._remove(conv_id)

    def _remove(self, conv_id: str) -> bool:
        """Supprime le fichier de ``conv_id`` ; faux s'il n'existe pas (ou si l'id est invalide)."""
        try:
            path = self._path(conv_id)
        except ValueError as e:
            log.warning("%s", e)
            return False
        self._uncache(conv_id)
        if not path.is_file():
            return False
        path.unlink(missing_ok=True)
        return True

    # --- cycle de vie -----------------------------------------------------------

    def _new(self) -> Conversation:
        now = time.time()
        while True:   # identifiant libre : ne jamais écraser une autre conversation
            conv_id = f"{datetime.fromtimestamp(now):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"
            if not (self.dir / f"{conv_id}.json").exists():
                return Conversation(id=conv_id, created=now, updated=now, brain=self.brain_kind)

    def new(self) -> None:
        """Sauvegarde la conversation en cours et en commence une nouvelle."""
        self.save()
        self.brain.reset()
        self.current = None

    def resume(self, conv_id: str) -> Conversation | None:
        """Sauvegarde la conversation en cours et recharge ``conv_id`` dans le cerveau."""
        target = self.load(conv_id)
        if target is None:
            return None
        self.save()
        if target.brain == self.brain_kind and _valid_state(target.state):
            self.brain.set_state(target.state)
        else:
            self.brain.set_state({"messages": transcript_messages(target.transcript)})
            target.brain = self.brain_kind
        target.updated = time.time()
        self.current = target
        return target

    def request_resume(self, conv_id: str) -> None:
        if not self.is_current(conv_id):   # déjà en cours : rien à faire
            self._pending.append(("switch", conv_id))

    def request_new(self) -> None:
        self._pending.append(("new", None))

    def begin_turn(self, text: str, speaker: str | None) -> None:
        """À appeler avant de passer une prise de parole au cerveau."""
        now = time.time()
        if self.current and now - self.current.updated > self.new_after_idle_s:
            log.info("Inactivité : nouvelle conversation")
            self.new()
        if self.current is None:
            self.current = self._new()
            self._suggested = None
        self._turn_start = len(self.current.transcript)
        self._turn_state = self.brain.get_state()
        self.record("user", text, speaker)

    def record(self, role: str, text: str, speaker: str | None = None) -> None:
        conv = self.current
        if conv is None:
            return
        text = "" if text is None else str(text)
        conv.transcript.append({"role": role, "text": text, "speaker": speaker, "ts": time.time()})
        conv.updated = time.time()
        if role == "user":
            if speaker and self._last_seen is not None:
                self._last_seen[speaker] = (conv.updated, conv.title or text, conv.id)
            if not conv.title:
                conv.title = text if len(text) <= self.title_chars else text[: self.title_chars].rsplit(" ", 1)[0] + "…"
            if speaker and speaker not in conv.speakers:
                conv.speakers.append(speaker)

    def name_turn_speaker(self, speaker: str) -> None:
        """Attribue à ``speaker`` les paroles anonymes du tour en cours (voix tout juste enregistrée)."""
        conv = self.current
        if conv is None:
            return
        for m in conv.transcript[self._turn_start:]:
            if m["role"] == "user" and not m.get("speaker"):
                m["speaker"] = speaker
                if self._last_seen is not None:
                    self._last_seen[speaker] = (m["ts"], conv.title, conv.id)
        if speaker not in conv.speakers:
            conv.speakers.append(speaker)

    def end_turn(self) -> None:
        """À appeler après le tour du cerveau : sauvegarde et applique un changement demandé."""
        pending, self._pending = self._pending, []
        for op, arg in pending:
            if op == "new":
                self.new()
            elif op == "switch":
                self._move_turn_to(arg)
            elif op == "forget":
                self._forget_now(*arg)
            elif op == "delete":
                self.delete(arg)
        self.save()

    def _move_turn_to(self, conv_id: str) -> None:
        """Reprend ``conv_id`` en y déplaçant l'échange qui a demandé la reprise.

        Si cette reprise répond à la proposition de ``related_note()``, toute la
        nouvelle conversation y est déplacée (elle ne contient que ce sujet).
        """
        old = self.current
        if old is not None and old.id == conv_id:   # déjà la conversation en cours
            return
        start = 0 if old and self._suggested == (old.id, conv_id) else self._turn_start
        turn = old.transcript[start:] if old else []
        if old:
            del old.transcript[start:]
            self.save(self._turn_state)   # l'ancienne conversation sans ce tour
            self.current = None
        if self.resume(conv_id) is None:
            self.current = old
            log.warning("Conversation introuvable : %s", conv_id)
            if old:
                old.transcript.extend(turn)
            self.save()
            return
        for m in turn:
            self.record(m["role"], m["text"], m.get("speaker"))
        for user, reply in _exchanges(turn):
            self.brain.append_exchange(user, reply)
        self.save()
        if old and not old.transcript and not self.is_current(old.id):   # vide après déplacement : supprimée
            self._remove(old.id)

    # --- compression -------------------------------------------------------------

    async def compress_all(self, lock: AbstractAsyncContextManager | None = None) -> int:
        """Résume le début des conversations trop volumineuses ; renvoie leur nombre.

        ``lock`` (verrou du pipeline) n'est pris que pour appliquer chaque résumé, pas
        pendant sa rédaction : aucune conversation n'est modifiée pendant un tour du
        cerveau, et l'utilisateur n'attend pas la fin d'un résumé pour être entendu.
        """
        await asyncio.to_thread(self._refresh)   # lecture des fichiers hors de la boucle
        done = 0
        for conv in self.all():
            if conv.size() <= self.compress_max_chars or len(conv.transcript) <= self.compress_keep + 1:
                continue
            if await self.compress(conv.id, lock):
                done += 1
        return done

    async def compress(self, conv_id: str, lock: AbstractAsyncContextManager | None = None) -> bool:
        """Remplace les messages de ``conv_id`` sauf les ``compress_keep`` derniers par un résumé.

        Le résumé est rédigé sans ``lock`` ; il n'est appliqué (sous ``lock``) que si ces
        messages n'ont pas changé entre-temps (oubli, suppression, autre compression…).
        """
        conv = self.current if self.is_current(conv_id) else self.load(conv_id)
        if conv is None or len(conv.transcript) <= self.compress_keep + 1:
            return False
        old = [dict(m) for m in conv.transcript[: len(conv.transcript) - self.compress_keep]]
        text = "\n".join(f"{_ROLE_LABELS.get(m['role'], 'Seldon')}"
                         f"{' (' + m['speaker'] + ')' if m.get('speaker') else ''} : {m['text']}" for m in old)
        summary = None
        try:
            summary = await self.brain.summarize(SUMMARY_INSTRUCTIONS, text)
        except Exception as e:
            log.warning("Résumé par le cerveau impossible (%s) : résumé local", e)
        if not isinstance(summary, str) or not summary.strip():
            part = Conversation(id=conv.id, created=old[0]["ts"], updated=old[-1]["ts"], brain=conv.brain,
                                title=conv.title, speakers=conv.speakers, transcript=old)
            summary = part.summary(requests=10)
        async with lock or nullcontext():
            return self._apply_summary(conv_id, old, summary.strip())

    def _apply_summary(self, conv_id: str, old: list[dict[str, Any]], summary: str) -> bool:
        """Remplace ``old``, début de ``conv_id``, par ``summary`` s'il y est toujours tel quel."""
        current = self.is_current(conv_id)
        conv = self.current if current else self.load(conv_id)
        cut = len(old)
        if conv is None or len(conv.transcript) < cut or not all(map(_same, conv.transcript[:cut], old)):
            log.info("Conversation %s modifiée pendant son résumé : compression reportée", conv_id)
            return False
        conv.transcript = [{"role": "summary", "text": summary, "speaker": None, "ts": old[0]["ts"]},
                           *conv.transcript[cut:]]
        log.info("Conversation « %s » compressée : %d messages résumés", conv.title, cut)
        if current:
            self.brain.set_state({"messages": transcript_messages(conv.transcript)})
            self.save()
        else:
            conv.state = {}   # reconstruit à partir de la transcription à la reprise
            self._write(conv)
        return True


def _exchanges(transcript: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Échanges (paroles de l'utilisateur, réponses de Seldon) d'un extrait de transcription."""
    out: list[tuple[str, str]] = []
    user: list[str] = []
    reply: list[str] = []
    for m in [*transcript, {"role": "user", "text": None}]:
        if m["role"] == "user":
            if user and reply:
                out.append(("\n".join(user), "\n".join(reply)))
                user, reply = [], []
            if m["text"] is not None:
                user.append(f"[locuteur : {neutralize(m.get('speaker') or 'inconnu')}]\n{neutralize(m['text'])}")
        elif m["role"] == "assistant":
            reply.append(m["text"])
    return out


def load_conversations(folder: str | Path) -> list[Conversation]:
    """Conversations enregistrées dans ``folder``, la plus récente d'abord (sans l'état du cerveau).

    L'identifiant de chacune est le nom de son fichier ; les fichiers illisibles sont ignorés.
    """
    folder = Path(folder).expanduser()
    convs = [c for path in (folder.glob("*.json") if folder.is_dir() else []) if (c := _read(path, path.stem))]
    return sorted(convs, key=lambda c: c.updated, reverse=True)


def transcript_messages(transcript: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Transcription -> messages texte alternés user/assistant (format commun aux cerveaux)."""
    messages: list[dict[str, Any]] = []
    for m in transcript:
        role = "user" if m["role"] in ("user", "summary") else "assistant"
        if m["role"] == "summary":
            text = f"[note : résumé des échanges précédents]\n{neutralize(m['text'])}"
        elif role == "user":   # crochets neutralisés comme à la première écoute (brain.prompts.user_message)
            text = f"[locuteur : {neutralize(m.get('speaker') or 'inconnu')}]\n{neutralize(m['text'])}"
        else:
            text = m["text"]
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"] += "\n" + text
        elif messages or role == "user":
            messages.append({"role": role, "content": text})
    while messages and messages[-1]["role"] == "user":  # le prochain message sera un tour utilisateur
        messages.pop()
    return messages
