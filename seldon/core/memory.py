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
cours de route.

La mémoire sait aussi quand chaque personne a parlé pour la dernière fois :
``absence_note()`` signale au cerveau quelqu'un qui revient après une longue
absence (ou qui lui parle pour la première fois), pour qu'il l'accueille.
"""

from __future__ import annotations

import json
import logging
import time
import unicodedata
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from seldon.brain.base import Brain

log = logging.getLogger(__name__)

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


def _fold(word: str) -> str:
    return unicodedata.normalize("NFKD", word.lower()).encode("ascii", "ignore").decode()


def _tokens(text: str) -> dict[str, str]:
    """Mots significatifs : forme sans accents ni majuscules -> forme d'origine (en minuscules)."""
    out: dict[str, str] = {}
    for word in "".join(c if c.isalnum() else " " for c in text).split():
        key = _fold(word)
        if len(key) > 2 and key not in _STOPWORDS:
            out.setdefault(key, word.lower())
    return out


def _words(text: str) -> set[str]:
    return set(_tokens(text))


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
        return "\n".join(
            f"{'Utilisateur' if m['role'] == 'user' else 'Seldon'} : {m['text']}" for m in self.transcript[-n:]
        )


class ConversationMemory:
    def __init__(
        self,
        brain: Brain,
        brain_kind: str,
        dir: str = "data/conversations",
        new_after_idle_s: float = 600,
        title_chars: int = 70,
        greet_after_s: float | None = 6 * 3600,   # absence au-delà de laquelle Seldon accueille ; None = jamais
    ) -> None:
        self.brain = brain
        self.brain_kind = brain_kind
        self.dir = Path(dir).expanduser()
        self.new_after_idle_s = new_after_idle_s
        self.title_chars = title_chars
        self.greet_after_s = greet_after_s
        self._last_seen: dict[str, tuple[float, str, str]] | None = None   # locuteur -> (instant, titre, id)
        self.current: Conversation | None = None
        self._pending: tuple[str, str | None] | None = None   # ("switch", id) | ("new", None)
        self._turn_start = 0   # début du tour en cours dans la transcription
        self._turn_state: dict[str, Any] = {}   # état du cerveau au début du tour

    # --- stockage -----------------------------------------------------------

    def _path(self, conv_id: str) -> Path:
        return self.dir / f"{conv_id}.json"

    def save(self, state: dict[str, Any] | None = None) -> None:
        conv = self.current
        if conv is None or not conv.transcript:
            return
        conv.state = self.brain.get_state() if state is None else state
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self._path(conv.id).with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(conv), ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._path(conv.id))

    def load(self, conv_id: str) -> Conversation | None:
        path = self._path(conv_id)
        if not path.is_file():
            return None
        return Conversation(**json.loads(path.read_text(encoding="utf-8")))

    def all(self) -> list[Conversation]:
        """Toutes les conversations, la plus récente d'abord (sans l'état du cerveau)."""
        convs = load_conversations(self.dir)
        if self.current and self.current.transcript:
            convs = [c for c in convs if c.id != self.current.id] + [self.current]
        return sorted(convs, key=lambda c: c.updated, reverse=True)

    def search(self, query: str = "", limit: int = 10) -> list[Conversation]:
        """Conversations correspondant le mieux aux mots de ``query`` (récentes si vide)."""
        convs = self.all()
        terms = _words(query)
        if not terms:
            return convs[:limit]
        scored = []
        for conv in convs:
            title, body = _words(conv.title), _words(" ".join(m["text"] for m in conv.transcript))
            score = 3 * len(terms & title) + len(terms & body)
            if score:
                scored.append((score, conv.updated, conv))
        scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
        return [c for _, _, c in scored[:limit]]

    def last_seen(self, speaker: str) -> tuple[float, str, str] | None:
        """(instant, titre, id de la conversation) de la dernière prise de parole de ``speaker``."""
        if self._last_seen is None:
            self._last_seen = {}
            for conv in reversed(self.all()):   # la plus récente en dernier : elle l'emporte
                for m in conv.transcript:
                    if m["role"] == "user" and m.get("speaker"):
                        self._last_seen[m["speaker"]] = (m["ts"], conv.title, conv.id)
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

    def delete(self, conv_id: str) -> bool:
        self._last_seen = None
        if self.current and self.current.id == conv_id:
            self.current = None
            self.brain.reset()
        path = self._path(conv_id)
        if path.is_file():
            path.unlink()
            return True
        return False

    # --- cycle de vie -----------------------------------------------------------

    def _new(self) -> Conversation:
        now = time.time()
        conv_id = f"{datetime.fromtimestamp(now):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"
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
        if target.brain == self.brain_kind and target.state:
            self.brain.set_state(target.state)
        else:
            self.brain.set_state({"messages": transcript_messages(target.transcript)})
            target.brain = self.brain_kind
        target.updated = time.time()
        self.current = target
        return target

    def request_resume(self, conv_id: str) -> None:
        self._pending = ("switch", conv_id)

    def request_new(self) -> None:
        self._pending = ("new", None)

    def begin_turn(self, text: str, speaker: str | None) -> None:
        """À appeler avant de passer une prise de parole au cerveau."""
        now = time.time()
        if self.current and now - self.current.updated > self.new_after_idle_s:
            log.info("Inactivité : nouvelle conversation")
            self.new()
        if self.current is None:
            self.current = self._new()
        self._turn_start = len(self.current.transcript)
        self._turn_state = self.brain.get_state()
        self.record("user", text, speaker)

    def record(self, role: str, text: str, speaker: str | None = None) -> None:
        conv = self.current
        if conv is None:
            return
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
        pending, self._pending = self._pending, None
        if pending is None:
            self.save()
        elif pending[0] == "new":
            self.new()
        else:
            self._move_turn_to(pending[1])

    def _move_turn_to(self, conv_id: str) -> None:
        """Reprend ``conv_id`` en y déplaçant l'échange qui a demandé la reprise."""
        old = self.current
        turn = old.transcript[self._turn_start:] if old else []
        if old:
            del old.transcript[self._turn_start:]
            self.save(self._turn_state)   # l'ancienne conversation sans ce tour
            self.current = None
        if self.resume(conv_id) is None:
            self.current = old
            log.warning("Conversation introuvable : %s", conv_id)
            if old:
                old.transcript.extend(turn)
            self.save()
            return
        conv = self.current
        for m in turn:
            self.record(m["role"], m["text"], m.get("speaker"))
        user = "\n".join(f"[locuteur : {m.get('speaker') or 'inconnu'}]\n{m['text']}" for m in turn if m["role"] == "user")
        reply = "\n".join(m["text"] for m in turn if m["role"] == "assistant")
        if user and reply:
            self.brain.append_exchange(user, reply)
        self.save()
        if old and not old.transcript:   # conversation vide après déplacement : supprimée
            self._path(old.id).unlink(missing_ok=True)


def load_conversations(folder: str | Path) -> list[Conversation]:
    """Conversations enregistrées dans ``folder``, la plus récente d'abord (sans l'état du cerveau)."""
    folder = Path(folder).expanduser()
    convs = []
    for path in folder.glob("*.json") if folder.is_dir() else []:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            data["state"] = {}
            convs.append(Conversation(**data))
        except (OSError, json.JSONDecodeError, TypeError):
            log.warning("Conversation illisible : %s", path)
    return sorted(convs, key=lambda c: c.updated, reverse=True)


def transcript_messages(transcript: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Transcription -> messages texte alternés user/assistant (format commun aux cerveaux)."""
    messages: list[dict[str, Any]] = []
    for m in transcript:
        role = "user" if m["role"] == "user" else "assistant"
        text = f"[locuteur : {m.get('speaker') or 'inconnu'}]\n{m['text']}" if role == "user" else m["text"]
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"] += "\n" + text
        elif messages or role == "user":
            messages.append({"role": role, "content": text})
    while messages and messages[-1]["role"] == "user":  # le prochain message sera un tour utilisateur
        messages.pop()
    return messages
