"""Outil « feishin » : pilote le lecteur de musique Feishin qui tourne sur cet ordinateur.

Deux voies, essayées dans cet ordre (``backend: auto``) :

* le « Remote control » de Feishin (Paramètres > Général > Remote control) :
  WebSocket sur le port 4333, module ``websockets``. Le client s'authentifie
  (``{"event": "authenticate", "header": "Basic <base64 user:mdp>"}``), reçoit
  l'état courant (``state``) puis chaque changement (``song``, ``playback``,
  ``volume``…). Tout est possible par cette voie, y compris la recherche dans la
  bibliothèque et la lecture d'un titre, d'un album ou d'une playlist ;
* MPRIS (D-Bus, via ``busctl --user``) en repli : lecture, volume, position,
  aléatoire, répétition et titre en cours, sans identifiants. Ni recherche, ni
  favori, ni note.

Identifiants du Remote control : ``FEISHIN_USERNAME`` / ``FEISHIN_PASSWORD``
dans le fichier ``.env`` (jamais dans ``config.yaml``).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
import re
import shutil
import uuid
from typing import Any

from seldon.base import Context, Result, Tool

log = logging.getLogger(__name__)

COMMANDS = [
    "status", "play", "pause", "toggle", "next", "previous",
    "volume", "volume_up", "volume_down", "seek", "forward", "rewind",
    "shuffle", "repeat", "favorite", "rating", "search", "play_media",
]
REMOTE_ONLY = {"favorite", "rating", "search", "play_media"}   # impossibles par MPRIS
KINDS = {"track": "tracks", "album": "albums", "playlist": "playlists"}
PLAY_MODES = ("now", "next", "last", "shuffle")
REPEAT_MODES = ("none", "all", "one")                 # ordre de la bascule dans Feishin
REPEAT_TO_MPRIS = {"none": "None", "all": "Playlist", "one": "Track"}
REPEAT_FR = {"none": "désactivée", "all": "toute la file", "one": "le titre en cours"}
STATUS_FR = {"playing": "en lecture", "paused": "en pause", "stopped": "à l'arrêt"}
AUTH_CLOSE = 4004                 # fermeture par le serveur : identifiants refusés
MAX_MESSAGE = 32 * 1024 * 1024    # « queue-state » peut porter des centaines de titres
SEARCH_LIMIT = 8                  # résultats rapportés par « search »
CONFIRM_DELAY = 1.5               # attente de la confirmation d'un changement (s)
_ID = re.compile(r"^[\w.:@-]{1,200}$")
_CTRL = re.compile(r"[\x00-\x1f\x7f]+")


class FeishinError(RuntimeError):
    """Échec du pilotage (message lisible, renvoyé tel quel au cerveau)."""


class Unavailable(FeishinError):
    """Voie indisponible (module absent, Feishin injoignable, identifiants refusés) : on essaie l'autre."""


def _clean(value: Any, limit: int = 120) -> str:
    """Texte venu de Feishin (titres…) : une ligne, sans caractères de contrôle, borné."""
    text = _CTRL.sub(" ", str(value or "")).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _clock(seconds: float | None) -> str:
    s = max(0, int(seconds or 0))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def describe(st: dict[str, Any]) -> str:
    """Résumé lisible d'un état (clés : status, title, artists, album, position, duration…)."""
    status = STATUS_FR.get(st.get("status") or "", "état inconnu")
    if not st.get("title"):
        return f"Aucun titre chargé dans Feishin (lecteur {status})."
    text = f"Feishin {status} : « {_clean(st['title'])} »"
    if st.get("artists"):
        text += f" de {_clean(st['artists'])}"
    if st.get("album"):
        text += f", album « {_clean(st['album'])} »"
    text += "."
    if st.get("duration"):
        text += f" Position {_clock(st.get('position'))} sur {_clock(st['duration'])}."
    extras = []
    if st.get("volume") is not None:
        extras.append(f"volume {round(st['volume'])} %")
    if st.get("shuffle") is not None:
        extras.append("aléatoire " + ("activé" if st["shuffle"] else "désactivé"))
    if st.get("repeat") in REPEAT_FR:
        extras.append(f"répétition {REPEAT_FR[st['repeat']]}")
    if st.get("favorite"):
        extras.append("dans les favoris")
    if st.get("rating"):
        extras.append(f"noté {st['rating']}/5")
    return text + (" " + ", ".join(extras).capitalize() + "." if extras else "")


# --- Remote control (WebSocket) ----------------------------------------------------

class Remote:
    """Une connexion authentifiée au Remote control de Feishin (``async with``)."""

    def __init__(self, url: str, username: str, password: str, timeout: float) -> None:
        self.url, self.username, self.password, self.timeout = url, username, password, timeout
        self.ws: Any = None
        self.raw: dict[str, Any] = {}   # dernier état connu, tenu à jour par les diffusions
        self._closed: type[BaseException] = ConnectionError

    async def __aenter__(self) -> Remote:
        try:
            import websockets
        except ImportError as e:
            raise Unavailable("Le module Python « websockets » n'est pas installé.") from e
        self._closed = websockets.ConnectionClosed
        try:
            self.ws = await asyncio.wait_for(websockets.connect(
                self.url, open_timeout=self.timeout, max_size=MAX_MESSAGE, close_timeout=1), self.timeout)
        except (OSError, TimeoutError, websockets.WebSocketException) as e:
            raise Unavailable(f"Feishin injoignable sur {self.url} ({type(e).__name__}). "
                              "Feishin est-il lancé avec le Remote control activé ?") from e
        try:
            if self.username or self.password:
                token = base64.b64encode(f"{self.username}:{self.password}".encode()).decode()
                await self.send("authenticate", header=f"Basic {token}")
            # sans identifiants, un serveur non protégé envoie l'état aussitôt ; protégé, rien ne vient
            wait = self.timeout if (self.username or self.password) else min(self.timeout, 2.0)
            if await self.wait_for("state", wait) is None:
                raise Unavailable("Le Remote control de Feishin demande un identifiant et un mot de passe "
                                  "(FEISHIN_USERNAME / FEISHIN_PASSWORD dans le fichier .env).")
        except BaseException:
            await self.__aexit__()
            raise
        return self

    async def __aexit__(self, *exc: Any) -> None:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(self.ws.close(), 2)

    async def send(self, event: str, **data: Any) -> None:
        await self.ws.send(json.dumps({"event": event, **data}))

    async def wait_for(self, event: str, timeout: float, match: Any = None) -> Any:
        """Lit les messages jusqu'à ``event`` (et ``match(data)`` vrai) ; None si le délai expire."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while (left := deadline - loop.time()) > 0:
            try:
                message = await asyncio.wait_for(self.ws.recv(), left)
            except TimeoutError:
                return None
            except self._closed as e:
                code = getattr(getattr(e, "rcvd", None), "code", None) or getattr(e, "code", None)
                if code == AUTH_CLOSE:
                    raise Unavailable("Feishin a refusé les identifiants du Remote control.") from e
                raise Unavailable(f"Connexion à Feishin fermée (code {code}).") from e
            try:
                msg = json.loads(message)
            except (TypeError, ValueError):
                continue
            if not isinstance(msg, dict):
                continue
            name, data = msg.get("event"), msg.get("data")
            self._apply(name, data)
            if name == "error" and event != "state":
                raise FeishinError(f"Feishin signale une erreur : {_clean(data, 200)}")
            if name == event and (match is None or match(data)):
                return data
        return None

    def _apply(self, event: str, data: Any) -> None:
        """Tient l'état à jour à partir d'une diffusion du serveur."""
        if event == "state" and isinstance(data, dict):
            self.raw = dict(data)
        elif event == "song":
            self.raw["song"] = data
        elif event == "playback":
            self.raw["status"] = data
        elif event in ("volume", "position", "shuffle", "repeat"):
            self.raw[event] = data
        elif event in ("favorite", "rating") and isinstance(data, dict):
            song = self.raw.get("song") or {}
            if song.get("id") == data.get("id"):
                song["userFavorite" if event == "favorite" else "userRating"] = data.get(event)

    async def request(self, event: str, response: str, **data: Any) -> Any:
        """Requête avec ``requestId`` ; renvoie les données de la réponse correspondante."""
        rid = uuid.uuid4().hex
        await self.send(event, requestId=rid, **data)
        found = await self.wait_for(response, self.timeout, lambda d: isinstance(d, dict) and d.get("requestId") == rid)
        if found is None:
            raise FeishinError("Feishin n'a pas répondu à temps (serveur musical lent ou déconnecté ?).")
        return found

    # --- interface commune avec Mpris ---

    async def state(self) -> dict[str, Any]:
        raw, song = self.raw, self.raw.get("song") or {}
        return {
            "status": raw.get("status"), "title": song.get("name"), "album": song.get("album"),
            "artists": song.get("artistName") or ", ".join(
                a.get("name", "") for a in song.get("artists") or [] if isinstance(a, dict)),
            "duration": (song.get("duration") or 0) / 1000 or None,   # ms -> s
            "position": raw.get("position"), "volume": raw.get("volume"),
            "shuffle": raw.get("shuffle"), "repeat": raw.get("repeat"),
            "favorite": song.get("userFavorite"), "rating": song.get("userRating"), "id": song.get("id"),
        }

    async def transport(self, command: str) -> None:
        st = await self.state()
        if command == "toggle":
            command = "pause" if st["status"] == "playing" else "play"
        await self.send(command)
        if command in ("next", "previous"):
            await self.wait_for("song", CONFIRM_DELAY, lambda d: (d or {}).get("id") != st["id"])
        elif st["status"] != ("playing" if command == "play" else "paused"):
            await self.wait_for("playback", CONFIRM_DELAY)

    async def set_volume(self, volume: int) -> None:
        await self.send("volume", volume=volume)

    async def seek(self, position: float, st: dict[str, Any]) -> None:
        await self.send("position", position=position)

    async def set_shuffle(self, on: bool) -> bool:
        if bool(self.raw.get("shuffle")) != on:   # le serveur ne connaît que la bascule
            await self.send("shuffle")
            await self.wait_for("shuffle", CONFIRM_DELAY)
        return bool(self.raw.get("shuffle"))

    async def set_repeat(self, mode: str) -> str | None:
        for _ in REPEAT_MODES:
            if self.raw.get("repeat") == mode:
                break
            await self.send("repeat")
            if await self.wait_for("repeat", CONFIRM_DELAY) is None:
                break
        return self.raw.get("repeat")


# --- MPRIS (D-Bus) -----------------------------------------------------------------

class Mpris:
    """Accès MPRIS minimal via ``busctl --user`` (aucune dépendance Python)."""

    BUS = "org.mpris.MediaPlayer2.Feishin"
    PATH = "/org/mpris/MediaPlayer2"
    IFACE = "org.mpris.MediaPlayer2.Player"

    def __init__(self, timeout: float) -> None:
        self.timeout = timeout

    async def _busctl(self, *args: str) -> str:
        exe = shutil.which("busctl")
        if not exe:
            raise Unavailable("MPRIS indisponible : commande « busctl » introuvable.")
        proc = await asyncio.create_subprocess_exec(
            exe, "--user", *args, stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            out, err = await asyncio.wait_for(proc.communicate(), self.timeout)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()
            raise Unavailable("MPRIS ne répond pas.") from None
        if proc.returncode:
            raise Unavailable(f"Feishin introuvable sur D-Bus (MPRIS) : {_clean(err.decode(errors='replace'), 200)}")
        return out.decode(errors="replace")

    async def get(self, *props: str) -> dict[str, Any]:
        out = await self._busctl("--json=short", "get-property", self.BUS, self.PATH, self.IFACE, *props)
        return dict(zip(props, [json.loads(line).get("data") for line in out.splitlines() if line.strip()]))

    async def set(self, prop: str, signature: str, value: str) -> None:
        await self._busctl("set-property", self.BUS, self.PATH, self.IFACE, prop, signature, value)

    async def call(self, method: str, *args: str) -> None:
        await self._busctl("call", self.BUS, self.PATH, self.IFACE, method, *args)

    # --- interface commune avec Remote ---

    async def state(self) -> dict[str, Any]:
        p = await self.get("PlaybackStatus", "Metadata", "Volume", "Position", "Shuffle", "LoopStatus")
        meta = {k: (v or {}).get("data") for k, v in (p.get("Metadata") or {}).items()}
        artists = meta.get("xesam:artist")
        loop = p.get("LoopStatus") or ""
        return {
            "status": str(p.get("PlaybackStatus") or "").lower() or None,
            "title": meta.get("xesam:title"), "album": meta.get("xesam:album"),
            "artists": ", ".join(map(str, artists)) if isinstance(artists, list) else artists,
            "duration": (meta.get("mpris:length") or 0) / 1e6 or None,
            "position": (p.get("Position") or 0) / 1e6,
            "volume": round((p.get("Volume") or 0) * 100),
            "shuffle": p.get("Shuffle"),
            "repeat": next((k for k, v in REPEAT_TO_MPRIS.items() if v == loop), None),
            "id": meta.get("mpris:trackid"),
        }

    async def transport(self, command: str) -> None:
        await self.call({"play": "Play", "pause": "Pause", "toggle": "PlayPause",
                         "next": "Next", "previous": "Previous"}[command])
        await asyncio.sleep(min(CONFIRM_DELAY, 0.5))   # le temps que les métadonnées suivent

    async def set_volume(self, volume: int) -> None:
        await self.set("Volume", "d", str(volume / 100))

    async def seek(self, position: float, st: dict[str, Any]) -> None:
        await self.call("SetPosition", "ox", str(st["id"]), str(int(position * 1e6)))

    async def set_shuffle(self, on: bool) -> bool:
        await self.set("Shuffle", "b", "true" if on else "false")
        return on

    async def set_repeat(self, mode: str) -> str:
        await self.set("LoopStatus", "s", REPEAT_TO_MPRIS[mode])
        return mode


# --- L'outil -----------------------------------------------------------------------

class Feishin(Tool):
    name = "control_music"
    description = (
        "Pilote le lecteur de musique Feishin de l'ordinateur. Commandes : status (titre en cours), "
        "play, pause, toggle, next, previous ; volume (value = 0 à 100), volume_up / volume_down "
        "(value = pas, défaut 10) ; seek (value = position en secondes), forward / rewind (value = "
        "secondes, défaut 10) ; shuffle (on = true/false, sinon bascule) ; repeat (repeat_mode = "
        "none/all/one, sinon bascule) ; favorite (on, défaut true) et rating (value = 0 à 5) pour le "
        "titre en cours ; search (query, kind = track/album/playlist) liste la bibliothèque ; "
        "play_media (query ou id, kind, play_mode = now/next/last/shuffle) lance un titre, un album "
        "ou une playlist. Pour un artiste, cherche ses albums ou ses titres (kind=album ou track)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "enum": COMMANDS, "description": "Action à effectuer."},
            "value": {"type": "number", "description": "Volume, pas, position ou note selon la commande."},
            "on": {"type": "boolean", "description": "shuffle / favorite : activer (true) ou désactiver (false)."},
            "repeat_mode": {"type": "string", "enum": list(REPEAT_MODES),
                            "description": "repeat : none (aucune), all (toute la file), one (le titre)."},
            "query": {"type": "string", "description": "search / play_media : texte cherché (titre, album, artiste…)."},
            "kind": {"type": "string", "enum": list(KINDS), "description": "search / play_media : type (défaut track)."},
            "id": {"type": "string", "description": "play_media : identifiant exact issu d'une recherche."},
            "play_mode": {"type": "string", "enum": list(PLAY_MODES),
                          "description": "play_media : now (remplace la file, défaut), next, last, shuffle."},
        },
        "required": ["command"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        url: str = "ws://127.0.0.1:4333",   # Remote control de Feishin
        backend: str = "auto",              # auto (Remote puis MPRIS) | remote | mpris
        timeout: float = 10.0,              # délai max d'une opération (s)
    ) -> None:
        if backend not in ("auto", "remote", "mpris"):
            raise ValueError(f"Feishin : backend inconnu {backend!r} (auto, remote, mpris)")
        if not str(url).startswith(("ws://", "wss://")):
            raise ValueError(f"Feishin : url doit commencer par ws:// ou wss:// ({url!r})")
        self.url, self.backend, self.timeout = url, backend, float(timeout)

    async def run(self, args: dict[str, Any], ctx: Context) -> Result:
        command = args.get("command")
        if command not in COMMANDS:
            return Result.fail(f"Commande inconnue : {str(command)[:40]}. Possibles : {', '.join(COMMANDS)}")
        if problem := _invalid(command, args):
            return Result.fail(problem)
        try:
            return Result(await asyncio.wait_for(self._dispatch(command, args), self.timeout * 3))
        except FeishinError as e:
            return Result.fail(str(e))
        except TimeoutError:
            return Result.fail("Feishin ne répond pas.")

    async def _dispatch(self, command: str, args: dict[str, Any]) -> str:
        if self.backend == "mpris":
            return await self._mpris(command, args)
        try:
            async with Remote(self.url, os.environ.get("FEISHIN_USERNAME", ""),
                              os.environ.get("FEISHIN_PASSWORD", ""), self.timeout) as remote:
                if command in REMOTE_ONLY:
                    return await self._remote_only(remote, command, args)
                return await _control(remote, command, args)
        except Unavailable as e:
            if self.backend == "remote" or command in REMOTE_ONLY:
                raise
            log.info("Feishin Remote indisponible (%s) : repli MPRIS", e)
            try:
                return await self._mpris(command, args)
            except Unavailable as e2:
                raise FeishinError(f"{e} Repli MPRIS impossible : {e2}") from e2

    async def _mpris(self, command: str, args: dict[str, Any]) -> str:
        if command in REMOTE_ONLY:
            raise FeishinError(f"« {command} » nécessite le Remote control de Feishin (non joignable).")
        return await _control(Mpris(self.timeout), command, args)

    async def _remote_only(self, remote: Remote, command: str, args: dict[str, Any]) -> str:
        st = await remote.state()
        kind = args.get("kind") or "track"
        if command == "search":
            items = await _search(remote, kind, args["query"])
            if not items:
                return f"Aucun résultat pour « {_clean(args['query'], 80)} »."
            lines = [f"- {_label(kind, it)} (id {_clean(it.get('id'), 80)})" for it in items[:SEARCH_LIMIT]]
            return f"Résultats ({kind}) pour « {_clean(args['query'], 80)} » :\n" + "\n".join(lines)
        if command == "play_media":
            return await _play_media(remote, kind, args)
        if not st["id"]:
            return "Aucun titre en cours."
        title = _clean(st["title"])
        if command == "favorite":
            on = args.get("on", True)
            await remote.send("favorite", favorite=on, id=st["id"])
            await remote.wait_for("favorite", CONFIRM_DELAY)
            return f"« {title} » {'ajouté aux' if on else 'retiré des'} favoris."
        rating = round(args["value"])
        await remote.send("rating", rating=rating, id=st["id"])
        await remote.wait_for("rating", CONFIRM_DELAY)
        return f"« {title} » noté {rating}/5." if rating else f"Note de « {title} » effacée."


async def _control(player: Remote | Mpris, command: str, args: dict[str, Any]) -> str:
    """Commandes communes aux deux voies."""
    st = await player.state()
    value = args.get("value")
    step = value if value is not None else 10
    if command == "status":
        return describe(st)
    if command in ("play", "pause", "toggle", "next", "previous"):
        await player.transport(command)
        if command == "pause" or (command == "toggle" and st["status"] == "playing"):
            return "Musique en pause."
        return describe(await player.state())
    if command in ("volume", "volume_up", "volume_down"):
        current = st["volume"] or 0
        target = value if command == "volume" else current + (step if command == "volume_up" else -step)
        target = round(min(100, max(0, target)))
        await player.set_volume(target)
        return f"Volume réglé à {target} %."
    if command in ("seek", "forward", "rewind"):
        if not st["title"]:
            return "Aucun titre en cours."
        target = value if command == "seek" else (st["position"] or 0) + (step if command == "forward" else -step)
        target = max(0.0, float(target))
        if st["duration"]:
            target = min(target, max(0.0, st["duration"] - 1))
        await player.seek(target, st)
        return f"Lecture placée à {_clock(target)}."
    if command == "shuffle":
        on = await player.set_shuffle(not st["shuffle"] if args.get("on") is None else args["on"])
        return "Lecture aléatoire " + ("activée." if on else "désactivée.")
    # repeat : le mode demandé, sinon le suivant dans l'ordre de Feishin
    want = args.get("repeat_mode") or REPEAT_MODES[(REPEAT_MODES.index(st["repeat"] or "none") + 1) % 3]
    mode = await player.set_repeat(want)
    if mode != want:
        return f"Répétition non confirmée (actuellement : {REPEAT_FR.get(mode, 'inconnue')})."
    return f"Répétition : {REPEAT_FR[mode]}."


async def _search(remote: Remote, kind: str, query: str) -> list[dict[str, Any]]:
    plural = KINDS[kind]
    data = await remote.request(f"{plural}-request", f"{plural}-response",
                                searchTerm=str(query).strip()[:200], startIndex=0, limit=10)
    items = data.get("items") if isinstance(data, dict) else None
    return [it for it in items or [] if isinstance(it, dict) and it.get("id")]


def _label(kind: str, item: dict[str, Any]) -> str:
    name = f"« {_clean(item.get('name'))} »"
    if kind == "track":
        extra = ", ".join(_clean(x, 80) for x in (item.get("artistName"), item.get("album")) if x)
    elif kind == "album":
        extra = ", ".join(str(x) for x in (_clean(item.get("albumArtistName"), 80), item.get("releaseYear")) if x)
    else:
        extra = f"{item['songCount']} titres" if item.get("songCount") else ""
    return f"{name} ({extra})" if extra else name


async def _play_media(remote: Remote, kind: str, args: dict[str, Any]) -> str:
    item_id, label = args.get("id"), f"l'élément {_clean(args.get('id'), 80)}"
    if not item_id:
        query = str(args["query"]).strip()
        items = await _search(remote, kind, query)
        if not items:
            return f"Rien trouvé pour « {_clean(query, 80)} » ({kind})."
        best = next((it for it in items if str(it.get("name", "")).casefold() == query.casefold()), items[0])
        item_id, label = best["id"], _label(kind, best)
    mode = args.get("play_mode") or "now"
    ack = await remote.request(f"play-{kind}", "operation-ack", id=item_id, playType=mode)
    if isinstance(ack, dict) and ack.get("error"):
        raise FeishinError(f"Feishin n'a pas pu lancer {label} : {_clean(ack['error'], 200)}")
    if mode in ("now", "shuffle"):
        return f"Lecture de {label}."
    return f"{label} ajouté {'après le titre en cours' if mode == 'next' else 'en fin de file'}."


def _invalid(command: str, a: dict[str, Any]) -> str | None:
    """Erreur dans les arguments, avant tout contact avec Feishin."""
    value = a.get("value")
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
        return "« value » doit être un nombre."
    if command == "volume" and (value is None or not 0 <= value <= 100):
        return "Indique un volume entre 0 et 100 (value)."
    if command == "seek" and (value is None or value < 0):
        return "Indique la position voulue en secondes (value)."
    if command == "rating" and (value is None or not 0 <= value <= 5):
        return "Indique une note entre 0 et 5 (value)."
    if command in ("volume_up", "volume_down", "forward", "rewind") and value is not None and value <= 0:
        return "« value » doit être positif."
    if command == "search" and not str(a.get("query") or "").strip():
        return "Indique quoi chercher (query)."
    if command == "play_media":
        if a.get("id") is not None and not _ID.match(str(a["id"])):
            return "Identifiant invalide."
        if not a.get("id") and not str(a.get("query") or "").strip():
            return "Indique quoi jouer (query) ou l'identifiant d'un résultat (id)."
    for key, allowed in (("kind", KINDS), ("play_mode", PLAY_MODES), ("repeat_mode", REPEAT_MODES)):
        if a.get(key) is not None and a[key] not in allowed:
            return f"{key} : {', '.join(allowed)}."
    if a.get("on") is not None and not isinstance(a["on"], bool):
        return "« on » doit valoir true ou false."
    return None
