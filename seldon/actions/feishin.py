"""Action « feishin » : pilote le lecteur de musique Feishin qui tourne sur cet ordinateur.

Deux voies, essayées dans cet ordre (``backend: auto``) :

* le serveur « Remote control » intégré à Feishin (Paramètres > Général >
  Remote control) : WebSocket sur le port 4333. Après la connexion, le client
  s'authentifie par un message ``{"event": "authenticate", "header": "Basic
  <base64 user:mot de passe>"}`` ; le serveur répond par l'état courant
  (``state``, ``queue-state``…) puis diffuse chaque changement (``song``,
  ``playback``, ``volume``, ``position``, ``shuffle``, ``repeat``…). Il accepte
  ``play``, ``pause``, ``next``, ``previous``, ``volume``, ``position`` (s),
  ``shuffle``/``repeat`` (bascules), ``favorite``, ``rating``, ainsi que la
  recherche dans la bibliothèque du serveur musical (``tracks-request``,
  ``albums-request``, ``playlists-request``) et la lecture d'un résultat
  (``play-track``, ``play-album``, ``play-playlist``, confirmée par
  ``operation-ack``). Nécessite le module ``websockets``.
* MPRIS (D-Bus, ``org.mpris.MediaPlayer2.Feishin``, via ``busctl --user``) en
  repli : lecture/pause, suivant/précédent, volume, position, aléatoire,
  répétition et titre en cours, sans identifiants. Ni recherche, ni favori,
  ni note par cette voie.

Identifiants : ceux saisis dans Feishin pour le Remote control. Ne les mettez pas
dans ``config.yaml`` : utilisez les variables d'environnement
``FEISHIN_USERNAME`` / ``FEISHIN_PASSWORD`` (lues si les options sont vides).

Exemple ``config.yaml`` :

    actions:
      - engine: feishin
        options:
          url: ws://127.0.0.1:4333
          backend: auto           # auto | remote | mpris
          timeout: 10
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

from seldon.actions.base import Action
from seldon.core.registry import register
from seldon.core.types import ActionContext, ActionResult

log = logging.getLogger(__name__)

COMMANDS = [
    "status", "play", "pause", "toggle", "next", "previous",
    "volume", "volume_up", "volume_down", "seek", "forward", "rewind",
    "shuffle", "repeat", "favorite", "rating", "search", "play_media",
]
# commandes possibles sans le serveur Remote (MPRIS)
MPRIS_COMMANDS = {
    "status", "play", "pause", "toggle", "next", "previous",
    "volume", "volume_up", "volume_down", "seek", "forward", "rewind", "shuffle", "repeat",
}
KINDS = {"track": "tracks", "album": "albums", "playlist": "playlists"}
PLAY_MODES = {"now": "now", "next": "next", "last": "last", "shuffle": "shuffle"}
REPEAT_MODES = ("none", "all", "one")
REPEAT_TO_MPRIS = {"none": "None", "all": "Playlist", "one": "Track"}
MPRIS_TO_REPEAT = {v: k for k, v in REPEAT_TO_MPRIS.items()}
REPEAT_FR = {"none": "désactivée", "all": "toute la file", "one": "le titre en cours"}
STATUS_FR = {"playing": "en lecture", "paused": "en pause", "stopped": "à l'arrêt"}
AUTH_CLOSE = 4004   # code de fermeture du serveur : identifiants refusés / délai dépassé
MAX_MESSAGE = 32 * 1024 * 1024   # « queue-state » peut porter des centaines de titres
_ID = re.compile(r"^[\w.:@-]{1,200}$")
_CTRL = re.compile(r"[\x00-\x1f\x7f]+")


class FeishinError(RuntimeError):
    """Échec du pilotage (message lisible, renvoyé tel quel au modèle)."""


class Unavailable(FeishinError):
    """Voie indisponible (dépendance absente, Feishin injoignable, identifiants refusés) : on peut essayer l'autre."""


def _clean(value: Any, limit: int = 120) -> str:
    """Texte venu de Feishin (titres…) : une ligne, sans caractères de contrôle, borné."""
    text = _CTRL.sub(" ", str(value or "")).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _clock(seconds: float | None) -> str:
    s = max(0, int(seconds or 0))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def describe(state: dict[str, Any]) -> str:
    """Résumé lisible d'un état normalisé (clés : status, title, artists, album, position, duration…)."""
    status = STATUS_FR.get(state.get("status") or "", "état inconnu")
    if not state.get("title"):
        return f"Aucun titre chargé dans Feishin (lecteur {status})."
    parts = [f"Feishin {status} : « {_clean(state['title'])} »"]
    if state.get("artists"):
        parts[0] += f" de {_clean(state['artists'])}"
    if state.get("album"):
        parts[0] += f", album « {_clean(state['album'])} »"
    parts[0] += "."
    if state.get("duration"):
        parts.append(f"Position {_clock(state.get('position'))} sur {_clock(state['duration'])}.")
    extras = []
    if state.get("volume") is not None:
        extras.append(f"volume {round(state['volume'])} %")
    if state.get("shuffle") is not None:
        extras.append("aléatoire " + ("activé" if state["shuffle"] else "désactivé"))
    if state.get("repeat") in REPEAT_FR:
        extras.append(f"répétition {REPEAT_FR[state['repeat']]}")
    if state.get("favorite"):
        extras.append("dans les favoris")
    if state.get("rating"):
        extras.append(f"noté {state['rating']}/5")
    if extras:
        parts.append(", ".join(extras).capitalize() + ".")
    return " ".join(parts)


def _remote_state(raw: dict[str, Any]) -> dict[str, Any]:
    """État du serveur Remote (``SongState``) -> état normalisé."""
    song = raw.get("song") or {}
    artists = song.get("artistName") or ", ".join(
        a.get("name", "") for a in song.get("artists") or [] if isinstance(a, dict))
    return {
        "status": raw.get("status"),
        "title": song.get("name"),
        "artists": artists,
        "album": song.get("album"),
        "duration": (song.get("duration") or 0) / 1000 or None,   # ms -> s
        "position": raw.get("position"),                           # s
        "volume": raw.get("volume"),                               # 0-100
        "shuffle": raw.get("shuffle"),
        "repeat": raw.get("repeat"),
        "favorite": song.get("userFavorite"),
        "rating": song.get("userRating"),
        "id": song.get("id"),
    }


# --- serveur Remote (WebSocket) ---

class RemoteSession:
    """Une connexion authentifiée au serveur Remote de Feishin (``async with``)."""

    def __init__(self, url: str, username: str, password: str, timeout: float) -> None:
        self.url, self.username, self.password, self.timeout = url, username, password, timeout
        self.ws: Any = None
        self.raw: dict[str, Any] = {}   # dernier SongState connu, tenu à jour par les diffusions
        self._closed_exc: type[BaseException] = ConnectionError

    @property
    def state(self) -> dict[str, Any]:
        return _remote_state(self.raw)

    async def __aenter__(self) -> "RemoteSession":
        try:
            import websockets
        except ImportError as e:
            raise Unavailable("Le module Python « websockets » n'est pas installé (pip install websockets).") from e
        connect = getattr(getattr(websockets, "asyncio", None), "client", None)
        connect = getattr(connect, "connect", None) or websockets.connect
        self._closed_exc = websockets.ConnectionClosed
        try:
            self.ws = await asyncio.wait_for(
                connect(self.url, open_timeout=self.timeout, max_size=MAX_MESSAGE, close_timeout=1),
                self.timeout)
        except (OSError, asyncio.TimeoutError, websockets.WebSocketException) as e:
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
                                  "(options username/password ou FEISHIN_USERNAME / FEISHIN_PASSWORD).")
        except BaseException:
            await self.close()
            raise
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def close(self) -> None:
        if self.ws is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self.ws.close(), 2)

    async def send(self, event: str, **data: Any) -> None:
        await self.ws.send(json.dumps({"event": event, **data}))

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

    async def wait_for(self, event: str, timeout: float, match: Any = None) -> Any:
        """Lit les messages jusqu'à ``event`` (et ``match(data)`` vrai) ; None si le délai expire."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while (left := deadline - loop.time()) > 0:
            try:
                message = await asyncio.wait_for(self.ws.recv(), left)
            except asyncio.TimeoutError:
                return None
            except self._closed_exc as e:
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
            if name == "error" and event not in ("state",):
                raise FeishinError(f"Feishin signale une erreur : {_clean(data, 200)}")
            if name == event and (match is None or match(data)):
                return data
        return None

    async def request(self, event: str, response: str, **data: Any) -> Any:
        """Requête avec ``requestId`` ; renvoie les données de la réponse correspondante."""
        rid = uuid.uuid4().hex
        await self.send(event, requestId=rid, **data)
        found = await self.wait_for(response, self.timeout,
                                    lambda d: isinstance(d, dict) and d.get("requestId") == rid)
        if found is None:
            raise FeishinError("Feishin n'a pas répondu à temps (serveur musical lent ou déconnecté ?).")
        return found


# --- MPRIS (D-Bus) ---

class Mpris:
    """Accès MPRIS minimal via ``busctl --user`` (aucune dépendance Python)."""

    PATH = "/org/mpris/MediaPlayer2"
    IFACE = "org.mpris.MediaPlayer2.Player"

    def __init__(self, bus_name: str, timeout: float) -> None:
        self.bus_name, self.timeout = bus_name, timeout

    async def _busctl(self, *args: str) -> str:
        exe = shutil.which("busctl")
        if not exe:
            raise Unavailable("MPRIS indisponible : commande « busctl » introuvable.")
        proc = await asyncio.create_subprocess_exec(
            exe, "--user", *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL)
        try:
            out, err = await asyncio.wait_for(proc.communicate(), self.timeout)
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()
            raise Unavailable("MPRIS ne répond pas.") from None
        if proc.returncode:
            raise Unavailable(f"Feishin introuvable sur D-Bus (MPRIS) : {_clean(err.decode(errors='replace'), 200)}")
        return out.decode(errors="replace")

    async def get(self, *props: str) -> dict[str, Any]:
        out = await self._busctl("--json=short", "get-property", self.bus_name, self.PATH, self.IFACE, *props)
        values = [json.loads(line).get("data") for line in out.splitlines() if line.strip()]
        return dict(zip(props, values))

    async def set(self, prop: str, signature: str, value: str) -> None:
        await self._busctl("set-property", self.bus_name, self.PATH, self.IFACE, prop, signature, value)

    async def call(self, method: str, *args: str) -> None:
        await self._busctl("call", self.bus_name, self.PATH, self.IFACE, method, *args)

    async def state(self) -> dict[str, Any]:
        p = await self.get("PlaybackStatus", "Metadata", "Volume", "Position", "Shuffle", "LoopStatus")
        meta = {k: (v or {}).get("data") for k, v in (p.get("Metadata") or {}).items()}
        artists = meta.get("xesam:artist")
        return {
            "status": str(p.get("PlaybackStatus") or "").lower() or None,
            "title": meta.get("xesam:title"),
            "artists": ", ".join(map(str, artists)) if isinstance(artists, list) else artists,
            "album": meta.get("xesam:album"),
            "duration": (meta.get("mpris:length") or 0) / 1e6 or None,
            "position": (p.get("Position") or 0) / 1e6,
            "volume": round((p.get("Volume") or 0) * 100),
            "shuffle": p.get("Shuffle"),
            "repeat": MPRIS_TO_REPEAT.get(p.get("LoopStatus") or ""),
            "trackid": meta.get("mpris:trackid"),
        }


@register("actions", "feishin")
class Feishin(Action):
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
    input_schema = {
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
        url: str = "ws://127.0.0.1:4333",
        username: str | None = None,   # vide : variable d'environnement FEISHIN_USERNAME
        password: str | None = None,   # vide : variable d'environnement FEISHIN_PASSWORD
        backend: str = "auto",         # auto (Remote puis MPRIS) | remote | mpris
        mpris_name: str = "org.mpris.MediaPlayer2.Feishin",
        timeout: float = 10.0,         # délai max d'une opération (s)
        search_limit: int = 8,         # résultats rapportés par « search »
        confirm_delay: float = 1.5,    # attente de la confirmation d'un changement (s)
    ) -> None:
        if backend not in ("auto", "remote", "mpris"):
            raise ValueError(f"Feishin : backend inconnu {backend!r} (auto, remote, mpris)")
        if not str(url).startswith(("ws://", "wss://")):
            raise ValueError(f"Feishin : url doit commencer par ws:// ou wss:// ({url!r})")
        self.url = url
        self.username = username or os.environ.get("FEISHIN_USERNAME", "")
        self.password = password or os.environ.get("FEISHIN_PASSWORD", "")
        self.backend = backend
        self.timeout = float(timeout)
        self.search_limit = max(1, int(search_limit))
        self.confirm_delay = float(confirm_delay)
        self.mpris = Mpris(mpris_name, self.timeout)

    def acknowledge(self, params: dict[str, Any]) -> str | None:
        return "Je lance ça." if params.get("command") == "play_media" else None

    async def run(self, params: dict[str, Any], ctx: ActionContext) -> ActionResult:
        command = params.get("command")
        if command not in COMMANDS:
            return ActionResult.error(f"Commande inconnue : {str(command)[:40]}. Possibles : {', '.join(COMMANDS)}")
        try:
            self._validate(command, params)
        except ValueError as e:
            return ActionResult.error(str(e))
        try:
            return ActionResult(text=await asyncio.wait_for(self._dispatch(command, params), self.timeout * 3))
        except FeishinError as e:
            return ActionResult.error(str(e))
        except asyncio.TimeoutError:
            return ActionResult.error("Feishin ne répond pas.")

    @staticmethod
    def _validate(command: str, p: dict[str, Any]) -> None:
        value = p.get("value")
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ValueError("« value » doit être un nombre.")
        if command == "volume" and (value is None or not 0 <= value <= 100):
            raise ValueError("Indique un volume entre 0 et 100 (value).")
        if command == "seek" and (value is None or value < 0):
            raise ValueError("Indique la position voulue en secondes (value).")
        if command == "rating" and (value is None or not 0 <= value <= 5):
            raise ValueError("Indique une note entre 0 et 5 (value).")
        if command in ("volume_up", "volume_down", "forward", "rewind") and value is not None and value <= 0:
            raise ValueError("« value » doit être positif.")
        if command == "search" and not str(p.get("query") or "").strip():
            raise ValueError("Indique quoi chercher (query).")
        if command == "play_media":
            if p.get("id") is not None and not _ID.match(str(p["id"])):
                raise ValueError("Identifiant invalide.")
            if not p.get("id") and not str(p.get("query") or "").strip():
                raise ValueError("Indique quoi jouer (query) ou l'identifiant d'un résultat (id).")
        if p.get("kind") is not None and p["kind"] not in KINDS:
            raise ValueError(f"Type inconnu : {str(p['kind'])[:20]} (track, album, playlist).")
        if p.get("play_mode") is not None and p["play_mode"] not in PLAY_MODES:
            raise ValueError("play_mode : now, next, last ou shuffle.")
        if p.get("repeat_mode") is not None and p["repeat_mode"] not in REPEAT_MODES:
            raise ValueError("repeat_mode : none, all ou one.")
        if p.get("on") is not None and not isinstance(p["on"], bool):
            raise ValueError("« on » doit valoir true ou false.")

    async def _dispatch(self, command: str, params: dict[str, Any]) -> str:
        if self.backend == "mpris":
            return await self._run_mpris(command, params)
        try:
            async with RemoteSession(self.url, self.username, self.password, self.timeout) as session:
                return await self._run_remote(session, command, params)
        except Unavailable as e:
            if self.backend == "remote" or command not in MPRIS_COMMANDS:
                raise
            log.info("Feishin Remote indisponible (%s) : repli MPRIS", e)
            try:
                return await self._run_mpris(command, params)
            except Unavailable as e2:
                raise FeishinError(f"{e} Repli MPRIS impossible : {e2}") from e2

    # --- via le serveur Remote ---

    async def _run_remote(self, s: RemoteSession, command: str, p: dict[str, Any]) -> str:
        st = s.state
        value = p.get("value")
        if command == "status":
            return describe(st)
        if command == "toggle":
            command = "pause" if st.get("status") == "playing" else "play"
        if command in ("play", "pause"):
            await s.send(command)
            if st.get("status") != ("playing" if command == "play" else "paused"):
                await s.wait_for("playback", self.confirm_delay)
            return describe(s.state) if command == "play" else "Musique en pause."
        if command in ("next", "previous"):
            old = st.get("id")
            await s.send(command)
            await s.wait_for("song", self.confirm_delay, lambda d: (d or {}).get("id") != old)
            return describe(s.state)
        if command in ("volume", "volume_up", "volume_down"):
            current = st.get("volume") or 0
            step = value if value is not None else 10
            target = value if command == "volume" else current + (step if command == "volume_up" else -step)
            target = round(min(100, max(0, target)))
            await s.send("volume", volume=target)
            return f"Volume réglé à {target} %."
        if command in ("seek", "forward", "rewind"):
            if not st.get("title"):
                return "Aucun titre en cours."
            pos = st.get("position") or 0
            step = value if value is not None else 10
            target = value if command == "seek" else pos + (step if command == "forward" else -step)
            target = max(0.0, float(target))
            if st.get("duration"):
                target = min(target, max(0.0, st["duration"] - 1))
            await s.send("position", position=target)
            return f"Lecture placée à {_clock(target)}."
        if command == "shuffle":
            want = p.get("on")
            if want is None or bool(st.get("shuffle")) != want:
                await s.send("shuffle")
                await s.wait_for("shuffle", self.confirm_delay)
            state = s.state.get("shuffle")
            return "Lecture aléatoire " + ("activée." if state else "désactivée.")
        if command == "repeat":
            want = p.get("repeat_mode")
            for _ in range(len(REPEAT_MODES) if want else 1):
                if want and s.state.get("repeat") == want:
                    break
                await s.send("repeat")
                if await s.wait_for("repeat", self.confirm_delay) is None:
                    break
            mode = s.state.get("repeat")
            if want and mode != want:
                return f"Répétition non confirmée (actuellement : {REPEAT_FR.get(mode, 'inconnue')})."
            return f"Répétition : {REPEAT_FR.get(mode, 'inconnue')}."
        if command in ("favorite", "rating"):
            if not st.get("id"):
                return "Aucun titre en cours."
            title = _clean(st.get("title"))
            if command == "favorite":
                on = p.get("on", True)
                await s.send("favorite", favorite=on, id=st["id"])
                await s.wait_for("favorite", self.confirm_delay)
                return f"« {title} » {'ajouté aux' if on else 'retiré des'} favoris."
            rating = round(value)
            await s.send("rating", rating=rating, id=st["id"])
            await s.wait_for("rating", self.confirm_delay)
            return f"« {title} » noté {rating}/5." if rating else f"Note de « {title} » effacée."
        if command == "search":
            kind = p.get("kind") or "track"
            items = await self._search(s, kind, p["query"])
            if not items:
                return f"Aucun résultat pour « {_clean(p['query'], 80)} »."
            lines = [f"- {self._label(kind, it)} (id {_clean(it.get('id'), 80)})" for it in items[: self.search_limit]]
            return f"Résultats ({kind}) pour « {_clean(p['query'], 80)} » :\n" + "\n".join(lines)
        if command == "play_media":
            return await self._play_media(s, p)
        raise FeishinError(f"Commande non prise en charge : {command}")

    async def _search(self, s: RemoteSession, kind: str, query: str) -> list[dict[str, Any]]:
        plural = KINDS[kind]
        data = await s.request(f"{plural}-request", f"{plural}-response",
                               searchTerm=str(query).strip()[:200], startIndex=0,
                               limit=max(self.search_limit, 10))
        items = data.get("items") if isinstance(data, dict) else None
        return [it for it in items or [] if isinstance(it, dict) and it.get("id")]

    @staticmethod
    def _label(kind: str, item: dict[str, Any]) -> str:
        name = f"« {_clean(item.get('name'))} »"
        if kind == "track":
            extra = ", ".join(_clean(x, 80) for x in (item.get("artistName"), item.get("album")) if x)
        elif kind == "album":
            extra = ", ".join(str(x) for x in (_clean(item.get("albumArtistName"), 80), item.get("releaseYear")) if x)
        else:
            extra = f"{item['songCount']} titres" if item.get("songCount") else ""
        return f"{name} ({extra})" if extra else name

    async def _play_media(self, s: RemoteSession, p: dict[str, Any]) -> str:
        kind = p.get("kind") or "track"
        label = f"l'élément {_clean(p.get('id'), 80)}"
        item_id = p.get("id")
        if not item_id:
            query = str(p["query"]).strip()
            items = await self._search(s, kind, query)
            if not items:
                return f"Rien trouvé pour « {_clean(query, 80)} » ({kind})."
            q = query.casefold()
            best = next((it for it in items if str(it.get("name", "")).casefold() == q), items[0])
            item_id, label = best["id"], self._label(kind, best)
        mode = PLAY_MODES[p.get("play_mode") or "now"]
        ack = await s.request(f"play-{kind}", "operation-ack", id=item_id, playType=mode)
        if isinstance(ack, dict) and ack.get("error"):
            raise FeishinError(f"Feishin n'a pas pu lancer {label} : {_clean(ack['error'], 200)}")
        if mode == "now" or mode == "shuffle":
            return f"Lecture de {label}."
        return f"{label} ajouté {'après le titre en cours' if mode == 'next' else 'en fin de file'}."

    # --- via MPRIS ---

    async def _run_mpris(self, command: str, p: dict[str, Any]) -> str:
        if command not in MPRIS_COMMANDS:
            raise FeishinError(f"« {command} » nécessite le Remote control de Feishin (non joignable).")
        m, value = self.mpris, p.get("value")
        if command == "status":
            return describe(await m.state())
        if command in ("play", "pause", "toggle", "next", "previous"):
            await m.call({"play": "Play", "pause": "Pause", "toggle": "PlayPause",
                          "next": "Next", "previous": "Previous"}[command])
            if command == "pause":
                return "Musique en pause."
            await asyncio.sleep(min(self.confirm_delay, 0.5))   # le temps que les métadonnées suivent
            return describe(await m.state())
        if command in ("volume", "volume_up", "volume_down"):
            if command == "volume":
                target = value
            else:
                current = (await m.get("Volume")).get("Volume") or 0
                step = value if value is not None else 10
                target = current * 100 + (step if command == "volume_up" else -step)
            target = round(min(100, max(0, target)))
            await m.set("Volume", "d", str(target / 100))
            return f"Volume réglé à {target} %."
        if command in ("seek", "forward", "rewind"):
            st = await m.state()
            if not st.get("trackid"):
                return "Aucun titre en cours."
            step = value if value is not None else 10
            target = value if command == "seek" else st["position"] + (step if command == "forward" else -step)
            target = max(0.0, float(target))
            if st.get("duration"):
                target = min(target, max(0.0, st["duration"] - 1))
            await m.call("SetPosition", "ox", str(st["trackid"]), str(int(target * 1e6)))
            return f"Lecture placée à {_clock(target)}."
        if command == "shuffle":
            want = p.get("on")
            if want is None:
                want = not (await m.get("Shuffle")).get("Shuffle")
            await m.set("Shuffle", "b", "true" if want else "false")
            return "Lecture aléatoire " + ("activée." if want else "désactivée.")
        # repeat
        want = p.get("repeat_mode")
        if want is None:   # bascule dans l'ordre de Feishin : aucune -> toute la file -> le titre
            current = MPRIS_TO_REPEAT.get((await m.get("LoopStatus")).get("LoopStatus") or "", "none")
            want = REPEAT_MODES[(("none", "all", "one").index(current) + 1) % 3]
        await m.set("LoopStatus", "s", REPEAT_TO_MPRIS[want])
        return f"Répétition : {REPEAT_FR[want]}."
