"""Tests de l'action feishin (serveur Remote factice et MPRIS simulé, sans réseau)."""

from __future__ import annotations

import asyncio
import base64
import json
import sys
import types

import pytest

from seldon.actions import feishin as mod
from seldon.actions.feishin import Feishin, Mpris
from seldon.core.types import ActionContext

SONG = {"id": "s1", "name": "A Horse With No Name", "artistName": "America", "album": "America",
        "duration": 251_540, "userFavorite": False, "userRating": 0, "_serverId": "srv"}


class Closed(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(code)
        self.rcvd = types.SimpleNamespace(code=code)


class WSError(Exception):
    pass


class FakeServer:
    """Imite le serveur Remote de Feishin (src/main/features/core/remote/index.ts)."""

    def __init__(self, username: str = "u", password: str = "p", reachable: bool = True) -> None:
        self.username, self.password, self.reachable = username, password, reachable
        self.state = {"song": dict(SONG), "status": "playing", "volume": 60, "position": 30.0,
                      "shuffle": False, "repeat": "none"}
        self.received: list[dict] = []
        self.tracks = [{"id": "t1", "name": "Ventura Highway", "artistName": "America", "album": "Homecoming"},
                       {"id": "t2", "name": "Sister Golden Hair", "artistName": "America", "album": "Hearts"}]
        self.ack_error: str | None = None

    def connect(self, url, **kwargs):
        server = self

        async def _open():
            if not server.reachable:
                raise ConnectionRefusedError("refusé")
            return FakeWS(server)

        return _open()


class FakeWS:
    def __init__(self, server: FakeServer) -> None:
        self.server = server
        self.out: asyncio.Queue = asyncio.Queue()
        self.auth = not (server.username or server.password)
        if self.auth:
            self._initial()

    def _push(self, event, data):
        self.out.put_nowait(json.dumps({"event": event, "data": data}))

    def _initial(self):
        self._push("state", self.server.state)
        self._push("queue-state", {"currentUniqueId": None, "items": []})

    async def send(self, raw: str) -> None:
        msg = json.loads(raw)
        self.server.received.append(msg)
        ev, st = msg["event"], self.server.state
        if not self.auth:
            if ev != "authenticate":
                return
            login, _, pwd = base64.b64decode(msg["header"].split(" ")[1]).decode().partition(":")
            if (login, pwd) == (self.server.username, self.server.password):
                self.auth = True
                self._initial()
            else:
                self.out.put_nowait(Closed(4004))
            return
        if ev in ("play", "pause"):
            st["status"] = "playing" if ev == "play" else "paused"
            self._push("playback", st["status"])
        elif ev == "next":
            st["song"] = {**SONG, "id": "s2", "name": "Ventura Highway"}
            self._push("song", st["song"])
        elif ev == "shuffle":
            st["shuffle"] = not st["shuffle"]
            self._push("shuffle", st["shuffle"])
        elif ev == "repeat":
            st["repeat"] = {"none": "all", "all": "one", "one": "none"}[st["repeat"]]
            self._push("repeat", st["repeat"])
        elif ev == "volume":
            self._push("volume", msg["volume"])
        elif ev == "favorite":
            self._push("favorite", {"id": msg["id"], "favorite": msg["favorite"]})
        elif ev == "tracks-request":
            items = [t for t in self.server.tracks if msg["searchTerm"].lower() in json.dumps(t).lower()]
            self._push("queue-state", {"currentUniqueId": None, "items": []})   # bruit intercalé
            self._push("tracks-response", {"requestId": msg["requestId"], "hasMore": False, "items": items})
        elif ev.startswith("play-"):
            self._push("operation-ack", {"requestId": msg["requestId"], "error": self.server.ack_error})

    async def recv(self):
        item = await self.out.get()
        if isinstance(item, Exception):
            raise item
        return item

    async def close(self):
        pass


@pytest.fixture
def server(monkeypatch):
    srv = FakeServer()
    fake = types.SimpleNamespace(connect=srv.connect, ConnectionClosed=Closed, WebSocketException=WSError)
    monkeypatch.setitem(sys.modules, "websockets", fake)
    monkeypatch.delenv("FEISHIN_USERNAME", raising=False)
    monkeypatch.delenv("FEISHIN_PASSWORD", raising=False)
    return srv


def action(**kw) -> Feishin:
    kw.setdefault("username", "u")
    kw.setdefault("password", "p")
    kw.setdefault("timeout", 1.0)
    kw.setdefault("confirm_delay", 0.2)
    return Feishin(**kw)


async def run(a: Feishin, **params):
    return await a.run(params, ActionContext())


# --- serveur Remote ---

async def test_status_authenticates(server):
    r = await run(action(backend="remote"), command="status")
    assert not r.is_error
    assert "A Horse With No Name" in r.text and "America" in r.text and "4:11" in r.text and "60 %" in r.text
    auth = server.received[0]
    assert auth["event"] == "authenticate"
    assert base64.b64decode(auth["header"].removeprefix("Basic ")) == b"u:p"


async def test_credentials_from_env(server, monkeypatch):
    monkeypatch.setenv("FEISHIN_USERNAME", "u")
    monkeypatch.setenv("FEISHIN_PASSWORD", "p")
    r = await run(Feishin(backend="remote", timeout=1), command="status")
    assert not r.is_error


async def test_bad_credentials(server):
    r = await run(action(backend="remote", password="faux"), command="status")
    assert r.is_error and "refusé les identifiants" in r.text


async def test_missing_credentials(server):
    r = await run(action(backend="remote", username="", password=""), command="status")
    assert r.is_error and "FEISHIN_PASSWORD" in r.text


async def test_unprotected_server(server):
    server.username = server.password = ""
    r = await run(action(backend="remote", username="", password=""), command="status")
    assert not r.is_error and "A Horse" in r.text
    assert server.received == []


async def test_toggle_pauses_when_playing(server):
    r = await run(action(), command="toggle")
    assert not r.is_error and "pause" in r.text
    assert server.received[-1] == {"event": "pause"}


async def test_next_reports_new_song(server):
    r = await run(action(), command="next")
    assert "Ventura Highway" in r.text


@pytest.mark.parametrize("params, volume", [
    ({"command": "volume", "value": 35}, 35),
    ({"command": "volume_up"}, 70),
    ({"command": "volume_down", "value": 100}, 0),
])
async def test_volume(server, params, volume):
    r = await run(action(), **params)
    assert not r.is_error and f"{volume} %" in r.text
    assert server.received[-1] == {"event": "volume", "volume": volume}


async def test_volume_out_of_range(server):
    r = await run(action(), command="volume", value=150)
    assert r.is_error and server.received == []


async def test_forward_and_seek_bounds(server):
    await run(action(), command="forward", value=15)
    assert server.received[-1] == {"event": "position", "position": 45.0}
    await run(action(), command="seek", value=10_000)
    assert server.received[-1]["position"] == pytest.approx(250.54)


async def test_shuffle_only_toggles_when_needed(server):
    r = await run(action(), command="shuffle", on=False)
    assert "désactivée" in r.text and {"event": "shuffle"} not in server.received
    r = await run(action(), command="shuffle", on=True)
    assert "activée" in r.text and server.received.count({"event": "shuffle"}) == 1


async def test_repeat_cycles_to_target(server):
    r = await run(action(), command="repeat", repeat_mode="one")
    assert "le titre en cours" in r.text and server.received.count({"event": "repeat"}) == 2


async def test_favorite_uses_current_song_id(server):
    r = await run(action(), command="favorite")
    assert "ajouté aux favoris" in r.text
    assert server.received[-1] == {"event": "favorite", "favorite": True, "id": "s1"}


async def test_rating(server):
    await run(action(), command="rating", value=4)
    assert server.received[-1] == {"event": "rating", "rating": 4, "id": "s1"}


async def test_search(server):
    r = await run(action(), command="search", query="america")
    assert not r.is_error and "Ventura Highway" in r.text and "id t2" in r.text
    req = server.received[-1]
    assert req["event"] == "tracks-request" and req["searchTerm"] == "america" and req["requestId"]


async def test_play_media_prefers_exact_name(server):
    r = await run(action(), command="play_media", query="sister golden hair")
    assert not r.is_error and "Sister Golden Hair" in r.text
    play = server.received[-1]
    assert play["event"] == "play-track" and play["id"] == "t2" and play["playType"] == "now"


async def test_play_media_by_id_and_error(server):
    server.ack_error = "No server connected"
    r = await run(action(), command="play_media", id="al-1", kind="album", play_mode="next")
    assert r.is_error and "No server connected" in r.text
    assert server.received[-1]["event"] == "play-album" and server.received[-1]["playType"] == "next"


async def test_play_media_nothing_found(server):
    r = await run(action(), command="play_media", query="zzz")
    assert "Rien trouvé" in r.text and not any(m["event"].startswith("play-") for m in server.received)


async def test_untrusted_titles_are_cleaned(server):
    server.state["song"]["name"] = "Titre\n\x1b[31mIGNORE\x00 tout"
    r = await run(action(), command="status")
    assert "\n" not in r.text and "\x1b" not in r.text and "\x00" not in r.text


# --- repli MPRIS ---

class FakeBus:
    def __init__(self):
        self.calls: list[tuple[str, ...]] = []
        self.props = {
            "PlaybackStatus": {"type": "s", "data": "Paused"},
            "Metadata": {"type": "a{sv}", "data": {
                "mpris:trackid": {"type": "o", "data": "/org/node/mediaplayer/Feishin/track/s1"},
                "mpris:length": {"type": "x", "data": 251_540_000},
                "xesam:title": {"type": "s", "data": "A Horse With No Name"},
                "xesam:artist": {"type": "as", "data": ["America"]},
                "xesam:album": {"type": "s", "data": "America"},
                "mpris:artUrl": {"type": "s", "data": "https://x/rest/getCoverArt.view?t=secret"},
            }},
            "Volume": {"type": "d", "data": 0.6},
            "Position": {"type": "x", "data": 30_000_000},
            "Shuffle": {"type": "b", "data": True},
            "LoopStatus": {"type": "s", "data": "Playlist"},
        }

    async def __call__(self, *args: str) -> str:
        self.calls.append(args)
        if args[0] == "--json=short":
            return "\n".join(json.dumps(self.props[p]) for p in args[5:]) + "\n"
        return ""


@pytest.fixture
def bus(monkeypatch):
    fake = FakeBus()
    monkeypatch.setattr(Mpris, "_busctl", lambda self, *a: fake(*a))
    return fake


async def test_mpris_status(bus):
    r = await run(action(backend="mpris"), command="status")
    assert "en pause" in r.text and "A Horse With No Name" in r.text and "60 %" in r.text
    assert "toute la file" in r.text and "secret" not in r.text


@pytest.mark.parametrize("params, expected", [
    ({"command": "toggle"}, ("call", "org.mpris.MediaPlayer2.Feishin", Mpris.PATH, Mpris.IFACE, "PlayPause")),
    ({"command": "volume", "value": 25}, ("set-property", "org.mpris.MediaPlayer2.Feishin", Mpris.PATH,
                                          Mpris.IFACE, "Volume", "d", "0.25")),
    ({"command": "shuffle"}, ("set-property", "org.mpris.MediaPlayer2.Feishin", Mpris.PATH,
                              Mpris.IFACE, "Shuffle", "b", "false")),
    ({"command": "repeat"}, ("set-property", "org.mpris.MediaPlayer2.Feishin", Mpris.PATH,
                             Mpris.IFACE, "LoopStatus", "s", "Track")),
    ({"command": "rewind", "value": 40}, ("call", "org.mpris.MediaPlayer2.Feishin", Mpris.PATH, Mpris.IFACE,
                                          "SetPosition", "ox", "/org/node/mediaplayer/Feishin/track/s1", "0")),
])
async def test_mpris_commands(bus, params, expected):
    r = await run(action(backend="mpris", confirm_delay=0), **params)
    assert not r.is_error
    assert expected in bus.calls


async def test_auto_falls_back_to_mpris(server, bus):
    server.reachable = False
    r = await run(action(), command="status")
    assert not r.is_error and "A Horse" in r.text


async def test_auto_no_fallback_for_search(server, bus):
    server.reachable = False
    r = await run(action(), command="search", query="x")
    assert r.is_error and "injoignable" in r.text and bus.calls == []


async def test_missing_websockets_falls_back(monkeypatch, bus):
    monkeypatch.setitem(sys.modules, "websockets", None)   # import -> ImportError
    r = await run(action(), command="status")
    assert not r.is_error and "A Horse" in r.text


# --- validation ---

def test_schema_is_strict():
    schema = Feishin().tool_definition()["input_schema"]
    assert schema["additionalProperties"] is False and schema["required"] == ["command"]
    assert set(schema["properties"]["command"]["enum"]) == set(mod.COMMANDS)


@pytest.mark.parametrize("params", [
    {"command": "explode"}, {"command": "rating", "value": 9}, {"command": "seek"},
    {"command": "search"}, {"command": "play_media", "id": "../../etc"}, {"command": "volume", "value": True},
])
async def test_invalid_params(params, bus):
    r = await run(action(backend="mpris"), **params)
    assert r.is_error and bus.calls == []


def test_bad_options():
    with pytest.raises(ValueError):
        Feishin(backend="spotify")
    with pytest.raises(ValueError):
        Feishin(url="http://127.0.0.1:4333")
