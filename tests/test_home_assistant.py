"""Tests de l'action home_assistant (client MCP Streamable HTTP simulé, sans réseau)."""

from __future__ import annotations

import json

import httpx
import pytest

from seldon.actions.home_assistant import HomeAssistant, _domains_by_name, _sse_messages
from seldon.core.types import ActionContext

URL = "http://ha.test/api/mcp"

TOOLS = [
    {"name": "HassTurnOn", "description": "Turns on/opens/presses a device or entity. More text.",
     "inputSchema": {"type": "object", "properties": {
         "name": {"type": "string"}, "area": {"type": "string"},
         "domain": {"type": "array", "items": {"type": "string", "enum": ["light", "lock", "cover"]}}}}},
    {"name": "HassTurnOff", "description": "Turns off/closes a device or entity.",
     "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}, "domain": {"type": "array"}}}},
    {"name": "HassLightSet", "description": "Sets the brightness or color of a light.",
     "inputSchema": {"type": "object", "properties": {
         "name": {"type": "string"}, "brightness": {"type": "integer", "minimum": 0, "maximum": 100}},
         "required": ["brightness"]}},
    {"name": "GetLiveContext", "description": "Provides real-time information about the CURRENT state.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "HassBroadcast", "description": "Broadcast a message.", "inputSchema": {"type": "object", "properties": {}}},
]

LIVE = ("Live Context: An overview of the areas and the devices in this smart home:\n"
        "- names: Lampe salon\n  domain: light\n  state: 'on'\n  areas: Salon\n"
        "- names: Entrée, Porte principale\n  domain: lock\n  state: locked\n")


class FakeHA:
    """Serveur MCP minimal : enregistre les requêtes, répond en JSON ou en SSE."""

    def __init__(self, sse: bool = False, session: str | None = "sess-1", token: str = "secret-token"):
        self.sse = sse
        self.session = session
        self.token = token
        self.requests: list[tuple[str, dict, httpx.Headers]] = []
        self.calls: list[tuple[str, dict]] = []
        self.expire_once = False
        self.results: dict[str, dict] = {}

    def reply(self, msg_id, result=None, error=None) -> httpx.Response:
        body = {"jsonrpc": "2.0", "id": msg_id, **({"error": error} if error else {"result": result})}
        if self.sse:
            text = (": ping\n\nevent: message\ndata: {\"jsonrpc\": \"2.0\", \"method\": \"notifications/progress\"}\n\n"
                    f"event: message\ndata: {json.dumps(body)}\n\n")
            return httpx.Response(200, text=text, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=body)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            return httpx.Response(200)
        if request.headers.get("authorization") != f"Bearer {self.token}":
            return httpx.Response(401, text="401: Unauthorized")
        msg = json.loads(request.content)
        self.requests.append((msg.get("method"), msg, request.headers))
        method = msg.get("method")
        if method == "initialize":
            resp = self.reply(msg["id"], {"protocolVersion": "2025-03-26", "capabilities": {"tools": {}},
                                          "serverInfo": {"name": "home-assistant", "version": "1"}})
            if self.session:
                resp.headers["mcp-session-id"] = self.session
            return resp
        if method == "notifications/initialized":
            return httpx.Response(202)
        if self.session and request.headers.get("mcp-session-id") != self.session:
            return httpx.Response(400, text="missing session")
        if self.expire_once:
            self.expire_once = False
            self.session = "sess-2"
            return httpx.Response(404)
        if method == "tools/list":
            return self.reply(msg["id"], {"tools": TOOLS})
        if method == "tools/call":
            name, args = msg["params"]["name"], msg["params"]["arguments"]
            self.calls.append((name, args))
            if name == "GetLiveContext":
                return self.reply(msg["id"], {"content": [{"type": "text", "text": json.dumps(
                    {"success": True, "result": LIVE})}], "isError": False})
            if name in self.results:
                return self.reply(msg["id"], self.results[name])
            return self.reply(msg["id"], {"content": [{"type": "text", "text": f"fait : {name}"}], "isError": False})
        return self.reply(msg["id"], error={"code": -32601, "message": "Method not found"})


def make(fake: FakeHA, **options) -> HomeAssistant:
    action = HomeAssistant(url=URL, token=options.pop("token", "secret-token"), **options)
    action.client = httpx.AsyncClient(transport=httpx.MockTransport(fake))
    return action


def make_ctx(answer: bool = True) -> tuple[ActionContext, list[str]]:
    asked: list[str] = []

    async def confirm(question: str) -> bool:
        asked.append(question)
        return answer

    return ActionContext(confirm=confirm), asked


# --- protocole ---

@pytest.mark.parametrize("sse", [False, True])
async def test_handshake_list_and_call(sse):
    fake = FakeHA(sse=sse)
    action = make(fake)
    await action.start()
    assert list(action.tools) == [t["name"] for t in TOOLS]
    methods = [m for m, _, _ in fake.requests]
    assert methods == ["initialize", "notifications/initialized", "tools/list"]
    init = fake.requests[0][1]
    assert init["params"]["protocolVersion"] and init["params"]["clientInfo"]["name"] == "seldon"
    assert "id" not in fake.requests[1][1]   # notification
    listed = fake.requests[2][2]
    assert listed["mcp-session-id"] == "sess-1" and listed["mcp-protocol-version"] == "2025-03-26"
    assert "text/event-stream" in listed["accept"]

    ctx, asked = make_ctx()
    r = await action.run({"tool": "HassTurnOn", "arguments": {"name": "Lampe salon"}}, ctx)
    assert not r.is_error and r.text == "fait : HassTurnOn" and asked == []
    # nom seul : domaine vérifié via GetLiveContext (lecture) avant la commande
    assert fake.calls == [("GetLiveContext", {}), ("HassTurnOn", {"name": "Lampe salon"})]
    await action.stop()


async def test_without_session_id():
    fake = FakeHA(session=None)
    action = make(fake)
    await action.start()
    assert "mcp-session-id" not in fake.requests[-1][2]
    r = await action.run({"tool": "GetLiveContext", "arguments": {}}, make_ctx()[0])
    assert not r.is_error and r.text.startswith("Live Context") and "\n- names" in r.text   # JSON déballé


async def test_session_expired_reinitializes():
    fake = FakeHA()
    action = make(fake)
    await action.start()
    fake.expire_once = True
    r = await action.run({"tool": "HassTurnOn", "arguments": {"name": "Lampe salon"}}, make_ctx()[0])
    assert not r.is_error
    assert [m for m, _, _ in fake.requests].count("initialize") == 2
    assert fake.requests[-1][2]["mcp-session-id"] == "sess-2"


def test_sse_parser():
    lines = ["event: message", "data: {\"id\": 1,", "data:  \"result\": {}}", "", ": commentaire", "data: pas du json", ""]
    assert _sse_messages(lines) == [{"id": 1, "result": {}}]


# --- démarrage et erreurs ---

async def test_unreachable_at_start_then_lazy_load():
    fake = FakeHA()
    down = True

    def handler(request):
        if down:
            raise httpx.ConnectError("connexion refusée")
        return fake(request)

    action = HomeAssistant(url=URL, token="secret-token")
    action.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await action.start()   # ne lève pas
    assert action.tools is None and "indisponible" in action.description
    assert "enum" not in action.input_schema["properties"]["tool"]

    r = await action.run({"tool": "HassTurnOn", "arguments": {}}, make_ctx()[0])
    assert r.is_error and "indisponible" in r.text

    down = False
    r = await action.run({"tool": "HassTurnOn", "arguments": {"name": "Lampe salon"}}, make_ctx()[0])
    assert not r.is_error
    assert action.input_schema["properties"]["tool"]["enum"][0] == "HassTurnOn"


async def test_bad_token():
    action = make(FakeHA(), token="mauvais")
    await action.start()
    r = await action.run({"tool": "HassTurnOn", "arguments": {}}, make_ctx()[0])
    assert r.is_error and "401" in r.text and "mauvais" not in r.text


async def test_missing_token(monkeypatch):
    monkeypatch.delenv("HA_TOKEN", raising=False)
    monkeypatch.delenv("HOME_ASSISTANT_TOKEN", raising=False)
    action = HomeAssistant(url=URL)
    await action.start()
    r = await action.run({"tool": "HassTurnOn"}, make_ctx()[0])
    assert r.is_error and "HA_TOKEN" in r.text


def test_token_from_env(monkeypatch):
    monkeypatch.delenv("HA_TOKEN", raising=False)
    monkeypatch.setenv("HOME_ASSISTANT_TOKEN", "env-token")
    assert HomeAssistant(url=URL)._headers()["Authorization"] == "Bearer env-token"


async def test_tool_is_error_and_jsonrpc_error():
    fake = FakeHA()
    fake.results["HassLightSet"] = {"content": [{"type": "text", "text": "Brightness required"}], "isError": True}
    action = make(fake)
    r = await action.run({"tool": "HassLightSet", "arguments": {"name": "Lampe salon"}}, make_ctx()[0])
    assert r.is_error and "Brightness required" in r.text and "brightness*" in r.text

    r = await action._request("tools/list", {})   # sanity : la session marche
    assert r["tools"]
    with pytest.raises(Exception, match="Method not found"):
        await action._request("resources/list", {})


async def test_unknown_and_denied_tools():
    fake = FakeHA()
    action = make(fake, denied_tools=["HassBroadcast"])
    await action.start()
    assert "HassBroadcast" not in action.tools and "HassBroadcast" not in action.description
    r = await action.run({"tool": "HassBroadcast", "arguments": {}}, make_ctx()[0])
    assert r.is_error and "non autorisé" in r.text
    r = await action.run({"tool": "Inexistant", "arguments": {}}, make_ctx()[0])
    assert r.is_error and "inconnu" in r.text and "HassTurnOn" in r.text
    assert fake.calls == []

    only = make(FakeHA(), allowed_tools=["GetLiveContext"])
    await only.start()
    assert list(only.tools) == ["GetLiveContext"]


async def test_result_truncated():
    fake = FakeHA()
    fake.results["HassTurnOn"] = {"content": [{"type": "text", "text": "x" * 10_000}], "isError": False}
    action = make(fake, max_result_chars=500)
    r = await action.run({"tool": "HassTurnOn", "arguments": {"name": "Lampe salon"}}, make_ctx()[0])
    assert len(r.text) <= 500 and "tronqué" in r.text


async def test_flat_and_string_arguments():
    fake = FakeHA()
    action = make(fake)
    await action.run({"tool": "HassTurnOn", "name": "Lampe salon"}, make_ctx()[0])
    await action.run({"tool": "HassTurnOff", "arguments": '{"name": "Lampe salon"}'}, make_ctx()[0])
    assert [c for c in fake.calls if c[0] != "GetLiveContext"] == [
        ("HassTurnOn", {"name": "Lampe salon"}), ("HassTurnOff", {"name": "Lampe salon"})]
    assert fake.calls.count(("GetLiveContext", {})) == 1   # contexte mis en cache


async def test_description_lists_tools():
    action = make(FakeHA())
    await action.start()
    d = action.description
    assert "- HassTurnOn : Turns on/opens/presses a device or entity." in d and "More text" not in d
    assert "domain (light|lock|cover)" in d and "brightness* (nombre 0-100)" in d


# --- confirmation ---

@pytest.mark.parametrize("params", [
    {"tool": "HassTurnOff", "arguments": {"name": "Porte principale"}},   # domaine trouvé via GetLiveContext
    {"tool": "HassTurnOn", "arguments": {"domain": ["lock"]}},
    {"tool": "HassTurnOn", "arguments": {"name": "Portail du jardin"}},  # mot-clé
    {"tool": "HassTurnOn", "arguments": {"name": "x", "device_class": ["garage"]}},
])
async def test_sensitive_requires_confirmation(params):
    fake = FakeHA()
    action = make(fake)
    await action.start()
    ctx, asked = make_ctx(answer=False)
    r = await action.run(params, ctx)
    assert r.is_error and "non confirmée" in r.text and len(asked) == 1
    assert all(name == "GetLiveContext" for name, _ in fake.calls)   # rien d'exécuté

    ctx, asked = make_ctx(answer=True)
    r = await action.run(params, ctx)
    assert not r.is_error and fake.calls[-1][0] == params["tool"]


async def test_harmless_no_confirmation_and_confirm_tools():
    fake = FakeHA()
    action = make(fake, confirm_tools=["HassBroadcast"])
    await action.start()
    ctx, asked = make_ctx(answer=False)
    assert not (await action.run({"tool": "HassTurnOff", "arguments": {"name": "Lampe salon"}}, ctx)).is_error
    assert not (await action.run({"tool": "GetLiveContext", "arguments": {}}, ctx)).is_error
    assert asked == []
    r = await action.run({"tool": "HassBroadcast", "arguments": {"message": "coucou"}}, ctx)
    assert r.is_error and len(asked) == 1


def test_live_context_parser():
    assert _domains_by_name(LIVE) == {"lampe salon": "light", "entrée": "lock", "porte principale": "lock"}
