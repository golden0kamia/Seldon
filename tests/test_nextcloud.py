"""Tests de l'action nextcloud (serveur MCP simulé, sans réseau)."""

from __future__ import annotations

import base64
import json

import httpx
import pytest

from seldon.actions.nextcloud import Nextcloud
from seldon.core.types import ActionContext

URL = "http://nc.test/mcp"


def tool(name: str, *required: str) -> dict:
    return {"name": name, "description": f"Outil {name}. Détails.",
            "inputSchema": {"type": "object", "properties": {r: {"type": "string"} for r in required},
                            "required": list(required)}}


TOOLS = [tool("nc_notes_search_notes", "query"), tool("nc_notes_create_note", "title"),
         tool("nc_notes_delete_note", "note_id"), tool("nc_calendar_get_upcoming_events"),
         tool("nc_webdav_read_file", "path"), tool("nc_auth_get_status"), tool("nc_get_capabilities"), tool("deck_get_boards"),
         tool("nc_share_create", "path")]


class FakeNC:
    """Serveur MCP minimal : mémorise les en-têtes Authorization et les appels."""

    def __init__(self):
        self.auth: list[str | None] = []
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            return httpx.Response(200)
        self.auth.append(request.headers.get("authorization"))
        msg = json.loads(request.content)
        method = msg.get("method")
        if method == "notifications/initialized":
            return httpx.Response(202)
        if method == "initialize":
            result = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}}
        elif method == "tools/list":
            result = {"tools": TOOLS}
        else:
            name, args = msg["params"]["name"], msg["params"]["arguments"]
            self.calls.append((name, args))
            result = {"content": [{"type": "text", "text": f"fait : {name}"}], "isError": False}
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": msg["id"], "result": result},
                              headers={"mcp-session-id": "s1"})


def make(fake: FakeNC, **options) -> Nextcloud:
    action = Nextcloud(url=URL, **options)
    action.client = httpx.AsyncClient(transport=httpx.MockTransport(fake))
    return action


def make_ctx(answer: bool = True) -> tuple[ActionContext, list[str]]:
    asked: list[str] = []

    async def confirm(question: str) -> bool:
        asked.append(question)
        return answer

    return ActionContext(confirm=confirm), asked


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    for var in ("NEXTCLOUD_USERNAME", "NEXTCLOUD_PASSWORD", "NEXTCLOUD_MCP_TOKEN"):
        monkeypatch.delenv(var, raising=False)


async def test_apps_filter_and_call_without_auth():
    fake = FakeNC()
    action = make(fake)
    await action.start()
    assert list(action.tools) == ["nc_notes_search_notes", "nc_notes_create_note", "nc_notes_delete_note",
                                  "nc_calendar_get_upcoming_events"]
    assert "nc_webdav_read_file" not in action.description and "query*" in action.description
    r = await action.run({"tool": "nc_notes_create_note", "arguments": {"title": "Courses"}}, make_ctx()[0])
    assert not r.is_error and r.text == "fait : nc_notes_create_note"
    assert set(fake.auth) == {None}
    r = await action.run({"tool": "nc_webdav_read_file", "arguments": {"path": "/a"}}, make_ctx()[0])
    assert r.is_error and "non autorisé" in r.text

    everything = make(FakeNC(), apps=None)
    await everything.start()
    assert len(everything.tools) == len(TOOLS)
    other = make(FakeNC(), apps=["deck", "sharing"])
    await other.start()
    assert list(other.tools) == ["deck_get_boards", "nc_share_create"]


async def test_destructive_tools_need_confirmation():
    fake = FakeNC()
    action = make(fake)
    ctx, asked = make_ctx(answer=False)
    r = await action.run({"tool": "nc_notes_delete_note", "arguments": {"note_id": 3}}, ctx)
    assert r.is_error and "non confirmée" in r.text and fake.calls == []
    assert asked and "note_id « 3 »" in asked[0]
    ctx, asked = make_ctx(answer=False)
    assert not (await action.run({"tool": "nc_notes_search_notes", "arguments": {"query": "x"}}, ctx)).is_error
    assert asked == []


async def test_basic_auth(monkeypatch):
    monkeypatch.setenv("NEXTCLOUD_USERNAME", "bastien")
    monkeypatch.setenv("NEXTCLOUD_PASSWORD", "app-pass")
    fake = FakeNC()
    action = make(fake, auth="basic")
    await action.start()
    assert set(fake.auth) == {"Basic " + base64.b64encode(b"bastien:app-pass").decode()}


async def test_missing_credentials():
    action = make(FakeNC(), auth="bearer")
    r = await action.run({"tool": "nc_notes_search_notes"}, make_ctx()[0])
    assert r.is_error and "NEXTCLOUD_MCP_TOKEN" in r.text
    with pytest.raises(ValueError):
        Nextcloud(auth="oauth")
