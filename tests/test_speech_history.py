"""Parole anticipée (découpage du flux) et historique borné des cerveaux."""

from __future__ import annotations

import asyncio
import json

import pytest

from seldon.brain import history
from seldon.brain.speech import SpeechStream, partial_json_string, sentence_end
from seldon.core.types import ActionResult


# -- flux de l'argument « text » ------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ('{"te', None),
    ('{"text": "', ("", False)),
    ('{"text":"Il est 14 h', ("Il est 14 h", False)),
    ('{"text": "Dis \\"oui\\" et \\', ('Dis "oui" et ', False)),    # échappement incomplet
    ('{"text": "caf\\u00e', ("caf", False)),
    ('{"text": "caf\\u00e9 !"}', ("café !", True)),
    ('{"text": "ligne\nbrute"', ("ligne\nbrute", True)),
])
def test_partial_json_string(raw, expected):
    assert partial_json_string(raw) == expected


def test_sentence_end_needs_following_text_and_skips_abbreviations():
    assert sentence_end("Bonjour.") == 0                   # la suite n'est pas encore écrite
    assert sentence_end("Bonjour. Il") == len("Bonjour.")
    assert sentence_end("Oui ! Non ? Peut-être… Bon") == len("Oui ! Non ? Peut-être…")
    assert sentence_end("Voyez M. Dupont à 3.5 km") == 0
    assert sentence_end("« C'est fini. » Puis") == len("« C'est fini. »")
    assert sentence_end("Un. Deux. Trois", start=4) == len("Un. Deux.")


async def test_speech_stream_says_sentences_in_order_and_stops_on_interruption():
    said = []

    async def execute(name, params):
        said.append(params["text"])
        await asyncio.sleep(0)
        if params["text"].startswith("Deux"):
            return ActionResult("Interrompu par l'utilisateur (« stop »)")
        return ActionResult("Prononcé.")

    speech = SpeechStream(execute)
    raw = json.dumps({"text": "Un. Deux. Trois. Quatre."})
    for k in range(1, len(raw) + 1):
        speech.feed(raw[:k])
        await asyncio.sleep(0)
    result = await speech.finish("Un. Deux. Trois. Quatre.")
    assert said == ["Un.", "Deux."] and speech.stopped
    assert result.text.startswith("Interrompu") and speech.spoken_text == "Un. Deux."


async def test_speech_stream_unused_without_text():
    async def execute(name, params):
        raise AssertionError("ne doit pas parler")

    speech = SpeechStream(execute)
    speech.feed('{"autre": 1')
    assert await speech.finish(None) is None and not speech.started


# -- historique borné ----------------------------------------------------------------

def openai_turn(i, output="Prononcé.", image=False):
    msgs = [{"role": "user", "content": f"question {i}"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}", "type": "function",
                                                                 "function": {"name": "x", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": f"c{i}", "content": output}]
    if image:
        msgs.append({"role": "user", "content": [{"type": "text", "text": "Image :"},
                                                 {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + "A" * 9999}}]})
    return msgs


def test_size_ignores_image_payloads():
    assert history.size(openai_turn(0, image=True)) < 200


def test_compact_shortens_old_results_then_drops_whole_turns():
    messages = [m for i in range(6) for m in openai_turn(i, output="sortie " * 400)]
    assert not history.compact(messages, history.OPENAI, max_chars=10**6, max_messages=1000)
    assert history.compact(messages, history.OPENAI, max_chars=8000, max_messages=1000)
    assert messages[-1]["content"] == "sortie " * 400                  # dernier échange intact
    assert all(len(m["content"]) < 1100 for m in messages[:-3] if m["role"] == "tool")
    assert history.size(messages) <= 8000
    # coupé au début d'un échange : chaque résultat suit son appel
    assert messages[0]["role"] == "user" and isinstance(messages[0]["content"], str)
    ids = {c["id"] for m in messages for c in m.get("tool_calls", [])}
    assert all(m["tool_call_id"] in ids for m in messages if m["role"] == "tool")


def test_compact_on_message_count():
    messages = [m for i in range(10) for m in openai_turn(i)]
    history.compact(messages, history.OPENAI, max_chars=10**6, max_messages=20)
    assert len(messages) <= 10 and messages[0]["content"].startswith("question")


def test_drop_old_images_keeps_most_recent():
    messages = [m for i in range(3) for m in openai_turn(i, image=True)]
    assert history.drop_old_images(messages, history.OPENAI, keep=1) == 2
    assert sum(history.OPENAI.image_count(m) for m in messages) == 1
    assert history.OPENAI.image_count(messages[-1]) == 1


def test_ollama_images():
    m = {"role": "tool", "tool_name": "cam", "content": "photo", "images": ["b64"]}
    history.OLLAMA.drop_images(m)
    assert "images" not in m and history.IMAGE_DROPPED in m["content"]


def test_anthropic_cut_removes_orphan_tool_results():
    def turn(i, pending=()):
        return [{"role": "user", "content": [*pending, {"type": "text", "text": f"q{i} " * 300}]},
                {"role": "assistant", "content": [{"type": "tool_use", "id": f"t{i}", "name": "speak", "input": {}}]}]

    messages = []
    for i in range(5):
        pending = [{"type": "tool_result", "tool_use_id": f"t{i - 1}",
                    "content": [{"type": "image", "source": {"data": "A" * 999}}, {"type": "text", "text": "ok"}]}] if i else []
        messages += turn(i, pending)
    history.compact(messages, history.ANTHROPIC, max_chars=3000, max_messages=1000)
    first = messages[0]["content"]
    assert all(b["type"] != "tool_result" for b in first) and first[-1]["type"] == "text"
    kept_ids = {b["id"] for m in messages if m["role"] == "assistant" for b in m["content"]}
    for m in messages:
        for b in m["content"]:
            if b.get("type") == "tool_result":
                assert b["tool_use_id"] in kept_ids
