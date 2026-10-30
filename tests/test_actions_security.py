"""Actions speak, camera et claude_code : paramètres hostiles ou invalides, robustesse."""

from __future__ import annotations

import asyncio
import gc
import json
import stat
import sys
import time

import pytest

from seldon.core.types import ActionContext, Utterance
from seldon.tools.camera import Camera
from seldon.tools.claude_code import ClaudeCode
from seldon.tools.speak import Speak
from seldon.tts.base import TTSEngine
from tests.test_actions import FakeCapture, fake_cv2  # noqa: F401  (fixture)
from tests.test_presence import InterruptingSTT, SlowTTS


# --- speak -----------------------------------------------------------------------

@pytest.mark.parametrize("params", [{}, {"text": None}, {"text": 42}, {"text": ["a"]}, {"text": "   "}])
async def test_speak_rejects_missing_or_invalid_text(params):
    tts, stt = SlowTTS(), InterruptingSTT()
    r = await Speak().run(params, ActionContext(services={"tts": tts, "stt": stt}))
    assert r.is_error and stt.paused == []


class FailingTTS(TTSEngine):
    async def speak(self, text):
        raise RuntimeError("haut-parleur débranché")


class BrokenListener(InterruptingSTT):
    async def interruption(self, spoken):
        raise RuntimeError("micro perdu")


async def test_speak_failure_resumes_listening_without_stray_task_errors():
    loop = asyncio.get_running_loop()
    errors = []
    loop.set_exception_handler(lambda _loop, context: errors.append(context))
    stt = BrokenListener()
    with pytest.raises(RuntimeError, match="débranché"):
        await Speak().run({"text": "bonjour"}, ActionContext(services={"tts": FailingTTS(), "stt": stt}))
    assert stt.paused == [True, False]
    gc.collect()
    await asyncio.sleep(0)
    assert errors == []   # pas de « Task exception was never retrieved »


async def test_speak_cancelled_stops_voice_then_resumes_listening():
    tts, stt = SlowTTS(), InterruptingSTT()
    job = asyncio.create_task(Speak().run({"text": "une longue phrase"}, ActionContext(services={"tts": tts, "stt": stt})))
    await asyncio.sleep(0.01)
    job.cancel()
    with pytest.raises(asyncio.CancelledError):
        await job
    assert tts.stopped.is_set() and stt.paused == [True, False]


# --- caméra ----------------------------------------------------------------------

async def test_camera_options_and_param_types(fake_cv2):
    cam = Camera(width="1280", height=720.0, devices={"bureau": "0", "porte": "/dev/video2"})
    assert cam.devices["bureau"] == {"device": 0, "width": 1280, "height": 720}
    for bad in (["bureau"], {"x": 1}, 3):
        assert (await cam.run({"camera": bad}, ActionContext())).is_error
    with pytest.raises(ValueError):
        Camera(width="large")


async def test_camera_photos_never_overwrite_and_are_private(fake_cv2, tmp_path, monkeypatch):
    folder = tmp_path / "photos"
    cam = Camera(save_dir=str(folder))
    frozen = time.localtime()
    monkeypatch.setattr("seldon.tools.camera.time.localtime", lambda *a: frozen)
    for _ in range(3):   # même seconde
        assert not (await cam.run({}, ActionContext())).is_error
    files = sorted(folder.iterdir())
    assert len(files) == 3 and all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in files)
    assert stat.S_IMODE(folder.stat().st_mode) == 0o700


async def test_camera_save_failure_still_returns_image(fake_cv2, tmp_path):
    blocker = tmp_path / "fichier"
    blocker.write_text("pas un dossier")
    r = await Camera(save_dir=str(blocker / "photos")).run({}, ActionContext())
    assert not r.is_error and r.images and "Enregistrement impossible" in r.text


async def test_camera_not_kept_open_after_stop(fake_cv2):
    cam = Camera(keep_open=True)
    await cam.stop()
    await cam.run({}, ActionContext())
    assert FakeCapture.instances[-1].released and cam._captures == {}


async def test_camera_open_failure_releases_device(fake_cv2, monkeypatch):
    import cv2

    class BadSet(FakeCapture):
        def set(self, prop, value):
            raise cv2.error("réglage refusé")
    monkeypatch.setattr(cv2, "VideoCapture", lambda dev: BadSet(dev))
    with pytest.raises(cv2.error):
        await Camera(width=640).run({}, ActionContext())
    assert FakeCapture.instances[-1].released


# --- claude_code ------------------------------------------------------------------

def fake_claude(tmp_path, output: str = "") -> str:
    """Faux binaire ``claude`` : note ses arguments et affiche ``output`` (ou un résultat JSON)."""
    script = tmp_path / "claude"
    reply = output or json.dumps({"result": "fait", "session_id": "s1", "num_turns": 1, "total_cost_usd": "gratuit",
                                  "permission_denials": ["Bash", {"tool_name": "Edit"}]})
    script.write_text(
        f"#!{sys.executable}\nimport json, sys\n"
        f"open({str(tmp_path / 'args.json')!r}, 'w').write(json.dumps(sys.argv[1:]))\n"
        f"sys.stdout.write({reply!r})\n"
    )
    script.chmod(0o755)
    return str(script)


def ctx(speaker="Bastien", answer=True):
    asked = []

    async def confirm(q):
        asked.append(q)
        return answer
    return ActionContext(utterance=Utterance(text="...", speaker=speaker), confirm=confirm), asked


async def test_claude_code_task_is_never_an_option(tmp_path):
    work = tmp_path / "projet"
    work.mkdir()
    action = ClaudeCode(claude_bin=fake_claude(tmp_path), cwd=str(work), confirm=False)
    c, _ = ctx()
    r = await action.run({"task": "--dangerously-skip-permissions"}, c)
    assert not r.is_error and "fait" in r.text and "permissions refusées : ?, Edit" in r.text
    args = json.loads((tmp_path / "args.json").read_text())
    assert args[:2] == ["-p", "Tâche : --dangerously-skip-permissions"]
    assert not any(a.startswith("--dangerously") for a in args)


async def test_claude_code_speakers_are_case_insensitive(tmp_path):
    action = ClaudeCode(claude_bin=fake_claude(tmp_path), cwd=str(tmp_path), allowed_speakers=["Bastien"])
    c, asked = ctx(speaker="bastien", answer=False)
    r = await action.run({"task": "x"}, c)
    assert "refusé de lancer" in r.text and asked   # autorisé, puis refusé par l'utilisateur
    c, asked = ctx(speaker=None)
    assert "pas autorisé" in (await action.run({"task": "x"}, c)).text and not asked


@pytest.mark.parametrize("requested", ["/", "..", "../..", "~", "/etc", "projet/../../"])
async def test_claude_code_cwd_stays_in_allowed_dirs(tmp_path, requested):
    work = tmp_path / "projet"
    work.mkdir()
    action = ClaudeCode(claude_bin=fake_claude(tmp_path), cwd=str(work), confirm=False)
    c, _ = ctx()
    r = await action.run({"task": "x", "cwd": requested}, c)
    assert r.is_error and ("non autorisé" in r.text or "introuvable" in r.text)
    assert not (tmp_path / "args.json").exists()


async def test_claude_code_relative_cwd_and_allowed_dirs(tmp_path):
    (tmp_path / "a" / "sous").mkdir(parents=True)
    (tmp_path / "b").mkdir()
    action = ClaudeCode(claude_bin=fake_claude(tmp_path), cwd=str(tmp_path / "a"), confirm=False,
                        allowed_dirs=[str(tmp_path / "a"), str(tmp_path / "b")])
    c, _ = ctx()
    assert str(tmp_path / "a" / "sous") in (await action.run({"task": "x", "cwd": "sous"}, c)).text
    assert not (await action.run({"task": "x", "cwd": str(tmp_path / "b")}, c)).is_error


def test_claude_code_permission_modes():
    with pytest.raises(ValueError):
        ClaudeCode(permission_mode="toutPermis")
    assert ClaudeCode(permission_mode="bypassPermissions", confirm=False).confirm is True
    assert ClaudeCode(permission_mode="plan", confirm=False).confirm is False


@pytest.mark.parametrize("output,message", [("[1, 2]", "a échoué"), ("x" * 5000, "trop volumineuse")])
async def test_claude_code_bad_output(tmp_path, output, message):
    action = ClaudeCode(claude_bin=fake_claude(tmp_path, output), cwd=str(tmp_path), confirm=False, max_output_bytes=1000)
    c, _ = ctx()
    r = await action.run({"task": "x"}, c)
    assert r.is_error and message in r.text


class RecordingTTS(TTSEngine):
    def __init__(self, fail=False):
        self.said, self.fail = [], fail

    async def speak(self, text):
        if self.fail:
            raise RuntimeError("synthèse en panne")
        self.said.append(text)


async def test_claude_code_background_waits_for_pipeline_and_survives_errors(tmp_path):
    busy = asyncio.Lock()
    tts = RecordingTTS()
    action = ClaudeCode(claude_bin=fake_claude(tmp_path), cwd=str(tmp_path), confirm=False)
    c, _ = ctx()
    c.services = {"tts": tts, "busy": busy}
    async with busy:   # un échange est en cours
        r = await action.run({"task": "x", "background": True}, c)
        assert "arrière-plan" in r.text
        await asyncio.sleep(0.5)
        assert tts.said == []   # ne parle pas par-dessus l'échange
    await asyncio.wait_for(asyncio.gather(*action._tasks), 5)
    assert tts.said and tts.said[0].startswith("Claude Code a terminé.")
    # une synthèse en panne n'est pas une exception orpheline
    c.services = {"tts": RecordingTTS(fail=True)}
    await action.run({"task": "x", "background": True}, c)
    results = await asyncio.wait_for(asyncio.gather(*action._tasks, return_exceptions=True), 5)
    assert all(res is None for res in results)
