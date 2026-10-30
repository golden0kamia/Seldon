import pytest

from seldon.brain.claude import ClaudeBrain
from seldon.brain.claude_account import NO_KEY, _result_content, transcript
from seldon.core.types import ActionResult


def test_auth_inconnu_refuse():
    with pytest.raises(ValueError):
        ClaudeBrain(auth="compte")


def test_abonnement_sans_cle_api():
    brain = ClaudeBrain(auth="abonnement")
    assert brain.account is not None
    assert brain.account.options().env == NO_KEY   # Claude Code n'utilise pas la clé du .env


def test_transcript_tous_formats_et_limite():
    messages = [
        {"role": "user", "content": "[x]\nbonjour"},
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "…"}, {"type": "text", "text": "salut"}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1", "content": []}]},
        {"role": "assistant", "content": "dernier"},
    ]
    assert transcript(messages, 1000) == "Utilisateur : [x] bonjour\nSeldon : salut\nSeldon : dernier"
    assert transcript(messages, 20) == "Seldon : dernier"


def test_resultat_fin_de_tour():
    out = _result_content(ActionResult(text="Dit."), last=True)
    assert not out["is_error"] and "termine ton tour" in out["content"][-1]["text"]


def test_reset_puis_echanges_transmis():
    brain = ClaudeBrain(auth="abonnement")
    brain.set_state({"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]})
    assert brain.account.backlog == 2
    brain.append_exchange("c", "d")
    assert brain.account.backlog == 4
    brain.reset()
    assert brain.account.backlog == 0 and brain.messages == []
