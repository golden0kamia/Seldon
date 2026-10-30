"""Non-régression : contournements de la classification des commandes terminal."""

from __future__ import annotations

import os

import pytest

from seldon.core.types import ActionContext, Utterance
from seldon.tools.terminal import BENIGN, SEVERE, STANDARD, Terminal


def make_ctx(answer: bool = True, speaker: str | None = "alice") -> tuple[ActionContext, list[str]]:
    asked: list[str] = []

    async def confirm(question: str) -> bool:
        asked.append(question)
        return answer

    return ActionContext(utterance=Utterance(text="...", speaker=speaker), confirm=confirm), asked


@pytest.mark.parametrize("cmd", [
    "env -S'touch x'",                       # env lance une commande cachée dans une option
    "GIT_EXTERNAL_DIFF=x git diff",          # variable qui fait lancer un programme
    "PATH=/tmp/x ls",
    "FOO=1 date",
    "./ls", "/tmp/x/ls -la",                 # programme hors des dossiers système
    "sort --compress-program=x data.txt",
    "tree -o sortie.txt", "tree -R",
    "ss -K dst 1.2.3.4",
    "ip -batch cmds.txt", "ip a a 10.0.0.1/24 dev eth0",
    "hostname -F nom.txt",
    "date -us 2020-01-01",
    "file -C -m magic",
    "git -c core.pager=x log", "git --exec-path=/tmp log",
    "git branch --set-upstream-to=origin/x", "git branch -m nouveau", "git tag -d v1",
    "journalctl --cursor-file=c.txt",
    "sort {-o,out} data.txt",                # expansion d'accolades
    "echo x >/dev/null.txt",
    "cat ~/.ssh/id_ed25519", "cat .env", "grep KEY projet/.env.local", "cat /proc/1/environ",
    "ps auxe", "ps eww",
])
def test_not_benign(cmd):
    assert Terminal(mode="allowlist").classify(cmd) != BENIGN


@pytest.mark.parametrize("cmd", [
    "ls -la", "df -h", "git status", "git log --oneline -n 5", "git --no-pager diff", "git branch -vv",
    "git tag -l 'v*'", "grep -rn foo . | head -n 20", "journalctl -u ollama -n 50 --no-pager",
    "ip -br a", "ip route get 1.1.1.1", "ss -tlnp", "ps aux | grep python", "ps -u steve",
    "LC_ALL=C sort data.txt", "tree -L 2", "find . -name '*.{log,txt}'", "cat a 2>/dev/null",
    "env", "printenv HOME",
])
def test_still_benign(cmd):
    assert Terminal(mode="allowlist").classify(cmd) == BENIGN


@pytest.mark.parametrize("cmd", [
    # effet illisible : code en ligne, enveloppes, noms de programme calculés
    "bash -c 'x'", "sh script.sh", "python3 -c 'print(1)'", "perl -ne 'print'", "node -e 1",
    "echo x | python3", "xargs touch < liste", "nohup make", "timeout 5 make", "env FOO=1 make",
    "find . -exec touch {} +", "awk '{system(\"x\")}' f", "sed 's/a/b/e' f", "sed -n '1e date' f",
    "$X -rf build", "/???/r? x", "{rm,-rf,x}", "$'\\x72\\x6d' x", "f() { x; }; f", "eval x",
    "git config alias.co '!x'", "export PATH=/tmp", "LD_PRELOAD=x.so make",
    # nom masqué par des guillemets ou un échappement : bash lit « rm »
    '"r"m fichier', "r\\m fichier", "git \"push\" --force",
    # écrire dans un fichier lancé automatiquement, ou toucher un secret
    "echo x >> ~/.bashrc", "cp cle ~/.ssh/authorized_keys", "cat ~/.ssh/id_rsa | head",
    "tar xf a.tar --to-command=x", "ssh -o ProxyCommand=x h",
])
def test_severe(cmd):
    assert Terminal(mode="allowlist").classify(cmd) == SEVERE


@pytest.mark.parametrize("cmd", [
    "mkdir build", "make", "python3 script.py", "python3 -m pytest", "awk '{print $1}' f",
    "sed -i 's/a/b/' f", "git commit -m x", "git config user.name Bob", "docker run --rm image",
    "FOO=1 make", "if true; then make; fi", "LC_ALL=C git commit -m x", "(cd /tmp && ls)",
])
def test_standard(cmd):
    assert Terminal(mode="allowlist").classify(cmd) == STANDARD


def test_denylist_sees_through_quotes():
    assert Terminal().check('"r"m -rf /') is not None
    assert Terminal().check("x" * 20_000) is not None
    assert Terminal().check("ls\0") is not None


def test_admin_cannot_skip_confirmation_with_obfuscation():
    t = Terminal(mode="allowlist", admins=["Bastien"])
    assert t.needs_confirmation('"r"m -rf build', "Bastien")
    assert t.needs_confirmation("bash -c 'x'", "Bastien")
    assert not t.needs_confirmation("make", "Bastien")


async def test_allowed_speakers_ignores_case():
    ctx, _ = make_ctx(speaker="ALICE")
    r = await Terminal(mode="auto", allowed_speakers=["alice"]).run({"command": "mkdir -p x", "cwd": "/tmp"}, ctx)
    assert not r.is_error


@pytest.mark.parametrize("timeout", ["abc", [1]])
async def test_bad_timeout(timeout):
    ctx, _ = make_ctx()
    r = await Terminal(mode="auto").run({"command": "echo x", "timeout": timeout}, ctx)
    assert r.is_error and "Délai invalide" in r.text


@pytest.mark.parametrize("timeout", [-5, float("nan"), float("inf")])
async def test_odd_timeout_uses_default(timeout):
    ctx, _ = make_ctx()
    r = await Terminal(mode="auto", max_timeout=2).run({"command": "echo ok", "timeout": timeout}, ctx)
    assert not r.is_error and "ok" in r.text


async def test_endless_output_is_cut():
    ctx, _ = make_ctx()
    t = Terminal(mode="auto", max_output_chars=200, max_read_bytes=2_000_000)
    r = await t.run({"command": "yes", "timeout": 20}, ctx)
    assert r.is_error and "trop volumineuse" in r.text and len(r.text) < 600


async def test_timeout_keeps_partial_output():
    ctx, _ = make_ctx()
    r = await Terminal(mode="auto").run({"command": "echo début; sleep 30", "timeout": 0.5}, ctx)
    assert r.is_error and "Délai" in r.text and "début" in r.text


async def test_secrets_hidden_from_commands(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-secret")
    monkeypatch.setenv("BASH_ENV", "/tmp/piege")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/run/agent")
    ctx, asked = make_ctx()
    r = await Terminal(mode="allowlist").run({"command": "env"}, ctx)
    assert asked == [] and "sk-secret" not in r.text and "ghp-secret" not in r.text and "piege" not in r.text
    assert "/run/agent" in r.text   # l'agent SSH reste utilisable (git push)
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-secret"   # Seldon garde ses variables


async def test_benign_git_runs_without_repo_hooks(monkeypatch):
    ctx, _ = make_ctx()
    r = await Terminal(mode="allowlist").run({"command": "env"}, ctx)
    assert "core.fsmonitor" in r.text and "core.hooksPath" in r.text
    r = await Terminal(mode="auto").run({"command": "mkdir -p x; env", "cwd": "/tmp"}, ctx)
    assert "core.hooksPath" not in r.text   # commandes validées : hooks de l'utilisateur intacts
