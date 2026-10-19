"""Action « terminal » : exécute une commande shell sur la machine de l'utilisateur.

Les commandes viennent de la voix puis d'un LLM : la sécurité prime.

- ``denylist`` : motifs regex toujours refusés (rm -rf /, mkfs, dd vers un disque,
  extinction, fork bomb, curl | sh...), quel que soit le mode ;
- ``mode`` : ``confirm`` (défaut, validation humaine avant chaque commande),
  ``allowlist`` (exécution directe des commandes en lecture seule : programmes de
  ``allowlist`` sans option ni sous-commande qui écrit, validation sinon) ou ``auto`` (tout s'exécute, à vos risques) ;
- ``allowed_speakers`` : liste des locuteurs autorisés (``None`` = tout le monde).

Exemple ``config.yaml`` :

    actions:
      - engine: terminal
        options:
          mode: allowlist
          allowed_speakers: [bastien]
          timeout: 30
"""

from __future__ import annotations

import asyncio
import os
import re
import shlex
import signal
from pathlib import Path
from typing import Any

from seldon.actions.base import Action
from seldon.core.registry import register
from seldon.core.types import ActionContext, ActionResult

MODES = ("confirm", "allowlist", "auto")

DEFAULT_DENYLIST: list[str] = [
    # rm récursif sur la racine, le home ou un répertoire système de premier niveau
    r"\brm\s+(?:-\S+\s+)*-\S*[rR]\S*\s+(?:-\S+\s+)*(?:/|/\*|~/?|~/\*|\$HOME/?|/(?:bin|boot|dev|etc|home|lib\w*|opt|proc|root|sbin|srv|sys|usr|var)/?)(?:\s|;|&|\||$)",
    r"--no-preserve-root",
    r"\bmkfs(?:\.\w+)?\b",
    r"\b(?:wipefs|fdisk|sfdisk|cfdisk|parted|gdisk|sgdisk)\b",
    r"\bdd\b.*\bof=/dev/(?!null\b)",
    r">\s*/dev/(?:sd|hd|vd|xvd|nvme|mmcblk|dm-)",
    r"\bshred\b.*\s/dev/",
    r"(?:^|[\s;&|(/`])(?:shutdown|reboot|poweroff|halt)(?=$|[\s;&|)`])",
    r"\binit\s+[06]\b",
    r"\bsystemctl\s+(?:poweroff|reboot|halt|kexec|suspend|hibernate|hybrid-sleep)\b",
    r":\s*\(\s*\)\s*\{.*:\s*\|\s*:.*\}",                      # fork bomb
    r"\b(?:curl|wget|fetch)\b[^|]*\|\s*(?:sudo\s+)?(?:\w*sh|python\d?|perl|ruby)\b",
    r"\b(?:ba|z|da)?sh\s+<\s*\(\s*(?:curl|wget)\b",           # sh <(curl ...)
    r"\bchmod\s+(?:-\S+\s+)*[0-7]*777\s+/(?:\s|$)",
    r"\bchown\s+(?:-\S+\s+)*-\S*R\S*\s+\S+\s+/(?:\s|$)",
    r">\s*/etc/(?:passwd|shadow|sudoers|fstab)\b",
    r"\bmv\s+(?:-\S+\s+)*(?:/|~|\$HOME)\s",                   # déplacer / ou ~
]

# Programmes en lecture seule : exécutés sans confirmation en mode ``allowlist``,
# sous réserve des règles d'arguments de ``_ARG_RULES``.
DEFAULT_ALLOWLIST: list[str] = [
    "ls", "pwd", "cd", "echo", "printf", "cat", "head", "tail", "wc", "grep", "find", "which",
    "whoami", "date", "cal", "uptime", "uname", "hostname", "df", "du", "free", "ps", "id",
    "lsblk", "lsusb", "lspci", "lscpu", "lsmod", "ip", "ss", "nproc", "sensors", "file",
    "stat", "sort", "uniq", "cut", "tr", "tree", "basename", "dirname", "realpath",
    "readlink", "env", "printenv", "jq", "diff", "md5sum", "sha256sum", "git", "systemctl",
    "journalctl", "true",
]

_FIND_WRITES = {"-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint", "-fprint0", "-fprintf", "-fls"}
_GIT_READ = {"status", "log", "diff", "show", "branch", "remote", "rev-parse", "ls-files",
             "blame", "describe", "shortlog", "tag"}
_SYSTEMCTL_READ = {"status", "show", "cat", "is-active", "is-enabled", "is-failed",
                   "list-units", "list-unit-files", "list-timers", "list-sockets"}
_JOURNALCTL_WRITES = ("--vacuum", "--rotate", "--flush", "--sync", "--relinquish",
                      "--setup-keys", "--update-catalog")


def _positional(args: list[str]) -> list[str]:
    return [a for a in args if not a.startswith("-")]


# Règles par programme : vrai si ces arguments ne modifient rien.
_ARG_RULES: dict[str, Any] = {
    "find": lambda a: not _FIND_WRITES & set(a),
    "sort": lambda a: not any(x.startswith("--output") or (x[:1] == "-" and x[1:2] != "-" and "o" in x)
                              for x in a),
    "uniq": lambda a: len(_positional(a)) <= 1,                       # uniq IN OUT écrit OUT
    "date": lambda a: not any(x.startswith(("-s", "--set")) for x in a),
    "hostname": lambda a: not _positional(a),                         # hostname NOM le change
    "env": lambda a: not _positional(a),                              # env CMD lancerait CMD
    "ip": lambda a: not {"add", "del", "delete", "set", "change", "replace", "flush",
                         "append", "prepend", "exec"} & set(a),
    "git": lambda a: bool(_positional(a)) and _positional(a)[0] in _GIT_READ
    and not any(x.startswith("--output") for x in a)
    and not (_positional(a)[0] in {"branch", "tag", "remote"} and len(_positional(a)) > 1),
    "systemctl": lambda a: not _positional(a) or _positional(a)[0] in _SYSTEMCTL_READ,
    "journalctl": lambda a: not any(x.startswith(_JOURNALCTL_WRITES) for x in a),
}

# Séparateurs de commandes du shell (pour trouver chaque programme lancé).
_SPLIT = re.compile(r"\|\||&&|[;|&\n]")
# Redirections sans effet sur les fichiers (2>&1, >/dev/null…), retirées avant analyse.
_HARMLESS_REDIRECT = re.compile(r"\d*>&\d+|&?\d*>>?\s*/dev/null\b")
# Constructions qui rendent l'analyse statique peu fiable -> confirmation obligatoire.
_UNSAFE_SYNTAX = re.compile(r"`|\$\(|<\(|>\(|>|\beval\b|\bexec\b|\bsudo\b")


def _commands(command: str) -> list[list[str]] | None:
    """Commandes simples (programme + arguments) ; ``None`` si l'analyse échoue."""
    commands = []
    for segment in _SPLIT.split(command):
        try:
            tokens = shlex.split(segment)
        except ValueError:
            return None
        while tokens and re.match(r"^\w+=", tokens[0]):  # VAR=val cmd
            tokens.pop(0)
        if tokens:
            commands.append([os.path.basename(tokens[0]), *tokens[1:]])
    return commands


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars * 2 // 3
    tail = max_chars - head
    cut = len(text) - head - tail
    return f"{text[:head]}\n[... {cut} caractères tronqués ...]\n{text[-tail:]}"


@register("actions", "terminal")
class Terminal(Action):
    name = "run_shell_command"
    description = (
        "Exécute une commande shell (bash) sur l'ordinateur Linux de l'utilisateur et "
        "renvoie le code de sortie et la sortie (stdout + stderr, éventuellement tronquée). "
        "La commande est non interactive : pas d'entrée standard, pas de terminal, pas "
        "d'éditeur ni de pager, et elle est interrompue après un délai. "
        "Privilégie les commandes en lecture seule (ls, cat, grep, df, ps...) et des "
        "options qui limitent la sortie (head, -n, --no-pager). Avant toute commande qui "
        "modifie ou supprime quelque chose, explique ce que tu vas faire. L'utilisateur "
        "peut devoir valider la commande ; certaines commandes dangereuses sont refusées "
        "d'office. En cas de refus, explique-le à l'utilisateur au lieu de contourner."
    )

    def __init__(
        self,
        mode: str = "confirm",
        allowlist: list[str] | None = None,
        denylist: list[str] | None = None,
        extra_denylist: list[str] | None = None,
        allowed_speakers: list[str] | None = None,
        timeout: float = 30.0,
        max_timeout: float = 300.0,
        cwd: str | None = None,
        max_output_chars: int = 8000,
        shell: str = "/bin/bash",
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode doit être l'un de {MODES}, reçu {mode!r}")
        self.mode = mode
        self.allowlist = set(DEFAULT_ALLOWLIST if allowlist is None else allowlist)
        patterns = (DEFAULT_DENYLIST if denylist is None else denylist) + (extra_denylist or [])
        self.denylist = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in patterns]
        self.allowed_speakers = None if allowed_speakers is None else set(allowed_speakers)
        self.timeout = timeout
        self.max_timeout = max_timeout
        self.cwd = Path(cwd or "~").expanduser()
        self.max_output_chars = max_output_chars
        self.shell = shell

    @property
    def input_schema(self) -> dict[str, Any]:  # type: ignore[override]
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Commande bash à exécuter."},
                "cwd": {
                    "type": "string",
                    "description": f"Répertoire de travail (défaut : {self.cwd}).",
                },
                "timeout": {
                    "type": "number",
                    "description": f"Délai max en secondes (défaut {self.timeout:g}, "
                    f"plafond {self.max_timeout:g}).",
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        }

    def check(self, command: str) -> str | None:
        """Motif de refus si la commande est interdite, sinon ``None``."""
        for pattern in self.denylist:
            if pattern.search(command):
                return f"Commande refusée par la liste d'interdiction (motif : {pattern.pattern})."
        return None

    def is_read_only(self, command: str) -> bool:
        """Vrai si la commande ne lance que des programmes autorisés, sans écriture."""
        command = _HARMLESS_REDIRECT.sub(" ", command)
        if _UNSAFE_SYNTAX.search(command):
            return False
        commands = _commands(command)
        return bool(commands) and all(
            prog in self.allowlist and _ARG_RULES.get(prog, lambda a: True)(args)
            for prog, *args in commands
        )

    def needs_confirmation(self, command: str) -> bool:
        if self.mode == "auto":
            return False
        return self.mode == "confirm" or not self.is_read_only(command)

    async def run(self, params: dict[str, Any], ctx: ActionContext) -> ActionResult:
        command = str(params.get("command", "")).strip()
        if not command:
            return ActionResult.error("Commande vide.")

        speaker = ctx.utterance.speaker if ctx.utterance else None
        if self.allowed_speakers is not None and speaker not in self.allowed_speakers:
            who = speaker or "un locuteur non identifié"
            return ActionResult.error(
                f"Refusé : {who} n'est pas autorisé à exécuter des commandes sur cette machine."
            )
        if reason := self.check(command):
            return ActionResult.error(reason)

        cwd = Path(params.get("cwd") or self.cwd).expanduser()
        if not cwd.is_dir():
            return ActionResult.error(f"Répertoire introuvable : {cwd}")
        timeout = min(float(params.get("timeout") or self.timeout), self.max_timeout)

        if self.needs_confirmation(command):
            question = f"Exécuter la commande « {command} » dans {cwd} ?"
            if not await ctx.confirm(question):
                return ActionResult.error("L'utilisateur a refusé l'exécution de la commande.")

        return await self._execute(command, cwd, timeout)

    async def _execute(self, command: str, cwd: Path, timeout: float) -> ActionResult:
        env = {**os.environ, "PAGER": "cat", "GIT_PAGER": "cat", "TERM": "dumb"}
        proc = await asyncio.create_subprocess_exec(
            self.shell, "-c", command,
            cwd=cwd, env=env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,  # groupe de processus dédié, tué d'un bloc
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout)
        except (TimeoutError, asyncio.CancelledError) as e:
            self._kill(proc)
            await proc.wait()
            if isinstance(e, asyncio.CancelledError):
                raise
            return ActionResult.error(f"Délai dépassé ({timeout:g} s) : commande interrompue.")

        output = _truncate(out.decode(errors="replace").strip(), self.max_output_chars)
        return ActionResult(text=f"Code de sortie : {proc.returncode}\n{output or '(aucune sortie)'}")

    @staticmethod
    def _kill(proc: asyncio.subprocess.Process) -> None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
