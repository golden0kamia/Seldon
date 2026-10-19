"""Action « claude_code » : délègue une tâche complexe à Claude Code (CLI ``claude``).

Claude Code tourne en mode non interactif (``claude -p``) dans un répertoire
donné : il peut lire, modifier du code et lancer des commandes selon le
``permission_mode`` choisi. Toute demande de permission non couverte par ce
mode est refusée automatiquement (``--permission-prompts none``), il ne
bloque donc jamais en attente d'une réponse.

Une tâche longue peut tourner en arrière-plan : Seldon annonce le résultat
à voix haute à la fin (événement ``claude_code.done`` émis également).

Exemple ``config.yaml`` :

    actions:
      - engine: claude_code
        options:
          cwd: ~/Documents
          permission_mode: acceptEdits
          confirm: true
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
from pathlib import Path
from typing import Any

from seldon.actions.base import Action
from seldon.core.registry import register
from seldon.core.types import ActionContext, ActionResult

log = logging.getLogger(__name__)

SPOKEN_SUMMARY = (
    "Tu es appelé par un assistant vocal. Termine toujours par un résumé de une à "
    "deux phrases en français, sans markdown ni code, adapté à la lecture à voix haute."
)


@register("actions", "claude_code")
class ClaudeCode(Action):
    name = "claude_code"
    description = (
        "Délègue une tâche complexe à Claude Code, un agent de programmation autonome qui "
        "travaille dans un répertoire de l'ordinateur : écrire ou modifier du code, déboguer, "
        "analyser un projet, faire une tâche en plusieurs étapes sur des fichiers. "
        "Pour une simple commande, utilise plutôt run_shell_command. Décris la tâche "
        "précisément et de façon autonome (Claude Code ne voit pas la conversation). "
        "Mets background=true pour une tâche longue : le résultat sera annoncé à "
        "l'utilisateur à la fin ; préviens-le alors que c'est lancé. Mets resume=true "
        "pour poursuivre la tâche précédente dans le même répertoire."
    )

    def __init__(
        self,
        claude_bin: str = "claude",
        cwd: str | None = None,
        permission_mode: str = "acceptEdits",   # acceptEdits | auto | plan | dontAsk | bypassPermissions
        allowed_tools: list[str] | None = None,
        model: str | None = None,
        max_budget_usd: float | None = None,
        timeout: float = 1800.0,
        confirm: bool = True,                    # validation humaine avant de lancer
        allowed_speakers: list[str] | None = None,
        max_result_chars: int = 4000,
        announce_chars: int = 400,               # longueur max annoncée à voix haute (arrière-plan)
    ) -> None:
        self.claude_bin = claude_bin
        self.cwd = Path(cwd or "~").expanduser()
        self.permission_mode = permission_mode
        self.allowed_tools = allowed_tools
        self.model = model
        self.max_budget_usd = max_budget_usd
        self.timeout = timeout
        self.confirm = confirm
        self.allowed_speakers = None if allowed_speakers is None else set(allowed_speakers)
        self.max_result_chars = max_result_chars
        self.announce_chars = announce_chars
        self._sessions: dict[Path, str] = {}       # dernier session_id par répertoire
        self._tasks: set[asyncio.Task] = set()

    input_schema = {
        "type": "object",
        "properties": {
            "task": {"type": "string", "description": "Description complète et autonome de la tâche."},
            "cwd": {"type": "string", "description": "Répertoire de travail (défaut : configuration)."},
            "resume": {"type": "boolean", "description": "Poursuivre la session précédente de ce répertoire."},
            "background": {"type": "boolean", "description": "Lancer en arrière-plan et annoncer le résultat à la fin."},
        },
        "required": ["task"],
        "additionalProperties": False,
    }

    async def stop(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    def _command(self, task: str, session: str | None) -> list[str]:
        cmd = [
            self.claude_bin, "-p", task,
            "--output-format", "json",
            "--permission-mode", self.permission_mode,
            "--permission-prompts", "none",
            "--append-system-prompt", SPOKEN_SUMMARY,
        ]
        if session:
            cmd += ["--resume", session]
        if self.model:
            cmd += ["--model", self.model]
        if self.max_budget_usd is not None:
            cmd += ["--max-budget-usd", str(self.max_budget_usd)]
        if self.allowed_tools:
            cmd += ["--allowed-tools", *self.allowed_tools]
        return cmd

    async def run(self, params: dict[str, Any], ctx: ActionContext) -> ActionResult:
        task = str(params.get("task", "")).strip()
        if not task:
            return ActionResult.error("Tâche vide.")
        speaker = ctx.utterance.speaker if ctx.utterance else None
        if self.allowed_speakers is not None and speaker not in self.allowed_speakers:
            return ActionResult.error(f"Refusé : {speaker or 'un locuteur non identifié'} n'est pas autorisé à utiliser Claude Code.")

        cwd = Path(params.get("cwd") or self.cwd).expanduser()
        if not cwd.is_dir():
            return ActionResult.error(f"Répertoire introuvable : {cwd}")
        session = self._sessions.get(cwd) if params.get("resume") else None

        if self.confirm and not await ctx.confirm(f"Lancer Claude Code dans {cwd} ? Tâche : « {task} »"):
            return ActionResult.error("L'utilisateur a refusé de lancer Claude Code.")

        if params.get("background"):
            job = asyncio.create_task(self._background(task, cwd, session, ctx))
            self._tasks.add(job)
            job.add_done_callback(self._tasks.discard)
            return ActionResult(text=f"Claude Code a démarré en arrière-plan dans {cwd}. Le résultat sera annoncé à la fin.")
        return await self._execute(task, cwd, session)

    async def _background(self, task: str, cwd: Path, session: str | None, ctx: ActionContext) -> None:
        result = await self._execute(task, cwd, session)
        events = ctx.services.get("events")
        if events is not None:
            await events.emit("claude_code.done", task=task, result=result)
        summary = result.text.split("\n\n", 1)[-1] if not result.is_error else result.text
        message = ("Claude Code a terminé. " if not result.is_error else "Claude Code a échoué. ") + summary
        tts, stt = ctx.services.get("tts"), ctx.services.get("stt")
        if tts is None:
            return
        if stt is not None:
            stt.pause()
        try:
            await tts.speak(message[: self.announce_chars])
        finally:
            if stt is not None:
                stt.resume()

    async def _execute(self, task: str, cwd: Path, session: str | None) -> ActionResult:
        try:
            proc = await asyncio.create_subprocess_exec(
                *self._command(task, session),
                cwd=cwd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except FileNotFoundError:
            return ActionResult.error(f"Commande « {self.claude_bin} » introuvable : Claude Code est-il installé ?")
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), self.timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError) as e:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await proc.wait()
            if isinstance(e, asyncio.CancelledError):
                raise
            return ActionResult.error(f"Claude Code interrompu après {self.timeout:g} s.")

        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            detail = (stderr or stdout).decode(errors="replace").strip()[-500:]
            return ActionResult.error(f"Claude Code a échoué (code {proc.returncode}) : {detail}")

        if data.get("session_id"):
            self._sessions[cwd] = data["session_id"]
        text = str(data.get("result") or "")
        if len(text) > self.max_result_chars:
            text = "[début tronqué]\n" + text[-self.max_result_chars:]
        info = [f"Répertoire : {cwd}", f"tours : {data.get('num_turns', '?')}"]
        if data.get("total_cost_usd") is not None:
            info.append(f"coût : {data['total_cost_usd']:.2f} $")
        denials = data.get("permission_denials") or []
        if denials:
            names = sorted({d.get("tool_name", "?") for d in denials})
            info.append(f"permissions refusées : {', '.join(names)}")
        return ActionResult(text=" | ".join(info) + "\n\n" + text, is_error=bool(data.get("is_error")))
