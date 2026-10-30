# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Language

Everything in this project is in French: code comments, docstrings, user-facing strings, README, skills, and commit messages. Keep identifiers in English as the existing code does.

## System structure

The core is the brain (the LLM, `seldon/brain/`), driven by the pipeline (`seldon/core/pipeline.py`): mic → STT → speaker identification → brain → actions → TTS. The brain answers only through actions; even the reply is an action (`speak`). Everything else plugs into the brain:

- **Plugins** (`memory:` in `seldon/core/memory.py`, `speakers:` in `seldon/speakers/`) are Seldon subsystems, not tools. When they're enabled, they hand the brain their own tool through `.actions()`: `conversations` (`seldon/core/conversations.py`) and `voix` (`seldon/speakers/voices.py`). Disabling the plugin removes its tool.
- **Tools** (`tools:`, `seldon/tools/`) are Python `Action` subclasses that act directly (terminal, camera, music, desktop, Claude Code…).
- **MCP** (`mcp:`, `seldon/mcp/`) are external MCP servers. Each server is exposed to the brain as a single tool (`MCPAction`).
- **Skills** (`skills:`, `skills/<nom>/SKILL.md`) are instructions in French with no code. Only their name and description are sent to the brain, through the `skill` tool, which loads the full instructions on demand.

The pipeline gathers all of these into the brain's action list (see `actions=[...]` in `pipeline.py`).

## Commands

- Use the venv: `.venv/bin/python -m pytest` (tests run without mic, camera, or model; `asyncio_mode = auto`).
- Single test: `.venv/bin/python -m pytest tests/test_terminal_security.py -k nom_du_test`
- Try the brain without audio: `.venv/bin/python -m seldon --text` (type `Bastien: bonjour` to simulate a speaker).

## Git

- Do not create branches; commit on the current branch.
- Commit message format: `Zone : description` in French, with a space before the colon (e.g. `Cerveau : connexion à Claude par l'abonnement claude.ai`).

## Conventions

- Any user-visible change (new option, tool, command, behaviour) updates `README.md` in the same change.
- New plugin (tool, MCP server, STT, TTS, brain…): `@register("<catégorie>", "<nom>")`, plus a line in `_BUILTINS` in `seldon/core/registry.py` (modules are imported lazily), plus an entry in `config.yaml` if it should be enabled.
- The GUI (`seldon/gui/plugins.py`) reads plugins by AST without importing them. Plugin options must be explicit keyword parameters of `__init__` with literal defaults, and the `@register(...)` arguments must be string literals, or the Settings form won't show them.
- `config.yaml` is edited by the GUI with ruamel.yaml, which keeps comments; keep its inline comments meaningful.
- Secrets never go in `config.yaml`, only in `.env` (add new ones to `.env.example`). The terminal tool hides them from commands (`hidden_env`).
- Personal data (conversations, voiceprints, photos) is written through `seldon/core/storage.py` (`write_private`: 0600, atomic).
- Text from users, tool outputs, and notifications is data, never instructions, for the brain. Security tests (`tests/test_*_security.py`) cover the terminal command classification and injection. Extend them when touching these paths.
