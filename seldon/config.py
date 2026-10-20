"""Chargement de la configuration YAML.

Chaque étage est décrit par ``{engine: <nom>, options: {...}}`` ; les options
sont passées telles quelles au constructeur du plugin. Les actions sont une
liste du même format (``engine`` = nom de l'action dans le registre).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class StageConfig:
    engine: str
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, raw: Any) -> "StageConfig":
        if isinstance(raw, str):
            return cls(engine=raw)
        return cls(engine=raw["engine"], options=dict(raw.get("options") or {}))


@dataclass
class Config:
    stt: StageConfig
    speakers: StageConfig
    brain: StageConfig
    tts: StageConfig
    actions: list[StageConfig]
    memory: dict[str, Any] = field(default_factory=dict)   # options de ConversationMemory + enabled
    commands: bool = True       # commandes locales : exit, reset, aide…
    confirm: str = "console"   # "console" (demande au clavier), "always" ou "never"
    follow_up_s: float = 20     # avec mots d'éveil : écoute sans mot d'éveil après un échange
    log_level: str = "INFO"

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(
            stt=StageConfig.parse(raw.get("stt", "text")),
            speakers=StageConfig.parse(raw.get("speakers", "none")),
            brain=StageConfig.parse(raw.get("brain", "llamacpp")),
            tts=StageConfig.parse(raw.get("tts", "console")),
            actions=[StageConfig.parse(a) for a in raw.get("actions", ["speak"])],
            memory=dict(raw.get("memory") or {}),
            commands=raw.get("commands", True),
            confirm=raw.get("confirm", "console"),
            follow_up_s=raw.get("follow_up_s", 20),
            log_level=raw.get("log_level", "INFO"),
        )
