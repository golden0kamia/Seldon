"""Bus d'événements minimal pour brancher des extensions sans toucher au pipeline.

Événements émis par le pipeline :
    utterance        (utterance: Utterance)
    overheard        (utterance: Utterance)       phrase entendue sans mot d'éveil, non traitée
    command          (name: str, args: str)
    acknowledge      (text: str)
    action.start     (name: str, params: dict)    params peut contenir des données sensibles
    action.end       (name: str, result: ActionResult)
    response         (text: str)
    error            (error: Exception, stage: str)   stage : brain, speakers, memory, pipeline

Les handlers sont appelés l'un après l'autre et attendus : un handler lent retarde le
pipeline ; une exception est journalisée sans interrompre les suivants.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Awaitable, Callable

log = logging.getLogger(__name__)

Handler = Callable[..., Awaitable[None] | None]


class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = defaultdict(list)

    def on(self, event: str, handler: Handler) -> None:
        self._handlers[event].append(handler)

    async def emit(self, event: str, **payload: Any) -> None:
        for handler in list(self._handlers[event]):
            try:
                result = handler(**payload)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                log.exception("Erreur dans un handler de l'événement %s", event)
