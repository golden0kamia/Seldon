"""Bus d'événements minimal pour brancher des extensions sans toucher au pipeline.

Événements émis par le pipeline :
    utterance        (utterance: Utterance)
    action.start     (name: str, params: dict)
    action.end       (name: str, result: ActionResult)
    response         (text: str)
    error            (error: Exception, stage: str)
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
