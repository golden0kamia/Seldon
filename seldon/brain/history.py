"""Historique borné des cerveaux.

Sans limite, l'historique grossit à chaque tour (sorties de commandes de 8000
caractères, images de la caméra en base64…) : chaque requête devient plus lente
et le contexte du modèle finit par déborder. Deux mécanismes :

- ``drop_old_images`` : seules les ``keep`` dernières captures restent dans
  l'historique, les plus anciennes deviennent une mention textuelle ;
- ``compact`` : au-delà de ``max_chars`` caractères ou ``max_messages``
  messages, les résultats d'outils et images des échanges passés (sauf le
  dernier) sont allégés, puis les plus anciens échanges oubliés, jusqu'à la
  moitié des limites.

L'historique n'est retouché qu'à ces moments-là, jamais à chaque tour : le
serveur (llama.cpp) peut ainsi réutiliser le cache du préfixe déjà calculé. Les
coupes se font au début d'un échange (message de l'utilisateur), si bien qu'un
appel d'outil n'est jamais séparé de son résultat.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

IMAGE_DROPPED = "(image retirée de l'historique)"
RESULT_CHARS = 1000   # longueur gardée des résultats d'outils des échanges passés
_SKIPPED = {"images", "image_url", "source", "signature", "data"}   # images, signatures : hors calcul


def shorten(text: str, max_chars: int) -> str:
    """Début et fin de ``text`` si trop long."""
    if len(text) <= max_chars:
        return text
    head = max_chars * 2 // 3
    tail = max_chars - head
    return f"{text[:head]}\n[… {len(text) - head - tail} caractères retirés de l'historique …]\n{text[-tail:]}"


def size(obj: Any) -> int:
    """Taille textuelle (caractères) d'un message ou d'une liste de messages, images exclues."""
    if isinstance(obj, str):
        return len(obj)
    if isinstance(obj, dict):
        return sum(size(v) for k, v in obj.items() if k not in _SKIPPED)
    if isinstance(obj, list):
        return sum(size(v) for v in obj)
    return 0


class OpenAIFormat:
    """Messages au format OpenAI (llama.cpp) : résultats ``role: tool``, images dans un message
    utilisateur (contenu en liste) qui suit le résultat."""

    def is_turn_start(self, m: dict[str, Any]) -> bool:
        return m.get("role") == "user" and isinstance(m.get("content"), str)

    def image_count(self, m: dict[str, Any]) -> int:
        content = m.get("content")
        return sum(1 for p in content if isinstance(p, dict) and p.get("type") == "image_url") \
            if isinstance(content, list) else 0

    def drop_images(self, m: dict[str, Any]) -> None:
        if self.image_count(m):
            m["content"] = [p for p in m["content"] if p.get("type") != "image_url"] + \
                [{"type": "text", "text": IMAGE_DROPPED}]

    def shorten_results(self, m: dict[str, Any], max_chars: int) -> None:
        if m.get("role") == "tool" and isinstance(m.get("content"), str):
            m["content"] = shorten(m["content"], max_chars)

    def after_cut(self, m: dict[str, Any]) -> None:
        pass


class OllamaFormat(OpenAIFormat):
    """Messages Ollama : les images sont attachées au résultat d'outil (clé ``images``)."""

    def image_count(self, m: dict[str, Any]) -> int:
        return len(m.get("images") or [])

    def drop_images(self, m: dict[str, Any]) -> None:
        if m.pop("images", None):
            m["content"] = f"{m.get('content') or ''}\n{IMAGE_DROPPED}".strip()


class AnthropicFormat:
    """Messages de l'API Claude : résultats d'outils (blocs ``tool_result``) dans les messages utilisateur."""

    @staticmethod
    def _results(m: dict[str, Any]) -> list[dict[str, Any]]:
        content = m.get("content")
        if m.get("role") != "user" or not isinstance(content, list):
            return []
        return [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]

    def is_turn_start(self, m: dict[str, Any]) -> bool:
        content = m.get("content")
        return m.get("role") == "user" and (
            isinstance(content, str) or any(isinstance(b, dict) and b.get("type") == "text" for b in content or [])
        )

    def image_count(self, m: dict[str, Any]) -> int:
        return sum(1 for r in self._results(m) if isinstance(r.get("content"), list)
                   for b in r["content"] if b.get("type") == "image")

    def drop_images(self, m: dict[str, Any]) -> None:
        for r in self._results(m):
            if isinstance(r.get("content"), list) and any(b.get("type") == "image" for b in r["content"]):
                r["content"] = [b for b in r["content"] if b.get("type") != "image"] + \
                    [{"type": "text", "text": IMAGE_DROPPED}]

    def shorten_results(self, m: dict[str, Any], max_chars: int) -> None:
        for r in self._results(m):
            content = r.get("content")
            if isinstance(content, str):
                r["content"] = shorten(content, max_chars)
            elif isinstance(content, list):
                for b in content:
                    if b.get("type") == "text" and isinstance(b.get("text"), str):
                        b["text"] = shorten(b["text"], max_chars)

    def after_cut(self, m: dict[str, Any]) -> None:
        # Les résultats d'outils en tête de ce message répondaient à un message oublié.
        if isinstance(m.get("content"), list):
            m["content"] = [b for b in m["content"] if not (isinstance(b, dict) and b.get("type") == "tool_result")]


OPENAI, OLLAMA, ANTHROPIC = OpenAIFormat(), OllamaFormat(), AnthropicFormat()


def drop_old_images(messages: list[dict[str, Any]], fmt: Any, keep: int) -> int:
    """Ne garde les images que des ``keep`` derniers messages qui en contiennent ; renvoie le nombre allégé."""
    seen = dropped = 0
    for m in reversed(messages):
        if fmt.image_count(m):
            if seen >= keep:
                fmt.drop_images(m)
                dropped += 1
            seen += 1
    return dropped


def compact(messages: list[dict[str, Any]], fmt: Any, max_chars: int, max_messages: int,
            result_chars: int = RESULT_CHARS) -> bool:
    """Allège ``messages`` (en place) s'il dépasse une des limites ; vrai si modifié."""
    if size(messages) <= max_chars and len(messages) <= max_messages:
        return False
    before = (len(messages), size(messages))
    starts = [i for i, m in enumerate(messages) if fmt.is_turn_start(m)]
    last = starts[-1] if starts else len(messages)
    for m in messages[:last]:   # le dernier échange reste intact : on peut y faire référence
        fmt.drop_images(m)
        fmt.shorten_results(m, result_chars)
    sizes = [size(m) for m in messages]
    total, cut = sum(sizes), 0
    for start in starts:
        if total <= max_chars // 2 and len(messages) - cut <= max_messages // 2:
            break
        if start > cut:
            total -= sum(sizes[cut:start])
            cut = start
    if cut:
        del messages[:cut]
        fmt.after_cut(messages[0])
    log.info("Historique allégé : %d → %d messages, %d → %d caractères",
             before[0], len(messages), before[1], size(messages))
    return True
