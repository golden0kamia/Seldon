"""Lecture / écriture de config.yaml en conservant commentaires et mise en forme.

``ruamel.yaml`` (aller-retour) garde les commentaires, l'ordre des clés et les
blocs commentés : l'interface modifie le document en place puis le réécrit.
"""

from __future__ import annotations

import copy
import io
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

STAGES = ("stt", "speakers", "brain", "tts")


def _yaml() -> YAML:
    y = YAML()
    y.preserve_quotes = True
    y.indent(mapping=2, sequence=4, offset=2)
    y.width = 4096
    y.representer.add_representer(type(None), lambda r, _: r.represent_scalar("tag:yaml.org,2002:null", "null"))
    return y


def loads(text: str) -> CommentedMap:
    doc = _yaml().load(text)
    return doc if isinstance(doc, CommentedMap) else CommentedMap()


def dumps(doc: Any) -> str:
    buf = io.StringIO()
    _yaml().dump(doc, buf)
    return buf.getvalue()


def load(path: str | Path) -> CommentedMap:
    p = Path(path)
    return loads(p.read_text(encoding="utf-8")) if p.exists() else CommentedMap()


def to_text(doc: CommentedMap) -> str:
    """Texte YAML du document, sections normalisées remises sous leur forme courte."""
    doc = copy.deepcopy(doc)
    for key in STAGES:
        compact_stage(doc, key)
    if isinstance(doc.get("actions"), CommentedSeq):
        compact_actions(doc["actions"])
    return dumps(doc)


def save(doc: CommentedMap, path: str | Path) -> str:
    text = to_text(doc)
    Path(path).write_text(text, encoding="utf-8")
    return text


def comment(container: Any, key: Any) -> str:
    """Commentaire de fin de ligne associé à une clé (ou un index de liste)."""
    try:
        tokens = container.ca.items.get(key)
    except AttributeError:
        return ""
    if not tokens:
        return ""
    texts = []
    for tok in tokens:
        if tok is None:
            continue
        for t in tok if isinstance(tok, list) else [tok]:
            if t is not None and hasattr(t, "value"):
                texts.append(t.value)
    # seul le commentaire sur la même ligne compte (pas les blocs commentés qui suivent)
    first = "".join(texts).split("\n")[0].strip()
    return first.lstrip("#").strip()


def stage(doc: CommentedMap, key: str) -> CommentedMap:
    """Section ``{engine, options}`` normalisée (une chaîne seule devient un dictionnaire)."""
    raw = doc.get(key)
    if not isinstance(raw, CommentedMap):
        node = CommentedMap(engine=raw if isinstance(raw, str) else "")
        doc[key] = node
        raw = node
    if not isinstance(raw.get("options"), CommentedMap):
        raw["options"] = CommentedMap(raw.get("options") or {})
    return raw


def action_entry(actions: CommentedSeq, index: int) -> CommentedMap:
    """Action ``index`` sous forme ``{engine, options}`` (convertit ``- speak``)."""
    item = actions[index]
    if not isinstance(item, CommentedMap):
        item = CommentedMap(engine=str(item))
        actions[index] = item
    if not isinstance(item.get("options"), CommentedMap):
        item["options"] = CommentedMap(item.get("options") or {})
    return item


def compact_actions(actions: CommentedSeq) -> None:
    """Réécrit ``{engine: x}`` sans options en ``- x`` pour garder le fichier lisible."""
    for i, item in enumerate(actions):
        if isinstance(item, CommentedMap):
            opts = item.get("options")
            if set(item) <= {"engine", "options"} and not opts:
                actions[i] = item.get("engine", "")


def compact_stage(doc: CommentedMap, key: str) -> None:
    node = doc.get(key)
    if isinstance(node, CommentedMap) and "options" in node and not node["options"]:
        del node["options"]
