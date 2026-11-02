"""Lecture / écriture de config.yaml en conservant commentaires et mise en forme.

``ruamel.yaml`` (aller-retour) garde les commentaires, l'ordre des clés et les
blocs commentés : l'interface modifie le document en place puis le réécrit.
"""

from __future__ import annotations

import copy
import io
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

STAGES = ("stt", "speakers", "brain", "tts")
LISTS = ("tools", "mcp")   # listes de {engine, options}


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


def to_text(doc: CommentedMap) -> str:
    """Texte YAML du document, sections normalisées remises sous leur forme courte."""
    doc = copy.deepcopy(doc)
    for key in STAGES:
        compact_stage(doc, key)
    for key in LISTS:
        if isinstance(doc.get(key), CommentedSeq):
            compact_list(doc[key])
    return dumps(doc)


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


def drop_comment(container: Any, key: Any) -> None:
    """Retire le commentaire de fin de ligne d'une clé (les blocs commentés qui suivent restent)."""
    try:
        tokens = container.ca.items.get(key)
    except AttributeError:
        return
    for tok in tokens or ():
        for t in tok if isinstance(tok, list) else [tok]:
            if t is not None and str(getattr(t, "value", "")).startswith("#"):
                rest = t.value.split("\n", 1)[1] if "\n" in t.value else ""
                t.value = "\n" + rest
                return


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


def entry(items: CommentedSeq, index: int) -> CommentedMap:
    """Élément ``index`` d'une liste tools / mcp sous forme ``{engine, options}`` (convertit ``- speak``)."""
    item = items[index]
    if not isinstance(item, CommentedMap):
        item = CommentedMap(engine=str(item))
        items[index] = item
    if not isinstance(item.get("options"), CommentedMap):
        item["options"] = CommentedMap(item.get("options") or {})
    return item


def compact_list(items: CommentedSeq) -> None:
    """Réécrit ``{engine: x}`` sans options en ``- x`` pour garder le fichier lisible."""
    for i, item in enumerate(items):
        if isinstance(item, CommentedMap) and set(item) <= {"engine", "options"} and not item.get("options"):
            items[i] = item.get("engine", "")


def compact_stage(doc: CommentedMap, key: str) -> None:
    node = doc.get(key)
    if isinstance(node, CommentedMap) and "options" in node and not node["options"]:
        del node["options"]
    if isinstance(node, CommentedMap) and isinstance(node.get("backup"), CommentedMap):
        compact_stage(node, "backup")
