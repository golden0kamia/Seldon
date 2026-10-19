"""Découverte des plugins par analyse statique (AST), sans les importer.

Importer un plugin peut charger torch, whisper ou opencv : l'interface lit donc
le code source pour trouver les classes ``@register("catégorie", "nom")`` et les
paramètres de leur ``__init__`` (valeur par défaut, annotation, commentaire de
fin de ligne). Ces informations alimentent les listes de moteurs et d'options.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MISSING = object()


@dataclass
class Param:
    name: str
    default: Any = MISSING      # valeur Python (littéral) ou MISSING si absente / non littérale
    default_src: str = ""       # texte source de la valeur par défaut
    annotation: str = ""
    comment: str = ""

    def hint(self) -> str:
        parts = []
        if self.annotation:
            parts.append(self.annotation)
        if self.default_src:
            parts.append(f"défaut : {self.default_src}")
        if self.comment:
            parts.append(self.comment)
        return " — ".join(parts)


@dataclass
class PluginInfo:
    category: str
    name: str
    path: Path
    doc: str = ""
    params: dict[str, Param] = field(default_factory=dict)


def _register_args(deco: ast.expr) -> tuple[str, str] | None:
    if not isinstance(deco, ast.Call) or len(deco.args) != 2:
        return None
    func = deco.func
    fname = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
    if fname != "register":
        return None
    a, b = deco.args
    if isinstance(a, ast.Constant) and isinstance(b, ast.Constant) and isinstance(a.value, str) and isinstance(b.value, str):
        return a.value, b.value
    return None


def _line_comment(lines: list[str], lineno: int) -> str:
    line = lines[lineno - 1] if 0 < lineno <= len(lines) else ""
    # pas de gestion fine des « # » dans les chaînes : suffisant pour des signatures
    return line.split("#", 1)[1].strip() if "#" in line else ""


def _params(init: ast.FunctionDef, lines: list[str]) -> dict[str, Param]:
    args = init.args
    positional = args.posonlyargs + args.args
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    pairs = list(zip(positional, defaults)) + list(zip(args.kwonlyargs, args.kw_defaults))
    out: dict[str, Param] = {}
    for arg, default in pairs:
        if arg.arg in ("self", "cls"):
            continue
        p = Param(arg.arg, annotation=ast.unparse(arg.annotation) if arg.annotation else "",
                  comment=_line_comment(lines, arg.lineno))
        if default is not None:
            p.default_src = ast.unparse(default)
            try:
                p.default = ast.literal_eval(default)
            except ValueError:
                pass
        out[arg.arg] = p
    return out


def scan_file(path: Path) -> list[PluginInfo]:
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError, UnicodeDecodeError):
        return []
    lines = source.splitlines()
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for deco in node.decorator_list:
            reg = _register_args(deco)
            if reg is None:
                continue
            doc = (ast.get_docstring(node) or ast.get_docstring(tree) or "").strip()
            init = next((n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
            found.append(PluginInfo(reg[0], reg[1], path, doc.split("\n\n")[0].replace("\n", " ").replace("``", ""),
                                    _params(init, lines) if init else {}))
    return found


def discover(package_dir: str | Path | None = None) -> dict[str, dict[str, PluginInfo]]:
    """Plugins du paquet seldon : ``{catégorie: {nom: PluginInfo}}``."""
    root = Path(package_dir) if package_dir else Path(__file__).resolve().parent.parent
    out: dict[str, dict[str, PluginInfo]] = {}
    for path in sorted(root.rglob("*.py")):
        if "gui" in path.relative_to(root).parts:
            continue
        for info in scan_file(path):
            out.setdefault(info.category, {})[info.name] = info
    return out


def class_params(path: str | Path, class_name: str) -> dict[str, Param]:
    """Paramètres de ``__init__`` d'une classe quelconque (ex. ConversationMemory)."""
    try:
        source = Path(path).read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError, UnicodeDecodeError):
        return {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            init = next((n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
            return _params(init, source.splitlines()) if init else {}
    return {}


_IDENT = r"[A-Za-z_][\w.\-]*"


def choices_from_comment(comment: str) -> list[str]:
    """Valeurs proposées par un commentaire : « a | b | c », « a / b / c » ou « "a" ou "b" »."""
    quoted = re.findall(r'"([^"\s]+)"', comment)
    if len(quoted) >= 2:
        return quoted
    for sep in (r"\|", "/"):
        m = re.search(rf"{_IDENT}(?:\s*{sep}\s*{_IDENT})+", comment)
        if m:
            return [s.strip() for s in re.split(sep, m.group(0))]
    return []
