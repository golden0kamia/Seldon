"""Widgets de saisie partagés par les formulaires de paramètres."""

from __future__ import annotations

from typing import Any, Callable

import yaml
from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QLabel, QLineEdit, QSpinBox, QWidget
from ruamel.yaml.scalarstring import ScalarString

FIXED_CHOICES = {
    "log_level": ["DEBUG", "INFO", "WARNING", "ERROR"],
    "compute_type": ["int8", "int8_float16", "float16", "float32"],
}
INT_RANGE = 1_000_000_000   # au-delà (ou décimales plus fines que DECIMALS) : saisie YAML, sans arrondi
DECIMALS = 3


class _WheelGuard(QObject):
    """La molette fait défiler la page au lieu de changer la valeur d'un champ qu'on survole.

    Sans cela, faire défiler le formulaire modifie au passage nombres et listes déroulantes.
    """

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() == QEvent.Type.Wheel and isinstance(obj, QWidget) and not obj.hasFocus():
            event.ignore()   # remonte jusqu'à la zone de défilement
            return True
        return False


_wheel_guard = _WheelGuard()


def guard_wheel(widget: QWidget) -> QWidget:
    widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)   # clic ou Tab avant la molette
    widget.installEventFilter(_wheel_guard)
    return widget


def _yaml_inline(value: Any) -> str:
    if value is None:
        return ""
    return yaml.safe_dump(_plain(value), default_flow_style=True, allow_unicode=True, width=10_000).strip().removesuffix("...").strip()


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, bool) or value is None:
        return value
    for t in (int, float, str):
        if isinstance(value, t):
            return t(value)
    return value


def make_editor(value: Any, choices: list[str], on_change: Callable[[Any], None]) -> QWidget:
    """Widget adapté au type de la valeur ; appelle ``on_change(nouvelle_valeur)``."""
    if isinstance(value, bool):
        w = QCheckBox()
        w.setChecked(value)
        w.toggled.connect(on_change)
        return w
    if isinstance(value, int) and abs(value) <= INT_RANGE:
        w = guard_wheel(QSpinBox(minimum=-INT_RANGE, maximum=INT_RANGE))
        w.setValue(value)
        w.valueChanged.connect(on_change)
        return w
    if isinstance(value, float) and abs(value) <= INT_RANGE and round(value, DECIMALS) == value:
        w = guard_wheel(QDoubleSpinBox(minimum=-INT_RANGE, maximum=INT_RANGE, decimals=DECIMALS, singleStep=0.05))
        w.setValue(value)
        w.valueChanged.connect(on_change)
        return w
    if isinstance(value, str):
        style = type(value) if isinstance(value, ScalarString) else str   # garde les guillemets d'origine
        if choices:
            w = guard_wheel(QComboBox(editable=True))
            w.addItems(choices if value in choices else [value, *choices])
            w.setCurrentText(value)
            w.currentTextChanged.connect(lambda t: on_change(style(t)))
            return w
        w = QLineEdit(value)
        w.setCursorPosition(0)   # champ étroit : on voit le début de la valeur, pas sa fin
        w.textEdited.connect(lambda t: on_change(style(t)))
        return w
    # null, listes, dictionnaires, nombres hors limites : saisie YAML en ligne ([a, b], {k: v}, vide = null)
    w = QLineEdit(_yaml_inline(value), placeholderText="null")
    w.setCursorPosition(0)
    w.setToolTip("Valeur YAML : texte, nombre, [liste], {clé: valeur} ; vide = null")

    def edited(text: str) -> None:
        try:
            parsed = yaml.safe_load(text) if text.strip() else None
        except yaml.YAMLError:
            w.setProperty("invalid", True)
            w.style().polish(w)
            return
        w.setProperty("invalid", False)
        w.style().polish(w)
        on_change(parsed)

    w.textEdited.connect(edited)
    return w


def hint_label(text: str) -> QLabel:
    label = QLabel(text, wordWrap=True, objectName="hint")
    font = label.font()
    font.setPointSizeF(font.pointSizeF() * 0.9)
    label.setFont(font)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return label
