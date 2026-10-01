"""Small explicit edit commands for atomic PatternDocument transactions."""
from __future__ import annotations

from dataclasses import dataclass

from ..document.model import PatternDocument


@dataclass(frozen=True)
class SetInputCommand:
    input_id: str
    value: float

    @property
    def command_type(self) -> str:
        return "SET_INPUT"

    def apply(self, document: PatternDocument) -> None:
        if self.input_id not in document.inputs:
            raise KeyError(self.input_id)
        document.inputs[self.input_id] = float(self.value)

    def to_dict(self) -> dict:
        return {"command_type": self.command_type, "input_id": self.input_id, "value": float(self.value)}


@dataclass(frozen=True)
class SetExpressionCommand:
    expression_id: str
    expression: str

    @property
    def command_type(self) -> str:
        return "SET_EXPRESSION"

    def apply(self, document: PatternDocument) -> None:
        if self.expression_id not in document.expressions:
            raise KeyError(self.expression_id)
        document.expressions[self.expression_id] = str(self.expression)

    def to_dict(self) -> dict:
        return {
            "command_type": self.command_type,
            "expression_id": self.expression_id,
            "expression": self.expression,
        }
