"""Quantity-aware evaluation of restricted parameter expressions."""
from __future__ import annotations

import ast

from .parser import attribute_path, parse_expression
from ..model.quantity import TypedValue, combine_dimensions, typed_literal


def _same_dimension(values: tuple[TypedValue, ...], operation: str) -> None:
    dimensions = {item.dimension for item in values}
    if len(dimensions) != 1:
        raise ValueError(f"quantity mismatch in {operation}")


def _binary(node: ast.BinOp, environment: dict[str, TypedValue]) -> TypedValue:
    left = _evaluate_node(node.left, environment)
    right = _evaluate_node(node.right, environment)
    if isinstance(node.op, (ast.Add, ast.Sub)):
        _same_dimension((left, right), "addition/subtraction")
        value = left.value_si + right.value_si if isinstance(node.op, ast.Add) else left.value_si - right.value_si
        return TypedValue(value, left.dimension)
    if isinstance(node.op, ast.Mult):
        return TypedValue(
            left.value_si * right.value_si,
            combine_dimensions(left.dimension, right.dimension, "MULTIPLY"),
        )
    if isinstance(node.op, ast.Div):
        if right.value_si == 0.0:
            raise ValueError("division by zero")
        return TypedValue(
            left.value_si / right.value_si,
            combine_dimensions(left.dimension, right.dimension, "DIVIDE"),
        )
    raise ValueError("unsupported binary operator")


def _call(node: ast.Call, environment: dict[str, TypedValue]) -> TypedValue:
    name = node.func.id
    values = tuple(_evaluate_node(argument, environment) for argument in node.args)
    if name == "abs":
        return TypedValue(abs(values[0].value_si), values[0].dimension)
    _same_dimension(values, name)
    raw = [item.value_si for item in values]
    if name == "min":
        result = min(raw)
    elif name == "max":
        result = max(raw)
    elif name == "clamp":
        if raw[1] > raw[2]:
            raise ValueError("clamp lower bound exceeds upper")
        result = min(max(raw[0], raw[1]), raw[2])
    else:
        raise ValueError(f"unsupported call: {name}")
    return TypedValue(result, values[0].dimension)


def _evaluate_node(node: ast.AST, environment: dict[str, TypedValue]) -> TypedValue:
    if isinstance(node, ast.Constant):
        return typed_literal(float(node.value))
    if isinstance(node, ast.Attribute):
        path = attribute_path(node)
        if path not in environment:
            raise ValueError(f"missing expression reference: {path}")
        return environment[path]
    if isinstance(node, ast.BinOp):
        return _binary(node, environment)
    if isinstance(node, ast.UnaryOp):
        value = _evaluate_node(node.operand, environment)
        sign = -1.0 if isinstance(node.op, ast.USub) else 1.0
        return TypedValue(sign * value.value_si, value.dimension)
    if isinstance(node, ast.Call):
        return _call(node, environment)
    raise ValueError(f"unsupported evaluation node: {type(node).__name__}")


def evaluate_expression(expression: str, environment: dict[str, TypedValue]) -> TypedValue:
    tree = parse_expression(expression)
    result = _evaluate_node(tree.body, environment)
    result.validate()
    return result
