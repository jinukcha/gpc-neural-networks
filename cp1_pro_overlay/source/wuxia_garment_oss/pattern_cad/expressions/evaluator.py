"""Evaluate a deterministic scalar-expression DAG without Python eval."""
from __future__ import annotations

import ast
import math
from typing import Mapping

_ALLOWED_CALLS = {
    "abs": abs,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
}


def expression_dependencies(expression: str) -> set[str]:
    tree = ast.parse(expression, mode="eval")
    dependencies: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in _ALLOWED_CALLS:
            dependencies.add(node.id)
    return dependencies


def _binary(node: ast.BinOp, values: Mapping[str, float]) -> float:
    left = _evaluate(node.left, values)
    right = _evaluate(node.right, values)
    if isinstance(node.op, ast.Add):
        return left + right
    if isinstance(node.op, ast.Sub):
        return left - right
    if isinstance(node.op, ast.Mult):
        return left * right
    if isinstance(node.op, ast.Div):
        if abs(right) < 1.0e-15:
            raise ValueError("division by zero")
        return left / right
    if isinstance(node.op, ast.Pow):
        return left**right
    raise ValueError(f"unsupported operator: {type(node.op).__name__}")


def _evaluate(node: ast.AST, values: Mapping[str, float]) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body, values)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id not in values:
            raise ValueError(f"unknown expression symbol: {node.id}")
        return float(values[node.id])
    if isinstance(node, ast.BinOp):
        return _binary(node, values)
    if isinstance(node, ast.UnaryOp):
        value = _evaluate(node.operand, values)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
        raise ValueError(f"unsupported unary operator: {type(node.op).__name__}")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        function = _ALLOWED_CALLS.get(node.func.id)
        if function is None or node.keywords:
            raise ValueError("unsupported expression call")
        return float(function(*[_evaluate(arg, values) for arg in node.args]))
    raise ValueError(f"unsupported expression node: {type(node).__name__}")


def _order(expressions: Mapping[str, str], inputs: Mapping[str, float]) -> list[str]:
    names = set(expressions)
    dependencies = {
        name: expression_dependencies(expression) - set(inputs)
        for name, expression in expressions.items()
    }
    unknown = sorted({item for refs in dependencies.values() for item in refs if item not in names})
    if unknown:
        raise ValueError(f"unknown DAG dependencies: {unknown}")
    ready = sorted(name for name, refs in dependencies.items() if not refs)
    ordered: list[str] = []
    remaining = {name: set(refs) for name, refs in dependencies.items()}
    while ready:
        current = ready.pop(0)
        ordered.append(current)
        for name in sorted(remaining):
            if current in remaining[name]:
                remaining[name].remove(current)
                if not remaining[name] and name not in ordered and name not in ready:
                    ready.append(name)
                    ready.sort()
    if len(ordered) != len(expressions):
        cycle = sorted(set(expressions) - set(ordered))
        raise ValueError(f"expression dependency cycle: {cycle}")
    return ordered


def evaluate_expression_dag(
    inputs: Mapping[str, float],
    expressions: Mapping[str, str],
) -> dict[str, float]:
    values = {name: float(value) for name, value in inputs.items()}
    for name in _order(expressions, values):
        value = _evaluate(ast.parse(expressions[name], mode="eval"), values)
        if not math.isfinite(value):
            raise ValueError(f"non-finite expression result: {name}")
        values[name] = value
    return {name: values[name] for name in expressions}
