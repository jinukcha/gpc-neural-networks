"""AST validation for deterministic garment parameter expressions."""
from __future__ import annotations

import ast


ALLOWED_FUNCTIONS = {"min", "max", "clamp", "abs"}
ALLOWED_ROOTS = {"param", "body", "block", "component", "boundary", "material"}
_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.USub,
    ast.UAdd,
    ast.Constant,
    ast.Name,
    ast.Attribute,
    ast.Call,
    ast.Load,
)


def attribute_path(node: ast.AST) -> str:
    parts = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        raise ValueError("expression attribute root must be a name")
    parts.append(current.id)
    path = ".".join(reversed(parts))
    if path.split(".", 1)[0] not in ALLOWED_ROOTS:
        raise ValueError(f"unsupported expression namespace: {path}")
    return path


def _validate_call(node: ast.Call) -> None:
    if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_FUNCTIONS:
        raise ValueError("unsupported expression function")
    expected = {"min": 2, "max": 2, "clamp": 3, "abs": 1}[node.func.id]
    if len(node.args) != expected or node.keywords:
        raise ValueError(f"invalid {node.func.id} argument count")


def parse_expression(expression: str) -> ast.Expression:
    if not expression or len(expression) > 1024:
        raise ValueError("invalid expression length")
    tree = ast.parse(expression, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"unsupported expression node: {type(node).__name__}")
        if isinstance(node, ast.Call):
            _validate_call(node)
        if isinstance(node, ast.Name) and not isinstance(getattr(node, "ctx", None), ast.Load):
            raise ValueError("expression assignment is forbidden")
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError("only numeric literals are allowed")
    return tree


def referenced_paths(tree: ast.Expression) -> tuple[str, ...]:
    paths = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            parent_is_attribute = any(
                isinstance(parent, ast.Attribute) and parent.value is node
                for parent in ast.walk(tree)
            )
            if not parent_is_attribute:
                paths.add(attribute_path(node))
    return tuple(sorted(paths))
