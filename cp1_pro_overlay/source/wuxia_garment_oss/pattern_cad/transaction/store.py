"""Atomic edit store with failure preservation, undo, and redo."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..document.model import PatternDocument
from ..document.resolver import require_resolved


class EditCommand(Protocol):
    @property
    def command_type(self) -> str: ...
    def apply(self, document: PatternDocument) -> None: ...
    def to_dict(self) -> dict: ...


@dataclass(frozen=True)
class TransactionResult:
    accepted: bool
    operation: str
    before_revision: int
    after_revision: int
    before_sha256: str
    after_sha256: str
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "operation": self.operation,
            "before_revision": self.before_revision,
            "after_revision": self.after_revision,
            "before_sha256": self.before_sha256,
            "after_sha256": self.after_sha256,
            "error": self.error,
        }


class PatternDocumentStore:
    def __init__(self, document: PatternDocument) -> None:
        require_resolved(document)
        self._document = document.clone()
        self._undo: list[PatternDocument] = []
        self._redo: list[PatternDocument] = []
        self._receipts: list[dict] = []

    @property
    def document(self) -> PatternDocument:
        return self._document.clone()

    @property
    def receipts(self) -> list[dict]:
        return list(self._receipts)

    def apply(self, command: EditCommand) -> TransactionResult:
        before = self._document.clone()
        before_payload = before.to_dict()
        candidate = before.clone()
        try:
            command.apply(candidate)
            candidate.parent_revision = before.revision
            candidate.revision = before.revision + 1
            require_resolved(candidate)
        except Exception as exc:
            result = TransactionResult(
                False,
                command.command_type,
                before.revision,
                before.revision,
                before_payload["document_sha256"],
                before_payload["document_sha256"],
                f"{type(exc).__name__}: {exc}",
            )
            self._receipts.append({**result.to_dict(), "command": command.to_dict()})
            return result
        self._undo.append(before)
        self._redo.clear()
        self._document = candidate
        after_payload = candidate.to_dict()
        result = TransactionResult(
            True,
            command.command_type,
            before.revision,
            candidate.revision,
            before_payload["document_sha256"],
            after_payload["document_sha256"],
        )
        self._receipts.append({**result.to_dict(), "command": command.to_dict()})
        return result

    def undo(self) -> TransactionResult:
        if not self._undo:
            raise ValueError("undo history is empty")
        before = self._document.clone()
        restored = self._undo.pop()
        self._redo.append(before)
        self._document = restored
        result = TransactionResult(
            True,
            "UNDO",
            before.revision,
            restored.revision,
            before.to_dict()["document_sha256"],
            restored.to_dict()["document_sha256"],
        )
        self._receipts.append(result.to_dict())
        return result

    def redo(self) -> TransactionResult:
        if not self._redo:
            raise ValueError("redo history is empty")
        before = self._document.clone()
        restored = self._redo.pop()
        require_resolved(restored)
        self._undo.append(before)
        self._document = restored
        result = TransactionResult(
            True,
            "REDO",
            before.revision,
            restored.revision,
            before.to_dict()["document_sha256"],
            restored.to_dict()["document_sha256"],
        )
        self._receipts.append(result.to_dict())
        return result
