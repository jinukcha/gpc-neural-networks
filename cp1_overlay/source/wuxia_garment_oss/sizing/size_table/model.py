"""Arbitrary-cardinality garment size table authority."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..body_profile.model import BodyMeasurements


@dataclass(frozen=True)
class SizeEntry:
    size_id: str
    target: BodyMeasurements

    def to_dict(self) -> dict:
        if not self.size_id:
            raise ValueError("size_id is required")
        return {"size_id": self.size_id, "target_body": self.target.to_dict()}


@dataclass(frozen=True)
class GarmentSizeTable:
    size_table_id: str
    base_size_id: str
    sizes: tuple[SizeEntry, ...]

    def validate(self) -> None:
        if not self.size_table_id or not self.sizes:
            raise ValueError("size table identity and entries are required")
        ids = [entry.size_id for entry in self.sizes]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate size IDs")
        if self.base_size_id not in ids:
            raise ValueError("base size is not present")
        for entry in self.sizes:
            entry.target.validate()

    def entry(self, size_id: str) -> SizeEntry:
        self.validate()
        for entry in self.sizes:
            if entry.size_id == size_id:
                return entry
        raise KeyError(size_id)

    def ordered_ids(self) -> tuple[str, ...]:
        self.validate()
        return tuple(entry.size_id for entry in self.sizes)

    def to_dict(self) -> dict:
        self.validate()
        return {
            "contract": "GarmentSizeTable/1",
            "size_table_id": self.size_table_id,
            "base_size_id": self.base_size_id,
            "sizes": [entry.to_dict() for entry in self.sizes],
        }

    @classmethod
    def from_entries(
        cls, size_table_id: str, base_size_id: str, entries: Iterable[SizeEntry]
    ) -> "GarmentSizeTable":
        result = cls(size_table_id, base_size_id, tuple(entries))
        result.validate()
        return result
