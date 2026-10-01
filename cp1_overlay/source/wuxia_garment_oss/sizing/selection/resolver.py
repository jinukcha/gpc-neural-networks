"""Resolve base size, body blocks, and bounded alteration admission."""
from __future__ import annotations

from dataclasses import dataclass

from ..body_profile.model import BodyMeasurementProfile
from ..instance.model import SelectionReceipt
from ..size_table.model import GarmentSizeTable
from .alteration import classify_admission, difference_plan, residuals
from .blocks import BlockSelection, apply_blocks, select_blocks
from .scoring import rank_sizes


@dataclass(frozen=True)
class SelectionRequest:
    request_id: str
    mode: str
    requested_size_id: str | None = None

    def validate(self) -> None:
        if self.mode not in {"STANDARD_SIZE", "AUTO_BODY_FIT", "CUSTOM_MEASUREMENTS"}:
            raise ValueError(f"unsupported sizing mode: {self.mode}")
        if self.mode == "STANDARD_SIZE" and not self.requested_size_id:
            raise ValueError("STANDARD_SIZE requires requested_size_id")


def _selected_score(ranked: list[dict], size_id: str) -> float:
    return next(float(item["score"]) for item in ranked if item["size_id"] == size_id)


def _regular_blocks() -> BlockSelection:
    return BlockSelection("REGULAR", "REGULAR", (), {})


def resolve_selection(
    request: SelectionRequest,
    table: GarmentSizeTable,
    body: BodyMeasurementProfile | None,
) -> SelectionReceipt:
    request.validate()
    table.validate()
    if request.mode == "STANDARD_SIZE":
        selected_id = str(request.requested_size_id)
        table.entry(selected_id)
        ranked = [{"size_id": selected_id, "score": 0.0, "dimensions": {}}]
        grade = difference_plan(table.entry(table.base_size_id).target, table.entry(selected_id).target, "GRADE")
        return SelectionReceipt(
            request.request_id, request.mode, selected_id, selected_id,
            "REGULAR", "REGULAR", "NORMAL_GRADE", ranked, grade, [], {}, [], {},
        )
    if body is None:
        raise ValueError(f"{request.mode} requires a body profile")
    body.validate()
    ranked = rank_sizes(body.measurements, table)
    recommended_id = str(ranked[0]["size_id"])
    selected_id = request.requested_size_id or recommended_id
    selected_entry = table.entry(selected_id)
    blocks = select_blocks(body.measurements, selected_entry.target)
    adjusted = apply_blocks(selected_entry.target, blocks)
    residual_by_name = residuals(body.measurements, adjusted)
    grade = difference_plan(table.entry(table.base_size_id).target, selected_entry.target, "GRADE")
    custom = difference_plan(adjusted, body.measurements, "CUSTOM_ALTERATION")
    admission, warnings = classify_admission(
        request.mode,
        body.measurements,
        blocks,
        residual_by_name,
        selected_id,
        recommended_id,
        _selected_score(ranked, selected_id),
        float(ranked[0]["score"]),
    )
    return SelectionReceipt(
        request.request_id,
        request.mode,
        selected_id,
        recommended_id,
        blocks.shape_block,
        blocks.height_block,
        admission,
        ranked,
        grade,
        custom,
        residual_by_name,
        warnings,
        blocks.evidence,
    )
