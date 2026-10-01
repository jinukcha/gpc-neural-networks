"""Pure transaction model used by Python qualification and Godot runtime."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..library.model import OutfitAssemblyPlan


@dataclass
class OutfitRuntimeState:
    equipped_garment_ids: tuple[str, ...] = ()
    active_outfit_id: str | None = None
    body_hide_mask_sha256: str | None = None
    revision: int = 0
    history: list[dict] = field(default_factory=list)

    def snapshot(self) -> dict:
        return {
            "equipped_garment_ids": list(self.equipped_garment_ids),
            "active_outfit_id": self.active_outfit_id,
            "body_hide_mask_sha256": self.body_hide_mask_sha256,
            "revision": self.revision,
        }


def commit_outfit(
    state: OutfitRuntimeState,
    plan: OutfitAssemblyPlan,
    body_hide_mask_sha256: str,
) -> dict:
    before = state.snapshot()
    if plan.status != "ACCEPTED":
        receipt = {
            "operation": "EQUIP_OUTFIT",
            "status": "REJECTED_ATOMIC",
            "outfit_id": plan.outfit_id,
            "reasons": list(plan.rejection_reasons),
            "before": before,
            "after": before,
            "state_preserved": True,
        }
        state.history.append(receipt)
        return receipt
    state.equipped_garment_ids = tuple(plan.layer_order)
    state.active_outfit_id = plan.outfit_id
    state.body_hide_mask_sha256 = body_hide_mask_sha256
    state.revision += 1
    after = state.snapshot()
    receipt = {
        "operation": "EQUIP_OUTFIT",
        "status": "COMMITTED",
        "outfit_id": plan.outfit_id,
        "reasons": [],
        "before": before,
        "after": after,
        "state_preserved": False,
    }
    state.history.append(receipt)
    return receipt


def unequip_outfit(state: OutfitRuntimeState) -> dict:
    before = state.snapshot()
    state.equipped_garment_ids = ()
    state.active_outfit_id = None
    state.body_hide_mask_sha256 = None
    state.revision += 1
    after = state.snapshot()
    receipt = {
        "operation": "UNEQUIP_OUTFIT",
        "status": "COMMITTED",
        "before": before,
        "after": after,
    }
    state.history.append(receipt)
    return receipt
