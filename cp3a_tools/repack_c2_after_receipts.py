from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "r0c_cp1"))
sys.path.insert(0, str(REPO_ROOT))

import package_tools as pt  # noqa: E402
from cp2d_tools import driver as cp2d  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: repack_c2_after_receipts.py WORK OUTPUT INPUTS")
    work = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()
    inputs = Path(sys.argv[3]).resolve()
    current = work / "current"
    baseline = work / "baseline"
    c1 = work / "c1"
    parent = work / "parent_c2c"
    cp2 = current / "RC_K0/child_designs/fortification/r0c_cp2"

    # The closeout document is carried in C2R but is not part of accepted C2 source authority.
    closeout_doc = cp2 / "docs/CP2D_CLOSEOUT.md"
    require(closeout_doc.is_file(), "receipt closeout document was not preserved until C2R packaging")
    closeout_doc.unlink()

    cp2d.remove_disposable_caches(current)
    pt.write_registry(current)
    registry = pt.validate_registry(current)
    require(registry.get("status") == "PASS", f"registry failed after receipt cleanup: {registry}")

    full = output / "C2.zip"
    patch = output / "C2P.zip"
    source = output / "C2S.zip"
    source_meta = pt.create_source_delta(c1, current, source, parent_name="C1.zip", stage=cp2d.STAGE)
    patch_meta = cp2d.create_direct_patch(baseline, current, patch, stage=cp2d.STAGE)
    pt.deterministic_zip(current, full)

    validations = {
        "full_reopen": pt.verify_zip_tree(full, current, work / "verify_post_receipt_full"),
        "baseline_plus_patch": cp2d.verify_baseline_plus_patch(baseline, patch, current, work / "verify_post_receipt_patch"),
        "C1_plus_source": pt.verify_parent_plus_delta(c1, source, current, work / "verify_post_receipt_source"),
        "patch_payload": pt.verify_patch_payload(patch, current, source, work / "verify_post_receipt_payload"),
    }
    for name, value in validations.items():
        require(value.get("status") == "PASS", f"post-receipt {name} failed: {value}")

    artifacts = {
        name: cp2d.zip_artifact(output / name)
        for name in ("C2.zip", "C2P.zip", "C2S.zip", "C2R.zip")
    }
    require(all(value.get("status") == "PASS" for value in artifacts.values()), "post-receipt ZIP integrity failed")

    check = {
        "schema": "royal-capital.fortification.c2-post-receipt-repack/1",
        "status": "PASS",
        "authority_inputs": cp2d.validate_inputs(inputs),
        "registry": registry,
        "tree_digest": cp2d.tree_digest(current),
        "tree_files": len(cp2d.inventory(current)),
        "source_delta": source_meta,
        "cumulative_patch": patch_meta,
        "artifacts": artifacts,
        "validation": validations,
        "receipt_zip_preserved_before_source_cleanup": True,
        "closeout_document_in_C2R": True,
        "closeout_document_in_C2_source": False,
    }
    (output / "C2_POST_RECEIPT_REPACK.json").write_text(
        json.dumps(check, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(check, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
