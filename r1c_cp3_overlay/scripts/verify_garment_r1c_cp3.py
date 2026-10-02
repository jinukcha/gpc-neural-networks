#!/usr/bin/env python3
"""Fresh-process CP3 reopen, deterministic rerun, and terminal publication."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from wuxia_garment_oss.completion.diagnosis import diagnose_completion
from wuxia_garment_oss.completion.evidence import render_cp3_evidence
from wuxia_garment_oss.completion.fixtures import guided_candidate, hold_candidate, safe_auto_candidate
from wuxia_garment_oss.completion.model import canonical_sha256
from wuxia_garment_oss.completion.planning import build_repair_plan
from wuxia_garment_oss.completion.preview import build_repair_preview
from wuxia_garment_oss.completion.snapshot import load_cp2_snapshot, write_json
from wuxia_garment_oss.completion.transaction import execute_transaction


BUILD_REL = Path("build/r1c_cp3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_case(canonical: dict, candidate: dict, name: str) -> tuple[dict, dict, dict]:
    report = diagnose_completion(canonical, candidate, f"R1C_CP3_{name.upper()}_DIAGNOSIS")
    plan = build_repair_plan(report, f"R1C_CP3_{name.upper()}_PLAN")
    preview = build_repair_preview(candidate, canonical, plan, f"R1C_CP3_{name.upper()}_PREVIEW")
    return report, plan, preview


def _rerun(root: Path) -> dict:
    canonical = load_cp2_snapshot(root)
    candidates = {
        "safe_auto": safe_auto_candidate(canonical),
        "guided": guided_candidate(canonical),
        "hold": hold_candidate(canonical),
    }
    products = {
        name: _build_case(canonical, candidate, name)
        for name, candidate in candidates.items()
    }
    safe_report, safe_plan, safe_preview = products["safe_auto"]
    guided_report, guided_plan, guided_preview = products["guided"]
    hold_report, hold_plan, hold_preview = products["hold"]
    repaired, safe_tx = execute_transaction(candidates["safe_auto"], canonical, safe_plan, "R1C_CP3_SAFE_COMMIT")
    _, guided_tx = execute_transaction(candidates["guided"], canonical, guided_plan, "R1C_CP3_GUIDED_WAIT")
    _, hold_tx = execute_transaction(candidates["hold"], canonical, hold_plan, "R1C_CP3_HOLD_REJECT")
    _, rollback_tx = execute_transaction(
        candidates["safe_auto"], canonical, safe_plan, "R1C_CP3_FORCED_ROLLBACK", fail_after_operations=2
    )
    return {
        "canonical": canonical,
        "candidates": candidates,
        "products": products,
        "repaired": repaired,
        "transactions": {
            "safe_commit": safe_tx,
            "guided_wait": guided_tx,
            "topology_hold": hold_tx,
            "forced_rollback": rollback_tx,
        },
        "previews": {"safe": safe_preview, "guided": guided_preview, "hold": hold_preview},
    }


def _published_hashes(build: Path) -> dict:
    return {
        "safe_report": read_json(build / "diagnosis/safe_auto.json")["report_sha256"],
        "safe_plan": read_json(build / "plans/safe_auto.json")["plan_sha256"],
        "safe_preview": read_json(build / "previews/safe_auto.json")["preview_sha256"],
        "repaired_state": read_json(build / "repaired/repaired_pattern_snapshot.json")["state_sha256"],
        "safe_tx": read_json(build / "transactions/safe_commit.json")["receipt_sha256"],
        "guided_tx": read_json(build / "transactions/guided_wait.json")["receipt_sha256"],
        "hold_tx": read_json(build / "transactions/topology_hold.json")["receipt_sha256"],
        "rollback_tx": read_json(build / "transactions/forced_rollback.json")["receipt_sha256"],
    }


def _rerun_hashes(products: dict) -> dict:
    safe_report, safe_plan, safe_preview = products["products"]["safe_auto"]
    transactions = products["transactions"]
    return {
        "safe_report": safe_report["report_sha256"],
        "safe_plan": safe_plan["plan_sha256"],
        "safe_preview": safe_preview["preview_sha256"],
        "repaired_state": products["repaired"]["state_sha256"],
        "safe_tx": transactions["safe_commit"]["receipt_sha256"],
        "guided_tx": transactions["guided_wait"]["receipt_sha256"],
        "hold_tx": transactions["topology_hold"]["receipt_sha256"],
        "rollback_tx": transactions["forced_rollback"]["receipt_sha256"],
    }


def _fresh_receipt(preliminary: dict, deterministic: bool, products: dict) -> dict:
    transactions = products["transactions"]
    payload = {
        "contract": "CompletionFreshProcessReopenReceipt/1",
        "writer_process_id": preliminary["writer_process_id"],
        "reopen_process_id": os.getpid(),
        "fresh_process_reopen_pass": preliminary["writer_process_id"] != os.getpid(),
        "deterministic_rerun_pass": deterministic,
        "safe_state_matches_canonical": products["repaired"]["state_sha256"] == products["canonical"]["state_sha256"],
        "guided_state_preserved": transactions["guided_wait"]["state_preserved"],
        "hold_state_preserved": transactions["topology_hold"]["state_preserved"],
        "rollback_state_preserved": transactions["forced_rollback"]["state_preserved"],
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def _terminal_receipt(preliminary: dict, fresh: dict) -> dict:
    gates = (
        preliminary["safe_auto_commit_pass"],
        preliminary["guided_wait_pass"],
        preliminary["topology_hold_pass"],
        preliminary["rollback_pass"],
        preliminary["partial_publication_count"] == 0,
        fresh["fresh_process_reopen_pass"],
        fresh["deterministic_rerun_pass"],
        fresh["safe_state_matches_canonical"],
        fresh["guided_state_preserved"],
        fresh["hold_state_preserved"],
        fresh["rollback_state_preserved"],
    )
    accepted = all(gates)
    payload = {
        **preliminary,
        "phase": "CLOSED",
        "terminal_decision": "CP3_COMPLETE_COMPLETION_REPAIR_TRANSACTION" if accepted else "HOLD_CP3",
        "cp3_acceptance": accepted,
        "fresh_process_reopen_pass": fresh["fresh_process_reopen_pass"],
        "deterministic_rerun_pass": fresh["deterministic_rerun_pass"],
        "next_checkpoint": "GARMENT_CAD_PRO_R1C_CP4",
    }
    payload.pop("receipt_sha256", None)
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def _write_docs(root: Path, receipt: dict) -> None:
    report = f"""# GARMENT-CAD-PRO-R1C / CP3 실행 보고서

```text
terminal decision       {receipt['terminal_decision']}
CP3 acceptance          {receipt['cp3_acceptance']}
safe issues             {receipt['safe_issue_count']}
safe operations         {receipt['safe_operation_count']}
GUIDED wait             {receipt['guided_wait_pass']}
topology HOLD           {receipt['topology_hold_pass']}
atomic rollback         {receipt['rollback_pass']}
partial publication     {receipt['partial_publication_count']}
triangulation           NOT EXECUTED
simulation              NOT EXECUTED
```

CP2 exact geometry and assembly products remain immutable. CP3 repairs only private source-pattern snapshots,
and commits a repaired product only when all operations succeed and the final state matches authority.
"""
    path = root / "docs/cp2b/GARMENT_CAD_PRO_R1C_CP3_EXECUTION_REPORT_KO.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    (root / "NEXT_TASK.md").write_text(
        "# Next task\n\n"
        "**`GARMENT-CAD-PRO-R1C / CP4 — SET-IN SLEEVE TUNIC TRIANGULATION / AVATAR ARRANGEMENT / MATERIAL SETTLING`**\n\n"
        "Use the CP3 committed repaired pattern snapshot as immutable input. Generate component-aware triangulation, "
        "avatar-aware 3D arrangement, seam constraints, and material-calibrated settling for the sleeved tunic.\n",
        encoding="utf-8",
    )


def main() -> int:
    root = parse_args().root.resolve()
    build = root / BUILD_REL
    preliminary = read_json(build / "cp3_preliminary_receipt.json")
    products = _rerun(root)
    deterministic = _published_hashes(build) == _rerun_hashes(products)
    fresh = _fresh_receipt(preliminary, deterministic, products)
    receipt = _terminal_receipt(preliminary, fresh)
    write_json(build / "fresh_process_reopen_receipt.json", fresh)
    write_json(build / "cp3_receipt.json", receipt)
    write_json(root / "R1C_STATUS.json", receipt)
    safe_report, safe_plan, _ = products["products"]["safe_auto"]
    render_cp3_evidence(
        build / "cp3_completion_repair_evidence.png",
        products["candidates"]["safe_auto"],
        products["repaired"],
        safe_report,
        safe_plan,
        products["transactions"],
        products["previews"],
        receipt,
    )
    _write_docs(root, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
