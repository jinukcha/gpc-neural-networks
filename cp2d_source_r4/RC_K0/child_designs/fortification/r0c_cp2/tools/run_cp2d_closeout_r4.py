#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
R3_RUNNER = HERE / "run_cp2d_closeout_r3.py"


def load_r3():
    spec = importlib.util.spec_from_file_location("rcf_cp2d_r3_runner_for_r4", R3_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load CP2-D R3 runner: {R3_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def canonical_source_path(value: str) -> str:
    normalized = value.replace("\\", "/")
    marker = "RC_K0/"
    position = normalized.find(marker)
    return normalized[position:] if position >= 0 else normalized


def normalize_json_paths(value: Any, key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {name: normalize_json_paths(child, name) for name, child in sorted(value.items())}
    if isinstance(value, list):
        return [normalize_json_paths(child, key) for child in value]
    if key == "path" and isinstance(value, str):
        return canonical_source_path(value)
    return value


def main() -> None:
    r3 = load_r3()

    def compare_reference(module: Any, accepted_root: Path, observed_root: Path) -> dict[str, Any]:
        accepted_inventory = module.inventory(accepted_root)
        observed_inventory = module.inventory(observed_root)
        missing = sorted(set(accepted_inventory) - set(observed_inventory))
        extra = sorted(set(observed_inventory) - set(accepted_inventory))
        different = sorted(
            rel
            for rel in set(accepted_inventory) & set(observed_inventory)
            if accepted_inventory[rel] != observed_inventory[rel]
        )
        module.require(not missing, f"accepted reference files missing from replay: {missing}")
        module.require(not extra, f"replay contains unexpected reference files: {extra}")

        normalized_reports: list[dict[str, Any]] = []
        for rel in different:
            module.require(rel.endswith(".json"), f"non-JSON accepted reference differs: {rel}")
            accepted_path = accepted_root / rel
            observed_path = observed_root / rel
            accepted_value = json.loads(accepted_path.read_text(encoding="utf-8"))
            observed_value = json.loads(observed_path.read_text(encoding="utf-8"))
            raw_differences = r3.diff_values(accepted_value, observed_value)
            non_path_differences = [row for row in raw_differences if not str(row["path"]).endswith(".path")]
            canonical_differences = r3.diff_values(
                normalize_json_paths(accepted_value),
                normalize_json_paths(observed_value),
            )
            module.require(not non_path_differences, f"{rel} differs outside path fields: {non_path_differences}")
            module.require(not canonical_differences, f"{rel} differs after canonical path normalization: {canonical_differences}")
            normalized_reports.append(
                {
                    "path": rel,
                    "status": "PASS_CANONICAL_WORKSPACE_PATH_NORMALIZED",
                    "raw_differences": raw_differences,
                    "non_path_differences": non_path_differences,
                    "canonical_differences": canonical_differences,
                }
            )

        return {
            "status": "PASS_EXACT" if not different else "PASS_CANONICAL_JSON_PATH_NORMALIZED",
            "accepted_files": len(accepted_inventory),
            "observed_files": len(observed_inventory),
            "raw_byte_differences": different,
            "normalized_json_files": normalized_reports,
            "binding_identity_status": "PASS",
            "path_normalization_rule": "only JSON values whose key is exactly path; prefix before RC_K0/ is non-identity",
            "non_json_byte_identity": True,
            "non_path_json_identity": True,
            "accepted_tree_digest": module.inventory_digest(accepted_inventory),
            "observed_tree_digest": module.inventory_digest(observed_inventory),
        }

    r3.compare_reference = compare_reference
    r3.main()

    tree_argument = Path(sys.argv[sys.argv.index("--tree") + 1]).resolve()
    cp2 = tree_argument / "RC_K0/child_designs/fortification/r0c_cp2"
    completion_path = cp2 / "reports/CP2D_COMPLETION.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    attempts = list(completion.get("failed_attempts_preserved", []))
    attempts.append(
        {
            "run_id": 36232961939,
            "status": "PRESERVED",
            "cause": "cad-provider receipt and result JSON contained execution-local absolute path fields",
            "failure_artifact_id": 10902579761,
            "logs_artifact_id": 10902887678,
            "rollback": False,
        }
    )
    completion["failed_attempts_preserved"] = attempts
    completion["final_clean_replay"]["path_normalization_contract"] = {
        "normalized_field": "JSON key exactly equal to path",
        "canonical_prefix": "RC_K0/",
        "non_path_json_fields": "BYTE_SEMANTIC_EQUAL",
        "non_json_files": "BYTE_EQUAL",
    }
    completion_path.write_text(
        json.dumps(completion, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    failed_path = cp2 / "reports/CP2D_FAILED_ATTEMPTS.json"
    failed = json.loads(failed_path.read_text(encoding="utf-8"))
    failed["attempts"] = attempts
    failed_path.write_text(
        json.dumps(failed, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(completion, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
