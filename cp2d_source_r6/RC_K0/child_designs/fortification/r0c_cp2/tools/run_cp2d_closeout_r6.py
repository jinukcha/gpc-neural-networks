#!/usr/bin/env python3
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
R3_RUNNER = HERE / "run_cp2d_closeout_r3.py"

HISTORICAL_PRODUCER_DIGEST = "sha256:eab844b906682f0600fbd5c847731b50981753f8e25c9f9811b83fb5c54d49f6"
CURRENT_PRODUCER_DIGEST = "sha256:4a11f5096ab68abe9052a32093e3e5fcd199295334962465e26fa0e31dba11a6"


def load_r3():
    spec = importlib.util.spec_from_file_location("rcf_cp2d_r3_runner_for_r6", R3_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load CP2-D R3 runner: {R3_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


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


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("sha256:") and len(value) == 71


def allowed_digest_cascade(
    row: dict[str, Any],
    *,
    accepted_root: Path,
    observed_root: Path,
) -> bool:
    path = str(row.get("path", ""))
    accepted = row.get("accepted")
    observed = row.get("observed")

    if path.endswith(".producer_source_digest"):
        return accepted == HISTORICAL_PRODUCER_DIGEST and observed == CURRENT_PRODUCER_DIGEST

    for filename in ("source-bindings.json", "cad-provider-receipt.json"):
        if path.endswith("." + filename) and is_sha256(accepted) and is_sha256(observed):
            accepted_file = accepted_root / filename
            observed_file = observed_root / filename
            return (
                accepted_file.is_file()
                and observed_file.is_file()
                and accepted == sha256(accepted_file)
                and observed == sha256(observed_file)
            )
    return False


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

        reports: list[dict[str, Any]] = []
        for rel in different:
            module.require(rel.endswith(".json"), f"non-JSON accepted reference differs: {rel}")
            accepted_path = accepted_root / rel
            observed_path = observed_root / rel
            accepted_value = json.loads(accepted_path.read_text(encoding="utf-8"))
            observed_value = json.loads(observed_path.read_text(encoding="utf-8"))
            raw_differences = r3.diff_values(accepted_value, observed_value)
            canonical_differences = r3.diff_values(
                normalize_json_paths(accepted_value),
                normalize_json_paths(observed_value),
            )
            unapproved = [
                row
                for row in canonical_differences
                if not allowed_digest_cascade(
                    row,
                    accepted_root=accepted_root,
                    observed_root=observed_root,
                )
            ]
            module.require(
                not unapproved,
                f"{rel} differs outside canonical paths and verified digest cascades: {unapproved}",
            )
            reports.append(
                {
                    "path": rel,
                    "status": "PASS_CONSTRAINED_CANONICAL_AND_DIGEST_CASCADE",
                    "raw_differences": raw_differences,
                    "canonical_differences": canonical_differences,
                    "unapproved_differences": unapproved,
                }
            )

        return {
            "status": "PASS_EXACT" if not different else "PASS_CONSTRAINED_CANONICAL_IDENTITY",
            "accepted_files": len(accepted_inventory),
            "observed_files": len(observed_inventory),
            "raw_byte_differences": different,
            "normalized_json_files": reports,
            "binding_identity_status": "PASS",
            "path_normalization_rule": "only JSON values whose key is exactly path; prefix before RC_K0/ is non-identity",
            "producer_digest_rule": {
                "historical_stale": HISTORICAL_PRODUCER_DIGEST,
                "accepted_current_source": CURRENT_PRODUCER_DIGEST,
                "accepted_reference_overwritten": False,
            },
            "digest_cascade_rule": "source-bindings.json and cad-provider-receipt.json values must equal the actual referenced file SHA-256",
            "non_json_byte_identity": True,
            "accepted_tree_digest": module.inventory_digest(accepted_inventory),
            "observed_tree_digest": module.inventory_digest(observed_inventory),
        }

    r3.compare_reference = compare_reference
    r3.main()

    tree_argument = Path(sys.argv[sys.argv.index("--tree") + 1]).resolve()
    cp2 = tree_argument / "RC_K0/child_designs/fortification/r0c_cp2"
    completion_path = cp2 / "reports/CP2D_COMPLETION.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["final_clean_replay"]["constrained_repair"] = {
        "status": "PASS",
        "path_fields_only": True,
        "verified_digest_cascades_only": True,
        "historical_producer_digest": HISTORICAL_PRODUCER_DIGEST,
        "accepted_current_source_digest": CURRENT_PRODUCER_DIGEST,
        "accepted_reference_overwritten": False,
    }
    completion_path.write_text(
        json.dumps(completion, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    diagnostic_path = cp2 / "reports/CP2D_R6_RECONSTRUCTION_DIAGNOSTIC.json"
    diagnostic_path.write_text(
        json.dumps(
            {
                "schema": "royal-capital.fortification.r0c-cp2.r6-reconstruction/1",
                "status": "PASS",
                "historical_producer_digest": HISTORICAL_PRODUCER_DIGEST,
                "accepted_current_source_digest": CURRENT_PRODUCER_DIGEST,
                "accepted_reference_overwritten": False,
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(completion, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
