from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceSpec:
    key: str
    repository: str
    url: str
    ref: str
    commit: str
    license_name: str
    adoption: str
    question: str
    paths: tuple[str, ...]


SOURCES = (
    SourceSpec(
        key="MOOS-IvP",
        repository="moos-ivp/moos-ivp",
        url="https://github.com/moos-ivp/moos-ivp.git",
        ref="v24.8.1",
        commit="477be7e91bf220185d55e5a2a9dba16dbde32c3f",
        license_name="GPL-3.0/LGPL-3.0 dual distribution; commercial terms available upstream",
        adoption="REFERENCE_ONLY_NO_CODE_COPY_NO_LINK",
        question="behavior arbitration, lifecycle events, helm decision boundaries",
        paths=(
            "ivp/src/lib_helmivp/BehaviorSet.h",
            "ivp/src/lib_helmivp/BehaviorSet.cpp",
            "ivp/src/lib_helmivp/BehaviorSetEntry.h",
            "ivp/src/lib_behaviors/IvPBehavior.h",
            "ivp/src/lib_behaviors/IvPBehavior.cpp",
            "ivp/src/pHelmIvP/HelmEngine.h",
            "ivp/src/pHelmIvP/HelmEngine.cpp",
        ),
    ),
    SourceSpec(
        key="PX4-Autopilot",
        repository="PX4/PX4-Autopilot",
        url="https://github.com/PX4/PX4-Autopilot.git",
        ref="v1.17.0",
        commit="d6f12ad1c4f70ad3230afd7d86e971421e02fef4",
        license_name="BSD-3-Clause",
        adoption="SEMANTIC_REFERENCE_REIMPLEMENT_PROJECT_CONTRACTS",
        question="mission item lifecycle, failure handling, resume and return semantics",
        paths=(
            "src/modules/navigator/mission_block.h",
            "src/modules/navigator/mission_block.cpp",
            "src/modules/navigator/mission_base.h",
            "src/modules/navigator/mission_base.cpp",
            "src/modules/navigator/mission.h",
            "src/modules/navigator/mission.cpp",
            "src/modules/navigator/CMakeLists.txt",
        ),
    ),
    SourceSpec(
        key="Unified-Planning",
        repository="aiplan4eu/unified-planning",
        url="https://github.com/aiplan4eu/unified-planning.git",
        ref="v1.3.0",
        commit="42e66926e400ab1367b5b02af504d8c7016b9243",
        license_name="Apache-2.0",
        adoption="OFFLINE_DESIGN_ORACLE_NO_GAMEPLAY_RUNTIME",
        question="HTN task, method, decomposition, ordering and plan validation semantics",
        paths=(
            "unified_planning/model/htn",
            "unified_planning/model/action.py",
            "unified_planning/model/problem.py",
            "unified_planning/plans/hierarchical_plan.py",
            "unified_planning/engines/mixins/plan_validator.py",
        ),
    ),
    SourceSpec(
        key="discrete-choosers",
        repository="CarrKnight/discrete-choosers",
        url="https://github.com/CarrKnight/discrete-choosers.git",
        ref="master",
        commit="1957b2642fcb48c498f737f7b1fcd657ffd11a73",
        license_name="MIT",
        adoption="SEMANTIC_REFERENCE_OWNED_IMPLEMENTATION",
        question="bounded exploration, exploitation and belief-based fishing-ground choice",
        paths=(
            "README.md",
            "src/main/java/io/github/carrknight",
        ),
    ),
    SourceSpec(
        key="DISPLACE",
        repository="frabas/DISPLACE_GUI",
        url="https://github.com/frabas/DISPLACE_GUI.git",
        ref="v1.8.0 lineage",
        commit="96eadecb1980d6f9ad22571cd6f963cc05379815",
        license_name="GPL-2.0",
        adoption="REFERENCE_ONLY_NO_CODE_COPY_NO_LINK",
        question="fishing decision trees, ground changes, quota, TAC and vessel voyage semantics",
        paths=(
            "commons/dtree",
            "commons/Vessel.cpp",
            "include/Vessel.h",
            "include/comstructs.h",
            "tests/unittests/dtrees.cpp",
        ),
    ),
    SourceSpec(
        key="mizer",
        repository="sizespectrum/mizer",
        url="https://github.com/sizespectrum/mizer.git",
        ref="v3.4.0",
        commit="bc15500d3ead330e79c86949830548c4b580bebf",
        license_name="GPL-3.0",
        adoption="OFFLINE_ORACLE_REFERENCE_ONLY_NO_CODE_COPY",
        question="selectivity, fishing mortality, catch and yield aggregation and input validation",
        paths=(
            "DESCRIPTION",
            "R/setFishing.R",
            "R/selectivity_funcs.R",
            "R/project_methods.R",
            "R/summary_methods.R",
            "tests/testthat/test-selectivity_funcs.R",
            "tests/testthat/test-setFishing.R",
        ),
    ),
    SourceSpec(
        key="OceanParcels",
        repository="OceanParcels/Parcels",
        url="https://github.com/OceanParcels/Parcels.git",
        ref="v3.1.4",
        commit="fb88390d2bc9e2d45cd3056c601c7acaa9afcdac",
        license_name="MIT",
        adoption="OFFLINE_ORACLE_REFERENCE_ONLY",
        question="advection and diffusion trajectory semantics for aging fishing-ground beliefs",
        paths=(
            "parcels/application_kernels/advection.py",
            "parcels/application_kernels/advectiondiffusion.py",
            "parcels/field.py",
            "parcels/fieldset.py",
            "parcels/grid.py",
            "parcels/particleset.py",
            "parcels/rng.py",
            "tests/test_advection.py",
            "tests/test_diffusion.py",
        ),
    ),
)


def run(*args: str) -> None:
    subprocess.run(args, check=True)


def fetch_source(spec: SourceSpec, source_root: Path) -> Path:
    target = source_root / spec.key
    run("git", "init", "-q", str(target))
    run("git", "-C", str(target), "remote", "add", "origin", spec.url)
    run("git", "-C", str(target), "fetch", "-q", "--depth=1", "origin", spec.commit)
    run("git", "-C", str(target), "checkout", "-q", "--detach", "FETCH_HEAD")
    actual = subprocess.check_output(
        ("git", "-C", str(target), "rev-parse", "HEAD"), text=True
    ).strip()
    if actual != spec.commit:
        raise RuntimeError(f"commit mismatch for {spec.key}: {actual}")
    return target


def copy_path(source_root: Path, output_root: Path, relative: str) -> None:
    source = source_root / relative
    target = output_root / "selected" / relative
    if not source.exists():
        raise FileNotFoundError(f"selected source missing: {source}")
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def copy_license(spec: SourceSpec, source: Path, output: Path) -> None:
    license_dir = output / "license"
    license_dir.mkdir(parents=True, exist_ok=True)
    candidates = sorted(
        path
        for path in source.iterdir()
        if path.is_file()
        and path.name.lower().startswith(("license", "copying", "notice"))
    )
    for path in candidates:
        shutil.copy2(path, license_dir / path.name)
    if any(license_dir.iterdir()):
        return
    if spec.key != "mizer":
        raise RuntimeError(f"license material missing: {spec.key}")
    shutil.copy2(source / "DESCRIPTION", license_dir / "PACKAGE_DESCRIPTION")
    system_gpl = Path("/usr/share/common-licenses/GPL-3")
    if system_gpl.exists():
        shutil.copy2(system_gpl, license_dir / "GPL-3.0.txt")
        return
    (license_dir / "GPL-3.0-DECLARATION.txt").write_text(
        "Upstream DESCRIPTION declares License: GPL-3. "
        "Full text: https://www.gnu.org/licenses/gpl-3.0.txt\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_record(path: Path, root: Path) -> dict[str, object]:
    data = path.read_bytes()
    try:
        lines: int | None = len(data.decode("utf-8").splitlines())
    except UnicodeDecodeError:
        lines = None
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "lines": lines,
    }


def write_project_receipt(spec: SourceSpec, folder: Path) -> dict[str, object]:
    files = [
        file_record(path, folder)
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    ]
    receipt: dict[str, object] = {
        "schema": "maritime.ai.r0a.oss_source_receipt/1",
        "repository": spec.repository,
        "ref": spec.ref,
        "commit": spec.commit,
        "license": spec.license_name,
        "adoption": spec.adoption,
        "question": spec.question,
        "selected_file_count": len(files),
        "selected_bytes": sum(int(item["bytes"]) for item in files),
        "files": files,
    }
    (folder / "SOURCE_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (folder / "REFERENCE_SCOPE.md").write_text(
        f"# {spec.key} reference scope\n\n"
        f"- Repository: `{spec.repository}`\n"
        f"- Ref: `{spec.ref}`\n"
        f"- Commit: `{spec.commit}`\n"
        f"- License: {spec.license_name}\n"
        f"- Adoption: `{spec.adoption}`\n"
        f"- Design question: {spec.question}\n\n"
        "This is a bounded source oracle, not a normal gameplay runtime dependency. "
        "Project-owned SDK contracts and native state authority remain canonical.\n",
        encoding="utf-8",
    )
    return receipt


def write_global_receipt(output: Path, receipts: dict[str, dict[str, object]]) -> None:
    inventory_path = output / "MARITIME_AI_R0A_OSS_SOURCE_INVENTORY.csv"
    rows: list[dict[str, object]] = []
    for spec in SOURCES:
        for path in sorted((output / spec.key).rglob("*")):
            if path.is_file():
                row = file_record(path, output / spec.key)
                row.update(
                    {
                        "project": spec.key,
                        "commit": spec.commit,
                        "license": spec.license_name,
                        "adoption": spec.adoption,
                    }
                )
                rows.append(row)
    columns = (
        "project",
        "path",
        "bytes",
        "sha256",
        "lines",
        "commit",
        "license",
        "adoption",
    )
    with inventory_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema": "maritime.ai.r0a.oss_workkit/1",
        "project_count": len(SOURCES),
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "projects": receipts,
    }
    (output / "MARITIME_AI_R0A_OSS_WORKKIT.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def write_hashes(output: Path) -> None:
    lines = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            lines.append(f"{sha256(path)}  {path.relative_to(output).as_posix()}")
    (output / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate(output: Path) -> None:
    for spec in SOURCES:
        folder = output / spec.key
        if not any((folder / "selected").rglob("*")):
            raise RuntimeError(f"empty selected source: {spec.key}")
        if not any((folder / "license").iterdir()):
            raise RuntimeError(f"empty license folder: {spec.key}")
        if not (folder / "SOURCE_RECEIPT.json").is_file():
            raise RuntimeError(f"receipt missing: {spec.key}")
    file_count = sum(1 for path in output.rglob("*") if path.is_file())
    if file_count < 40:
        raise RuntimeError(f"unexpectedly small workkit: {file_count} files")
    print(json.dumps({"projects": len(SOURCES), "files": file_count}, indent=2))


def main() -> None:
    source_root = Path("/tmp/maritime-ai-r0a-source")
    output = Path("workkit")
    shutil.rmtree(source_root, ignore_errors=True)
    shutil.rmtree(output, ignore_errors=True)
    source_root.mkdir(parents=True)
    output.mkdir(parents=True)
    receipts: dict[str, dict[str, object]] = {}
    for spec in SOURCES:
        source = fetch_source(spec, source_root)
        project_output = output / spec.key
        for relative in spec.paths:
            copy_path(source, project_output, relative)
        copy_license(spec, source, project_output)
        receipts[spec.key] = write_project_receipt(spec, project_output)
    write_global_receipt(output, receipts)
    write_hashes(output)
    validate(output)


if __name__ == "__main__":
    main()
