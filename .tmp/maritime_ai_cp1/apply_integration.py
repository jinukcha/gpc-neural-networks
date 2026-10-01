from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()

sources = root / "sdk/src/sources.cmake"
text = sources.read_text()
entries = [
    "    src/mission/profiles/profile_validator.cpp",
    "    src/mission/resources/resource_ledger.cpp",
    "    src/mission/receipts/receipt_log.cpp",
    "    src/mission/persistence/mission_snapshot.cpp",
    "    src/mission/runtime/mission_service.cpp",
    "    src/fisheries/autonomy/beliefs/belief_store.cpp",
    "    src/fisheries/autonomy/ground_selection/selector.cpp",
    "    src/fisheries/autonomy/voyage_policy/policy.cpp",
    "    src/fisheries/autonomy/persistence/snapshot.cpp",
    "    src/fisheries/autonomy/runtime/fishing_skipper.cpp",
]
if entries[0] not in text:
    closing = text.rfind("\n)")
    if closing < 0:
        raise RuntimeError("MARITIME_SOURCES closing delimiter not found")
    text = text[:closing] + "\n" + "\n".join(entries) + text[closing:]
    sources.write_text(text)

core = root / "sdk/cmake/core_targets.cmake"
text = core.read_text()
private_include = (
    "target_include_directories(maritime_core_objects PRIVATE\n"
    "    ${CMAKE_CURRENT_SOURCE_DIR}/src)\n"
)
if private_include not in text:
    marker = "target_compile_features(maritime_core_objects PUBLIC cxx_std_20)"
    text = text.replace(marker, private_include + marker)
    core.write_text(text)

top = root / "sdk/CMakeLists.txt"
text = top.read_text()
subdirectory = "add_subdirectory(tests/maritime_ai_cp1)"
if subdirectory not in text:
    marker = "include(${CMAKE_CURRENT_SOURCE_DIR}/tests/fisheries_suites.cmake)\n"
    text = text.replace(marker, marker + subdirectory + "\n")
    top.write_text(text)

cp1 = root / "sdk/tests/maritime_ai_cp1/CMakeLists.txt"
text = cp1.read_text().replace(
    "${CMAKE_CURRENT_SOURCE_DIR}/../../fixtures",
    "${CMAKE_CURRENT_SOURCE_DIR}/../fixtures",
)
cp1.write_text(text)
