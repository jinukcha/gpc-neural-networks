extends Node

const REQUIRED_PHASES := [
    "belief", "selection", "deploy", "relocate", "processing",
    "return", "landing", "restore", "policy_guards"
]
const RECEIPT_PATH := "user://maritime_ai_cp1_godot_receipt.json"

func _ready() -> void:
    var result := _run_native_vertical()
    _write_receipt(result)
    if result.get("passed", false):
        print("MARITIME_AI_CP1_GODOT_PASS ", JSON.stringify(result))
        get_tree().quit(0)
        return
    push_error("MARITIME_AI_CP1_GODOT_FAIL %s" % JSON.stringify(result))
    get_tree().quit(1)

func _run_native_vertical() -> Dictionary:
    var script_path := ProjectSettings.globalize_path(
        "res://tests/maritime_ai_cp1/run_native_cp1.sh"
    )
    var native_receipt := ProjectSettings.globalize_path(
        "res://tests/maritime_ai_cp1/native_receipt.json"
    )
    var output: Array = []
    var exit_code := OS.execute(
        script_path, PackedStringArray([native_receipt]), output, true
    )
    if exit_code != 0:
        return {
            "schema": "maritime.ai.cp1.godot.vertical/1",
            "passed": false,
            "exit_code": exit_code,
            "runner_output": output,
        }
    var parsed := _read_json(native_receipt)
    var phases_ok := _phases_complete(parsed.get("phases", []))
    return {
        "schema": "maritime.ai.cp1.godot.vertical/1",
        "passed": bool(parsed.get("passed", false)) and phases_ok,
        "native": parsed,
        "phase_contract_complete": phases_ok,
    }

func _phases_complete(entries: Array) -> bool:
    var present := {}
    for entry in entries:
        if entry is Dictionary and bool(entry.get("contract_present", false)):
            present[String(entry.get("name", ""))] = true
    for phase in REQUIRED_PHASES:
        if not present.has(phase):
            return false
    return true

func _read_json(path: String) -> Dictionary:
    if not FileAccess.file_exists(path):
        return {}
    var file := FileAccess.open(path, FileAccess.READ)
    if file == null:
        return {}
    var parsed = JSON.parse_string(file.get_as_text())
    return parsed if parsed is Dictionary else {}

func _write_receipt(receipt: Dictionary) -> void:
    var file := FileAccess.open(RECEIPT_PATH, FileAccess.WRITE)
    if file != null:
        file.store_string(JSON.stringify(receipt, "  ") + "\n")
