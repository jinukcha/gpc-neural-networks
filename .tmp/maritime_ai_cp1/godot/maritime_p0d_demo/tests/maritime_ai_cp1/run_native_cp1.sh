#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
OUT="${1:-$ROOT/godot/maritime_p0d_demo/tests/maritime_ai_cp1/native_receipt.json}"
LOG="${OUT%.json}.log"
RAW="${OUT%.json}.raw.json"
BINARY="$ROOT/build_cp1_native/maritime_ai_cp1_contract_test"
SOURCE="$ROOT/sdk/tests/maritime_ai_cp1/fishing_cp1_contract_test.cpp"
PHASES=(belief selection deploy relocate processing return landing restore policy_guards)

if [[ ! -x "$BINARY" || ! -f "$SOURCE" ]]; then
  printf '{"schema":"maritime.ai.cp1.godot.native/1","passed":false,"reason":"native_cp1_contract_unavailable"}\n' > "$OUT"
  exit 33
fi

set +e
"$BINARY" "$RAW" > "$LOG" 2>&1
STATUS=$?
set -e
python3 - "$SOURCE" "$LOG" "$RAW" "$OUT" "$STATUS" <<'PY'
from pathlib import Path
import hashlib,json,sys
source,log,raw,out,status=sys.argv[1:]
phases=['belief','selection','deploy','relocate','processing','return','landing','restore','policy_guards']
text=Path(log).read_text(errors='replace')
try: native=json.loads(Path(raw).read_text())
except Exception: native={}
receipt={
 'schema':'maritime.ai.cp1.godot.native/1',
 'passed':int(status)==0 and native.get('passed') is True,
 'native_test_source':'sdk/tests/maritime_ai_cp1/fishing_cp1_contract_test.cpp',
 'native_test_source_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),
 'ctest_status':int(status),
 'phases':[{'name':p,'contract_present':p in native.get('phases',[]) and f'phase={p}' in text} for p in phases],
 'native_receipt':native,
 'log_tail':text[-12000:],
}
Path(out).write_text(json.dumps(receipt,indent=2)+'\n')
PY
exit "$STATUS"
