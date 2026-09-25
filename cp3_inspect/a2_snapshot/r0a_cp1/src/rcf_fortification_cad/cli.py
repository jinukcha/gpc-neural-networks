from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contract import CadStatus
from .provider import Build123dProviderAdapter


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--cp0-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    result = Build123dProviderAdapter(args.cp0_root).execute(request, args.output)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.status is CadStatus.SUCCEEDED else 2 if result.status is CadStatus.REJECTED else 1


if __name__ == "__main__":
    raise SystemExit(main())
