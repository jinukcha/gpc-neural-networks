#!/usr/bin/env python3
"""Run CP6 with the explicit shell/waistband/gusset topology contract."""
from __future__ import annotations

import run_garment_cad_pro_cp6 as pipeline
from wuxia_garment_oss.garments.trousers.topology import topology_receipt


def main() -> int:
    pipeline.topology_receipt = topology_receipt
    return pipeline.main()


if __name__ == "__main__":
    raise SystemExit(main())
