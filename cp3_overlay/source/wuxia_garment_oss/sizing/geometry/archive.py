"""Canonical NPZ writer with fixed member ordering and timestamps."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np


_FIXED_TIME = (1980, 1, 1, 0, 0, 0)


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, np.asarray(array), allow_pickle=False)
    return buffer.getvalue()


def write_canonical_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", _FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, _npy_bytes(arrays[name]))
    temporary.replace(path)
