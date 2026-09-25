from __future__ import annotations

import inspect
import typing
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from rcf_fortification_cad import (
    Build123dProviderAdapter,
    CadProviderRequest,
    CadProviderResult,
    CadStatus,
    PROJECT_FRAME_ID,
    request_from_dict,
)
from rcf_fortification_cad.canonical import PROJECT_TO_PROVIDER_MATRIX, PROVIDER_TO_PROJECT_MATRIX, project_to_provider, provider_to_project


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cp1 = Path(os.environ["RCF_CP1_ROOT"])
        cls.cp0 = Path(os.environ["RCF_CP0_ROOT"])
        cls.request_data = json.loads((cls.cp1 / "fixtures/profile_extrusion.request.json").read_text())

    def test_request_round_trip_and_frame(self):
        request = request_from_dict(self.request_data)
        self.assertIsInstance(request, CadProviderRequest)
        self.assertEqual(request.to_dict(), self.request_data)
        self.assertEqual(request.frame, PROJECT_FRAME_ID)
        self.assertEqual(PROJECT_TO_PROVIDER_MATRIX[1][2], -1000.0)
        self.assertEqual(PROVIDER_TO_PROJECT_MATRIX[2][1], -0.001)
        point = (3.0, 4.0, 5.0)
        self.assertEqual(provider_to_project(project_to_provider(point)), point)

    def test_public_contract_has_no_upstream_types(self):
        import rcf_fortification_cad as public
        for name in public.__all__:
            obj = getattr(public, name)
            if not callable(obj):
                continue
            try:
                hints = typing.get_type_hints(obj)
            except (TypeError, NameError):
                hints = {}
            pending = list(hints.values())
            while pending:
                current = pending.pop()
                module = getattr(current, "__module__", "")
                self.assertFalse(module == "build123d" or module.startswith("build123d.") or module == "OCP" or module.startswith("OCP."))
                pending.extend(typing.get_args(current))

    def test_successful_execution_and_stored_copy(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out"
            result = Build123dProviderAdapter(self.cp0).execute(self.request_data, out)
            self.assertIsInstance(result, CadProviderResult)
            self.assertIs(result.status, CadStatus.SUCCEEDED)
            self.assertEqual(result.shape["volume_m3"], 48.0)
            self.assertEqual(result.shape["bounds_m"], {"min": [-1.0, 1.0, -3.0], "max": [3.0, 3.0, 3.0]})
            self.assertEqual({x["format"] for x in result.stored_copies}, {"STEP", "BREP"})
            self.assertEqual(sorted(p.name for p in out.iterdir()), ["cad-provider-receipt.json", "neutral-mesh.json", "result.json", "shape.brep", "shape.step"])
            receipt = json.loads((out / "cad-provider-receipt.json").read_text())
            self.assertFalse(receipt["contract"]["public_upstream_types_exposed"])
            self.assertFalse(receipt["runtime"]["global_install"])

    def test_rejections_publish_no_geometry(self):
        cases = {
            "invalid_units.request.json": "UNIT_CONTRACT_MISMATCH",
            "invalid_frame.request.json": "FRAME_CONTRACT_MISMATCH",
            "invalid_tolerance.request.json": "TOLERANCE_OUT_OF_DOMAIN",
            "runtime_mismatch.request.json": "CAD_RUNTIME_VERSION_MISMATCH",
            "budget_exceeded.request.json": "GEOMETRY_BUDGET_EXCEEDED",
        }
        with tempfile.TemporaryDirectory() as td:
            for index, (name, code) in enumerate(cases.items()):
                request = json.loads((self.cp1 / "fixtures" / name).read_text())
                out = Path(td) / str(index)
                result = Build123dProviderAdapter(self.cp0).execute(request, out)
                self.assertIs(result.status, CadStatus.REJECTED)
                self.assertEqual(result.failure["code"], code)
                self.assertEqual([p.name for p in out.iterdir()], ["result.json"])


if __name__ == "__main__":
    unittest.main()
