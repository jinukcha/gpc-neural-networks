from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import unittest

from rcf_fortification_path_span.model import PathContractError, canonicalize_centerline, socket_plan, validate_fixture

ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    return json.loads((ROOT / "fixtures" / name).read_text(encoding="utf-8"))


class ContractTests(unittest.TestCase):
    def test_straight_canonical(self):
        centerline, frames = canonicalize_centerline(load("straight_path_span.fixture.json"))
        self.assertEqual(centerline["kind"], "LINE")
        self.assertEqual(centerline["length_m"], 24.0)
        self.assertEqual(centerline["segment_count"], 1)
        self.assertEqual(centerline["bounds_m"], {"min": [0.0, 0.0, 0.0], "max": [24.0, 0.0, 0.0]})
        self.assertEqual(frames["frames"][0]["tangent"], [1.0, 0.0, 0.0])
        self.assertEqual(frames["frames"][0]["inside"], [0.0, 0.0, 1.0])
        self.assertEqual(frames["frames"][0]["outside"], [0.0, 0.0, -1.0])

    def test_curved_sampling_and_bounds(self):
        centerline, frames = canonicalize_centerline(load("curved_path_span.fixture.json"))
        self.assertEqual(centerline["kind"], "CIRCULAR_ARC")
        self.assertEqual(centerline["segment_count"], 6)
        self.assertLessEqual(centerline["maximum_observed_chord_error_m"], 0.05)
        self.assertAlmostEqual(centerline["length_m"], 48.0 * math.radians(30.0), places=8)
        self.assertEqual(centerline["bounds_m"]["min"], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(centerline["bounds_m"]["max"][0], 24.0, places=8)
        self.assertAlmostEqual(centerline["bounds_m"]["max"][2], 6.430780618, places=8)
        self.assertEqual(len(frames["frames"]), 7)

    def test_frames_orthonormal_and_right_handed(self):
        _, frames = canonicalize_centerline(load("curved_path_span.fixture.json"))
        for frame in frames["frames"]:
            t, up, inside, outside = frame["tangent"], frame["up"], frame["inside"], frame["outside"]
            self.assertAlmostEqual(sum(x*x for x in t), 1.0, places=8)
            self.assertAlmostEqual(sum(t[i]*up[i] for i in range(3)), 0.0, places=8)
            self.assertEqual([round(-x, 9) for x in inside], outside)
            self.assertEqual(frame["orientation_determinant"], 1.0)

    def test_socket_stability(self):
        centerline, _ = canonicalize_centerline(load("curved_path_span.fixture.json"))
        sockets = socket_plan(centerline)
        self.assertEqual(len(sockets), 10)
        self.assertEqual(len({s["socket_id"] for s in sockets}), 10)
        self.assertEqual(sockets[0]["socket_id"], "span_start")
        self.assertEqual(sockets[1]["socket_id"], "span_end")

    def test_key_order_does_not_change_digest(self):
        fixture = load("curved_path_span.fixture.json")
        reversed_fixture = dict(reversed(list(fixture.items())))
        a, _ = canonicalize_centerline(fixture)
        b, _ = canonicalize_centerline(reversed_fixture)
        self.assertEqual(a, b)

    def test_negative_contracts(self):
        fixture = load("straight_path_span.fixture.json")
        bad = copy.deepcopy(fixture); bad["centerline"]["end_m"] = [0.0, 0.0, 0.0]
        with self.assertRaises(PathContractError): validate_fixture(bad)
        bad = copy.deepcopy(fixture); bad["centerline"]["end_m"] = [24.0, 1.0, 0.0]
        with self.assertRaises(PathContractError): validate_fixture(bad)
        arc = load("curved_path_span.fixture.json")
        bad = copy.deepcopy(arc); bad["centerline"]["radius_m"] = 3.0
        with self.assertRaises(PathContractError): validate_fixture(bad)
        bad = copy.deepcopy(arc); bad["centerline"]["sweep_angle_deg"] = 40.0
        with self.assertRaises(PathContractError): validate_fixture(bad)


class OutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = os.environ.get("RCF_R0B_CP1_OUTPUT_ROOT")
        if not root:
            raise unittest.SkipTest("RCF_R0B_CP1_OUTPUT_ROOT not set")
        cls.root = Path(root)

    def test_output_contract(self):
        for family in ("straight", "curved"):
            result = json.loads((self.root / family / "reference" / "result.json").read_text())
            centerline = json.loads((self.root / family / "reference" / "canonical-centerline.json").read_text())
            frames = json.loads((self.root / family / "reference" / "local-frames.json").read_text())
            self.assertEqual(result["status"], "SUCCEEDED")
            self.assertEqual(result["solid_count"], 5)
            self.assertEqual(result["socket_count"], 10)
            self.assertEqual(centerline["canonical_digest"], frames["canonical_centerline_digest"])
            self.assertGreater(result["vertex_count"], 0)
            self.assertGreater(result["triangle_count"], 0)


if __name__ == "__main__":
    unittest.main()
