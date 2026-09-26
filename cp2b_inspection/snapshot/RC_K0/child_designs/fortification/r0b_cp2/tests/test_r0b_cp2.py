from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import unittest

from rcf_fortification_join.model import (
    JoinContractError,
    bounded_overlap,
    interface_sockets,
    section_plan,
    socket_alignment,
    validate_fixture,
)

ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    return json.loads((ROOT / "fixtures" / name).read_text(encoding="utf-8"))


class ContractTests(unittest.TestCase):
    def test_family_section_counts(self):
        expected = {
            "miter_join.fixture.json": ("MITER", 3),
            "bevel_join.fixture.json": ("BEVEL", 4),
            "profile_transition_join.fixture.json": ("PROFILE_TRANSITION", 3),
        }
        for fixture_name, (family, count) in expected.items():
            spec = validate_fixture(load(fixture_name))
            self.assertEqual(spec["family"], family)
            self.assertEqual(len(section_plan(spec)), count)

    def test_socket_alignment_exact(self):
        spec = validate_fixture(load("miter_join.fixture.json"))
        sockets = interface_sockets(spec)
        alignment = socket_alignment(spec, sockets)
        self.assertEqual(alignment["status"], "PASS")
        self.assertEqual(alignment["alignment_count"], 6)
        self.assertTrue(all(row["exact_numeric_equal"] for row in alignment["alignments"]))

    def test_bounded_overlap(self):
        for name in ("miter_join.fixture.json", "bevel_join.fixture.json", "profile_transition_join.fixture.json"):
            spec = validate_fixture(load(name))
            evidence = bounded_overlap(spec, section_plan(spec))
            self.assertEqual(evidence["status"], "PASS")
            self.assertLessEqual(evidence["incoming"]["ratio"], 0.5)
            self.assertLessEqual(evidence["outgoing"]["ratio"], 0.5)

    def test_transition_profiles_differ(self):
        spec = validate_fixture(load("profile_transition_join.fixture.json"))
        self.assertNotEqual(spec["incoming_profile"], spec["outgoing_profile"])
        sections = section_plan(spec)
        self.assertNotEqual(sections[0]["profile"]["profile_id"], sections[-1]["profile"]["profile_id"])

    def test_negative_contracts(self):
        fixture = load("miter_join.fixture.json")
        bad = copy.deepcopy(fixture); bad["overlap"]["incoming_m"] = 20.0
        with self.assertRaises(JoinContractError): validate_fixture(bad)
        bad = copy.deepcopy(fixture); bad["budget"]["max_sections"] = 2
        with self.assertRaises(JoinContractError): validate_fixture(bad)
        bad = copy.deepcopy(fixture); bad["outgoing"]["sockets"]["span"]["position_m"][0] += 0.1
        with self.assertRaises(JoinContractError): validate_fixture(bad)


class OutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = os.environ.get("RCF_R0B_CP2_OUTPUT_ROOT")
        if not root:
            raise unittest.SkipTest("RCF_R0B_CP2_OUTPUT_ROOT not set")
        cls.root = Path(root)

    def test_output_contracts(self):
        for family in ("miter", "bevel", "transition"):
            root = self.root / family / "reference"
            result = json.loads((root / "result.json").read_text())
            alignment = json.loads((root / "socket-alignment.json").read_text())
            overlap = json.loads((root / "bounded-overlap.json").read_text())
            parts = json.loads((root / "semantic-parts.json").read_text())
            self.assertEqual(result["status"], "SUCCEEDED")
            self.assertEqual(result["solid_count"], 5)
            self.assertEqual(result["socket_count"], 6)
            self.assertEqual(alignment["status"], "PASS")
            self.assertEqual(overlap["status"], "PASS")
            self.assertEqual(len(parts["parts"]), 5)

    def test_stored_copy_count(self):
        for family in ("miter", "bevel", "transition"):
            stored = json.loads((self.root / family / "reference/stored-copies.json").read_text())
            self.assertEqual(len(stored["copies"]), 10)


if __name__ == "__main__":
    unittest.main()
