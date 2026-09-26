from __future__ import annotations

import json
import os
from pathlib import Path
import unittest


class TerrainSpanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(os.environ["RCF_R0B_CP3_OUTPUT_ROOT"])
        cls.stepped = cls.root / "stepped/reference"
        cls.retaining = cls.root / "retaining/reference"

    def read(self, root: Path, name: str):
        return json.loads((root / name).read_text(encoding="utf-8"))

    def test_stepped_result(self):
        result = self.read(self.stepped, "result.json")
        self.assertEqual(result["family"], "TERRAIN_STEPPED")
        self.assertEqual(result["segment_count"], 3)
        self.assertEqual(result["unit_count"], 15)
        self.assertEqual(result["solid_count"], 15)
        self.assertEqual(result["part_count"], 5)
        self.assertEqual(result["socket_count"], 14)
        self.assertEqual(result["maximum_gap_m"], 0.0)
        self.assertEqual(result["maximum_embedment_depth_m"], 2.0)
        self.assertEqual(result["maximum_step_height_m"], 1.0)
        self.assertAlmostEqual(result["overall_centerline_grade"], 2.0 / 30.0, places=9)
        self.assertFalse(result["terrain_mutation"])

    def test_stepped_interface_and_contact(self):
        interface = self.read(self.stepped, "foundation-interface.json")
        contact = self.read(self.stepped, "contact-evidence.json")
        grade = self.read(self.stepped, "grade-evidence.json")
        self.assertEqual(interface["summary"]["interface_count"], 3)
        self.assertEqual(interface["summary"]["transition_count"], 2)
        self.assertEqual(interface["summary"]["total_contact_area_m2"], 240.0)
        self.assertEqual(contact["summary"]["sample_count"], 15)
        self.assertEqual(contact["summary"]["contact_ratio"], 1.0)
        self.assertEqual(grade["status"], "PASS")
        self.assertTrue(all(row["wall_walk_transition_required"] for row in interface["transitions"]))

    def test_retaining_result(self):
        result = self.read(self.retaining, "result.json")
        self.assertEqual(result["family"], "RETAINING")
        self.assertEqual(result["segment_count"], 1)
        self.assertEqual(result["unit_count"], 5)
        self.assertEqual(result["solid_count"], 5)
        self.assertEqual(result["socket_count"], 8)
        self.assertEqual(result["maximum_gap_m"], 0.0)
        self.assertEqual(result["maximum_embedment_depth_m"], 5.0)
        self.assertEqual(result["retained_height_m"], 3.0)
        self.assertEqual(result["overall_centerline_grade"], 0.0)
        self.assertFalse(result["terrain_mutation"])

    def test_retaining_interface_and_contact(self):
        interface = self.read(self.retaining, "foundation-interface.json")
        contact = self.read(self.retaining, "contact-evidence.json")
        grade = self.read(self.retaining, "grade-evidence.json")
        self.assertEqual(interface["summary"]["interface_count"], 3)
        self.assertEqual(interface["summary"]["total_contact_area_m2"], 264.0)
        self.assertEqual(contact["summary"]["sample_count"], 27)
        self.assertEqual(contact["summary"]["contact_count"], 27)
        self.assertAlmostEqual(grade["equivalent_cross_slope_ratio"], 1.0 / 3.0, places=9)

    def test_stored_copies_and_parts(self):
        stepped_stored = self.read(self.stepped, "stored-copies.json")
        retaining_stored = self.read(self.retaining, "stored-copies.json")
        self.assertEqual(len(stepped_stored["copies"]), 30)
        self.assertEqual(len(retaining_stored["copies"]), 10)
        for root in (self.stepped, self.retaining):
            parts = self.read(root, "semantic-parts.json")
            self.assertEqual([row["part_id"] for row in parts["parts"]], ["foundation", "wall_body", "wall_walk", "inner_parapet", "outer_parapet"])
            self.assertTrue(all(row["surface_coverage"] == "DEFERRED_TO_R0B_CP4" for row in parts["parts"]))

    def test_evidence_receipt_linkage(self):
        for root in (self.stepped, self.retaining):
            profile = self.read(root, "canonical-terrain-profile.json")
            plan = self.read(root, "terrain-span-plan.json")
            receipt = self.read(root, "cad-provider-receipt.json")
            self.assertEqual(profile["profile_digest"], plan["terrain_profile_digest"])
            self.assertEqual(profile["profile_digest"], receipt["terrain_profile_digest"])
            self.assertFalse(receipt["terrain_mutation"])


if __name__ == "__main__":
    unittest.main()
