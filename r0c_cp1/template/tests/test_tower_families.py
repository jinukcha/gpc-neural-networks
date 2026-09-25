from __future__ import annotations

import json
import os
from pathlib import Path
import unittest


class TowerFamilyAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(os.environ["RCF_TOWER_REFERENCE_ROOT"])
        cls.docs = {}
        for family in ("round", "square", "polygonal"):
            root = cls.root / family
            cls.docs[family] = {
                "result": json.loads((root / "result.json").read_text()),
                "plan": json.loads((root / "tower-plan.json").read_text()),
                "parts": json.loads((root / "semantic-parts.json").read_text()),
                "sockets": json.loads((root / "sockets.json").read_text()),
                "bounds": json.loads((root / "bounds-attachment.json").read_text()),
                "foundation": json.loads((root / "foundation-interface.json").read_text()),
                "stored": json.loads((root / "stored-copies.json").read_text()),
            }

    def test_all_families_succeeded(self):
        self.assertEqual({doc["result"]["family"] for doc in self.docs.values()}, {"ROUND", "SQUARE", "POLYGONAL"})
        self.assertTrue(all(doc["result"]["status"] == "SUCCEEDED" for doc in self.docs.values()))

    def test_semantic_parts(self):
        expected = ["foundation", "tower_body", "tower_crown"]
        for doc in self.docs.values():
            self.assertEqual([part["part_id"] for part in doc["parts"]["parts"]], expected)
            self.assertTrue(all(part["surface_coverage"] == "DEFERRED_TO_R0C_CP4" for part in doc["parts"]["parts"]))

    def test_socket_contract(self):
        expected = [
            "span_in", "span_out", "wall_walk_in", "wall_walk_out",
            "foundation_in", "foundation_out", "foundation_center",
            "foundation_outside", "foundation_inside", "roof_socket",
        ]
        for doc in self.docs.values():
            sockets = doc["sockets"]["sockets"]
            self.assertEqual([socket["socket_id"] for socket in sockets], expected)
            self.assertEqual(len({socket["socket_id"] for socket in sockets}), 10)
            self.assertTrue(all(socket["frame"]["determinant"] == 1.0 for socket in sockets))

    def test_foundation_contact(self):
        for doc in self.docs.values():
            foundation = doc["foundation"]
            self.assertEqual(foundation["status"], "PASS")
            self.assertEqual(foundation["contact_ratio"], 1.0)
            self.assertEqual(foundation["maximum_gap_m"], 0.0)
            self.assertFalse(foundation["terrain_mutation"])

    def test_attachment_and_projection(self):
        for doc in self.docs.values():
            evidence = doc["bounds"]
            self.assertEqual(evidence["attachment_intersection_count"], 2)
            self.assertGreater(evidence["attachment_width_m"], 0.0)
            self.assertLessEqual(evidence["body_outside_projection_m"], evidence["maximum_body_projection_m"])
            self.assertEqual(evidence["bounds_status"], "PASS")

    def test_stored_copies(self):
        for doc in self.docs.values():
            copies = doc["stored"]["copies"]
            self.assertEqual(len(copies), 6)
            self.assertEqual({copy["format"] for copy in copies}, {"STEP", "BREP"})
            self.assertEqual({copy["part_id"] for copy in copies}, {"foundation", "tower_body", "tower_crown"})

    def test_round_approximation_contract(self):
        round_doc = self.docs["round"]
        self.assertEqual(round_doc["plan"]["construction_route"], "POLYGONAL_APPROXIMATION_32")
        self.assertEqual(round_doc["plan"]["side_count"], 32)
        tess = json.loads((self.root / "round/fixed-tessellation.json").read_text())
        self.assertLessEqual(tess["round_radial_chord_error_m"], tess["linear_deflection_m"])

    def test_no_deferred_scope_claims(self):
        for doc in self.docs.values():
            plan = doc["plan"]
            self.assertEqual(plan["tower_join_realization"], "DEFERRED_TO_R0C_CP2")
            self.assertEqual(plan["battlement_realization"], "DEFERRED_TO_R0C_CP3")
            self.assertEqual(plan["surface_source_coverage"], "DEFERRED_TO_R0C_CP4")


if __name__ == "__main__":
    unittest.main()
