from __future__ import annotations

import json
import os
from pathlib import Path
import unittest


class MixedSpanCloseoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(os.environ["RCF_CP4_ACCEPTED_OUTPUT"])
        cls.cp4_root = Path(os.environ["RCF_CP4_ROOT"])
        cls.result = json.loads((cls.root / "result.json").read_text(encoding="utf-8"))
        cls.segmentation = json.loads((cls.root / "segmentation-plan.json").read_text(encoding="utf-8"))
        cls.coverage = json.loads((cls.root / "source-coverage-map.json").read_text(encoding="utf-8"))
        cls.alignment = json.loads((cls.root / "socket-alignment.json").read_text(encoding="utf-8"))
        cls.module_index = json.loads((cls.root / "module-index.json").read_text(encoding="utf-8"))
        cls.mesh = json.loads((cls.root / "neutral-mesh.json").read_text(encoding="utf-8"))
        cls.qualification = json.loads((cls.cp4_root / "reports/qualification.json").read_text(encoding="utf-8"))

    def test_result_counts(self) -> None:
        self.assertEqual(self.result["status"], "SUCCEEDED")
        self.assertEqual(self.result["module_count"], 7)
        self.assertEqual(self.result["span_segment_count"], 4)
        self.assertEqual(self.result["join_count"], 3)
        self.assertEqual(self.result["connection_count"], 6)
        self.assertEqual(self.result["vertex_count"], 544)
        self.assertEqual(self.result["triangle_count"], 908)

    def test_segmentation_order_and_continuity(self) -> None:
        self.assertEqual(
            self.segmentation["chain_order"],
            ["span/straight", "join/miter", "span/curved", "join/bevel", "span/stepped", "span/retaining", "join/transition"],
        )
        spans = self.segmentation["span_segments"]
        for left, right in zip(spans, spans[1:]):
            self.assertAlmostEqual(left["station"]["end_m"], right["station"]["start_m"], places=9)
        self.assertFalse(self.segmentation["input_array_order_is_identity"])

    def test_socket_alignment(self) -> None:
        self.assertEqual(self.alignment["status"], "PASS")
        self.assertEqual(self.alignment["alignment_count"], 6)
        self.assertTrue(all(row["status"] == "PASS" for row in self.alignment["alignments"]))
        self.assertTrue(all(row["errors"]["position_m"] <= 1e-6 for row in self.alignment["alignments"]))

    def test_source_coverage(self) -> None:
        summary = self.coverage["summary"]
        self.assertEqual(summary["mesh_triangle_count"], 908)
        self.assertEqual(summary["covered_triangle_count"], 908)
        self.assertEqual(summary["unique_triangle_count"], 908)
        self.assertEqual(summary["gap_count"], 0)
        self.assertEqual(summary["overlap_count"], 0)
        indices = [index for row in self.coverage["surfaces"] for index in row["global_triangle_indices"]]
        self.assertEqual(len(indices), len(set(indices)))
        self.assertEqual(set(indices), set(range(908)))

    def test_module_ranges(self) -> None:
        ranges = self.mesh["module_ranges"]
        self.assertEqual(len(ranges), 7)
        self.assertEqual(sum(row["vertex_count"] for row in ranges), 544)
        self.assertEqual(sum(row["triangle_count"] for row in ranges), 908)
        self.assertEqual([row["module_id"] for row in ranges], self.segmentation["chain_order"])

    def test_exact_source_refs(self) -> None:
        self.assertEqual(self.module_index["module_count"], 7)
        for module in self.module_index["modules"]:
            self.assertTrue(module["source_refs"]["result"]["sha256"].startswith("sha256:"))
            self.assertTrue(module["source_refs"]["neutral_mesh"]["sha256"].startswith("sha256:"))
            self.assertTrue(module["source_refs"]["provider_receipt"]["sha256"].startswith("sha256:"))

    def test_clean_replay_and_source_replay(self) -> None:
        self.assertEqual(self.qualification["clean_replay"]["status"], "PASS")
        self.assertTrue(self.qualification["clean_replay"]["byte_identical"])
        self.assertEqual(self.qualification["source_replay"]["case_count"], 7)
        self.assertTrue(all(row["byte_identical"] for row in self.qualification["source_replay"]["cases"]))

    def test_negative_gates(self) -> None:
        gates = self.qualification["negative_gates"]
        self.assertEqual(gates["case_count"], 10)
        self.assertTrue(gates["all_rejected"])
        self.assertTrue(gates["no_accepted_negative_output"])


if __name__ == "__main__":
    unittest.main()
