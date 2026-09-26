from __future__ import annotations
import json, os, unittest
from pathlib import Path

class WallSpanTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.root=Path(os.environ['RCF_CP2_ACCEPTED_OUTPUT']); cls.result=json.loads((cls.root/'result.json').read_text()); cls.mesh=json.loads((cls.root/'neutral-mesh.json').read_text()); cls.parts=json.loads((cls.root/'semantic-parts.json').read_text()); cls.sockets=json.loads((cls.root/'sockets.json').read_text())
 def test_total_envelope(self): self.assertEqual(self.result['bounds_m'],{'min':[0.0,-2.0,-4.0],'max':[24.0,14.4,4.0]}); self.assertEqual(self.result['volume_m3'],2273.28)
 def test_semantic_parts(self): self.assertEqual([p['part_id'] for p in self.parts['parts']],['foundation','wall_body','wall_walk','inner_parapet','outer_parapet'])
 def test_mesh(self): self.assertEqual(len(self.mesh['vertices_m']),40); self.assertEqual(len(self.mesh['triangles']),60); self.assertTrue(all(0<=i<40 for t in self.mesh['triangles'] for i in t))
 def test_sockets(self): self.assertEqual(len(self.sockets['sockets']),10); self.assertEqual(len({s['socket_id'] for s in self.sockets['sockets']}),10)
 def test_no_cp3_claim(self): self.assertTrue(all(p['surface_coverage']=='DEFERRED_TO_R0A_CP3' for p in self.parts['parts']))
if __name__=='__main__': unittest.main()
