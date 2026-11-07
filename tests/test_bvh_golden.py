import json
import unittest
from pathlib import Path

import numpy as np

from tests.bvh_reference import closest_point, ray_trace

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "data/bvh-golden/v1"


class BvhGoldenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = np.load(GOLDEN / "inputs.npz")
        cls.outputs = np.load(GOLDEN / "outputs.npz")

    def test_old_backend_and_exact_miss_contract(self):
        metadata = json.loads((GOLDEN / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["backend"], "cumesh._cubvh")
        self.assertEqual(metadata["gpu"], "AMD Radeon RX 9070 XT")
        self.assertEqual(int(self.outputs["cube_ray_face_id"][1]), -1)
        self.assertEqual(float(self.outputs["cube_ray_depth"][1]), 1000.0)

    def test_golden_rays_match_independent_reference(self):
        # The legacy backend incorrectly reports depth 0 on the degenerate face at the origin.
        # The approved replacement contract explicitly excludes that zero-area candidate.
        for mesh in ("cube", "open"):
            triangles = self.inputs[f"{mesh}_vertices"][self.inputs[f"{mesh}_faces"]]
            for index, (origin, direction) in enumerate(zip(self.inputs["ray_origins"], self.inputs["ray_directions"])):
                expected_position, expected_face, expected_depth = ray_trace(origin, direction, triangles)
                actual_face = int(self.outputs[f"{mesh}_ray_face_id"][index])
                if expected_face == -1:
                    self.assertEqual(actual_face, -1)
                else:
                    self.assertGreaterEqual(actual_face, 0)
                self.assertAlmostEqual(float(self.outputs[f"{mesh}_ray_depth"][index]), expected_depth, places=5)
                np.testing.assert_allclose(
                    self.outputs[f"{mesh}_ray_positions"][index], expected_position, rtol=2e-5, atol=2e-5,
                )

    def test_golden_distances_uvw_and_reconstruction_match_reference(self):
        for mesh in ("cube", "open", "degenerate"):
            triangles = self.inputs[f"{mesh}_vertices"][self.inputs[f"{mesh}_faces"]]
            for index, point in enumerate(self.inputs["points"]):
                expected = closest_point(point, triangles)
                self.assertAlmostEqual(float(self.outputs[f"{mesh}_udf"][index]), expected.distance, places=5)
                face_id = int(self.outputs[f"{mesh}_udf_uvw_face_id"][index])
                uvw = self.outputs[f"{mesh}_uvw"][index]
                self.assertAlmostEqual(float(uvw.sum()), 1.0, places=5)
                self.assertGreaterEqual(float(uvw.min()), -2e-5)
                reconstructed = uvw @ triangles[face_id]
                reconstructed_distance = float(np.linalg.norm(reconstructed - point))
                self.assertAlmostEqual(reconstructed_distance, expected.distance, places=5)


if __name__ == "__main__":
    unittest.main()
