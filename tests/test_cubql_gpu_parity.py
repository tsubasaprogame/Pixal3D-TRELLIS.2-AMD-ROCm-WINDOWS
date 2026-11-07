import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np
import torch

from tests.bvh_reference import closest_point, ray_trace

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "data/bvh-golden/v1"
RUNTIME = ROOT / "data/runtime/experimental-runtime"
TOLERANCE = 2e-5


def load_experimental_backend():
    """Load only the new extension, without importing the legacy package."""
    pyd = next((RUNTIME / "cumesh").glob("_bvh*.pyd"))
    package = types.ModuleType("cumesh")
    package.__path__ = [str(RUNTIME / "cumesh")]
    package.__package__ = "cumesh"
    sys.modules["cumesh"] = package
    spec = importlib.util.spec_from_file_location("cumesh._bvh", pyd)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(torch.cuda.is_available(), "HIP GPU is required")
class CuBQLGpuParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = load_experimental_backend()
        cls.inputs = np.load(GOLDEN / "inputs.npz")
        cls.outputs = np.load(GOLDEN / "outputs.npz")
        cls.meshes = {
            name: cls.backend.create_cuBVH(
                cls.inputs[f"{name}_vertices"], cls.inputs[f"{name}_faces"]
            )
            for name in ("cube", "open", "degenerate")
        }
        cls.points = torch.from_numpy(cls.inputs["points"]).cuda()
        cls.ray_origins = torch.from_numpy(cls.inputs["ray_origins"]).cuda()
        cls.ray_directions = torch.from_numpy(cls.inputs["ray_directions"]).cuda()

    def assert_close(self, actual, expected):
        np.testing.assert_allclose(actual, expected, rtol=TOLERANCE, atol=TOLERANCE)

    def ray_trace(self, mesh, origins=None, directions=None):
        origins = self.ray_origins if origins is None else origins
        directions = self.ray_directions if directions is None else directions
        count = origins.shape[0]
        positions = torch.empty((count, 3), dtype=torch.float32, device=origins.device)
        face_id = torch.empty(count, dtype=torch.int64, device=origins.device)
        depth = torch.empty(count, dtype=torch.float32, device=origins.device)
        mesh.ray_trace(origins, directions, positions, face_id, depth)
        return tuple(value.cpu().numpy() for value in (positions, face_id, depth))

    def unsigned_distance(self, mesh, points=None, return_uvw=True):
        points = self.points if points is None else points
        count = points.shape[0]
        distances = torch.empty(count, dtype=torch.float32, device=points.device)
        face_id = torch.empty(count, dtype=torch.int64, device=points.device)
        uvw = torch.empty((count, 3), dtype=torch.float32, device=points.device) if return_uvw else None
        mesh.unsigned_distance(points, distances, face_id, uvw)
        result = [distances.cpu().numpy(), face_id.cpu().numpy()]
        result.append(None if uvw is None else uvw.cpu().numpy())
        return tuple(result)

    def signed_distance(self, mesh, mode, points=None):
        points = self.points if points is None else points
        count = points.shape[0]
        distances = torch.empty(count, dtype=torch.float32, device=points.device)
        face_id = torch.empty(count, dtype=torch.int64, device=points.device)
        uvw = torch.empty((count, 3), dtype=torch.float32, device=points.device)
        mesh.signed_distance(points, distances, face_id, uvw, mode)
        return tuple(value.cpu().numpy() for value in (distances, face_id, uvw))

    def assert_uvw_reconstructs_closest_points(self, mesh_name, distances, face_ids, uvw):
        triangles = self.inputs[f"{mesh_name}_vertices"][self.inputs[f"{mesh_name}_faces"]]
        for index, point in enumerate(self.inputs["points"]):
            expected = closest_point(point, triangles)
            self.assertAlmostEqual(float(distances[index]), expected.distance, delta=TOLERANCE)
            face_id = int(face_ids[index])
            self.assertGreaterEqual(face_id, 0)
            self.assertLess(face_id, len(triangles))
            self.assert_close(uvw[index].sum(), 1.0)
            self.assertGreaterEqual(float(uvw[index].min()), -TOLERANCE)
            reconstructed = uvw[index] @ triangles[face_id]
            reconstructed_distance = float(np.linalg.norm(reconstructed - point))
            self.assertAlmostEqual(reconstructed_distance, expected.distance, delta=TOLERANCE)

    def test_rays_match_golden_and_independent_reference(self):
        for mesh_name in ("cube", "open"):
            positions, face_ids, depths = self.ray_trace(self.meshes[mesh_name])
            self.assert_close(positions, self.outputs[f"{mesh_name}_ray_positions"])
            self.assert_close(depths, self.outputs[f"{mesh_name}_ray_depth"])
            triangles = self.inputs[f"{mesh_name}_vertices"][self.inputs[f"{mesh_name}_faces"]]
            for index, (origin, direction) in enumerate(zip(
                self.inputs["ray_origins"], self.inputs["ray_directions"]
            )):
                expected_position, expected_face, expected_depth = ray_trace(origin, direction, triangles)
                if expected_face == -1:
                    self.assertEqual(int(face_ids[index]), -1)
                else:
                    self.assertGreaterEqual(int(face_ids[index]), 0)
                self.assertAlmostEqual(float(depths[index]), expected_depth, delta=TOLERANCE)
                self.assert_close(positions[index], expected_position)

        miss_positions, miss_faces, miss_depths = self.ray_trace(
            self.meshes["cube"],
            torch.tensor([[2.0, 2.0, 1.0]], device="cuda"),
            torch.tensor([[0.0, 0.0, -1.0]], device="cuda"),
        )
        self.assertEqual(int(miss_faces[0]), -1)
        self.assertEqual(float(miss_depths[0]), 1000.0)
        self.assert_close(miss_positions[0], [2.0, 2.0, -999.0])

    def test_degenerate_face_does_not_create_false_ray_hit(self):
        positions, face_ids, depths = self.ray_trace(
            self.meshes["degenerate"],
            torch.tensor([[0.0, 0.0, 0.0]], device="cuda"),
            torch.tensor([[1.0, 0.0, 0.0]], device="cuda"),
        )
        self.assertGreaterEqual(int(face_ids[0]), 0)
        self.assertNotEqual(int(face_ids[0]), 12)
        self.assertAlmostEqual(float(depths[0]), 1.0, delta=TOLERANCE)
        self.assert_close(positions[0], [1.0, 0.0, 0.0])

    def test_unsigned_distance_and_uvw_match_golden(self):
        for mesh_name, mesh in self.meshes.items():
            with self.subTest(mesh=mesh_name):
                distances, face_ids, uvw = self.unsigned_distance(mesh)
                self.assert_close(distances, self.outputs[f"{mesh_name}_udf"])
                self.assert_uvw_reconstructs_closest_points(mesh_name, distances, face_ids, uvw)
                without_uvw = self.unsigned_distance(mesh, return_uvw=False)
                self.assertIsNone(without_uvw[2])
                self.assert_close(without_uvw[0], distances)

    def test_signed_distance_modes_match_golden_and_reference_magnitude(self):
        mode_ids = {"watertight": 0, "raystab": 1}
        for mesh_name, mesh in self.meshes.items():
            triangles = self.inputs[f"{mesh_name}_vertices"][self.inputs[f"{mesh_name}_faces"]]
            for mode, mode_id in mode_ids.items():
                with self.subTest(mesh=mesh_name, mode=mode):
                    distances, face_ids, uvw = self.signed_distance(mesh, mode_id)
                    expected = self.outputs[f"{mesh_name}_{mode}"]
                    self.assert_close(distances, expected)
                    outside_surface_band = np.abs(expected) > TOLERANCE
                    self.assertTrue(np.all(np.sign(distances[outside_surface_band]) == np.sign(expected[outside_surface_band])))
                    self.assert_uvw_reconstructs_closest_points(mesh_name, np.abs(distances), face_ids, uvw)
                    for index, point in enumerate(self.inputs["points"]):
                        reference = closest_point(point, triangles)
                        self.assertAlmostEqual(abs(float(distances[index])), reference.distance, delta=TOLERANCE)

    def test_empty_queries_return_empty_outputs(self):
        empty = torch.empty((0, 3), dtype=torch.float32, device="cuda")
        empty_directions = torch.empty((0, 3), dtype=torch.float32, device="cuda")
        for mesh in (self.meshes["cube"],):
            positions, face_ids, depths = self.ray_trace(mesh, empty, empty_directions)
            self.assertEqual(positions.shape, (0, 3))
            self.assertEqual(face_ids.shape, (0,))
            self.assertEqual(depths.shape, (0,))
            distances, face_ids, uvw = self.unsigned_distance(mesh, empty)
            self.assertEqual(distances.shape, (0,))
            self.assertEqual(face_ids.shape, (0,))
            self.assertEqual(uvw.shape, (0, 3))
            distances, face_ids, uvw = self.signed_distance(mesh, 0, empty)
            self.assertEqual(distances.shape, (0,))
            self.assertEqual(face_ids.shape, (0,))
            self.assertEqual(uvw.shape, (0, 3))

    def test_query_tensor_validation_and_sdf_mode(self):
        mesh = self.meshes["cube"]
        origins = self.ray_origins
        directions = self.ray_directions
        count = origins.shape[0]
        positions = torch.empty((count, 3), dtype=torch.float32, device="cuda")
        face_ids = torch.empty(count, dtype=torch.int64, device="cuda")
        depths = torch.empty(count, dtype=torch.float32, device="cuda")
        with self.assertRaisesRegex(RuntimeError, "float32"):
            mesh.ray_trace(origins.double(), directions, positions, face_ids, depths)
        with self.assertRaisesRegex(RuntimeError, "contiguous"):
            mesh.ray_trace(origins, directions.t().contiguous().t(), positions, face_ids, depths)
        with self.assertRaisesRegex(RuntimeError, "same shape"):
            mesh.ray_trace(origins, directions[:1], positions, face_ids, depths)
        with self.assertRaisesRegex(RuntimeError, "face_id"):
            mesh.ray_trace(origins, directions, positions, face_ids[:1], depths)
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            mesh.ray_trace(origins, directions, positions.cpu(), face_ids, depths)
        distances = torch.empty(count, dtype=torch.float32, device="cuda")
        uvw = torch.empty((count, 3), dtype=torch.float32, device="cuda")
        with self.assertRaisesRegex(RuntimeError, "mode"):
            mesh.signed_distance(self.points, distances, face_ids, uvw, 2)

    def test_public_wrapper_uses_new_backend(self):
        source = (ROOT / "scripts/native/cubql-bvh/cumesh/bvh.py").read_text(encoding="utf-8")
        self.assertIn("from . import _bvh as _backend", source)
        self.assertNotIn("_cubvh", source)

        from cumesh.bvh import cuBVH
        wrapped = cuBVH(self.inputs["cube_vertices"], self.inputs["cube_faces"])
        positions, face_ids, depths = wrapped.ray_trace(self.ray_origins, self.ray_directions)
        self.assert_close(positions.cpu().numpy(), self.outputs["cube_ray_positions"])
        self.assert_close(depths.cpu().numpy(), self.outputs["cube_ray_depth"])
        distances, face_ids, uvw = wrapped.unsigned_distance(self.points, return_uvw=True)
        self.assert_close(distances.cpu().numpy(), self.outputs["cube_udf"])
        self.assert_uvw_reconstructs_closest_points(
            "cube", distances.cpu().numpy(), face_ids.cpu().numpy(), uvw.cpu().numpy()
        )
        signed, _, _ = wrapped.signed_distance(self.points, return_uvw=True, mode="watertight")
        self.assert_close(signed.cpu().numpy(), self.outputs["cube_watertight"])

    def test_large_100001_query_batch(self):
        count = 100001
        points_np = np.resize(self.inputs["points"], (count, 3)).astype(np.float32)
        origins_np = np.resize(self.inputs["ray_origins"], (count, 3)).astype(np.float32)
        directions_np = np.resize(self.inputs["ray_directions"], (count, 3)).astype(np.float32)
        points = torch.from_numpy(points_np).cuda()
        origins = torch.from_numpy(origins_np).cuda()
        directions = torch.from_numpy(directions_np).cuda()

        positions, face_ids, depths = self.ray_trace(self.meshes["cube"], origins, directions)
        self.assert_close(positions, np.resize(self.outputs["cube_ray_positions"], (count, 3)))
        self.assert_close(depths, np.resize(self.outputs["cube_ray_depth"], count))
        distances, face_ids, uvw = self.unsigned_distance(self.meshes["cube"], points)
        self.assert_close(distances, np.resize(self.outputs["cube_udf"], count))
        self.assert_close(uvw.sum(axis=1), np.ones(count, dtype=np.float32))
        self.assertGreaterEqual(float(uvw.min()), -TOLERANCE)
        for mode_id, mode in ((0, "watertight"), (1, "raystab")):
            signed, signed_faces, signed_uvw = self.signed_distance(self.meshes["cube"], mode_id, points)
            self.assert_close(signed, np.resize(self.outputs[f"cube_{mode}"], count))
            self.assert_close(signed_uvw.sum(axis=1), np.ones(count, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
