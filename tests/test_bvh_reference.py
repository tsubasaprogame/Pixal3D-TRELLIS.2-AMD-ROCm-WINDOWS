import unittest

import numpy as np

try:
    from tests.bvh_reference import closest_point, ray_triangle
except ImportError:
    closest_point = None
    ray_triangle = None

TRIANGLE = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)


class BvhReferenceTests(unittest.TestCase):
    def require_reference(self):
        self.assertIsNotNone(closest_point, "bvh_reference is not implemented")
        self.assertIsNotNone(ray_triangle, "bvh_reference is not implemented")

    def test_plane_closest_point_and_uvw(self):
        self.require_reference()
        result = closest_point(np.array([0.25, 0.25, 1.0]), TRIANGLE[None, :, :])
        np.testing.assert_allclose(result.point, [0.25, 0.25, 0.0], atol=2e-5)
        np.testing.assert_allclose(result.uvw, [0.5, 0.25, 0.25], atol=2e-5)
        self.assertEqual(result.face_id, 0)
        self.assertAlmostEqual(result.distance, 1.0, places=5)

    def test_closest_point_outside_triangle_uses_edge(self):
        self.require_reference()
        result = closest_point(np.array([1.0, 1.0, 0.0]), TRIANGLE[None, :, :])
        np.testing.assert_allclose(result.point, [0.5, 0.5, 0.0], atol=2e-5)
        np.testing.assert_allclose(result.uvw, [0.0, 0.5, 0.5], atol=2e-5)

    def test_ray_triangle_hit_and_parallel_miss(self):
        self.require_reference()
        hit = ray_triangle(np.array([0.25, 0.25, 1.0]), np.array([0.0, 0.0, -2.0]), TRIANGLE)
        self.assertIsNotNone(hit)
        self.assertAlmostEqual(hit.depth, 0.5, places=5)
        np.testing.assert_allclose(hit.uvw, [0.5, 0.25, 0.25], atol=2e-5)
        miss = ray_triangle(np.array([0.25, 0.25, 1.0]), np.array([1.0, 0.0, 0.0]), TRIANGLE)
        self.assertIsNone(miss)


if __name__ == "__main__":
    unittest.main()
