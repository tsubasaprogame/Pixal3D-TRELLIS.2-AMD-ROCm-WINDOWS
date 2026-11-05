import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.acceptance.equirectangular as equirectangular
from tools.acceptance.equirectangular import sample_capture_positions


def capture_from_world(world):
    world = np.asarray(world, dtype=np.float64)
    return np.column_stack((world[..., 0], -world[..., 2], world[..., 1]))


def reference_sample(color_map, capture_positions):
    capture_positions = np.asarray(capture_positions, dtype=np.float64)
    world = np.column_stack((
        capture_positions[..., 0],
        capture_positions[..., 2],
        -capture_positions[..., 1],
    ))
    world /= np.linalg.norm(world, axis=-1, keepdims=True)
    longitude = np.arctan2(-world[..., 2], world[..., 0])
    latitude = np.arcsin(np.clip(world[..., 1], -1.0, 1.0))
    height, width = color_map.shape[:2]
    u = np.mod((longitude + np.pi) / (2 * np.pi) * width - 0.5, width)
    v = np.clip((np.pi / 2 - latitude) / np.pi * height - 0.5, 0, height - 1)
    u0 = np.floor(u).astype(np.int64) % width
    u1 = (u0 + 1) % width
    v0 = np.floor(v).astype(np.int64)
    v1 = np.minimum(v0 + 1, height - 1)
    fu = (u - np.floor(u))[..., None]
    fv = (v - np.floor(v))[..., None]
    return (
        (color_map[v0, u0] * (1 - fu) + color_map[v0, u1] * fu) * (1 - fv)
        + (color_map[v1, u0] * (1 - fu) + color_map[v1, u1] * fu) * fv
    )


class EquirectangularSamplingTests(unittest.TestCase):
    def setUp(self):
        height, width = 8, 16
        yy, xx = np.meshgrid(
            np.linspace(0, 1, height),
            np.linspace(0, 1, width),
            indexing="ij",
        )
        self.color_map = np.stack((xx, yy, np.full_like(xx, 0.25)), axis=-1)
        self.world = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.3, 0.4, -0.5],
            [-0.7, 0.2, 0.1],
        ])
        self.positions = capture_from_world(self.world)

    def test_matches_independent_bilinear_reference(self):
        expected = reference_sample(self.color_map, self.positions)
        actual = sample_capture_positions(
            torch.from_numpy(self.color_map).permute(2, 0, 1).float(),
            torch.from_numpy(self.positions).float(),
        ).numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_east_of_greenwich_appears_on_screen_right(self):
        color_map = np.zeros((2, 16, 3), dtype=np.float32)
        color_map[..., 0] = np.linspace(0.0, 1.0, 16, dtype=np.float32)
        # Camera at +X with +Y up has screen-right = -Z. East must move right.
        world = np.array([[1.0, 0.0, -0.25], [1.0, 0.0, 0.25]])
        positions = capture_from_world(world)
        actual = sample_capture_positions(
            torch.from_numpy(color_map).permute(2, 0, 1).float(),
            torch.from_numpy(positions).float(),
        ).numpy()
        self.assertGreater(actual[0, 0], actual[1, 0])

    def test_longitude_wraps_across_equirectangular_seam(self):
        epsilon = math.radians(0.1)
        world = np.array([
            [math.cos(epsilon), 0.0, math.pi - epsilon],
            [math.cos(epsilon), 0.0, -math.pi + epsilon],
        ])
        world[0, 2] = math.sin(math.pi - epsilon)
        world[1, 2] = math.sin(-math.pi + epsilon)
        positions = capture_from_world(world)
        expected = reference_sample(self.color_map, positions)
        actual = sample_capture_positions(
            torch.from_numpy(self.color_map).permute(2, 0, 1).float(),
            torch.from_numpy(positions).float(),
        ).numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_scale_does_not_change_surface_color(self):
        color_map = torch.from_numpy(self.color_map).permute(2, 0, 1).float()
        positions = torch.from_numpy(self.positions).float()
        unit = sample_capture_positions(color_map, positions)
        scaled = sample_capture_positions(color_map, positions * 0.5)
        torch.testing.assert_close(unit, scaled)

    def test_direct_map_attributes_set_expected_pbr_channels(self):
        direct_map_attributes = getattr(equirectangular, "direct_map_attributes", None)
        self.assertIsNotNone(direct_map_attributes, "direct-map PBR attributes are not implemented")
        color_map = torch.from_numpy(self.color_map).permute(2, 0, 1).float()
        positions = torch.from_numpy(self.positions).float()
        attrs = direct_map_attributes(color_map, positions)
        torch.testing.assert_close(attrs[:, :3], sample_capture_positions(color_map, positions))
        torch.testing.assert_close(attrs[:, 3], torch.zeros(len(positions)))
        torch.testing.assert_close(attrs[:, 4], torch.full((len(positions),), 0.85))
        torch.testing.assert_close(attrs[:, 5], torch.ones(len(positions)))

    def test_capture_to_glb_preserves_north_and_right_handedness(self):
        capture_to_glb = getattr(equirectangular, "capture_to_glb_array", None)
        self.assertIsNotNone(capture_to_glb, "capture-to-GLB conversion is not implemented")
        capture = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.5], [0.0, 0.5, 0.0]])
        expected = np.array([[1.0, 3.0, -2.0], [0.0, 0.5, 0.0], [0.0, 0.0, -0.5]])
        actual = capture_to_glb(capture)
        np.testing.assert_allclose(actual, expected)
        np.testing.assert_allclose(capture[0], [1.0, 2.0, 3.0])

    def test_raster_uv_is_exported_with_gltf_vertical_orientation(self):
        raster_uv_to_glb = getattr(equirectangular, "raster_uv_to_glb", None)
        self.assertIsNotNone(raster_uv_to_glb, "raster-to-GLB UV conversion is not implemented")
        raster = np.array([[0.2, 0.8], [0.7, 0.1]], dtype=np.float32)
        exported = raster_uv_to_glb(raster)
        np.testing.assert_allclose(exported, [[0.2, 0.2], [0.7, 0.9]])
        np.testing.assert_allclose(raster, [[0.2, 0.8], [0.7, 0.1]])

    @unittest.skipUnless(torch.cuda.is_available(), "GPU is required")
    def test_gpu_matches_cpu(self):
        color_map = torch.from_numpy(self.color_map).permute(2, 0, 1).float()
        positions = torch.from_numpy(self.positions).float()
        expected = sample_capture_positions(color_map, positions)
        actual = sample_capture_positions(color_map.cuda(), positions.cuda()).cpu()
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
