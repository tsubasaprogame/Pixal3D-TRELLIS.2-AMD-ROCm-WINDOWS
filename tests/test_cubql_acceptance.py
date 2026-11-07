import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.acceptance.cubql_acceptance import capture_inputs, compare_backend_results, reconstruct_positions
from tools.acceptance.comfy_capture import comfy_argv, configure_comfy_args


class AcceptanceHarnessTests(unittest.TestCase):
    def test_capture_preserves_mesh_uv_and_sparse_voxels_on_cpu(self):
        mesh = SimpleNamespace(
            vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
            faces=np.array([[0, 1, 2]], dtype=np.int32),
            vertex_normals=np.array([[0, 0, 1]] * 3, dtype=np.float32),
            visual=SimpleNamespace(uv=np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)),
        )
        voxels = SimpleNamespace(
            feats=torch.arange(12, dtype=torch.float32).reshape(2, 6),
            coords=torch.tensor([[0, 1, 2, 3], [0, 2, 3, 4]], dtype=torch.int32),
            shape=torch.Size([1, 6]),
            spatial_shape=torch.Size([4, 4, 4]),
        )

        captured = capture_inputs(mesh, voxels, resolution=1024, texture_size=2048)

        self.assertEqual(captured["vertices"].shape, (3, 3))
        self.assertEqual(captured["faces"].shape, (1, 3))
        self.assertEqual(captured["uv"].shape, (3, 2))
        self.assertEqual(captured["voxel_feats"].shape, (2, 6))
        self.assertEqual(captured["voxel_shape"], [1, 6])
        self.assertEqual(captured["voxel_spatial_shape"], [4, 4, 4])
        self.assertEqual(captured["resolution"], 1024)
        self.assertEqual(captured["texture_size"], 2048)
        for key in ("vertices", "faces", "normals", "uv", "voxel_feats", "voxel_coords"):
            self.assertEqual(captured[key].device.type, "cpu")

    def test_reconstructs_positions_from_face_ids_and_barycentric_coordinates(self):
        vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        faces = torch.tensor([[0, 1, 2]])
        face_ids = torch.tensor([0, 0])
        uvw = torch.tensor([[0.25, 0.25, 0.5], [1., 0., 0.]])

        positions = reconstruct_positions(vertices, faces, face_ids, uvw)

        torch.testing.assert_close(positions, torch.tensor([[0.25, 0.5, 0.], [0., 0., 0.]]))

    def test_comfy_wrapper_removes_duplicate_script_argument(self):
        argv = ["capture.py", "ComfyUI/main.py", "--port", "8194", "--listen", "127.0.0.1"]

        result = comfy_argv(argv, "C:/runtime/ComfyUI/main.py")

        self.assertEqual(result, ["C:/runtime/ComfyUI/main.py", "--port", "8194", "--listen", "127.0.0.1"])

    def test_comfy_args_are_enabled_before_pipeline_import(self):
        options = SimpleNamespace(calls=[])
        options.enable_args_parsing = lambda enabled=True: options.calls.append(enabled)
        argv = ["capture.py", "ComfyUI/main.py", "--port", "8194"]

        result = configure_comfy_args(argv, "C:/runtime/ComfyUI/main.py", options)

        self.assertEqual(options.calls, [True])
        self.assertEqual(result, ["C:/runtime/ComfyUI/main.py", "--port", "8194"])

    def test_capture_hook_patches_relative_pipeline_import(self):
        import importlib
        import tempfile
        from tools.acceptance import comfy_capture

        install = getattr(comfy_capture, "install_capture_hook", None)
        self.assertIsNotNone(install)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "packed" / "trellis2" / "pipelines"
            package.mkdir(parents=True)
            for directory in (root / "packed", root / "packed" / "trellis2", package):
                (directory / "__init__.py").write_text("", encoding="utf-8")
            (package / "trellis2_image_to_3d.py").write_text(
                "class Trellis2ImageTo3DPipeline:\n"
                "    def postprocess_mesh(self):\n"
                "        return 'original'\n",
                encoding="utf-8",
            )
            sys.path.insert(0, str(root))

            def capture(self):
                return "captured"

            restore = install(capture)
            try:
                module = importlib.import_module("packed.trellis2.pipelines.trellis2_image_to_3d")
                self.assertEqual(module.Trellis2ImageTo3DPipeline().postprocess_mesh(), "captured")
            finally:
                restore()
                sys.path.remove(str(root))
                for name in list(sys.modules):
                    if name == "packed" or name.startswith("packed."):
                        del sys.modules[name]

    def test_compare_cli_runs_in_isolated_python(self):
        import subprocess

        result = subprocess.run(
            [sys.executable, "-I", str(ROOT / "tools" / "acceptance" / "compare_baton.py"), "--help"],
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_query_comparison_requires_same_capture_and_mask(self):
        import importlib.util

        self.assertIsNotNone(importlib.util.find_spec("tools.acceptance.compare_baton"))
        from tools.acceptance.compare_baton import compare_query_results

        old = {
            "distances": torch.zeros(2),
            "face_ids": torch.tensor([0, 1]),
            "uvw": torch.tensor([[1., 0., 0.], [1., 0., 0.]]),
            "positions": torch.zeros(2, 3),
            "attrs": torch.ones(2, 6),
            "mask": torch.tensor([[True, False]]),
            "capture_sha256": "same",
        }
        self.assertTrue(compare_query_results(old, old)["passed"])
        self.assertFalse(compare_query_results(old, {**old, "capture_sha256": "other"})["passed"])
        self.assertFalse(compare_query_results(old, {**old, "mask": torch.tensor([[False, False]])})["passed"])

    def test_equivalent_positions_pass_even_when_face_ids_differ(self):
        old = {
            "distances": torch.zeros(2),
            "face_ids": torch.tensor([0, 1]),
            "uvw": torch.tensor([[1., 0., 0.], [1., 0., 0.]]),
            "positions": torch.zeros(2, 3),
            "attrs": torch.ones(2, 6),
        }
        new = {
            "distances": torch.zeros(2),
            "face_ids": torch.tensor([1, 0]),
            "uvw": torch.tensor([[0., 1., 0.], [0., 1., 0.]]),
            "positions": torch.zeros(2, 3),
            "attrs": torch.ones(2, 6),
        }

        result = compare_backend_results(old, new)

        self.assertTrue(result["passed"])
        self.assertEqual(result["face_mismatches"], 2)
        self.assertEqual(result["max_position_abs_diff"], 0.0)


if __name__ == "__main__":
    unittest.main()
