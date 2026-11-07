import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "native"))

from install_pixal3d_amd import REQUIRED_WHEELS, find_wheel_set, validate_runtime_root
from probe_trellis2_extensions import MANDATORY_SHAPE_CHECKS, shape_ready
from verify_pixal3d_amd import MANDATORY_CHECKS, native_ready


class NativeContractTests(unittest.TestCase):
    def test_runtime_path_rejects_wsl_and_protected_runtimes(self):
        values = (
            "/mnt/c/amd",
            r"C:\AMD-AI\ComfyUI-Pixal3D",
            r"C:\AMD-AI\ComfyUI-Pixal3D-AMD",
        )
        for value in values:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_runtime_root(value)
        self.assertTrue(validate_runtime_root(r"C:\AMD-AI\Pixal3D-ROCm").is_absolute())

    def test_official_wheel_set_is_exact(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in REQUIRED_WHEELS:
                (root / name).write_bytes(b"test")
            self.assertEqual([path.name for path in find_wheel_set(root)], list(REQUIRED_WHEELS))
            (root / REQUIRED_WHEELS[0]).unlink()
            with self.assertRaises(FileNotFoundError):
                find_wheel_set(root)

    def test_installer_downloads_wheels_before_validating(self):
        installer = (ROOT / "installer" / "amd" / "Pixal3DNative.ps1").read_text(encoding="utf-8")
        download = installer.index("$wheels = @(Get-AmdNativeWheelPaths)")
        validation = installer.index("install_pixal3d_amd.py")
        self.assertLess(download, validation, "Official wheels must be downloaded before wheel-set validation")

    def test_extension_probe_runtime_dependencies_are_declared(self):
        requirements = (ROOT / "scripts" / "native" / "requirements-base.txt").read_text(encoding="utf-8").splitlines()
        expected = ("easydict==", "opencv-python-headless==", "plyfile==", "trimesh==", "tqdm==", "zstandard==")
        missing = [name for name in expected if not any(line.startswith(name) for line in requirements)]
        self.assertEqual(missing, [], "CuMesh and O-Voxel runtime dependencies must be pinned")

    def test_base_and_extension_readiness_fail_closed(self):
        self.assertIn("cumesh_bvh", MANDATORY_SHAPE_CHECKS)
        base = [{"name": name, "status": "pass"} for name in MANDATORY_CHECKS]
        shape = [{"name": name, "status": "pass"} for name in MANDATORY_SHAPE_CHECKS]
        self.assertTrue(native_ready(base))
        self.assertTrue(shape_ready(shape))
        self.assertFalse(native_ready(base[:-1]))
        for index, item in enumerate(shape):
            with self.subTest(missing=item["name"]):
                self.assertFalse(shape_ready(shape[:index] + shape[index + 1:]))
            with self.subTest(failed=item["name"]):
                failed = [dict(value) for value in shape]
                failed[index]["status"] = "fail"
                self.assertFalse(shape_ready(failed))
        shape[0]["status"] = "unsupported"
        self.assertFalse(shape_ready(shape))


if __name__ == "__main__":
    unittest.main()
