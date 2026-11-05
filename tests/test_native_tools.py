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

    def test_base_and_extension_readiness_fail_closed(self):
        base = [{"name": name, "status": "pass"} for name in MANDATORY_CHECKS]
        shape = [{"name": name, "status": "pass"} for name in MANDATORY_SHAPE_CHECKS]
        self.assertTrue(native_ready(base))
        self.assertTrue(shape_ready(shape))
        self.assertFalse(native_ready(base[:-1]))
        self.assertFalse(shape_ready(shape[:-1]))
        shape[0]["status"] = "unsupported"
        self.assertFalse(shape_ready(shape))


if __name__ == "__main__":
    unittest.main()
