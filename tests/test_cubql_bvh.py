import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CUMESH_ROOT = (ROOT / "../afamd/src/CuMesh").resolve()


class CuBQLSourceContractTests(unittest.TestCase):
    def test_cubql_source_pin_and_probe_exist(self):
        metadata_path = ROOT / "installer/amd/cubql-source.json"
        self.assertTrue(metadata_path.is_file(), "cubql-source.json is not implemented")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["commit"], "a7c1072119b34312e8b5dbf73a2e22628309acb7")
        self.assertEqual(metadata["url"], "https://github.com/NVIDIA/cuBQL.git")
        self.assertTrue((ROOT / "scripts/native/cubql_hip_probe.cu").is_file())
        self.assertTrue((ROOT / "licenses/cuBQL-Apache-2.0.txt").is_file())

    def test_new_backend_and_untouched_cumesh_extensions(self):
        setup = (CUMESH_ROOT / "setup.py").read_text(encoding="utf-8")
        self.assertIn('name="cumesh._bvh"', setup)
        self.assertIn('name="cumesh._C"', setup)
        self.assertIn('name="cumesh._xatlas"', setup)
        self.assertNotIn('name="cumesh._cubvh"', setup)


if __name__ == "__main__":
    unittest.main()
