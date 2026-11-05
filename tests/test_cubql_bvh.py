import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class CuBQLSourceContractTests(unittest.TestCase):
    def test_cubql_source_pin_and_probe_exist(self):
        metadata_path = ROOT / "installer/amd/cubql-source.json"
        self.assertTrue(metadata_path.is_file(), "cubql-source.json is not implemented")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["commit"], "a7c1072119b34312e8b5dbf73a2e22628309acb7")
        self.assertEqual(metadata["url"], "https://github.com/NVIDIA/cuBQL.git")
        self.assertTrue((ROOT / "scripts/native/cubql_hip_probe.cu").is_file())
        self.assertTrue((ROOT / "licenses/cuBQL-Apache-2.0.txt").is_file())


if __name__ == "__main__":
    unittest.main()
