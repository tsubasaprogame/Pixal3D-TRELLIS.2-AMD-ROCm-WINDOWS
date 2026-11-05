import json
import struct
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "baton-uv2048"


def read_glb():
    payload = (EXAMPLE / "baton-uv2048.glb").read_bytes()
    magic, version, length = struct.unpack_from("<4sII", payload)
    if magic != b"glTF" or version != 2 or length != len(payload):
        raise AssertionError("Invalid GLB header")
    json_length, chunk_type = struct.unpack_from("<II", payload, 12)
    if chunk_type != 0x4E4F534A:
        raise AssertionError("Missing GLB JSON chunk")
    document = json.loads(payload[20:20 + json_length].decode("utf-8").rstrip("\x00 "))
    return payload, document


class PublicationTests(unittest.TestCase):
    def test_reviewed_glb_geometry_and_embedded_atlas(self):
        payload, document = read_glb()
        self.assertEqual(len(payload), 22_138_764)
        vertices = 0
        triangles = 0
        for mesh in document["meshes"]:
            for primitive in mesh["primitives"]:
                vertices += document["accessors"][primitive["attributes"]["POSITION"]]["count"]
                triangles += document["accessors"][primitive["indices"]]["count"] // 3
        self.assertEqual(vertices, 357_819)
        self.assertEqual(triangles, 662_201)
        image = next(item for item in document["images"] if item.get("mimeType") == "image/png")
        view = document["bufferViews"][image["bufferView"]]
        offset = 20 + struct.unpack_from("<I", payload, 12)[0] + 8 + view.get("byteOffset", 0)
        png = payload[offset:offset + view["byteLength"]]
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(struct.unpack_from(">II", png, 16), (2048, 2048))

    def test_sanitized_measured_result(self):
        result = json.loads((EXAMPLE / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["glbPath"], "examples/baton-uv2048/baton-uv2048.glb")
        self.assertEqual(result["fallbackEvents"], [])
        self.assertEqual(result["seconds"], 499)
        self.assertTrue(result["validation"]["accepted"])

    def test_required_third_party_notices(self):
        required = (
            "CuMesh-MIT.txt", "FlexGEMM-MIT.txt", "TRELLIS2-MIT.txt",
            "cubvh-MIT.txt", "cubvh-NVIDIA-Source-Code-License.txt",
            "xatlas-MIT.txt", "Eigen-CuMesh-MPL-2.0.txt",
            "Eigen-O-Voxel-MPL-2.0.txt", "PyTorch-LICENSE.txt",
            "PyTorch-NOTICE.txt", "Torchvision-LICENSE.txt", "Torchaudio-LICENSE.txt",
        )
        for name in required:
            with self.subTest(name=name):
                self.assertTrue((ROOT / "licenses" / name).is_file())
        notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        self.assertIn("3.3", notices)
        self.assertIn("non-commercial", notices.lower())

    def test_documentation_and_ci_contract(self):
        license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        self.assertIn("Copyright (c) 2026 Михаил Бойко (Mikhail Boyko)", license_text)
        for name in ("README.md", "README.ru.md"):
            text = (ROOT / name).read_text(encoding="utf-8")
            self.assertIn("examples/baton-uv2048/preview.png", text)
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for option in ("-DryRun", "-ProbeOnly", "-ForceProbe"):
            self.assertIn(option, readme)
        evidence = (ROOT / "docs" / "RX9070XT.md").read_text(encoding="utf-8")
        self.assertIn("nine base checks", evidence)
        self.assertIn("five mandatory extension checks", evidence)
        self.assertIn("NATTEN", evidence)
        self.assertIn("nvdiffrast", evidence)
        self.assertIn("unsupported", evidence)
        workflow = (ROOT / ".github" / "workflows" / "smoke.yml").read_text(encoding="utf-8")
        self.assertIn("windows-latest", workflow)
        self.assertNotIn("Pixal3DNative.ps1 -RuntimeRoot", workflow)

    def test_public_text_has_no_private_or_studio_content(self):
        parts = (
            ("Astra", "Forge"), ("Artem", " Doronin"), ("C:\\", "Users\\"),
            ("gh", "p_"), ("github", "_pat_"), ("PRIVATE", " KEY"),
        )
        forbidden = ["".join(part) for part in parts]
        files = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
        for relative in files:
            path = ROOT / relative
            with self.subTest(file=relative):
                self.assertNotIn(path.name.lower(), {".e" + "nv", "vault" + ".json"})
            if path.suffix.lower() in {".glb", ".whl", ".png", ".jpg", ".jpeg", ".pyc"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            lowered = text.lower()
            for pattern in forbidden:
                with self.subTest(file=relative, pattern=pattern):
                    self.assertNotIn(pattern.lower(), lowered)


if __name__ == "__main__":
    unittest.main()
