# Cinematic Earth Assets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a cinematic cloud-free Earth GLB with 2048 × 2048 textures through the controlled cuBQL texture-transfer path, then optionally produce an independent 2048 × 2048 cloud shell.

**Architecture:** Generate and inspect one source image, use a regular sphere as the input geometry, capture TRELLIS.2 sparse texture features, and perform the final closest-point transfer with `cumesh._bvh`. Keep machine-specific launch imports in a local host adapter and user-facing artifacts in the workspace `earth-assets/` directory. Generate clouds separately with deterministic seeded noise and accept them only after independent alpha/GLB validation.

**Tech Stack:** Native Windows AMD TRELLIS.2, PyTorch ROCm, cuBQL/HIP, Node.js ESM, Python 3.12, trimesh, NumPy, Pillow, GLB 2.0.

## Global Constraints

- Use seed `42`; preserve the exact source prompt and source image.
- Earth must contain no clouds, atmosphere shell, text, labels, stars or background geometry.
- Use Standard resolution `1024` and textures exactly `2048 × 2048` for this run.
- The final Earth texture transfer must use `cumesh._bvh`, not merely load it in the runtime.
- Do not modify or replace the stable runtime or stable `main`.
- Run only one GPU process at a time; no background GPU jobs and no retry loop after driver instability.
- Set `REPO_ROOT`, `HOST_ROOT`, `EARTH_DIR` and `STABLE_PYTHON` to the existing repository, native host, workspace output and embedded Python paths before executing the commands.
- Large intermediate captures remain diagnostics; deliver only GLB, PNG, previews and `parameters.json`.

---

### Task 1: Source Image and Input Sphere

**Files:**
- Create: `$EARTH_DIR/source-prompt.txt`
- Create: `$EARTH_DIR/earth-source.png`
- Create: `$EARTH_DIR/create_sphere.py`
- Create: `$EARTH_DIR/earth-sphere.glb`

**Interfaces:**
- Consumes: the approved cinematic style and seed `42`.
- Produces: `earth-source.png` for `AMD_UV_INPUT` and `earth-sphere.glb` for `AMD_UV_GLB`.

- [ ] **Step 1: Record the exact source prompt**

Write this prompt to `source-prompt.txt`:

```text
A single complete planet Earth centered in frame, cinematic semi-realistic game asset, orthographic three-quarter globe view, deep sapphire blue oceans, richly detailed saturated green and ochre continents, recognizable large-scale Earth geography, deserts forests and mountain regions, clean white polar ice caps, crisp coastal detail, subtle realistic surface relief, premium cinematic color grading, evenly lit entire globe, high detail, isolated object. No clouds, no atmosphere, no glow, no stars, no space background, no text, no labels, no logos, no borders, no extra planets, no cropped globe.
```

- [ ] **Step 2: Generate and inspect the source**

Use the image-generation workflow with a transparent background and the largest available square source size, using seed `42` if the tool exposes it. Save the result as `earth-source.png`. Check its byte size before viewing, inspect the image, and reject it if it contains clouds, text, extra planets or a cropped globe.

Expected: one complete cloud-free Earth matching the recorded prompt.

- [ ] **Step 3: Verify that the sphere test initially fails**

Run:

```bash
"$STABLE_PYTHON" -c "import trimesh; m=trimesh.load(r'$EARTH_DIR/earth-sphere.glb',force='mesh',process=False); assert len(m.vertices)>10000 and len(m.faces)>20000; print(len(m.vertices),len(m.faces))"
```

Expected: FAIL with `FileNotFoundError` because `earth-sphere.glb` does not exist.

- [ ] **Step 4: Create the deterministic input sphere**

Write `create_sphere.py`:

```python
from pathlib import Path
import trimesh

output = Path(__file__).resolve().parent / "earth-sphere.glb"
mesh = trimesh.creation.uv_sphere(radius=1.0, count=[192, 96])
mesh.export(output)
print({"vertices": len(mesh.vertices), "faces": len(mesh.faces), "path": str(output)})
```

Run:

```bash
"$STABLE_PYTHON" "$EARTH_DIR/create_sphere.py"
"$STABLE_PYTHON" -c "import trimesh; m=trimesh.load(r'$EARTH_DIR/earth-sphere.glb',force='mesh',process=False); assert len(m.vertices)>10000 and len(m.faces)>20000; print(len(m.vertices),len(m.faces))"
```

Expected: a non-empty sphere with more than 10000 vertices and 20000 faces.

### Task 2: Host-Neutral Capture Through TRELLIS.2

**Files:**
- Create: `$EARTH_DIR/host-adapter.mjs`
- Create: `$REPO_ROOT/data/runtime/diagnostics/native-amd/baton-cubql-capture.pt`
- Create: `$REPO_ROOT/data/runtime/diagnostics/native-amd/baton-cubql-capture.json`

**Interfaces:**
- Consumes: `earth-source.png`, `earth-sphere.glb`, the existing native host launcher and UV workflow.
- Produces: the capture consumed by `tools/acceptance/run_transfer.py`.

- [ ] **Step 1: Create the local launch adapter**

Write `host-adapter.mjs`:

```javascript
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const hostRoot = path.resolve(process.env.CUBQL_HOST_ROOT);
const load = (relative) => import(pathToFileURL(path.join(hostRoot, relative)).href);

export async function buildUvAtlasWorkflow(options) {
  const module = await load('scripts/native/uv_atlas_workflow.mjs');
  return module.buildUvAtlasWorkflow(options);
}

export async function createCaptureLaunch({ url, env, tempDir, dbFile }) {
  const { buildNativePixal3DLaunch } = await load('server/pixal3d-native.mjs');
  const { pixal3dConfig, pixal3dEnv } = await load('server/pixal3d.mjs');
  const { getPixal3DBackend } = await load('server/pixal3d-backends.mjs');
  const hostEnv = {
    ...env,
    ASTRA_PIXAL3D_BACKEND: 'windows-rocm',
    ASTRA_PIXAL3D_URL: url,
    ASTRA_AMD_NAF_TILE: '2',
    ASTRA_PIXAL3D_NATIVE_DIR: path.resolve(process.env.CUBQL_STABLE_RUNTIME),
  };
  const config = pixal3dConfig({ env: hostEnv, appRoot: hostRoot });
  const descriptor = getPixal3DBackend('windows-rocm', hostEnv, hostRoot);
  return buildNativePixal3DLaunch({
    config,
    descriptor,
    tempDir,
    dbFile,
    env: pixal3dEnv(config, hostEnv),
  });
}
```

- [ ] **Step 2: Check adapter syntax and required paths**

Run:

```bash
node --check "$EARTH_DIR/host-adapter.mjs"
test -f "$EARTH_DIR/earth-source.png" && test -f "$EARTH_DIR/earth-sphere.glb"
```

Expected: exit code `0`; no server process is started.

- [ ] **Step 3: Capture the sparse texture features in one foreground GPU process**

Run with a 300-second foreground limit:

```bash
export CUBQL_HOST_ADAPTER="$EARTH_DIR/host-adapter.mjs"
export CUBQL_HOST_ROOT="$HOST_ROOT"
export CUBQL_STABLE_RUNTIME="$HOST_ROOT/runtime/ComfyUI-Pixal3D-AMD-Native"
export AMD_UV_INPUT="$EARTH_DIR/earth-source.png"
export AMD_UV_GLB="$EARTH_DIR/earth-sphere.glb"
node "$REPO_ROOT/tools/acceptance/capture_baton.mjs"
```

Expected: `capture_complete` evidence, a non-empty `baton-cubql-capture.pt`, texture size `2048`, seed `42`, and no fallback or driver error. Stop immediately on display instability or a non-recoverable HIP error.

### Task 3: cuBQL Texture Transfer and Earth Validation

**Files:**
- Create: `$EARTH_DIR/validate_earth.py`
- Create: `$EARTH_DIR/transfer/baton-new-cubql.glb`
- Create: `$EARTH_DIR/earth.glb`
- Create: `$EARTH_DIR/earth-base-color.png`
- Create: `$EARTH_DIR/earth-metallic-roughness.png`
- Create: `$EARTH_DIR/earth-preview.png`
- Create: `$EARTH_DIR/parameters.json`

**Interfaces:**
- Consumes: `baton-cubql-capture.pt` and the experimental `cumesh._bvh` runtime.
- Produces: the final cloud-free Earth asset and its validation record.

- [ ] **Step 1: Write the validator and verify red before transfer**

Write `validate_earth.py`:

```python
import io
import json
import struct
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

root = Path(__file__).resolve().parent
path = root / "earth.glb"
data = path.read_bytes()
magic, version, length = struct.unpack_from("<4sII", data, 0)
assert magic == b"glTF" and version == 2 and length == len(data)
json_length, json_type = struct.unpack_from("<I4s", data, 12)
assert json_type == b"JSON"
document = json.loads(data[20:20 + json_length])
binary_header = 20 + json_length
binary_length, binary_type = struct.unpack_from("<I4s", data, binary_header)
assert binary_type == b"BIN\x00"
binary = data[binary_header + 8:binary_header + 8 + binary_length]
images = []
for image in document["images"]:
    assert "bufferView" in image and "uri" not in image
    view = document["bufferViews"][image["bufferView"]]
    start = view.get("byteOffset", 0)
    decoded = Image.open(io.BytesIO(binary[start:start + view["byteLength"]]))
    decoded.load()
    images.append(decoded)
mesh = trimesh.load(path, force="mesh", process=False)
assert len(mesh.vertices) and len(mesh.faces)
assert np.isfinite(mesh.vertices).all()
assert mesh.visual.uv is not None and len(mesh.visual.uv) == len(mesh.vertices)
primitive = document["meshes"][0]["primitives"][0]
pbr = document["materials"][primitive["material"]]["pbrMetallicRoughness"]
base_index = document["textures"][pbr["baseColorTexture"]["index"]]["source"]
mr_index = document["textures"][pbr["metallicRoughnessTexture"]["index"]]["source"]
base = Image.open(root / "earth-base-color.png").convert("RGBA")
mr = Image.open(root / "earth-metallic-roughness.png").convert("RGB")
assert images[base_index].size == (2048, 2048)
assert images[mr_index].size == (2048, 2048)
assert np.array_equal(images[base_index].convert("RGBA"), base)
assert np.array_equal(images[mr_index].convert("RGB"), mr)
report = {"passed": True, "vertices": len(mesh.vertices), "triangles": len(mesh.faces), "textureSize": 2048}
(root / "validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
```

Run:

```bash
"$STABLE_PYTHON" "$EARTH_DIR/validate_earth.py"
```

Expected: FAIL with `FileNotFoundError` because `earth.glb` has not been generated.

- [ ] **Step 2: Run the actual cuBQL closest-point transfer**

Run with a 300-second foreground limit:

```bash
export CAPTURE="$REPO_ROOT/data/runtime/diagnostics/native-amd/baton-cubql-capture.pt"
export CUBQL_STABLE_RUNTIME="$HOST_ROOT/runtime/ComfyUI-Pixal3D-AMD-Native"
"$STABLE_PYTHON" "$REPO_ROOT/tools/acceptance/run_transfer.py" --label new --runtime "$REPO_ROOT/data/runtime/experimental-runtime" --stable-runtime "$CUBQL_STABLE_RUNTIME" --capture "$CAPTURE" --output-dir "$EARTH_DIR/transfer"
```

Expected: `cumesh_backend` is `cumesh._bvh`, texture size is 2048, outputs are finite, and fallback/driver error arrays are empty.

- [ ] **Step 3: Name the user-facing artifacts and run validation**

Run:

```bash
cp "$EARTH_DIR/transfer/baton-new-cubql.glb" "$EARTH_DIR/earth.glb"
cp "$EARTH_DIR/transfer/baton-new-cubql-atlas.png" "$EARTH_DIR/earth-base-color.png"
cp "$EARTH_DIR/transfer/baton-new-cubql-metallic-roughness.png" "$EARTH_DIR/earth-metallic-roughness.png"
"$STABLE_PYTHON" "$EARTH_DIR/validate_earth.py"
```

Expected: `passed: true` and exact 2048 × 2048 embedded/standalone pixel matches.

- [ ] **Step 4: Render and inspect multiple views**

Use the installed 3D preview/render workflow to open `earth.glb` and save a main preview plus front, rear and polar inspection views under `earth-assets/`. Inspect the full atlas and enlarged seam crops. Reject unintended clouds, missing regions, severe stretching, duplicated shells or color displacement.

Expected: one coherent cloud-free Earth and no obvious seam regression.

- [ ] **Step 5: Record reproducibility and measured results**

Write `parameters.json` with seed `42`, the complete prompt from `source-prompt.txt`, source-image SHA-256, cuBQL revision `a7c1072119b34312e8b5dbf73a2e22628309acb7`, Standard `1024`, texture size `2048`, texture sampling settings from the workflow, output hashes, measured time/VRAM from the transfer report, and empty fallback/driver-error arrays only if confirmed by that report. Run:

```bash
"$STABLE_PYTHON" -c "import json; p=json.load(open(r'$EARTH_DIR/parameters.json',encoding='utf-8')); assert p['seed']==42 and p['textureSize']==2048 and p['backend']=='cumesh._bvh'; print('parameters valid')"
```

Expected: `parameters valid`.

### Task 4: Separate Cloud Layer and Delivery

**Files:**
- Create: `$EARTH_DIR/create_clouds.py`
- Create: `$EARTH_DIR/clouds-rgba.png`
- Create: `$EARTH_DIR/clouds.glb`
- Create: `$EARTH_DIR/clouds-preview.png`

**Interfaces:**
- Consumes: the accepted Earth asset and seed `42`.
- Produces: an optional independent cloud shell; Earth files remain unchanged.

- [ ] **Step 1: Verify the cloud test initially fails**

Run:

```bash
"$STABLE_PYTHON" -c "from PIL import Image; im=Image.open(r'$EARTH_DIR/clouds-rgba.png'); assert im.size==(2048,2048) and im.mode=='RGBA'; a=im.getchannel('A'); assert a.getextrema()[0]==0 and a.getextrema()[1]>0"
```

Expected: FAIL with `FileNotFoundError`.

- [ ] **Step 2: Generate deterministic RGBA clouds and a separate shell**

Write `create_clouds.py`:

```python
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

root = Path(__file__).resolve().parent
rng = np.random.default_rng(42)
noise = np.zeros((2048, 2048), dtype=np.float32)
for size, weight in ((8, 0.40), (16, 0.25), (32, 0.17), (64, 0.11), (128, 0.07)):
    layer = Image.fromarray(rng.integers(0, 256, (size, size), dtype=np.uint8))
    layer = layer.resize((2048, 2048), Image.Resampling.BICUBIC)
    noise += np.asarray(layer, dtype=np.float32) / 255.0 * weight
alpha = np.clip((noise - 0.54) * 5.0, 0.0, 1.0)
rgba = np.full((2048, 2048, 4), 255, dtype=np.uint8)
rgba[..., 0] = np.clip(245 + noise * 10, 0, 255).astype(np.uint8)
rgba[..., 1] = np.clip(248 + noise * 7, 0, 255).astype(np.uint8)
rgba[..., 2] = 255
rgba[..., 3] = np.round(alpha * 220).astype(np.uint8)
texture = Image.fromarray(rgba, "RGBA")
texture.save(root / "clouds-rgba.png")
mesh = trimesh.creation.uv_sphere(radius=1.012, count=[192, 96])
vertices = mesh.vertices
u = (np.arctan2(vertices[:, 1], vertices[:, 0]) / (2 * np.pi) + 0.5) % 1.0
v = np.arccos(np.clip(vertices[:, 2], -1.0, 1.0)) / np.pi
material = PBRMaterial(baseColorTexture=texture, alphaMode="BLEND", doubleSided=True, metallicFactor=0.0, roughnessFactor=1.0)
mesh.visual = TextureVisuals(uv=np.column_stack((u, 1.0 - v)), material=material)
mesh.export(root / "clouds.glb")
print({"texture": [2048, 2048], "alphaExtrema": texture.getchannel("A").getextrema(), "vertices": len(mesh.vertices)})
```

Run:

```bash
"$STABLE_PYTHON" "$EARTH_DIR/create_clouds.py"
"$STABLE_PYTHON" -c "from PIL import Image; im=Image.open(r'$EARTH_DIR/clouds-rgba.png'); assert im.size==(2048,2048) and im.mode=='RGBA'; a=im.getchannel('A'); assert a.getextrema()[0]==0 and a.getextrema()[1]>0; print('cloud texture valid')"
```

Expected: 2048 × 2048 RGBA, transparent and non-transparent pixels, and a separate `clouds.glb`.

- [ ] **Step 3: Inspect and conditionally accept clouds**

Render `clouds.glb` over a copy of Earth without changing `earth.glb`. Inspect the alpha map and cloud seam. If the shell, transparency or seam is unacceptable, remove `clouds.glb` and `clouds-preview.png` from the delivery set and record the limitation; never bake this texture into Earth.

Expected: either an independent accepted cloud layer or a clear exclusion with Earth still accepted.

- [ ] **Step 4: Final verification and delivery**

Run the Earth validator again, verify `parameters.json`, and deliver `earth.glb`, `earth-base-color.png`, `earth-metallic-roughness.png`, `earth-preview.png`, `parameters.json`, and only the accepted cloud files through the configured file-delivery command.

Expected: every delivered file exists, Earth validation remains `passed: true`, and the user receives seed/style information for the next planets.
