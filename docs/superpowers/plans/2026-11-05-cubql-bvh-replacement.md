# cuBQL BVH Replacement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace CuMesh's NVIDIA-restricted cubvh BVH backend with a pinned cuBQL/HIP backend while preserving the public `cumesh.cuBVH` API and TRELLIS.2 texture transfer on RX 9070 XT.

**Architecture:** Keep CuMesh `_C` and `_xatlas` unchanged. Add a new `cumesh._bvh` extension containing project-owned pybind/HIP query code and pinned Apache-2.0 cuBQL sources; switch only `cumesh/bvh.py` after golden outputs are captured. Build and test experimentally without replacing the stable wheel or runtime.

**Tech Stack:** CPython 3.12, PyTorch 2.9.1+rocm7.2.1, ROCm 7.2.1 HIP, C++20, cuBQL 1.3.1, pybind11, NumPy, unittest, RX 9070 XT (`gfx1201`).

## Global Constraints

- Work only on `feature/cubql-bvh`; stable `main`, its manifest, and published wheel remain unchanged.
- Pin cuBQL to `a7c1072119b34312e8b5dbf73a2e22628309acb7` from `https://github.com/NVIDIA/cuBQL.git`.
- Do not copy NVIDIA-restricted cubvh implementation code into the replacement.
- No CPU runtime fallback and no replacement of stable runtime files before all acceptance checks pass.
- Preserve `create_cuBVH(vertices, triangles)`, `ray_trace`, `unsigned_distance`, and `signed_distance` including UVW and both SDF modes.
- Compare non-tied geometry with absolute/relative tolerance `2e-5`; ray miss is exactly `face_id=-1`, `depth=1000.0`.
- Build only for the approved Windows/ROCm/PyTorch/`gfx1201` configuration during this task.
- Run GPU work in small batches and stop immediately on driver reset, display instability, or non-recoverable HIP errors.

---

### Task 1: Reproducible cuBQL HIP Probe

**Files:**
- Create: `scripts/native/cubql_hip_probe.cu`
- Create: `installer/amd/cubql-source.json`
- Create: `licenses/cuBQL-Apache-2.0.txt`
- Test: `tests/test_cubql_bvh.py`

**Interfaces:**
- Consumes: pinned cuBQL headers under `../afamd/src/cuBQL` and ROCm `hipcc.exe` under `../afamd/venv/Scripts`.
- Produces: a standalone probe returning closest triangle `0`, closest point `(0.25,0.25,0)`, squared distance `0.25`, plus source revision metadata.

- [ ] **Step 1: Write the failing host contract test**

```python
def test_cubql_source_pin_and_probe_exist(self):
    metadata = json.loads((ROOT / "installer/amd/cubql-source.json").read_text())
    self.assertEqual(metadata["commit"], "a7c1072119b34312e8b5dbf73a2e22628309acb7")
    self.assertEqual(metadata["url"], "https://github.com/NVIDIA/cuBQL.git")
    self.assertTrue((ROOT / "scripts/native/cubql_hip_probe.cu").is_file())
    self.assertTrue((ROOT / "licenses/cuBQL-Apache-2.0.txt").is_file())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_cubql_bvh -v`
Expected: FAIL because `cubql-source.json` does not exist.

- [ ] **Step 3: Add metadata, Apache-2.0 text, and minimal probe**

```json
{
  "url": "https://github.com/NVIDIA/cuBQL.git",
  "commit": "a7c1072119b34312e8b5dbf73a2e22628309acb7",
  "version": "1.3.1",
  "license": "Apache-2.0",
  "gpuArchitecture": "gfx1201"
}
```

The probe defines `CUBQL_GPU_BUILDER_IMPLEMENTATION=1` and `CUBQL_TRIANGLE_CPAT_IMPLEMENTATION=1`, uploads one `cuBQL::Triangle`, builds `cuBQL::bvh3f` with `cuBQL::gpuBuilder`, runs `cuBQL::triangles::CPAT` at `(0.25,0.25,1)`, copies the result back, checks the expected values within `2e-5`, and prints `CUBQL_HIP_PROBE_OK`.

- [ ] **Step 4: Build and run only gfx1201 probe**

Run from the task root with ROCm paths prepended to `PATH`:

```powershell
& ..\afamd\venv\Scripts\hipcc.exe --offload-arch=gfx1201 -O3 -std=c++20 -I ..\afamd\src\cuBQL scripts/native/cubql_hip_probe.cu -o data/runtime/diagnostics/native-amd/cubql_hip_probe.exe
& data/runtime/diagnostics/native-amd/cubql_hip_probe.exe
```

Expected: exit code `0` and `CUBQL_HIP_PROBE_OK`; no driver/display instability.

- [ ] **Step 5: Commit the probe**

```bash
git add scripts/native/cubql_hip_probe.cu installer/amd/cubql-source.json licenses/cuBQL-Apache-2.0.txt tests/test_cubql_bvh.py docs/superpowers/plans/2026-11-05-cubql-bvh-replacement.md
git commit -m "test: add pinned cuBQL HIP probe"
```

### Task 2: Capture Old-Backend Golden Geometry

**Files:**
- Create: `tests/bvh_reference.py`
- Create: `tests/capture_cumesh_bvh_golden.py`
- Create: `data/bvh-golden/v1/inputs.npz`
- Create: `data/bvh-golden/v1/outputs.npz`
- Test: `tests/test_bvh_reference.py`

**Interfaces:**
- Consumes: current stable `cumesh._cubvh` in the isolated runtime.
- Produces: versioned ray/UDF/SDF/UVW golden tensors and independent CPU reference functions `ray_triangle`, `closest_point`, and `signed_distance_reference`.

- [ ] **Step 1: Write failing CPU-reference tests**

```python
def test_plane_closest_point_and_uvw(self):
    result = closest_point(np.array([.25, .25, 1.]), TRIANGLE)
    np.testing.assert_allclose(result.point, [.25, .25, 0.], atol=2e-5)
    np.testing.assert_allclose(result.uvw, [.5, .25, .25], atol=2e-5)
    self.assertAlmostEqual(result.distance, 1.0, places=5)
```

- [ ] **Step 2: Verify reference tests fail**

Run: `python -m unittest tests.test_bvh_reference -v`
Expected: FAIL with `No module named tests.bvh_reference`.

- [ ] **Step 3: Implement independent NumPy references**

Implement Möller–Trumbore ray intersection without looking at restricted source, closest-point-on-triangle regions (vertex, edge, face), barycentric reconstruction, deterministic `+X` ray parity, and nearest-normal ray-stab sign. Reject zero-area candidates when a valid triangle exists.

- [ ] **Step 4: Verify references and capture golden data**

Run: `python -m unittest tests.test_bvh_reference -v`
Expected: all CPU reference tests PASS.

Run with the old runtime Python: `python tests/capture_cumesh_bvh_golden.py --output data/bvh-golden/v1`.
Expected: ray hit/miss, plane/cube/open mesh, UDF/UVW, both SDF modes, degenerate and tied cases saved with finite outputs and exact miss sentinel.

- [ ] **Step 5: Commit golden contract**

```bash
git add tests/bvh_reference.py tests/test_bvh_reference.py tests/capture_cumesh_bvh_golden.py data/bvh-golden/v1
git commit -m "test: capture cubvh BVH golden geometry"
```

### Task 3: New Extension and Input Contract

**Files:**
- Create: `../afamd/src/CuMesh/src/bvh/bindings.cpp`
- Create: `../afamd/src/CuMesh/src/bvh/bvh_state.hip`
- Create: `../afamd/src/CuMesh/src/bvh/bvh_state.h`
- Modify: `../afamd/src/CuMesh/setup.py`
- Test: `tests/test_cubql_bvh.py`

**Interfaces:**
- Consumes: golden inputs and pinned cuBQL headers.
- Produces: `cumesh._bvh.create_cuBVH(vertices, triangles)` with owned device state and explicit validation; `_C` and `_xatlas` remain registered.

- [ ] **Step 1: Add failing build/source contract tests**

```python
def test_new_backend_and_untouched_cumesh_extensions(self):
    setup = (CUMESH_ROOT / "setup.py").read_text()
    self.assertIn('name="cumesh._bvh"', setup)
    self.assertIn('name="cumesh._C"', setup)
    self.assertIn('name="cumesh._xatlas"', setup)
    self.assertNotIn('name="cumesh._cubvh"', setup)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m unittest tests.test_cubql_bvh -v`
Expected: FAIL because `_bvh` is absent and `_cubvh` is still registered.

- [ ] **Step 3: Implement state and validation**

Validate `[N,3]` float32 vertices, `[M,3]` int32 triangles, non-empty geometry, index bounds, finite vertices, at least one non-degenerate triangle, compatible output devices, and zero-query early return. Store packed `cuBQL::Triangle` per original face and `bvh3f`; preserve `primIDs` as original face IDs. Check every HIP allocation/launch error with the operation name.

- [ ] **Step 4: Build experimental wheel without installing it**

Run with `BUILD_TARGET=rocm` and `GPU_ARCHS=gfx1201`:

```powershell
& ..\afamd\venv\Scripts\python.exe -m pip wheel --no-deps --no-build-isolation ../afamd/src/CuMesh -w data/runtime/experimental-wheels
```

Expected: wheel contains `cumesh._bvh`, `_C`, `_xatlas`, no `_cubvh`, and no `third_party/cubvh` files.

- [ ] **Step 5: Commit extension contract tests**

The adjacent CuMesh checkout is an externally managed, detached and already patched source tree; do not stage it into the public repository. Its reproducible replacement sources are copied into the public packaging task only after GPU parity passes.

```bash
git add tests/test_cubql_bvh.py
git commit -m "test: add cuBQL BVH extension contract"
```

### Task 4: Ray, UDF, SDF, and UVW Parity

**Files:**
- Create: `../afamd/src/CuMesh/src/bvh/queries.hip`
- Modify: `../afamd/src/CuMesh/src/bvh/bindings.cpp`
- Modify: `../afamd/src/CuMesh/cumesh/bvh.py`
- Create: `tests/test_cubql_gpu_parity.py`

**Interfaces:**
- Consumes: `create_cuBVH` and golden v1 data.
- Produces: `ray_trace(rays_o,rays_d,positions,face_id,depth)`, `unsigned_distance(positions,distances,face_id,uvw)`, and `signed_distance(...,mode)` on the current PyTorch HIP stream.

- [ ] **Step 1: Write failing GPU parity tests**

Load golden v1 and assert depths/positions/distances within `2e-5`, exact miss sentinel, UVW sums within `2e-5`, reconstructed closest points within `2e-5`, and SDF sign agreement outside the surface band. Permit another face ID only when reconstructed geometry meets the same tolerances.

- [ ] **Step 2: Run tests against missing backend**

Run with experimental wheel on `PYTHONPATH`: `python -m unittest tests.test_cubql_gpu_parity -v`.
Expected: FAIL because `_bvh` query operations are unavailable.

- [ ] **Step 3: Implement independent HIP query kernels**

Use cuBQL ray traversal and project-owned triangle intersection for nearest non-negative hits; shrinking-radius traversal plus project-owned closest-point logic for UDF/UVW; deterministic ray parity for watertight SDF; deterministic ray stabbing plus nearest-surface orientation for ray-stab SDF. Exclude zero-area candidates when valid geometry exists and pass the current stream from `at::cuda::getCurrentCUDAStream()`.

- [ ] **Step 4: Run small then large parity batches**

Run: `python -m unittest tests.test_cubql_gpu_parity -v`.
Expected: all plane/cube/open/degenerate/tied cases PASS, followed by a 100001-item query batch PASS with no fallback or HIP error.

- [ ] **Step 5: Commit query parity tests**

Do not stage the adjacent CuMesh checkout into the public repository. Reproducible query sources enter the public packaging task after their behavior is verified.

```bash
git add tests/test_cubql_gpu_parity.py
git commit -m "test: verify cuBQL BVH query parity"
```

### Task 5: Fail-Closed Readiness and Packaging

**Files:**
- Modify: `scripts/native/probe_trellis2_extensions.py`
- Modify: `THIRD_PARTY_NOTICES.md`
- Modify: `installer/amd/native-wheels.json`
- Test: `tests/test_native_tools.py`
- Test: `tests/test_publication.py`

**Interfaces:**
- Consumes: experimental replacement wheel and BVH operation probes.
- Produces: readiness details containing backend/revision, ray/UDF/SDF/UVW and large-batch status; publication manifest with measured wheel size/SHA-256.

- [ ] **Step 1: Add failing readiness and publication tests**

Require `cumesh_bvh` in mandatory checks; verify missing/failed BVH operations make `shape_ready=False`; verify cuBQL Apache-2.0 notice and exact source commit; inspect wheel members to forbid `_cubvh` and `third_party/cubvh` while requiring `_C` and `_xatlas`.

- [ ] **Step 2: Verify tests fail**

Run: `python -m unittest discover -s tests -p "test_*.py" -v`.
Expected: FAIL on missing `cumesh_bvh` readiness and replacement wheel metadata.

- [ ] **Step 3: Extend operational probe and update notices**

Add actual BVH operations to `probe_cumesh`, return backend `cuBQL`, full commit, parity status and large-batch details. Remove cubvh license entries only from replacement notices; retain CuMesh, Eigen and xatlas notices. Record toolchain, architecture, wheel size and SHA-256.

- [ ] **Step 4: Run host and PowerShell suites**

Run: `python -m unittest discover -s tests -p "test_*.py" -v` and `powershell -NoProfile -ExecutionPolicy Bypass -File tests/Smoke.ps1`.
Expected: all tests PASS and replacement wheel inspection contains no restricted component.

- [ ] **Step 5: Commit readiness and packaging**

```bash
git add scripts/native/probe_trellis2_extensions.py tests installer/amd licenses THIRD_PARTY_NOTICES.md
git commit -m "build: package cuBQL BVH replacement"
```

### Task 6: RX 9070 XT Acceptance and Report

**Files:**
- Create: `data/runtime/diagnostics/native-amd/cubql-bvh-acceptance.json`
- Create: `docs/CUBQL-BVH-VERIFICATION.md`
- Test: existing TRELLIS.2 baton capture and texture-transfer inputs.

**Interfaces:**
- Consumes: identical captured intermediate mesh and texture inputs for old/new backends.
- Produces: verified attribute transfer, Standard 1024 + UV 2048 baton GLB/atlas, timings, VRAM telemetry, fallback and driver-stability report.

- [ ] **Step 1: Compare old/new transfer on captured inputs**

Run old and new backends separately on the same captured mesh. Assert finite attributes, matching geometry/face assignment, and atlas differences within the documented numerical/image tolerance; save both outputs for visual inspection.

- [ ] **Step 2: Run complete baton workflow once**

Run Standard 1024 with UV 2048 using the replacement wheel. Expected: non-empty valid GLB, finite geometry, embedded base-color atlas exactly `2048 x 2048`, no missing texture and no fallback.

- [ ] **Step 3: Inspect atlas and rendered preview**

Compare UV seams, color placement and face/attribute assignment with the approved example. Any obvious color displacement or seam regression blocks acceptance.

- [ ] **Step 4: Record verification and rerun all tests**

Record elapsed time, available peak VRAM telemetry, fallback count, watchdog/driver resets, probe reports, old/new performance difference, wheel SHA-256 and source commit. Rerun Python, PowerShell and mandatory GPU probes.

- [ ] **Step 5: Commit only verified evidence**

```bash
git add docs/CUBQL-BVH-VERIFICATION.md data/runtime/diagnostics/native-amd/cubql-bvh-acceptance.json
git commit -m "docs: verify cuBQL BVH on RX 9070 XT"
```

If compilation or GPU reliability fails, stop at the experimental branch with logs and a precise blocker; do not replace the stable wheel or weaken readiness checks.
