# cuBQL BVH verification on RX 9070 XT

Verification date: 2026-11-05.

## Result

The `cumesh._bvh` replacement passed the reviewed BVH acceptance checks on AMD Radeon RX 9070 XT (`gfx1201`) with cuBQL revision `a7c1072119b34312e8b5dbf73a2e22628309acb7`.

Verified wheel:

- `installer/amd/wheels/cumesh-1.0-cp312-cp312-win_amd64.whl`
- SHA-256: `8b48bc2730524a4760fecef64a15a87dc2763e489856e4f4d73dc06150918083`

The stable runtime and `main` were not replaced or modified by this acceptance work.

## Controlled old/new comparison

Both backends processed the same captured TRELLIS.2 mesh and sparse texture inputs in separate, sequential GPU processes. The capture is 97,865,801 bytes, has SHA-256 `e2998746546846da7c0b7f7cdcfef322a08fc0c719f4d201e020710467416b4d`, and was produced in 83.6 seconds. It contains 357,819 vertices, 662,201 triangles, UV coordinates, 1,476,717 × 6 voxel features with spatial shape `[1024, 830, 671]`, and finite floating-point values.

| Metric | Old `cumesh._cubvh` | New `cumesh._bvh` |
|---|---:|---:|
| Covered texels | 2,205,811 | 2,205,811 |
| Total transfer time | 7.2898 s | 6.9417 s |
| BVH build | 0.23699 s | 0.04907 s |
| BVH queries | 0.01883 s | 0.01546 s |
| `grid_sample` | 0.09492 s | 0.09353 s |
| Peak allocated VRAM | 2,768,168,960 bytes | 2,768,168,960 bytes |
| Peak reserved VRAM | 3,238,002,688 bytes | 3,238,002,688 bytes |
| GLB size | 26,432,824 bytes | 26,432,812 bytes |
| Recorded fallback events | 0 | 0 |
| Recorded driver errors | 0 | 0 |

These timings are measurements of this controlled transfer, not a general performance guarantee.

## Numerical and visual equivalence

The comparison passed with identical capture and coverage masks and finite outputs:

- Maximum distance difference: `1.9397813844079792e-07` (tolerance `2e-5`).
- Maximum reconstructed-position difference: `3.1441450119018555e-06` (tolerance `2e-5`).
- Maximum attribute difference: `9.53376293182373e-05` (tolerance `1e-4`).
- Fifty-one samples selected another face ID. Their reconstructed positions and attributes remain within the stated tolerances; equal-distance face selection is not required to be identical.
- Maximum raw UVW difference is `0.9974327087402344`. This is expected when a different tied face and barycentric representation are selected; the reconstructed geometry, not raw UVW equality, is the compatibility criterion.

The 2048 × 2048 atlases differ in 1,598 texels, with a maximum channel difference of 3 levels and mean channel difference `0.0001040697`. The full atlases and enlarged UV-seam crops were visually inspected; no obvious color displacement or seam regression was observed.

GLB validation passed for both outputs. Geometry and UV match the capture, old/new geometry is identical, base-color and metallic-roughness images are embedded at 2048 × 2048, and decoded texture pixels match the saved atlases.

## Fresh test results

The final acceptance suite was rerun after adding the capture/comparison harness:

```text
../afamd/venv/Scripts/python.exe -m unittest discover -s tests -p "test_*.py" -v
Ran 34 tests in 1.679s
OK
```

This includes real GPU golden/reference parity for rays, UDF, both SDF modes, UVW, degenerate faces, empty queries, validation, the public wrapper and a 100001-query batch.

```text
powershell -NoProfile -ExecutionPolicy Bypass -File tests/Smoke.ps1
Smoke checks passed.
```

The separate mandatory `probe_cumesh_bvh()` check also passed on the GPU: backend `cumesh._bvh`, exact ray miss, UDF, watertight and ray-stab SDF, UVW and 100001 queries. The probe-only virtual environment lacks `tqdm`, so the standalone probe invocation supplies a minimal import shim for that unused dependency; the BVH operations themselves use the real HIP extension and no CPU fallback. The stable embedded runtime was not used for this experimental-overlay probe because its isolated Python configuration ignores `PYTHONPATH` and still contains the old backend.

## Authoritative equirectangular Earth transfer

On 2026-11-06 the transfer harness gained an optional `--color-map` path. It samples an approved equirectangular RGB map directly at the closest-point positions returned by `cumesh._bvh`; the original generative-voxel path remains available when the option is omitted. Direct sampling is reported separately as `equirectangular_sample`, not as FlexGEMM `grid_sample` time.

The Earth run reused the completed multiview capture with SHA-256 `245566c83f3b3de9247d9267e68a4ca5af008ce7e5f7c97be8d44a4106060f30` and the approved map with SHA-256 `de812ed0e68c13ba8f57cefdd08ee89f7cca2ed62ecf31d9b2499ae9c5518314`. It produced 37,940 vertices, 72,960 triangles and 2,320,614 covered texels at 2048 × 2048. The final measured transfer took 6.8493 seconds, direct sampling took 0.1355 seconds, peak allocated VRAM was 853,702,144 bytes, and no fallback or driver errors were recorded.

Visual review rejected the first output despite valid GLB structure, and an axis-export defect was corrected. A later review in a real Three.js GLTFLoader/WebGL viewer rejected the supposedly accepted replacement too: large stretched chart regions were visible on several sides. The remaining defect was a missing raster-to-glTF V conversion. Trimesh flips V while importing GLB, so the earlier CPU preview had masked the error. The exporter now inverts raster V exactly once; a regression test pins that conversion, and the artifact validator compares raw-accessor colors against the authoritative map.

A subsequent user review caught horizontal mirroring that the V-only checks could not detect. Longitude sampling used `atan2(world.z, world.x)`, which put east on screen-left. It now uses `atan2(-world.z, world.x)`, matching Three.js SphereGeometry's equirectangular orientation. A regression test verifies that east of Greenwich appears on screen-right, and the GLB validator independently compares against the mirrored orientation. The corrected GLB has raw-UV color error `0.0130039`, flipped-V error `0.2027875`, and mirrored-longitude error `0.1439687`; the previous artifact fails the new longitude check. Four real WebGL views show unmirrored Americas, Asia, Arctic, Africa and Europe without chart breaks. The validated local deliverables are `earth-assets/earth.glb`, `earth-preview.png`, `earth-webgl-contact.jpg` and `parameters.json`; the current GLB SHA-256 is `c92dd811b773258ab085255796498b882c773e1e704e0d2631e8eab568f2b59b`.

The final suite for this update passed 42 Python tests and the PowerShell Smoke checks. The Edit3D project also passed compilation/build and real offscreen initialization with the named source material and embedded 2048 × 2048 textures.

## Reproducing the capture harness

`tools/acceptance/capture_baton.mjs` is host-neutral. Set `CUBQL_HOST_ADAPTER` to an integration module exporting `createCaptureLaunch(options)` and `buildUvAtlasWorkflow(options)`, set `CUBQL_STABLE_RUNTIME` to the native runtime root, and set `AMD_UV_INPUT` and `AMD_UV_GLB` to the input files. The adapter owns host-specific launch configuration; no private host path or machine-specific input location is stored in the public harness.

## Scope and limitations

The production AMD `postprocess_mesh` path uses `_rasterize_uv_barycentric`, `_interp_vertex_attr` and `grid_sample_3d` and bypasses BVH. Consequently, this report does **not** claim that an ordinary production baton exercised cuBQL. The BVH-specific end-to-end evidence is the separate old/new closest-point texture transfer on identical captured inputs, together with GPU parity tests and the operational probe.

No fallback events or driver errors were observed in the measured old/new transfers. This does not prove absence of future driver resets, display instability or micro-freezes.

## Evidence index

Compact machine-readable summary:

- `data/runtime/diagnostics/native-amd/cubql-bvh-acceptance.json`

Local diagnostic inputs and outputs:

- `data/runtime/diagnostics/native-amd/baton-cubql-capture.pt`
- `data/runtime/diagnostics/native-amd/baton-old-cubql.json`
- `data/runtime/diagnostics/native-amd/baton-new-cubql.json`
- `data/runtime/diagnostics/native-amd/baton-cubql-comparison.json`
- `data/runtime/diagnostics/native-amd/baton-cubql-glb-validation.json`
- `data/runtime/diagnostics/native-amd/cubql-bvh-probe.json`
- `data/runtime/diagnostics/native-amd/baton-cubql-comparison.png`
- `data/runtime/diagnostics/native-amd/baton-cubql-atlas-diff.png`

Large capture, query, GLB and PNG artifacts remain local and are intentionally not committed.
