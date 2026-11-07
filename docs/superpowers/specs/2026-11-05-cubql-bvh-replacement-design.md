# Replacing cubvh with cuBQL/HIP

Date: 2026-11-05
Status: approved design, implementation pending
Target: native Windows Pixal3D / TRELLIS.2 on AMD Radeon RX 9070 XT (`gfx1201`)

## Purpose

Remove the nested cubvh component and its NVIDIA non-commercial research/evaluation restriction from the new CuMesh HIP wheel without changing the public `cumesh.cuBVH` Python API or losing TRELLIS.2 texture transfer.

The implementation uses Apache-2.0 cuBQL as the BVH builder/traversal layer and independently implemented HIP query kernels. NVIDIA-restricted implementation code must not be copied into the replacement.

This work happens on `feature/cubql-bvh`. Stable `main`, its manifest, and its published wheel remain unchanged until all acceptance checks pass.

## Goals

- Preserve `create_cuBVH(vertices, triangles)` and the existing methods `ray_trace`, `unsigned_distance`, and `signed_distance` with `return_uvw` and both SDF modes.
- Preserve the existing `CuMesh.cuBVH` consumer interface so CuMesh remeshing and TRELLIS.2 nodes require no call-site changes.
- Return geometrically correct closest-face IDs and barycentric coordinates for texture and attribute transfer.
- Build and run natively on Windows with ROCm 7.2.1, PyTorch 2.9.1+rocm7.2.1, CPython 3.12, and `gfx1201`.
- Remove `third_party/cubvh`, `cumesh._cubvh`, and their NVIDIA-restricted implementation from the replacement wheel.
- Provide reproducible sources, a pinned cuBQL revision, license notices, GPU parity tests, and an end-to-end textured example.
- Measure query/runtime time, VRAM, fallbacks, and driver stability.

## Non-goals

- Supporting other GPU architectures, PyTorch versions, or operating systems.
- Replacing the independent CuMesh `_C` or `_xatlas` extensions.
- Reimplementing cubvh flood-fill, hash-table, marching-cubes, or decimation helpers that are not called by the CuMesh package or the verified TRELLIS.2 path.
- Adding a CPU runtime fallback.
- Guaranteeing bitwise-identical face IDs for geometrically tied closest primitives or identical floating-point traversal order.
- Setting a hard performance-regression threshold before measurements. Correctness and quality are blockers; measured slowdown is reported and optimized separately unless it prevents a run from completing reliably.

## Chosen architecture

### 1. New BVH extension

Build a new extension named `cumesh._bvh` from project-owned binding/query sources and a pinned cuBQL source revision. It exports:

- `create_cuBVH(vertices, triangles)`;
- `cuBVH.ray_trace(rays_o, rays_d, positions, face_id, depth)`;
- `cuBVH.unsigned_distance(positions, distances, face_id, optional_uvw)`;
- `cuBVH.signed_distance(positions, distances, face_id, optional_uvw, mode)`.

The extension owns device buffers for packed triangle vertices, original face IDs, primitive bounds, and the cuBQL BVH. Original face IDs remain stable even when cuBQL reorders primitive IDs.

`cumesh/bvh.py` imports `._bvh` as its backend while retaining the existing Python wrapper, tensor allocation, shapes, dtypes, and public class name. TRELLIS.2 continues to call `cumesh.cuBVH`.

### 2. Removal of the old component

The replacement CuMesh source tree and wheel contain no `third_party/cubvh` directory and no `cumesh._cubvh` module. The old extension is not retained as an optional backend in the replacement wheel.

The other nested cubvh helpers are outside the verified CuMesh/TRELLIS.2 call graph. Their removal does not alter the independent CuMesh `_C` hole-filling implementation or `_xatlas`.

### 3. Query implementation

- **Ray tracing:** traverse cuBQL and independently intersect candidate triangles. Return the nearest non-negative hit, the original face ID, ray depth, and `origin + depth * direction`.
- **Ray miss:** preserve the current contract: `face_id = -1`, `depth = 1000.0`, and the position evaluated at that depth.
- **Unsigned distance:** use a shrinking-radius closest-surface query. Return distance, original face ID, and optionally barycentric UVW of the closest point.
- **Watertight SDF:** combine the UDF magnitude and closest-face result with deterministic ray-parity inside/outside classification. Interior distances are negative and exterior distances positive.
- **Ray-stab SDF:** use deterministic ray stabbing and nearest-surface orientation for non-watertight meshes; preserve the same sign convention and UDF/UVW outputs.
- **Degenerate triangles:** exclude zero-area candidates from ray hits and closest-point selection when a valid candidate exists. Reject a mesh containing no non-degenerate triangles at construction with an explicit error.

No algorithm is copied from the NVIDIA-restricted implementation. Public API behavior and numerical references are the compatibility contract.

## Data flow

1. The Python wrapper converts input vertices/triangles to CPU NumPy arrays and allocates GPU output tensors as it does today.
2. The binding validates dimensions, dtypes, index ranges, non-empty geometry, and output devices before allocating device state.
3. The adapter builds primitive bounds and the cuBQL BVH on the current PyTorch HIP stream.
4. Each query uses packed contiguous float32 inputs and writes into caller-provided tensors on that same stream.
5. UVW is computed from the selected closest point and original triangle vertices. Consumers reconstruct attributes exactly as before.
6. Destruction releases BVH and triangle device allocations without synchronizing unrelated streams.

Existing query chunking in CuMesh/TRELLIS.2, including the 100000-item batches where present, remains in place. New native code does not silently move queries to CPU or allocate unbounded temporary buffers.

## Error handling and watchdog safety

- Invalid shapes, unsupported dtypes, out-of-range triangle indices, empty meshes, and device mismatches raise explicit errors before kernel launch.
- HIP allocation, launch, and stream errors are checked and surfaced with the operation name.
- Zero-query inputs return empty outputs without launching a kernel.
- Large queries retain caller-side chunking; any additional native chunking must be bounded and reported in diagnostics.
- A failed experimental build or probe never replaces the stable wheel or marks the runtime ready.
- The installer and readiness probes fail closed if `_bvh`, any BVH operation, or required license/source metadata is missing.

## Test strategy

### Host-side contract tests

Run without a GPU:

- setup and source manifests reference the new `_bvh` module and pinned cuBQL revision;
- no replacement source or wheel contains `third_party/cubvh`, `_cubvh`, or NVIDIA-restricted source notices as an included component;
- existing CuMesh `_C` and `_xatlas` entries remain;
- wheel size/SHA-256 manifest validation remains mandatory;
- readiness fails when any BVH check is missing or failed;
- Apache-2.0 and other required third-party notices are present.

### GPU golden and reference parity

Before removing the old backend, capture versioned golden inputs/outputs from it. Test the replacement against both those golden results and an independent small-mesh CPU reference. CPU code is test-only.

Cases:

- plane and tetrahedron/cube;
- open square and non-watertight mesh;
- inside, outside, on-surface, near-edge, and near-vertex queries;
- ray hit, miss, parallel ray, origin on surface, and non-unit direction;
- UDF with and without UVW;
- watertight and ray-stab SDF signs and magnitudes;
- degenerate triangles and wholly degenerate input;
- tied closest faces;
- queries larger than one batch and a production-sized mesh.

For non-tied results, compare distances, depths, positions, and reconstructed closest points with absolute/relative tolerance `2e-5`. UVW must sum to one within `2e-5`, remain on the selected triangle within numerical tolerance, and reconstruct the closest point within `2e-5`. SDF signs must agree outside a `2e-5` surface band.

For tied candidates, a different face ID is valid only when its distance and reconstructed closest point satisfy the same tolerances. A ray miss must match the sentinel contract exactly. No test may require traversal-order equality.

### Mandatory extension probe

Extend `probe_trellis2_extensions.py` with actual BVH operations rather than import-only checks. Report backend name, source revision, ray/UDF/SDF/UVW results, large-batch completion, and GPU errors. `shapeReady` is false unless all mandatory BVH checks pass.

### End-to-end acceptance on RX 9070 XT

1. Run attribute/texture transfer from the same captured intermediate mesh and inputs with old and new backends. Compare the transferred attributes and atlas without regenerating the shape.
2. Run the complete baton Standard 1024 + UV 2048 workflow with the replacement wheel.
3. Verify a non-empty valid GLB, embedded base-color atlas exactly 2048 x 2048, finite geometry, and no missing textures.
4. Review rendered preview and atlas against the approved example, especially UV seams, face/attribute assignment, and obvious color displacement.
5. Record elapsed time, peak available VRAM telemetry, fallback events, watchdog/driver resets, and probe reports.
6. Run existing PowerShell Smoke and Python unit tests.

Acceptance requires correct geometry/texture behavior, no hidden CPU fallback, no driver reset, and all tests passing. Performance differences are included in the verification report.

## Build, source, and license provenance

- Pin cuBQL by full commit SHA; do not track a moving branch during builds.
- Record the source URL, commit, build toolchain, GPU architecture, and wheel SHA-256.
- Include the cuBQL Apache-2.0 text and any applicable NOTICE attribution.
- Keep CuMesh, Eigen, xatlas, and other applicable notices.
- Remove cubvh/NVIDIA-restricted license entries only when the corresponding code is actually absent from the replacement source and wheel.
- Review the wheel file list before publication.

## Rollout

- Commit design and implementation only to `feature/cubql-bvh` during development.
- Keep the existing `main` wheel and examples untouched.
- Publish the branch for review after host tests and at least the gfx1201 smoke pass.
- Merge or replace the stable release only after the full parity and end-to-end acceptance checks pass.
- If cuBQL cannot compile or run reliably on `gfx1201`, stop at an experimental branch with logs and a clear blocker; do not weaken readiness checks or substitute a hidden fallback.
