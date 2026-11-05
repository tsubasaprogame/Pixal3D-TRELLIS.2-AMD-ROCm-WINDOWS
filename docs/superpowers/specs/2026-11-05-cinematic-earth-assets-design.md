# Cinematic Earth and Separate Cloud Layer Design

Date: 2026-11-05
Status: approved design, written specification pending user review

## Purpose

Create a visually striking Earth asset for a coherent series of planets. The first asset is Earth without clouds, generated through the native AMD TRELLIS.2/cuBQL workflow with textures no smaller than 2048 × 2048. A separate cloud layer is optional and must not be baked into the Earth surface.

## Visual direction

Use a cinematic semi-realistic style rather than strict satellite photography or low-poly art:

- deep blue oceans with readable coastal water;
- saturated, detailed continents with recognizable large-scale geography;
- deserts, vegetation and mountain regions with distinct but harmonious colors;
- clean polar ice caps;
- subtle geometric relief that remains a single coherent planet;
- no clouds, atmospheric shell, stars, text, labels, logos or background scenery in the Earth asset.

The series contract records the prompt, seed, source image, geometry settings and palette. Future planets should change planetary features while retaining the same rendering and generation parameters.

## Reproducibility

Use seed `42` for the TRELLIS.2 workflow and for any source-generation tool that exposes a seed. Preserve the exact source image because a seed alone does not reproduce a pipeline when models or external image generators change.

Save a `parameters.json` file containing:

- seed `42`;
- the complete source prompt and negative constraints;
- SHA-256 of the source image;
- backend and cuBQL revision;
- Standard geometry resolution `1024`;
- texture size `2048`;
- texture sampling settings used by the workflow;
- output file names, dimensions and SHA-256 hashes;
- measured runtime, peak VRAM when available, fallback events and observed driver errors.

## Generation pipeline

1. Create or select one clean source image showing a single cloud-free Earth in the approved style. Inspect it before GPU generation; reject text, clouds, extra planets and cropped geometry.
2. Run the native Windows AMD pipeline sequentially in small batches. Use the experimental runtime containing `cumesh._bvh`; do not replace the stable runtime.
3. Capture the intermediate Standard 1024 mesh, UVs and sparse texture features before postprocessing.
4. Produce the final textured asset through the controlled closest-point BVH texture-transfer path with `cumesh._bvh`, so the new backend is actually exercised rather than merely present in the runtime.
5. Export a GLB with embedded base-color and metallic-roughness textures at 2048 × 2048, plus standalone PNG textures and rendered previews.
6. Only after Earth passes validation, attempt a separate cloud layer. The preferred cloud deliverable is a slightly larger cloud shell GLB with an independent 2048 × 2048 RGBA texture. If alpha, seams or geometry are not acceptable, omit the cloud asset and report that limitation rather than baking clouds into Earth.

The ordinary production AMD `postprocess_mesh` path bypasses BVH. Therefore the controlled BVH transfer in step 4 is required to satisfy the request to use the new cuBQL method.

## Deliverables

Place user-facing artifacts in the current workspace under `earth-assets/`:

- `earth.glb` — cloud-free Earth with embedded 2048 × 2048 textures;
- `earth-base-color.png` and `earth-metallic-roughness.png`;
- `earth-preview.png` and additional inspection views;
- `parameters.json` — reproducibility and validation record;
- `clouds.glb` and `clouds-rgba.png` only if the separate cloud layer passes validation;
- `clouds-preview.png` if clouds are accepted.

Large intermediate captures and query tensors remain diagnostic files and are not user-facing deliverables.

## Validation and acceptance

Earth is accepted only when:

- the GLB opens successfully and contains non-empty finite geometry and UV coordinates;
- the asset is a single coherent cloud-free planet with no obvious duplicated shells, text or background geometry;
- base-color and metallic-roughness images are embedded and are at least 2048 × 2048;
- decoded embedded texture pixels match the standalone PNG files;
- full-atlas and enlarged UV-seam inspections show no obvious color displacement or seam regression;
- previews from multiple sides do not reveal severe stretching, missing regions or unintended clouds;
- `parameters.json` records seed `42` and the actual generation settings;
- no CPU fallback is used and no driver error is observed during the measured run.

Clouds are accepted separately only when the layer is independent, transparent where intended, at least 2048 × 2048, and free of obvious seam or depth artifacts.

Passing these checks does not prove absence of future driver resets or display instability. Stop GPU work immediately on display instability, a driver reset or a non-recoverable HIP error; preserve the diagnostic log and do not retry in a tight loop.
