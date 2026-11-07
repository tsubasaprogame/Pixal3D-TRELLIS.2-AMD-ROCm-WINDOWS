import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", choices=("old", "new"), required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--stable-runtime", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--color-map", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(args.stable_runtime / "ComfyUI"))
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(args.runtime))
    os.environ["ASTRA_AMD_ROCM"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    import cumesh
    from comfy_extras.nodes_mesh_postprocess import _interp_vertex_attr, _rasterize_uv_barycentric, _uv_unwrap
    from cumesh.bvh import cuBVH
    from flex_gemm.ops.grid_sample import grid_sample_3d
    from tools.acceptance.cubql_acceptance import reconstruct_positions
    from tools.acceptance.equirectangular import capture_to_glb_array, direct_map_attributes, raster_uv_to_glb

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = torch.load(args.capture, map_location="cpu", weights_only=False)
    if not capture.get("capture_complete"):
        raise RuntimeError("Capture is incomplete")
    color_map = None
    if args.color_map:
        with Image.open(args.color_map) as image:
            colors = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        color_map = torch.from_numpy(colors).permute(2, 0, 1).contiguous().cuda()

    vertices = capture["vertices"].numpy().astype(np.float32, copy=True)
    faces = capture["faces"].numpy().astype(np.int32, copy=True)
    normals = capture["normals"].numpy().astype(np.float32, copy=True)
    uv = capture["uv"].numpy().astype(np.float32, copy=True)
    texture_size = int(capture["texture_size"])
    resolution = int(capture["resolution"])
    if uv.shape[0] == 0:
        vmapping, faces, uv = _uv_unwrap(capture["vertices"], capture["faces"].long(), "adaptive", 1024, 2, 0.0)
        # _uv_unwrap returns numpy arrays (vmapping int64, faces int64, uv float32);
        # cuBVH requires int32 triangles
        vertices = vertices[np.asarray(vmapping)]
        normals = normals[np.asarray(vmapping)]
        faces = np.ascontiguousarray(faces, dtype=np.int32)
        uv = np.ascontiguousarray(uv, dtype=np.float32)
    uv[:, 1] = 1 - uv[:, 1]

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    face_idx, bary, mask = _rasterize_uv_barycentric(faces, uv, texture_size)
    if not bool(mask.any()):
        raise RuntimeError("UV rasterization produced an empty texture")
    vertices_torch = torch.from_numpy(vertices).float().cuda()
    faces_torch = torch.from_numpy(faces).int().cuda()
    pos = _interp_vertex_attr(vertices_torch, faces_torch.long(), face_idx, bary, mask)
    torch.cuda.synchronize()
    raster_seconds = time.perf_counter() - started

    build_started = time.perf_counter()
    bvh = cuBVH(vertices, faces)
    torch.cuda.synchronize()
    build_seconds = time.perf_counter() - build_started

    valid_pos = pos[mask].contiguous()
    distance_parts = []
    face_parts = []
    uvw_parts = []
    position_parts = []
    attr_parts = []
    query_seconds = 0.0
    sample_seconds = 0.0
    for start in range(0, valid_pos.shape[0], 100001):
        batch = valid_pos[start:start + 100001]
        query_started = time.perf_counter()
        distances, face_ids, uvw = bvh.unsigned_distance(batch, return_uvw=True)
        closest = reconstruct_positions(vertices_torch, faces_torch.long(), face_ids, uvw)
        torch.cuda.synchronize()
        query_seconds += time.perf_counter() - query_started

        sample_started = time.perf_counter()
        if color_map is None:
            attrs = grid_sample_3d(
                capture["voxel_feats"].cuda(),
                capture["voxel_coords"].cuda(),
                shape=torch.Size([*capture["voxel_shape"], *capture["voxel_spatial_shape"]]),
                grid=((closest + 0.5) * resolution).reshape(1, -1, 3),
                mode="trilinear",
            )
        else:
            attrs = direct_map_attributes(color_map, closest)
        torch.cuda.synchronize()
        sample_seconds += time.perf_counter() - sample_started
        distance_parts.append(distances.cpu())
        face_parts.append(face_ids.cpu())
        uvw_parts.append(uvw.cpu())
        position_parts.append(closest.cpu())
        attr_parts.append(attrs.reshape(-1, 6).cpu())

    result = {
        "distances": torch.cat(distance_parts),
        "face_ids": torch.cat(face_parts),
        "uvw": torch.cat(uvw_parts),
        "positions": torch.cat(position_parts),
        "attrs": torch.cat(attr_parts),
        "mask": mask.cpu(),
        "capture_sha256": hashlib.sha256(args.capture.read_bytes()).hexdigest(),
    }
    for key in ("distances", "uvw", "positions", "attrs"):
        if not bool(torch.isfinite(result[key]).all()):
            raise RuntimeError(f"Non-finite values in {key}")

    attrs_map = torch.zeros((*mask.shape, result["attrs"].shape[1]), dtype=result["attrs"].dtype)
    attrs_map[result["mask"]] = result["attrs"]
    attrs_np = attrs_map.numpy()
    base_color = np.clip(attrs_np[..., 0:3] * 255, 0, 255).astype(np.uint8)
    metallic = np.clip(attrs_np[..., 3:4] * 255, 0, 255).astype(np.uint8)
    roughness = np.clip(attrs_np[..., 4:5] * 255, 0, 255).astype(np.uint8)
    alpha = np.clip(attrs_np[..., 5:6] * 255, 0, 255).astype(np.uint8)
    inpaint_mask = (~result["mask"].numpy()).astype(np.uint8)
    base_color = cv2.inpaint(base_color, inpaint_mask, 3, cv2.INPAINT_TELEA)
    metallic = cv2.inpaint(metallic, inpaint_mask, 1, cv2.INPAINT_TELEA)[..., None]
    roughness = cv2.inpaint(roughness, inpaint_mask, 1, cv2.INPAINT_TELEA)[..., None]
    alpha = cv2.inpaint(alpha, inpaint_mask, 1, cv2.INPAINT_TELEA)[..., None]
    base_texture = Image.fromarray(np.concatenate([base_color, alpha], axis=-1))
    mr_texture = Image.fromarray(np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=-1))

    out_vertices = capture_to_glb_array(vertices)
    out_normals = capture_to_glb_array(normals)
    uv = raster_uv_to_glb(uv)
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=base_texture,
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicRoughnessTexture=mr_texture,
        metallicFactor=1.0,
        roughnessFactor=1.0,
        alphaMode="OPAQUE",
        doubleSided=True,
    )
    mesh = trimesh.Trimesh(
        vertices=out_vertices,
        faces=faces,
        vertex_normals=out_normals,
        process=False,
        visual=trimesh.visual.TextureVisuals(uv=uv, material=material),
    )

    stem = f"baton-{args.label}-cubql"
    result_file = output_dir / f"{stem}-queries.pt"
    atlas_file = output_dir / f"{stem}-atlas.png"
    mr_file = output_dir / f"{stem}-metallic-roughness.png"
    glb_file = output_dir / f"{stem}.glb"
    report_file = output_dir / f"{stem}.json"
    torch.save(result, result_file)
    base_texture.save(atlas_file)
    mr_texture.save(mr_file)
    glb_file.write_bytes(mesh.export(file_type="glb"))
    report = {
        "label": args.label,
        "stage": "closest-point-bvh-texture-transfer",
        "cumesh_module": cumesh.__file__,
        "cumesh_backend": cumesh.bvh._backend.__name__,
        "capture": str(args.capture),
        "capture_sha256": result["capture_sha256"],
        "color_source": "generative-voxel-features" if color_map is None else "equirectangular-map",
        "color_map": str(args.color_map) if args.color_map else None,
        "color_map_sha256": hashlib.sha256(args.color_map.read_bytes()).hexdigest() if args.color_map else None,
        "vertices": int(out_vertices.shape[0]),
        "triangles": int(faces.shape[0]),
        "covered_texels": int(result["mask"].sum()),
        "atlas_size": [texture_size, texture_size],
        "glb_bytes": glb_file.stat().st_size,
        "seconds": {
            "total": time.perf_counter() - started,
            "rasterize": raster_seconds,
            "bvh_build": build_seconds,
            "bvh_queries": query_seconds,
            "grid_sample": sample_seconds if color_map is None else None,
            "equirectangular_sample": sample_seconds if color_map is not None else None,
        },
        "vram_bytes": {
            "peak_allocated": int(torch.cuda.max_memory_allocated()),
            "peak_reserved": int(torch.cuda.max_memory_reserved()),
        },
        "fallback_events": [],
        "driver_errors": [],
        "result_file": str(result_file),
        "atlas_file": str(atlas_file),
        "glb_file": str(glb_file),
    }
    report_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
