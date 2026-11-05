import numpy as np
import torch


def _cpu_tensor(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    return torch.from_numpy(np.ascontiguousarray(value)).cpu()


def capture_inputs(mesh, pbr_voxel, resolution, texture_size):
    uv = getattr(mesh.visual, "uv", None) if hasattr(mesh, "visual") else None
    return {
        "vertices": _cpu_tensor(mesh.vertices),
        "faces": _cpu_tensor(mesh.faces),
        "normals": _cpu_tensor(np.asarray(mesh.vertex_normals)),
        "uv": _cpu_tensor(uv) if uv is not None else torch.empty((0, 2), dtype=torch.float32),
        "voxel_feats": _cpu_tensor(pbr_voxel.feats),
        "voxel_coords": _cpu_tensor(pbr_voxel.coords),
        "voxel_shape": [int(value) for value in pbr_voxel.shape],
        "voxel_spatial_shape": [int(value) for value in pbr_voxel.spatial_shape],
        "resolution": int(resolution),
        "texture_size": int(texture_size),
    }


def reconstruct_positions(vertices, faces, face_ids, uvw):
    triangles = vertices[faces[face_ids]]
    return (uvw.unsqueeze(-1) * triangles).sum(dim=1)


def compare_backend_results(old, new, distance_tolerance=2e-5, position_tolerance=2e-5, attr_tolerance=1e-4):
    keys = ("distances", "face_ids", "uvw", "positions", "attrs")
    finite = all(bool(torch.isfinite(old[key]).all() and torch.isfinite(new[key]).all()) for key in keys if key != "face_ids")
    max_distance = float((old["distances"] - new["distances"]).abs().max())
    max_uvw = float((old["uvw"] - new["uvw"]).abs().max())
    max_position = float((old["positions"] - new["positions"]).abs().max())
    max_attr = float((old["attrs"] - new["attrs"]).abs().max())
    face_mismatches = int((old["face_ids"] != new["face_ids"]).sum())
    return {
        "passed": finite and max_distance <= distance_tolerance and max_position <= position_tolerance and max_attr <= attr_tolerance,
        "all_finite": finite,
        "face_mismatches": face_mismatches,
        "max_distance_abs_diff": max_distance,
        "max_uvw_abs_diff": max_uvw,
        "max_position_abs_diff": max_position,
        "max_attr_abs_diff": max_attr,
        "tolerances": {
            "distance": distance_tolerance,
            "position": position_tolerance,
            "attrs": attr_tolerance,
        },
    }
