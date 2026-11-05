"""Capture the stable cubvh backend without installing or replacing its wheel."""
import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OLD_ROOT = (ROOT / "../afamd/wheels/old-cumesh").resolve()
sys.path.insert(0, str(OLD_ROOT))
sys.modules["tqdm"] = types.SimpleNamespace(tqdm=lambda value, *args, **kwargs: value)

import torch
from cumesh import cuBVH
from cumesh import bvh as bvh_wrapper


def cube_mesh():
    vertices = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ], dtype=np.float32)
    faces = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
    ], dtype=np.int32)
    return vertices, faces


def degenerate_cube_mesh():
    vertices, faces = cube_mesh()
    vertices = np.vstack((vertices, np.zeros((1, 3), dtype=np.float32)))
    faces = np.vstack((faces, np.array([[8, 8, 8]], dtype=np.int32)))
    return vertices, faces


def open_grid_mesh():
    vertices = np.array([[x, y, 0] for y in range(4) for x in range(4)], dtype=np.float32)
    faces = []
    for y in range(3):
        for x in range(3):
            a = y * 4 + x
            b, c, d = a + 1, a + 5, a + 4
            faces.extend(([a, b, c], [a, c, d]))
    return vertices, np.array(faces, dtype=np.int32)


def numpy_values(tensors):
    return [tensor.detach().cpu().numpy() for tensor in tensors]


def capture_mesh(name, vertices, faces, points, ray_origins, ray_directions):
    backend = cuBVH(vertices, faces)
    points_gpu = torch.from_numpy(points).cuda()
    origins_gpu = torch.from_numpy(ray_origins).cuda()
    directions_gpu = torch.from_numpy(ray_directions).cuda()
    result = {}
    result[f"{name}_ray_positions"], result[f"{name}_ray_face_id"], result[f"{name}_ray_depth"] = numpy_values(
        backend.ray_trace(origins_gpu, directions_gpu)
    )
    udf_without_uvw = backend.unsigned_distance(points_gpu, return_uvw=False)
    assert udf_without_uvw[2] is None
    result[f"{name}_udf"], result[f"{name}_udf_face_id"] = numpy_values(udf_without_uvw[:2])
    udf, face_id, uvw = backend.unsigned_distance(points_gpu, return_uvw=True)
    result[f"{name}_udf_uvw"], result[f"{name}_udf_uvw_face_id"], result[f"{name}_uvw"] = numpy_values(
        (udf, face_id, uvw)
    )
    for mode in ("watertight", "raystab"):
        distances, face_ids, uvw = backend.signed_distance(points_gpu, return_uvw=True, mode=mode)
        result[f"{name}_{mode}"], result[f"{name}_{mode}_face_id"], result[f"{name}_{mode}_uvw"] = numpy_values(
            (distances, face_ids, uvw)
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data/bvh-golden/v1")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    points = np.array([
        [0, 0, 0], [2, 0, 0], [1, 0, 0], [1, 1, 0.2], [1.1, 1.1, 1.1],
        [0.25, 0.5, 2], [0.25, 0.25, 1], [0.25, 0.25, -1], [2, 2, 1],
    ], dtype=np.float32)
    ray_origins = np.array([[0.25, 0.25, 1], [2, 2, 1], [0, 0, 0], [1, 0, 0]], dtype=np.float32)
    ray_directions = np.array([[0, 0, -2], [0, 0, -1], [1, 0, 0], [-1, 0, 0]], dtype=np.float32)

    inputs, outputs = {}, {}
    factories = (("cube", cube_mesh), ("open", open_grid_mesh), ("degenerate", degenerate_cube_mesh))
    for name, factory in factories:
        vertices, faces = factory()
        inputs[f"{name}_vertices"], inputs[f"{name}_faces"] = vertices, faces
        outputs.update(capture_mesh(name, vertices, faces, points, ray_origins, ray_directions))
    inputs.update(points=points, ray_origins=ray_origins, ray_directions=ray_directions)

    assert np.isfinite(outputs["cube_ray_depth"]).all()
    assert int(outputs["cube_ray_face_id"][1]) == -1
    assert float(outputs["cube_ray_depth"][1]) == 1000.0
    assert abs(float(outputs["cube_udf"][1]) - 1.0) <= 2e-5
    assert abs(float(outputs["cube_watertight"][0]) + 1.0) <= 2e-5

    np.savez_compressed(args.output / "inputs.npz", **inputs)
    np.savez_compressed(args.output / "outputs.npz", **outputs)
    metadata = {
        "schemaVersion": 1,
        "backend": bvh_wrapper._backend.__name__,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "tolerance": 2e-5,
        "meshes": ["cube", "open", "degenerate"],
        "queries": len(points),
        "rays": len(ray_origins),
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
