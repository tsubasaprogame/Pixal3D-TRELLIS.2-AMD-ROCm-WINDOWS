"""Sample an equirectangular color map at cuBQL capture-space positions."""
import math

import numpy as np
import torch


def capture_to_glb_array(values: np.ndarray) -> np.ndarray:
    """Convert capture coordinates to right-handed GLB world coordinates."""
    values = np.asarray(values)
    if values.shape[-1] != 3:
        raise ValueError("values must end in three coordinates")
    converted = values.copy()
    converted[..., 1] = values[..., 2]
    converted[..., 2] = -values[..., 1]
    return converted


def raster_uv_to_glb(uv: np.ndarray) -> np.ndarray:
    """Convert rasterizer V to the glTF orientation used by this exporter."""
    converted = np.asarray(uv, dtype=np.float32).copy()
    converted[..., 1] = 1.0 - converted[..., 1]
    return converted


def sample_capture_positions(color_map: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample CHW colors; longitude wraps and latitude clamps.

    The GLB export transform maps capture coordinates to world coordinates as
    ``(x, z, -y)``. World +Y is north and longitude increases toward -Z, so
    east appears on screen-right when Greenwich is viewed from +X.
    """
    if color_map.ndim != 3:
        raise ValueError("color_map must have CHW dimensions")
    if positions.shape[-1] != 3:
        raise ValueError("positions must end in three coordinates")
    if color_map.device != positions.device:
        raise ValueError("color_map and positions must share a device")

    world = torch.stack((positions[..., 0], positions[..., 2], -positions[..., 1]), dim=-1)
    world = torch.nn.functional.normalize(world, p=2, dim=-1, eps=1e-12)
    longitude = torch.atan2(-world[..., 2], world[..., 0])
    latitude = torch.asin(world[..., 1].clamp(-1.0, 1.0))

    _, height, width = color_map.shape
    u = ((longitude + math.pi) / (2 * math.pi) * width - 0.5) % width
    v = ((math.pi / 2 - latitude) / math.pi * height - 0.5).clamp(0, height - 1)
    u0 = torch.floor(u).long() % width
    u1 = (u0 + 1) % width
    v0 = torch.floor(v).long()
    v1 = (v0 + 1).clamp(max=height - 1)
    fu = (u - torch.floor(u)).unsqueeze(-1)
    fv = (v - torch.floor(v)).unsqueeze(-1)

    def sample(rows, columns):
        return color_map[:, rows, columns].movedim(0, -1)

    top = sample(v0, u0) * (1 - fu) + sample(v0, u1) * fu
    bottom = sample(v1, u0) * (1 - fu) + sample(v1, u1) * fu
    return top * (1 - fv) + bottom * fv


def direct_map_attributes(color_map: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Return RGB, metallic, roughness and alpha for an authoritative color map."""
    attrs = positions.new_zeros((*positions.shape[:-1], 6))
    attrs[..., :3] = sample_capture_positions(color_map, positions)
    attrs[..., 4] = 0.85
    attrs[..., 5] = 1.0
    return attrs
