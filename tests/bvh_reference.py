"""Small independent NumPy geometry references used only by tests."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RayHit:
    depth: float
    uvw: np.ndarray


@dataclass(frozen=True)
class ClosestResult:
    distance: float
    face_id: int
    point: np.ndarray
    uvw: np.ndarray


def ray_triangle(origin, direction, triangle):
    """Moller-Trumbore intersection, with depth in direction-vector units."""
    origin = np.asarray(origin, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    a, b, c = np.asarray(triangle, dtype=np.float64)
    ab, ac = b - a, c - a
    pvec = np.cross(direction, ac)
    determinant = float(np.dot(ab, pvec))
    if abs(determinant) < 1e-10:
        return None
    inv_determinant = 1.0 / determinant
    tvec = origin - a
    u = float(np.dot(tvec, pvec)) * inv_determinant
    if u < -1e-9 or u > 1.0 + 1e-9:
        return None
    qvec = np.cross(tvec, ab)
    v = float(np.dot(direction, qvec)) * inv_determinant
    if v < -1e-9 or u + v > 1.0 + 1e-9:
        return None
    depth = float(np.dot(ac, qvec)) * inv_determinant
    if depth < 0.0:
        return None
    return RayHit(depth, np.array([1.0 - u - v, u, v], dtype=np.float64))


def _closest_point_on_triangle(point, triangle):
    a, b, c = np.asarray(triangle, dtype=np.float64)
    ab, ac = b - a, c - a
    ap = point - a
    d1, d2 = float(np.dot(ab, ap)), float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return a.copy(), np.array([1.0, 0.0, 0.0])

    bp = point - b
    d3, d4 = float(np.dot(ab, bp)), float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return b.copy(), np.array([0.0, 1.0, 0.0])

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + v * ab, np.array([1.0 - v, v, 0.0])

    cp = point - c
    d5, d6 = float(np.dot(ab, cp)), float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return c.copy(), np.array([0.0, 0.0, 1.0])

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + w * ac, np.array([1.0 - w, 0.0, w])

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and d4 - d3 >= 0.0 and d5 - d6 >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b), np.array([0.0, 1.0 - w, w])

    denominator = 1.0 / (va + vb + vc)
    v, w = vb * denominator, vc * denominator
    return a + v * ab + w * ac, np.array([1.0 - v - w, v, w])


def closest_point(point, triangles):
    point = np.asarray(point, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.float64)
    best = None
    best_squared = np.inf
    for face_id, triangle in enumerate(triangles):
        a, b, c = triangle
        if np.linalg.norm(np.cross(b - a, c - a)) <= 1e-12:
            continue
        candidate, uvw = _closest_point_on_triangle(point, triangle)
        squared = float(np.dot(candidate - point, candidate - point))
        if squared < best_squared:
            best_squared = squared
            best = ClosestResult(float(np.sqrt(squared)), face_id, candidate, uvw)
    if best is None:
        raise ValueError("Mesh contains no non-degenerate triangles")
    return best


def ray_trace(origin, direction, triangles):
    origin = np.asarray(origin, dtype=np.float64)
    on_surface = closest_point(origin, triangles)
    if on_surface.distance <= 1e-8:
        return origin.copy(), on_surface.face_id, 0.0
    best_face = -1
    best_depth = 1000.0
    for face_id, triangle in enumerate(np.asarray(triangles, dtype=np.float64)):
        hit = ray_triangle(origin, direction, triangle)
        if hit is not None and hit.depth < best_depth:
            best_face, best_depth = face_id, hit.depth
    position = np.asarray(origin, dtype=np.float64) + best_depth * np.asarray(direction, dtype=np.float64)
    return position, best_face, best_depth
