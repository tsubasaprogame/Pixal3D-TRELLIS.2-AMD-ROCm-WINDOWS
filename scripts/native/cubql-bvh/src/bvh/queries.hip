#include "bvh_state_internal.h"

#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPGuard.h>
#include <cuBQL/traversal/rayQueries.h>
#include <cuBQL/traversal/shrinkingRadiusQuery.h>

namespace cumesh_bvh {
namespace {

struct ClosestPoint {
  cuBQL::vec3f point;
  cuBQL::vec3f uvw;
};

struct ClosestHit {
  float squared_distance = CUBQL_INF;
  uint32_t face_id = 0;
  cuBQL::vec3f uvw = cuBQL::vec3f(1.0f, 0.0f, 0.0f);
  cuBQL::vec3f normal = cuBQL::vec3f(0.0f, 0.0f, 0.0f);
};

__device__ bool non_degenerate(const cuBQL::Triangle &triangle) {
  const cuBQL::vec3f normal = cuBQL::cross(triangle.b - triangle.a,
                                           triangle.c - triangle.a);
  return cuBQL::dot(normal, normal) > 0.0f;
}

__device__ bool intersect_triangle(const cuBQL::vec3f &origin,
                                   const cuBQL::vec3f &direction,
                                   const cuBQL::Triangle &triangle,
                                   float &depth) {
  const cuBQL::vec3f ab = triangle.b - triangle.a;
  const cuBQL::vec3f ac = triangle.c - triangle.a;
  const cuBQL::vec3f pvec = cuBQL::cross(direction, ac);
  const float determinant = cuBQL::dot(ab, pvec);
  if (fabsf(determinant) < 1e-10f) return false;
  const float inverse = 1.0f / determinant;
  const cuBQL::vec3f tvec = origin - triangle.a;
  const float u = cuBQL::dot(tvec, pvec) * inverse;
  if (u < 0.0f || u > 1.0f) return false;
  const cuBQL::vec3f qvec = cuBQL::cross(tvec, ab);
  const float v = cuBQL::dot(direction, qvec) * inverse;
  if (v < 0.0f || u + v > 1.0f) return false;
  depth = cuBQL::dot(ac, qvec) * inverse;
  return depth >= 0.0f;
}

__device__ ClosestPoint closest_point_on_triangle(
    const cuBQL::vec3f &point, const cuBQL::Triangle &triangle) {
  const cuBQL::vec3f &a = triangle.a;
  const cuBQL::vec3f &b = triangle.b;
  const cuBQL::vec3f &c = triangle.c;
  const cuBQL::vec3f ab = b - a;
  const cuBQL::vec3f ac = c - a;
  const cuBQL::vec3f ap = point - a;
  const float d1 = cuBQL::dot(ab, ap);
  const float d2 = cuBQL::dot(ac, ap);
  if (d1 <= 0.0f && d2 <= 0.0f)
    return {a, cuBQL::vec3f(1.0f, 0.0f, 0.0f)};

  const cuBQL::vec3f bp = point - b;
  const float d3 = cuBQL::dot(ab, bp);
  const float d4 = cuBQL::dot(ac, bp);
  if (d3 >= 0.0f && d4 <= d3)
    return {b, cuBQL::vec3f(0.0f, 1.0f, 0.0f)};

  const float vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
    const float v = d1 / (d1 - d3);
    return {a + v * ab, cuBQL::vec3f(1.0f - v, v, 0.0f)};
  }

  const cuBQL::vec3f cp = point - c;
  const float d5 = cuBQL::dot(ab, cp);
  const float d6 = cuBQL::dot(ac, cp);
  if (d6 >= 0.0f && d5 <= d6)
    return {c, cuBQL::vec3f(0.0f, 0.0f, 1.0f)};

  const float vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
    const float w = d2 / (d2 - d6);
    return {a + w * ac, cuBQL::vec3f(1.0f - w, 0.0f, w)};
  }

  const float va = d3 * d6 - d5 * d4;
  if (va <= 0.0f && d4 - d3 >= 0.0f && d5 - d6 >= 0.0f) {
    const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    return {b + w * (c - b), cuBQL::vec3f(0.0f, 1.0f - w, w)};
  }

  const float denominator = 1.0f / (va + vb + vc);
  const float v = vb * denominator;
  const float w = vc * denominator;
  return {a + v * ab + w * ac, cuBQL::vec3f(1.0f - v - w, v, w)};
}

__device__ void visit_closest_triangle(const cuBQL::vec3f &point,
                                       const cuBQL::Triangle &triangle,
                                       uint32_t face_id, ClosestHit &hit) {
  if (!non_degenerate(triangle)) return;
  const ClosestPoint candidate = closest_point_on_triangle(point, triangle);
  const cuBQL::vec3f difference = candidate.point - point;
  const float squared_distance = cuBQL::dot(difference, difference);
  if (squared_distance < hit.squared_distance ||
      (squared_distance == hit.squared_distance && face_id < hit.face_id)) {
    hit.squared_distance = squared_distance;
    hit.face_id = face_id;
    hit.uvw = candidate.uvw;
    hit.normal = cuBQL::cross(triangle.b - triangle.a, triangle.c - triangle.a);
  }
}

__device__ ClosestHit find_closest(const cuBQL::bvh3f &bvh,
                                   const cuBQL::Triangle *triangles,
                                   const cuBQL::vec3f &point) {
  ClosestHit hit;
  auto visit = [&](uint32_t face_id) -> float {
    visit_closest_triangle(point, triangles[face_id], face_id, hit);
    return hit.squared_distance;
  };
  cuBQL::shrinkingRadiusQuery::forEachPrim<float, 3>(visit, bvh, point,
                                                      CUBQL_INF);
  return hit;
}

__global__ void ray_trace_kernel(const cuBQL::Triangle *triangles,
                                 cuBQL::bvh3f bvh,
                                 const float *origins, const float *directions,
                                 float *positions, int64_t *face_ids,
                                 float *depths, int64_t count) {
  const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const cuBQL::vec3f origin(origins[index * 3], origins[index * 3 + 1],
                            origins[index * 3 + 2]);
  const cuBQL::vec3f direction(directions[index * 3], directions[index * 3 + 1],
                               directions[index * 3 + 2]);
  const ClosestHit on_surface = find_closest(bvh, triangles, origin);
  if (on_surface.squared_distance <= 1e-16f) {
    face_ids[index] = static_cast<int64_t>(on_surface.face_id);
    depths[index] = 0.0f;
    positions[index * 3] = origin.x;
    positions[index * 3 + 1] = origin.y;
    positions[index * 3 + 2] = origin.z;
    return;
  }
  cuBQL::ray3f ray(origin, direction, 0.0f, 1000.0f);
  int64_t best_face = -1;
  auto visit = [&](uint32_t face_id) -> float {
    const cuBQL::Triangle &triangle = triangles[face_id];
    if (!non_degenerate(triangle)) return ray.tMax;
    float depth = 0.0f;
    if (intersect_triangle(origin, direction, triangle, depth) &&
        (depth < ray.tMax ||
         (depth == ray.tMax && (best_face < 0 || face_id < best_face)))) {
      best_face = static_cast<int64_t>(face_id);
      return depth;
    }
    return ray.tMax;
  };
  cuBQL::shrinkingRayQuery::forEachPrim(visit, bvh, ray);
  face_ids[index] = best_face;
  depths[index] = ray.tMax;
  const cuBQL::vec3f position = origin + ray.tMax * direction;
  positions[index * 3] = position.x;
  positions[index * 3 + 1] = position.y;
  positions[index * 3 + 2] = position.z;
}

__global__ void unsigned_distance_kernel(const cuBQL::Triangle *triangles,
                                         cuBQL::bvh3f bvh, const float *points,
                                         float *distances, int64_t *face_ids,
                                         float *uvw, int64_t count) {
  const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const cuBQL::vec3f point(points[index * 3], points[index * 3 + 1],
                           points[index * 3 + 2]);
  const ClosestHit hit = find_closest(bvh, triangles, point);
  distances[index] = sqrtf(hit.squared_distance);
  face_ids[index] = static_cast<int64_t>(hit.face_id);
  if (uvw != nullptr) {
    uvw[index * 3] = hit.uvw.x;
    uvw[index * 3 + 1] = hit.uvw.y;
    uvw[index * 3 + 2] = hit.uvw.z;
  }
}

__global__ void signed_distance_kernel(const cuBQL::Triangle *triangles,
                                       cuBQL::bvh3f bvh, const float *points,
                                       float *distances, int64_t *face_ids,
                                       float *uvw, int64_t count, int mode) {
  const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const cuBQL::vec3f point(points[index * 3], points[index * 3 + 1],
                           points[index * 3 + 2]);
  const ClosestHit hit = find_closest(bvh, triangles, point);

  // A deterministic slightly perturbed +X ray avoids systematic double hits
  // on triangle diagonals while retaining the legacy parity convention.
  const cuBQL::vec3f direction(1.0f, 0.0001220703125f, 0.00006103515625f);
  cuBQL::ray3f ray(point, direction, 1e-6f, CUBQL_INF);
  int crossing_count = 0;
  auto visit = [&](uint32_t face_id) -> int {
    const cuBQL::Triangle &triangle = triangles[face_id];
    if (!non_degenerate(triangle)) return CUBQL_CONTINUE_TRAVERSAL;
    float depth = 0.0f;
    if (intersect_triangle(point, direction, triangle, depth) && depth > 1e-6f)
      ++crossing_count;
    return CUBQL_CONTINUE_TRAVERSAL;
  };
  cuBQL::fixedRayQuery::forEachPrim(visit, bvh, ray);

  bool negative = (crossing_count & 1) != 0;
  if (mode == 0 && crossing_count == 0) {
    // Open surfaces have no globally defined inside. Preserve the observed
    // watertight convention by orienting their nearest face; closed meshes
    // with no forward crossings are outside and have positive orientation.
    const cuBQL::Triangle &triangle = triangles[hit.face_id];
    const cuBQL::vec3f closest = hit.uvw.x * triangle.a +
                                 hit.uvw.y * triangle.b +
                                 hit.uvw.z * triangle.c;
    negative = cuBQL::dot(point - closest, hit.normal) < 0.0f;
  }
  const float magnitude = sqrtf(hit.squared_distance);
  distances[index] = negative ? -magnitude : magnitude;
  face_ids[index] = static_cast<int64_t>(hit.face_id);
  if (uvw != nullptr) {
    uvw[index * 3] = hit.uvw.x;
    uvw[index * 3 + 1] = hit.uvw.y;
    uvw[index * 3 + 2] = hit.uvw.z;
  }
}

void check_vector3(const at::Tensor &tensor, const char *name, int device) {
  TORCH_CHECK(tensor.defined(), name, " must be defined");
  TORCH_CHECK(tensor.dim() == 2 && tensor.size(1) == 3,
              name, " must have shape [N, 3]");
  TORCH_CHECK(tensor.scalar_type() == at::kFloat,
              name, " must have dtype float32");
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA/HIP tensor");
  TORCH_CHECK(tensor.get_device() == device,
              name, " must be on the BVH CUDA device");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_float_output(const at::Tensor &tensor, const char *name,
                        int64_t count, int device) {
  TORCH_CHECK(tensor.defined(), name, " must be defined");
  TORCH_CHECK(tensor.dim() == 1 && tensor.size(0) == count,
              name, " must have shape [N]");
  TORCH_CHECK(tensor.scalar_type() == at::kFloat,
              name, " must have dtype float32");
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA/HIP tensor");
  TORCH_CHECK(tensor.get_device() == device,
              name, " must be on the BVH CUDA device");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_face_output(const at::Tensor &tensor, int64_t count, int device) {
  TORCH_CHECK(tensor.defined(), "face_id must be defined");
  TORCH_CHECK(tensor.dim() == 1 && tensor.size(0) == count,
              "face_id must have shape [N]");
  TORCH_CHECK(tensor.scalar_type() == at::kLong,
              "face_id must have dtype int64");
  TORCH_CHECK(tensor.is_cuda(), "face_id must be a CUDA/HIP tensor");
  TORCH_CHECK(tensor.get_device() == device,
              "face_id must be on the BVH CUDA device");
  TORCH_CHECK(tensor.is_contiguous(), "face_id must be contiguous");
}

void check_distance_tensors(const at::Tensor &points,
                            const at::Tensor &distances,
                            const at::Tensor &face_id,
                            const c10::optional<at::Tensor> &uvw, int device) {
  check_vector3(points, "points", device);
  const int64_t count = points.size(0);
  check_float_output(distances, "distances", count, device);
  check_face_output(face_id, count, device);
  if (uvw.has_value()) check_vector3(*uvw, "uvw", device);
  if (uvw.has_value())
    TORCH_CHECK(uvw->size(0) == count, "uvw must have the same N as points");
}

int launch_blocks(int64_t count) {
  constexpr int threads = 256;
  return static_cast<int>((count + threads - 1) / threads);
}

}  // namespace

void BVHState::ray_trace(at::Tensor rays_o, at::Tensor rays_d,
                         at::Tensor positions, at::Tensor face_id,
                         at::Tensor depth) {
  const c10::hip::HIPGuard device_guard(state_->device);
  check_vector3(rays_o, "rays_o", state_->device);
  check_vector3(rays_d, "rays_d", state_->device);
  TORCH_CHECK(rays_o.sizes() == rays_d.sizes(),
              "rays_o and rays_d must have the same shape");
  const int64_t count = rays_o.size(0);
  check_vector3(positions, "positions", state_->device);
  TORCH_CHECK(positions.size(0) == count, "positions must have the same N as rays_o");
  check_face_output(face_id, count, state_->device);
  check_float_output(depth, "depth", count, state_->device);
  if (count == 0) return;
  const hipStream_t stream =
      c10::hip::getCurrentHIPStreamMasqueradingAsCUDA(state_->device).stream();
  ray_trace_kernel<<<launch_blocks(count), 256, 0, stream>>>(
      state_->triangles, state_->bvh, rays_o.data_ptr<float>(),
      rays_d.data_ptr<float>(), positions.data_ptr<float>(),
      face_id.data_ptr<int64_t>(), depth.data_ptr<float>(), count);
  check_hip(hipGetLastError(), "ray_trace_kernel launch");
}

void BVHState::unsigned_distance(at::Tensor points, at::Tensor distances,
                                 at::Tensor face_id,
                                 c10::optional<at::Tensor> uvw) {
  const c10::hip::HIPGuard device_guard(state_->device);
  check_distance_tensors(points, distances, face_id, uvw, state_->device);
  const int64_t count = points.size(0);
  if (count == 0) return;
  const hipStream_t stream =
      c10::hip::getCurrentHIPStreamMasqueradingAsCUDA(state_->device).stream();
  unsigned_distance_kernel<<<launch_blocks(count), 256, 0, stream>>>(
      state_->triangles, state_->bvh, points.data_ptr<float>(),
      distances.data_ptr<float>(), face_id.data_ptr<int64_t>(),
      uvw.has_value() ? uvw->data_ptr<float>() : nullptr, count);
  check_hip(hipGetLastError(), "unsigned_distance_kernel launch");
}

void BVHState::signed_distance(at::Tensor points, at::Tensor distances,
                               at::Tensor face_id,
                               c10::optional<at::Tensor> uvw, int64_t mode) {
  TORCH_CHECK(mode == 0 || mode == 1, "SDF mode must be 0 (watertight) or 1 (raystab)");
  const c10::hip::HIPGuard device_guard(state_->device);
  check_distance_tensors(points, distances, face_id, uvw, state_->device);
  const int64_t count = points.size(0);
  if (count == 0) return;
  const hipStream_t stream =
      c10::hip::getCurrentHIPStreamMasqueradingAsCUDA(state_->device).stream();
  signed_distance_kernel<<<launch_blocks(count), 256, 0, stream>>>(
      state_->triangles, state_->bvh, points.data_ptr<float>(),
      distances.data_ptr<float>(), face_id.data_ptr<int64_t>(),
      uvw.has_value() ? uvw->data_ptr<float>() : nullptr, count,
      static_cast<int>(mode));
  check_hip(hipGetLastError(), "signed_distance_kernel launch");
}

}  // namespace cumesh_bvh
