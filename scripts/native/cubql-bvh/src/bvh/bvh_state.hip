#define CUBQL_GPU_BUILDER_IMPLEMENTATION 1

#include "bvh_state_internal.h"

#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPGuard.h>

#include <memory>
#include <vector>

namespace cumesh_bvh {
namespace {

__global__ void make_boxes(const cuBQL::Triangle *triangles,
                           cuBQL::box3f *boxes, int64_t count) {
  const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const cuBQL::Triangle triangle = triangles[index];
  const cuBQL::vec3f edge1 = triangle.b - triangle.a;
  const cuBQL::vec3f edge2 = triangle.c - triangle.a;
  const cuBQL::vec3f normal = cuBQL::cross(edge1, edge2);
  const float area_twice_squared = cuBQL::dot(normal, normal);
  boxes[index] = area_twice_squared > 0.0f ? triangle.bounds() : cuBQL::box3f();
}

}  // namespace

BVHState::BVHState(const float *vertices, int64_t vertex_count,
                   const int32_t *triangles, int64_t triangle_count) {
  auto state = std::make_unique<DeviceState>();
  state->triangle_count = triangle_count;
  check_hip(hipGetDevice(&state->device), "hipGetDevice(create_cuBVH)");
  const c10::hip::HIPGuard device_guard(state->device);
  const hipStream_t stream = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA(state->device).stream();

  std::vector<cuBQL::Triangle> host_triangles;
  host_triangles.reserve(triangle_count);
  for (int64_t face = 0; face < triangle_count; ++face) {
    const int32_t *ids = triangles + face * 3;
    const float *a = vertices + ids[0] * 3;
    const float *b = vertices + ids[1] * 3;
    const float *c = vertices + ids[2] * 3;
    host_triangles.emplace_back(cuBQL::vec3f(a[0], a[1], a[2]),
                                cuBQL::vec3f(b[0], b[1], b[2]),
                                cuBQL::vec3f(c[0], c[1], c[2]));
  }

  const size_t triangle_bytes = host_triangles.size() * sizeof(cuBQL::Triangle);
  const size_t box_bytes = host_triangles.size() * sizeof(cuBQL::box3f);
  check_hip(hipMalloc(&state->triangles, triangle_bytes), "hipMalloc(triangles)");
  check_hip(hipMalloc(&state->boxes, box_bytes), "hipMalloc(boxes)");
  check_hip(hipMemcpyAsync(state->triangles, host_triangles.data(), triangle_bytes,
                           hipMemcpyHostToDevice, stream),
            "hipMemcpyAsync(triangles)");
  constexpr int threads = 256;
  const int blocks = static_cast<int>((triangle_count + threads - 1) / threads);
  make_boxes<<<blocks, threads, 0, stream>>>(state->triangles, state->boxes,
                                             triangle_count);
  check_hip(hipGetLastError(), "make_boxes launch");
  check_hip(hipStreamSynchronize(stream), "make_boxes synchronization");
  cuBQL::gpuBuilder(state->bvh, state->boxes,
                    static_cast<uint32_t>(triangle_count), cuBQL::BuildConfig(),
                    stream, state->memory);
  check_hip(hipGetLastError(), "cuBQL::gpuBuilder");
  check_hip(hipStreamSynchronize(stream), "cuBQL::gpuBuilder synchronization");
  check_hip(hipFree(state->boxes), "hipFree(boxes after build)");
  state->boxes = nullptr;
  state_ = state.release();
}

BVHState::~BVHState() {
  if (state_ == nullptr) return;
  const c10::hip::HIPGuard device_guard(state_->device);
  const hipStream_t stream = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA(state_->device).stream();
  cuBQL::cuda::free(state_->bvh, stream, state_->memory);
  if (state_->boxes != nullptr) hipFree(state_->boxes);
  if (state_->triangles != nullptr) hipFree(state_->triangles);
  delete state_;
}

}  // namespace cumesh_bvh
