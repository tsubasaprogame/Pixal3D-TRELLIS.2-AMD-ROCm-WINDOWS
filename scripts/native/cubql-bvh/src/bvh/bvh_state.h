#pragma once

#include <torch/extension.h>

#include <cstdint>
#include <memory>

#if defined(__HIPCC__) || defined(__CUDACC__)
#include <hip/hip_runtime.h>
#endif

namespace cumesh_bvh {

#if defined(__HIPCC__) || defined(__CUDACC__)
inline void check_hip(hipError_t status, const char *operation) {
  TORCH_CHECK(status == hipSuccess, operation, ": ", hipGetErrorString(status));
}
#endif

class BVHState {
 public:
  BVHState(const float *vertices, int64_t vertex_count,
           const int32_t *triangles, int64_t triangle_count);
  ~BVHState();

  BVHState(const BVHState &) = delete;
  BVHState &operator=(const BVHState &) = delete;

  void ray_trace(at::Tensor rays_o, at::Tensor rays_d,
                 at::Tensor positions, at::Tensor face_id, at::Tensor depth);
  void unsigned_distance(at::Tensor points, at::Tensor distances,
                         at::Tensor face_id, c10::optional<at::Tensor> uvw);
  void signed_distance(at::Tensor points, at::Tensor distances,
                       at::Tensor face_id, c10::optional<at::Tensor> uvw,
                       int64_t mode);

 private:
  struct DeviceState;
  DeviceState *state_ = nullptr;
};

}  // namespace cumesh_bvh
