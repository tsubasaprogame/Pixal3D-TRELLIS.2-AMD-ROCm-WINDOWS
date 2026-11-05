#pragma once

#include "bvh_state.h"

#include <cuBQL/bvh.h>
#include <cuBQL/queries/triangleData/Triangle.h>

namespace cumesh_bvh {

struct BVHState::DeviceState {
  cuBQL::Triangle *triangles = nullptr;
  cuBQL::box3f *boxes = nullptr;
  cuBQL::bvh3f bvh;
  cuBQL::DeviceMemoryResource memory;
  int64_t triangle_count = 0;
  int device = -1;
};

}  // namespace cumesh_bvh
