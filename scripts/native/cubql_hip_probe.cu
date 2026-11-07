// Minimal independent HIP/cuBQL geometry probe for the approved gfx1201 runtime.
#define CUBQL_GPU_BUILDER_IMPLEMENTATION 1
#define CUBQL_TRIANGLE_CPAT_IMPLEMENTATION 1
#include "cuBQL/bvh.h"
#include "cuBQL/queries/triangleData/closestPointOnAnyTriangle.h"

#include <cmath>
#include <iostream>

using cuBQL::Triangle;
using cuBQL::box3f;
using cuBQL::bvh3f;
using cuBQL::vec3f;

struct ProbeResult {
  int triangleID;
  vec3f point;
  float sqrDist;
};

__global__ void generateBoxes(box3f *boxes, const Triangle *triangles, int count) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < count) boxes[index] = triangles[index].bounds();
}

__global__ void runQuery(bvh3f bvh, const Triangle *triangles, ProbeResult *result) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  cuBQL::triangles::CPAT query;
  query.runQuery(triangles, bvh, vec3f(0.25f, 0.25f, 1.0f));
  result->triangleID = query.triangleIdx;
  result->point = query.P;
  result->sqrDist = query.sqrDist;
}

int main() {
  Triangle hostTriangle(
      vec3f(0.0f, 0.0f, 0.0f),
      vec3f(1.0f, 0.0f, 0.0f),
      vec3f(0.0f, 1.0f, 0.0f));
  Triangle *deviceTriangles = nullptr;
  box3f *deviceBoxes = nullptr;
  ProbeResult *result = nullptr;
  bvh3f bvh;

  CUBQL_CUDA_CALL(Malloc((void **)&deviceTriangles, sizeof(Triangle)));
  CUBQL_CUDA_CALL(Malloc((void **)&deviceBoxes, sizeof(box3f)));
  CUBQL_CUDA_CALL(MallocManaged((void **)&result, sizeof(ProbeResult)));
  CUBQL_CUDA_CALL(Memcpy(deviceTriangles, &hostTriangle, sizeof(Triangle), cudaMemcpyDefault));
  generateBoxes<<<1, 1>>>(deviceBoxes, deviceTriangles, 1);
  CUBQL_CUDA_SYNC_CHECK();
  cuBQL::gpuBuilder(bvh, deviceBoxes, 1, cuBQL::BuildConfig());
  runQuery<<<1, 1>>>(bvh, deviceTriangles, result);
  CUBQL_CUDA_SYNC_CHECK();

  const float tolerance = 2e-5f;
  bool valid = result->triangleID == 0
      && std::fabs(result->point.x - 0.25f) <= tolerance
      && std::fabs(result->point.y - 0.25f) <= tolerance
      && std::fabs(result->point.z) <= tolerance
      && std::fabs(result->sqrDist - 1.0f) <= tolerance;
  std::cout << "triangle=" << result->triangleID
            << " point=(" << result->point.x << "," << result->point.y << "," << result->point.z << ")"
            << " sqrDist=" << result->sqrDist << std::endl;

  CUBQL_CUDA_CALL(Free(deviceBoxes));
  CUBQL_CUDA_CALL(Free(deviceTriangles));
  CUBQL_CUDA_CALL(Free(result));
  cuBQL::cuda::free(bvh);
  CUBQL_CUDA_SYNC_CHECK();
  if (!valid) {
    std::cerr << "CUBQL_HIP_PROBE_MISMATCH" << std::endl;
    return 1;
  }
  std::cout << "CUBQL_HIP_PROBE_OK" << std::endl;
  return 0;
}
