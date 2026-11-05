#include <torch/extension.h>

#include <pybind11/numpy.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>

#include "bvh_state.h"

namespace py = pybind11;
using cumesh_bvh::BVHState;

namespace {

std::shared_ptr<BVHState> create_cu_bvh(
    py::array_t<float, py::array::c_style> vertices,
    py::array_t<int32_t, py::array::c_style> triangles) {
  TORCH_CHECK(vertices.ndim() == 2 && vertices.shape(1) == 3,
              "vertices must have shape [N, 3] and dtype float32");
  TORCH_CHECK(triangles.ndim() == 2 && triangles.shape(1) == 3,
              "triangles must have shape [M, 3] and dtype int32");
  const int64_t vertex_count = vertices.shape(0);
  const int64_t triangle_count = triangles.shape(0);
  TORCH_CHECK(vertex_count > 0, "vertices must not be empty");
  TORCH_CHECK(triangle_count > 0, "triangles must not be empty");
  TORCH_CHECK(vertex_count <= std::numeric_limits<uint32_t>::max() &&
                  triangle_count <= std::numeric_limits<uint32_t>::max(),
              "mesh exceeds the 32-bit cuBQL index range");

  const float *vertex_data = vertices.data();
  const int32_t *triangle_data = triangles.data();
  for (int64_t i = 0; i < vertex_count * 3; ++i) {
    TORCH_CHECK(std::isfinite(vertex_data[i]), "vertices must be finite");
  }

  bool has_non_degenerate_triangle = false;
  for (int64_t face = 0; face < triangle_count; ++face) {
    const int32_t *ids = triangle_data + face * 3;
    for (int corner = 0; corner < 3; ++corner) {
      TORCH_CHECK(ids[corner] >= 0 && ids[corner] < vertex_count,
                  "triangle index out of bounds at face ", face);
    }
    const float *a = vertex_data + ids[0] * 3;
    const float *b = vertex_data + ids[1] * 3;
    const float *c = vertex_data + ids[2] * 3;
    double normal[3] = {0.0, 0.0, 0.0};
    for (int axis = 0; axis < 3; ++axis) {
      const int next = (axis + 1) % 3;
      const int previous = (axis + 2) % 3;
      const double ab_next = static_cast<double>(b[next]) - a[next];
      const double ab_previous = static_cast<double>(b[previous]) - a[previous];
      const double ac_next = static_cast<double>(c[next]) - a[next];
      const double ac_previous = static_cast<double>(c[previous]) - a[previous];
      normal[axis] = ab_next * ac_previous - ab_previous * ac_next;
    }
    const double area_twice_squared =
        normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2];
    has_non_degenerate_triangle |= area_twice_squared > 0.0;
  }
  TORCH_CHECK(has_non_degenerate_triangle,
              "mesh must contain at least one non-degenerate triangle");

  return std::make_shared<BVHState>(vertex_data, vertex_count,
                                    triangle_data, triangle_count);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<BVHState, std::shared_ptr<BVHState>>(m, "cuBVH")
      .def("ray_trace", &BVHState::ray_trace)
      .def("unsigned_distance", &BVHState::unsigned_distance)
      .def("signed_distance", &BVHState::signed_distance);
  m.def("create_cuBVH", &create_cu_bvh);
}
