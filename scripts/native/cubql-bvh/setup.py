from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, IS_HIP_EXTENSION
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
BUILD_TARGET = os.environ.get("BUILD_TARGET", "auto")

if BUILD_TARGET == "auto":
    if IS_HIP_EXTENSION:
        IS_HIP = True
    else:
        IS_HIP = False
else:
    if BUILD_TARGET == "cuda":
        IS_HIP = False
    elif BUILD_TARGET == "rocm":
        IS_HIP = True

if not IS_HIP:
    cc_flag = [f"-allow-unsupported-compiler"]
else:
    archs = os.getenv("GPU_ARCHS", "native").split(";")
    cc_flag = [f"--offload-arch={arch}" for arch in archs]

setup(
    name="cumesh",
    packages=[
        'cumesh',
    ],
    ext_modules=[
        CUDAExtension(
            name="cumesh._C",
            sources=[
                # Hashmap functions
                "src/hash/hash.cu",

                # CuMesh
                "src/atlas.cu",
                "src/clean_up.cu",
                "src/cumesh.cu",
                "src/connectivity.cu",
                "src/geometry.cu",
                "src/io.cu",
                "src/simplify.cu",
                "src/shared.cu",

                # Remeshing
                "src/remesh/simple_dual_contour.cu",
                "src/remesh/svox2vert.cu",

                # main: HIP headers require clang rather than MSVC on Windows.
                "src/ext.cu" if IS_HIP and os.name == "nt" else "src/ext.cpp",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++20"],
                "nvcc": ["-O3","-std=c++20"] + cc_flag,
            }
        ),
        CUDAExtension(
            name="cumesh._bvh",
            sources=[
                "src/bvh/bvh_state.hip",
                "src/bvh/queries.hip",
                "src/bvh/bindings.cpp",
            ],
            include_dirs=[
                os.path.abspath(os.environ.get(
                    "CUBQL_ROOT", os.path.join(ROOT, "..", "cuBQL")
                )),
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++20"],
                "nvcc": ["-O3", "-std=c++20"] + cc_flag,
            }
        ),
        CUDAExtension(
            name="cumesh._xatlas",
            sources=[
                'third_party/xatlas/xatlas_mod.cpp',
                'third_party/xatlas/binding.cpp',
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++20"],
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
)