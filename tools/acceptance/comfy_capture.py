import importlib.machinery
import os
import runpy
import sys
from pathlib import Path

import torch


class CaptureComplete(Exception):
    pass


def comfy_argv(argv, main_path):
    return [str(main_path), *argv[2:]]


def configure_comfy_args(argv, main_path, options):
    options.enable_args_parsing()
    return comfy_argv(argv, main_path)


def install_capture_hook(capture):
    original = importlib.machinery.SourceFileLoader.exec_module

    def exec_module(loader, module):
        original(loader, module)
        if module.__name__.endswith("trellis2.pipelines.trellis2_image_to_3d"):
            module.Trellis2ImageTo3DPipeline.postprocess_mesh = capture

    importlib.machinery.SourceFileLoader.exec_module = exec_module

    def restore():
        importlib.machinery.SourceFileLoader.exec_module = original

    return restore


def main():
    repo = Path(os.environ["CUBQL_REPO"]).resolve()
    runtime = Path(os.environ["CUBQL_STABLE_RUNTIME"]).resolve()
    experimental = Path(os.environ["CUBQL_EXPERIMENTAL_RUNTIME"]).resolve()
    output = Path(os.environ["CUBQL_CAPTURE"]).resolve()

    sys.path.insert(0, str(experimental))
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(runtime / "src" / "ComfyUI-Trellis2"))
    sys.path.insert(0, str(runtime / "ComfyUI"))

    import comfy.options
    main_path = runtime / "ComfyUI" / "main.py"
    sys.argv = configure_comfy_args(sys.argv, main_path, comfy.options)

    import cumesh
    from tools.acceptance.cubql_acceptance import capture_inputs

    def capture(self, mesh, pbr_voxel, resolution=1024, texture_size=2048, *args, **kwargs):
        payload = capture_inputs(mesh, pbr_voxel, resolution, texture_size)
        payload.update({
            "capture_complete": True,
            "seed": 42,
            "source_image": os.environ["AMD_UV_INPUT"],
            "source_glb": os.environ["AMD_UV_GLB"],
            "mesh_orientation": "90 degrees",
            "texture_steps": 12,
            "cumesh_module": cumesh.__file__,
            "cumesh_backend": cumesh.bvh._backend.__name__,
        })
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, output)
        print(f"[CUBQL-CAPTURE] saved {output}", flush=True)
        raise CaptureComplete(str(output))

    install_capture_hook(capture)
    runpy.run_path(str(main_path), run_name="__main__")


if __name__ == "__main__":
    main()
