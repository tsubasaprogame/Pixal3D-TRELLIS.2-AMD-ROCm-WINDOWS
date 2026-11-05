# Third-party notices

The original toolkit scripts, documentation and examples are covered by the repository MIT license. Third-party software remains under its own licenses. This file is a notice summary, not legal advice and not a replacement for the complete texts in `licenses/`.

## Bundled HIP extensions and their sources

| Component | Source and pinned revision | Notice |
|---|---|---|
| CuMesh | https://github.com/visualbruno/CuMesh.git — `d10e54c30ddd03d11472c1431693f985501c7966` | MIT: `CuMesh-MIT.txt` |
| FlexGEMM | https://github.com/visualbruno/FlexGEMM.git — `db388bd17c34abc697792aa718dca780734c2783` | MIT: `FlexGEMM-MIT.txt` |
| TRELLIS.2 / O-Voxel | https://github.com/visualbruno/TRELLIS.2.git — `65d1e13b4a92296036044df0633242bb9e95abf6` | MIT: `TRELLIS2-MIT.txt` |
| cubvh in CuMesh | Nested component of the pinned CuMesh revision | MIT: `cubvh-MIT.txt`; NVIDIA Source Code License: `cubvh-NVIDIA-Source-Code-License.txt` |
| xatlas in CuMesh | Nested component of the pinned CuMesh revision | MIT: `xatlas-MIT.txt` |
| Eigen for CuMesh | https://gitlab.com/libeigen/eigen.git — `3147391d946bb4b6c68edd901f2add6ac1f31f8c` | Primarily MPL-2.0: `Eigen-CuMesh-MPL-2.0.txt`; explanation: `Eigen-CuMesh-README.txt` |
| Eigen for O-Voxel | https://gitlab.com/libeigen/eigen.git — `21e4582d1739107337a03460c81412981130373e` | MPL-2.0: `Eigen-O-Voxel-MPL-2.0.txt` |

The Eigen source archives are available from the pinned upstream repositories and revisions above. Eigen files were not modified by the AMD patches.

## NVIDIA / cubvh usage restriction

The cubvh NVIDIA Source Code License, section 3.3, permits use of the covered work and derivatives only for **non-commercial research or evaluation**. The MIT terms of other components do not remove this separate restriction. Review `licenses/cubvh-NVIDIA-Source-Code-License.txt` before any use or redistribution.

## ROCm and PyTorch

Official ROCm 7.2.1 and PyTorch 2.9.1+rocm7.2.1 packages are downloaded by the installer from pinned `repo.radeon.com` URLs and are not committed to this repository. Their exact sizes and SHA-256 hashes are listed in `installer/amd/native-wheels.json`.

- ROCm Python packaging is produced by AMD's TheRock project: https://github.com/ROCm/TheRock. The upstream packaging license is reproduced in `ROCm-TheRock-MIT.txt`. Individual ROCm libraries retain their own upstream notices inside the official packages.
- PyTorch LICENSE and NOTICE are reproduced as `PyTorch-LICENSE.txt` and `PyTorch-NOTICE.txt`.
- Torchvision and Torchaudio licenses are reproduced as `Torchvision-LICENSE.txt` and `Torchaudio-LICENSE.txt`.
- CPython is downloaded from python.org and remains subject to the Python Software Foundation license included in the official distribution.
