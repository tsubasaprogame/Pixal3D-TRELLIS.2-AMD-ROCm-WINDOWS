$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '..\installer\Install-Hunyuan3D-AMD.ps1')
function Assert($condition, $message) { if (!$condition) { throw $message } }
$cards = @(
  [pscustomobject]@{ Name = 'AMD Radeon(TM) Graphics' },
  [pscustomobject]@{ Name = 'AMD Radeon RX 9070 XT' }
)
Assert ((Select-AmdGpu $cards).Name -eq 'AMD Radeon RX 9070 XT') 'discrete RX 9070 XT must beat the iGPU'
Assert ((Select-AmdGpu @($cards[0])) -eq $null) 'unsupported iGPU must not be selected'
Assert ((Get-AmdRocmPackageUrls).Count -eq 4) 'four ROCm runtime packages are required'
Assert ((Get-AmdTorchPackageUrls).Count -eq 3) 'three PyTorch packages are required'
Assert (((Get-AmdTorchPackageUrls) -notmatch 'rocm-rel-7.2.1').Count -eq 0) 'all wheels must be pinned to ROCm 7.2.1'
Assert (((Get-AmdTorchPackageUrls) -match 'cu128|download.pytorch.org').Count -eq 0) 'AMD profile must not contain CUDA wheels'
$installer = Get-Content (Join-Path $PSScriptRoot '..\installer\Install-Hunyuan3D-AMD.ps1') -Raw
Assert ($installer.Contains('InstallAmdComfyDeps')) 'AMD ComfyUI installer must exist'
Assert ($installer.Contains('Assert-AmdTorch')) 'ROCm must be validated'
Assert ($installer.Contains('hunyuan3d-dit-v2.safetensors')) 'Hunyuan3D-2 model is required'
Assert ($installer.Contains('u2net.onnx')) 'background-removal model is required'
Assert ($installer.Contains('extra_model_paths.yaml')) 'ComfyUI model path configuration is required'
Assert (!$installer.Contains('InstallAmdPixal3d')) 'Pixal3D must not be installed by this focused package'
Assert (!$installer.Contains('wsl.exe')) 'WSL must not be used'
Write-Host 'Hunyuan3D AMD static unit tests passed'
