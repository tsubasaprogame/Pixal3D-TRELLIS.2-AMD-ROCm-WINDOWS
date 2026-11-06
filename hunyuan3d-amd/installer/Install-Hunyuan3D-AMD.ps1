param([switch]$DryRun)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$Root = Split-Path -Parent $PSScriptRoot
$Rt = Join-Path $Root 'runtime'
$Dl = Join-Path $PSScriptRoot 'downloads'
$Log = Join-Path $PSScriptRoot 'install.log'
$ComfyTag = 'v0.33.2'
$UvZip = 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip'
$Curl = Join-Path $env:SystemRoot 'System32\curl.exe'
if (!(Test-Path -LiteralPath $Curl)) { $Curl = 'curl.exe' }
$Tar = Join-Path $env:SystemRoot 'System32\tar.exe'
if (!(Test-Path -LiteralPath $Tar)) { $Tar = $null }
. (Join-Path $PSScriptRoot 'amd\AmdCore.ps1')

function Say($t, $c = 'Gray') { Write-Host $t -ForegroundColor $c; Add-Content -LiteralPath $Log -Value $t -Encoding UTF8 }
function Ok($t) { Say ("  [found]    " + $t) 'Green' }
function Title($t) { Write-Host ''; Say ('== ' + $t + ' ==') 'Cyan' }
function Native([scriptblock]$b) { $ErrorActionPreference = 'Continue'; & $b }
function Download($url, $dst) {
  New-Item -ItemType Directory -Force -Path (Split-Path $dst) | Out-Null
  if (Test-Path -LiteralPath $dst) { Say ("    already downloaded: " + (Split-Path $dst -Leaf)); return }
  Say ("    downloading " + $url)
  Native { & $Curl -L --fail --retry 5 --retry-delay 3 -C - -# -o ($dst + '.part') $url }
  if ($LASTEXITCODE -ne 0) { throw "Download failed: $url (code $LASTEXITCODE). Run again to resume." }
  Move-Item -LiteralPath ($dst + '.part') -Destination $dst -Force
}
function Unzip($zip, $dst) {
  New-Item -ItemType Directory -Force -Path $dst | Out-Null
  if ($Tar) { Native { & $Tar -xf $zip -C $dst }; if ($LASTEXITCODE -eq 0) { return } }
  try { Expand-Archive -LiteralPath $zip -DestinationPath $dst -Force } catch { throw "Could not unpack $zip" }
}
function Run($exe, [string[]]$argv, $what) {
  Say ("    " + $what)
  Native { & $exe @argv }
  if ($LASTEXITCODE -ne 0) { throw "$what - failed (code $LASTEXITCODE)" }
}
function Uv() {
  $uv = Join-Path $Rt 'uv\uv.exe'
  if (!(Test-Path -LiteralPath $uv)) { $zip = Join-Path $Dl 'uv.zip'; Download $UvZip $zip; Unzip $zip (Join-Path $Rt 'uv') }
  $env:UV_PYTHON_INSTALL_DIR = Join-Path $Rt 'python'
  $env:UV_CACHE_DIR = Join-Path $Rt 'uv-cache'
  return $uv
}
function Model($url, $name) { Download $url (Join-Path $Root ('data\models\checkpoints\' + $name)) }
function SaveAmdComfyModelPaths($dir) {
  $base = (Join-Path $Root 'data\models').Replace('\', '/')
  $yaml = "hunyuan3d_amd:`n  base_path: '$base'`n  checkpoints: checkpoints`n"
  Set-Content -LiteralPath (Join-Path $dir 'extra_model_paths.yaml') -Value $yaml -Encoding UTF8
}

# AMD functions extracted from owner AMD toolkit installer/Install-AMD.ps1.
function InstallAmdComfyDeps($dir) {
  $uv = Uv
  $py = Join-Path $dir '.venv\Scripts\python.exe'
  if (!(Test-Path -LiteralPath $py)) {
    Push-Location $dir
    try { Run $uv @('venv', '--python', '3.12', '.venv') 'Python 3.12 for AMD ComfyUI' } finally { Pop-Location }
  }
  $hip = Native { & $py -c "import torch; print(torch.version.hip or '')" 2>$null }
  if ($LASTEXITCODE -ne 0 -or !$hip) {
    Run $uv @('pip', 'uninstall', '--python', $py, 'torch', 'torchvision', 'torchaudio') 'removing non-ROCm PyTorch'
    Run $uv (@('pip', 'install', '--python', $py, '--no-cache-dir') + (Get-AmdRocmPackageUrls)) 'AMD ROCm 7.2.1 runtime'
    Run $uv (@('pip', 'install', '--python', $py, '--no-cache-dir') + (Get-AmdTorchPackageUrls)) 'AMD ROCm PyTorch 2.9.1'
  }
  Assert-AmdTorch $py 'AMD Radeon RX 9070 XT'
  Run $uv @('pip', 'install', '--python', $py, '-r', (Join-Path $dir 'requirements.txt')) 'ComfyUI libraries'
  Run $uv @('pip', 'install', '--python', $py, 'rembg', 'onnxruntime', 'numba', 'scikit-image', 'xatlas') 'Hunyuan3D support libraries'
  Assert-AmdTorch $py 'AMD Radeon RX 9070 XT'
}
function InstallAmdComfy() {
  $zip = Join-Path $Dl "ComfyUI-$ComfyTag.zip"
  Download "https://github.com/comfyanonymous/ComfyUI/archive/refs/tags/$ComfyTag.zip" $zip
  $dir = Join-Path $Rt 'ComfyUI'
  if (!(Test-Path -LiteralPath (Join-Path $dir 'main.py'))) {
    Unzip $zip $Rt
    $src = Get-ChildItem -Path $Rt -Directory -Filter 'ComfyUI-*' | Select-Object -First 1
    Rename-Item -LiteralPath $src.FullName -NewName 'ComfyUI'
  }
  InstallAmdComfyDeps $dir
  SaveAmdComfyModelPaths $dir
  return $dir
}

if ($MyInvocation.InvocationName -eq '.') { return }
Clear-Host
Say 'Hunyuan3D - AMD ROCm setup (extracted from owner AMD toolkit)' 'White'
$gpu = Select-AmdGpu @(Get-CimInstance Win32_VideoController)
if (!$gpu) { Say 'Supported discrete AMD GPU not found (RX 9070 XT is the target).' 'Red'; exit 1 }
Ok ('AMD GPU: ' + $gpu.Name)
Say ('Installation folder: ' + $Root)
if ($DryRun) {
  Say "Would install ComfyUI $ComfyTag, ROCm 7.2.1, PyTorch 2.9.1, Hunyuan3D-2 and u2net."
  Say 'Dry run: no downloads or changes.'
  exit 0
}
Title 'ComfyUI with native AMD ROCm'
$comfy = InstallAmdComfy
Title 'Hunyuan3D-2 model'
Model 'https://huggingface.co/tencent/Hunyuan3D-2/resolve/main/hunyuan3d-dit-v2-0/model.fp16.safetensors' 'hunyuan3d-dit-v2.safetensors'
Title 'Background removal model'
Download 'https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx' (Join-Path $Root 'data\models\background\models\u2net\u2net.onnx')
Say 'Hunyuan3D AMD installation finished.' 'Green'
Say ('ComfyUI: ' + $comfy)
Say 'Start with: Start ComfyUI AMD.cmd'
