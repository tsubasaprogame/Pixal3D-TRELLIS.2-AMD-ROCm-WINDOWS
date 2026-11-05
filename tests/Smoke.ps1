[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$Root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))

$required = @(
  'installer\amd\Pixal3DNative.ps1',
  'installer\amd\native-runtime.json',
  'installer\amd\native-wheels.json',
  'scripts\native\install_pixal3d_amd.py',
  'scripts\native\verify_pixal3d_amd.py',
  'scripts\native\probe_trellis2_extensions.py'
)
foreach ($relative in $required) {
  if (!(Test-Path -LiteralPath (Join-Path $Root $relative) -PathType Leaf)) {
    throw "Required file is missing: $relative"
  }
}

$runtime = Get-Content -LiteralPath (Join-Path $Root 'installer\amd\native-runtime.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if ($runtime.python.version -ne '3.12.10' -or $runtime.python.signer -ne 'Python Software Foundation') {
  throw 'Portable Python metadata does not match the reviewed runtime.'
}
$manifest = Get-Content -LiteralPath (Join-Path $Root 'installer\amd\native-wheels.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if (@($manifest.official).Count -ne 7) { throw 'Exactly seven official ROCm/PyTorch entries are required.' }
if (@($manifest.extensions).Count -ne 3) { throw 'Exactly three bundled HIP wheel entries are required.' }
foreach ($entry in $manifest.extensions) {
  $path = Join-Path $Root ("installer\amd\wheels\" + $entry.file)
  if (!(Test-Path -LiteralPath $path -PathType Leaf)) { throw "Bundled HIP wheel is missing: $($entry.file)" }
  $item = Get-Item -LiteralPath $path
  if ($item.Length -ne [long]$entry.size) { throw "Bundled HIP wheel size mismatch: $($entry.file)" }
  $hash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
  if ($hash -ne $entry.sha256.ToLowerInvariant()) { throw "Bundled HIP wheel SHA-256 mismatch: $($entry.file)" }
}
foreach ($entry in $manifest.official) {
  if ($entry.url -notmatch '^https://repo\.radeon\.com/') { throw "Official package has a non-ROCm URL: $($entry.file)" }
  if ($entry.file -match '(?i)cuda|cu128') { throw "CUDA package found in AMD manifest: $($entry.file)" }
}

$installer = Get-Content -LiteralPath (Join-Path $Root 'installer\amd\Pixal3DNative.ps1') -Raw -Encoding UTF8
if ($installer -notmatch [regex]::Escape("installer\amd\native-runtime.json")) { throw 'Installer does not use the standalone runtime manifest.' }
if ($installer -notmatch [regex]::Escape('--no-index') -or $installer -notmatch [regex]::Escape('--no-deps')) {
  throw 'Installer must use offline, no-dependency wheel installation.'
}
$forbidden = @('native-ready.json', 'Start-Process', 'shutdown', 'wsl.exe --install', 'pnputil', 'devcon')
foreach ($pattern in $forbidden) {
  if ($installer -match [regex]::Escape($pattern)) { throw "Unsafe or production-only installer operation: $pattern" }
}
Write-Host 'Smoke checks passed.'
