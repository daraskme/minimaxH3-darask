param(
    [string]$PlayerSource = "",
    [string]$DlssSdk = ""
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $PlayerSource) { $PlayerSource = Join-Path $root ".cache\research\dlss5-video-player" }
if (-not $DlssSdk) { $DlssSdk = Join-Path $root ".cache\research\NVIDIA-DLSS-310.9.1" }
$playerCommit = "ffb01f5d015ee1a0544a6a46452cfe4b334553a3"
$dlssCommit = "374959484e79a640feaba44c93ac8cfb0a03f5b5"
$runtimeSha = "3975567B8943C53ACCE397F2B72380092F84F162D00B0D2C7D08A1025C563983"

function Invoke-Native([string]$Program, [string[]]$Arguments) {
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program failed with exit code $LASTEXITCODE" }
}

function Ensure-Checkout([string]$Path, [string]$Url, [string]$Commit) {
    if (-not (Test-Path (Join-Path $Path ".git"))) {
        New-Item -ItemType Directory -Force -Path (Split-Path $Path) | Out-Null
        Invoke-Native "git.exe" @("clone", "--filter=blob:none", "--no-checkout", $Url, $Path)
    }
    $head = (& git.exe -C $Path rev-parse HEAD 2>$null).Trim()
    if ($LASTEXITCODE -ne 0 -or $head -ne $Commit) {
        Invoke-Native "git.exe" @("-C", $Path, "fetch", "--depth=1", "origin", $Commit)
        Invoke-Native "git.exe" @("-C", $Path, "checkout", "--detach", $Commit)
    }
    $verified = (& git.exe -C $Path rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $verified -ne $Commit) {
        throw "Checkout revision mismatch: $Path"
    }
}

Ensure-Checkout $PlayerSource "https://github.com/2600th/dlss5-video-player.git" $playerCommit
Ensure-Checkout $DlssSdk "https://github.com/NVIDIA/DLSS.git" $dlssCommit

$runtime = Join-Path $DlssSdk "lib\Windows_x86_64\rel\nvngx_dlss.dll"
if (-not (Test-Path $runtime)) { throw "Official DLSS SR runtime is missing: $runtime" }
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $runtime).Hash -ne $runtimeSha) {
    throw "Official DLSS SR runtime hash mismatch."
}
$signature = Get-AuthenticodeSignature -LiteralPath $runtime
if ($signature.Status -ne "Valid" -or $signature.SignerCertificate.Subject -notlike "*NVIDIA Corporation*") {
    throw "Official DLSS SR runtime signature is not valid NVIDIA Authenticode."
}

$cmake = (Get-Command cmake.exe -ErrorAction Stop).Source
$vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) { throw "Visual Studio Build Tools discovery is unavailable." }
$vsInstall = (& $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath).Trim()
if ($LASTEXITCODE -ne 0 -or -not $vsInstall) { throw "MSVC x64 Build Tools are unavailable." }
$devShell = Join-Path $vsInstall "Common7\Tools\Launch-VsDevShell.ps1"
& $devShell -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
if (-not $env:VSCMD_VER) { throw "MSVC developer environment could not be initialized." }
$ninja = Join-Path $vsInstall "Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
if (-not (Test-Path $ninja)) { throw "Visual Studio Ninja is unavailable." }
$source = Join-Path $root "tools\dlss_exporter"
$build = Join-Path $root ".cache\build\dlss-exporter-ninja"
$destination = Join-Path $source "bin"
New-Item -ItemType Directory -Force -Path $build,$destination | Out-Null

# The upstream raw encoder always forces Matroska, even for an .mp4 path. That
# millisecond Matroska time base turns 24 fps into 24.024 fps after stream-copy
# remux. Generate one audited build-only source with exactly this container
# selection changed; all process, cancellation and encoding code stays pinned.
$upstreamMedia = Join-Path $PlayerSource "src\MediaPipeline.cpp"
$patchedMedia = Join-Path $build "MediaPipeline.H3Studio.cpp"
$mediaText = [IO.File]::ReadAllText($upstreamMedia)
$needle = @'
    arguments.insert(arguments.end(), {L"-f", L"matroska", output.wstring()});
'@
$replacement = @'
    const wchar_t* container = output.extension() == L".mp4" ? L"mp4" : L"matroska";
    if (output.extension() == L".mp4") arguments.insert(arguments.end(), {L"-movflags", L"+faststart"});
    arguments.insert(arguments.end(), {L"-f", container, output.wstring()});
'@
if ([regex]::Matches($mediaText, [regex]::Escape($needle)).Count -ne 1) { throw "Pinned MediaPipeline MP4 patch anchor mismatch." }
[IO.File]::WriteAllText($patchedMedia, $mediaText.Replace($needle, $replacement), [Text.UTF8Encoding]::new($false))

Invoke-Native $cmake @(
    "-S", $source, "-B", $build, "-G", "Ninja", "-DCMAKE_MAKE_PROGRAM=$ninja", "-DCMAKE_BUILD_TYPE=Release",
    "-DDLSS_PLAYER_SOURCE=$PlayerSource", "-DDLSS_SDK=$DlssSdk", "-DH3_MEDIA_PIPELINE_SOURCE=$patchedMedia"
)
Invoke-Native $cmake @("--build", $build, "--target", "H3DLSSExporter", "--parallel")

$worker = Join-Path $build "H3DLSSExporter.exe"
if (-not (Test-Path $worker)) { throw "Exporter build produced no executable." }
Copy-Item -LiteralPath $worker -Destination (Join-Path $destination "H3DLSSExporter.exe") -Force
Copy-Item -LiteralPath $runtime -Destination (Join-Path $destination "nvngx_dlss.dll") -Force
$stagedWorker = Join-Path $destination "H3DLSSExporter.exe"
$manifest = [ordered]@{
    schema_version = 1
    contract = "h3studio-dlss-sr-v1"
    player_commit = $playerCommit
    dlss_commit = $dlssCommit
    worker_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $stagedWorker).Hash.ToLowerInvariant()
    runtime_sha256 = $runtimeSha.ToLowerInvariant()
    runtime_signature = "Valid NVIDIA Corporation"
    media_pipeline_patch = "direct-mp4-container-v1"
    media_pipeline_source_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $patchedMedia).Hash.ToLowerInvariant()
    smoke_validated = $false
    smoke_validation = $null
}
$manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $destination "build-manifest.json") -Encoding utf8NoBOM
Write-Host "Built and staged the official DLSS SR exporter. GPU smoke validation is still required."
