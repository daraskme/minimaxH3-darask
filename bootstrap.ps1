$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $VenvPython)) {
    $BasePythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    $BasePythonPath = if ($BasePythonCommand) { $BasePythonCommand.Source } else { $null }
    if (-not $BasePythonPath) {
        $Candidate = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"
        if (Test-Path -LiteralPath $Candidate) { $BasePythonPath = $Candidate }
    }
    if (-not $BasePythonPath) { throw "Python 3.12 was not found." }
    & $BasePythonPath -m venv (Join-Path $ProjectRoot ".venv")
    if ($LASTEXITCODE -ne 0) { throw "Could not create .venv." }
}

& $VenvPython -m ensurepip --upgrade
if ($LASTEXITCODE -ne 0) { throw "Could not bootstrap pip in .venv." }
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Could not update pip." }
& $VenvPython -c "import torch,torchvision; assert torch.__version__ == '2.14.0+cu130'; assert torchvision.__version__ == '0.29.0+cu130'" 2>$null
if ($LASTEXITCODE -ne 0) {
    & $VenvPython -m pip install "torch==2.14.0+cu130" "torchvision==0.29.0+cu130" --index-url "https://download.pytorch.org/whl/cu130"
    if ($LASTEXITCODE -ne 0) { throw "Could not install the validated CUDA 13.0 PyTorch and TorchVision builds." }
}

& $VenvPython -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Could not install H3 Studio requirements." }
& $VenvPython -m pip install -r (Join-Path $ProjectRoot "requirements-seedvr2.txt")
if ($LASTEXITCODE -ne 0) { throw "Could not install the pinned SeedVR2 requirements." }
& $VenvPython -m pip install "triton-windows==3.8.0.post28"
if ($LASTEXITCODE -ne 0) { throw "Could not install Triton for Windows." }
& $VenvPython -m pip install "https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post6/sageattention-2.2.0%2Bcu130torch2.10.0andhigher.post6-cp310-abi3-win_amd64.whl"
if ($LASTEXITCODE -ne 0) { throw "Could not install SageAttention." }
& $VenvPython (Join-Path $ProjectRoot "scripts\download_postprocess_models.py")
if ($LASTEXITCODE -ne 0) { throw "Could not install verified postprocess models." }
& $VenvPython -X utf8 (Join-Path $ProjectRoot "scripts\download_seedvr2_models.py")
if ($LASTEXITCODE -ne 0) { throw "Could not install the verified SeedVR2 model bundle." }
& $VenvPython -X utf8 (Join-Path $ProjectRoot "scripts\seedvr2_selfcheck.py")
if ($LASTEXITCODE -ne 0) { throw "SeedVR2 real-model self-check failed." }
& $VenvPython -c "import torch,diffusers,transformers,torchao,sageattention; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw "GPU runtime verification failed." }
Write-Host "H3 Studio setup complete." -ForegroundColor Green
