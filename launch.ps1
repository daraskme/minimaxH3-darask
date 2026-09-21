$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project environment is missing. Run .\bootstrap.ps1 first."
}
$env:PYTHONUTF8 = "1"
Set-Location -LiteralPath $ProjectRoot
$Port = [int](& $Python -c "from h3studio.config import settings; print(settings.port)")
if ($LASTEXITCODE -ne 0) { throw "Could not read config.json." }
$Url = "http://127.0.0.1:$Port"

function Get-H3Status {
    try {
        $Status = Invoke-RestMethod -Uri "$Url/api/status" -Method Get -TimeoutSec 2
        if ($Status.engine.engine -eq "standalone-diffusers") { return $Status }
    } catch { }
    return $null
}

$ExistingStatus = Get-H3Status
if ($null -ne $ExistingStatus) {
    Start-Process $Url
    Write-Host "H3 Studio is already running at $Url"
    exit 0
}

# A responding port that did not return H3 Studio's status belongs to another
# process. Fail here instead of opening an unrelated local service.
$Client = [System.Net.Sockets.TcpClient]::new()
try {
    $ConnectTask = $Client.ConnectAsync("127.0.0.1", $Port)
    [void]$ConnectTask.Wait(300)
    $PortOccupied = $Client.Connected
} catch {
    $PortOccupied = $false
} finally {
    $Client.Dispose()
}
if ($PortOccupied) {
    throw "Port $Port is occupied by another service. Change port in config.json or stop that service."
}

$Server = Start-Process -FilePath $Python -ArgumentList @(
    "-m", "uvicorn", "h3studio.main:app", "--host", "127.0.0.1", "--port", "$Port"
) -WindowStyle Hidden -PassThru
try {
    $Ready = $false
    $StartupTimer = [System.Diagnostics.Stopwatch]::StartNew()
    while ($StartupTimer.Elapsed.TotalSeconds -lt 30) {
        if ($Server.HasExited) {
            throw "H3 Studio exited during startup (exit code $($Server.ExitCode))."
        }
        if ($null -ne (Get-H3Status)) {
            $Ready = $true
            break
        }
        Start-Sleep -Milliseconds 500
    }
    if (-not $Ready) { throw "H3 Studio did not become healthy within 30 seconds." }
    Start-Process $Url
    Write-Host "H3 Studio is running at $Url" -ForegroundColor Green
    Wait-Process -Id $Server.Id
    if ($Server.ExitCode -ne 0) { throw "H3 Studio stopped with exit code $($Server.ExitCode)." }
} finally {
    if (-not $Server.HasExited) { Stop-Process -Id $Server.Id -Force }
}
