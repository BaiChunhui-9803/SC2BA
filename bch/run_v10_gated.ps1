param(
    [ValidateSet(2,3,4)]
    [int]$StartStage = 2,
    [ValidateRange(4,6)]
    [int]$Workers = 6
)

$ErrorActionPreference = 'Stop'
$Repo = 'D:\Binich\project\SC2BA-Lab'
$Root = Join-Path $Repo 'bch\runs\v10_all13'
$Launcher = Join-Path $Repo 'bch\run_v10.ps1'
$Conda = 'C:\Users\Binich\.conda\envs\smacv2'
$Python = Join-Path $Conda 'python.exe'
$env:PATH = "$Conda;$Conda\Library\bin;$Conda\Scripts;$env:PATH"
$env:PYTHONPATH = $Repo
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
Set-Location $Repo

$LogDir = Join-Path $Root 'driver_logs'
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
$Token = Get-Date -Format 'yyyyMMdd-HHmmss'
$StageNumber = $StartStage

function Write-Status([string]$State, [int]$Number, [string]$Detail = '') {
    $path = Join-Path $Root 'driver_status.json'
    $record = [ordered]@{state=$State; stage=$Number; workers=$Workers; message=$Detail; pid=$PID; updated_at=(Get-Date -Format o)}
    $record | ConvertTo-Json | Set-Content -LiteralPath $path -Encoding utf8
}

function Assert-NoEngines([string]$Context) {
    $sc2 = @(Get-Process -Name 'SC2','SC2_x64' -ErrorAction SilentlyContinue)
    if ($sc2.Count) { throw "SC2 engines active at $Context (PIDs $($sc2.Id -join ', '))" }
}

function Invoke-Run([string]$Name, [string]$Group) {
    $file = Join-Path $LogDir "$Name-$Token.log"
    powershell -NoProfile -ExecutionPolicy Bypass -File $Launcher -Stage $Name -Scenarios $Group -OutRoot $Root -Workers $Workers -TrainSteps 8000 -EvalEvery 2000 -Seeds '101,202,303' 2>&1 | Tee-Object -FilePath $file
    if ($LASTEXITCODE -ne 0) { throw "$Name failed. See $file." }
}

function Invoke-Gate([int]$Number) {
    & $Python -m bch.verify_v10_gates $Number --out-root $Root
    if ($LASTEXITCODE -ne 0) { throw "Gate $Number failed." }
}

function Confirm-WebRows([string]$RelativePath, [int]$Rows) {
    $esc = [Uri]::EscapeDataString($RelativePath)
    $url = "http://127.0.0.1:4321/studio/experiment-hub-asset?project=sc2ba-lab&root=sc2baLab&file=$esc"
    $payload = Invoke-WebRequest -Uri $url -TimeoutSec 15
    $actual = @($payload.Content | ConvertFrom-Csv).Count
    if ($actual -ne $Rows) { throw "White Night City returned $actual rows for $RelativePath; expected $Rows." }
    "White Night City verified $actual rows in $RelativePath."
}

try {
    if ((git remote get-url --push --all upstream) -ne 'DISABLE') { throw 'Disabled upstream push configuration is required.' }
    Assert-NoEngines 'driver start'
    Invoke-Gate 0
    Invoke-Gate 1
    if ($StartStage -le 2) {
        $StageNumber = 2
        Write-Status 'running' $StageNumber 'v10-A1'
        Invoke-Run 'v10a-run' 'all13'
        Assert-NoEngines 'Stage 2 completion'
        Invoke-Run 'analyze' 'all13'
        Invoke-Gate 2
        Confirm-WebRows 'bch/runs/v10_all13/analysis/summary_by_condition.csv' 52
    } elseif (!(Test-Path -LiteralPath (Join-Path $Root 'gates\stage2.json'))) {
        throw 'Gate 2 must be recorded before resuming at Stage 3 or 4.'
    }
    if ($StartStage -le 3) {
        $StageNumber = 3
        Write-Status 'running' $StageNumber 'v10-B1'
        Invoke-Run 'v10b-fixed-vs-pool' 'core4'
        Assert-NoEngines 'Stage 3 completion'
        Invoke-Run 'analyze' 'core4'
        Invoke-Gate 3
        Confirm-WebRows 'bch/runs/v10_all13/analysis/summary_by_condition.csv' 148
    } else {
        Invoke-Gate 3
    }
    $StageNumber = 4
    Write-Status 'running' $StageNumber 'analysis and manuscript figures'
    & $Python -m bch.plot_v10_results --out-root $Root --report (Join-Path $Repo 'refine-logs\V10_EXPERIMENT_RESULTS.md')
    if ($LASTEXITCODE -ne 0) { throw 'Stage 4 report or figure generation failed.' }
    Invoke-Run 'analyze' 'all13'
    Confirm-WebRows 'bch/runs/v10_all13/analysis/summary_by_condition.csv' 148
    Assert-NoEngines 'v10 completion'
    Write-Status 'complete' 4 'gates passed, analysis synchronized, report and figures written'
} catch {
    Write-Status 'failed' $StageNumber $_.Exception.Message
    throw
}

