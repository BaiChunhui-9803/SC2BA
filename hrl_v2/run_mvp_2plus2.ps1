<#
.SYNOPSIS
    Isolated single-worker 2+2 MVP runner for DyIM-CHRL (hrl_v2).
    Runs strictly outside bch/ and bch/runs/ so active v11 jobs are never disturbed.
    Requires interactive Windows desktop (WinSta0\Default) for SC2_x64.exe.
#>
param(
    [int]$Seed = 101,
    [int]$TrainSteps = 40000,
    [int]$EvalEvery = 8000,
    [int]$RolloutSteps = 1024,
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "cuda",
    [ValidateSet("all", "3m", "2s3z", "MM4m", "MM8")]
    [string]$OnlyScenario = "all",
    [string]$OutRoot = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Py = "C:\Users\Binich\.conda\envs\smacv2\python.exe"

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = $RepoRoot
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"

if (-not $OutRoot) {
    $OutRoot = Join-Path $PSScriptRoot ("runs\mvp_2plus2_steps{0}_seed{1}" -f $TrainSteps, $Seed)
}

$Jobs = @(
    @{ Scenario = "3m";   Regime = "pool-all9" },
    @{ Scenario = "3m";   Regime = "fixed-bot" },
    @{ Scenario = "2s3z"; Regime = "pool-all9" },
    @{ Scenario = "2s3z"; Regime = "fixed-bot" },
    @{ Scenario = "MM4m"; Regime = "fixed-bot" },
    @{ Scenario = "MM8";  Regime = "fixed-bot" }
)

if ($OnlyScenario -ne "all") {
    $Jobs = $Jobs | Where-Object { $_.Scenario -eq $OnlyScenario }
}

Write-Host "=== DyIM-CHRL (hrl_v2) Isolated 2+2 MVP Runner ===" -ForegroundColor Cyan
Write-Host "Output root: $OutRoot (100% isolated from bch/runs/)"
Write-Host "Jobs queued: $($Jobs.Count) | Seed: $Seed | Steps: $TrainSteps | Device: $Device"

foreach ($job in $Jobs) {
    $sc = $job.Scenario
    $rg = $job.Regime
    $jobDir = Join-Path $OutRoot "$sc\$rg\seed_$Seed"
    $doneFile = Join-Path $jobDir "done.json"
    if (Test-Path $doneFile) {
        Write-Host "[SKIP completed] $sc / $rg" -ForegroundColor Green
        continue
    }
    Write-Host "`n>>> Launching $sc / $rg -> $jobDir" -ForegroundColor Yellow
    & $Py -m hrl_v2.run_experiment `
        --scenario $sc `
        --regime $rg `
        --seed $Seed `
        --train-steps $TrainSteps `
        --eval-every $EvalEvery `
        --rollout-steps $RolloutSteps `
        --device $Device `
        --out-dir $jobDir
    if ($LASTEXITCODE -ne 0) {
        throw "Job failed: $sc / $rg (exit code $LASTEXITCODE)"
    }
}

Write-Host "`n=== All selected DyIM-CHRL MVP jobs completed cleanly ===" -ForegroundColor Green
