param(
    [ValidateSet("smoke", "v10a-collect-fit", "v10a-run", "v10b-fixed-vs-pool", "analyze", "all")]
    [string]$Stage = "smoke",
    [string]$Scenarios = "all13",
    [int]$Workers = 4,
    [int]$TrainSteps = 8000,
    [int]$EvalEvery = 2000,
    [string]$Seeds = "101,202,303",
    [string]$OutRoot = "D:\Binich\project\SC2BA-Lab\bch\runs\v10_all13"
)

$ErrorActionPreference = "Stop"
$RepoRoot = "D:\Binich\project\SC2BA-Lab"
$EnvRoot = "C:\Users\Binich\.conda\envs\smacv2"
$Python = Join-Path $EnvRoot "python.exe"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = $RepoRoot
$env:PATH = "$EnvRoot;$EnvRoot\Library\bin;$EnvRoot\Scripts;$env:PATH"
Set-Location $RepoRoot

function Run-Stage([string]$Current) {
    switch ($Current) {
        "smoke" {
            & $Python -m bch.v10_suite smoke `
                --scenarios $Scenarios --workers $Workers --out-root $OutRoot
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        }
        "v10a-collect-fit" {
            & $Python -m bch.v10_suite collect `
                --scenarios $Scenarios --repeats 2 --workers $Workers --out-root $OutRoot
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
            & $Python -m bch.v10_suite fit `
                --scenarios $Scenarios --methods "im-medoid,unit-set,im-grid" `
                --epochs 160 --codes 32 --device cuda --workers $Workers --out-root $OutRoot
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        }
        "v10a-run" {
            & $Python -m bch.v10_suite run `
                --scenarios $Scenarios `
                --methods "im-hash,im-medoid,unit-set,im-grid" `
                --regimes "pool-all9" `
                --seeds $Seeds `
                --train-steps $TrainSteps --eval-every $EvalEvery --eval-init `
                --val-repeats 2 --test-repeats 5 `
                --workers $Workers --out-root $OutRoot
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        }
        "v10b-fixed-vs-pool" {
            $TargetScenarios = if ($Scenarios -eq "all13") { "core4" } else { $Scenarios }
            & $Python -m bch.v10_suite run `
                --scenarios $TargetScenarios `
                --methods "im-hash,im-medoid,unit-set,im-grid" `
                --regimes "fixed-bot,fixed-qmix,fixed-dop,fixed-vdn,fixed-coma,pool-algos8,pool-all9" `
                --seeds $Seeds `
                --train-steps $TrainSteps --eval-every $EvalEvery --eval-init `
                --val-repeats 2 --test-repeats 5 `
                --workers $Workers --out-root $OutRoot
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        }
        "analyze" {
            & $Python -m bch.v10_suite analyze --out-root $OutRoot
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        }
    }
}

if ($Stage -eq "all") {
    foreach ($s in @("smoke", "v10a-collect-fit", "v10a-run", "v10b-fixed-vs-pool", "analyze")) {
        Write-Host "`n==================== [v10] STAGE: $s ====================" -ForegroundColor Cyan
        Run-Stage $s
    }
} else {
    Run-Stage $Stage
}
