$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = 'D:\Binich\project\SC2BA-Lab'
$env:PATH = 'C:\Users\Binich\.conda\envs\smacv2;' + $env:PATH
$global:GatePython = 'C:\Users\Binich\.conda\envs\smacv2\python.exe'
$global:GateGuard = Join-Path $PSScriptRoot 'gate2_guard.py'
$global:GateRoot = Join-Path $PSScriptRoot 'mvp_2plus2_steps40000_seed101'
$runner = 'D:\Binich\project\SC2BA-Lab\hrl_v2\run_mvp_2plus2.ps1'
Set-Location -LiteralPath 'D:\Binich\project\SC2BA-Lab'
# A non-stopping debugger action checks the previous arm before the next iteration.
# The original runner remains unchanged, including its done.json resume behavior.
$lines = Get-Content -LiteralPath $runner
$line = 1 + [Array]::IndexOf($lines, '    $sc = $job.Scenario')
if ($line -le 0) { throw 'Cannot locate the serial job boundary; refusing unguarded launch.' }
$Host.Runspace.Debugger.SetDebugMode([System.Management.Automation.DebugModes]::LocalScript)
$boundary = Set-PSBreakpoint -Script $runner -Line $line -Action {
    try {
        & $global:GatePython $global:GateGuard --root $global:GateRoot
        if ($LASTEXITCODE -ne 0) { [Environment]::Exit($LASTEXITCODE) }
    } catch {
        [Environment]::Exit(43)
    }
}
try {
    & $runner -Seed 101 -TrainSteps 40000 -EvalEvery 8000 -RolloutSteps 1024 -Device cuda
} finally {
    Remove-PSBreakpoint -Breakpoint $boundary
}
