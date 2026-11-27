param(
    [ValidateRange(1, 8)]
    [int]$Workers = 6,
    [string]$OutRoot = 'D:\Binich\project\SC2BA-Lab\bch\runs\v11_40k_10seed',
    [string]$Sc2Version = '5.0.16.97425',
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
$RepoRoot = 'D:\Binich\project\SC2BA-Lab'
$EnvRoot = 'C:\Users\Binich\.conda\envs\smacv2'
$Python = Join-Path $EnvRoot 'python.exe'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONPATH = $RepoRoot
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:CUDA_VISIBLE_DEVICES = ''
$env:PATH = "$EnvRoot;$EnvRoot\Library\bin;$EnvRoot\Scripts;$env:PATH"
Set-Location -LiteralPath $RepoRoot

$commandLine = (Get-CimInstance Win32_Process -Filter "ProcessId = $PID").CommandLine
if ($commandLine -notmatch '(?i)(?:^|\s)-NoProfile(?:\s|$)') {
    throw 'Launch this script using powershell -NoProfile -ExecutionPolicy Bypass -File.'
}
if (!(Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Missing interpreter $Python" }

Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
using System.Text;
public class Sc2baV11Desktop {
    [DllImport("user32.dll")] public static extern IntPtr GetProcessWindowStation();
    [DllImport("user32.dll")] public static extern IntPtr GetThreadDesktop(uint threadId);
    [DllImport("kernel32.dll")] public static extern uint GetCurrentThreadId();
    [DllImport("user32.dll", CharSet=CharSet.Unicode)]
    public static extern bool GetUserObjectInformation(IntPtr handle, int index, StringBuilder buffer, int length, out int needed);
    [DllImport("user32.dll", SetLastError=true)] public static extern IntPtr OpenInputDesktop(uint flags, bool inherit, uint access);
    [DllImport("user32.dll")] public static extern bool CloseDesktop(IntPtr desktop);
    public static string Name(IntPtr handle) {
        var buffer = new StringBuilder(256);
        int needed;
        return GetUserObjectInformation(handle, 2, buffer, 512, out needed) ? buffer.ToString() : "unavailable";
    }
}
'@

$station = [Sc2baV11Desktop]::Name([Sc2baV11Desktop]::GetProcessWindowStation())
$desktop = [Sc2baV11Desktop]::Name([Sc2baV11Desktop]::GetThreadDesktop([Sc2baV11Desktop]::GetCurrentThreadId()))
$inputHandle = [Sc2baV11Desktop]::OpenInputDesktop(0, $false, 1)
$inputDesktop = 'unavailable'
if ($inputHandle -ne [IntPtr]::Zero) {
    try { $inputDesktop = [Sc2baV11Desktop]::Name($inputHandle) }
    finally { [void][Sc2baV11Desktop]::CloseDesktop($inputHandle) }
}
if ($station -ne 'WinSta0' -or $desktop -ne 'Default' -or $inputDesktop -ne 'Default') {
    throw "Active interactive desktop required. Current $station\$desktop, input $inputDesktop."
}

$engines = @(Get-Process -Name SC2_x64 -ErrorAction SilentlyContinue)
$manifest = Join-Path $OutRoot 'v11_queue_manifest.json'
$preflight = [ordered]@{
    host_id = 'local-win'
    station = $station
    desktop = $desktop
    input_desktop = $inputDesktop
    sc2_processes = $engines.Count
    manifest_present = Test-Path -LiteralPath $manifest -PathType Leaf
    python = $Python
    workers = $Workers
}
if ($CheckOnly) {
    $preflight | ConvertTo-Json
    exit 0
}
if ($engines.Count) { throw "SC2 engines already active. PIDs $($engines.Id -join ', ')." }
if (!$preflight.manifest_present) { throw "Initialize the verified v11 manifest before launching $manifest" }

$LogDir = Join-Path $OutRoot 'driver_logs'
$RunnerDir = Join-Path $OutRoot 'runners'
New-Item -ItemType Directory -Path $LogDir, $RunnerDir -Force | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$stdoutPath = Join-Path $LogDir "local-win-$stamp.stdout.log"
$stderrPath = Join-Path $LogDir "local-win-$stamp.stderr.log"
$launchPath = Join-Path $RunnerDir 'local-win-launch.json'
$launch = [ordered]@{
    launcher_pid = $PID
    host_id = 'local-win'
    direction = 'forward'
    state = 'starting'
    workers = $Workers
    python = $Python
    stdout = $stdoutPath
    stderr = $stderrPath
    desktop = "$station\$desktop"
    updated_at = [DateTime]::UtcNow.ToString('o')
}

function Save-LaunchStatus {
    $launch.updated_at = [DateTime]::UtcNow.ToString('o')
    $temporary = "$launchPath.$PID.tmp"
    [IO.File]::WriteAllText($temporary, ($launch | ConvertTo-Json), [Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $temporary -Destination $launchPath -Force
}

try {
    Save-LaunchStatus
    $arguments = @(
        '-u', '-m', 'bch.v11_distributed', 'run',
        '--direction', 'forward', '--workers', $Workers.ToString(), '--host-id', 'local-win',
        '--sc2-version', $Sc2Version,
        '--out-root', ('"{0}"' -f $OutRoot),
        '--encoders-root', ('"{0}"' -f (Join-Path $RepoRoot 'bch\runs\v10_all13\encoders'))
    )
    $worker = Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $RepoRoot `
        -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
    $launch.runner_pid = $worker.Id
    $launch.state = 'running'
    Save-LaunchStatus
    $worker.WaitForExit()
    $worker.Refresh()
    $exitCode = $worker.ExitCode
    $launch.exit_code = $exitCode
    $launch.state = if ($exitCode -eq 0) { 'completed' } else { 'blocked' }
    Save-LaunchStatus
    exit $exitCode
} catch {
    $launch.state = 'blocked'
    $launch.error = $_.Exception.Message
    Save-LaunchStatus
    throw
}
