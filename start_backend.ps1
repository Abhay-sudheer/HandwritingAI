$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot
$CompatPython = Join-Path $ProjectRoot '.backend-pure-venv\Scripts\python.exe'
if (Test-Path $CompatPython) {
    $CompatSitePackages = Join-Path $ProjectRoot '.backend-pure-venv\Lib\site-packages'
    $SitePackages = Join-Path $ProjectRoot '.venv\Lib\site-packages'
    $OldPythonPath = $env:PYTHONPATH
    try {
        $env:PYTHONPATH = if ($OldPythonPath) { "$CompatSitePackages$([IO.Path]::PathSeparator)$SitePackages$([IO.Path]::PathSeparator)$OldPythonPath" } else { "$CompatSitePackages$([IO.Path]::PathSeparator)$SitePackages" }
        & $CompatPython -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
    } finally { $env:PYTHONPATH = $OldPythonPath }
} elseif (Test-Path '.venv\Scripts\python.exe') {
    & '.venv\Scripts\python.exe' -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
} else { throw 'No project Python environment was found.' }
