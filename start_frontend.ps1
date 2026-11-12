$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Join-Path $ProjectRoot 'frontend')
if (-not (Test-Path 'node_modules')) { npm install }
npm run dev
