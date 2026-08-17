# Verify honest install path: dist=mlspec, import=mltk
# Usage: pwsh -File scripts/verify_install.ps1
$ErrorActionPreference = "Stop"
Write-Host "==> pip show mlspec (expect installed after: pip install mlspec)"
pip show mlspec 2>&1 | Select-Object -First 8

Write-Host "==> import mltk"
python -c "import mltk; print('mltk', mltk.__version__)"

Write-Host "==> mltk doctor (if CLI installed)"
try {
  mltk doctor
} catch {
  Write-Host "CLI not on PATH — try: pip install 'mlspec[cli]'"
}

Write-Host "OK: install story uses mlspec; import remains mltk"
