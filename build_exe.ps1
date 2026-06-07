# Build the unsigned Windows desktop distribution for cell_3d_analysis.
#
# Output: dist/cell3d_gui/
# This build is intentionally unsigned. Windows SmartScreen can show an
# "unknown publisher" warning; choose "More info" -> "Run anyway" if you trust it.
# The distribution can be several GB because it bundles torch, cellpose and
# native scientific dependencies.

$ErrorActionPreference = "Stop"

if (-not (Test-Path ".\venv\Scripts\python.exe")) {
    throw "No se encontro .\venv\Scripts\python.exe. Ejecuta .\setup.ps1 primero."
}

$python = ".\venv\Scripts\python.exe"
& $python -m pip install -r requirements-build.txt
& $python -m PyInstaller build\cell3d_gui.spec --noconfirm

Write-Host ""
Write-Host "Build listo en dist\cell3d_gui\cell3d_gui.exe"
Write-Host "Nota: ejecutable sin firma digital; SmartScreen puede mostrar 'editor desconocido'."
