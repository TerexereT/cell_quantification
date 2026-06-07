# Crea el entorno virtual, detecta la GPU e instala las dependencias del proyecto
# (incluyendo el build correcto de PyTorch segun el hardware disponible).
#
# Uso (desde la raiz de cell_3d_analysis/):
#   .\setup.ps1

$ErrorActionPreference = "Stop"

# 1. Crear el entorno virtual si no existe
if (-not (Test-Path ".\venv\Scripts\python.exe")) {
    Write-Host "Creando entorno virtual..."
    python -m venv venv
} else {
    Write-Host "Entorno virtual ya existe, lo reutilizo."
}

$python = ".\venv\Scripts\python.exe"
& $python -m pip install --upgrade pip

# 2. Detectar GPU NVIDIA
$gpu = Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match "NVIDIA" } | Select-Object -First 1

if ($gpu) {
    Write-Host "GPU NVIDIA detectada: $($gpu.Name)"
    Write-Host "Instalando PyTorch con soporte CUDA (build cu128, compatible con GPUs Blackwell como RTX 50xx y arquitecturas anteriores)..."
    & $python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
} else {
    Write-Host "No se detecto GPU NVIDIA -> se instalara PyTorch para CPU (vendra como dependencia de cellpose)."
}

# 3. Instalar el resto de dependencias del proyecto
Write-Host "Instalando requirements.txt..."
& $python -m pip install -r requirements.txt

# 4. Verificar deteccion de GPU
Write-Host ""
Write-Host "Verificando PyTorch / CUDA:"
& $python -c "import torch; available = torch.cuda.is_available(); print('CUDA disponible:', available); print('Dispositivo:', torch.cuda.get_device_name(0) if available else 'CPU')"

Write-Host ""
Write-Host "Listo. Activa el entorno en cada sesion nueva con: .\venv\Scripts\Activate.ps1"
