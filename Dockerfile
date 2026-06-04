# ============================================================
# Imagen reproducible para el pipeline de análisis 3D de células
# ============================================================
FROM python:3.10-slim

# Dependencias de sistema requeridas por Cellpose / OpenCV / matplotlib.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instala primero las dependencias (mejor cacheo de capas).
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia el código fuente y la configuración.
COPY . .

# Crea la estructura de carpetas dentro del contenedor.
RUN mkdir -p input/raw_zstacks input/metadata config \
             output/masks_3d output/projections output/meshes \
             output/measurements output/figures_qc output/logs

# Comando por defecto: ejecuta el pipeline completo.
CMD ["python", "src/main.py", "--config", "config/config.yaml"]
