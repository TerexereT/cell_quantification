# cell_3d_analysis — Análisis 3D de células a partir de z-stacks de microscopía

Pipeline automatizado en Python que segmenta células en 3D con **Cellpose**
(`do_3D=True`), reconstruye su superficie con *marching cubes* y exporta
mediciones volumétricas por célula en CSV, junto con máscaras, proyecciones y
figuras de control de calidad.

---

## 1. Qué tipo de imágenes acepta

- Formato nativo: **TIFF** (`.tif` o `.tiff`).
- Formato **`.czi`** (Zeiss): no se lee directamente; primero se convierte a TIFF
  con `tools/czi_to_tiff.py` (ver sección 9). El conversor extrae la calibración
  física automáticamente del encabezado del `.czi`.
- Contenido: un **z-stack** (varios cortes en Z), no una sola imagen 2D.
- Layouts soportados (configurables en `config.yaml`):
  - Grayscale `Z × Y × X` (caso por defecto, `channel_axis: null`).
  - Multicanal `Z × C × Y × X` → `channel_axis: 1`.
  - Multicanal `Z × Y × X × C` → `channel_axis: 3`.

> ⚠️ **Importante:** una sola imagen 2D **no** permite reconstrucción 3D real.
> El volumen, el área de superficie y la anisotropía solo tienen sentido con un
> z-stack que tenga varios cortes en Z. Si el stack tiene un único corte, el
> pipeline lo procesa pero emite una advertencia y las métricas 3D serán
> degeneradas.

---

## 2. Cómo organizar las carpetas

```
cell_3d_analysis/
├── input/
│   ├── raw_zstacks/        # ← coloca aquí tus .tif/.tiff
│   │   └── ejemplo_zstack.tif
│   └── metadata/
│       └── metadata.csv     # ← una fila por imagen
├── config/
│   └── config.yaml          # ← parámetros del pipeline
├── output/                  # ← se llena automáticamente
│   ├── ejemplo_zstack/
│   │   ├── 1/              # ← salidas de Fase 1 (segmentación)
│   │   │   ├── masks_3d/
│   │   │   ├── projections/
│   │   │   ├── meshes/
│   │   │   ├── measurements/
│   │   │   └── figures_qc/
│   │   └── 2/              # ← salidas de Fase 2 (clasificación Dbc1)
│   │       ├── masks_3d/
│   │       ├── measurements/
│   │       └── figures_qc/
│   └── logs/               # log general de la corrida
└── src/                     # código del pipeline
```

---

## 3. Cómo llenar `metadata.csv`

Una fila por imagen. Columnas mínimas obligatorias:

| Columna | Tipo | Significado |
|---|---|---|
| `filename` | texto | Nombre del archivo dentro de `input/raw_zstacks/` |
| `px_xy_um` | float | Tamaño de pixel lateral en micras (µm/píxel en X y Y) |
| `px_z_um` | float | Distancia entre cortes Z en micras (µm/corte) |
| `channel_to_segment` | entero | Índice de canal a segmentar (0 si es grayscale) |
| `notes` | texto | Comentario libre (opcional) |

Ejemplo:

```csv
filename,px_xy_um,px_z_um,channel_to_segment,notes
ejemplo_zstack.tif,0.108,0.300,0,"control sample"
otra_muestra.tif,0.108,0.300,1,"canal verde"
```

Los valores `px_xy_um` y `px_z_um` salen de los metadatos de tu microscopio
(tamaño de píxel y paso del eje Z). **Son imprescindibles**: sin ellos las
medidas no tienen unidades físicas.

---

## 4. Cómo modificar `config.yaml`

```yaml
input_dir: "input/raw_zstacks"
metadata_file: "input/metadata/metadata.csv"
output_dir: "output"

cellpose:
  gpu: false              # true si tienes GPU con CUDA
  model_type: "cyto3"     # modelo Cellpose (en Cellpose-SAM v4 se ignora)
  diameter: null          # null = autodetección; o un número en píxeles
  do_3D: true             # NO cambiar: segmentación volumétrica
  z_axis: 0               # eje Z dentro del array
  channel_axis: null      # null = grayscale; 1 = Z×C×Y×X; 3 = Z×Y×X×C
  flow_threshold: 0.4     # ↑ más permisivo con la forma del flujo
  cellprob_threshold: 0.0 # ↓ detecta más células (incluye más dudosas)
  min_size_voxels: 100    # descarta objetos con menos voxeles que esto

measurements:
  save_individual_cell_meshes: true   # exporta un .obj por célula
  calculate_surface_area: true
  calculate_projected_area_xy: true

qc:
  save_overlay_projection: true
  save_mask_projection: true
```

Parámetros que normalmente querrás tocar:
- **`min_size_voxels`**: súbelo si aparecen muchos objetos espurios pequeños.
- **`cellprob_threshold`**: bájalo (p.ej. `-1.0`) para detectar más células;
  súbelo para ser más estricto.
- **`diameter`**: fíjalo en píxeles si conoces el tamaño típico de tus células
  (mejora velocidad y consistencia).

---

## 5. Cómo ejecutar el pipeline

### Preparar el entorno virtual e instalar dependencias (recomendado)

Aísla las dependencias del proyecto en un entorno virtual. Desde la raíz del
proyecto (`cell_3d_analysis/`):

**Windows (PowerShell):**

```powershell
# 1. Crear el entorno virtual (solo la primera vez)
python -m venv .venv

# 2. Activarlo (cada vez que abras una terminal nueva)
.\.venv\Scripts\Activate.ps1

# 3. Instalar dependencias dentro del entorno
pip install -r requirements.txt
pip install czifile        # solo si vas a convertir archivos .czi (sección 9)
```

> Si PowerShell bloquea la activación con un error de *execution policy*, ejecútalo
> una vez en esa terminal:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

**Linux / macOS (bash):**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install czifile        # solo si vas a convertir archivos .czi
```

Con el entorno **activado** (verás `(.venv)` al inicio del prompt), ejecuta el
pipeline (paso siguiente). Para salir del entorno: `deactivate`.

> **Nota sobre GPU**: `requirements.txt` instala la versión CPU de PyTorch. Para
> usar GPU (config `gpu: true`), instala el build de torch con CUDA que
> corresponda a tu tarjeta **dentro del entorno activado** antes que el resto,
> p.ej. desde https://pytorch.org/get-started/locally/.

### Local (con el entorno virtual activado)

Desde la raíz del proyecto (`cell_3d_analysis/`):

```bash
python src/main.py --config config/config.yaml
```

### Con conda

```bash
conda env create -f environment.yml
conda activate cell_3d_analysis
python src/main.py --config config/config.yaml
```

### Con Docker

```bash
docker build -t cell_3d_analysis .
# Monta input/, output/ y config/ para usar tus datos y conservar resultados:
docker run --rm \
  -v "$PWD/input:/app/input" \
  -v "$PWD/output:/app/output" \
  -v "$PWD/config:/app/config" \
  cell_3d_analysis
```

---

## 6. Qué significa cada archivo de salida

**Fase 1** (segmentación — `python src/main.py`):

| Carpeta | Archivo | Contenido |
|---|---|---|
| `output/<nombre>/1/masks_3d/` | `<nombre>_masks_3d.tif` | Máscara 3D etiquetada (0 = fondo, 1..N = células). |
| `output/<nombre>/1/measurements/` | `<nombre>_measurements_3d.csv` | Una fila por célula con todas las métricas 3D. |
| `output/<nombre>/1/projections/` | `<nombre>_max_projection.tif` | Proyección de intensidad máxima de la imagen original. |
| `output/<nombre>/1/projections/` | `<nombre>_mask_projection.tif` | Proyección máxima de la máscara. |
| `output/<nombre>/1/meshes/` | `<nombre>_cell_<id>.obj` | Malla 3D de la superficie de cada célula (si está habilitado). |
| `output/<nombre>/1/figures_qc/` | `<nombre>_qc_overlay.png` | Figura 3 paneles (original / máscara / overlay) para revisar la segmentación. |
| `output/logs/` | `pipeline_log.txt` | Registro completo de la ejecución. |

**Fase 2** (clasificación Dbc1 — `python tools/phase2_intensity.py`):

| Carpeta | Archivo | Contenido |
|---|---|---|
| `output/<nombre>/2/masks_3d/` | `<nombre>_masks_dbc1_positive.tif` | Máscara original con labels Dbc1− puestos a 0. |
| `output/<nombre>/2/measurements/` | `<nombre>_dbc1_intensity.csv` | Intensidad por célula + clasificación Dbc1+/Dbc1−. |
| `output/<nombre>/2/figures_qc/` | `<nombre>_dbc1_classification.png` | 3 paneles: canal rojo / máscara clasificada (verde=+, rojo=−) / overlay. |
| `output/<nombre>/2/figures_qc/` | `<nombre>_qc_red_overlay.png` | Overlay canal rojo con todas las máscaras. |
| `output/<nombre>/2/figures_qc/` | `<nombre>_qc_blue_overlay.png` | Overlay canal azul (DAPI) con todas las máscaras. |

### Columnas del CSV — Fase 1 (`_measurements_3d.csv`)

`filename, cell_id, voxel_count, volume_um3, surface_area_um2,
projected_area_xy_um2, z_slices_detected, bbox_z_min, bbox_z_max, bbox_y_min,
bbox_y_max, bbox_x_min, bbox_x_max`

Los `bbox_*` son índices (en voxeles) **inclusivos** del bounding box 3D.

### Columnas del CSV — Fase 2 (`_dbc1_intensity.csv`)

`cell_id, area_px, mean_intensity_red, mean_intensity_corr, IntDen,
IntDen_corregida, clasificacion, bkg_pp, PromIntDen_BKG, umbral,
n_positivas, n_negativas`

- **`mean_intensity_corr`** — intensidad media por pixel corregida por fondo (`mean_intensity_red − bkg_pp`). Es la métrica usada para la clasificación.
- **`IntDen` / `IntDen_corregida`** — densidad integrada (área × intensidad media), reportada para referencia con el protocolo Fiji.
- **`clasificacion`** — `Dbc1+` o `Dbc1−`. La última fila (`__metadata__`) contiene los parámetros del umbral aplicado.

---

## 7. Cómo interpretar las métricas

- **`volume_um3`** — Volumen de la célula en micras cúbicas.
  Se calcula como `voxel_count × px_xy_um² × px_z_um`. Es la suma del volumen
  físico de todos los voxeles que pertenecen a la célula.

- **`surface_area_um2`** — Área de la superficie 3D en micras cuadradas.
  Se obtiene reconstruyendo la superficie con *marching cubes* (con el
  espaciado físico real `(px_z, px_xy, px_xy)`) y sumando el área de los
  triángulos de la malla. Refleja lo "rugosa" o ramificada que es la célula:
  para un mismo volumen, mayor área = forma más irregular.
  Puede ser `NaN` si la célula es demasiado pequeña para reconstruir una malla.

- **`projected_area_xy_um2`** — Área de la "sombra" de la célula sobre el plano
  XY. Es el número de píxeles que ocupa la célula al colapsar el eje Z,
  multiplicado por `px_xy_um²`. Útil para comparar con mediciones clásicas 2D.

- **`z_slices_detected`** — En cuántos cortes Z aparece la célula. Da una idea
  de su extensión en profundidad.

---

## 8. Dependencias

Ver `requirements.txt` / `environment.yml`:
`cellpose, numpy, pandas, tifffile, scikit-image, matplotlib, pyyaml`.

Para convertir archivos `.czi` (sección 9) se necesita además **`czifile`**
(`pip install czifile`), que no es parte del pipeline principal.

---

## 9. Guía rápida: ejecutar con tus propias imágenes

### A) Si tus imágenes ya son TIFF (`.tif` / `.tiff`)

1. **Copia** tus archivos a `input/raw_zstacks/`:

   ```powershell
   Copy-Item "C:\ruta\a\tus\imagenes\*.tif" "input\raw_zstacks\"
   ```

2. **Edita** `input/metadata/metadata.csv` — una fila por imagen. Las columnas
   `px_xy_um` (µm/píxel lateral) y `px_z_um` (µm entre cortes Z) son
   **obligatorias** y salen de los metadatos de tu microscopio:

   ```csv
   filename,px_xy_um,px_z_um,channel_to_segment,notes
   muestra1.tif,0.099,2.0,0,mi experimento
   muestra2.tif,0.099,2.0,0,mi experimento
   ```

   El pipeline procesa **solo** las imágenes listadas aquí. Si una imagen está en
   la carpeta pero no en el CSV, se ignora; si está en el CSV pero falta el
   archivo, se registra un error y se continúa con las demás.

3. **Ejecuta** desde la raíz del proyecto:

   ```powershell
   python src/main.py --config config/config.yaml
   ```

4. **Revisa** los resultados en `output/` (ver sección 6). Mira primero el PNG de
   `output/<nombre>/1/figures_qc/` para confirmar que la segmentación es razonable antes de
   confiar en las métricas.

### B) Si tienes un archivo `.czi` (Zeiss)

El pipeline no lee `.czi` directamente; conviértelo a TIFF primero con el
conversor incluido, que **además extrae la calibración** (`px_xy_um`, `px_z_um`)
del encabezado del `.czi`:

```powershell
pip install czifile   # solo la primera vez

# Convierte el canal 0 y agrega la fila a metadata.csv automáticamente:
python tools/czi_to_tiff.py "C:\ruta\a\imagen.czi" --channel 0 --append-metadata
```

El conversor:
- Extrae el canal indicado (`--channel`, default 0) y reduce el volumen a `(Z, Y, X)`.
- Guarda el TIFF en `input/raw_zstacks/<nombre>.tif`.
- Imprime `px_xy_um` y `px_z_um` leídos del `.czi`.
- Con `--append-metadata`, agrega/actualiza la fila correspondiente en
  `metadata.csv` con esa calibración (si la omites, copia los valores impresos
  a mano en el CSV).

Luego ejecuta el pipeline normal (paso 3 de la sección A). Para varios canales,
repite la conversión con el `--channel` que quieras segmentar.

> **Nota sobre tus imágenes `cre+342_17`**: el `.czi` de ejemplo es
> `C × Z × Y × X = 2 × 6 × 1024 × 1024` (2 canales, 6 cortes Z), con calibración
> `px_xy ≈ 0.099 µm` y `px_z = 2.0 µm`. Tras convertir, queda un z-stack
> grayscale `(6, 1024, 1024)` que el pipeline procesa con la config por defecto
> (`channel_axis: null`).

---

## 10. Ajuste de parámetros para núcleos (DAPI) y reprocesamiento manual

Esta sección traduce el flujo manual de Fiji + Cellpose GUI (segmentación de
**núcleos DAPI**, medición de intensidad en el canal de señal) a los parámetros
de `config.yaml`, para que puedas evaluar la forma de la segmentación y reprocesar
tú mismo ajustando valores, sin tocar código.

### 10.1 Qué canal segmentar

El `.czi` `cre+342_17` tiene 2 canales:

| Canal | Fluoróforo | Uso |
|---|---|---|
| 0 | AF647 (Alexa Fluor 647) | Señal a **medir** (intensidad), NO se segmenta |
| 1 | DAPI | **Núcleos** → este es el que se segmenta |

Por eso la conversión usa `--channel 1`:

```powershell
./venv/Scripts/python.exe tools/czi_to_tiff.py "C:\ruta\a\imagen.czi" --channel 1 --append-metadata
```

### 10.2 Valores recomendados (de las notas de Fiji/Cellpose)

Estos son los valores ya puestos en `config.yaml`, equivalentes a los de la GUI:

| Parámetro GUI (notas) | Valor | `config.yaml` | Efecto |
|---|---|---|---|
| flow threshold | 0.2 | `cellpose.flow_threshold` | Más bajo separa mejor núcleos pegados |
| cellprob threshold | -3 | `cellpose.cellprob_threshold` | Más bajo capta núcleos más débiles (más detecciones) |
| Chan to segment: blue (DAPI) | canal 1 | (se elige al convertir el `.czi`, ver 10.1) | Segmenta el canal de núcleos |
| Diámetro (medido en Fiji) | autodetección | `cellpose.diameter` | Ver 10.4 para fijarlo manualmente |
| Modelo: cyto2 | `cyto2` | `cellpose.model_type` | ⚠️ Cellpose-SAM v4 **ignora** este campo (ver 10.5) |

> **Calibración**: `px_xy_um = 0.099`, `px_z_um = 2.0` se extraen solos del `.czi`
> y quedan en `metadata.csv`. No los edites a mano salvo que cambies de microscopio.

### 10.3 Cómo evaluar la forma y qué tocar

Tras cada corrida, abre `output/<nombre>/1/figures_qc/<nombre>_qc_overlay.png` (3 paneles:
original / máscara / overlay) y compara contra lo que esperas de tus núcleos:

| Lo que ves en el overlay | Qué ajustar en `config.yaml` | Hacia dónde |
|---|---|---|
| Muchas manchitas espurias / ruido segmentado | `min_size_voxels` | Súbelo: 100 → 500 → 1000 |
| Dos o más núcleos pegados como uno solo | `flow_threshold` | Bájalo: 0.2 → 0.1 |
| Un núcleo partido en varios pedazos | `flow_threshold` | Súbelo: 0.2 → 0.4 |
| Faltan núcleos tenues (no los detecta) | `cellprob_threshold` | Bájalo: -3 → -5 |
| Detecta de más (fondo tomado como núcleo) | `cellprob_threshold` | Súbelo: -3 → -1 → 0 |
| Núcleos sub/sobre-dimensionados | `diameter` | Fíjalo en px (ver 10.4) |

Cambia **un parámetro a la vez**, reprocesa y vuelve a mirar el overlay. Así
aíslas el efecto de cada ajuste.

### 10.4 Fijar el diámetro nuclear (opcional, mejora la separación)

Como en tus notas (medir ~6 núcleos con la línea en Fiji y promediar):

1. En Fiji, traza la línea sobre el diámetro de ~6 núcleos y `Analyze → Measure`;
   promedia los `Length`.
2. Conviértelo a **píxeles**:
   - Si mediste con la imagen **sin calibrar** (px), ese promedio ya está en px.
   - Si mediste en µm (con `pixel width = 0.099`), entonces `px = µm / 0.099`.
3. Pon ese número en `config.yaml`:
   ```yaml
   cellpose:
     diameter: 95        # ejemplo: ~9.4 µm / 0.099 ≈ 95 px
   ```
   Déjalo en `null` para autodetección.

### 10.5 Nota sobre el modelo (cyto2 vs Cellpose-SAM)

Tus notas usan **cyto2 + denoise** (Cellpose v3). El entorno tiene **Cellpose
4.1.1 (Cellpose-SAM)**, que **ignora `model_type`** y no usa el denoise de v3 — por
eso en el log aparece `model_type argument is not used in v4.0.1+`. Los valores de
`flow_threshold`, `cellprob_threshold` y `diameter` **sí** se aplican.

Si quieres reproducir tus notas al pie de la letra (cyto2 + denoise), instala
Cellpose v3 en el entorno:

```powershell
./venv/Scripts/python.exe -m pip install "cellpose<4"
```

(con v3, `model_type: "cyto2"` y el campo `diameter` se respetan; el denoise no está
cableado en este pipeline y seguiría siendo un paso manual de la GUI).

### 10.6 Reprocesar tras ajustar

Edita `config/config.yaml`, guarda y vuelve a correr (sobrescribe `output/`):

```powershell
./venv/Scripts/python.exe src/main.py --config config/config.yaml
```

> **GPU automática**: con `gpu: auto` el pipeline usa la GPU si hay CUDA disponible
> y cae a CPU si no, sin que cambies nada (ver sección 5, nota de GPU).

---

## 11. Fase 2 — Clasificación Dbc1+/Dbc1− por intensidad

Tras ejecutar el pipeline de Fase 1 (que genera las máscaras 3D), la Fase 2
mide la intensidad del marcador Dbc1 (canal rojo, Alexa Fluor 647) en cada
célula detectada y las clasifica como positivas o negativas.

### 11.1 Prerequisitos

- Haber corrido la Fase 1 al menos una vez (carpetas con `masks_3d/` en `output/`).
- Tener el archivo `.czi` original (ambos canales: rojo=0, azul=1).
- Dependencia `czifile` instalada:
  ```powershell
  ./venv/Scripts/pip.exe install czifile
  ```

### 11.2 Ejecutar la Fase 2

Desde la raíz del proyecto (`cell_3d_analysis/`):

```powershell
./venv/Scripts/python.exe tools/phase2_intensity.py "..\ruta\a\imagen.czi" output/
```

Con factor personalizado (ver 11.4):

```powershell
./venv/Scripts/python.exe tools/phase2_intensity.py "..\ruta\a\imagen.czi" output/ --factor 1.0
```

El script descubre automáticamente todas las subcarpetas de `output/` que
tengan `masks_3d/` y las procesa en secuencia.

### 11.3 Archivos generados por carpeta

| Archivo | Contenido |
|---|---|
| `figures_qc/<stem>_qc_red_overlay.png` | Overlay canal rojo con todas las máscaras |
| `figures_qc/<stem>_qc_blue_overlay.png` | Overlay canal azul (DAPI) con todas las máscaras |
| `figures_qc/<stem>_dbc1_classification.png` | 3 paneles: canal rojo / máscara clasificada (verde=Dbc1+, rojo=Dbc1−) / overlay |
| `measurements/<stem>_dbc1_intensity.csv` | Una fila por célula con métricas de intensidad y clasificación |
| `masks_3d/<stem>_masks_dbc1_positive.tif` | Máscara igual a la original pero con labels Dbc1− puestos a 0 |

### 11.4 Ajustar el umbral de clasificación

La clasificación usa la **intensidad media corregida por background** (`mean_intensity_corr`),
que es independiente del tamaño celular:

```
bkg_pp         = intensidad media de pixels fuera de todas las máscaras
mean_int_corr  = mean_intensity_red − bkg_pp  (por célula)
umbral         = media(todas) − factor × SD(todas)
célula Dbc1−   si mean_int_corr < umbral
```

El parámetro `--factor` controla la sensibilidad:

| `--factor` | Efecto | Cuándo usarlo |
|---|---|---|
| `0.5` | Detecta ~30% de células como negativas (umbral alto) | Si ves muchas células tenues que se pierden |
| `1.0` | Umbral moderado | Punto de partida recomendado |
| `1.6` | Default (umbral más bajo) | Si la mayoría son positivas y pocas son negativas |
| `2.0` | Umbral muy bajo, pocas negativas | Si sólo quieres capturar las más débiles |

**Flujo de ajuste recomendado:**

1. Corre con `--factor 1.0` y abre `*_dbc1_classification.png`.
2. Si aún faltan células negativas visibles, baja: `--factor 0.5`.
3. Si clasifica como negativas células que visualmente tienen señal, sube: `--factor 1.5`.
4. Compara el overlay 3 de la figura contra el canal rojo (panel 1) para validar.

El CSV siempre incluye `mean_intensity_corr` y `IntDen_corregida` para que
puedas inspecionar los valores y decidir el factor apropiado.

---

## Notas de compatibilidad de Cellpose

El módulo de segmentación detecta automáticamente la firma de la API e incluye
solo los argumentos soportados:
- **Cellpose v3** usa `model_type` (p.ej. `cyto3`) y `channels`.
- **Cellpose v4 (Cellpose-SAM)** eliminó `channels` y deprecó `model_type`;
  en ese caso el parámetro se ignora sin romper la ejecución.
"# cell_quantification" 
