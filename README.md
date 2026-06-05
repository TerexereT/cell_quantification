# cell_3d_analysis — Análisis 3D de células a partir de z-stacks de microscopía

Pipeline en Python que segmenta células en 3D con **Cellpose** (`do_3D=True`), reconstruye su superficie con *marching cubes* y exporta mediciones volumétricas (Fase 1), y clasifica cada célula por intensidad de señal Dbc1 (Fase 2).

---

## 1. Instalación

### Requisitos previos

- Python 3.9 o superior instalado y en el PATH.
- El proyecto clonado o descargado en tu máquina.

### Crear y activar el entorno virtual

Ejecuta los siguientes comandos desde la carpeta raíz del proyecto (`cell_3d_analysis/`):

**Windows (PowerShell)**

```powershell
# Solo la primera vez
python -m venv venv

# Cada vez que abras una terminal nueva
.\venv\Scripts\Activate.ps1
```

> Si PowerShell bloquea la activación, ejecuta primero:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

**macOS / Linux (bash)**

```bash
python3 -m venv venv
source venv/bin/activate
```

Cuando el entorno está activo verás `(venv)` al inicio del prompt.

### Instalar dependencias

Con el entorno activado:

```bash
pip install -r requirements.txt
```

Para trabajar con archivos `.czi` de Zeiss (Fase 1B y Fase 2):

```bash
pip install czifile
```

> **GPU (opcional):** `requirements.txt` instala la versión CPU de PyTorch.
> Para usar GPU, instala primero el build de torch con CUDA para tu tarjeta
> desde https://pytorch.org/get-started/locally/ y luego instala el resto
> de dependencias.

---

## 2. Preparar los datos de entrada

### Estructura de carpetas

```
cell_3d_analysis/
├── input/
│   ├── raw_zstacks/        # coloca aquí tus .tif/.tiff
│   └── metadata/
│       └── metadata.csv    # una fila por imagen
├── config/
│   └── config.yaml         # parámetros del pipeline
└── output/                 # se genera automáticamente
```

### metadata.csv

Una fila por imagen. Columnas mínimas obligatorias:

| Columna              | Tipo   | Significado                                |
| -------------------- | ------ | ------------------------------------------ |
| `filename`           | texto  | Nombre del archivo en `input/raw_zstacks/` |
| `px_xy_um`           | float  | Tamaño de pixel lateral en µm              |
| `px_z_um`            | float  | Distancia entre cortes Z en µm             |
| `channel_to_segment` | entero | Canal a segmentar (0 si es grayscale)      |
| `notes`              | texto  | Comentario libre (opcional)                |

```csv
filename,px_xy_um,px_z_um,channel_to_segment,notes
muestra1.tif,0.099,2.0,1,nucleo DAPI
muestra2.tif,0.108,0.300,0,control
```

### Convertir archivos .czi a TIFF (si aplica)

El pipeline no lee `.czi` directamente. Usa el conversor incluido, que además
extrae la calibración del encabezado:

```powershell
# Convierte el canal 1 (DAPI) y agrega la fila a metadata.csv automáticamente
.\venv\Scripts\python.exe tools/czi_to_tiff.py "C:\ruta\a\imagen.czi" --channel 1 --append-metadata
```

El conversor:
- Extrae el canal indicado y guarda el TIFF en `input/raw_zstacks/`.
- Lee `px_xy_um` y `px_z_um` del `.czi` e imprime sus valores.
- Con `--append-metadata`, los escribe directamente en `metadata.csv`.

> **Nota sobre tus imágenes `cre+342_17`:** el `.czi` es `C × Z × Y × X = 2 × 6 × 1024 × 1024`.
> Canal 0 = AF647 (señal a medir), canal 1 = DAPI (núcleos a segmentar).
> Calibración extraída automáticamente: `px_xy = 0.099 µm`, `px_z = 2.0 µm`.

---

## 3. Fase 1 — Segmentación 3D

### Ejecutar

Desde la raíz del proyecto, con el entorno activado:

```powershell
.\venv\Scripts\python.exe src/main.py --config config/config.yaml
```

El pipeline procesa todas las imágenes listadas en `metadata.csv`.

### Parametrizar

Edita `config/config.yaml` para ajustar el comportamiento:

```yaml
input_dir: "input/raw_zstacks"
metadata_file: "input/metadata/metadata.csv"
output_dir: "output"

cellpose:
  gpu: auto                # true si tienes GPU con CUDA; "auto" detecta automáticamente
  model_type: "cyto3"       # ignorado en Cellpose v4 (Cellpose-SAM)
  diameter: null            # null = autodetección; o un número en píxeles
  do_3D: true               # NO cambiar
  z_axis: 0
  channel_axis: null        # null = grayscale; 1 = Z×C×Y×X; 3 = Z×Y×X×C
  flow_threshold: 0.2       # más bajo = mejor separación de núcleos pegados
  cellprob_threshold: -3.0  # más bajo = detecta núcleos más tenues
  min_size_voxels: 100      # descarta objetos con menos voxeles que este valor

measurements:
  save_individual_cell_meshes: true
  calculate_surface_area: true
  calculate_projected_area_xy: true

qc:
  save_overlay_projection: true
  save_mask_projection: true
```

Guía de ajuste rápido (revisa `output/<nombre>/1/figures_qc/*_qc_overlay.png` después de cada corrida):

| Lo que ves en el overlay       | Parámetro            | Acción                  |
| ------------------------------ | -------------------- | ----------------------- |
| Muchas manchitas espurias      | `min_size_voxels`    | Subir: 100 → 500 → 1000 |
| Núcleos pegados como uno       | `flow_threshold`     | Bajar: 0.2 → 0.1        |
| Un núcleo partido en varios    | `flow_threshold`     | Subir: 0.2 → 0.4        |
| Faltan núcleos tenues          | `cellprob_threshold` | Bajar: -3 → -5          |
| Fondo detectado como célula    | `cellprob_threshold` | Subir: -3 → -1 → 0      |
| Núcleos sub/sobredimensionados | `diameter`           | Fijar en px (ver abajo) |

**Cómo fijar el diámetro nuclear:** mide ~6 núcleos en Fiji con la herramienta
de línea (`Analyze → Measure`), promedia los `Length` y conviértelo a píxeles
(`px = µm / px_xy_um`). Ejemplo: `9.4 µm / 0.099 ≈ 95 px`.

```yaml
cellpose:
  diameter: 95
```

### Salidas de Fase 1

Las salidas quedan en `output/<nombre_imagen>/1/`:

| Carpeta         | Archivo                        | Contenido                                                    |
| --------------- | ------------------------------ | ------------------------------------------------------------ |
| `masks_3d/`     | `<nombre>_masks_3d.tif`        | Máscara 3D etiquetada (0 = fondo, 1..N = células)            |
| `measurements/` | `<nombre>_measurements_3d.csv` | Una fila por célula con métricas morfológicas 3D             |
| `projections/`  | `<nombre>_max_projection.tif`  | Proyección de intensidad máxima                              |
| `projections/`  | `<nombre>_mask_projection.tif` | Proyección máxima de la máscara                              |
| `meshes/`       | `<nombre>_cell_<id>.obj`       | Malla 3D por célula (si `save_individual_cell_meshes: true`) |
| `figures_qc/`   | `<nombre>_qc_overlay.png`      | 3 paneles: original / máscara / overlay                      |

**Columnas del CSV (`_measurements_3d.csv`):**

`filename, cell_id, voxel_count, volume_um3, surface_area_um2, projected_area_xy_um2, z_slices_detected, bbox_z_min, bbox_z_max, bbox_y_min, bbox_y_max, bbox_x_min, bbox_x_max`

Los `bbox_*` son índices en voxeles (inclusivos) del bounding box 3D.

---

## 4. Fase 2 — Clasificación Dbc1+/Dbc1−

### Prerequisitos

- Haber ejecutado la Fase 1 al menos una vez (carpetas `output/*/1/masks_3d/` deben existir).
- Tener el archivo `.czi` original con ambos canales (rojo = AF647, azul = DAPI).
- `czifile` instalado (`pip install czifile`).

### Ejecutar

```powershell
.\venv\Scripts\python.exe tools/phase2_intensity.py "C:\ruta\a\imagen.czi" output/
```

El script descubre automáticamente todas las subcarpetas de `output/` que contengan
`masks_3d/` y las procesa en secuencia.

### Parametrizar el umbral de clasificación

Hay tres modos de umbral, en orden de prioridad:

| Modo               | Argumento       | Comportamiento                                       |
| ------------------ | --------------- | ---------------------------------------------------- |
| **Otsu** (default) | _(ninguno)_     | Valle natural del histograma de `mean_intensity_red` |
| **Factor SD**      | `--factor k`    | `mean - k × SD` de `mean_intensity_red`              |
| **Valor fijo**     | `--threshold T` | Valor fijo en unidades raw del detector (0–65535)    |

Ejemplo con factor personalizado:

```powershell
.\venv\Scripts\python.exe tools/phase2_intensity.py "C:\ruta\a\imagen.czi" output/ --factor 1.0
```

Guía de ajuste del `--factor`:

| `--factor` | Efecto                                                      |
| ---------- | ----------------------------------------------------------- |
| `0.5`      | Umbral alto — más células clasificadas como negativas       |
| `1.0`      | Umbral moderado — punto de partida recomendado              |
| `1.6`      | Default — umbral más bajo, pocas negativas                  |
| `2.0`      | Umbral muy bajo — solo las más tenues quedan como negativas |

**Flujo recomendado:** corre con `--factor 1.0`, abre `*_dbc1_classification.png`
y compara el panel de clasificación (verde = Dbc1+, rojo = Dbc1−) contra el
canal rojo. Ajusta el factor y repite hasta que coincida con lo esperado.

### Resumen en consola

Al terminar, el script imprime:

```
Resumen:
carpeta                          | N_células | N_Dbc1+ | N_Dbc1- | umbral
output\cre+342_17...\2           | 347       | 310     | 37      | 8547.231
```

El `umbral` está en unidades raw de intensidad del detector (escala 0–65535
para imágenes 16-bit). Para verificarlo en Fiji: mide la intensidad media
en el canal rojo de una célula que visualmente parezca negativa — debe estar
por debajo de ese valor.

### Salidas de Fase 2

Las salidas quedan en `output/<nombre_imagen>/2/`:

| Carpeta         | Archivo                            | Contenido                                        |
| --------------- | ---------------------------------- | ------------------------------------------------ |
| `measurements/` | `<nombre>_dbc1_intensity.csv`      | Intensidad por célula + clasificación            |
| `masks_3d/`     | `<nombre>_masks_dbc1_positive.tif` | Máscara original con labels Dbc1− puestos a 0    |
| `figures_qc/`   | `<nombre>_dbc1_classification.png` | 3 paneles: canal rojo / clasificación / overlay  |
| `figures_qc/`   | `<nombre>_qc_red_overlay.png`      | Overlay canal rojo con todas las máscaras        |
| `figures_qc/`   | `<nombre>_qc_blue_overlay.png`     | Overlay canal azul (DAPI) con todas las máscaras |

**Columnas del CSV (`_dbc1_intensity.csv`):**

*Métricas 2D (proyección máxima en Z):*
`cell_id, area_px, mean_intensity_red, mean_intensity_corr, IntDen, IntDen_corregida`

*Métricas 3D (volumen completo):*
`voxel_count_3d, IntDen_3D, IntDen_3D_corr, mean_intensity_3D`

*Clasificación (última fila `__metadata__`):*
`clasificacion, bkg_pp, bkg_pp_3d, PromIntDen_BKG, umbral, metodo_umbral, n_positivas, n_negativas`

---

## 5. Referencia de fórmulas

### Fase 1 — Morfología 3D

Sea `V` el conjunto de voxeles de la célula en la máscara `(Z, Y, X)`:

| Columna                 | Fórmula                                                                          |
| ----------------------- | -------------------------------------------------------------------------------- |
| `voxel_count`           | `\|V\|`                                                                          |
| `volume_um3`            | `\|V\| × px_xy² × px_z`                                                          |
| `projected_area_xy_um2` | Píxeles únicos en proyección XY × `px_xy²`                                       |
| `z_slices_detected`     | Cortes Z con al menos un voxel de la célula                                      |
| `surface_area_um2`      | Área de la malla triangular (*marching cubes*, `spacing = (px_z, px_xy, px_xy)`) |

### Fase 2 — Intensidad

| Columna               | Fórmula                                                       |
| --------------------- | ------------------------------------------------------------- |
| `mean_intensity_red`  | Media de `red_proj` sobre los píxeles de la célula            |
| `bkg_pp`              | Media de `red_proj` sobre píxeles fuera de todas las máscaras |
| `mean_intensity_corr` | `mean_intensity_red − bkg_pp`                                 |
| `IntDen`              | `area_px × mean_intensity_red`                                |
| `IntDen_corregida`    | `IntDen − PromIntDen_BKG`                                     |
| `IntDen_3D`           | Suma de `red_volume` sobre todos los voxeles de la célula     |
| `IntDen_3D_corr`      | `IntDen_3D − bkg_pp_3d × voxel_count_3d`                      |

---

## 6. Notas

**Cellpose v4 (Cellpose-SAM):** el campo `model_type` se ignora. Los parámetros
`flow_threshold`, `cellprob_threshold` y `diameter` sí se aplican. Para usar
cyto2 con denoise (Cellpose v3): `pip install "cellpose<4"`.

**Imágenes soportadas:** TIFF grayscale `(Z, Y, X)` o multicanal
(`Z × C × Y × X` con `channel_axis: 1`, o `Z × Y × X × C` con `channel_axis: 3`).
Una sola imagen 2D no permite reconstrucción 3D real.

**Log:** cada corrida genera `output/logs/pipeline_log.txt` con el registro
completo de la ejecución.
