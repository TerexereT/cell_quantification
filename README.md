# cell_3d_analysis — Análisis 3D de células a partir de z-stacks de microscopía

Pipeline en Python que segmenta células en 3D con **Cellpose** (`do_3D=True`), reconstruye su superficie con *marching cubes* y exporta mediciones volumétricas (Fase 1), clasifica cada célula por intensidad de señal Dbc1 (Fase 2) y grafica de forma interactiva cualquier par de columnas de los CSV de mediciones exportando a PNG (pestaña Graficar).

---

## 1. Instalación

### Requisitos previos

- Python 3.9 o superior instalado y en el PATH.
- El proyecto clonado o descargado en tu máquina.

### Instalar

El script `setup.ps1` hace todo el proceso en un solo paso: crea el entorno
virtual, detecta si tienes una GPU NVIDIA y, si la hay, instala el build de
PyTorch con soporte CUDA correcto (incluyendo GPUs Blackwell como la serie
RTX 50xx) antes de instalar el resto de dependencias.

Desde la carpeta raíz del proyecto (`cell_3d_analysis/`):

```powershell
# Si PowerShell bloquea la ejecución de scripts, primero:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

.\setup.ps1
```

Al terminar, verás impreso si CUDA está disponible y el nombre de la GPU
detectada. Activa el entorno en cada sesión nueva con:

```powershell
.\venv\Scripts\Activate.ps1
```

Cuando el entorno está activo verás `(venv)` al inicio del prompt.

> El driver de NVIDIA **no** se instala automáticamente (requiere permisos de
> administrador y normalmente un reinicio). Si `setup.ps1` reporta que CUDA
> no está disponible pese a tener una GPU NVIDIA, instala/actualiza el driver
> desde https://www.nvidia.com/drivers y vuelve a correr `.\setup.ps1`.

---

## Aplicación de escritorio (.exe)

La app de escritorio tiene tres pestañas: **"Fase 1"** (segmentación 3D),
**"Fase 2"** (medición de intensidad y clasificación Dbc1) y **"Graficar"**
(visualizar y exportar los CSV de mediciones, ver sección 4b). El archivo CZI,
el canal y la carpeta de Salida se eligen en la zona común superior y son
compartidos por las Fases 1 y 2; el log de ejecución es común abajo.

Las pestañas Fase 1 y Fase 2 permiten correr el pipeline sin usar PowerShell ni
editar `config/config.yaml` a mano. Veras selectores para:

- Archivo `.czi` de entrada.
- Canal de segmentacion para Fase 1.
- Parametros de Cellpose (`diameter`, `flow_threshold`, `cellprob_threshold`,
  `min_size_voxels`, `gpu`).
- Parametros de Dbc1 para Fase 2 (`otsu`, `factor` o `fixed`, canales rojo/azul).
- Carpeta raiz de salida.

Los valores iniciales salen de `config/config.yaml`. Puedes cambiarlos en el
formulario antes de ejecutar; esos cambios aplican solo a esa corrida y no
sobrescriben el YAML.

La salida mantiene la misma estructura que el CLI:

```text
<carpeta_elegida>/<nombre_czi>/1/
<carpeta_elegida>/<nombre_czi>/2/
```

El panel GPU muestra si CUDA esta disponible. Si tienes una GPU AMD en Windows,
la app informa que PyTorch no acelera con ROCm en Windows y que se usara CPU.
El mensaje recomienda instalar o actualizar drivers de GPU para obtener
respuestas mas rapidas.

El ejecutable se entrega sin firma digital. Windows SmartScreen puede mostrar
una advertencia de "editor desconocido"; para continuar, selecciona
`Mas informacion` y luego `Ejecutar de todas formas`.

---

## 2. Preparar los datos de entrada

Para Fase 1 solo necesitas el archivo `.czi` original — no hace falta copiarlo
al proyecto ni llenar `metadata.csv`. El pipeline:

- Extrae el canal indicado del `.czi`.
- Lee `px_xy_um` y `px_z_um` directamente del encabezado.
- Genera las salidas en `output/<nombre_czi>/1/`.

> **Nota sobre las imágenes `cre+342_17`:** el `.czi` es `C × Z × Y × X = 2 × 6 × 1024 × 1024`.
> Canal 0 = AF647 (señal a medir), canal 1 = DAPI (núcleos a segmentar).
> Calibración extraída automáticamente: `px_xy = 0.099 µm`, `px_z = 2.0 µm`.

> ¿Prefieres convertir el `.czi` a TIFF y trabajar con `metadata.csv` a mano?
> Consulta el [Apéndice: procesamiento manual desde TIFF](#apéndice-procesamiento-manual-desde-tiff-y-metadatacsv)
> al final de este documento.

---

## 3. Fase 1 — Segmentación 3D

### Ejecutar

Desde la raíz del proyecto, con el entorno activado, apunta directo al `.czi`:

```powershell
.\venv\Scripts\python.exe src/main.py --czi "C:\ruta\a\imagen.czi" --channel 1
```

Antes de procesar, el programa imprime un resumen para confirmar:

- Ruta del `.czi` de entrada.
- Canal que se va a segmentar.
- Forma del volumen extraído como `(Z, Y, X)`.
- Calibración detectada (`px_xy_um` y `px_z_um`) o `NO detectada`.
- Carpeta de salida que se va a crear: `output/<nombre_czi>/1/`.

Si los parámetros son correctos, responde `y`, `yes`, `s`, `si` o `sí`. Cualquier
otra respuesta, incluido Enter vacío, cancela antes de correr Cellpose.

Opciones de ejecución:

| Opción | Uso |
| ------ | --- |
| `--czi "C:\ruta\a\imagen.czi"` | Procesa ese `.czi` directamente, sin leer `metadata.csv`. |
| `--channel 0` | Canal a extraer y segmentar. Default: `0`. |
| `--yes` o `-y` | Omite la confirmación interactiva. Útil para scripts o CI. |
| `--config config/config.yaml` | Mantiene los parámetros de segmentación, medición, QC y `output_dir`. |

Ejemplos:

```powershell
# Interactivo: muestra resumen y pide confirmación
.\venv\Scripts\python.exe src/main.py --czi "C:\ruta\a\imagen.czi" --channel 1

# Automatizado: muestra resumen pero no pide confirmación
.\venv\Scripts\python.exe src/main.py --czi "C:\ruta\a\imagen.czi" --channel 1 --yes

# Usando otra configuración
.\venv\Scripts\python.exe src/main.py --config config/config.yaml --czi "C:\ruta\a\imagen.czi" --channel 0 --yes
```

Si el `.czi` no tiene calibración legible, el resumen muestra `NO detectada` y
el log registra una advertencia. El procesamiento requiere `px_xy_um` y
`px_z_um`; si alguno queda sin detectar, Fase 1 falla con un error de calibración.

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

Orden de generación:

| Orden | Carpeta         | Archivo                        | Contenido                                                    |
| ----- | --------------- | ------------------------------ | ------------------------------------------------------------ |
| 1     | `masks_3d/`     | `<nombre>_masks_3d.tif`        | Máscara 3D etiquetada (0 = fondo, 1..N = células)            |
| 2     | `meshes/`       | `<nombre>_cell_<id>.obj`       | Malla 3D por célula (si `save_individual_cell_meshes: true`) |
| 3     | `measurements/` | `<nombre>_measurements_3d.csv` | Una fila por célula con métricas morfológicas 3D             |
| 4     | `projections/`  | `<nombre>_max_projection.tif`  | Proyección de intensidad máxima de la imagen segmentada      |
| 5     | `projections/`  | `<nombre>_mask_projection.tif` | Proyección máxima de la máscara                              |
| 6     | `figures_qc/`   | `<nombre>_qc_overlay.png`      | 3 paneles: original / máscara / overlay                      |

Para entrada CZI, `<nombre>` sale del nombre del archivo `.czi` sin extensión.
Ejemplo: `imagen.czi` genera `output/imagen/1/`.

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

## 4b. Graficar — Gráficas interactivas de las mediciones

La función Graficar vive en la **app de escritorio**, en la pestaña **"Graficar"**.
Toma cualquier CSV de mediciones generado por la Fase 1
(`*_measurements_3d.csv`) o la Fase 2 (`*_dbc1_intensity.csv`) y lo grafica sin
escribir código.

### Cómo usarla

1. Abre la app (`.exe` o `python -m gui`) y ve a la pestaña **"Graficar"**.
2. El selector **CSV mediciones** se autocompleta con todos los CSV encontrados
   bajo la carpeta de salida actual (botón **"Buscar CSV en carpeta de salida"**
   para refrescar). También puedes elegir cualquier CSV con **`...`**.
3. Elige la columna del eje **X** y la del eje **Y** (solo se listan columnas
   numéricas).
4. Elige el **Tipo** de gráfica: `Dispersion (puntos)` o `Linea`.
5. La gráfica se **actualiza automáticamente** al cambiar cualquier selector.
6. Cuando quede como quieres, pulsa **"Exportar PNG"** y elige dónde guardarla
   (200 dpi).

### Detalles

- La fila de metadatos de la Fase 2 (`cell_id == "__metadata__"`) se descarta
  automáticamente, así no contamina la gráfica.
- En modo `Linea` los puntos se ordenan por X antes de unirlos.
- El nombre sugerido del PNG es `<csv>_<Y>_vs_<X>.png` junto al CSV de origen.
- La lógica de datos vive en `gui/plot_panel.py` (testeada en
  `tests/test_plot_panel.py`); la UI se embebe con `matplotlib` (`FigureCanvasTkAgg`).

> Graficar no agrega dependencias nuevas: `pandas` y `matplotlib` ya formaban
> parte de `requirements.txt`. No hace falta reinstalar nada si ya seguiste el
> paso de instalación; si partes de cero, `.\setup.ps1` instala todo lo necesario.

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

---

## Reconstruir el .exe

Se usa PyInstaller en modo `onedir`, mas confiable para dependencias nativas
como `torch`, `cellpose`, `scikit-image` y `matplotlib`.

Desde la raiz del proyecto (`cell_3d_analysis/`):

```powershell
.\setup.ps1
.\build_exe.ps1
```

El script instala `requirements-build.txt`, ejecuta
`pyinstaller build/cell3d_gui.spec --noconfirm` y deja el resultado en:

```text
dist/cell3d_gui/cell3d_gui.exe
```

Notas de distribucion:

- El build no incluye firma digital ni pasos de `signtool`.
- El resultado puede pesar varios GB porque empaqueta dependencias cientificas y
  PyTorch.
- SmartScreen puede mostrar "editor desconocido" la primera vez que se ejecute.

---

## Apéndice: procesamiento manual desde TIFF y metadata.csv

Si prefieres convertir tus `.czi` a TIFF antes de correr el pipeline (por
ejemplo, para inspeccionar o reutilizar el TIFF), sigue este flujo manual.

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

### Convertir un .czi a TIFF

Usa el conversor incluido para generar el TIFF y agregar su fila a `metadata.csv`:

```powershell
# Convierte el canal 1 (DAPI) y agrega la fila a metadata.csv automáticamente
.\venv\Scripts\python.exe tools/czi_to_tiff.py "C:\ruta\a\imagen.czi" --channel 1 --append-metadata
```

El conversor:
- Extrae el canal indicado y guarda el TIFF en `input/raw_zstacks/`.
- Lee `px_xy_um` y `px_z_um` del `.czi` e imprime sus valores.
- Con `--append-metadata`, los escribe directamente en `metadata.csv`.

### Ejecutar Fase 1 desde metadata.csv

Con el TIFF ya en `input/raw_zstacks/` y su fila en `metadata.csv`, procesa
todas las imágenes listadas:

```powershell
.\venv\Scripts\python.exe src/main.py --config config/config.yaml
```
