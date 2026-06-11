## Fase 1

**Que hace y por que:** la Fase 1 toma el z-stack (TIF o CZI) y detecta cada
nucleo celular como un objeto 3D usando Cellpose (`do_3D=True`). El resultado
es una "mascara" 3D: una imagen del mismo tamano que el z-stack donde cada
voxel vale 0 (fondo) o un numero entero 1..N que identifica a que nucleo
pertenece. A partir de esa mascara se reconstruye la superficie de cada
nucleo con *marching cubes* y se calculan medidas de tamano y forma. Estas
medidas sirven para caracterizar la morfologia de los nucleos (tamano,
volumen, area) independientemente de cualquier marcador de fluorescencia.
La mascara 3D generada aqui (`output/<archivo>/1/masks_3d/<archivo>_masks_3d.tif`)
es el insumo obligatorio de la Fase 2.

Sea `V` el conjunto de voxeles de la célula en la máscara `(Z, Y, X)`:

| Columna                 | Fórmula                                                                          |
| ----------------------- | -------------------------------------------------------------------------------- |
| `voxel_count`           | `\|V\|`                                                                          |
| `volume_um3`            | `\|V\| × px_xy² × px_z`                                                          |
| `projected_area_xy_um2` | Píxeles únicos en proyección XY × `px_xy²`                                       |
| `z_slices_detected`     | Cortes Z con al menos un voxel de la célula                                      |
| `surface_area_um2`      | Área de la malla triangular (*marching cubes*, `spacing = (px_z, px_xy, px_xy)`) |

## Fase 2

**Que hace y por que:** la Fase 2 NO vuelve a segmentar nada; reutiliza las
mascaras 3D que dejo la Fase 1 en `output/<archivo>/1/masks_3d/`. Para cada
nucleo de esa mascara, mide cuanta senal hay en el canal rojo (Dbc1/AF647) del
mismo CZI, tanto en la proyeccion 2D (maxima intensidad en Z) como en el
volumen 3D completo. Con esas mediciones clasifica cada nucleo como `Dbc1+`
(senal alta) o `Dbc1-` (senal baja/fondo) segun un umbral. Si la Fase 1 no se
ejecuto antes (no existen mascaras), la Fase 2 no tiene nada que medir y no
genera resultados.

Notación: `red_proj` / `mask_proj` son las proyecciones máximas en Z del canal
rojo y de la máscara (2D); `red_volume` / `mask_3d` son los volúmenes 3D
completos (Z, Y, X) tal como se cargaron del CZI y de la máscara guardada en
Fase 1.

### Canales (`red_channel` / `blue_channel`)

El CZI tiene varios canales de fluorescencia. `red_channel` indica el índice
(0-based) del canal donde está la señal Dbc1/AF647 que se va a medir y
clasificar; `blue_channel` indica el índice del canal nuclear (DAPI) que solo
se usa para generar las figuras de control de calidad (overlay azul). En Fiji
los canales se numeran C=1, C=2... (1-based), así que C=1 en Fiji equivale a
`red_channel=0` aquí. Si los valores no coinciden con el CZI real, las
mediciones de intensidad serán incorrectas aunque la segmentación sea correcta.

### Columnas de la tabla de resultados

| Columna               | Fórmula                                                       | Notas |
| --------------------- | ------------------------------------------------------------- | ----- |
| `area_px`             | Cantidad de píxeles de la célula en `mask_proj`                | Medida 2D (proyección) |
| `mean_intensity_red`  | Media de `red_proj` sobre los píxeles de la célula            | Valor usado para clasificar Dbc1+/− |
| `bkg_pp`              | Media de `red_proj` sobre píxeles fuera de todas las máscaras | Fondo "por pixel" en 2D |
| `mean_intensity_corr` | `mean_intensity_red - bkg_pp`                                 | Intensidad media corregida por fondo |
| `IntDen`              | `area_px × mean_intensity_red`                                | Densidad de intensidad integrada (2D) |
| `IntDen_corregida`    | `IntDen - PromIntDen_BKG`                                     | `IntDen` corregida por el fondo promedio |
| `voxel_count_3d`      | Cantidad de voxeles de la célula en `mask_3d`                  | Tamaño 3D real del núcleo (no la proyección) |
| `IntDen_3D`           | Suma de `red_volume` sobre todos los voxeles de la célula     | Señal total acumulada en todo el volumen |
| `IntDen_3D_corr`      | `IntDen_3D - bkg_pp_3d × voxel_count_3d`                      | `IntDen_3D` corregida por fondo 3D |
| `mean_intensity_3D`   | Media de `red_volume` sobre los voxeles de la célula           | Promedio de señal en todo el volumen del núcleo |
| `clasificacion`       | `Dbc1+` si `mean_intensity_red >= umbral`, si no `Dbc1-`       | Ver "Modos de umbral" abajo |

### `PromIntDen_BKG` (fondo usado para `IntDen_corregida`)

`PromIntDen_BKG` es el promedio de dos estimaciones de fondo:

- `bkg_image = bkg_pp × mediana(area_px de todas las células)`: cuánta señal
  de fondo "esperaríamos" si el fondo ocupara el área típica de una célula.
- `bkg_cell = mínimo de IntDen entre todas las células`: la célula con menor
  `IntDen` se usa como aproximación de "célula sin señal real".

`PromIntDen_BKG = (bkg_image + bkg_cell) / 2`. Esta corrección solo afecta a
`IntDen_corregida`; no se usa para `IntDen_3D_corr` (que usa `bkg_pp_3d`
directamente).

### Modos de umbral (`threshold_mode`)

El umbral decide el corte entre `Dbc1+` y `Dbc1-` y se calcula sobre
`mean_intensity_red` (2D) de todas las células de la imagen:

- **`otsu`** (por defecto): calcula el umbral automáticamente con el método de
  Otsu, que busca el valor que mejor separa dos grupos (alto/bajo) en el
  histograma de `mean_intensity_red`. Recomendado cuando hay una mezcla clara
  de células positivas y negativas y no se conoce un umbral fijo de antemano.
- **`factor`**: `umbral = media(mean_intensity_red) - factor × SD(mean_intensity_red)`.
  Útil cuando Otsu da resultados inestables (p.ej. pocas células o
  distribución poco bimodal). Un `factor` bajo (ej. 0.5) sube el umbral
  (más células clasificadas como `Dbc1-`); un `factor` alto (ej. 1.6-2.0) baja
  el umbral (más células `Dbc1+`).
- **`fixed`**: usa directamente el valor de `threshold` como umbral en la
  escala raw del detector (p.ej. 0-65535 para 16 bits), sin calcular nada a
  partir de los datos. Útil para reproducir un umbral validado manualmente en
  Fiji/ImageJ o para comparar lotes con el mismo criterio absoluto.

El campo `metodo_umbral` del CSV registra cuál de los tres modos se usó
realmente (`otsu`, `mean-{factor}sd` o `fixed({valor})`).
