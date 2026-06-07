## Fase 1

La Fase 1 segmenta células en 3D con Cellpose (`do_3D=True`), reconstruye su
superficie con marching cubes y exporta mediciones volumétricas y morfológicas.

Sea `V` el conjunto de voxeles de la célula en la máscara `(Z, Y, X)`:

| Columna                 | Fórmula                                                                          |
| ----------------------- | -------------------------------------------------------------------------------- |
| `voxel_count`           | `\|V\|`                                                                          |
| `volume_um3`            | `\|V\| × px_xy² × px_z`                                                          |
| `projected_area_xy_um2` | Píxeles únicos en proyección XY × `px_xy²`                                       |
| `z_slices_detected`     | Cortes Z con al menos un voxel de la célula                                      |
| `surface_area_um2`      | Área de la malla triangular (*marching cubes*, `spacing = (px_z, px_xy, px_xy)`) |

## Fase 2

La Fase 2 mide la intensidad Dbc1 en el canal rojo para cada célula segmentada
en Fase 1 y clasifica cada objeto como Dbc1+ o Dbc1− según el umbral elegido.

| Columna               | Fórmula                                                       |
| --------------------- | ------------------------------------------------------------- |
| `mean_intensity_red`  | Media de `red_proj` sobre los píxeles de la célula            |
| `bkg_pp`              | Media de `red_proj` sobre píxeles fuera de todas las máscaras |
| `mean_intensity_corr` | `mean_intensity_red - bkg_pp`                                 |
| `IntDen`              | `area_px × mean_intensity_red`                                |
| `IntDen_corregida`    | `IntDen - PromIntDen_BKG`                                     |
| `IntDen_3D`           | Suma de `red_volume` sobre todos los voxeles de la célula     |
| `IntDen_3D_corr`      | `IntDen_3D - bkg_pp_3d × voxel_count_3d`                      |
