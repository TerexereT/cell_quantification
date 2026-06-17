# Convenciones de UI (Tkinter) — cell_3d_analysis

Guía para mantener consistencia y usabilidad en la app de escritorio (`gui/`).

## Botones de ayuda `?` pegados a su campo

**Regla:** el botón de ayuda `?` de un parámetro debe quedar **inmediatamente a
la derecha del campo que explica**, nunca en una columna `grid` separada que
pueda estirarse y alejarlo.

**Por qué:** cuando el `?` se coloca en `column=2` de un `grid` cuya `column=1`
tiene `weight=1`, la columna se estira y el botón se va al borde derecho,
quedando visualmente desconectado del valor. Un usuario final no lo asocia con
el campo y no lo encuentra.

**Cómo aplicarlo:** usa el helper `Cell3DApp._field_row(parent, row, label,
widget_factory, key)`. Mete el campo (Entry/Combobox) y el `?` dentro de un
mismo contenedor `ttk.Frame` con `pack(side="left")`, de modo que viajen juntos
sin depender del estiramiento de la columna.

```python
self._field_row(
    phase1, row, label,
    lambda c, k=key: ttk.Entry(c, textvariable=phase1_vars[k], width=16),
    key,  # clave en FIELD_HELP para el texto del popup
)
```

`widget_factory` recibe el contenedor y devuelve el widget ya creado dentro de
él (no lo posiciones tú: el helper hace el `pack`). Captura variables de bucle
con argumentos por defecto (`k=key`, `v=var`) para evitar el late-binding de
lambdas.

## Área de progreso / log redimensionable

El notebook (Fase 1/2/Graficar) y el panel inferior (métricas + barra de
progreso + log) viven en un **`ttk.PanedWindow(orient="vertical")`**. El **sash**
da al usuario **altura editable** del cuadro de salida; al construir se coloca con
`after_idle` en `notebook.winfo_reqheight()` para dejar el notebook compacto (gap
mínimo con el botón "Ejecutar Fase X") y el log alto.

El panel inferior solo es visible en Fase 1/Fase 2: en `_on_tab_changed` se
`add`/`forget` el pane (no `grid_remove`). Encima del log, una fila muestra
`Tiempo · %/ETA (Fase 2) · CPU · GPU` (`resource_monitor.format_metrics`),
alimentada por un hilo daemon (`_metrics_loop`) que se refresca cada ~1 s mientras
`self.running`. El progreso es **híbrido**: Fase 1 barra indeterminada animada;
Fase 2 determinada por máscara (cuenta mensajes con prefijo
`phase2_intensity.MASK_PROGRESS_PREFIX`).

## Acciones de Fase 1

Fase 1 usa dos botones con responsabilidades distintas:

- `Generar`: siempre disponible cuando no hay ejecución activa. Crea o reutiliza
  una variante QC de segmentación en `1/*/variants/<variant_id>/` y actualiza
  `1/figures_qc/medidas.md` para lectura manual y
  `1/figures_qc/phase1_cache.json` para el programa.
- `Finalizar`: solo disponible cuando hay una variante QC pendiente. Copia la
  variante activa a las rutas canónicas de Fase 1 y genera mediciones/mallas para
  habilitar Fase 2. Al cerrar la Fase 1 conserva solo las 3 variantes QC más
  recientes y elimina las carpetas de variantes más antiguas.

No habilitar Fase 2 por la sola presencia de cualquier `.tif` en `1/masks_3d/`.
La UI debe exigir cache finalizado y mostrar un mensaje accionable: ejecutar
`Generar` y luego `Finalizar Fase 1`.

## Textos instructivos

Cada pestaña abre con una instrucción en **negrita** y pasos numerados
(`① ② ③`) que dicen al usuario qué hacer, en orden. Mantener este patrón en
nuevas pestañas/fases.
