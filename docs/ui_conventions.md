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

## Textos instructivos

Cada pestaña abre con una instrucción en **negrita** y pasos numerados
(`① ② ③`) que dicen al usuario qué hacer, en orden. Mantener este patrón en
nuevas pestañas/fases.
