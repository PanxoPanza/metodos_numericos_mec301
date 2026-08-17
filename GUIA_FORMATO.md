# Guía de formato y diseño — MEC301 · Métodos Numéricos

Convenciones del material de cátedra (`material_catedra/`). Cada capítulo es un notebook que cumple **dos funciones a la vez**: apuntes del curso (jupyter book) y slides en clase (extensión RISE). Todo lo de abajo existe para que las dos funcionen sin conflicto.

---

## 1. Estructura de celdas y slides

**Regla base: una idea por celda.** El texto se revela a medida que avanza la narración; una celda con tres ideas obliga a mostrarlas todas de golpe.

| `slide_type` | uso |
|---|---|
| `slide` | inicia una diapositiva nueva. Lleva el título o la idea principal |
| `fragment` | aparece con un clic sobre la diapositiva actual. Es el modo por defecto para desarrollar una idea |
| `-` (vacío) | se pega a la celda anterior sin clic. Útil para una animación bajo su enunciado |
| `skip` | no aparece en la presentación (se ejecuta igual) |

Presupuesto por diapositiva: **título + 2 a 4 fragmentos**, o una ecuación grande, o una figura. Como referencia, ~900 caracteres por diapositiva ya es demasiado.

Estructura típica de una sección:

```
[slide]     ## Título de la sección
[fragment]  planteamiento / ecuación central
[fragment]  > nota o consecuencia
[slide]     ejemplo o figura
[fragment]  código
[fragment]  > interpretación del resultado
```

Las citas `>` se reservan para **conclusiones, advertencias y notas** — son el "subrayado" del apunte.

---

## 2. Notación matemática

Fijada para todo el curso; respetarla evita las ambigüedades más caras.

| símbolo | significado |
|---|---|
| $m$ | número de datos. El índice de dato es $i = 1,\dots,m$ |
| $n$ | grado del modelo / última función base. El índice es $k = 0,\dots,n$ |
| $\mathbf{x}$ | variable independiente en negrita cuando puede ser un vector, $\mathbf{x} = x_1, x_2, x_3, \ldots$ |
| $S_r$ | suma de los cuadrados de los residuos (nunca solo $S$) |
| $S_t$ | suma total de cuadrados |
| $\eta$ | tasa de aprendizaje ($\alpha$ ya se usa como coeficiente de modelos de potencia) |
| $Z$ | matriz de diseño, $Z_{ik} = \phi_k(\mathbf{x}_i)$ |

**El subíndice simple identifica el dato.** Si una variable independiente tiene varias componentes, se distinguen por contexto o con letras físicas ($V$, $A$, $T$), no reutilizando el índice de dato.

Ecuaciones con `\begin{equation*}` (o `equation` si necesita numeración) y `\begin{align*}`, nunca `$$`. El notebook tiene `amsmath` y `dollarmath` habilitados en `_config.yml`.

---

## 3. Código en las celdas

**Menos código visible es mejor.** El notebook enseña métodos numéricos, no programación; cada línea en pantalla compite con la explicación.

- **`print` siempre con f-strings:**
  ```python
  print(f'y = {a[0]:.3f} + {a[1]:.3f}*x + {a[2]:.5f}*x^2')
  ```
  Nunca `%`-formatting ni `.format()`.
- Preferir una función de librería antes que programar una fórmula a mano, salvo que el objetivo pedagógico *sea* mostrar el algoritmo.
- Comentarios al margen, alineados, en minúscula y en español.
- El formato de los gráficos (`plt.rcParams`) se define **una sola vez** al inicio; después solo `plt.figure(figsize=(5,4))`.
- Nombres consistentes en todo el curso: `xi`, `yi` para los datos; `a` para el vector de coeficientes; `x` para el arreglo de graficado.

---

## 4. Figuras estáticas (matplotlib)

```python
plt.rcParams.update({'font.size': 10, 'figure.dpi': 200})
fig, ax = plt.subplots(figsize=(5, 4))
...
ax.legend(loc='upper left', fontsize=8.5, framealpha=0.95,
          handlelength=1.9, borderpad=0.4, labelspacing=0.32)
fig.tight_layout()
fig.savefig('images/nombre.png', bbox_inches='tight')
```

- **`figsize=(5,4)`** es el estándar del curso. Para diagramas o figuras de dos paneles se puede ampliar, manteniendo el alto.
- Guardar en `images/` del capítulo, en PNG.
- Insertar con HTML, no con markdown, para poder controlar el ancho:
  ```html
  <img src="./images/nombre.png" width="500" align= center>
  ```
  Anchos habituales: **300–600 px**. Con `figsize=(5,4)` (relación 1.25), `width="500"` ocupa una diapositiva cómoda; ajustar hacia abajo si el texto de la celda no cabe.
- Colores: azul `#1f77b4` para datos, rojo `#d62728` para el modelo, gris `#555555` para elementos secundarios, verde `#2ca02c` / `#1c7a3e` para "óptimo/correcto".
- Todo el texto de la figura en español, con unidades.

---

## 5. Animaciones interactivas

Referencia canónica: `02-Algebra_lineal/condicionamiento_interactivo.html`.

**Formato obligatorio:**

1. **Fragmento HTML, no documento.** Solo `<meta charset="utf-8">`, `<style>`, un `<div id="xxx-app">` y `<script>`. **Nunca un iframe**: impide que la animación se adapte al ancho de la diapositiva.
2. **CSS completamente scopeado**: todas las reglas prefijadas con `#xxx-app` y las clases con `xxx-`. Prefijo de 3 letras único por animación (`cnd`, `cdr`, `dgr`, `sbr`). Sin esto, dos animaciones en el mismo notebook se pisan.
3. **JavaScript dentro de un IIFE** `(function(){ ... })()`, con todos los `id` prefijados.
4. **Raíz:**
   ```css
   width:100%; max-width:min(800px, 94vw); padding:4px 2px; background:transparent;
   font-family:"Segoe UI",Calibri,system-ui,sans-serif; font-size:15px; line-height:1.45
   ```
5. **Fuente adaptativa**: un `ResizeObserver` sobre el div raíz fija
   `base = max(13, min(17, ancho/50))` px; todo lo interno en `em`. Guardar el valor anterior para no entrar en bucle con el observer.
6. **Gráficos en SVG con `viewBox`**, nunca canvas:
   ```css
   svg{width:100%; height:auto; display:block; background:#fff;
       border:1px solid var(--linea); border-radius:0.55em}
   ```
   El `viewBox` fija la relación de aspecto; el SVG escala solo.
7. **Texto del gráfico a tamaño aparente constante**: `esc = W_viewBox / anchoRealSVG`, luego `fT = 13*esc` (rótulos) y `fE = 15*esc` (nombres de eje). Márgenes derivados: `ml = max(52, 3.6*fT)`, `mb = max(42, 3.2*fT)` — con menos, la etiqueta del eje x se pisa con los números.
8. **Layout de dos columnas** que colapsa en pantallas angostas:
   ```css
   .wrap{display:flex; gap:1.05em; align-items:flex-start; flex-wrap:wrap}
   .izq {flex:3 1 264px; min-width:0; max-width:448px}   /* gráfico */
   .der {flex:2 1 250px; min-width:0; font-size:0.85em}  /* panel   */
   ```
9. **Paleta:**
   ```
   --tinta:#1a2733  --suave:#5d6b78  --linea:#d8dee4  --panel:#f6f8fa
   --c1:#1f77b4     --c2:#ff7f0e     --rojo:#d62728   --acento:#00afd8
   --verde:#1c7a3e
   ```
10. **Componentes reutilizables:** `.ctrl` (caja de deslizador con `label` + `.val` flotante en color acento + `.hint`), `.eqbox` (fórmula, serif "Cambria Math"), `table.t` (lecturas numéricas, columna `.num` a la derecha con `tabular-nums`), `.badge` (píldora de estado: verde `#e3f4e8`/`#1c7a3e`, ámbar `#fdf3dd`/`#8a6100`, rojo `#fbe4e4`/`#a11`), `.btns` (grid de botones), `.ley` (leyenda bajo el gráfico).
11. **Subíndices:** `<sub>` en HTML, `<tspan baseline-shift="sub">` en SVG. **No usar** los caracteres Unicode `ᵣ`/`ᵢ` en texto corrido: varias fuentes los dibujan como coma.
12. **Embebido**, en celda con tag `remove-input`:
    ```python
    from pathlib import Path
    from IPython.display import HTML
    HTML(Path('interactive/A1_nombre.html').read_text(encoding='utf-8'))
    ```
    (los dos `import` solo la primera vez). Los archivos van junto al notebook o en una subcarpeta `interactive/` si son varios.

**Qué justifica una animación:** que transmita algo que el texto estático no puede — buscar a mano un óptimo, ver una dinámica iterativa, comparar ajuste contra generalización. Dos o tres por capítulo, no más.

---

## 6. Lenguaje

- Español de Chile, registro de cátedra, tuteo al estudiante ("mueve", "pruébalo").
- **"no lineal"**, sin guion. Igual "no lineales".
- **Python**, **NumPy**, **SciPy** con mayúscula cuando se nombra la herramienta; en `código` cuando se nombra el módulo (`numpy.polyfit`).
- Anglicismos solo cuando no hay equivalente instalado (*outlier*, *machine learning*, *output*); en cursiva la primera vez.
- Términos técnicos aceptados: sobreajuste, multivariable, univariable, linealización, sobredeterminado, minimax, matriz de diseño.
- Revisar tildes antes de cerrar: `hunspell -d es_ES -i UTF-8 -l` sobre la prosa extraída atrapa casi todo.

---

## 7. Flujo de trabajo al editar

1. **Backup antes de tocar nada**: copia del notebook original en `NN-Capitulo/backup/` con fecha.
2. **Partir siempre del archivo en disco**, no de una copia previa: el notebook se edita a mano entre sesiones. Diffear antes de escribir.
3. **Escribir solo los archivos que realmente cambiaron** en esa edición.
4. Si se sobrescribe algo por error, `.ipynb_checkpoints/NN-Capitulo-checkpoint.ipynb` guarda la última versión guardada en Jupyter y suele ser recuperable.
5. **Verificación antes de cerrar:**
   - ejecutar el notebook completo desde un kernel limpio, sin errores;
   - revisar la estructura de diapositivas (ninguna sobrecargada);
   - comprobar que todas las imágenes referenciadas existen;
   - validar el LaTeX (KaTeX/MathJax) y correr el corrector ortográfico;
   - probar las animaciones a dos anchos distintos (~1180 px y ~820 px).
6. **Versionar**: rama por capítulo (`mejora-unidadN`) antes de aplicar cambios grandes.

---

## 8. Enfoque pedagógico

Lo que hace que el material funcione, más allá del formato:

- **Un ejemplo conductor por capítulo**, con datos reales, que recorra toda la unidad (en el capítulo 3: el túnel de viento).
- **Motivar antes de formalizar**: primero el problema o la exploración manual, después la derivación.
- **Cerrar cada resultado con su interpretación**, en cita `>`.
- **Conectar hacia adelante y hacia atrás** con otras unidades explícitamente (ajuste ↔ interpolación, ecuaciones normales ↔ álgebra lineal, $\nabla S_r=0$ ↔ búsqueda de raíces).
- **Terminar con una síntesis**: un diagrama del capítulo y una tabla de "qué método uso cuándo".
- Preferir cortar contenido redundante antes que agregar: el material se usa en clase y el tiempo es el recurso escaso.

---

## 9. Bibliografía de referencia

- Chapra S., Canale R. *Métodos Numéricos para Ingenieros*, McGraw Hill — texto base en español.
- Chapra S. *Applied Numerical Methods with MATLAB for Engineers*, 3rd Ed.
- Press W. H. et al. *Numerical Recipes: The Art of Scientific Computing*, 3rd Ed., Cambridge — fundamento teórico y justificación estadística.
- Kong Q., Siauw T., Bayen A. M. *Python Programming and Numerical Methods*, Academic Press.

Citar como: `Autor. **Capítulo N: Título** en *Libro*, Ed., Editorial, Año`.
