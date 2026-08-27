# Guía de formato y diseño — MEC301 · Métodos Numéricos

**Versión 2026.** Convenciones del material de cátedra (`material_catedra/`).

Cada capítulo es un notebook que cumple **dos funciones a la vez**: apuntes del curso
(Jupyter Book) y diapositivas en clase (extensión RISE). Todo lo que sigue existe para que
esas dos funciones convivan sin conflicto, y para que **las once unidades se lean como un
solo curso encadenado y no como once islas**.

Esta guía es **autocontenida**: no requiere leer ningún otro documento del repositorio.

> **Cómo leerla.** Las secciones 1 a 4 son las que definen el material (arquitectura,
> esqueleto, encadenamiento, diapositivas). Las secciones 5 a 9 son convenciones de
> ejecución. Las secciones 10 a 12 son el procedimiento de trabajo. Los anexos son
> registro: qué hay hoy en cada unidad y qué falta.

**Jerarquía de reglas.** Cuando dos criterios choquen, gana el de más arriba:

1. que el estudiante entienda;
2. que la unidad se conecte con el resto del curso;
3. que la diapositiva se lea en proyector;
4. que el apunte se lea bien en el libro web;
5. la consistencia cosmética.

---

## 1. Arquitectura del curso

El curso tiene once unidades en tres partes (`_toc.yml`):

| Parte | Unidades |
|---|---|
| 1. Introducción a los métodos numéricos | 01 Aspectos generales · 02 Álgebra lineal · 03 Ajuste de curvas · 04 Interpolación |
| 2. Métodos basados en series de Taylor | 05 Series de Taylor · 06 Búsqueda de raíces · 07 Integración |
| 3. Métodos para ecuaciones diferenciales | 08 Derivación numérica · 09 EDO valor inicial · 10 EDO valor de frontera · 11 Diferencias finitas para EDP |

### 1.1 Las tres ideas que atraviesan todo el curso

Casi cualquier contenido del curso se puede colgar de uno de estos tres ejes. Nombrarlos
explícitamente en cada unidad es lo que evita que el material se sienta como una colección
de recetas:

**(A) Costo — ¿cuánto cuesta?** Introducido en la Unidad 1 con la notación $O(N)$.
Reaparece en el conteo de operaciones de la eliminación de Gauss ($O(n^3)$), en por qué LU
se factoriza una sola vez, en por qué el algoritmo de Thomas vale la pena, en el número de
evaluaciones de función de una cuadratura y en el tamaño de paso de una EDO.

**(B) Error — ¿cuánto me equivoco?** Introducido en la Unidad 1 (error absoluto y
relativo), formalizado en la Unidad 5 (truncamiento vs. redondeo) y aplicado en cada método
posterior: orden de convergencia, condicionamiento, estabilidad, rigidez.

**(C) Reducción — ¿a qué problema ya resuelto lo llevo?** Es el eje más importante y el
menos evidente para el estudiante. Casi todo método numérico del curso termina en uno de
tres problemas canónicos:

> **resolver $A\mathbf{x} = \mathbf{b}$** (Unidad 2) · **encontrar la raíz de $f(x)=0$**
> (Unidad 6) · **aproximar por un polinomio** (Unidades 4 y 5).

Cuando un capítulo llegue a uno de esos tres, hay que **decirlo con todas sus letras**.

### 1.2 Mapa de dependencias (el engranaje)

Flechas: "lo de la izquierda es lo que hace funcionar a lo de la derecha".

```
U1 Costo O(N) y error  ─────────────────────────────────────────────► (todo el curso)

U2 Sistemas lineales   ──► U3  ecuaciones normales Z'Z a = Z'y
   Ax = b, Cond(A)     ──► U4  matriz de Vandermonde · sistema tridiagonal del spline
   Gauss / LU / iter.  ──► U6  Newton vectorial: J·Δx = −F
                       ──► U10 diferencias finitas ⇒ sistema tridiagonal
                       ──► U11 EDP ⇒ sistema grande y disperso (métodos iterativos)

U3 Mínimos cuadrados   ──► U6  ∇Sr = 0 es búsqueda de raíces (Gauss-Newton, LM)
   ∇Sr = 0             ──► (machine learning: pérdida + descenso de gradiente)

U4 Interpolación       ──► U7  integrar el interpolante = trapecio / Simpson
                       ──► U8  derivar el interpolante = diferencias finitas
                       ──► U10 método de colocación

U5 Series de Taylor    ──► U6  Newton-Raphson = Taylor truncado a primer orden
                       ──► U7  orden de error de las cuadraturas
                       ──► U8  deducción y orden de las diferencias finitas
                       ──► U9  Euler = Taylor de primer orden; RK = órdenes superiores

U6 Raíces              ──► U9  Euler implícito: cada paso es una ecuación no lineal
                       ──► U10 método de disparo = raíz en la condición inicial faltante
                       ──► U11 EDP no lineal

U7 Integración         ──► U9  integrar la EDO; RK como cuadratura del paso

U8 Diferencias finitas ──► U10 y U11 (discretización del operador diferencial)

U9 EDO valor inicial   ──► U10 el disparo llama a solve_ivp

U10 EDO frontera 1D    ──► U11 la misma receta en 2D
```

### 1.3 Regla de oro del encadenamiento

**Ninguna unidad abre en el vacío ni cierra en el vacío.** Ver sección 3.

---

## 2. Anatomía de un capítulo

Esta estructura es **norma del curso**. Un capítulo nuevo o rediseñado la sigue; las
desviaciones deben justificarse.

```
# Título de la unidad                                  [slide]
   celda de librerías del capítulo                     [skip, tag remove-input]

## Introducción                                        [slide]
   problema de ingeniería concreto, con figura o datos reales
   qué herramienta de la unidad anterior nos deja cortos aquí   ← puente hacia atrás
   qué vamos a poder hacer al final del capítulo

## <Familia de métodos 1>                              [slide]
### <método A>   ...   ### <método B>
   por cada método: idea → derivación → ejemplo con el problema conductor → interpretación

## <Familia de métodos 2>                              [slide]
   ...

## <Tema> en Python                                    [slide]
   la función de librería que se usa en la práctica, con sus trampas

## Resumen: ¿qué método uso?                           [slide]
   tabla situación → método → costo → función de Python
   diagrama del capítulo (images/mapa_unidadN.png)
   puente hacia adelante: dónde reaparece esto            ← puente hacia adelante

## Referencias                                         [slide]
```

**Obligatorios:** el título `#`, `## Introducción`, `## Resumen: ¿qué método uso?` y
`## Referencias`. La sección "en Python" es obligatoria salvo que las funciones de librería
ya se hayan mostrado método por método.

**Jerarquía de encabezados:** `#` solo el título del capítulo · `##` sección mayor ·
`###` método o subtema · `####` matiz dentro de un método (usar con moderación; si un
capítulo tiene muchos `####`, probablemente hay que reorganizar en `###`).

**Tamaño.** Entre 90 y 130 celdas es el rango cómodo para una unidad completa
(U3 tiene 117, U1 tiene 94). Sobre 200 celdas (U2 tiene 241) el capítulo ya no cabe en el
tiempo de clase y conviene evaluar qué recortar.

---

## 3. Encadenamiento entre unidades

Lo que convierte once capítulos en un curso. Tres mecanismos, todos obligatorios:

### 3.1 Puente hacia atrás (en la Introducción)

La introducción debe dejar claro **por qué esta unidad viene después de la anterior**: qué
problema nuevo aparece, o qué límite de lo ya visto obliga a una herramienta nueva.
No basta con "en la unidad anterior vimos X"; hay que mostrar el problema que X no resuelve.

Ejemplo real (U3, celda 13): el sistema de $m$ ecuaciones y 2 incógnitas de una recta
está **sobredeterminado** — la maquinaria de la U2 no aplica directamente, y de ahí nace
mínimos cuadrados.

### 3.2 Notas puente (en cualquier punto del capítulo)

Cuando aparece un objeto que ya se vio o que se verá, se marca con una cita `>` que nombra
la unidad **con número**:

> El sistema $2\times2$ que derivamos para la recta es exactamente $Z^TZ\,\mathbf{a} = Z^T\mathbf{y}$
> con $n=1$ — un sistema lineal de los de la **Unidad 2**.

> Por eso `curve_fit` pide un punto inicial (`p0`) y puede converger a una solución local:
> son los mismos riesgos de cualquier método de búsqueda de raíces no lineal (**Unidad 6**).

Reglas de la nota puente:

- **Nombrar la unidad por número en negrita** (`**Unidad 6**`), no "más adelante".
- Decir **qué objeto** viaja (una matriz tridiagonal, una raíz, un polinomio), no solo el tema.
- Si el puente apunta hacia adelante, no adelantar el método: solo plantar la pregunta.
- Entre 3 y 6 notas puente por capítulo. Menos, y la unidad flota; más, y se vuelve ruido.

### 3.3 Cierre con mapa

La sección `## Resumen: ¿qué método uso?` cierra con:

1. **la tabla de decisión** (situación → método → costo → función de Python);
2. **un diagrama** del capítulo, `images/mapa_unidadN.png`, que muestre cómo se relacionan
   los métodos vistos (referencia: `03-Ajuste_de_curvas/images/mapa_unidad3.png`);
3. **verificaciones que hay que hacer siempre** antes de confiar en un resultado
   (referencia: U2, celdas 237-239);
4. **el puente hacia adelante**: en qué unidad reaparece esto.

### 3.4 Catálogo de puentes ya escritos

Registro de lo que ya existe en el material, para no duplicarlo ni contradecirlo.

| Desde | Hacia | Puente |
|---|---|---|
| U2 §Matriz singular | U3 | "reescritura algebraica exacta ≠ linealización" |
| U2 §Repaso de matrices | U9, U10, U11 | matriz tridiagonal: aparece en EDO y diferencias finitas |
| U2 §Hilbert | U3 | la matriz de Hilbert nace de ajustar polinomios por mínimos cuadrados |
| U2 §Resumen | U10, U11 | Thomas: LU tridiagonal en $O(n)$ |
| U3 §Ajuste vs. interpolación | U4 | ruido ⇒ tendencia; sin ruido ⇒ pasar por los puntos |
| U3 §Ecuaciones normales | U2 | $Z^TZ\mathbf{a}=Z^T\mathbf{y}$ es un sistema lineal |
| U3 §En el fondo todo es buscar una raíz | U6 | $\nabla S_r=\mathbf{0}$; Gauss-Newton y LM son Newton |
| U3 §Del ajuste al machine learning | — | modelo → pérdida → optimización |
| U4 §Introducción | U3 | dato con ruido ⇒ tendencia; dato exacto ⇒ pasar por él |
| U4 §Vandermonde | U2 | la matriz $V$ y su $\mathrm{Cond}(V)$ |
| U4 §Newton y Lagrange | U7, U8 | la base $L_i(x)$: integrarla y derivarla |
| U4 §Error de interpolación | U5 | el resto de Taylor |
| U4 §Sistema tridiagonal | U2, U10, U11 | la matriz tridiagonal y el algoritmo de Thomas |
| U4 §Derivadas del spline | U8 | derivar el interpolante |

**Puentes pendientes de escribir** (unidades aún no rediseñadas): ver Anexo C.

---

## 4. Celdas y diapositivas (RISE + Jupyter Book)

**Regla base: una idea por celda.** El texto se revela a medida que avanza la narración;
una celda con tres ideas obliga a mostrarlas todas de golpe.

| `slide_type` | uso |
|---|---|
| `slide` | inicia una diapositiva nueva. Lleva el título o la idea principal |
| `fragment` | aparece con un clic sobre la diapositiva actual. Modo por defecto para desarrollar una idea |
| `-` (vacío) | se pega a la celda anterior sin clic. Útil para una animación o figura bajo su enunciado |
| `skip` | no aparece en la presentación, pero se ejecuta igual (celda de librerías, generación de figuras) |

Presupuesto por diapositiva: **título + 2 a 4 fragmentos**, o una ecuación grande, o una
figura, o una animación. Como referencia, ~900 caracteres en una diapositiva ya es demasiado.

Patrón típico de una sección:

```
[slide]     ### Título del método
[fragment]  planteamiento / ecuación central
[fragment]  > nota, consecuencia o advertencia
[slide]     ejemplo con el problema conductor (figura o datos)
[fragment]  código
[fragment]  > interpretación del resultado
```

**Las citas `>` son el subrayado del apunte.** Se reservan para conclusiones,
advertencias, notas puente e interpretaciones de resultado. Una cita que solo repite el
párrafo anterior es ruido.

**En el libro web** las citas se ven como bloques con barra lateral y el `slide_type` se
ignora, así que el capítulo debe leerse de corrido: una celda `fragment` tiene que tener
sentido leída inmediatamente después de la anterior, sin la pausa del clic.

---

## 5. Notación matemática

### 5.1 Convenciones globales del curso

| Convención | Detalle |
|---|---|
| Índice de dato / nodo | $i = 1,\dots,m$ (o $i=0,\dots,n$ cuando la numeración natural parte en 0, como en nodos de una malla) |
| Índice de iteración | superíndice entre paréntesis o $k$: $x^{(k)}$, $x_k$. **Nunca** mezclar índice de dato e índice de iteración en el mismo símbolo sin advertirlo |
| Vectores y matrices | vectores en negrita minúscula ($\mathbf{x}$, $\mathbf{a}$), matrices en mayúscula ($A$, $Z$, $J$). Un escalar nunca va en negrita |
| Sistema lineal canónico | $A\mathbf{x} = \mathbf{b}$. Cuando el lado derecho son datos medidos, se admite $\mathbf{y}$ |
| Aproximado vs. exacto | el valor exacto sin adorno, la aproximación numérica con tilde o subíndice: $x$ vs. $\tilde{x}$ o $x_\mathrm{num}$ |
| Error | $\varepsilon$ para error relativo, $e$ para residuo o error puntual. Definirlos la primera vez que aparecen en el capítulo |
| Paso | $h$ para paso genérico o espacial, $\Delta t$ para paso temporal |
| Variables físicas | conservan su letra habitual ($T$ temperatura, $V$ velocidad, $F$ fuerza, $k$ rigidez) y **siempre llevan unidades** cuando se tabulan o grafican |
| Orden de complejidad | $O(\cdot)$ con la letra del tamaño del problema explícita: $O(n^3)$, $O(N\log N)$ |

**Formato.** Ecuaciones desplegadas con `\begin{equation*}` (o `equation` si se necesita
numerar) y `\begin{align*}`; **nunca `$$`**. Matemática en línea con `$...$`.
`amsmath` y `dollarmath` están habilitados en `_config.yml`.

**Compatibilidad KaTeX.** Lo que se renderiza en el libro web es KaTeX: no usar `\mbox`,
`\eqnarray` ni entornos de `mathtools`. `\mathrm`, `\mathbf`, `\text`, `align*`,
`equation*`, `cases`, `pmatrix` y `bmatrix` funcionan.

**Colisiones de símbolos.** Antes de introducir una letra nueva en un capítulo, revisar
que no choque con un uso ya fijado en otra unidad (Anexo A). Si choca y no hay alternativa,
advertirlo con una cita `>`.

### 5.2 Notación local de cada unidad

Cada capítulo puede fijar símbolos propios, pero:

- se **declaran donde se introducen**, no se asumen;
- se usan **consistentemente dentro del capítulo**;
- **no se exportan** a otras unidades salvo que el Anexo A lo diga.

El registro de la notación local vive en el **Anexo A** de esta guía. Al fijar un símbolo
nuevo en un capítulo, agregarlo ahí.

---

## 6. Código en las celdas

**Menos código visible es mejor.** El notebook enseña métodos numéricos, no programación:
cada línea en pantalla compite con la explicación.

### 6.1 Reglas

- **`print` siempre con f-strings.** Nunca `%`-formatting ni `.format()`, ni en `print`
  ni en etiquetas de gráficos:
  ```python
  print(f'y = {a[0]:.3f} + {a[1]:.3f}*x')
  plt.plot(x, y(x), '-r', label=f'y = {a[0]:.2f} + {a[1]:.2f}x')
  ```
- **Preferir la función de librería** antes que programar la fórmula a mano. La excepción
  es cuando el objetivo pedagógico *es* mostrar el algoritmo (eliminación de Gauss,
  Gauss-Seidel, bisección, Euler): en ese caso, la implementación a mano va **una sola vez**
  y después se usa la de librería.
- **Celda de librerías al inicio del capítulo**, con `slide_type: skip` y tag
  `remove-input`, encabezada por `# Librerías utilizadas en este capítulo`. Los `import`
  que aparecen otra vez más adelante son deliberados: sirven para que el estudiante pueda
  copiar una celda suelta a Colab. No hay que eliminarlos.
- **Comentarios al margen**, alineados, en minúscula y en español.
- **`plt.rcParams` se define una sola vez** en la celda de librerías; después, en cada
  figura, basta `plt.figure(figsize=(5,4))`. Los capítulos aún no rediseñados repiten
  `plt.rcParams.update({'font.size': ...})` en cada celda de gráfico, con tamaños que van
  de 10 a 18: al pasar por un capítulo, consolidar eso en la celda de librerías con
  `font.size: 10`.
- Nada de estado oculto: una celda de código no debe depender de una variable definida
  20 celdas antes sin que el texto lo recuerde.

### 6.2 Nombres de variables (comunes a todo el curso)

Son los nombres que ya usa el material. Aplicarlos en código nuevo; **no** hacer renombres
masivos en capítulos que funcionan.

| Rol | Nombre |
|---|---|
| Datos tabulados (medidos o dados) | `xi`, `yi` |
| Arreglo denso para graficar | `x` (y `y` o `y(x)` para el modelo evaluado) |
| Coeficientes del modelo / solución del ajuste | `a` |
| Matriz del sistema y lado derecho | `A`, `b` |
| Solución del sistema | `x_sol` (o `x` si no hay ambigüedad con el arreglo de ploteo) |
| Función y su derivada | `f`, `dfdx` |
| Nº de datos · nº de nodos o incógnitas | `m` · `n` (o `N` para tamaño de input en U1) |
| Pasos | `h`, `dx`, `dt` |
| Condición inicial | `x0`, `y0`, `t0` |
| Tolerancia e iteraciones | `tol`, `max_iter` |
| Errores | `error_abs`, `error_rel` |
| Objeto de resultado de SciPy | `sol` |
| Variables físicas | su letra física: `T`, `V`, `F`, `Ta`, `Tb` |

**No** usar `l` ni `O` como nombres de variable (se confunden con 1 y 0), ni tildes ni `ñ`
en identificadores.

### 6.3 Librerías del curso

`requirements.txt` declara: **numpy, scipy, pandas, matplotlib, scikit-learn**. En la
práctica el curso vive de NumPy, Matplotlib y SciPy, con scikit-learn usado puntualmente
(`r2_score`). No introducir una librería nueva sin que resuelva algo que las anteriores no
pueden, y si se introduce, agregarla a `requirements.txt`.

---

## 7. Figuras estáticas (Matplotlib)

```python
plt.rcParams.update({'font.size': 10, 'figure.dpi': 200})
fig, ax = plt.subplots(figsize=(5, 4))
...
ax.legend(loc='upper left', fontsize=8.5, framealpha=0.95,
          handlelength=1.9, borderpad=0.4, labelspacing=0.32)
fig.tight_layout()
fig.savefig('images/nombre.png', bbox_inches='tight')
```

- **`figsize=(5, 4)` o `(5, 3)`** son los tamaños de referencia; ancho ~5 pulgadas y alto
  entre 3 y 4. Para diagramas o paneles múltiples se puede ampliar el ancho manteniendo el
  alto, de modo que todas las figuras del curso se vean a la misma escala relativa.
- Guardar en la carpeta de imágenes del capítulo, en PNG. La carpeta es `images/` en todos
  los capítulos **salvo la Unidad 1, que usa `imagenes/`**: respetar la que exista.
- Insertar con HTML, no con markdown, para poder controlar el ancho:
  ```html
  <img src="./images/nombre.png" width="500" align= center>
  ```
- **Ojo con el ancho y RISE.** El CSS del capítulo escala las imágenes en modo presentación
  (`--rise-image-scale: 1.6` para imágenes con atributo `width`, `--rise-figure-scale: 1.3`
  para las figuras generadas por código). Es decir, `width="500"` en el apunte se ve a ~800 px
  en la diapositiva. Elegir el ancho pensando en el **apunte**; la presentación se ajusta sola.
- Texto de la figura en español, con unidades en los ejes.
- **Paleta por defecto** (no es obligación, es el punto de partida): datos `#1f77b4`,
  modelo/resultado `#d62728`, elementos secundarios `#555555`, óptimo o correcto `#2ca02c`
  / `#1c7a3e`. Lo que sí es obligatorio: que el mismo objeto tenga el mismo color dentro
  de un capítulo.
- Las figuras conceptuales (esquemas, mapas de unidad) pueden generarse con el código que
  sea; lo que importa es que el PNG quede versionado en `images/`.

---

## 8. Animaciones interactivas

Referencia canónica: `02-Algebra_lineal/condicionamiento_interactivo.html`.
Ejemplos vivos: `03-Ajuste_de_curvas/interactive/` (A1 cazador de rectas,
A2 descenso de gradiente, A3 sobreajuste).

**Qué justifica una animación:** que transmita algo que el texto estático no puede —
buscar a mano un óptimo, ver una dinámica iterativa, comparar ajuste contra generalización,
ver cómo se deforma una solución mal condicionada. **Dos o tres por capítulo, no más.**

### 8.1 Contrato técnico (obligatorio)

Estos cuatro puntos son los que hacen que la animación funcione en el notebook, en el libro
web y en el proyector. No son negociables:

1. **Fragmento HTML, no documento, y nunca un iframe.** Solo `<meta charset="utf-8">`,
   `<style>`, un `<div id="xxx-app">` y `<script>`. Un iframe impide que la animación se
   adapte al ancho de la diapositiva.
2. **CSS y JS completamente scopeados.** Todas las reglas prefijadas con `#xxx-app`, las
   clases con `xxx-`, y el JS dentro de un IIFE `(function(){ ... })()` con todos los `id`
   prefijados. Prefijo de 3 letras único por animación (`cnd`, `cdr`, `dgr`, `sbr`). Sin
   esto, dos animaciones en el mismo notebook se pisan.
3. **Gráficos en SVG con `viewBox`, nunca canvas.**
   ```css
   svg{width:100%; height:auto; display:block; background:#fff;
       border:1px solid var(--linea); border-radius:0.55em}
   ```
4. **Tipografía adaptativa.** Un `ResizeObserver` sobre el div raíz fija
   `base = max(13, min(17, ancho/50))` px, y todo lo interno se mide en `em`.
   Guardar el valor anterior para no entrar en bucle con el observer.
   Para el texto dentro del SVG: `esc = W_viewBox / anchoRealSVG`, luego `fT = 13*esc`
   (rótulos) y `fE = 15*esc` (nombres de eje), con márgenes `ml = max(52, 3.6*fT)` y
   `mb = max(42, 3.2*fT)` — con menos, la etiqueta del eje x se pisa con los números.

### 8.2 Convenciones visuales (recomendadas)

Existen para que las animaciones de distintos capítulos se vean como una familia. Se pueden
ampliar si el contenido lo pide.

- **Raíz:** `width:100%; max-width:min(800px, 94vw); padding:4px 2px; background:transparent;
  font-family:"Segoe UI",Calibri,system-ui,sans-serif; font-size:15px; line-height:1.45`
- **Layout de dos columnas** que colapsa en pantallas angostas:
  ```css
  .wrap{display:flex; gap:1.05em; align-items:flex-start; flex-wrap:wrap}
  .izq {flex:3 1 264px; min-width:0; max-width:448px}   /* gráfico */
  .der {flex:2 1 250px; min-width:0; font-size:0.85em}  /* panel   */
  ```
- **Paleta:**
  ```
  --tinta:#1a2733  --suave:#5d6b78  --linea:#d8dee4  --panel:#f6f8fa
  --c1:#1f77b4     --c2:#ff7f0e     --rojo:#d62728   --acento:#00afd8
  --verde:#1c7a3e
  ```
- **Componentes:** `.ctrl` (deslizador con `label`, `.val` en color acento y `.hint`),
  `.eqbox` (fórmula, serif "Cambria Math"), `table.t` (lecturas numéricas, columna `.num`
  a la derecha con `tabular-nums`), `.badge` (píldora de estado: verde `#e3f4e8`/`#1c7a3e`,
  ámbar `#fdf3dd`/`#8a6100`, rojo `#fbe4e4`/`#a11`), `.btns` (grid de botones),
  `.ley` (leyenda bajo el gráfico).
- **Subíndices:** `<sub>` en HTML, `<tspan baseline-shift="sub">` en SVG. **No usar** los
  caracteres Unicode `ᵣ`/`ᵢ` en texto corrido: varias fuentes los dibujan como coma.

### 8.3 Embebido

Celda de código con `slide_type: -` (para que quede pegada bajo su enunciado) y tag
`remove-input`:

```python
from pathlib import Path
from IPython.display import HTML
HTML(Path('interactive/A1_cazador_de_rectas.html').read_text(encoding='utf-8'))
```

Los dos `import` van en la celda de librerías del capítulo; en las animaciones siguientes
basta la línea `HTML(...)`. Un archivo suelto junto al notebook si es una sola animación;
subcarpeta `interactive/` con nombres `A1_`, `A2_`, `A3_` si son varias.

### 8.4 El enunciado importa tanto como la animación

Una animación sin instrucción no se usa. La celda inmediatamente anterior debe decir
**qué mover y qué observar**, en primera persona plural (§9):

> En la siguiente animación, movamos $a_0$ y $a_1$ para hacer el error $S_r$ lo más pequeño
> posible.

Y la celda inmediatamente posterior debe cerrar con la conclusión a la que se quería llegar.

---

## 9. Lenguaje

- Español de Chile, registro de cátedra, en **primera persona plural**: "analicemos el
  siguiente ejemplo", "probemos cómo funciona este caso particular", "consideremos",
  "derivamos", "graficamos", "notemos que". Es la voz del material — el profesor y el curso
  haciendo el desarrollo juntos — y aplica también cuando se le pide algo al estudiante:
  "movamos $a_0$", no "mueve $a_0$".
- Evitar el tuteo ("mueve", "pruébalo") y también el impersonal frío ("se procede a
  calcular"). Si una frase queda forzada en plural, reescribirla en vez de cambiar de voz.
- **"no lineal"**, sin guion. Igual "no lineales".
- **Python**, **NumPy**, **SciPy**, **Matplotlib** con mayúscula cuando se nombra la
  herramienta; en `código` cuando se nombra el módulo o la función (`numpy.polyfit`).
- Anglicismos solo cuando no hay equivalente instalado (*outlier*, *machine learning*,
  *spline*, *output*); en cursiva la primera vez que aparecen en el capítulo.
- Términos técnicos aceptados: sobreajuste, multivariable, univariable, linealización,
  sobredeterminado, minimax, matriz de diseño, rigidez, condicionamiento.
- Negrita para el término que se está definiendo, cursiva para énfasis. No usar ambas.
- Revisar tildes antes de cerrar: `hunspell -d es_ES -i UTF-8 -l` sobre la prosa extraída
  atrapa casi todo.

---

## 10. Organización de archivos

```
material_catedra/
├── _config.yml, _toc.yml, myst.yml, intro.md, requirements.txt
├── GUIA_FORMATO.md                   ← esta guía
└── NN-Nombre_unidad/
    ├── NN-Nombre_unidad.ipynb        ← el capítulo (único archivo del TOC)
    ├── NN-Nombre_unidad.css          ← CSS de RISE del capítulo
    ├── images/                       ← PNG del capítulo (U1: imagenes/)
    ├── interactive/                  ← animaciones HTML, si hay varias
    └── backup/                       ← copias fechadas antes de una edición grande
```

- **Dos motores de construcción conviven.** `_config.yml` + `_toc.yml` son Jupyter Book 1;
  `myst.yml` es Jupyter Book 2 / MyST, donde la numeración de capítulos se restaura con
  `numbering: {headings: true, title: {enabled: true, offset: 0}}`. Un cambio que afecte la
  numeración o el TOC hay que reflejarlo en **ambos** archivos.
- **Un solo notebook por unidad en el TOC.** Los `*_old.ipynb`, `* - edit.ipynb`,
  `*_2025.ipynb` son historia y no se construyen (`only_build_toc_files: true`), pero
  conviene no acumularlos: lo que se quiere conservar va a `backup/`.
- El `.css` del capítulo controla el tamaño de fuente en RISE, el color de los encabezados
  (`rgb(0,175,216)`) y el escalado de imágenes y figuras descrito en la sección 7. Los
  capítulos rediseñados comparten el mismo CSS; al crear un capítulo nuevo, copiar el de
  `03-Ajuste_de_curvas/`.
- Nombres de imágenes en minúscula, con guion bajo, descriptivos y en el idioma que ya use
  el capítulo (`mapa_unidad3.png`, `error_surface.png`, `LU_schematic.png`).

---

## 11. Flujo de trabajo al editar

1. **Respaldar antes de tocar nada**: copia del notebook en `NN-Capitulo/backup/` con fecha.
2. **Partir siempre del archivo en disco**, no de una copia previa ni de un script
   generador: el notebook se edita a mano entre sesiones. Diffear antes de escribir.
3. **Escribir solo los archivos que realmente cambiaron** en esa edición.
4. Si se sobrescribe algo por error, `.ipynb_checkpoints/NN-Capitulo-checkpoint.ipynb`
   guarda la última versión guardada en Jupyter y suele ser recuperable.
5. **Versionar**: rama por capítulo (`mejora-unidadN`) antes de aplicar cambios grandes.
   Los comandos `git` se corren manualmente desde una terminal en Windows; ejecutarlos a
   través de herramientas remotas sobre la carpeta de OneDrive se cuelga.
6. Los cambios grandes de una unidad se registran en `NN-Capitulo/CAMBIOS_unidadN.md`
   (referencia: `03-Ajuste_de_curvas/CAMBIOS_unidad3.md`).

---

## 12. Checklist de verificación antes de cerrar

**Ejecución**

- [ ] El notebook corre completo desde un kernel limpio, sin errores.
- [ ] Todas las imágenes referenciadas existen en `images/` (o `imagenes/` en U1).
- [ ] Las rutas de las animaciones existen y el `read_text(encoding='utf-8')` está presente.

**Presentación**

- [ ] Ninguna diapositiva sobrecargada (título + 2 a 4 fragmentos).
- [ ] Ninguna celda de código con más de ~15 líneas visibles.
- [ ] Las animaciones se ven bien a ~1180 px y a ~820 px de ancho.
- [ ] Las figuras se leen en el proyector con el escalado de RISE aplicado.

**Apunte**

- [ ] El capítulo se lee de corrido, sin depender de las pausas de la presentación.
- [ ] El LaTeX renderiza con KaTeX (sin `$$`, sin `\mbox`, sin `\eqnarray`).
- [ ] Corrector ortográfico pasado sobre la prosa.

**Encadenamiento** (lo que se olvida)

- [ ] La Introducción tiene un puente hacia atrás explícito.
- [ ] Hay entre 3 y 6 notas puente con la unidad nombrada en negrita.
- [ ] El capítulo cierra con tabla de decisión, mapa y puente hacia adelante.
- [ ] La notación nueva no choca con la del Anexo A, y lo nuevo quedó agregado ahí.

---

## 13. Enfoque pedagógico

Lo que hace que el material funcione, más allá del formato:

- **Un ejemplo conductor por capítulo**, con datos reales, que recorra toda la unidad
  (U2: los tres cuerpos unidos por elásticos; U3: el túnel de viento). El ejemplo se
  presenta en la Introducción y se retoma al cerrar, ya resuelto con la herramienta nueva.
- **Motivar antes de formalizar**: primero el problema o la exploración manual, después la
  derivación. Una animación donde el estudiante busca el óptimo a mano vale más que la
  derivación que viene después, porque le da un lugar donde poner la derivación.
- **Cerrar cada resultado con su interpretación**, en cita `>`. Un número sin lectura no
  enseña nada: "$r^2 = 0.88$" tiene que ir seguido de "el modelo lineal explica un 88% de
  la varianza de los datos".
- **Mostrar por qué el método existe, no solo cómo se aplica.** Por qué el error es
  *cuadrático* y no absoluto; por qué no se resuelve un sistema invirtiendo la matriz; por
  qué `lstsq` no arma $Z^TZ$ explícitamente. Es el contenido que sobrevive al examen.
- **Nombrar la reducción.** Cuando el método se apoya en uno de los tres problemas
  canónicos (§1.1), decirlo.
- **Anticipar el error del estudiante** con advertencias en cita `>`: qué se confunde con
  qué (residuo vs. error, linealización vs. reescritura algebraica, condición necesaria vs.
  suficiente).
- **Preferir cortar contenido redundante antes que agregar.** El material se usa en clase
  y el tiempo es el recurso escaso. Si algo entra, algo sale.

---

## 14. Bibliografía y formato de las referencias

Textos base del curso (`referencias/` en la carpeta del curso):

- Chapra S., Canale R. *Métodos Numéricos para Ingenieros*, 6ta Ed., McGraw Hill, 2011 —
  texto base en español.
- Chapra S. *Applied Numerical Methods with MATLAB for Engineers*, 3rd Ed., McGraw Hill.
- Press W. H., Teukolsky S. A., Vetterling W. T., Flannery B. P. *Numerical Recipes: The
  Art of Scientific Computing*, 3rd Ed., Cambridge University Press, 2007 — fundamento
  teórico y justificación estadística.
- Kong Q., Siauw T., Bayen A. M. *Python Programming and Numerical Methods — A Guide for
  Engineers and Scientists*, 1st Ed., Academic Press, 2021 — implementación en Python.
- Kiusalaas J. *Numerical Methods in Engineering with Python 3*, 3rd Ed., Cambridge
  University Press, 2013.

**Formato de la sección `## Referencias`:** una viñeta por texto, capítulo en negrita,
título del libro en cursiva, y —cuando ayude— una sub-viñeta indicando qué secciones
corresponden a qué parte del capítulo:

```markdown
- Press W., Teukolsky S., Vetterling W., Flannery B. **Chapter 2: Solution of Linear
  Algebraic Equations** in *Numerical Recipes: The Art of Scientific Computing*, 3rd Ed.,
  Cambridge University Press, 2007.
  - §2.4 sistemas tridiagonales y banda · §2.5 refinamiento iterativo · §2.11 por qué no invertir la matriz
```

Los enlaces a documentación en línea (NumPy, SciPy) van en el cuerpo del capítulo, no en
Referencias.

---

# Anexo A — Notación local por unidad

Símbolos fijados dentro de un capítulo. **Solo son válidos dentro de su unidad**, salvo
donde se indique. Al fijar un símbolo nuevo, agregarlo aquí.

### U1 · Aspectos generales

| símbolo | significado | alcance |
|---|---|---|
| $N$ | tamaño del input de un algoritmo | **global**: se reutiliza cada vez que se discute costo |
| $O(\cdot)$ | orden de complejidad | **global** |
| error absoluto / relativo | definidos aquí | **global** |

### U2 · Álgebra lineal

| símbolo | significado | alcance |
|---|---|---|
| $A$, $\mathbf{x}$, $\mathbf{y}$ | sistema $A\mathbf{x}=\mathbf{y}$ (el lado derecho son fuerzas medidas en el ejemplo conductor) | **global** para $A$; el lado derecho es $\mathbf{b}$ fuera de esta unidad |
| $n$ | dimensión del sistema | local |
| $\mathrm{Cond}(A)$ | número de condición | **global** |
| $K$ | matriz de rigidez del ejemplo de los elásticos | local |

### U3 · Ajuste de curvas

| símbolo | significado | alcance |
|---|---|---|
| $m$ | número de datos, índice $i=1,\dots,m$ | **global** para datos tabulados |
| $n$ | grado del modelo / última función base, índice $k=0,\dots,n$ ($n+1$ coeficientes) | local |
| $S_r$ | suma de cuadrados de los residuos (**nunca solo $S$**) | local |
| $S_t$ | suma total de cuadrados | local |
| $r^2$ | coeficiente de determinación | local |
| $Z$ | matriz de diseño, $Z_{ik} = \phi_k(\mathbf{x}_i)$ | local |
| $\mathbf{a}$ | vector de coeficientes del modelo | **global** para coeficientes de un ajuste |
| $\alpha$, $\beta$ | coeficientes del modelo de potencia $y=\alpha x^\beta$ | local |
| $\eta$ | tasa de aprendizaje del descenso de gradiente (se usa $\eta$ y no $\alpha$ porque $\alpha$ ya está tomado) | local |
| $\nabla S_r$ | gradiente de la función de pérdida | local, con puente a U6 |

**Precisiones que hay que respetar en U3:** el vector de derivadas de un escalar es el
*gradiente*; el Jacobiano es el de la función residuo ($\nabla S_r = -2J^T\mathbf{e}$) y el
Jacobiano de $\nabla S_r$ es el Hessiano. $\nabla S_r=\mathbf{0}$ es condición **necesaria
pero no suficiente**. El descenso de gradiente es iteración de punto fijo sobre esa
ecuación; Gauss-Newton y Levenberg-Marquardt son Newton sobre la misma.

### U4 · Interpolación

| símbolo | significado | alcance |
|---|---|---|
| $n$ | grado del polinomio de interpolación; hay $n+1$ nodos, índice $i=0,\dots,n$ | local (coherente con U3, donde $n$ es el grado del modelo) |
| $x_i$, $y_i$ | nodos de interpolación — **dato exacto**, no medición con ruido | **global** desde acá |
| $p_n(x)$ | polinomio de interpolación de grado $n$ | local |
| $L_i(x)$ | base de Lagrange, $L_i(x_j)=\delta_{ij}$ | local, con puente a U7 y U8 |
| $V$ | matriz de Vandermonde | local, con puente a U2 |
| $S_i(x)$ | tramo $i$ del *spline*, definido en $[x_i, x_{i+1}]$ | local |
| $h_i = x_{i+1}-x_i$ | ancho del tramo $i$ | **global** (mismo $h$ del paso en U7-U11) |

**Precisión que hay que respetar en U4:** en la interpolación el grado y el número de datos
están amarrados ($m = n+1$); eso es justamente lo que la distingue del ajuste de la U3, y
conviene decirlo explícitamente.

### U5 a U11 · pendiente de fijar

Al rediseñar cada unidad, completar aquí su notación. Símbolos que ya circulan y conviene
consolidar: $h$ y $\Delta t$ (paso), $x_i$ (nodos de la malla), $T$ (temperatura, ejemplo
conductor recurrente de las unidades de EDO y EDP), $J$ (Jacobiano, U6), $\mathbf{k}_i$
(etapas de Runge-Kutta, U9).

---

# Anexo B — Plantilla de capítulo nuevo

Copiar esta estructura de celdas al empezar una unidad:

| # | tipo | slide_type | tags | contenido |
|---|---|---|---|---|
| 0 | markdown | `slide` | | `# Título de la unidad` |
| 1 | code | `skip` | `remove-input` | `# Librerías utilizadas en este capítulo` + imports + `plt.rcParams` |
| 2 | markdown | `slide` | | `## Introducción` — el problema de ingeniería |
| 3 | markdown | `fragment` | | figura o datos del problema conductor |
| 4 | markdown | `fragment` | | **puente hacia atrás**: qué de la unidad anterior no alcanza |
| 5 | markdown | `slide` | | qué vamos a poder hacer al final del capítulo |
| … | | | | desarrollo por familias de métodos |
| n−4 | markdown | `slide` | | `## Resumen: ¿qué método uso?` |
| n−3 | markdown | `fragment` | | tabla situación → método → costo → función de Python |
| n−2 | markdown | `slide` | | `<img src="./images/mapa_unidadN.png" width="860" align= center>` |
| n−1 | markdown | `fragment` | | **puente hacia adelante** en cita `>` |
| n | markdown | `slide` | | `## Referencias` |

---

# Anexo C — Estado del rediseño 2026

| Unidad | Estado | Tiene animación | Tiene `## Resumen` | Tiene mapa |
|---|---|---|---|---|
| 01 Aspectos generales | rediseñada | no | no | no |
| 02 Álgebra lineal | rediseñada | sí (condicionamiento) | sí | no |
| 03 Ajuste de curvas | rediseñada | sí (A1, A2, A3) | sí (¿qué herramienta uso?) | sí |
| 04 Interpolación | rediseñada (2026-08-19) | sí (A1, A2, A3) | sí | sí |
| 05 Series de Taylor | pendiente | — | — | — |
| 06 Búsqueda de raíces | pendiente | — | — | — |
| 07 Integración | pendiente | — | — | — |
| 08 Derivación numérica | pendiente | — | — | — |
| 09 EDO valor inicial | pendiente | — | — | — |
| 10 EDO valor de frontera | pendiente | — | — | — |
| 11 Diferencias finitas EDP | pendiente | — | — | — |

**Deudas conocidas al cerrar esta versión de la guía:**

- U1 y U2 no tienen mapa de unidad; U1 no tiene sección de resumen.
- Quedan celdas antiguas con `%`-formatting en etiquetas de gráficos (U3, celda de la
  primera figura del túnel de viento). Corregir al pasar por ahí, sin abrir una campaña.
- U1 usa `imagenes/` mientras el resto usa `images/`. No unificar por ahora: rompería
  rutas por una ganancia cosmética.
- Quedan enunciados en tuteo de antes de fijar la voz en primera persona plural (§9); el
  más visible es el de la animación A1 en U3 ("mueve $a_0$ y $a_1$"). Corregir al pasar por
  cada capítulo, sin abrir una campaña.
- Los puentes hacia adelante que faltan escribir, en orden de prioridad:
  U5 → U6 (Newton-Raphson como Taylor truncado), U2 → U10 y U11 (la matriz tridiagonal que
  vuelve), U6 → U9 (Euler implícito como ecuación no lineal por paso).
  *(U4 → U7 y U8 quedó escrito el 2026-08-19.)*
- La U4 conserva su propio `04-Interpolacion.css`, más liviano que el compartido por U2 y U3.
  Unificarlo cambiaría el escalado de las figuras ya calibradas para el proyector: decisión
  pendiente del usuario.
