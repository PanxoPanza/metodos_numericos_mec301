# Plan de mejora — Unidad 4 · Métodos de Interpolación

**Estado actual:** `04-Interpolacion/04-Interpolacion.ipynb`, 75 celdas.
**Estado propuesto:** ~101 celdas (dentro del rango 90–130 de la §2 de `GUIA_FORMATO.md`).
**Balance:** +39 celdas nuevas, −13 por consolidación de secciones redundantes.

Todo lo que sigue se apoya en la guía de formato 2026 y en los dos textos guía:
Chapra, *Applied Numerical Methods with MATLAB* 3rd Ed., **cap. 17 y 18**; Press et al.,
*Numerical Recipes* 3rd Ed., **cap. 3** (§3.0, §3.3, §3.5) y §5.8.

---

## 0. Respaldo y control de versiones (antes de tocar nada)

1. Crear `04-Interpolacion/backup/04-Interpolacion_2026-08-19.ipynb` — copia byte a byte
   del archivo en disco, **antes** de cualquier edición.
2. Verificar que el respaldo abre correctamente (conteo de celdas idéntico: 75).
3. Rama de trabajo, para correr manualmente en la terminal de Windows (los comandos `git`
   sobre OneDrive se cuelgan desde acá):

   ```
   cd "...\material_catedra"
   git checkout -b mejora-unidad4
   git add 04-Interpolacion/backup/
   git commit -m "backup unidad 4 antes del rediseño"
   ```
4. Al cerrar, registrar los cambios en `04-Interpolacion/CAMBIOS_unidad4.md`.
5. Toda edición parte del archivo en disco y se aplica celda por celda (nunca regenerando
   el notebook desde un script).

**Punto de retorno:** en cualquier momento, `backup/04-Interpolacion_2026-08-19.ipynb`
restaura la versión de hoy; `.ipynb_checkpoints/04-Interpolacion-checkpoint.ipynb` es la
red secundaria.

---

## 1. Diagnóstico

### 1.1 Lo que está bien y se conserva

- El ejemplo conductor de la introducción (tabla de propiedades vs. temperatura) es el
  correcto y engancha con U3.
- La secuencia polinomial → Runge → por tramos → spline es la secuencia canónica de los
  dos textos guía.
- El video de condiciones de borde (`_static/videos/BoundaryConditions.mp4`, de Matías
  Rojas) es material propio y de buena calidad: se mantiene tal cual.
- El tratamiento de `numpy.interp` y `scipy.interpolate.CubicSpline` cubre lo que el
  estudiante usará en la práctica.

### 1.2 Errores de contenido detectados (corrección obligatoria)

| # | Celda | Problema | Corrección |
|---|---|---|---|
| E1 | 2 | La tabla dice "propiedades **del agua**". La imagen es la Tabla 17.1 de Chapra: **aire** a 1 atm ($\rho = 1{,}29\ \mathrm{kg/m^3}$ a 0 °C). | Cambiar a "aire" y citar la fuente |
| E2 | 14 | "un sistema de $n$ ecuaciones lineales con $n$ incógnitas" | Son $n+1$ ecuaciones y $n+1$ incógnitas |
| E3 | 16 vs. 17 | El texto dice interpolar $\{x_0,x_2,x_4\}$; el código usa `xi[0], xi[1], xi[4]` | Alinear texto y código |
| E4 | 19 | Afirma que la curva azul pasa por $(0{,}5;\,2{,}0)$, pero por el error E3 pasa por $(-2;\,-0{,}1)$ | Se resuelve al corregir E3 |
| E5 | 23 vs. 24–26 | El texto anuncia $p_5$ y $p_7$; el código genera grados **4 y 6** (5 y 7 *nodos*) | Redactar en términos de nodos y grado explícito |
| E6 | 21 | "Runge demostró el **mal condicionamiento de los polinomios**" | Lo mal condicionado es la interpolación de alto orden **sobre nodos equiespaciados**, no el polinomio (NR §3.0) |
| E7 | 45 vs. 48 | $S_i$ se define en $[x_i, x_{i+1}]$ y dos celdas después se evalúa en $x_{i-1}$ con $i=1,\dots,n$ | Fijar una sola convención de índices en todo el capítulo |
| E8 | 48, 49 | Ecuaciones con `$$` y `&=&` estilo `eqnarray` | Migrar a `align*` (§5.1 de la guía: KaTeX no admite `eqnarray`) |
| E9 | 63 | "condiciones de borde `not-a-knot`, `clamped` y **`spline`**" | Es `natural` |
| E10 | varias | Tildes y tipeo: *incognitas*, *contruir*, *domino*, *periodicas*, *condicion*, *quién* → *quien* | Pasada de corrector |

### 1.3 Explicaciones inconclusas (el corazón del encargo)

| # | Dónde | Qué queda colgando | Cómo se cierra |
|---|---|---|---|
| I1 | Celda 10 | "formar $n+1$ ecuaciones… **sin embargo, este método no es eficiente**". Nunca se dice por qué. | Mostrar la **matriz de Vandermonde** y calcular su $\mathrm{Cond}(V)$ → puente directo a **Unidad 2** (Chapra §17.1.1, NR ec. 3.5.2) |
| I2 | Celda 11 | Newton y Lagrange aparecen solo como dos enlaces web. | Una sección breve: *son el mismo polinomio, escrito distinto*, y por qué eso importa (§3 de este plan) |
| I3 | Celda 14 | Se afirma que `polyfit` "estrictamente no es interpolación" sin cerrar la idea. | NR §3.5 lo dice con precisión: ajustar suaviza (menos coeficientes que datos), interpolar toma los datos como exactos (igual número). Es el mismo comando con dos propósitos |
| I4 | Celdas 20–27 | Se muestra que el polinomio oscila, pero no se dice **qué hacer al respecto**. El capítulo salta a "por tramos" sin explicar por qué eso lo arregla. | Cerrar con la regla práctica de NR (3–4 puntos sí, 5–6 tal vez, más casi nunca) y con la causa real: los **nodos**, no el grado (animación A1) |
| I5 | Celdas 47–54 | Se cuentan $4n$ incógnitas y $4n$ ecuaciones… y ahí termina. Nunca se dice cómo se resuelve el sistema ni cuánto cuesta. | **El remate que falta:** eliminando incógnitas, todo se reduce a un sistema **tridiagonal** en las segundas derivadas $y''_i$, que se resuelve en $O(n)$ con el algoritmo de Thomas (NR §3.3, ec. 3.3.7) → puente a **Unidad 2** |
| I6 | Celda 45 | Nunca se explica **por qué cúbico** y no cuadrático o de grado 5. | Chapra §18.3: para continuidad de la derivada $n$-ésima se necesita orden $n+1$; el ojo detecta discontinuidades de curvatura, la tercera derivada no |
| I7 | Celdas 67–70 | Las derivadas del spline se presentan como una *feature* de SciPy. | Es la base de la **Unidad 8**: derivar el interpolante *es* la diferencia finita |
| I8 | Celdas 42–44, 71–73 | La extrapolación aparece dos veces, como detalle de API, sin advertencia conceptual. | Una sola sección con la advertencia de NR §3.0 y Chapra §17.5.1 |
| I9 | Celda 3 | Se plantea "¿cuánto vale $\rho$ a 10 °C?" y **nunca se responde**. | Cerrar el capítulo con ese número (§13 de la guía: el ejemplo conductor se retoma resuelto) |
| I10 | Todo el capítulo | No hay `## Resumen`, ni mapa de unidad, ni un solo puente hacia adelante. | §5 de este plan |

### 1.4 Deudas de formato

- **No existe celda de librerías** (`skip` + `remove-input`). Los `import` y
  `plt.rcParams.update({'font.size': 10})` se repiten en 7 celdas de código.
- `figsize` va entre `(6,5)` y `(4,3)`. **Se mantiene tal cual en las figuras existentes**
  (están calibradas para el proyector y no vale la pena tocarlas); el estándar `(5,4)` o
  `(5,3)` de la guía aplica solo a figuras nuevas.
- El capítulo abre con `#` y salta a `##`, pero no tiene ninguna de las secciones
  obligatorias salvo `## Referencias`.
- Dos imágenes existen en `images/` y **no se usan**: `newton_recursion.png` (diagrama de
  diferencias divididas), que pasa a la diapositiva de Newton y Lagrange, y
  `2dinterpolation.png`, que por decisión D3 se queda sin usar.
- Las referencias citan Chapra 6ta Ed. y Kong, pero no *Numerical Recipes* ni Chapra
  *Applied Numerical Methods*, que son los textos que el curso declara como guía.

---

## 2. Estructura propuesta

Cada línea es una diapositiva o un bloque de fragmentos. **N** = celda nueva,
**M** = celda modificada, **=** = celda que se conserva, **✂** = celda que se elimina o
se funde.

```
#  Métodos de interpolación                                          [slide]  =
   celda de librerías (numpy, matplotlib, scipy, IPython.display,
   pathlib, plt.rcParams)                                            [skip]   N

## Introducción                                                      [slide]  =
   tabla de propiedades del aire (corregida)                                  M
   ¿cuánto vale rho a 10 °C? el valor no está en la tabla                     =
   puente atrás → Unidad 3: allá el dato tenía ruido y buscábamos la
   tendencia; acá el dato es exacto y la curva debe pasar por él               M
   qué podremos hacer al final del capítulo                          [slide]  N

## Interpolación polinomial                                          [slide]  =
   existe un único polinomio de grado n por n+1 puntos                        =
### La forma directa: el sistema de Vandermonde                      [slide]  N
   plantear el sistema V a = y para 3 puntos de la tabla                      N
   código: armar V con np.vander y resolver                                   N
   > Cond(V) = 5,9e6 para 3 puntos; 1e27 para los 11 de la tabla              N
   > nota puente **Unidad 2**: es el mismo Cond(A) del capítulo anterior      N
### Newton y Lagrange: el mismo polinomio, escrito distinto          [slide]  M
   base de Lagrange: cada L_i vale 1 en su nodo y 0 en los demás              N
   diferencias divididas de Newton — images/newton_recursion.png              N
   > nota puente **Unidades 7 y 8**: integrar y derivar estas bases           N
### `numpy.polyfit`: ajustar e interpolar con el mismo comando       [slide]  M
   ejemplo corregido (E3/E4), reanclado en la tabla de aire                   M
   > ajustar suaviza, interpolar toma el dato como exacto (NR §3.5)           N
### El fenómeno de Runge                                             [slide]  M
   la función de Runge y el enunciado correcto (E6)                           M
   figura grado 4 vs grado 6 (corregida, figsize sin cambios)                 M
   ▶ ANIMACIÓN A1 — nodos y oscilación                                        N
   > la culpa es de los nodos equiespaciados, no del grado                    N
   > regla práctica de NR: 3–4 puntos sí, 5–6 tal vez, más casi nunca         N
   (opcional) el término de error de interpolación → puente **Unidad 5**      N

## Interpolación por tramos                                          [slide]  =
   figura piecewise_interpolation.png                                         =
### Interpolación lineal (`numpy.interp`)                            [slide]  =
   fórmula de la recta por tramo                                              =
   ejemplo corto + gráfico                                           [3 celdas] ✂ (de 12)
   > la derivada salta en cada nodo: es continua, pero no suave               N
### ¿Por qué cúbico?                                                 [slide]  N
   para continuidad de la derivada n-ésima hace falta orden n+1 (Chapra §18.3) N
   > la curvatura discontinua se ve; la tercera derivada no                   N
### Spline cúbico: contar ecuaciones                                 [slide]  M
   4n incógnitas (índices unificados, E7)                                     M
   2n condiciones de paso por los nodos (align*, E8)                          M
   2(n−1) condiciones de suavidad                                             M
   2 condiciones de borde                                                     =
### El remate: el sistema es tridiagonal                             [slide]  N
   eliminando, quedan n−1 ecuaciones en las y''_i (NR ec. 3.3.7)              N
   > nota puente **Unidad 2**: Thomas resuelve esto en O(n)                   N
   > por eso el spline escala a tablas de miles de puntos                     N
### Condiciones de borde                                             [slide]  M
   las cuatro alternativas en una tabla (de 4 celdas a 1)                     M ✂
   video BoundaryConditions.mp4                                               =
   `bc_type` en CubicSpline (de 6 celdas a 2)                                 M ✂
   ▶ ANIMACIÓN A2 — global vs. lineal vs. spline, con derivadas               N
### Derivadas del interpolante                                       [slide]  =
   figura de y, y', y'', y'''                                                 M
   > solo hasta la tercera derivada                                           =
   > nota puente **Unidad 8**: derivar el interpolante es la diferencia finita N
### Extrapolación: por qué no                                        [slide]  M ✂
   `numpy.interp` congela, `CubicSpline` extrapola, `extrapolate=False`       M
   > advertencia de NR §3.0 y Chapra §17.5.1                                  N

## Interpolación en Python                                           [slide]  N
   tabla comparativa: np.interp vs. CubicSpline (cuándo usar cada una)        N
### Volvamos a la pregunta: rho del aire a 10 °C                     [slide]  N
   ▶ ANIMACIÓN A3 — la tabla real, los tres métodos y el valor exacto         N
   > rho(10 °C) ≈ 1,245 kg/m³ contra 1,2469 del gas ideal: 0,15 % de error    N

## Resumen: ¿qué método uso?                                         [slide]  N
   tabla situación → método → costo → función de Python                       N
   images/mapa_unidad4.png                                                    N
   verificaciones que hay que hacer siempre                                   N
   > puentes hacia adelante: **Unidades 5, 7, 8 y 10**                        N

## Referencias                                                       [slide]  M
```

---

## 3. Decisiones de alcance

Los tres puntos de alcance quedaron resueltos. Los dejo registrados con su justificación,
porque condicionan el conteo de celdas y el trabajo de las secciones 2 y 6.

**D1 · ¿Cuánta profundidad para Newton y Lagrange? — RESUELTO: opción (a), solo la idea.**

Una sola diapositiva, sin desarrollo formal ni ejemplo a mano:

- la base de Lagrange en una línea: cada $L_i(x)$ vale 1 en su nodo y 0 en todos los demás,
  así que $p_n(x) = \sum_i y_i L_i(x)$ pasa por los datos por construcción;
- las diferencias divididas de Newton mencionadas como la forma recursiva, apoyadas en la
  figura `newton_recursion.png` que hoy está sin usar;
- el punto que importa: **es el mismo polinomio único de la sección anterior, escrito en
  otra base**, y se prefieren esas formas porque evitan armar y resolver el sistema de
  Vandermonde;
- nota puente a las **Unidades 7 y 8**.

Total: 4 celdas. La razón: el curso usa librerías, y la base de Lagrange se necesita de
verdad recién en la **Unidad 7** (Newton–Cotes es integrar el interpolante de Lagrange).
Acá basta plantar la semilla.

**D2 · ¿Entran los nodos de Chebyshev? — RESUELTO: opción (a), como interruptor en A1.**

Aparecen únicamente como una casilla dentro de la animación A1 (equiespaciados ↔ Chebyshev),
sin fórmula ni desarrollo teórico en el cuerpo del capítulo. El único texto que se agrega es
la conclusión de la celda de cierre de la animación.

La razón: es la respuesta honesta a "¿y entonces qué hago?" y evita que el
estudiante se lleve la idea falsa de que "polinomio de grado alto = malo". Los números son
elocuentes: con 15 nodos sobre la función de Runge el error máximo es **7,2** con nodos
equiespaciados y **0,047** con Chebyshev. Y el cierre pedagógico es el que necesitamos:
*en una tabla de ingeniería los nodos vienen dados, no se eligen* → por eso existe el
spline (NR §3.0 y §5.8).

**D3 · ¿Interpolación en 2D? — RESUELTO: opción (c), no entra.**

La Unidad 4 se queda en una dimensión. No se agrega diapositiva conceptual ni
`RegularGridInterpolator`, y `2dinterpolation.png` sigue sin usarse en el capítulo (se
mantiene en `images/` por si más adelante hace falta). El puente hacia la **Unidad 11**
queda cubierto por la matriz tridiagonal del spline, que es el objeto que efectivamente
viaja hacia allá.

Con esto el capítulo baja a **~101 celdas**, más holgado para el tiempo de clase.

---

## 4. Cortes aprobados

**Los cinco se aplican.** Todos son consolidaciones, no pérdidas de contenido: suman
**−13 celdas**, que es lo que financia las secciones nuevas.

| # | Celdas | Qué es hoy | Qué propongo | Δ |
|---|---|---|---|---|
| C1 | 33–41 | Envolver `np.interp` en un `lambda` y evaluarlo en `0.3`, luego en `0.5`, luego en un arreglo, cada cosa en su celda; después un gráfico | Una celda con la función `lambda` y su evaluación vectorizada, y una con el gráfico | −5 |
| C2 | 42–44 y 71–73 | La extrapolación se trata dos veces, separada por 25 celdas: una vez para `np.interp` y otra para `CubicSpline` | Una sola subsección "Extrapolación: por qué no", con los dos comportamientos contrastados en un gráfico | −3 |
| C3 | 51–54 | Las cuatro condiciones de borde, una por celda | Una tabla de cuatro filas: nombre · condición · cuándo usarla. Sigue siendo *una idea* | −3 |
| C4 | 57–62 | Seis celdas de sintaxis de `bc_type` (equivalencias `(1,0.0)`↔`clamped`, etc.) | Dos celdas: un bloque de código comentado y una nota con el valor por defecto | −4 |
| C5 | 15–19 | Ejemplo de `polyfit` con cinco puntos inventados, desconectado del ejemplo conductor | El mismo ejemplo, pero sobre tres filas reales de la tabla de aire (y con los índices corregidos) | 0 |

**C5 no es un corte sino un reanclaje**: los cinco puntos inventados se reemplazan por tres
filas reales de la tabla de aire. Los datos originales quedan en el respaldo.

Lo que **no** se corta: el video de condiciones de borde (es material propio y funciona) y
la figura de derivadas del spline (celda 69), que pasa a ser el puente a la Unidad 8.

---

## 5. Encadenamiento: los puentes que se agregan

La Unidad 4 es, según el mapa de la guía (§1.2), una de las que más alimenta al resto del
curso, y hoy no tiene ni un solo puente escrito.

**Hacia atrás (Introducción):** la Unidad 3 cerró distinguiendo ajuste de interpolación
(celdas 7–8 de U3, figura `interpolacion_1D.png`). La Unidad 4 abre retomando esa figura:
*allá el dato traía ruido y buscábamos la tendencia; acá el dato es exacto y la curva tiene
que pasar por él*. Con eso la unidad deja de abrir en el vacío.

**Notas puente dentro del capítulo (5, dentro del rango 3–6 de la guía):**

1. **Unidad 2** — la matriz de Vandermonde y su $\mathrm{Cond}(V)$. Es el reencuentro
   literal con el número de condición de U2, y explica el "no es eficiente" que hoy queda
   colgando.
2. **Unidad 2** — el sistema del spline es **tridiagonal**: Thomas lo resuelve en $O(n)$.
   El mismo objeto que reaparecerá en U10 y U11.
3. **Unidades 7 y 8** — la base de Lagrange: integrarla da trapecio y Simpson, derivarla da
   las diferencias finitas.
4. **Unidad 8** — las derivadas de `CubicSpline`: derivar el interpolante *es* derivar
   numéricamente.
5. **Unidad 5** — el término de error $f^{(n+1)}(\xi)\prod(x-x_i)/(n+1)!$ es un resto de
   Taylor, y explica de dónde salen los órdenes de error de U7 y U8. *(Opcional: una sola
   diapositiva; se puede dejar fuera si el capítulo queda largo.)*

**Cierre (mapa):** `images/mapa_unidad4.png`, generado con Matplotlib y versionado, con la
estructura: *datos exactos* → un polinomio global (Vandermonde / Newton / Lagrange) →
oscilación → por tramos (lineal / cúbico) → tridiagonal → hacia U7, U8, U10.

---

## 6. Las tres animaciones

Cumplen el contrato técnico de la §8.1 de la guía: fragmento HTML sin iframe, CSS y JS
scopeados con prefijo de tres letras, SVG con `viewBox`, tipografía adaptativa con
`ResizeObserver`. Van en `04-Interpolacion/interactive/`.

### A1 · `A1_nodos_y_oscilacion.html` — prefijo `rng`

**Qué transmite:** que "más puntos" no es "mejor", y que el culpable son los nodos.

- Deslizador: número de nodos, de 3 a 17.
- Selector de función: Runge $1/(1+25x^2)$ · una función suave · una con esquina ($|x|$).
- Interruptor: nodos equiespaciados ↔ nodos de Chebyshev.
- Panel de lectura: error máximo $\max|f - p_n|$ y $\mathrm{Cond}(V)$, con píldora de
  estado verde/ámbar/roja.

**Enunciado (celda anterior):** «Subamos el número de nodos y observemos qué le pasa al
error en el centro y qué le pasa en los extremos.»
**Cierre (celda posterior):** con nodos equiespaciados el error crece sin control al subir
el grado; con nodos bien elegidos, no. En una tabla de ingeniería los nodos vienen dados.

### A2 · `A2_global_vs_tramos.html` — prefijo `trm`

**Qué transmite:** por qué el spline es el compromiso correcto entre pasar por los puntos y
no inventar oscilaciones. Es la Figura 18.1 de Chapra, pero manipulable.

- Datos con un cambio brusco (el caso donde el polinomio global se descompone).
- Tres botones: polinomio global · lineal por tramos · spline cúbico (se pueden superponer).
- Casilla "mostrar derivadas": panel inferior con $y'(x)$ y $y''(x)$ — se ve el escalón del
  lineal y la continuidad del cúbico. Es la justificación visual de "¿por qué cúbico?".
- Deslizador: número de datos.

**Cierre:** el spline compra suavidad sin comprar oscilación, y lo hace resolviendo un
sistema tridiagonal.

### A3 · `A3_tabla_de_aire.html` — prefijo `air`

**Qué transmite:** cierra el ejemplo conductor y la advertencia sobre extrapolación, en un
solo objeto.

- La tabla real de Chapra ($T$ de −40 a 500 °C, $\rho$ del aire), dibujada como nodos.
- Deslizador de temperatura de −60 a 560 °C, con las zonas de extrapolación sombreadas.
- Lecturas simultáneas: interpolación lineal, spline cúbico y polinomio global de grado 10.
- **Valor de referencia:** la ley de gases ideales $\rho = p/(RT)$ reproduce la tabla
  completa con tres cifras significativas (verificado: 1,293 vs. 1,29 a 0 °C; 0,946 vs.
  0,946 a 100 °C). Eso nos da un error verdadero, no estimado.
- Al salir del rango tabulado los tres métodos divergen, y el polinomio global lo hace de
  forma espectacular.

**Cierre:** $\rho(10\,^\circ\mathrm{C}) \approx 1{,}245\ \mathrm{kg/m^3}$ por interpolación
lineal, contra $1{,}2469$ del gas ideal — un 0,15 % de error, con dos filas de una tabla.
Y del otro lado del borde, ninguno de los tres sirve.

---

## 7. Sección `## Resumen: ¿qué método uso?`

Tabla de decisión propuesta:

| Situación | Método | Costo | En Python |
|---|---|---|---|
| 2–4 puntos, valor puntual | polinomio de interpolación | $O(n^2)$ | `numpy.polyfit` + `polyval` |
| Tabla larga, se necesita un valor | lineal por tramos | $O(\log n)$ por consulta | `numpy.interp` |
| Tabla larga, se necesita una curva suave | spline cúbico | $O(n)$ al construir | `scipy.interpolate.CubicSpline` |
| Los datos tienen ruido | **no interpolar**: ajustar | — | **Unidad 3** |

Verificaciones que hay que hacer siempre (siguiendo el modelo de U2):

1. ¿Los datos tienen ruido? Si lo tienen, esto es un problema de ajuste, no de interpolación.
2. ¿El valor pedido está dentro del rango tabulado? Si no, es extrapolación.
3. ¿Los nodos están ordenados y sin repetir?
4. ¿El resultado es físicamente posible? (una densidad negativa entre dos nodos delata una
   oscilación).

---

## 8. Notación de la unidad (para el Anexo A de la guía)

| símbolo | significado | alcance |
|---|---|---|
| $n$ | grado del polinomio de interpolación; hay $n+1$ nodos, $i = 0,\dots,n$ | local (coherente con U3, donde $n$ es el grado del modelo) |
| $x_i$, $y_i$ | nodos de interpolación (dato exacto, no medición con ruido) | **global** desde acá |
| $p_n(x)$ | polinomio de interpolación de grado $n$ | local |
| $L_i(x)$ | base de Lagrange | local, con puente a U7 |
| $V$ | matriz de Vandermonde | local, con puente a U2 |
| $S_i(x)$ | tramo $i$ del spline, definido en $[x_i, x_{i+1}]$ | local |
| $h_i = x_{i+1} - x_i$ | ancho del tramo $i$ | **global** (mismo $h$ del paso en U7–U11) |

Un cuidado: en U3, $m$ es el número de datos y $n$ el grado. Acá el número de datos es
$n+1$ **porque grado y número de datos están amarrados** — eso es precisamente lo que
distingue interpolar de ajustar, y conviene decirlo explícitamente en vez de dejar que el
estudiante lo choque solo.

---

## 9. Orden de ejecución propuesto

1. Respaldo + rama (§0). **Bloqueante.**
2. Correcciones de contenido E1–E10 sobre el archivo en disco. Cambio pequeño, valor alto,
   independiente del resto — se puede aplicar y revisar por separado antes de seguir.
3. Celda de librerías: imports y `plt.rcParams` una sola vez. Los `figsize` de cada figura
   quedan como están.
4. Introducción reescrita con el puente a U3.
5. Vandermonde y `Cond(V)` (cierra I1) + la diapositiva de Newton y Lagrange (cierra I2).
6. Animación A1 + cierre de la sección de Runge (cierra I4).
7. Cortes C1–C4 en la sección de interpolación por tramos.
8. "¿Por qué cúbico?" + el remate tridiagonal (cierra I5 e I6).
9. Animación A2.
10. Sección "Interpolación en Python" + animación A3 (cierra I9).
11. `## Resumen`, `mapa_unidad4.png`, puentes hacia adelante, Referencias.
12. Verificación final (§10) y `CAMBIOS_unidad4.md`.

Los pasos 2 y 3 se pueden entregar como una primera tanda revisable, antes de comprometerse
con el rediseño completo.

---

## 10. Verificación antes de cerrar

- [ ] El notebook corre completo desde un kernel limpio.
- [ ] Las tres animaciones se ven bien a ~1180 px y a ~820 px, y no se pisan entre sí
      (prefijos `rng`, `trm`, `air` verificados como únicos en el archivo).
- [ ] Ninguna diapositiva con más de 4 fragmentos; ninguna celda de código sobre 15 líneas.
- [ ] Sin `$$`, sin `eqnarray`, sin `\mbox`: todo renderiza en KaTeX.
- [ ] `images/mapa_unidad4.png` existe y está versionado.
- [ ] Los números citados en el texto coinciden con lo que imprime el código
      (ya verificados: $\mathrm{Cond}(V) = 5{,}89\times10^6$ para 3 puntos y
      $9{,}9\times10^{26}$ para los 11 de la tabla; error de Runge 7,2 vs. 0,047 con 15
      nodos; $\rho(10\,^\circ\mathrm{C})$: 1,245 interpolado vs. 1,2469 exacto).
- [ ] Voz en primera persona plural en todo el capítulo, incluidos los enunciados de las
      tres animaciones.
- [ ] Corrector ortográfico sobre la prosa.
- [ ] Diff contra `backup/04-Interpolacion_2026-08-19.ipynb` revisado celda por celda.

---

## 11. Referencias actualizadas (sección `## Referencias`)

```markdown
- Chapra S. **Chapter 17: Polynomial Interpolation** y **Chapter 18: Splines and
  Piecewise Interpolation** en *Applied Numerical Methods with MATLAB for Engineers and
  Scientists*, 3rd Ed., McGraw Hill, 2012.
  - §17.1.1 la matriz de Vandermonde y su condicionamiento · §17.2 y §17.3 Newton y
    Lagrange · §17.5 extrapolación y oscilaciones · §18.1 por qué un spline · §18.4 spline
    cúbico

- Press W., Teukolsky S., Vetterling W., Flannery B. **Chapter 3: Interpolation and
  Extrapolation** en *Numerical Recipes: The Art of Scientific Computing*, 3rd Ed.,
  Cambridge University Press, 2007.
  - §3.0 orden, nodos equiespaciados y los peligros de extrapolar · §3.3 el spline cúbico
    como sistema tridiagonal · §3.5 interpolar no es ajustar

- Chapra S., Canale R. **Capítulo 18: Interpolación** en *Métodos Numéricos para
  Ingenieros*, 6ta Ed., McGraw Hill, 2011.

- Kong Q., Siauw T., Bayen A. M. **Chapter 17: Interpolation** en *Python Programming and
  Numerical Methods*, 1st Ed., Academic Press, 2021.
```
