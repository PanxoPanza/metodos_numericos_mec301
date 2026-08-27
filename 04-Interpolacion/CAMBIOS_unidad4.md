# Cambios — Unidad 4 · Métodos de Interpolación

**Fecha:** 2026-08-19
**Antes:** 75 celdas · **Después:** 119 celdas (rango cómodo de la guía: 90–130)
**Respaldo:** `backup/04-Interpolacion_2026-08-19.ipynb` (copia byte a byte del original)
**Plan que se ejecutó:** `PLAN_unidad4.md`

Todo se apoya en `GUIA_FORMATO.md` (versión 2026) y en los dos textos guía: Chapra,
*Applied Numerical Methods with MATLAB* 3rd Ed., cap. 17 y 18; Press et al.,
*Numerical Recipes* 3rd Ed., cap. 3.

---

## 1. Errores de contenido corregidos

| # | Qué decía | Qué dice ahora |
|---|---|---|
| E1 | "propiedades **del agua**" | La imagen es la Tabla 17.1 de Chapra: **aire** a 1 atm. Corregido y con la fuente citada |
| E2 | "un sistema de $n$ ecuaciones con $n$ incógnitas" | $n+1$ ecuaciones y $n+1$ incógnitas |
| E3 | El texto interpolaba $\{x_0,x_2,x_4\}$ y el código usaba `xi[0], xi[1], xi[4]` | El ejemplo se rehízo sobre la tabla de aire, con texto y código alineados |
| E4 | "la curva azul pasa por $(0{,}5;\,2{,}0)$" — era falso | Resuelto al rehacer el ejemplo |
| E5 | El texto anunciaba $p_5$ y $p_7$; el código generaba grados 4 y 6 | Redactado en términos de nodos y grado explícito |
| E6 | "Runge demostró el mal condicionamiento **de los polinomios**" | Lo mal condicionado es la interpolación de alto orden **sobre nodos equiespaciados** (NR §3.0) |
| E7 | $S_i$ definido en $[x_i,x_{i+1}]$ y evaluado en $x_{i-1}$ dos celdas después | Una sola convención de índices en todo el capítulo |
| E8 | Ecuaciones con `$$` y `&=&` estilo `eqnarray` | `align*`, compatible con KaTeX |
| E9 | "condiciones de borde `not-a-knot`, `clamped` y **`spline`**" | `natural` |
| E10 | *incognitas*, *contruir*, *domino*, *periodicas*, *condicion*, *quién*, "una único", "esta implementada" | Corregidos |
| — | "en esa clase revisaremos" | "en este capítulo revisaremos" |
| — | Título de figura "Interpolación lineal (interp1d de scipy)" cuando el código usa `numpy.interp` | Corregido |
| — | Comentario "# segunda derivada" sobre la línea de la tercera derivada | Corregido |
| — | `numpy interp` (sin punto) en el texto | `numpy.interp` |
| — | `y_1(15)` colgando al final de una celda (evaluación fuera de rango sin explicación) | Eliminado |
| — | La figura de extrapolación heredaba `xi`, `yi` de una celda 30 posiciones antes, que ya habían sido sobrescritos | La celda define sus propios datos |

## 2. Explicaciones que quedaban inconclusas y ahora se cierran

- **"Este método no es eficiente"** (sin decir por qué) → nueva sección
  **"La forma directa: el sistema de Vandermonde"**, con la matriz explícita y el cálculo
  de $\mathrm{Cond}(V)$: $5{,}89\times10^6$ con 3 datos y $9{,}85\times10^{26}$ con los 11
  de la tabla. Puente directo al número de condición de la **Unidad 2**.
  (Chapra §17.1.1 — el valor $5{,}89\times10^6$ coincide con el del libro.)
- **Newton y Lagrange como dos enlaces web** → una sección breve: son el mismo polinomio
  en otra base, y se prefieren porque evitan resolver el sistema de Vandermonde. Usa
  `images/newton_recursion.png`, que estaba sin usar.
- **`polyfit` "estrictamente no es interpolación"** → tabla que distingue los dos usos según
  el número de datos (NR §3.5: ajustar suaviza, interpolar toma el dato como exacto).
- **Runge sin desenlace** → animación A1, la regla práctica de NR (3-4 puntos sí, 5-6 tal
  vez, más casi nunca) y la fórmula del error de interpolación, que explica por qué el error
  explota en los bordes y conecta con el resto de Taylor (**Unidad 5**).
- **El conteo de $4n$ ecuaciones que no terminaba en nada** → nueva sección
  **"El sistema que hay que resolver es tridiagonal"** con la ecuación de NR (3.3.7) en las
  segundas derivadas, y el remate: Thomas lo resuelve en $O(n)$ (**Unidad 2**, y la misma
  matriz reaparece en **U10** y **U11**).
- **Por qué cúbico y no otro grado** → nueva sección con la regla de Chapra §18.3
  (para continuidad de la derivada $k$-ésima hace falta orden $k+1$) y la tabla
  grado → qué queda continuo.
- **Las derivadas del *spline* presentadas como una *feature* de SciPy** → nota puente:
  derivar el interpolante *es* la diferencia finita de la **Unidad 8**.
- **La extrapolación tratada dos veces como detalle de API** → una sola sección
  "Extrapolación: por qué no", con los tres métodos contrastados y la advertencia de
  NR §3.0 / Chapra §17.5.1.
- **La pregunta de la introducción ($\rho$ a 10 °C) nunca se respondía** → se responde al
  cierre: 1,2450 kg/m³ por interpolación lineal contra 1,2469 del gas ideal, 0,15 % de error.

## 3. Secciones nuevas

- Celda de librerías (`skip` + `remove-input`) con los imports y `plt.rcParams` una sola vez.
- `### La forma directa: el sistema de Vandermonde`
- `### Newton y Lagrange: el mismo polinomio, escrito distinto`
- `### ¿Por qué cúbico?`
- `### El sistema que hay que resolver es tridiagonal`
- `### Extrapolación: por qué no`
- `## Interpolación en Python` (tabla `numpy.interp` vs. `CubicSpline`)
- `## Resumen: ¿qué método uso?` con tabla de decisión, `images/mapa_unidad4.png`,
  cuatro verificaciones y el puente hacia adelante.
- Referencias reescritas con Chapra *Applied Numerical Methods* cap. 17-18 y
  *Numerical Recipes* cap. 3, con las subsecciones que corresponden a cada parte.

## 4. Cortes aplicados (−13 celdas)

| # | Qué era | Qué quedó |
|---|---|---|
| C1 | 9 celdas para envolver `np.interp` en un `lambda` y evaluarlo en 0.3, en 0.5 y en un arreglo, cada cosa por separado | 1 celda con la función y su evaluación vectorizada, más el gráfico |
| C2 | La extrapolación tratada dos veces, separada por 25 celdas | Una sola subsección |
| C3 | Las cuatro condiciones de borde, una por celda | Una tabla: condición · qué impone · cuándo usarla |
| C4 | 6 celdas de sintaxis de `bc_type` | 1 bloque de código comentado + 1 nota |
| C5 | Ejemplo de `polyfit` con cinco puntos inventados | El mismo ejemplo sobre cinco filas reales de la tabla de aire |

## 5. Animaciones (`interactive/`)

Cumplen el contrato técnico de la §8.1 de la guía: fragmento HTML sin iframe, CSS y JS
scopeados, SVG con `viewBox`, tipografía adaptativa con `ResizeObserver`. Prefijos únicos
verificados: `rng`, `trm`, `air`.

- **`A1_nodos_y_oscilacion.html`** — deslizador de número de nodos (3 a 17), tres funciones
  (Runge, suave, con esquina) e interruptor equiespaciados ↔ Chebyshev. Lee el error máximo,
  el del centro y el del borde por separado. Con 15 nodos sobre Runge: error 7,19 con nodos
  equiespaciados y 0,047 con Chebyshev.
- **`A2_global_vs_tramos.html`** — los mismos datos vistos por un polinomio global, por
  rectas y por un *spline* cúbico, con panel opcional de $y'(x)$ donde se ve el escalón del
  lineal y la continuidad del cúbico. Es la Figura 18.1 de Chapra, manipulable. El *spline*
  se calcula en el propio JS resolviendo el sistema tridiagonal con el algoritmo de Thomas.
- **`A3_tabla_de_aire.html`** — la tabla real de Chapra con deslizador de temperatura de
  −60 a 560 °C y las zonas de extrapolación sombreadas. Compara lineal, *spline* y polinomio
  de grado 10 contra la ley de gases ideales, que reproduce la tabla con tres cifras. Fuera
  del rango el polinomio de grado 10 se va a −14,6 kg/m³.

Las tres se probaron en Chromium a 1180 px y a 820 px de ancho, sin errores de consola.

## 6. Puentes agregados (encadenamiento)

| Dónde | Hacia | Qué objeto viaja |
|---|---|---|
| Introducción | **U3** | dato con ruido ⇒ tendencia; dato exacto ⇒ pasar por él |
| Vandermonde | **U2** | la matriz y su $\mathrm{Cond}$ |
| Newton y Lagrange | **U7**, **U8** | la base de Lagrange: integrarla y derivarla |
| Error de interpolación | **U5** | el resto de Taylor |
| Sistema del *spline* | **U2**, **U10**, **U11** | la matriz tridiagonal y el algoritmo de Thomas |
| Derivadas de `CubicSpline` | **U8** | derivar el interpolante |
| Cierre | **U7**, **U8**, **U10** | integrar, derivar y colocar el interpolante |

## 7. Notación de la unidad (para el Anexo A de la guía)

| símbolo | significado | alcance |
|---|---|---|
| $n$ | grado del polinomio; hay $n+1$ nodos, $i=0,\ldots,n$ | local (coherente con U3) |
| $x_i$, $y_i$ | nodos de interpolación (dato exacto, no medición) | **global** desde acá |
| $p_n(x)$ | polinomio de interpolación de grado $n$ | local |
| $L_i(x)$ | base de Lagrange | local, con puente a U7 |
| $V$ | matriz de Vandermonde | local, con puente a U2 |
| $S_i(x)$ | tramo $i$ del *spline*, definido en $[x_i, x_{i+1}]$ | local |
| $h_i = x_{i+1}-x_i$ | ancho del tramo $i$ | **global** |

## 8. Lo que NO se tocó

- Los `figsize` de las figuras existentes (`(6,5)` y `(4,3)`), por pedido explícito: están
  calibrados para el proyector. El estándar `(5,4)`/`(5,3)` se aplicó solo a lo nuevo.
- El video `_static/videos/BoundaryConditions.mp4` (Matías Rojas).
- Los `import` repetidos en celdas sueltas: la guía §6.1 dice que son deliberados para poder
  copiar una celda a Colab. Solo se consolidó `plt.rcParams`.
- `04-Interpolacion.css`. **Queda pendiente para su decisión:** la guía §10 dice que los
  capítulos rediseñados comparten el CSS de `03-Ajuste_de_curvas/`, pero ese CSS escala las
  imágenes en presentación (`--rise-image-scale: 1.6`) y cambiaría cómo se ven las figuras
  actuales en el proyector. No se copió.
- `images/2dinterpolation.png` sigue sin usarse (decisión D3 del plan: la interpolación en
  2D no entra en esta unidad).

## 9. Verificación realizada

- El notebook corre completo desde un kernel limpio, sin errores (`nbclient`).
- `nbformat.validate` sin observaciones.
- Sin `$$`, sin `eqnarray`, sin `\mbox`: 14 `equation*` y 4 `align*`, todo compatible con KaTeX.
- Sin `%`-formatting ni `.format()`: todos los `print` usan f-strings.
- 44 diapositivas, ninguna con más de 4 fragmentos.
- Los números citados en el texto coinciden con lo que imprime el código:
  $\mathrm{Cond}(V) = 5{,}89\times10^6$ y $9{,}85\times10^{26}$;
  $\rho(350\,^\circ\mathrm{C}) = 0{,}5676$ (Chapra: 0,567625);
  $\rho(10\,^\circ\mathrm{C})$ = 1,2450 lineal / 1,2431 *spline* / 1,2469 gas ideal.
- Las seis figuras se revisaron una por una después de ejecutar.
- Voz en primera persona plural en todo el capítulo, incluidos los enunciados y los textos
  de ayuda de las tres animaciones.

## 10. Comandos de git (correr en la terminal de Windows)

```
cd "...\material_catedra"
git checkout -b mejora-unidad4
git add 04-Interpolacion/
git commit -m "Unidad 4: rediseño 2026 (Vandermonde, spline tridiagonal, 3 animaciones, resumen y mapa)"
```
