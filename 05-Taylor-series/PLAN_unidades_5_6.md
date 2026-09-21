# Plan de rediseño — Unidades 5 y 6

**Fecha:** 2026-09-20 · **Estado: IMPLEMENTADO** el 2026-09-20. Registro detallado en `CAMBIOS_unidad5.md` y `../06-Root-finding/CAMBIOS_unidad6.md`.

Resultado: **U5 de 31 a 69 celdas**, **U6 de 105 a 125**. Ambos notebooks corren limpios desde kernel vacío; las tres animaciones pasan Playwright a 1180 px y 820 px sin errores. Queda abierto el corte C3.
**Alcance:** `05-Taylor-series/05-Taylor-series.ipynb` (31 celdas) y `06-Root-finding/06-Root-finding.ipynb` (105 celdas), tratadas como una sola unidad narrativa en dos archivos.

Respaldos ya creados (verificados idénticos al original por `md5sum`):

- `05-Taylor-series/backup/05-Taylor-series_2026-09-20.ipynb`
- `06-Root-finding/backup/06-Root-finding_2026-09-20.ipynb`

## 0. Decisiones tomadas (2026-09-20)

1. **C1 no se corta.** Las celdas 79-94 de U6 (búsqueda lineal y región de confianza) se mantienen como están: es la única vez que el curso nombra optimización multidimensional. Sí se corrigen dentro de ellas los defectos D8 y D9, que son de notación y no de contenido: $\vec{F},\ \bar{J} \rightarrow \mathbf{F},\ J$, y $\mathbf{x}_k = \mathbf{x}_{k-1} - J^{-1}\mathbf{F}$ pasa a $J\,\Delta\mathbf{x} = -\mathbf{F}$, que es la forma en que efectivamente se resuelve y la que conecta con la Unidad 2.
2. **Ejemplo conductor de U5: el péndulo**, $\sin\theta \approx \theta$.
3. **Animación A4 (Colebrook en vivo): fuera.** Quedan tres: A1 y A2 en U5, A3 en U6.
4. **CSS: se unifica.** Las Unidades 5 y 6 adoptan el archivo compartido de `03-Ajuste_de_curvas/`. Consecuencia concreta, porque hoy los CSS de U5 y U6 no escalan imágenes: al entrar `--rise-image-scale: 1.6`, las seis imágenes de U6 pasan de 350-600 px a 560-960 px en la diapositiva. Hay que recalibrar sus anchos pensando en el apunte, según la regla de §7 de la guía. Propuesta: `presure_drop` 350 → 300, `valor_intermedio` 400 → 380, `bisection` 400 → 380, `newton_raphson` 400 → 380. Las dos GIF quedan sujetas a C5.

5. **Cortes aprobados: C2, C4, C5 y C6.** Queda abierto solo **C3** (comprimir el volcado de la firma de `root_scalar`, celdas 55-59, −2). Mientras no lo definas, la firma se mantiene como está y solo se corrige la tabla de métodos (D5).

Con eso, el tamaño final proyectado es **U5 ≈ 62 celdas** y **U6 ≈ 124** (122 si entra C3).

**C5 aprobado** implica que las celdas 34 y 44 de U6 pierden su figura. La animación A3 las reemplaza, así que A3 deja de ser opcional: es la que sostiene la sección de convergencia de Newton.

---

## 1. Diagnóstico

### Unidad 5

La unidad no es un capítulo, es un anexo de 31 celdas. Le falta el objeto que el resto del curso le va a pedir prestado: **el resto de Taylor $R_n$ y la forma en paso $h$**.

Se identifican las siguientes brechas:

1. **No existe el resto.** El capítulo define la serie, la trunca y dice "a mayor orden, mejor la aproximación", pero nunca escribe $R_n = \frac{f^{(n+1)}(\xi)}{(n+1)!}(x-a)^{n+1}$. Sin ese término no hay forma de decir *cuánto* mejor, ni de dónde sale el $O(h^{n+1})$ que usarán las Unidades 7, 8 y 9.
2. **No existe la forma en paso $h$.** El capítulo trabaja siempre en $(x-a)$. La forma que realmente se usa después es $f(x_i+h) = f(x_i) + f'(x_i)h + \tfrac{1}{2}f''(x_i)h^2 + O(h^3)$. Es la misma ecuación, pero si no se escribe aquí, cada unidad posterior la reintroduce sola.
3. **No hay introducción ni puente hacia atrás.** El capítulo abre con la fórmula, sin problema de ingeniería y sin conexión con la Unidad 4, aunque ambos resuelven el mismo problema canónico: aproximar por un polinomio.
4. **No hay resumen, mapa ni puente hacia adelante.** La Unidad 5 existe para abrir la Parte 2 y no lo dice.
5. **La sección de errores queda a medias.** Están truncamiento ($e^2$) y redondeo ($e^{-30}$) por separado, pero falta lo que los une: el error total y el $h$ óptimo (Chapra §4.4). Es justamente el puente hacia la Unidad 8.
6. **Error factual, celda 17.** El texto dice "para el orden 8 la aproximación es casi perfecta"; el código recorre `range(4)`, es decir órdenes 1, 3, 5 y 7, y el error máximo con orden 7 en $[-\pi,\pi]$ es $7{,}5\times10^{-2}$. Ni es orden 8 ni es casi perfecta.

### Unidad 6

El tamaño es correcto. El problema es de estructura:

1. **El ejemplo conductor se abandona.** Colebrook se plantea en las celdas 1-7 y no se resuelve nunca. El capítulo cierra con $f(x)=x^3-1$.
2. **Treinta celdas finales de prosa descriptiva.** Las celdas 79-94 (búsqueda lineal, región de confianza) describen algoritmos sin una sola ecuación que el estudiante vaya a usar ni un ejemplo.
3. **Newton-Raphson aparece sin origen.** Dice "se origina a partir de series de Taylor" y pasa de largo. Es la conexión más importante de toda la Parte 2 y ocupa media línea.
4. **Falta el orden de convergencia.** No se dice que la bisección es lineal ni que Newton es cuadrático. Es el eje "costo" del curso aplicado a este capítulo, y está en Chapra §5.4/§6.2 y en Numerical Recipes §9.1/§9.4.
5. **No hay resumen, tabla de decisión, mapa ni puente hacia adelante.**
6. **Dos GIF prestados de un blog**, en inglés y con notación ajena ($y=f(x_k)$, $n$ para la iteración).

### Defectos de código y notación (ambas unidades)

| # | Dónde | Qué |
|---|---|---|
| D1 | U6 celda 36 | `newton_raphson(x0, f, fprime, ...)` usa `df(xk)` global en vez de `fprime(xk)`. Funciona por accidente |
| D2 | U6 celda 24 | `bisection` retorna `m`, que no existe si converge en `k = 0` |
| D3 | U6 celda 47 | El texto anuncia $x_0=0$ y $x_0=0{,}01$, el código solo corre $x_0=0$, y la celda 48 reporta ambos resultados |
| D4 | U6 celda 49 | Dice "la segunda iteración"; es la primera |
| D5 | U6 celda 59 | La tabla de `root_scalar` tiene una `o` de más por fila: 10 entradas para 9 columnas |
| D6 | U6 celdas 9, 10, 74 | "una o más variables **dependientes**" debe ser *independientes* |
| D7 | U6 | $x_0$ se usa como raíz (celdas 11-12) y como valor inicial (celda 27 en adelante) |
| D8 | U6 | $\vec{F}$, $\bar{J}$, $\vec{x}$ contradicen §5.1 de la guía: vector negrita minúscula, matriz mayúscula |
| D9 | U6 celda 79 | $\mathbf{x}_k = \mathbf{x}_{k-1} - J^{-1}\mathbf{F}$ invita a invertir la matriz, justo lo que la Unidad 2 dice que no se hace |
| D10 | U5 celda 3, U6 celda 48 | `$$` con `aligned`, prohibido por la guía |
| D11 | U5 celdas 15, 25, 28 · U6 celdas 25, 38, 47, 50, 67, 69 | `%`-formatting en vez de f-strings |
| D12 | U5 | No hay celda de librerías al inicio (`skip` + `remove-input`); los `import` están dispersos |
| D13 | Ortografía | *Analisis, susesiva, incluído, erronea, fluído, dimencional, interfalo, vetorial, derviada, incognitas, polinómio, menciona* (por *mencionar*), *Usamor `roor_scalar`* |
| D14 | U6 celda 3 | El texto dice "donde $f$ es el factor de fricción"; la ecuación usa $f_c$ |

---

## 2. Arquitectura propuesta

### 2.1 La bisagra U5 → U6

La conexión entre ambas unidades se apoya en una sola frase, que se escribe en U5 y se cobra en U6:

> Truncar la serie de Taylor en el primer orden reemplaza la función por su recta tangente. Newton-Raphson es exactamente eso: reemplazar $f$ por su recta tangente y resolver la ecuación lineal que queda.

En U5 la frase cierra la sección de aproximación lineal y planta la pregunta, sin dar el método. En U6 la misma frase abre Newton-Raphson, con hipervínculo de vuelta.

**Mecanismo de hipervínculo.** Se usa el patrón que ya existe en las Unidades 10 y 11:

```markdown
[Unidad 5: aproximación lineal](../05-Taylor-series/05-Taylor-series.ipynb)
```

Para los tres destinos que se citan por sección, se agrega un ancla HTML sobre el encabezado (`<a id="aproximacion-lineal"></a>`), invisible en RISE y funcional en el libro. Enlaces previstos:

| Desde | Hacia |
|---|---|
| U6 §Newton-Raphson | U5 §Aproximación lineal |
| U6 §Criterio de convergencia | U5 §Error de truncamiento |
| U6 §Newton vectorial | U2 §Sistemas lineales |
| U5 §Resumen | U6 §Búsqueda de raíces |
| U5 §Introducción | U4 §Interpolación |

**Contra la redundancia.** La Unidad 6 no vuelve a derivar la serie de Taylor ni a definir error absoluto y relativo. Solo repite lo que necesita reformulado: la recta tangente en la notación de iteración ($x_k$, $x_{k+1}$), en una celda, con el enlace al desarrollo completo.

### 2.2 Unidad 5 — de 31 a ~62 celdas

**Ejemplo conductor: el péndulo simple.** La linealización $\sin\theta \approx \theta$ es la aproximación de primer orden más usada en ingeniería, y el resto de Taylor entrega el número que justifica hasta dónde vale.

Valores verificados para el capítulo:

| $\theta_0$ | error de $\sin\theta\approx\theta$ | error en el periodo $T$ |
|---|---|---|
| 5° | 0,13 % | 0,05 % |
| 10° | 0,51 % | 0,19 % |
| 14° | 1,00 % | — |
| 15° | 1,15 % | 0,43 % |
| 30° | 4,72 % | 1,71 % |

La cota del resto, $|R_2| \le \theta^3/3!$, predice el error real con razón 1,00 a 1,13 en $\theta \in [0{,}1;\ \pi/2]$. Es decir, la cota no es un adorno teórico: da el número.

Estructura:

```
# Series de Taylor                                     [slide]
   celda de librerías                                  [skip, remove-input]
## Introducción                                        [slide]
   el péndulo: ¿por qué todo el mundo escribe sin θ ≈ θ?
   puente atrás a U4: mismo problema canónico, dos informaciones distintas
   (U4: el polinomio pasa por los datos · U5: el polinomio copia las derivadas en un punto)
## Expansión en series de Taylor                       [slide]
   definición · polinomio de un polinomio (comprimido) · sin(x)
## Truncamiento y orden de aproximación                [slide]
   serie truncada · gráfico de sin(x) corregido · A1
## El resto de Taylor                                  [slide]   ← NUEVO
   R_n en forma de Lagrange · la cota aplicada al péndulo · O(h^{n+1})
## La forma en paso h                                  [slide]   ← NUEVO
   f(x_i+h) = f(x_i) + f'(x_i)h + ... + O(h^{n+1})
   aproximación lineal · nota puente a U6, U7, U8, U9
## Error de truncamiento                               [slide]
   e^2 por orden (tabla existente, con f-strings e interpretación)
## Error de redondeo                                   [slide]
   e^{-30} (ejemplo existente) + por qué falla + cómo se arregla
## El error total y el h óptimo                        [slide]   ← NUEVO
   la curva en V · A2 · puente a U8
## Resumen: ¿qué me llevo de Taylor?                   [slide]   ← NUEVO
   tabla + images/mapa_unidad5.png + puente adelante
## Referencias                                         [slide]
```

### 2.3 Unidad 6 — de 105 a ~122 celdas

Entran 29 celdas y salen 12, si se aprueban C2 a C5.

```
# Algoritmos de búsqueda de raíces                     [slide]
   celda de librerías                                  [skip, remove-input]
## Introducción
   Colebrook (existente) + puente atrás a U5           ← AMPLIADO
## Búsqueda de raíces para una función escalar         (comprimido, C4)
### Método de la bisección
   teorema del valor intermedio · algoritmo · criterio de convergencia
   cuántas iteraciones toma: n = log2((b−a)/tol)       ← NUEVO
### Método de Newton-Raphson
   de la recta tangente a la iteración, con enlace a U5 ← AMPLIADO
   A1 (reemplaza los dos GIF)                          ← NUEVO
   orden de convergencia: los dígitos correctos se duplican ← NUEVO
### Cuando Newton falla
   f'(x_k) ≈ 0 · control de la raíz (con el código corregido, D3)
### Método de la secante
### Métodos combinados (Brent)
### Raíces de función escalar en Python                (comprimido, C2 y C3)
## Cerramos Colebrook                                  [slide]   ← NUEVO
   Swamee-Jain como x0 · brentq · ΔP · bisección vs Newton en iteraciones
## Búsqueda de raíces para funciones vectoriales       (se mantiene, notación corregida)
   Newton vectorial: J Δx = −F  ← la reducción a Ax = b, con enlace a U2
   búsqueda lineal y región de confianza (texto actual)
   fsolve (ejemplo existente)
## Resumen: ¿qué método uso?                           [slide]   ← NUEVO
   tabla + images/mapa_unidad6.png + verificaciones + puente a U9 y U10
## Referencias                                         (ampliadas)
```

---

## 3. Cortes (✔ aprobado · ⧗ pendiente)

| # | Celdas | Qué es hoy | Propuesta | Saldo |
|---|---|---|---|---|
| ~~C1~~ | U6 79-94 | Búsqueda lineal y región de confianza | **Descartado.** Se mantienen; solo se corrige la notación (D8, D9) | 0 |
| C2 ✔ | U6 66-72 | `sol.root`, `xtol`, `rtol`, `maxiter` en 7 celdas | 3 celdas | −4 |
| C3 ⧗ | U6 55-59 | Volcado completo de la firma de `root_scalar` + tabla de métodos | Mantener la tabla (corregida, D5), comprimir la firma | −2 |
| C4 ✔ | U6 9-13 | Definición formal de función escalar $f:\mathbb{R}^n\to\mathbb{R}$ y clasificación de métodos, que se repite en la celda 74 | 2 celdas | −3 |
| C5 ✔ | U6 34, 44 | Los dos GIF prestados del blog | Reemplazados por A1 | −3 |
| C6 ✔ | U5 2-6 | Expansión de $5x^2+3x+5$ en $a=0$ y en $a=2$, en 5 celdas | 2 celdas, conservando el corolario "la expansión de Taylor de un polinomio es el mismo polinomio" | −3 |

Aprobados C2, C4, C5 y C6: **−13 celdas** (−3 en U5, −10 en U6). C3 sigue abierto (−2 más).

Tamaño final proyectado: **U5 ≈ 62** y **U6 ≈ 124**, dentro del rango cómodo de 90-130 de la guía.

---

## 4. Animaciones

Tres animaciones. Todas siguen el contrato técnico de §8.1 de la guía: fragmento HTML sin iframe, CSS y JS scopeados con prefijo de 3 letras, SVG con `viewBox`, tipografía adaptativa por `ResizeObserver`.

### A1 · U5 — "Taylor bajo la lupa" (prefijo `tay`)

- **Controles:** orden $N$ (0-12) · punto de expansión $a$ (arrastrable sobre el eje) · función ($\sin x$, $e^x$, $\ln(1+x)$, $1/(1-x)$).
- **Muestra:** la función, el polinomio $p_N$, y un punto $x$ arrastrable con dos lecturas: error real $|f(x)-p_N(x)|$ y cota del resto $|R_N|$.
- **Qué enseña que el texto no puede:** que el error crece con la distancia a $a$ y baja con $N$, pero que en $1/(1-x)$ evaluado en $x=1{,}2$ subir $N$ **empeora** el resultado (verificado: $N=5 \to 9{,}93$; $N=20 \to 225$; $N=40 \to 8814$; el valor exacto es $-5$). Ahí aparece el radio de convergencia sin necesidad de teoría.
- **Cierre:** cerca de $a$ la recta tangente basta. Esa es la puerta a la Unidad 6.

### A2 · U5 — "El $h$ óptimo" (prefijo `err`)

- **Controles:** deslizador de $h$ en escala logarítmica.
- **Muestra:** curva en V en log-log del error de $(f(x+h)-f(x))/h$ contra $h$, con las dos asíntotas dibujadas: truncamiento $\propto h$ y redondeo $\propto \varepsilon_\mathrm{maq}/h$.
- **Números verificados** ($f=\sin$, $x=1$): el error baja hasta $3{,}0\times10^{-9}$ en $h\approx10^{-8}$ y después sube; en $h=10^{-16}$ vale $0{,}54$. La predicción teórica es $h_\mathrm{opt}\approx\sqrt{\varepsilon_\mathrm{maq}}=1{,}5\times10^{-8}$.
- **Qué enseña:** más resolución no es más precisión. Es la figura de Chapra §4.4 y el puente directo a la Unidad 8.

### A3 · U6 — "Bisección contra Newton" (prefijo `rai`)

- **Controles:** función (4 casos preconfigurados) · $x_0$ arrastrable · intervalo $[a,b]$ arrastrable · botón "iterar" paso a paso.
- **Muestra:** dos paneles con la misma función; abajo, tabla de $|x_k - x^*|$ por iteración.
- **Qué enseña:** en bisección el error se corta a la mitad cada paso; en Newton los dígitos correctos se duplican. Verificado sobre Colebrook desde $x_0=0{,}02$: $1{,}4\times10^{-3} \to 8{,}1\times10^{-5} \to 2{,}7\times10^{-7} \to 2{,}8\times10^{-12}$.
- **Casos que rompen Newton**, incluidos como preajustes: $x^3-100x^2-x+100$ desde $x_0=0$ salta a $x^*=100$ en una iteración, mientras que desde $x_0=0{,}01$ llega a $x^*=1$ en 10 pasos. Reemplaza al GIF prestado y además arregla D3.

### A4 · U6 — descartada

"Colebrook en vivo" queda fuera. El cierre del ejemplo conductor se resuelve en texto y código.

---

## 5. Números verificados

No recalcular a ciegas. Todo lo de abajo está comprobado con NumPy y SciPy.

**Colebrook — ejemplo conductor cerrado.** Agua a 20 °C ($\nu = 1{,}004\times10^{-6}$ m²/s, $\rho = 998$ kg/m³), tubería de acero comercial ($\varepsilon = 0{,}045$ mm), $D = 0{,}1$ m, $L = 100$ m, $V = 2$ m/s:

- $\mathrm{Re} = 1{,}992\times10^5$ · $\varepsilon/D = 4{,}5\times10^{-4}$
- $f_c = 0{,}018567$ · $\Delta P = 37{,}1$ kPa · $h_f = 3{,}79$ m
- Swamee-Jain como valor inicial: $f_0 = 0{,}018672$, a 0,57 % de la solución
- Bisección en $[0{,}008;\ 0{,}08]$ con tolerancia $10^{-8}$: **23 iteraciones**, exactamente las que predice $n = \lceil\log_2((b-a)/\mathrm{tol})\rceil = 23$
- Newton desde $x_0 = 0{,}02$: **4 iteraciones**

**Taylor de $\sin(x)$, error máximo en $[-\pi,\pi]$:** orden 1 → 3,14 · orden 3 → 2,03 · orden 5 → 0,524 · orden 7 → $7{,}5\times10^{-2}$ · orden 9 → $6{,}9\times10^{-3}$ · orden 11 → $4{,}5\times10^{-4}$.

**$e^{-30}$ por serie directa (200 términos):** entrega $-8{,}55\times10^{-5}$ contra el exacto $9{,}36\times10^{-14}$. El término más grande de la serie es $7{,}76\times10^{11}$ en $i=29$, y $\varepsilon_\mathrm{maq}\times7{,}76\times10^{11} = 1{,}7\times10^{-4}$, del orden del resultado erróneo. Ese es el mecanismo exacto de la falla. La cura, que conviene mostrar: calcular $1/\sum 30^i/i!$ entrega $9{,}357623\times10^{-14}$, con error relativo $4\times10^{-16}$.

**Detalle de implementación a conservar.** En la celda de $e^{-30}$, `x = -30` debe seguir siendo entero: `x**i / math.factorial(i)` es una división exacta entre enteros grandes. Si se escribe `x = -30.0`, Python levanta `OverflowError` al convertir `factorial(171)` a flotante.

**$e^x - x^2$ (ejemplo de ambos métodos):** la raíz es $-0{,}703467$. Bisección en $[-1,1]$ con tolerancia $10^{-5}$: 18 iteraciones. Newton desde $x_0=1$: 6 iteraciones, y el primer paso lo lanza a $-1{,}39$, fuera del intervalo. Vale la pena mostrarlo: es la diferencia entre acotar y no acotar.

---

## 6. Notación a fijar (Anexo A de la guía)

| símbolo | significado | unidad |
|---|---|---|
| $a$ | punto de expansión de la serie | U5 |
| $N$ | orden de la aproximación | U5 |
| $R_n$ | resto de Taylor, forma de Lagrange | U5, con puente a U7 y U8 |
| $\xi$ | punto intermedio del resto, $\xi \in (a,x)$ | U5 |
| $h$ | paso, $h = x - x_i$ | global, mismo $h$ de U4 y U7-U11 |
| $\varepsilon_\mathrm{maq}$ | épsilon de máquina | global desde U5 |
| $x^*$ | raíz, $f(x^*)=0$ | U6 |
| $x_k$ | iterado $k$; $x_0$ es **solo** el valor inicial | U6 |
| $\mathbf{F}$, $J$ | función vectorial y su Jacobiano, $J_{ij}=\partial f_i/\partial x_j$ | U6, con puente a U2 |

El cambio de $\vec{F}$, $\bar{J}$ a $\mathbf{F}$, $J$ (D8) alinea la Unidad 6 con §5.1 de la guía y con las Unidades 2, 3 y 4.

---

## 7. Referencias a ampliar

**Unidad 5**, que hoy cita un solo texto:

- Chapra S. **Chapter 4: Roundoff and Truncation Errors** en *Applied Numerical Methods with MATLAB for Engineers*, 3rd Ed.
  - §4.3 errores de truncamiento y el resto · §4.4 error total y el $h$ óptimo
- Press W., Teukolsky S., Vetterling W., Flannery B. **Chapter 5: Evaluation of Functions** en *Numerical Recipes*, 3rd Ed.
  - §5.1 evaluación de polinomios (Horner) · §5.3 series y su convergencia · §5.7 derivadas numéricas y el $h$ óptimo
  - §1.1 error, exactitud y estabilidad
- Kong Q., Siauw T., Bayen A. M. **Chapter 18: Taylor Series** (la que ya está).

**Unidad 6:**

- Chapra S. **Chapter 5: Roots — Bracketing Methods** y **Chapter 6: Roots — Open Methods** en *Applied Numerical Methods with MATLAB for Engineers*, 3rd Ed.
  - §5.4 bisección y su número de iteraciones · §6.2 Newton-Raphson · §6.3 secante · §6.4 método de Brent · **§6.7 caso de estudio: fricción en tuberías**, que es el mismo problema de Colebrook del capítulo
- Press W. *et al.* **Chapter 9: Root Finding and Nonlinear Sets of Equations** en *Numerical Recipes*, 3rd Ed. (corregir la autoría: hoy aparece como "Williams H. P.")
  - §9.1 acotamiento y bisección · §9.4 Newton-Raphson con derivada · §9.6 Newton para sistemas no lineales · §9.7 métodos globalmente convergentes

---

## 8. Flujo de trabajo

1. Respaldos con fecha. **Hecho.**
2. Rama de trabajo. Los comandos `git` sobre la carpeta de OneDrive se cuelgan si se ejecutan por herramientas remotas, así que van para correr en una terminal de Windows:

   ```
   cd "C:\Users\frami\OneDrive - Universidad Adolfo Ibanez\Teaching - MEC301 - Metodos Numericos\material_catedra"
   git checkout -b mejora-unidades-5-6
   git add -A
   git commit -m "backup U5 y U6 antes del rediseno"
   ```

3. Orden de ejecución sugerido: U5 primero (define la notación y el resto de Taylor que U6 necesita), después U6, y las animaciones al final, cuando el texto que las enuncia ya esté fijo.
4. Verificación antes de cerrar: ejecutar ambos notebooks completos con `nbclient` en un kernel limpio, extraer las figuras generadas y revisarlas una por una; probar las tres animaciones con Playwright a 1180 px y 820 px, capturando `pageerror`.
5. Registro en `CAMBIOS_unidad5.md` y `CAMBIOS_unidad6.md`.
6. Al cerrar: actualizar el Anexo A (notación de U5 y U6) y el Anexo C (estado del rediseño y catálogo de puentes §3.4) de `GUIA_FORMATO.md`.

---

## 9. Lo que queda por decidir

1. **C3**: comprimir el volcado de la firma de `root_scalar` (celdas 55-59, −2). Por defecto se mantiene.
2. **Tamaño de U5.** Queda en ~62 celdas, bajo el rango cómodo de 90-130 de la guía. Me parece correcto: es una unidad bisagra, no una unidad de métodos. Si prefieres acercarla al rango, el material natural para engordarla es el ejemplo de $e^x$ y las series alternantes de Numerical Recipes §5.3.
3. **Los anchos de imagen de U6 tras unificar el CSS** hay que verificarlos en el proyector. Eso no lo puedo hacer desde acá.
