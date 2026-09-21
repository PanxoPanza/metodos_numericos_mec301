# Cambios — Unidad 6 · Búsqueda de raíces

**Fecha:** 2026-09-20 · **Plan:** `../05-Taylor-series/PLAN_unidades_5_6.md`
**Respaldo del original:** `backup/06-Root-finding_2026-09-20.ipynb` (105 celdas)
**Resultado:** 125 celdas.

## Por qué

El tamaño era correcto; el problema era de estructura. El ejemplo conductor (Colebrook) se planteaba en la celda 7 y no se resolvía nunca. Newton-Raphson aparecía sin explicar de dónde sale. No había orden de convergencia, ni resumen, ni mapa, ni puente hacia adelante. Y el código tenía errores reales.

## Qué se agregó

- **Puente hacia atrás a la Unidad 5**, con hipervínculo, en la introducción y otra vez al abrir Newton-Raphson. La derivación ahora dice explícitamente que sale de truncar la serie de Taylor en primer orden.
- **¿Cuántas iteraciones cuesta la bisección?** — $n = \lceil\log_2((b-a)/\mathrm{tol})\rceil$, la única garantía de este tipo en el capítulo. Verificado en el ejemplo de Colebrook: la predicción da 23 y la bisección toma exactamente 23.
- **Orden de convergencia**, con tabla comparando bisección (lineal), secante (superlineal) y Newton (cuadrático), y la advertencia de que la promesa de Newton es local.
- **Sección nueva "Cerramos el problema de Colebrook"** — 8 celdas que resuelven el problema del comienzo con datos reales: agua a 20 °C, acero comercial, $D = 0{,}1$ m, $L = 100$ m, $V = 2$ m/s. Resultado $f_c = 0{,}018567$, $\Delta P = 37{,}1$ kPa, $h_f = 3{,}79$ m. Incluye la comparación de costo entre métodos y Swamee-Jain como valor inicial (0,57 % de error sin iterar).
- **Resumen, tabla de decisión, verificaciones y mapa** `images/mapa_unidad6.png`, con puente hacia adelante a las Unidades 9 y 10.
- **Celda de librerías** al inicio (`skip` + `remove-input`). Los imports repetidos más adelante se conservan a propósito, para poder copiar una celda suelta a Colab.
- **Animación A3** (ver más abajo).
- **Cinco notas puente** con la unidad en negrita: Unidad 1 (error absoluto y relativo), Unidad 5 (dos veces), Unidad 2 (el sistema lineal de Newton vectorial), Unidad 8 (la secante como diferencia dividida), y el cierre hacia las Unidades 9, 10 y 11.

## Qué se corrigió

| # | Dónde | Qué pasaba |
|---|---|---|
| D1 | `newton_raphson` | Recibía `fprime` pero usaba la `df` global. Funcionaba solo por accidente, porque `df` estaba definida afuera |
| D2 | `bisection` | Retornaba `m`, que no existía si la función convergía en `k = 0` |
| D3 | Control de la solución | El texto anunciaba $x_0=0$ y $x_0=0{,}01$, el código corría solo uno, y la celda siguiente reportaba ambos resultados. Ahora el loop corre los dos |
| D4 | Misma sección | "la segunda iteración nos da $x_1 = 100$" — es la primera |
| D5 | Tabla de `root_scalar` | Cada fila tenía una `o` de más: 10 entradas para 9 columnas |
| D6 | Definición de función escalar y vectorial | "una o más variables **dependientes**" — son *independientes* |
| D7 | Notación | $x_0$ se usaba como raíz y como valor inicial. Ahora $x^*$ es la raíz, $x_k$ el iterado y $x_0$ solo el valor inicial |
| D8 | Sección vectorial | $\vec{F}$, $\vec{x}$, $\bar{J}$ contradecían §5.1 de la guía. Ahora $\mathbf{F}$, $\mathbf{x}$, $J$ |
| D9 | Newton vectorial | Estaba escrito $\mathbf{x}_k = \mathbf{x}_{k-1} - J^{-1}\mathbf{F}$, que invita a invertir la matriz, justo lo que la Unidad 2 dice que no se hace. Ahora $J\,\Delta\mathbf{x} = -\mathbf{F}$, con nota puente a la Unidad 2 |
| D10 | Control de la solución | `$$ ... \begin{aligned}` → `align*` |
| D11 | Seis celdas de código | `%`-formatting → f-strings |
| D12 | Figura del polinomio cúbico | `figsize=(9,6)` y `plt.rcParams` local → `figsize=(5,4)`, con las dos raíces marcadas |
| D13 | Ortografía | *fluído, dimencional, interfalo, vetorial, derviada, incognitas, menciona* (por *mencionar*), *Usamor `roor_scalar`*, "método de Brent's" |
| D14 | Introducción | El texto decía "donde $f$ es el factor de fricción"; la ecuación usa $f_c$ |

## Cortes aplicados

| Corte | Qué | Saldo |
|---|---|---|
| C2 | `sol.root`, `xtol`, `rtol` y `maxiter` ocupaban 7 celdas de texto casi idéntico. Quedan en 3, sin perder ningún argumento | −4 |
| C4 | La definición formal $f:\mathbb{R}^n\to\mathbb{R}$ y la clasificación de métodos ocupaban 5 celdas y se repetían más adelante. Quedan en 2 | −3 |
| C5 | Las dos GIF prestadas de un blog (en inglés, con notación ajena) se sacaron del capítulo. La animación A3 las reemplaza | −0 celdas |

**C1 quedó descartado por decisión del usuario:** las 16 celdas de búsqueda lineal y región de confianza se mantienen; solo se corrigió su notación (D8 y D9).
**C3 sigue abierto:** comprimir el volcado de la firma de `root_scalar`.

> Los archivos `images/newton_raphson_good.gif` y `newton_raphson_bad.gif` quedan en la carpeta sin usar, igual que `2dinterpolation.png` en la Unidad 4. No se borraron.

## Animación

`interactive/A3_biseccion_vs_newton.html` (prefijo `rai`), contrato técnico de §8.1 de la guía.

Cuatro funciones preconfiguradas, valor inicial de Newton arrastrable, e iteración paso a paso con una tabla que compara la cota de la bisección contra el error de Newton. Con Colebrook desde $x_0 = 0{,}02$ la tabla muestra la convergencia cuadrática en crudo: $1{,}4\times10^{-3} \to 8{,}1\times10^{-5} \to 2{,}7\times10^{-7} \to 2{,}8\times10^{-12}$, mientras la cota de la bisección solo se va a la mitad. Los preajustes $x^3-100x^2-x+100$ desde $x_0=0$ y $\arctan(x)$ desde $x_0=1{,}5$ muestran los dos modos de falla de un método abierto.

## Otros archivos

- `06-Root-finding.css` reemplazado por el compartido de `03-Ajuste_de_curvas/`. **Consecuencia:** las imágenes con atributo `width` ahora se escalan por `--rise-image-scale: 1.6` en RISE. Los anchos se bajaron en consecuencia: `presure_drop` 350 → 300, y `valor_intermedio`, `bisection` y `newton_raphson` de 400 a 380. Conviene revisarlos en el proyector.
- `images/mapa_unidad6.png`, nueva.

## Verificación

- El notebook corre completo desde un kernel limpio, sin errores (`nbclient`).
- Las cifras del texto se contrastaron contra la salida real: bisección 18 iteraciones y Newton 6 en $e^x-x^2$; $x_0=0 \to x^*=100$ en 2 pasos y $x_0=0{,}01 \to x^*=1$ en 9; Colebrook `bisect` 23, `brentq` 10, `secant` 4.
- La animación se probó con Playwright a 1180 px y 820 px, sin `pageerror`.

---

## Escalado de animaciones en RISE (2026-09-21)

`06-Root-finding.css` (compartido con U3, U4 y U5) escala las animaciones en modo presentación con `--rise-anim-scale: 1.45` y `--rise-anim-max: 820px`, aplicados por `zoom` a los elementos con `id` terminado en `-app`. A3 pasó además a medir el ancho del SVG con `clientWidth`, que es inmune al `zoom`; con `getBoundingClientRect()` el texto del SVG quedaba clavado en 13 px mientras el resto crecía. Detalle completo en §8.1.1 de `GUIA_FORMATO.md`. Respaldos en `backup/` e `interactive/backup/`.

El contenido del capítulo no se tocó en esa sesión.

**Corrección del 2026-09-21 (misma sesión):** la primera versión de esa regla usaba `div[id$="-app"]` sin acotar, que también calza con `#ipython-main-app`, el contenedor de todo el notebook en Jupyter clásico. En RISE eso agrandaba la página completa y la dejaba en una columna angosta. La regla quedó limitada a `div.output_area` / `div.output_subarea` / `.jp-OutputArea-output`.

---

## Animación de región de confianza (2026-09-21)

`interactive/A4_region_de_confianza.html`, prefijo `reg`. Respaldo del notebook previo en `backup/06-Root-finding_2026-09-21b.ipynb`. Se insertó al cierre de `### Métodos de región de confianza`: enunciado, animación y dos notas. El resto del capítulo no se tocó.

**Qué muestra.** El sistema $\mathbf{F}(x,y) = \big(10(y-x^2),\ 1-x\big)$, cuya norma al cuadrado tiene un valle curvo y raíz en $(1,1)$. Se eligió por sus escalas comparables: la ventana se calcula isotrópica a partir del alto del gráfico, de modo que la región de confianza se dibuja como un círculo real y no como una elipse. Con el sistema de la celda 101 del capítulo esto no es posible, porque $x$ recorre un rango seis veces mayor que $y$.

En cada iteración se ve el paso de Newton completo (azul punteado), el paso que el radio permite (rojo), la región (círculo naranjo) y el camino aceptado (negro). El panel entrega $\Delta$, $|\mathbf{F}|$, el largo del paso de Newton y $\rho$, el cociente entre la reducción real y la predicha por el modelo. El radio se duplica cuando $\rho > 0.75$ con el paso recortado, y se reduce a la mitad cuando $\rho < 0.25$.

**Preajustes y lo que muestra cada uno.** Desde $(-1.2,\,1.0)$ con $\Delta = 0.5$, el paso de Newton mide 5.32 y la región lo recorta a 0.5. Desde $(-0.5,\,-0.4)$ con $\Delta = 2$, los dos primeros pasos se rechazan con $\rho = -10.4$ y $\rho = -0.86$, y el radio baja de 2 a 0.5 antes de aceptar el primero. Con $\Delta$ grande el método se comporta como Newton puro. También se puede hacer clic en el gráfico para mover el punto de partida.

**Implementación.** Curvas de nivel por *marching squares* sobre una malla de 72×72, agrupadas en una ruta por nivel para no inflar el número de nodos del SVG (28 en total). El subproblema se resuelve exacto: si el paso de Newton cabe en el disco se toma entero, y si no se minimiza el modelo sobre el borde con un barrido angular de 720 muestras. Probado con Playwright a 1180, 1366 y 1920 px, recorriendo los cuatro preajustes, sin `pageerror` ni scroll horizontal.

## Corrección de la sección de sistemas no lineales (2026-09-21)

Respaldo previo en `backup/06-Root-finding_2026-09-21c.ipynb`. Solo se tocaron las celdas 73 a 107. El total del capítulo no cambió: 114 celdas.

**El ejemplo que abre la sección es ahora el que se resuelve al final.** Se planteaba $x\log(y^2-1)=3$, $y\sin(2x^3)+e^y=2$ y después se resolvía otro sistema distinto con `fsolve`. Además ese primero no es resoluble tal como estaba: `fsolve` no converge desde seis puntos de partida, devuelve `ier` 4 o 5 con $|\mathbf{F}| \approx 2$. Se reemplazó por $x\cos y = 4$, $xy-y=5$, que es el que el capítulo ya resolvía, y la celda del final dice "volvamos al sistema que planteamos al abrir esta sección".

**Los dos ejes quedaron separados.** El texto mezclaba cómo se obtiene el Jacobiano con cuánto se avanza, y presentaba a Broyden como método de búsqueda lineal cuando es una forma de estimar $J$. Ahora la sección del criterio de minimización cierra con las dos preguntas independientes: cómo obtener $J$ (analítico, diferencias finitas, Broyden) y cuánto avanzar (paso completo, búsqueda lineal, región de confianza).

**Región de confianza reescrita.** Decía que estos métodos nacen porque calcular el Jacobiano es complicado y que la aproximación paraboloide "simplifica el cálculo del Jacobiano". Eso describe a Broyden, no a la región de confianza, y además contradecía la celda de `fsolve`, donde `hybrj` recibe el Jacobiano analítico del usuario. Ahora: la región responde la pregunta del largo del paso, fija el radio antes de calcular el paso y no después, la definición es el disco de radio $\Delta$ donde damos por válida la aproximación cuadrática, y los tres pasos del algoritmo nombran $\rho$. La celda de MINPACK aclara que `hybrd` estima el Jacobiano por diferencias finitas y `hybrj` lo recibe.

**Otros arreglos de la misma sección.** Notación $\mathbf{F}$, $\mathbf{x}$ en las definiciones, que usaban $f$ y $x$ sin negrita · "El método consiste en encontrar $\mathbf{x}_{k+1}$ a partir de la pendiente descendiente" pasa a describir el paso como solución del sistema lineal · "el Jacobiano entrega múltiples direcciones posibles" se reemplaza por el problema real, que es cuánto avanzar · $\phi = \mathbf{F}\cdot\mathbf{F}$ queda con nombre, para poder referirse a ella después · la celda `func(root)`, que imprimía un array suelto, ahora verifica $|\mathbf{F}|$ en la raíz y da $1.66\times10^{-11}$ · typos: Jacoviano, parabolide, incognitas, busqueda · la diapositiva del criterio de minimización tenía 6 bloques y se partió en dos.

Fuera de alcance por ahora, anotado en la revisión: Colebrook calcula $f_c$ pero no la caída de presión, "cuadrático" y "superlineal" solo aparecen en la tabla resumen, faltan el puente hacia adelante y la referencia al mapa, y quedan comas decimales en las celdas 39, 48 y 69.

## Broyden, largo del paso y escala de color de la animación (2026-09-21)

Respaldo previo en `backup/06-Root-finding_2026-09-21d.ipynb` y `interactive/backup/A4_region_de_confianza_2026-09-21.html`. El capítulo pasa de 114 a 116 celdas.

**Las tres estrategias de avance quedaron definidas.** La celda de las dos decisiones nombraba "paso completo", "búsqueda lineal" y "región de confianza" sin decir qué es cada una, y la lista se leía como tres sinónimos. Ahora cada una se define en la misma línea: el paso completo es $\mathbf{x}_{k+1}=\mathbf{x}_k+\Delta\mathbf{x}$; la búsqueda lineal conserva la dirección de $\Delta\mathbf{x}$ y solo la acorta a $\alpha\,\Delta\mathbf{x}$ hasta que $\phi$ baje; la región de confianza fija primero el radio y después busca el mejor paso dentro de él. El contraste que importa: la búsqueda lineal fija la dirección y busca el largo, la región fija el largo máximo y busca la dirección.

**Broyden tiene ahora su propia nota.** Aparecía nombrado y nunca explicado, y se confundía con la aproximación cuadrática de $\phi$, que pertenece a la otra pregunta. La nota dice lo que hace: le exige a la nueva matriz cumplir la condición secante $B_{k+1}\,\delta\mathbf{x} = \delta\mathbf{F}$, que es la versión en $n$ dimensiones del método de la secante, y la corrige solo en la dirección del último paso. El costo es cero evaluaciones extra de $\mathbf{F}$, porque $\delta\mathbf{F}$ ya se calculó para chequear convergencia; el precio es convergencia superlineal en vez de cuadrática. La celda de la pregunta "¿cómo obtengo $J$?" dice además que las diferencias finitas cuestan $n$ evaluaciones extra por iteración, que es el número contra el que Broyden compite.

**Mapa de color en `A4_region_de_confianza.html`.** Las curvas de nivel en un solo tono se leían como una imagen plana: no se veía hacia dónde desciende $|\mathbf{F}|^2$. Ahora el fondo es un mapa de ocho bandas de color con barra de escala rotulada a la derecha del gráfico, claro en el fondo del valle y oscuro lejos de la raíz, con las curvas de nivel encima para tapar el pixelado de las bandas. El canal pálido que cruza el mapa es el valle, y ahí se ve de inmediato por qué el paso de Newton completo se dispara fuera de él.

**Implementación del mapa.** La malla subió de 72×72 a 120×120 y se evalúa una sola vez por redibujo, compartida entre el relleno y las curvas de nivel (antes cada nivel reevaluaba $\phi$ en toda la malla). El relleno es una ruta por banda, fusionando celdas vecinas de la misma banda dentro de cada fila: 8 nodos SVG y unas 1800 subrutas en vez de 14 400 rectángulos. Como el campo solo depende de la ventana, el resultado se cachea y al iterar no se recalcula nada. La ventana se corrió a $x$ centrado en 0.52 porque la barra de color le quita ancho al gráfico y el preajuste $(-1.2,\,1.0)$ quedaba fuera; verificado a 415, 640 y 900 px de ancho que los cuatro preajustes caen dentro y que la escala sigue siendo isotrópica (130.6 px por unidad en ambos ejes a 640 px).
