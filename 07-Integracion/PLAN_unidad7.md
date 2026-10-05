# Plan de mejora — Unidad 7 · Integración numérica

**Estado actual:** `07-Integracion/07-Integracion.ipynb`, 87 celdas, sin animaciones, sin celda de librerías, sin `## Resumen`.
**Estado propuesto:** ~100 celdas. Se conserva la estructura básica: (1) datos tabulados → Newton-Cotes, (2) función conocida → Gauss y cuadratura adaptativa.
**Textos guía contrastados:** Chapra & Canale, *Métodos Numéricos para Ingenieros* 7ma Ed., **cap. 21** (Newton-Cotes) y **cap. 22** (integración de ecuaciones: §22.2 Romberg, §22.3 cuadratura adaptiva, §22.4 Gauss, §22.5 impropias). Chapra, *Applied Numerical Methods with MATLAB* 3rd Ed., **cap. 19-20** (§19.9 caso de estudio trabajo, §20.4 y ej. 20.5 con la función `humps`).
**Verificado con:** SciPy 1.18.1 y NumPy 2.x en el contenedor (2026-10-04).

---

## 0. Respaldo y flujo

1. `07-Integracion/backup/07-Integracion_2026-10-XX.ipynb` antes de tocar nada.
2. Rama (correr en terminal de Windows): `git checkout -b mejora-unidad7`.
3. Editar celda por celda sobre el archivo en disco. Registrar en `CAMBIOS_unidad7.md`.
4. Adoptar el CSS compartido de `03-Ajuste_de_curvas/` (el actual `07-Integracion.css` es de 2023 y no escala animaciones en RISE). Revisar anchos de imagen después.

---

## 1. Diagnóstico

### 1.1 Lo que funciona y se conserva

- La división datos tabulados / función conocida. Es la misma de Chapra (cap. 21 vs. cap. 22) y es la decisión que el estudiante enfrenta en la práctica.
- La derivación de Gauss de dos puntos por coeficientes indeterminados (igual a Chapra §22.4.1-22.4.2).
- La comparación de error trapecio vs. Gauss contra número de puntos (celdas 69-70).
- Los parámetros prácticos de `quad`: `epsabs`, `epsrel`, `limit`, intervalos infinitos. Los valores por defecto están correctos (1.49e-8 y 50).

### 1.2 Errores de contenido y de código (corrección obligatoria)

| # | Celda | Problema | Corrección |
|---|---|---|---|
| E1 | 26 | `np.arange(a,b,h)` excluye $b$: integra 99 tramos y deja fuera el último. Error 6.6e-4; con `linspace(a,b,101)` es 1.6e-4 | `np.linspace` |
| E2 | 25-27 | El texto dice $h=\pi/10$, el código usa $\pi/100$, y la celda 27 compara con $h^2 = 9.87\times10^{-2}$ (que es el de $\pi/10$) | Alinear texto, código y cota |
| E3 | 21, 28, 38 | Alias `trapz`, `simps` y `np.trapz` ya **no existen** (eliminados en SciPy ≥1.14 y NumPy 2). Verificado: `AttributeError` | Borrar las menciones. `np.trapezoid` sí existe |
| E4 | 41, 47 | Describe `simpson` con número par de puntos como "promedio de dos variantes". Eso era antes de SciPy 1.11. Ahora aplica la corrección de Cartwright en el último tramo y el argumento `even` desapareció. La conclusión numérica sobrevive (10/11/12 pts: 7.5e-4 / 1.1e-4 / 3.4e-4) pero la explicación no | Reescribir. Corregir también "castigo por considerar un método de orden de truncamiento mayor" (es al revés) |
| E5 | 31 | Producto de Lagrange desde $j=1$ (debe ser $j=0$). Notación $L(x)$, $P_i(x)$ contradice la U4, que usa $p_n(x)$ y bases $L_i(x)$ | Usar la notación de U4 |
| E6 | 32 | `$$` + `aligned` con `&=&` (estilo eqnarray). Viola la regla KaTeX del curso | `align*` |
| E7 | 51-56 | El texto usa $c_0,c_1,x_0,x_1$ y el sistema $c_1,c_2,x_1,x_2$ | Unificar en $c_0,c_1$ |
| E8 | 52 | "$c_0=c_1=h/2$" en $[-1,1]$ con $h$ sin definir | $c_0=c_1=1$ (Chapra: $(b-a)/2$ en general) |
| E9 | 60-61 | Cambio de variable con signo menos. Válido por simetría, pero distinto a Chapra (22.29) | $x = \frac{b+a}{2} + \frac{b-a}{2}t$ |
| E10 | 70 | `'$\int_0^\pi ...$'` sin `r''`: `SyntaxWarning` en Python ≥3.12 | raw string, f-string |
| E11 | 74 | $\epsilon = I_1 - I_2$ sin valor absoluto. "Determine" (usted) | $\lvert I_1-I_2\rvert$, primera persona plural |
| E12 | 7, 12, 13 | Intervalo escrito al revés $[x_{i+1},x_i]$. Integrales sin $dx$ (13, 17, 18) | Corregir |
| E13 | 26, 45, 79, 82 | `%`-formatting. Celda 79 llama `quad` dos veces | f-strings, una llamada |
| E14 | — | No hay celda de librerías: los `import` se repiten y `plt.rcParams` se fija en la celda 70 | Celda de librerías `skip` + `remove-input` |
| E15 | 86 | El enlace de Kong apunta al capítulo 17 (interpolación). Chapra citado en 6ta Ed.; en `referencias/` está la 7ma (mismos números de capítulo) | Corregir enlace, decidir edición (la guía §14 dice 6ta) |

### 1.3 Contraste con Chapra: lo que falta o se presenta distinto

1. **Origen de Newton-Cotes.** Chapra (21.1) define todas las fórmulas como *integrar el polinomio interpolante* $\int p_n(x)\,dx$. El curso deriva Riemann y trapecio por Taylor + diferencia hacia adelante, y Simpson por Lagrange. Son dos caminos distintos para lo mismo. Propuesta: una sola idea (integrar $p_n$ de la U4, grado 0, 1, 2, 3) y Taylor (resto $R_N$ de la U5) solo para el orden del error. Es exactamente el engranaje U4→U7 y U5→U7 de la guía §1.2.
2. **Error con su constante.** Chapra da $E_t = -\frac{1}{12}f''(\xi)(b-a)^3$ (trapecio simple) y $E_a = -\frac{b-a}{12}h^2\bar f''$ (compuesto). Dice algo que $O(h^2)$ no dice: el trapecio es exacto para rectas y el error depende de la curvatura. Es la misma ecuación que después usa Richardson.
3. **Por qué Simpson gana un orden.** Chapra lo marca como objetivo de estudio (PT6.3, obj. 3): Simpson 1/3 integra exacto una cúbica aunque use una parábola. El curso remite a un enlace externo. Basta una celda.
4. **Simpson 3/8** (§21.2.3). Resuelve el número impar de segmentos, que es justamente el problema que la celda 37 deja planteado.
5. **Extrapolación de Richardson / Romberg** (§22.2). No está. Es la pieza que justifica el paso "tenemos dos estimaciones $I_1$, $I_2$" de la cuadratura adaptativa, que hoy aparece sin explicación. `scipy.integrate.romberg` fue eliminada (SciPy 1.15), así que va como idea + 5 líneas de código, no como función de librería.
6. **Cuadratura adaptativa.** Chapra §22.3 usa Simpson en dos niveles + Richardson ($I_2 + (I_2-I_1)/15$) y recursión local. `quad` hace otra cosa (ver §3). El curso no distingue ninguna de las dos.
7. **Por qué Gauss no basta.** Con $\sin x$ Gauss parece perfecto. Con la función `humps` de Chapra (MATLAB ej. 20.5), `fixed_quad` sobre $[0,1]$ con 8 puntos todavía yerra en 1.34 (exacto: 29.858). Ese contraste es lo que motiva la adaptatividad, y hoy falta.
8. **Integrales impropias** (§22.5). El curso las resuelve con `quad(..., -inf, inf)` sin decir cómo. Una nota: `quad` mapea el intervalo infinito a $(0,1]$ con un cambio de variable, igual que la ec. 22.33 de Chapra.
9. **Omitido a propósito** (no agrega al hilo): fórmulas abiertas (§21.4), integrales múltiples (§21.5), Newton-Cotes de orden superior (Boole y más).

### 1.4 Encadenamiento

- **Los puentes U4→U7 y U5→U7 ya no existen en esos notebooks.** Verificado hoy: ni `04-Interpolacion.ipynb` ni `05-Taylor-series.ipynb` (49 celdas) mencionan la Unidad 7 ni la integración. El catálogo §3.4 de la guía los da por escritos: está desactualizado. La U7 tiene que construir sus propios puentes hacia atrás.
- Falta el puente U7→U9 (pendiente en el Anexo C): Euler = Riemann, Heun = trapecio, RK4 = Simpson cuando $dy/dt = f(t)$.
- Puente nuevo posible U7→U6: las 4 ecuaciones para $c_0,c_1,x_0,x_1$ de Gauss son un **sistema no lineal**. Reducción a un problema canónico.
- Puente U7→U8: Richardson reaparece en derivación numérica (Chapra §23.2). Hoy la U8 no lo tiene.

---

## 2. Flujo propuesto (misma estructura, celdas reordenadas o reescritas)

| Bloque | Celdas actuales | Propuesta | Δ |
|---|---|---|---|
| Título + librerías | 0 | + celda de librerías | +1 |
| `## Introducción` | 1-2 | Conservar la imagen. Agregar el ejemplo conductor (§4). Puente atrás: "en la **Unidad 4** aproximamos datos por $p_n(x)$. Integrar $p_n$ es fácil. En la **Unidad 5** vimos que el término descartado fija el error". Pregunta eje: ¿tenemos datos o la función? | +2 |
| `## Newton-Cotes` · idea | 3-5 | $\int_a^b f\,dx \approx \int_a^b p_n\,dx$. Grado 0, 1, 2, 3 → Riemann, trapecio, Simpson 1/3, 3/8 | 0 |
| Riemann | 6-15 | Comprimir la derivación de error (10 → 6 celdas) | −4 |
| Trapecio | 16-24, 28 | Fórmula integrando $p_1$. Error con constante de Chapra. Cortar 28 (`np.trapz`) | −1 |
| Simpson 1/3 | 29-47 | Notación U4, `align*`, celda "exacta para cúbicas". Reescribir 40-41 y 46-47 con el comportamiento actual de `simpson` | −2 |
| Simpson 3/8 | — | Fórmula, error, cuándo se usa | +3 |
| Orden verificado | — | Una figura log-log del error vs. $h$ para trapecio y Simpson, con pendientes 2 y 4 | +2 |
| `## Funciones conocidas` · motivación | 48-49 | El costo se mide en **evaluaciones de $f$** (eje Costo) | 0 |
| Richardson | — | Dos estimaciones $I(h)$, $I(h/2)$ → estimación del error y valor mejorado $\frac43 I(h/2) - \frac13 I(h)$. Tabla de Romberg mencionada, no desarrollada. Puente a U8 | +5 |
| Gauss | 50-70 | Índices, cambio de variable de Chapra, "exacta hasta grado $2n-1$", puente U6. Tabla 1 como tabla markdown. Comparación trapecio vs. Gauss **con `humps`**: Gauss no converge rápido con un pico | +2 |
| Cuadratura adaptativa | 71-79 | Reescribir: idea de Chapra (dos estimaciones, error, subdividir) → qué hace `quad` → **animación A1** → `full_output` con `humps` (partición real: 5 subintervalos, 189 evaluaciones). Cortar `adaptive_quadrature.png` (la animación la reemplaza) | +4 |
| Impropias y parámetros | 80-85 | Nota del cambio de variable. f-strings | 0 |
| `## Resumen` | — | Tabla de decisión (justificada: hay varias familias): datos equiespaciados → `simpson` · datos desiguales → `trapezoid` · función suave → `quad` (o `fixed_quad` si se repite mucho) · pico conocido → `quad(..., points=[...])` · intervalo infinito → `quad` con `np.inf`. Verificaciones. Puente U9. Sin mapa | +3 |
| `## Referencias` | 86 | Capítulos y secciones según §14 de la guía | 0 |

**Total aproximado:** 87 → ~100 celdas.

**No se crea `## Integración en Python` aparte:** las funciones de SciPy siguen junto a cada método, como hoy. Moverlas cambiaría la estructura básica.

---

## 3. Animación A1 — Cuadratura adaptativa de `quad`

**Archivo:** `07-Integracion/interactive/A1_cuadratura_adaptativa.html`, prefijo `qad`.

### 3.1 Qué hace realmente `quad` (y en qué difiere de Chapra)

`quad` llama a QUADPACK (rutina QAGS para intervalo finito):

1. En cada subintervalo evalúa una regla de **Gauss-Kronrod de 21 puntos**: Gauss de 10 puntos y Kronrod de 21 comparten los 10 nodos de Gauss, así que las dos estimaciones cuestan 21 evaluaciones y no 31.
2. El error del subintervalo se estima a partir de $\lvert K_{21}-G_{10}\rvert$ con una corrección empírica de QUADPACK, no con la diferencia cruda.
3. **Estrategia global**: divide en dos el subintervalo con el *mayor error estimado* de toda la partición, no recorre recursivamente como Chapra.
4. Para cuando $\sum E_k \le \max(\texttt{epsabs},\ \texttt{epsrel}\,\lvert I\rvert)$ o al llegar a `limit` subintervalos.
5. Si detecta una singularidad, acelera con extrapolación (algoritmo ε de Wynn).

Costo exacto: `neval` $= 21\,(2L-1)$ con $L$ subintervalos finales.

### 3.2 Verificación ya hecha (contenedor, 2026-10-04)

Implementé en Python Gauss-Kronrod 21 con el estimador de QUADPACK + bisección del peor subintervalo (QAG sin extrapolación) y comparé la partición con `quad(..., full_output=1)` (`alist`, `blist`):

| función | tolerancia | `quad`: subint. / evals | réplica | ¿misma partición? |
|---|---|---|---|---|
| `humps` en [0,1] | 1e-3 / 1e-6 / 1.49e-8 | 3/105 · 3/105 · 5/189 | igual | sí |
| pico $e^{-((x-0.3)/0.01)^2}$ en [0,1] | idem | 5 · 6 · 7 | igual | sí |
| $\sin(1/x)$ en [0.05,1] | idem | 4 · 5 · 6 | igual | sí |
| $\sqrt{x}$, $\ln x$, $\lvert x-1/3\rvert$ | 1e-6 | 6 · 6 · 5 | 10 · 21 · 10 | **no** (ahí entra la extrapolación) |

Conclusión: para funciones suaves con picos, la animación puede reproducir `quad` **exactamente**. Las singulares quedan fuera de los preajustes, o se rotulan como "sin extrapolación".

### 3.3 Diseño

- **Gráfico (SVG):** $f(x)$, la partición actual sombreada por subintervalo, cada uno coloreado según su error estimado (escala log, barra de color a la derecha). En el subintervalo recién dividido, los 21 nodos de Kronrod como puntos (los 10 de Gauss marcados distinto).
- **Controles:** preajuste (`humps`, pico angosto, $\sin(1/x)$), tolerancia (deslizador 1e-2 … 1e-10), botones **Paso**, **Hasta converger**, **Reiniciar**.
- **Opcional — conmutador de regla base:** "Simpson + Richardson (Chapra §22.3)" vs. "Gauss-Kronrod 21 (`quad`)". Medido con `humps`: a 1e-4 Simpson adaptivo usa 23 subintervalos y 93 evaluaciones, `quad` 3 y 105. A 1e-6, 66 y 265 contra 5 y 189. Muestra que lo caro es evaluar $f$ y que la regla de alto orden gana cuando se pide precisión.
- **Panel de lecturas:** subintervalos $L$, `neval`, $I$ estimada, $\sum E_k$, tolerancia efectiva, error real (valor exacto de `humps`: 29.85832539549867), insignia convergido / no.
- **Contrato:** fragmento sin iframe, IIFE, SVG con `viewBox`, `ResizeObserver`, ancho con `clientWidth`, variables de tinta para tema oscuro, leyenda bajo el gráfico, `<clipPath>` para la curva, verificar `</script>` de cierre.
- **Prueba:** exportar desde SciPy las particiones de cada preajuste a JSON y compararlas contra la animación (node con shim de DOM, o Playwright en el contenedor). Cero `pageerror` a 1180 y 820 px.

### 3.4 Enunciado (primera persona plural)

Antes: "Avancemos paso a paso con `humps` y observemos dónde se concentran los subintervalos. Bajemos la tolerancia y contemos cuántas evaluaciones de $f$ cuesta cada cifra extra."
Después: "En la animación, vemos que..." (la partición se refina solo cerca de los picos, el error estimado acota al real).

---

## 4. Ejemplo conductor (decisión del usuario)

Hoy toda la unidad usa $\int_0^\pi \sin x\,dx = 2$. Sirve para verificar órdenes, pero es tan suave que oculta el problema que resuelve la adaptatividad. Opciones:

- **(a) Dos casos, uno por bloque.** Datos: trabajo de una fuerza tabulada cada 5 m (Chapra MATLAB §19.9, $W=\int F\cos\theta\,dx$, 7 datos). Su lección es útil: con datos gruesos, más orden no da mejor resultado. Función: `humps` (Chapra MATLAB ej. 20.5), valor exacto conocido. **Recomendado.**
- **(b) Solo `humps`**, muestreado como "datos" en el primer bloque. Más simple, menos ingenieril.
- **(c) Radiación de cuerpo negro:** espectro tabulado (datos) y función de Planck con intervalo infinito ($\int_0^\infty x^3/(e^x-1)\,dx = \pi^4/15$). Cubre impropias, pero exige agregar un archivo de datos.

$\sin x$ se queda como caso de verificación de órdenes.

---

## 5. Notación a fijar (Anexo A de la guía)

| símbolo | significado | alcance |
|---|---|---|
| $I$, $I(h)$, $E(h)$ | integral exacta, estimación con paso $h$, su error (Chapra) | local |
| $n$, $h=(b-a)/n$ | número de segmentos, paso | $h$ global |
| $p_n(x)$, $L_i(x)$ | interpolante y base de Lagrange | heredado de U4 |
| $t_i$, $c_i$ | nodos y pesos de Gauss en $[-1,1]$ | local. $t_i$ y no $x_i$, para no chocar con los nodos de la malla |
| $G_{10}$, $K_{21}$ | estimaciones de Gauss y Kronrod | local |

---

## 6. Decisiones del usuario (2026-10-04)

1. Ejemplo conductor: opción (a). Trabajo de una fuerza tabulada (Chapra MATLAB §19.9) para datos y `humps` para funciones.
2. Richardson entra, en versión sencilla y solo como instrumento para introducir la cuadratura adaptativa. Sin tabla de Romberg.
3. Simpson 3/8 no entra.
4. La animación A1 lleva el conmutador Simpson + Richardson (Chapra) vs. Gauss-Kronrod 21 (`quad`).
5. Entra la segunda animación A2: arrastrar los dos puntos del trapecio hasta llegar a Gauss.
6. Referencias: se mantiene Chapra 6ta Ed. Es la que tienen los estudiantes y el contenido no cambia de forma significativa entre ediciones.
