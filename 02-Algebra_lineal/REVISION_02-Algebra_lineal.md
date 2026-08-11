# Revisión docente — `02-Algebra_lineal.ipynb`

> **Estado: implementado el 10-ago-2026.** Todas las correcciones y secciones nuevas descritas aquí están aplicadas en `02-Algebra_lineal.ipynb`. El notebook original quedó respaldado en `02-Algebra_lineal.BACKUP-20260810.ipynb`.
>
> **Fuera de alcance por decisión del docente:** unificar la edición de Chapra, mencionar SVD/QR, agregar ejercicios propuestos y agregar objetivos de aprendizaje.
>
> **Movido a evaluación:** las secciones de relajación (SOR), refinamiento iterativo y sistemas tridiagonales/dispersos se retiraron de la cátedra por tiempo y quedaron como base de la Tarea 1, en `2026 - Sem2/evaluaciones/tarea1/tarea1_ideas.md`. En el capítulo quedó solo una nota indicando que la LU tridiagonal es $O(n)$ y se llama algoritmo de Thomas, para dar sentido a la tabla de resumen.
>
> **Agregado en la segunda ronda:** unidades en formato `\mathrm`, definición de *flop* y deducción de la complejidad $2n^3/3 + n^2$, explicación de la matriz de prueba `+ n*np.eye(n)`, sección completa sobre por qué no se resuelve con la matriz inversa (costo, precisión y destrucción de la estructura), la figura de desplazamientos como imagen estática, y una herramienta interactiva de condicionamiento (`condicionamiento_interactivo.html`, HTML+JS puro) verificada en un build real de Jupyter Book.

**Revisado:** 10-ago-2026 · 122 celdas (~85 slides RISE) · Contrastado con Chapra & Canale *Applied Numerical Methods with MATLAB* (3rd Ed) y Press et al. *Numerical Recipes* (3rd Ed).

**Veredicto general:** la secuencia conceptual es sólida y bien pensada (problema físico motivador → representación matricial → caracterización del sistema → directos → iterativos → librería). El problema no es el *qué* sino el *cómo*: es el único capítulo de los 11 sin una sola figura generada por código, y hay un puñado de errores de signo/fórmula en celdas que los alumnos copian literalmente. Abajo, en orden de urgencia.

---

## 1. Bloqueante: el build de Jupyter Book falla

`_config.yml` tiene `execute_notebooks: force` y la **celda 54** (`print('inv(P) = ', inv(P))` con `P` singular) lanza `LinAlgError` a propósito. Ninguna celda del libro completo tiene el tag `raises-exception`, así que `jb build` aborta en este capítulo.

Provocar el error es una **buena decisión pedagógica** — no la elimines. Solo agrega el tag:

```python
"metadata": {"tags": ["raises-exception"]}
```

---

## 2. Errores técnicos a corregir

| # | Celda | Problema | Corrección |
|---|-------|----------|------------|
| 1 | **6** | Signo en la 2ª ecuación de Newton: `- k_2(x_1 - x_2)` equivale a `+k_2(x_2-x_1)`, signo invertido. Es **inconsistente con la celda 7**, que sí está correcta. | `- k_2(x_2 - x_1)` |
| 2 | **77** | Sustitución hacia atrás con coeficientes intercambiados: `x_2 = (4.25 - (-2)x_3)/5`. El valor reportado (−2.583) es correcto, el error está solo en la fórmula escrita — y es exactamente lo que el alumno transcribe. | `x_2 = (4.25 - 5·x_3)/(-2)` |
| 3 | **69** | `x_3 = (y_3' - a_{3,4}x_4)/a_{3,4}` — el denominador debe ser el elemento diagonal. | `/a_{3,3}'` |
| 4 | **58** | **"Una matriz está mal condicionada si det(A) ≈ 0"** — es falso, y es el error conceptual más costoso del capítulo. El determinante depende de la escala; el condicionamiento no. Verificado: `B = 0.01·I₁₀` tiene `det(B) ≈ 1e-20` y `cond(B) = 1.0` (perfectamente condicionada). A la inversa, `[[1,1],[1,1.0001]]` tiene `det ≈ 1e-4` y `cond ≈ 4·10⁴`. | Redefinir el mal condicionamiento *solo* vía `Cond(A)`. Chapra (§11.2, p.272) usa el determinante escalado apenas como indicador preliminar, nunca como definición. |
| 5 | **26** | La expresión $\sqrt[p]{\sum_i\sum_j\|a_{ij}\|^p}$ es la norma **entrywise**, no la *p*-norma matricial (inducida) $\max \|Ax\|_p/\|x\|_p$. Consecuencia práctica: en la celda 31 `norm(A)` devuelve **Frobenius**, pero en la celda 62 `cond(P)` usa por defecto la **norma-2 espectral** (SVD). El alumno no puede reproducir `cond` con el `norm` que se le enseñó. | Chapra §11.2.1 (p.274) es explícito: *"the 2 norm and the Frobenius norm for a matrix are not the same"*. Introduce norma de fila (∞) y norma-2, y aclara qué usa `numpy.linalg.cond`. |
| 6 | **96** | Dominancia diagonal escrita con `≥`. La condición suficiente de convergencia de Gauss-Seidel requiere desigualdad **estricta** (`>`) en al menos una fila; con `≥` débil hace falta además irreducibilidad. Misma imprecisión en el código de la celda 104. | `\|a_{ii}\| > \sum_{j\neq i}\|a_{ij}\|` |
| 7 | **74** | El "pivoteo parcial" mostrado reordena las filas como (3,1,2) — una rotación, no un intercambio. El resultado neto coincide con la `P` de la celda 89 porque hay un segundo swap después, pero no es lo que hace el algoritmo ni lo que devuelve `scipy.linalg.lu`. | Intercambiar solo filas 1↔3 → orden (3,2,1). |
| 8 | **80** | La tabla Doolittle/Crout/Choleski (tomada de Kiusalaas, Tabla 2.1) omite que Cholesky exige **A simétrica y definida positiva**. Sin eso, `L = Uᵀ` parece una opción libre. | Agregar la condición; Chapra §10.3 (p.263). |
| 9 | **25** | "El producto de la matriz inversa por **su diagonal** es igual a la matriz identidad". | "…por la matriz…" |
| 10 | **42** | Último elemento de la matriz aumentada aparece como `y_n`, debe ser `y_m`. | `y_m` |
| 11 | **13** | Llamar "linealizar" a multiplicar por $x_3$ confunde: es una manipulación algebraica exacta (válida solo si $x_3 \neq 0$), no una linealización. El término reaparecerá con otro significado en ajuste de curvas y en Newton. | Reformular como "se puede reescribir en forma lineal, siempre que $x_3 \neq 0$". |
| 12 | **95** | "Los métodos iterativos no requieren gran capacidad de memoria (LU requiere almacenar L, U y P)" — argumento débil: `L` y `U` se guardan *in-place* sobre `A` y `P` es un vector de `n` enteros. | El argumento real es el **fill-in**: en una matriz dispersa, `L` y `U` se densifican. Esto además prepara el terreno para los caps. 10 y 11. |

---

## 3. Bugs verificados en `gauss_seidel()` (celda 106)

Ejecuté la función. Cinco problemas, dos de ellos serios:

**a) Muta el argumento del usuario.** `x = x0` no copia; el `for i in range(n): x[i] = ...` escribe sobre el array del llamador.

```
x0 antes:   [1. 1. 1.]
x0 después: [ 2.0887 -1.5535 -0.6499]   ← destruido
```

Fix: `x = np.array(x0, dtype=float).copy()`.

**b) `dtype` entero → resultado silenciosamente incorrecto.** Con `x0 = np.array([1,1,1])` (sin puntos), la asignación trunca a entero en cada iteración:

```
resultado: [ 2  -1   0]        (correcto: [2.0888, -1.5534, -0.6499])
```

Sin error, sin warning. Un alumno que omita los puntos decimales obtiene basura y no tiene forma de saberlo. Esto es, de hecho, **una excelente oportunidad didáctica**: conecta directo con el cap. 01 (representación de punto flotante). Pero el código debe forzar `float`.

**c) `DeprecationWarning` en NumPy ≥ 1.25.** `y = y.reshape(-1,1)` convierte `y[i]` en un array de forma `(1,)`; asignarlo a `x[i]` está deprecado y **será un error** en versiones futuras. Los alumnos con Anaconda reciente ya ven el warning. El `reshape` existía solo para armar la matriz aumentada del `assert`, así que se resuelve junto con el punto (e): basta con eliminarlo y dejar `y` como vector 1-D.

**d) Off-by-one + crash.** `for k in range(1, max_iter)` ejecuta `max_iter - 1` iteraciones (50 → 49). Con `max_iter=1` no entra al loop y revienta:

```
UnboundLocalError: local variable 'error_abs' referenced before assignment
```

Fix: `range(max_iter)` e inicializar `error_abs = np.inf`.

**e) Eliminar el `assert matrix_rank(...)`.** Es más costoso que el propio solver: O(n³) frente a O(n²) por iteración, justo el costo que la celda 95 dice que los métodos iterativos evitan. Junto con él se pueden eliminar `matrix_rank` y `concatenate` de los imports, y las líneas que arman la matriz aumentada `Ay`. Reemplazar `n = len(y)` por `n = nrows` (ya disponible de `A.shape`).

**f)** Solo hay criterio absoluto. `norm(x - x_old) < abs_tol` depende de la escala de la solución; agregar criterio relativo.

---

## 4. Contraste con los textos base: qué falta

### 4.1 Sistemas tridiagonales y dispersos — la omisión más importante

Ninguna mención. Ambos textos base le dedican secciones completas:

- Chapra **§9.4 Tridiagonal Systems** (p.245) y su caso de estudio **§9.5 Model of a Heated Rod** (p.247)
- Numerical Recipes **§2.4 Tridiagonal and Band-Diagonal Systems** (p.56) y **§2.7 Sparse Linear Systems** (p.75)

Esto no es un detalle de completitud, es un problema de **diseño curricular**: los caps. **10 (EDO con condición de borde)** y **11 (Diferencias finitas para EDP)** de tu propio libro generan exactamente sistemas tridiagonales y dispersos. Ahora mismo el alumno llega allá con `np.linalg.solve` sobre matrices densas como única herramienta y sin entender por qué eso no escala. El algoritmo de Thomas (O(n) en vez de O(n³)) y `scipy.sparse.linalg.spsolve` cierran ese hueco, y de paso le dan sentido a la sección de métodos iterativos, que hoy queda huérfana de motivación.

### 4.2 La interpretación física de A⁻¹ — está en el texto base, con tu mismo ejemplo

Chapra §11.1 (p.272) usa **literalmente la misma matriz** `K = [150 -100 0; -100 150 -50; 0 -50 50]` de tu celda 115, y con ella enseña que cada $k^{-1}_{ij}$ es *el desplazamiento del saltador i por cada newton aplicado sobre j* — de ahí superposición y proporcionalidad. Tu notebook calcula `inv(A)` en la celda 31 de forma puramente abstracta y nunca cierra el círculo. Ganancia pedagógica alta, costo cero: el texto ya lo tiene desarrollado.

### 4.3 El significado cuantitativo del número de condición

La celda 60 dice "matrices mal condicionadas tienen Cond(A) alto" — circular y no accionable. Chapra §11.2.2 (p.275) da el resultado que un ingeniero necesita:

$$\frac{\|\Delta X\|}{\|X\|} \le \mathrm{Cond}(A)\,\frac{\|\Delta A\|}{\|A\|}$$

y la regla de bolsillo: si los coeficientes tienen $t$ dígitos y $\mathrm{Cond}(A)=10^c$, la solución es confiable a $t-c$ dígitos. **Esa frase sola justifica toda la sección.** Complementar con la matriz de Hilbert (Ejemplo 11.3, p.275): el alumno ve `cond` explotar con `n` y dígitos evaporarse.

Bonus verificable: `cond(P)` de tu matriz singular devuelve `3.76e16`, no `inf`. Mostrar eso enseña de un golpe que el rango numérico es una decisión de tolerancia, no un hecho exacto.

### 4.4 Otros vacíos frente a la bibliografía

| Tema | Fuente | Comentario |
|---|---|---|
| **Jacobi** y **relajación / SOR** | Chapra §12.1 (p.284) | Solo se cubre Gauss-Seidel. Jacobi cuesta 10 líneas y hace visible *por qué* Gauss-Seidel converge más rápido. |
| "¿Es la inversión un proceso N³?" y advertencia contra resolver vía A⁻¹ | NR §2.11 (p.106) | La celda 48 abre con *"la forma más directa de resolver es x = A⁻¹y"* sin advertir que en la práctica **nunca** se hace así (más caro y menos preciso que LU). |
| **Refinamiento iterativo** | NR §2.5 (p.61) | Barato de agregar y conecta perfecto con la celda 111 (residuo vs. error). |
| Conteo de operaciones ($\sim 2n^3/3$) | Chapra §9.2 | El notebook dice "O(N³)" pero nunca cuantifica ni compara Gauss vs. LU con múltiples RHS — que es justo el argumento de la celda 84. |

### 4.5 Residuo vs. error: oportunidad perdida en la celda 111

La celda explica bien que $\|Ax_{sol}-y\|$ no es la tolerancia del criterio de parada. Pero se queda a un paso del cierre elegante del capítulo:

$$\frac{\|x - x_{exacto}\|}{\|x\|} \le \mathrm{Cond}(A)\,\frac{\|r\|}{\|y\|}$$

Aquí el número de condición de la sección 2 reaparece explicando el resultado numérico de la sección 4. Ese tipo de retorno es lo que convierte una lista de métodos en un capítulo.

### 4.6 Referencias (celda 121)

- **No cita Numerical Recipes ni Kiusalaas**, aunque están en la bibliografía y la tabla Doolittle/Crout/Choleski proviene de Kiusalaas.
- Sugerencia: referencias por sección, no solo al final (ej. "profundizar en LU → Chapra 6ta ed., Cap. 10").

---

## 5. Diseño instruccional

**El capítulo menos visual del libro.** Revisé los 11 capítulos: este es el **único con 0 menciones de matplotlib**. 122 celdas de LaTeX y `print()`. Y está en la posición #2, donde se fija el hábito de trabajo del alumno y donde todavía se está definiendo si "esto es un ramo de matemáticas escritas en pantalla" o "esto es un ramo de experimentación numérica".

**Carga excesiva para una sesión.** ~85 slides RISE es demasiado para 90 minutos. Hay redundancia clara: el sistema de los elásticos se repite completo en las celdas 7, 14, 20 y 21. Consolidando eso y las celdas de una línea (4, 8, 9, 16, 33, 44, 55, 63, 103, 107) se recuperan 15–20 slides sin perder contenido.

**Falta una tabla de decisión de cierre.** El capítulo termina abruptamente en `scipy.linalg.lu`. Falta el resumen que responde la pregunta del ingeniero: *¿cuándo uso Gauss, cuándo LU, cuándo Cholesky, cuándo iterativo, cuándo sparse?* — en función de tamaño, estructura, simetría y número de lados derechos.

---

## 6. Seis intervenciones interactivas concretas

Ordenadas por retorno pedagógico sobre esfuerzo. Ninguna requiere dependencias nuevas más allá de `ipywidgets` (agregar a `requirements.txt`).

1. **Condicionamiento como geometría** *(el de mayor impacto)*. Slider de $\varepsilon$ en $\begin{bmatrix}1&1\\1&1+\varepsilon\end{bmatrix}$, graficando las dos rectas y la solución. El alumno **ve** dos rectas casi paralelas y entiende de inmediato por qué la intersección es incierta y por qué `cond` explota — sin una sola fórmula. Esto reemplaza la definición errónea vía determinante (§2, punto 4) por intuición correcta.

2. **`plt.spy()` de A, L y U para una matriz tridiagonal de 200×200.** Tres imágenes lado a lado muestran el **fill-in**: L y U se llenan de elementos que A no tenía. Justifica los métodos iterativos, el almacenamiento disperso, y conecta explícitamente con los caps. 10–11. Reemplaza el argumento débil de la celda 95.

3. **Convergencia de Gauss-Seidel en escala log.** $\|x^{(k)}-x^{(k-1)}\|$ vs. $k$ para: (a) matriz diagonal dominante, (b) Jacobi sobre la misma, (c) una matriz **no** diagonal dominante que diverge. Un gráfico responde "¿por qué me exigen verificar dominancia diagonal?" mejor que la celda 96.

4. **Verificación empírica de O(n³).** Cronometrar `np.linalg.solve` para n = 10…2000, graficar en log-log, ajustar la pendiente → ≈3. El alumno *mide* la complejidad en vez de creerla, y se enlaza con el cap. 01. Añadir la misma medición para `spsolve` sobre la versión dispersa: la diferencia es espectacular.

5. **Eliminación gaussiana animada.** `imshow` de la matriz aumentada paso a paso con los valores anotados y la fila pivote resaltada. Convierte las celdas 74–76 (tres bloques de LaTeX estáticos) en algo que se puede seguir. Con un slider de paso funciona igual de bien en RISE.

6. **Cerrar el problema motivador.** El capítulo abre con los tres cuerpos colgando (celda 2) y nunca los vuelve a dibujar. Un gráfico a escala con posiciones inicial y final usando el resultado de la celda 118 cierra el arco narrativo — y es la ocasión natural para agregar la interpretación de $A^{-1}$ de Chapra (§4.2).

---

## Resumen ejecutivo

**Arreglar ya:** tag `raises-exception` en la celda 54 (build roto) · signo en celda 6 · fórmula en celda 77 · definición de mal condicionamiento en celda 58 · copia de `x0` y `dtype=float` en `gauss_seidel`.

**Mayor retorno pedagógico:** agregar sistemas tridiagonales/dispersos (puente hacia los caps. 10–11) · la desigualdad de Chapra para `Cond(A)` · la interpretación física de $A^{-1}$ con el ejemplo que ya usas · las intervenciones interactivas 1, 2 y 3.

**Estructural:** tabla de decisión al cierre · consolidar redundancias para bajar de ~85 a ~65 slides.
