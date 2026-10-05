# Cambios — Unidad 7 · Integración numérica (2026-10-04)

Plan y decisiones: `PLAN_unidad7.md`. Respaldos: `backup/07-Integracion_2026-10-04.ipynb`, `backup/07-Integracion_2026-10-04.css`, `backup/GUIA_FORMATO_2026-10-04.md`.

**87 → 109 celdas.** Estructura conservada: datos tabulados (Newton-Cotes) y funciones conocidas (Gauss, adaptativa).

## Contenido
- Celda de librerías nueva (`skip` + `remove-input`). Los `import` sueltos se eliminaron.
- Introducción: ejemplo conductor del trabajo de una fuerza variable (Chapra y Canale, caso 24.4; 7 datos cada 5 m) y puente hacia atrás a U4 ($p_n$) y U5 (resto de Taylor).
- Newton-Cotes presentado como integrar $p_n$ (grado 0, 1, 2). Trapecio con el error de Chapra $E_t=-\frac{1}{12}f''(\xi)h^3$ y su forma compuesta. Simpson con la notación de Lagrange de U4 y la celda "exacta hasta grado 3".
- Trabajo con los datos: trapecio 119.09 J (8.1 %), Simpson 117.13 J (9.6 %) contra la referencia 129.52 J. Simpson sale peor: los datos no resuelven los máximos.
- Figura log-log nueva: pendientes 2 y 4 del trapecio y Simpson con $\sin x$.
- Funciones conocidas: `humps` (Chapra MATLAB, ej. 20.5) como ejemplo conductor, valor exacto 29.85832539549867.
- Gauss: animación A2 antes de la derivación. Nodos renombrados a $t_i$. Puente a U6 (sistema no lineal). Cambio de variable con la forma de Chapra. "Exacta hasta grado $2n-1$".
- Comparación Gauss vs. trapecio con $\sin x$ y `humps`: con 10 puntos Gauss sigue sobre 1 % de error en `humps`.
- Richardson nuevo, en versión mínima (4 celdas), solo como instrumento para la adaptativa: error estimado 1.18e-2 contra real 1.21e-2 (trapecio, 16 y 32 subintervalos, `humps`).
- Cuadratura adaptativa reescrita: pasos de Chapra §22.3 y las dos diferencias de `quad` (Gauss-Kronrod 21, estrategia global). Animación A1. `full_output` con `humps`: 5 subintervalos, 189 evaluaciones, $21(2L-1)$.
- `quad`: argumento `points` agregado. Impropias con nota del cambio de variable.
- `## Resumen` nuevo (tabla de decisión + verificaciones + puente a U9). Sin mapa.
- Referencias: Chapra y Canale 6ta Ed. (cap. 21, 22, 24, sin números de sección), Chapra MATLAB cap. 20, Kong (enlace corregido al cap. 21), Numerical Recipes §4.1, 4.3, 4.6, 4.7.

## Errores corregidos
`np.arange` que dejaba fuera el último tramo · $h$ del texto distinto al del código · alias `trapz`, `simps`, `np.trapz` (ya no existen) · comportamiento obsoleto de `simpson` con número par de puntos · producto de Lagrange desde $j=1$ · `$$` con `&=&` · índices $c_0,c_1$ vs. $c_1,c_2$ · "$c_0=c_1=h/2$" · intervalos al revés y $dx$ faltantes · `SyntaxWarning` por `'\int'` sin raw string · `%`-formatting · `quad` llamado dos veces · "Determine" (usted).

Celdas originales eliminadas: 15, 27, 28, 30, 32, 45, 47, 70 (fusionadas o reemplazadas).

## Animaciones (prefijos `qad`, `gau`)
- `interactive/A1_cuadratura_adaptativa.html`: conmutador Gauss-Kronrod 21 (`quad`) vs. Simpson + Richardson (Chapra), funciones `humps`, pico angosto y $\sin(1/x)$ en [0.02, 1], tolerancia 1e-2 a 1e-10. **Verificado**: en los 27 casos (3 funciones × 9 tolerancias) la animación da el mismo número de subintervalos, de evaluaciones y el mismo valor que `scipy.integrate.quad(f, a, b, epsabs=tol, epsrel=tol)` (SciPy 1.18.1). El modo Simpson coincide en los 27 casos con el algoritmo de Chapra fig. 22.5. Replica la regla de QUADPACK que rechaza la primera estimación cuando el error queda saturado (sin eso el pico angosto convergía mal con tol 1e-2). No incluye la extrapolación de Wynn: por eso no hay funciones singulares entre los preajustes.
- `interactive/A2_trapecio_a_gauss.html`: dos puntos arrastrables, recta por ellos, área que sobra y que falta. Reproduce el ejemplo 22.3 de Chapra: trapecio 0.172800, Gauss 1.822578 (11.1 %).
- Probadas con Playwright a 1366 px con DOM de Jupyter, con y sin `rise-enabled`: cero `pageerror`, `#ipython-main-app` en zoom 1.

## Pendiente
- `images/adaptive_quadrature.png` quedó sin uso. No se borró.
- CSS compartido adoptado: revisar anchos de imagen en el proyector.
- git (en terminal de Windows): `git add 07-Integracion/` incluyendo el `.ipynb` **y** los `.html`.

# Segunda ronda (2026-10-05), sobre la versión editada por el usuario (111 → 117 celdas)

- Trapecio: vuelve la deducción original por Taylor (expansión de primer orden + derivada aproximada por la pendiente entre los dos puntos). Nota nueva: esa expansión es la recta $p_1$, lo que la conecta con la presentación de Newton-Cotes como "integrar el interpolante".
- **El error se escribe solo como $O(h^n)$.** Se eliminaron $E_t=-\frac{1}{12}f''(\xi)h^3$, el término con $\bar f''$ y el $-\frac{1}{90}f^{(4)}h^5$ de Simpson.
- Criterio para saber si los datos resuelven la función: comparar la integral con paso $h$ y $2h$. Trabajo: 119.09 → 124.98 lb·ft, cambio de 4.9 %.
- Ejemplo del trabajo en las unidades de Chapra y Canale §24.4 (ft, lb, lb·ft).
- Órdenes del trapecio y Simpson obtenidos con `polyfit` en escala log-log: pendientes 2.01 y 4.04.
- Gauss: $x\in[-1,1]$ con nodos $x_i$ (como la Tabla 1) y $t\in[a,b]$ solo en el cambio de variable. A2 pasó a $x$.
- Polinomio de grado 5 identificado: Chapra y Canale, ejemplos 21.1 y 22.3, en la nota bajo la animación A2 y en el notebook.
- Gauss vs. trapecio separado en dos figuras: $\sin x$ (lo que se gana) y `humps` (Gauss no basta → adaptativa).
- "pico" → "pulso" en el texto y en A1.
- Referencias de Chapra según la 7ma Ed. (2015), con números de sección verificados en `referencias/`. `humps` citada como función de gibas (problemas 18.27 y 25.23). Se quitó el libro de MATLAB.
- Respetado lo que el usuario eliminó: puente a U6 en Gauss, nota de `neval`, puente a U9.
