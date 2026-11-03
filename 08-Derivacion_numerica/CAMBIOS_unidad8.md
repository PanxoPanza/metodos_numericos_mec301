# Cambios Unidad 8 — Derivación numérica (2026-11-02)

Respaldo: `backup/08-Derivacion_numerica_2026-11-02.ipynb` y `.css`. 63 → 77 celdas.

## Puentes
- Introducción: puente hacia atrás con U5 (diferencia hacia adelante y curva en V) y U4 (derivar el interpolante).
- Nota puente U4: la diferencia central es la derivada de p2 por tres nodos.
- Gráfico de error contra h ahora va de 1e-10 a 1: muestra la curva en V de U5. Pendientes con polyfit en h > 1e-3: 1.05 (adelante) y 2.00 (central). Mínimos cerca de 1e-8 y 1e-6.
- Nota puente U10/U2: la segunda derivada central produce un sistema tridiagonal.
- Nota puente U7: integrar atenúa el ruido, derivar lo amplifica.
- Cierre con puente a U9: Euler = diferencia hacia adelante, Euler implícito y BDF = diferencia hacia atrás.

## Contenido nuevo
- Sección "Derivar datos medidos": péndulo grabado a 100 cuadros/s, datos sintéticos 10°·cos(ωt), l = 0.3 m, ruido normal 0.2°, semilla 0.
  - gradient: error RMS 14.3 °/s sobre amplitud 57.2 °/s (25 %). Predicción δ/(√2 h) = 14.1.
  - Submuestreo h = 0.01/0.02/0.05/0.10 s: 14.3 / 7.1 / 2.7 / 4.1 °/s.
  - savgol_filter(theta, 31, 3, deriv=1, delta=dt): 1.3 °/s (2.3 %).
- Resumen con tabla de decisión (sin mapa) y verificaciones.

## Correcciones
- `scipy.misc.derivative` (eliminada en SciPy 1.12) → `scipy.differentiate.derivative` (SciPy ≥ 1.15): sin(π/3) con 11 evaluaciones, error 1e-14.
- Definición de derivada con h (antes usaba a como punto y como paso), f⁽⁵⁾ en vez de f''', f'(0.5) en vez de f'(5), (x − x_j) → (x − x_i), índices y_j en derivadas parciales.
- Polinomio identificado como ejemplo 4.4 de Chapra y Canale (7ma Ed.).
- f-strings, rcParams único, figsize (5,3)/(5,4), un solo bloque de código por figura.
- Link roto a la U4 reemplazado por **Unidad 4**.
- CSS compartido de U3; anchos de imagen 560-600.
- Referencias a Chapra y Canale 7ma Ed. (cap. 4 y 23, verificados en el PDF) y Kong cap. 20.

## Revisión 2026-11-02 (tarde), sobre la versión local del usuario
- δ/(√2 h): deducción explícita (θ̃ = θ + e, suma de varianzas), celdas 75-76.
- Derivar datos con ruido: `savgol_filter` reemplazado por `CubicSpline` con 21 nodos fijos (cada 0.15 s) ajustado con `curve_fit` y derivado con `.derivative()` (Unidades 3 y 4). Error RMS 1.1 °/s (vs 14.3 con gradient). Nodos cada 0.1 s: 1.5; cada 0.25 s: 3.8.
- Nota con enlaces a `savgol_filter`, `make_lsq_spline` y `make_smoothing_spline` como alternativas más eficientes. Import de `savgol_filter` reemplazado por `curve_fit`. Tabla del Resumen actualizada.
- Corregido en la versión del usuario: `h_array` → `hi` (celdas 31 y 33, rompía la ejecución limpia), la curva "backward" graficaba `error_fw`, comentarios invertidos en la celda 29.
