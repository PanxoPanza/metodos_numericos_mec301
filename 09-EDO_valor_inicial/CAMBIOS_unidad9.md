# Cambios Unidad 9 — EDO valor inicial (2026-11-02)

Respaldo: `backup/09-EDO_valor_inicial_2026-11-02.ipynb` y `.css`. 189 → 106 celdas.

## Estructura
1. Introducción resumida al péndulo (ejemplo conductor). Clasificación de EDP, Navier-Stokes, placa 2D y condiciones de borde reducidas a una nota puente a U10/U11.
2. Reducción a sistema de primer orden (sin `$$`, notación y en negrita).
3. Euler explícito, precisión (local O(h²), global O(h)), estabilidad con R = 1 + hλ, Euler implícito.
4. Runge-Kutta contado según es.wikipedia (forma general b_i, c_i, a_ij; dos etapas; RK4 con la lectura de k1..k4), con puente previo a U7 (Euler/punto medio/trapecio/Simpson cuando F = F(t)). Orden y costo con el ej. 25.8 de Chapra y Canale. Tabla de Butcher (en.wikipedia): esquema, tablas, explícito vs implícito, implementación genérica `paso_rk`, métodos embebidos.
5. Multipaso: AB2 (interpolar e integrar F) y BDF2 (diferencia hacia atrás O(h²) de U8). Animación A3.
6. solve_ivp con el péndulo (una figura con t_eval y los pasos del método).
7. Estabilidad y rigidez: tabla de R(z), definición de rigidez, animación A2, λ = −1e5.
8. Cierre: péndulo sin roce, Euler vs Heun vs RK4 (energía).
9. Resumen y puente a U10 (disparo, sin adelantar el método).

## Números verificados (SciPy 1.18.1)
- Euler en dy/dt = e^{-2t}: error en t = 1 de 0.0447 (h = 0.1) y 0.0043 (h = 0.01).
- dy/dt = −20y: h = 0.02 decae, 0.1 oscila constante (z = −2), 0.12 crece.
- Ej. 25.8 (y(4) = 75.33896): órdenes 1.00, 2.02, 4.00. Con 1024 evaluaciones: Euler 9.5e-4, Heun 5.7e-6, RK4 8.0e-11.
- Tabla de Butcher de RK4 y función rk4 dan el mismo y(4) = 75.34533606 con h = 0.5.
- Péndulo m = 0.1 kg, l = 0.3 m, κ = 0.05 kg/s, θ0 = 30°, t ∈ [0, 5]: 35 pasos, 212 evaluaciones (rtol por defecto).
- λ = −1e5, y(0) = 1, t ∈ [0, 10]: RK45 302 037 pasos / 2 113 262 evaluaciones; Radau 7 / 44; BDF 56 / 160. Predicción por estabilidad: 10·1e5/3.31 ≈ 302 400 (límite real de RK45 z ≈ −3.306).
- t_eval con 51 puntos: error máximo Radau 1.5, BDF 4.4e-4.
- Péndulo sin roce, θ0 = 30°, h = 0.05, 10 s: E/E0 = 61.4 (Euler), 1.355 (Heun), 0.999 (RK4).

## Animaciones (preliminares)
- `interactive/A1_paso_runge_kutta.html` (prefijo rkp): un paso de Euler/Punto medio/Heun/RK4 con las pendientes k_i y el área de la regla equivalente.
- `interactive/A2_estabilidad_rigidez.html` (prefijo est): trayectorias de dy/dt = λ(y − cos t) y |R(z)| para Euler, RK4, Euler implícito y BDF2.
- `interactive/A3_multipaso.html` (prefijo mpa): AB2 y BDF2 paso a paso, con arranque RK4 y el cruce de BDF2 como raíz.
- Probadas con Playwright a 1180 y 820 px, cero pageerror. Falta pulido visual y prueba en RISE/proyector.

## Correcciones al material anterior
- Condición de estabilidad estaba invertida (h > 2/a), texto y código usaban h distintos, signo de Euler implícito, error global atribuido a O(h²), parámetros del péndulo inconsistentes entre texto y código, solución de un oscilador escrita con exponenciales reales, `solve_ipv`, "Randau".
- Afirmación falsa "los métodos de paso simple son ineficientes para problemas rígidos": lo que importa es explícito vs implícito.
- Afirmación "n etapas dan orden n" acotada a n ≤ 4.
- Imágenes que quedaron sin uso (no se borraron): Heun_method.png, Midpoint_method.png, stiffness.png, multistep.png, navierstokes.png/.svg.

## Revisión 2026-11-02 (tarde), sobre la versión local del usuario
- Nota de estabilidad (celda 26): queda solo que el factor de amplificación se extiende a EDOs no lineales, fuera de los contenidos del curso; la estabilidad se evalúa observando la solución. Sin ∂F/∂y ni Jacobiano.
- Cierre del péndulo: eliminada la celda del λ imaginario (|1 + iω0h|). La conclusión se lee del gráfico. 106 → 105 celdas.
- Respaldo previo: `backup/09-EDO_valor_inicial_2026-11-02_revision-usuario.ipynb`.
- Repuestas a pedido del usuario las figuras `Heun_method.png` y `Midpoint_method.png` (predictor-corrector), bajo cada método (celdas 44 y 46, ancho 500). Punto medio pasa a diapositiva propia. 105 → 107 celdas. Respaldo: `backup/09-EDO_valor_inicial_2026-11-02_antes-figuras-heun.ipynb`.

## Revisión 2026-11-03: rigidez vuelve al formato original
- Eliminada la tabla de R(z), el análisis con z = hλ y la predicción de pasos de RK45 por estabilidad.
- Restaurado el formato cualitativo original: definición, ejemplo dy/dt = −1000y + 3000 − 2000e^{−t} con su solución analítica, recomendaciones, ejemplo λ = −1e5 con RK45/Radau/BDF (302 037 / 7 / 56 pasos), gráfico de pasos y la discrepancia Radau vs BDF con t_eval.
- A2 rehecha (más simple): usa la misma EDO del ejemplo; panel del transitorio (0–0.01 s) y solución completa; un deslizador de h y dos botones (0.0015 / 0.05); tabla estable/inestable y error en t = 4. Con h = 0.05: Euler explícito y RK4 inestables, Euler implícito 9.7e-7, BDF2 4.9e-9. Respaldo: `interactive/backup/A2_estabilidad_rigidez_2026-11-02.html`.
- 106 → 108 celdas. Respaldo: `backup/09-EDO_valor_inicial_2026-11-03_antes-rigidez.ipynb`.
