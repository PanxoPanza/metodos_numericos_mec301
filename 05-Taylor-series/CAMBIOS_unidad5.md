# Cambios — Unidad 5 · Series de Taylor

**Fecha:** 2026-09-20 · **Plan:** `PLAN_unidades_5_6.md`
**Respaldo del original:** `backup/05-Taylor-series_2026-09-20.ipynb` (31 celdas)
**Resultado:** 69 celdas.

## Por qué

La unidad era un anexo de 31 celdas, sin introducción, sin resumen y sin los dos objetos que el resto del curso le pide prestado: el resto de Taylor $R_N$ y la forma en paso $h$. Las Unidades 6 a 9 se apoyan en ella y no los encontraban escritos en ninguna parte.

## Estructura nueva

```
# Series de Taylor
   celda de librerías                          [skip, remove-input]   NUEVA
## Introducción                                                       NUEVA
   el péndulo · puente atrás a la Unidad 4 · qué vamos a poder hacer
## Expansión de funciones en series de Taylor
## Aproximación de funciones no polinomiales
   serie truncada · gráfico corregido · A1
## El resto de Taylor                                                 NUEVA
## La forma en paso h                                                 NUEVA
## Errores de truncamiento
## Errores de redondeo
## El error total y el paso óptimo                                    NUEVA
   A2
## Resumen: ¿qué me llevo de Taylor?                                  NUEVA
## Referencias                                                        ampliada
```

## Qué se agregó

- **Ejemplo conductor: el péndulo.** La linealización $\sin\theta\approx\theta$ abre el capítulo y se cierra con el número: la aproximación cuesta menos de 1 % de error hasta los 14°, y 4,7 % a los 30°. Figura nueva `images/pendulo.png`.
- **El resto de Taylor en forma de Lagrange**, con $\xi$ declarado, y su uso como cota. Aplicado al péndulo: la cota $\theta^3/3!$ predice el error real con menos de 4 % de margen.
- **La forma en paso $h$**, $f(x_i+h) = f(x_i) + f'(x_i)h + \cdots + O(h^{N+1})$, con la definición de orden de un método numérico. Es la forma que usan las Unidades 6 a 9.
- **El error total y el paso óptimo** (Chapra §4.4): la curva en V del error de $(f(x+h)-f(x))/h$, con las dos asíntotas. Verificado: mínimo $3\times10^{-9}$ en $h\approx10^{-8}$, contra la predicción $\sqrt{\varepsilon_\mathrm{maq}}=1{,}5\times10^{-8}$.
- **Diagnóstico y cura del ejemplo de $e^{-30}$.** El capítulo mostraba que fallaba, sin explicar por qué ni cómo arreglarlo. Ahora se identifica el término más grande de la serie ($7{,}76\times10^{11}$ en $i=29$, cuyo error de máquina es $1{,}7\times10^{-4}$) y se muestra la cura: $1/\sum 30^i/i!$ acierta en todas las cifras.
- **Resumen, tabla de decisión, verificaciones y mapa** `images/mapa_unidad5.png`.
- **Dos animaciones** en `interactive/` (ver más abajo).
- **Cuatro notas puente**: hacia atrás a la Unidad 4 e hipervínculo a la Unidad 1; hacia adelante a las Unidades 6, 7, 8 y 9.

## Qué se corrigió

| # | Qué decía | Qué dice |
|---|---|---|
| 1 | "para el orden 8 la aproximación es casi perfecta" | El código recorría `range(4)`, es decir órdenes 1, 3, 5 y 7. Ahora el gráfico muestra hasta el orden 9 y **calcula e imprime el error máximo de cada curva**: 3,1 · 2,0 · 0,52 · 0,075 · 0,0069 |
| 2 | `$$ ... \begin{aligned}` (celda 3) | `align*`, según §5.1 de la guía |
| 3 | `\begin{equation}` numerado (celda 1) | `equation*` |
| 4 | `%`-formatting en tres celdas de código | f-strings |
| 5 | Imports dispersos, `plt.rcParams` dentro de una celda de gráfico | Celda de librerías al inicio, `skip` + `remove-input`, con `plt.rcParams` una sola vez |
| 6 | *Analisis, susesiva, incluído, erronea* | Corregido |

## Corte aplicado

**C6** — la expansión de $5x^2+3x+5$ en $a=0$ y en $a=2$ ocupaba 5 celdas. Queda en 2, conservando el corolario de que la expansión de Taylor de un polinomio es el mismo polinomio.

## Animaciones

`interactive/`, contrato técnico de §8.1 de la guía (fragmento HTML sin iframe, CSS y JS scopeados, SVG con `viewBox`, tipografía adaptativa).

- **A1_taylor_bajo_la_lupa** (prefijo `tay`) — orden $N$, punto de expansión $a$, cuatro funciones y un punto de evaluación arrastrable. Lee el error real junto a la cota del resto. El caso $1/(1-x)$ en $x=1{,}2$ muestra que subir el orden **empeora** el resultado fuera del radio de convergencia.
- **A2_h_optimo** (prefijo `err`) — curva en V del error total contra $h$ en log-log, con las asíntotas de truncamiento y redondeo dibujadas, y el $h$ óptimo marcado.

## Otros archivos

- `05-Taylor-series.css` reemplazado por el compartido de `03-Ajuste_de_curvas/`. **Consecuencia:** ahora las imágenes con atributo `width` se escalan por `--rise-image-scale: 1.6` en RISE. Los anchos nuevos (`pendulo.png` 330, `mapa_unidad5.png` 700) se eligieron pensando en el apunte, pero conviene revisarlos en el proyector.
- `images/pendulo.png` y `images/mapa_unidad5.png`, nuevas.

## Verificación

- El notebook corre completo desde un kernel limpio, sin errores (`nbclient`).
- Las figuras generadas se revisaron una por una.
- Las animaciones se probaron con Playwright a 1180 px y 820 px, sin `pageerror`.
- Todas las cifras del texto se contrastaron contra la salida real del notebook.

---

## Revisión del 2026-09-21

Tras la revisión del usuario (69 → 56 celdas; su versión respaldada en `backup/05-Taylor-series_2026-09-21_revision-usuario.ipynb`):

- Comentario `#` explicando `np.finfo(float).eps*abs(t)` en la celda del diagnóstico de redondeo, variable renombrada a `resolucion` y print rotulado "Resolución de ese término" en vez de "última cifra significativa".
- Celda nueva que explica qué es esa resolución: un `float` guarda ~16 cifras y el salto entre representables es relativo, $\varepsilon_\mathrm{maq}|t|$; en $7.76\times10^{11}$ da $1.72\times10^{-4}$.
- Separador decimal con punto en todo el notebook.
- **Corregido un defecto introducido en la revisión**: en la celda del péndulo el `zip` pasó a `(theta_deg, cota, error_abs, error_rel)` pero los nombres del `for` quedaron como estaban, así que la columna "R2" imprimía el error real y "error abs." la cota. Con `for td, c, ea, er` la cota vuelve a salir por encima del error en las seis filas.

Pendiente de decisión del usuario: al sacar la celda de la aproximación lineal en notación de paso se fueron los dos puentes hacia adelante, y la unidad ya no menciona la Unidad 6. `images/mapa_unidad5.png` quedó sin uso.

## Escalado de animaciones en RISE (2026-09-21)

`05-Taylor-series.css` (compartido con U3, U4 y U6) escala las animaciones en modo presentación con `--rise-anim-scale: 1.45` y `--rise-anim-max: 820px`, aplicados por `zoom` a los elementos con `id` terminado en `-app`. Las dos animaciones pasaron además a medir el ancho del SVG con `clientWidth`, que es inmune al `zoom`; con `getBoundingClientRect()` el texto del SVG quedaba clavado en 13 px mientras el resto crecía. Detalle completo en §8.1.1 de `GUIA_FORMATO.md`. Respaldos en `backup/` e `interactive/backup/`.

**Corrección del 2026-09-21 (misma sesión):** la primera versión de esa regla usaba `div[id$="-app"]` sin acotar, que también calza con `#ipython-main-app`, el contenedor de todo el notebook en Jupyter clásico. En RISE eso agrandaba la página completa y la dejaba en una columna angosta. La regla quedó limitada a `div.output_area` / `div.output_subarea` / `.jp-OutputArea-output`.

**Leyenda de A2 (2026-09-21).** `A2_h_optimo.html` ganó una leyenda con muestras de línea para las cuatro curvas: error total medido, truncamiento ∝ h, redondeo ∝ ε/h y h óptimo. Va **bajo el gráfico**, como fila en HTML con `border-top` sólido o punteado en cada muestra, no dentro del SVG: la curva en V y el aspa de las asíntotas no dejan ninguna zona libre que sobreviva al cambio de función, y una caja fija terminaba tapando la rama descendente. Al ir en `em`, escala sola con el zoom de RISE. El pie dejó de repetir los colores y ahora dice lo que la leyenda no puede. La etiqueta verde dentro del gráfico y la entrada de la leyenda usan el mismo nombre, "h óptimo", para que el lector las una. Respaldo previo en `interactive/backup/A2_h_optimo_2026-09-21b.html`.

---

## Refinamiento del 2026-09-21 (sobre la reestructuración del usuario)

Punto de partida: la versión del usuario con las secciones agrupadas en subsecciones, 47 celdas. Respaldo en `backup/05-Taylor-series_2026-09-21c.ipynb`. Resultado: 48 celdas.

**`slide_type` reasignados** (10 celdas). Al mover celdas durante la reagrupación quedaron con el tipo que tenían antes, y en RISE eso juntaba cinco bloques y dos encabezados en una misma diapositiva, con `## Análisis de errores` y `## Reflexiones finales` colgando del final de la sección anterior. Ahora son 18 diapositivas, ninguna sobre el presupuesto de título + 2 a 4 fragmentos. Los dos encabezados `##` abren su diapositiva llevando como primer fragmento el contenido que les sigue, para no dejar diapositivas de solo título.

**$O(h^n)$ aclarado.** Tres precisiones, sin sección nueva: la celda que introduce la notación ahora dice que agrupa los términos descartados y por qué basta la potencia más baja; una nota nueva advierte que describe el error con $h \to 0$ y $N$ fijo, que no dice nada para un $h$ fijo como el $h=2$ del ejemplo de $e^2$, y que una aproximación de orden $N$ tiene error $O(h^{N+1})$; y la definición del residuo cierra con $R_N = [f^{(N+1)}(\xi)/(N+1)!]\,h^{N+1}$, que es de donde sale el $O(h^{N+1})$.

**Causa del error de redondeo corregida.** La celda decía que la falla venía de que `x**i/math.factorial(i)` es división exacta entre enteros grandes, y que el error superaba una resolución de $\sim 10^{-14}$. Las dos cosas estaban mal orientadas: esa división exacta es justamente lo que hace que cada término sea correcto (error relativo 0.0), y el piso real es $\varepsilon_\mathrm{maq}\cdot 7.8\times10^{11} = 1.7\times10^{-4}$, que es $1.8\times10^{9}$ veces el valor buscado. Ahora la celda atribuye la falla a la cancelación al acumular, con esos números.

**Residuo vs. cota.** La celda de $e^2$ anunciaba "la cota es" y escribía una fórmula con $\xi$ adentro; ahora escribe $|R_N| \le e^2 h^{N+1}/(N+1)!$. Las columnas de las dos tablas se llamaban `residuo` y `R2` e imprimían la cota: ahora dicen `cota`. El código del ejemplo usa `h**n` en vez de `x**n`, coherente con el `h` que ya definía, y `np.exp(x)` en vez de `np.exp(2)`.

**Detalles menores.** Punto sobrante antes de la ecuación del péndulo · la pregunta de la introducción vuelve a conectar con la herramienta en vez de prometer la respuesta · la lectura de A1 dice dónde arrastrar el punto ($x = \pm\pi$) para leer el error máximo · el resto del péndulo explica por qué se usa $R_2$ en una aproximación de orden 1 · "expasión" → "expansión".

**Referencias colgantes eliminadas**, por la decisión de dejar fuera la sección del error total: la pregunta 4 de Reflexiones finales, que hablaba del $h$ óptimo, y las sub-viñetas §4.4 de Chapra y §5.7 de Numerical Recipes. `interactive/A2_h_optimo.html` queda en la carpeta sin uso.
