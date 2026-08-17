# Unidad 3 — registro de cambios (2026-08-17)

Backup del original: `backup/03-Ajuste_de_curvas_ORIGINAL_2026-08-17.ipynb`

**85 → 108 celdas · 49 slides.** Archivos nuevos: `interactive/` (3 animaciones + `embed.py`) y 4 figuras en `images/`.

---

## Correcciones de errores (F0)

| Dónde | Qué |
|---|---|
| Regresión multivariable | Matriz de ecuaciones normales con término $(2,2)$ incorrecto y signo equivocado en $S_r$ — **eliminadas** al reemplazarse por la forma general |
| Tabla del túnel de viento | Mostraba 6 puntos, el código usa 8 → tabla corregida a 8 columnas |
| Definición de $S_t$ | Se describía como "desviación estándar"; ahora "suma total de cuadrados" |
| Interpretación de $r^2$ | "explica un 88.05% *de los datos*" → "*de la varianza* de los datos" |
| `np.corrcoef` | Se indicaba `np.corrcoef(yi, y(xi))` (devuelve una matriz 2×2) → `[0,1]` y se acota la relación $r=\pm\sqrt{r^2}$ al caso lineal |
| Notación | El original usaba $n$ como último índice de dato y también como grado; unificado a $i=1,\dots,m$ datos |
| Tasa de aprendizaje | Se usa $\eta$ (no $\alpha$, ya ocupado por el modelo de potencia $y=\alpha x^\beta$) |
| Typos | túnel, algunos, unidimensional, esperábamos, estándar, específicamente, `curve_fit`, "lineal o no lineal" |

---

## Secciones nuevas

**§3 ¿Por qué el error cuadrático? (NR §15.1)** — tres criterios de error comparados en una figura; justificación por máxima verosimilitud con ruido gaussiano; cierre explícito: la pérdida cuadrática del ML es este mismo criterio.

**§4 Incertidumbre de los coeficientes (NR §15.2)** — $\sigma_{a_0}$, $\sigma_{a_1}$ con el ejemplo del túnel de viento ($a_1 = 19.47 \pm 2.93$). Hábito para informes de laboratorio.

**§6 Forma general: matriz de diseño (NR §15.4)** — modelo $y=\sum a_k\phi_k(x)$ → sistema sobredeterminado $Z\mathbf a\approx\mathbf y$ → ecuaciones normales $Z^TZ\mathbf a = Z^T\mathbf y$. Una demo con `np.linalg.lstsq`; el caso multivariable queda conceptual. Conecta con la Unidad 2.

**§8 Puente a machine learning** — tabla "modelo → pérdida → optimización" (regresión lineal / `curve_fit` / red neuronal); por qué el paraboloide admite solución cerrada y lo no lineal no; descenso de gradiente; sobreajuste y generalización.

**§9 Síntesis** — mapa de la unidad + tabla "qué herramienta uso cuándo".

También: slide "ajuste vs. interpolación" al inicio (usa `interpolacion_1D.png`, antes huérfana) y referencia a NR cap. 15 agregada.

---

## Cortes aplicados

- **C1** Derivaciones a mano de las ecuaciones normales polinomial y multivariable → casos particulares de la matriz de diseño.
- **C2** Demo en código de `curve_fit` sobre modelo lineal → una línea de texto.
- **C3** Nota log vs. ln → un solo fragment.
- **C4** Bloques repetidos de `plt.rcParams.update(...)` → se define una sola vez.

> **Punto a evaluar:** si en tus evaluaciones pides armar *a mano* el sistema 3×3 de un ajuste cuadrático, ese paso intermedio explícito ya no está en el material (queda implícito en $Z^TZ\mathbf a=Z^T\mathbf y$). Se puede reponer como un fragment si lo necesitas.

---

## Animaciones

Archivos HTML autocontenidos en `interactive/`, sin dependencias externas. Se embeben con dos líneas:

```python
from interactive.embed import animacion
animacion('A1_cazador_de_rectas.html')
```

`embed.py` lee el HTML y lo inserta como `<iframe srcdoc="...">`. Ventaja: al ejecutarse el notebook, la animación queda **embebida en la salida**, así que funciona igual en RISE (kernel vivo) y en el jupyter book publicado, sin depender de rutas relativas ni de que Sphinx copie archivos. Las celdas llevan el tag `remove-input` para que el libro muestre solo el widget.

| | Qué hace | Sección |
|---|---|---|
| **A1** Cazador de rectas | Sliders $a_0,a_1$; los residuos son cuadrados cuya área total es $S_r$. Botón "revelar óptimo" y toggle $\Sigma\|e_i\|$ | §2 (antes de derivar) |
| **A2** Descenso de gradiente | Superficie de error + recta en vivo; slider $\eta$ (chico → lento, grande → zigzag, muy grande → diverge); punto de partida elegible con clic | §8 |
| **A3** Sobreajuste | Grado 1–7; $r^2$ de ajuste sube siempre, pero el error en datos nuevos explota | §8 |

---

## Verificación

- Notebook ejecutado de punta a punta desde kernel limpio, sin errores; `execution_count` secuencial.
- Todos los números citados en el texto contrastados contra los outputs reales.
- Fórmulas de incertidumbre contraverificadas contra $\sqrt{\mathrm{diag}(\sigma^2 (Z^TZ)^{-1})}$ → coincidencia exacta.
- `lstsq` cuadrático idéntico a `polyfit` de orden 2.
- Las 10 imágenes referenciadas existen; las 3 animaciones renderizan en el HTML exportado.

## Para versionar

```bash
cd "material_catedra"
git switch -c mejora-unidad3
git add 03-Ajuste_de_curvas/
git commit -m "Unidad 3: MLE, matriz de diseño, puente a ML y animaciones interactivas"
```

Para volver atrás: `git switch main`, o copiar de vuelta el archivo de `backup/`.
