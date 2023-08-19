#!/usr/bin/env python
# coding: utf-8

# # Algebra lineal y sistemas de ecuaciones lineales

# ## Introducción a los sistemas de ecuaciones lineales

# Consideremos el caso de tres personas conectada por cuerdas elásticas.
# 
# <img src="./images/bungee_man.png" width="300" align= center>

# En la primera figura (a), los tres cuerpos están en la posición inicial de forma que los elásticos están totalmente extendidos, **pero no estirados**. Definimos el cambio en la posición inicial de cada persona, como: $x_1$, $x_2$, $x_3$.
# 
# Cuando los cuerpos se dejan caer, los elásticos se extienden por la gravedad y cada cuerpo toma la posición indicada en (b).

# Analizamos el cambio en la posición de cada persona utilizando la ley de Newton:

# *Diagrama de cuerpo libre*
# 
# <img src="./images/bungee_man_solve.png" width="300" align= center>

# \begin{align*}
# m_1\frac{d^2 x_1}{dt^2} &= m_1g + k_2(x_2 - x_1) - k_1 x_1 \\
# m_2\frac{d^2 x_2}{dt^2} &= m_2g + k_3(x_3 - x_2) - k_2 (x_1 - x_2) \\
# m_3\frac{d^2 x_3}{dt^2} &= m_3g + k_3(x_2 - x_3)
# \end{align*}

# En condiciones de equilibrio:
# 
# \begin{align*}
# (k_1 + k_2)x_1\;\;\;\;\;\;\;\;\; - k_2x_2 \;\;\;\;\;\;\;\;\;\;\,&=  m_1g\\
# - k_2x_1 + (k_2 + k_3)x_2 - k_3x_3  &= m_2g\\
# - k_3x_2 + k_3x_3 &= m_3g
# \end{align*}

# En el ejemplo anterior, derivamos un sistema de ecuaciones lineales con 3 incognitas el cual podemos resolver con técnicas analíticas.

# Sin embargo, si el sistema es más grande tenemos un sistema de ecuaciones con un gran número de incognitas y debemos recurrir a métodos más eficientes para poder resolverlos. 

# Por ejemplo, un reticulado de barras, donde cada barra se puede modelar como un resorte.
# 
# <img src="./images/beam_lattice.png" width="800px" align= center>

# Este es el enfoque que utilizan los software de modelación computacional, tales como: el método de elementos finitos (FEM), métodos de los momentos (MoM), o volúmenes finitos (VEM).
# 
# <img src="./images/fem_beam_lattice.png" width="600" align= center>

# ### Definición general
# Decimos que una ecuación es lineal cuando: 
# 1. Todas sus incognitas están **únicamente** separadas por sumas o restas
# 2. El exponente de cada incognita es $1$.

# Por ejemplo,
# - $3x_1 + 4x_2 - 3 = -5x_3$ (lineal)
# 
# - $\frac{-3x_1 + x_2}{x_3} = 2$ (no es lineal, pero se puede linealizar como $ -3x_1 + x_2 -2x_3 = 0 )$
# 
# - $x_1 x_2 + x_3 = 5$ (no lineal)
# 
# - $x_1 + 3x_2 + x_3^4 = 3$ (no lineal)

# > Un sistema de ecuaciones lineales esta compuesto por más de una ecuación lineal, tal como en el ejemplo de las personas conectadas por cuerdas elásticas
# 
# \begin{align*}
# (k_1 + k_2)x_1\;\;\;\;\;\;\;\;\; - k_2x_2 \;\;\;\;\;\;\;\;\;\;\,&=  m_1g\\
# - k_2x_1 + (k_2 + k_3)x_2 - k_3x_3  &= m_2g\\
# - k_3x_2 + k_3x_3 &= m_3g
# \end{align*}

# ### Representación matricial

# Para resolver sistemas de ecuaciones lineal se utiliza la representación matricial. Esto permite la implementación computacional de los algoritmos.
# 
# Por ejemplo, consideremos una ecuación lineal en su forma general:
# 
# \begin{eqnarray*}
# \begin{array}{rcrcccccrcc}
# a_{1,1} x_1 &+& a_{1,2} x_2 &+& {\ldots}& +& a_{1,n-1} x_{n-1} &+&a_{1,n} x_n &=& y_1,\\
# a_{2,1} x_1 &+& a_{2,2} x_2 &+&{\ldots}& +& a_{2,n-1} x_{n-1} &+& a_{2,n} x_n &=& y_2, \\
# &&&&{\ldots} &&{\ldots}&&&& \\
# a_{m-1,1}x_1 &+& a_{m-1,2}x_2&+ &{\ldots}& +& a_{m-1,n-1} x_{n-1} &+& a_{m-1,n} x_n &=& y_{m-1},\\
# a_{m,1} x_1 &+& a_{m,2}x_2 &+ &{\ldots}& +& a_{m,n-1} x_{n-1} &+& a_{m,n} x_n &=& y_{m}.
# \end{array}
# \end{eqnarray*}
# 
# donde $a_{i,j}$ y $y_i$ son números reales.

# La forma matricial de esta ecuación tiene la siguiente forma:

# $$\begin{bmatrix}
# a_{1,1} & a_{1,2} & ... & a_{1,n}\\
# a_{2,1} & a_{2,2} & ... & a_{2,n}\\
# ... & ... & ... & ... \\
# a_{m,1} & a_{m,2} & ... & a_{m,n}
# \end{bmatrix}\left[\begin{array}{c} x_1 \\x_2 \\ ... \\x_n \end{array}\right] =
# \left[\begin{array}{c} y_1 \\y_2 \\ ... \\y_m \end{array}\right]$$

# O, similarmente, 
# 
# $$Ax = y,$$
# 
# donde:
# 
# $$A = \begin{bmatrix}
# a_{1,1} & a_{1,2} & ... & a_{1,n}\\
# a_{2,1} & a_{2,2} & ... & a_{2,n}\\
# ... & ... & ... & ... \\
# a_{m,1} & a_{m,2} & ... & a_{m,n}
# \end{bmatrix},\;\;x= \left[\begin{array}{c} x_1 \\x_2 \\ ... \\x_n \end{array}\right],\;\;\ y = \left[\begin{array}{c} y_1 \\y_2 \\ ... \\y_m \end{array}\right]$$ 

# De igual forma, el problema de las personas sujetas con elásticos, 
# 
# \begin{align*}
# (k_1 + k_2)x_1\;\;\;\;\;\;\;\;\; - k_2x_2 \;\;\;\;\;\;\;\;\;\;\,&=  m_1g\\
# - k_2x_1 + (k_2 + k_3)x_2 - k_3x_3  &= m_2g\\
# - k_3x_2 + k_3x_3 &= m_3g,
# \end{align*}

# se puede representar de forma matricial como:
# 
# $$\begin{bmatrix}
# k_1 + k_2 & -k_2 & 0 \\
# -k_2 & k_2 + k_3 & -k_3\\
# 0 & -k_3 & k_3
# \end{bmatrix}
# \left[\begin{array}{c} x_1 \\x_2 \\ x_3 \end{array}\right] =
# \left[\begin{array}{c} m_1g \\m_2g \\m_3g \end{array}\right]$$

# ### Repaso de matrices

# **Norma matricial.** Existen distintos tipos. La más conocida es la *p-norma*:
# 
#    $$\Vert A \Vert_{p} = \sqrt[p]{(\sum_i^m \sum_j^n |a_{ij}|^p)}$$
#    
#    Para $p = 2$, se llama *norma de Frobenius*

# **Determinante.** Se denota como $det(A)$, o $|A|$. **Solo se aplica a matrices cuadradas**.
# 
#    Por ejemplo, para una matriz $2\times2$, el determinante es:
#    
#    $$ |A| = \begin{vmatrix} a & b \\ 
#                              c & d\\ 
#             \end{vmatrix} = ad - bc,$$
# 
#    para una matrix $3\times3$:
#    
# \begin{eqnarray*}
# |A| = \begin{vmatrix}
# a & b & c \\
# d & e & f \\
# g & h & i \\
# \end{vmatrix} & = & a\begin{vmatrix}
# \Box &\Box  &\Box  \\
# \Box & e & f \\
# \Box & h & i \\
# \end{vmatrix} - b\begin{vmatrix}
# \Box &\Box  &\Box  \\
# d & \Box & f \\
# g & \Box & i \\
# \end{vmatrix}+c\begin{vmatrix}
# \Box &\Box  &\Box  \\
# d & e & \Box \\
# g & h & \Box \\
# \end{vmatrix} \\
# &&\\
# & = & a\begin{vmatrix}
# e & f \\
# h & i \\
# \end{vmatrix} - b\begin{vmatrix}
# d & f \\
# g & i \\
# \end{vmatrix}+c\begin{vmatrix}
# d & e \\
# g & h \\
# \end{vmatrix} \\ 
# &&\\
# & = & aei - afh + bfg - bdi + cdh - ceg 
# \end{eqnarray*}

# **Matriz identidad ($I$).** es una matriz cuadarada con $1$ en la diagonal, y $0$ en el resto de los elementos:
# 
# $$I = \begin{bmatrix}
# 1 & 0 & 0 \\
# 0 & 1 & 0 \\
# 0 & 0 & 1 \\
# \end{bmatrix}$$

# **Matriz inversa.** Definimos la matriz inversa de $A$ como: $A^{-1}$. 
# 
#    - Solo existe para matrices cuadradas. 
#    
#    - El producto de la matriz inversa por su diagonal es igual a la matriz identidad $A\cdot A^{-1} = I$
# 
#    - Para una matriz $2\times2$, la matriz inversa está definida por:
#    
#    $$
# A^{-1} = \begin{bmatrix}
# a & b \\
# c & d\\
# \end{bmatrix}^{-1} = \frac{1}{|A|}\begin{bmatrix}
# d & -b \\
# -c & a\\
# \end{bmatrix}$$
# 
# >La solución analítica para determinar la matriz inversa se vuelve mas complicada a medida que aumentan las dimensiones de la matriz.

# ### Representación en python
# Para representar sistemas de ecuaciones lineales en python utilizamos variables del tipo *numyp array* de la libreria `numpy`.

# Por ejemplo, para representar el sistema:
# 
# \begin{eqnarray*}
# 3x_1 + 1x_2 - 5x_3 &=& 2 \\
# -2x_1 - 2x_2 + 5x_3 &=& 5 \\
# 8x_1 + 3x_2  &=& -3 \\
# \end{eqnarray*}

# In[1]:


import numpy as np

A = np.array([[ 3,  1, -5],
              [-2, -2,  5],
              [ 8,  3,  0]])

y = np.array([[2], [5], [-3]])

print('A:\n',A)
print('\ny:\n',y)


# La librería `linalg` de `numpy` tiene funciones predefinidas para calcular la norma (`norm`), determinante (`det`), matriz inversa (`inv`). La matriz identidad (`eye`) viene desde `numpy`

# In[2]:


from numpy.linalg import norm, det, inv # norm, det y inv son parte de numpy.linalg

print('norm(A) = %.4f (Frobenius por defecto)'% norm(A))
print('det(A) = %.4f' % det(A))
print('inv(A):\n',inv(A))
print()
print('eye(3) = (Matriz identidad 3x3) \n', np.eye(3))  # eye es parte de numpy


# > Notar que `norm`, `det` y `inv` son parte de la librería `numpy.linalg`, mientras que `eye` es parte de `numpy`.

# Comprobamos la identidad $AA^{-1} = I$ usando `python`.

# In[3]:


A.dot(inv(A)) # usamos numpy.dot() para multiplicar matrices


# ## Caracterización de sistemas de ecuaciones lineales
# 

# ### Anáisis mediante el rango
# 
# Un **sistema de ecuaciones** lineales tiene **solución única**, si y solo si el **número de incognitas es igual al número de ecuaciones linealmente independientes** en el sistema.

# Por ejemplo, el siguiente sistema de ecuaciones lineales:
# 
# \begin{eqnarray*}
# 3x_1 + 1x_2 - 5x_3 &=& 2 \\
# -2x_1 - 2x_2 + 5x_3 &=& 5 \\
# 4x_1 -5x_3  &=& 9 \\
# \end{eqnarray*}
# 
# No tiene solución única, ya que: $(\mathrm{ec.~}3) = 2\times(\mathrm{ec.~}1) + (\mathrm{ec.~}2)$

# Definimos el **rango de la matriz, $\mathrm{rank}(A)$**, como el número de filas (o columnas) linealmente independenientes.

# Consideremos la matrix aumentada $[A|y]$  como:
# 
# $$[A|y] = \begin{bmatrix}
# a_{1,1} & a_{1,2} & ... & a_{1,n} & y_1\\
# a_{2,1} & a_{2,2} & ... & a_{2,n} & y_2\\
# ... & ... & ... & ... & ...\\
# a_{m,1} & a_{m,2} & ... & a_{m,n} & y_n
# \end{bmatrix}$$ 
# 
# donde $A$ es una matriz $m\times n$. Es decir, $m$ ecuaciones y $n$ incognitas.
# 
# - **El sistema tiene solución única** si $\mathrm{rank}\left([A|y]\right) = \mathrm{rank}\left(A\right)$, y $\mathrm{rank}\left(A\right) = n$
# 
# 
# - **El sistema tiene infinitas soluciones** si $\mathrm{rank}\left([A|y]\right) = \mathrm{rank}\left(A\right)$, y $\mathrm{rank}\left(A\right) < n$
# 
# 
# - **El no tiene soluciones** si $\mathrm{rank}\left([A|y]\right) = \mathrm{rank}\left(A\right) + 1$

# > En python, $\mathrm{rank}(A)$ está dado por la función `matrix_rank` de la librería `numpy.linalg`

# In[4]:


from numpy.linalg import matrix_rank
M = np.array([[ 3,  1, -5],
              [-2, -2,  5],
              [ 4,  0, -5]])
matrix_rank(M)


# En el caso del ejemplo anterior:
# 
# \begin{eqnarray*}
# 3x_1 + 1x_2 - 5x_3 &=& 2 \\
# -2x_1 - 2x_2 + 5x_3 &=& 5 \\
# 4x_1 -5x_3  &=& 9 \\
# \end{eqnarray*}

# In[5]:


# y = np.array([[2], [5], [9]])
My_aug = np.concatenate((M,y),axis = 1)
print('[M|y] =\n', My_aug)
print('\n')
print('rank(M|b) =', matrix_rank(My_aug))
print('rank(M) =', matrix_rank(M))
print('Número de incognitas, n =', M.shape[1])


# \begin{align*}
# \mathrm{rank}\left([M|y]\right) &= \mathrm{rank}\left(M\right) + 1 \\
# \mathrm{rank}\left(M\right) &\lt n
# \end{align*}
# 
#  **Por lo tanto, el sistema tiene infinitas soluciones**

# ### Matriz singular
# La forma más directa de resolver un sistema es mediante la matriz inversa: $x = A^{-1}y$. Sin embargo, la estabilidad de esta expresión dependerá si la matriz es invertible o no.

# Recordamos de álgebra lineal, que la matriz inversa está dada por 
# \begin{equation*}
# A^{-1} = \frac{1}{|A|}\mathrm{adj}(A),
# \end{equation*}
# donde $\mathrm{adj}(A)$ es la matriz adjunta.

# Si $\mathrm{det}(A) = 0$,  decimos que **la matriz es singular** y, por lo tanto, no es invertible.

# Por ejemplo, la matriz:
# 
# $$P = \begin{bmatrix}
# 1 & 2 & -1 \\
# 2 & 3 &  0 \\
# 1 & 1 &  1 \\
# \end{bmatrix},$$
# 
# es singular.

# In[6]:


P = np.array([[ 1, 2,-1],
              [ 2, 3, 0],
              [ 1, 1, 1]])
print('det(P) = ', det(P))


# Como resultado, al intentar calcular la matriz inversa, `python` nos arrojará un error.

# In[7]:


print('inv(P) = ', inv(P))


# ### Condición de una matriz
# 
# *Decimos que una matriz $A$ está **mal condicionada**, si $\mathrm{det}(A) \approx 0$.* 

# Si bien, las **matrices mal condicionadas** tienen inversa, son **numericamente problemáticas**, ya que pueden inducir errores de redondeo, *overflow* o *underflow* como resultado de la división por un número muy pequeño

# Para determinar si una matriz está mal condicionada utilizamos el **número de condición**, definido como:
# 
# $$\mathrm{Cond}(A) = \|A\|\cdot\|A^{-1}\|$$
# 
# *Matrices mal condicionadas están caracterizadas por "$\mathrm{Cond}(A)$" altos*

# En python, $\mathrm{Cond}(A)$ está dado por la función `cond` de la librería `numpy.linalg`

# In[40]:


from numpy.linalg import cond
print('Cond(P) = ',cond(P))


# Notar que $\mathrm{det}(A)= 0$, no necesariamente significa que el sistema no tiene solución

# Por ejemplo, en el ejemplo anterior
# \begin{eqnarray*}
# 3x_1 + 1x_2 - 5x_3 &=& 2 \\
# -2x_1 - 2x_2 + 5x_3 &=& 5 \\
# 4x_1 -5x_3  &=& 9 \\
# \end{eqnarray*}
# 

# In[9]:


print('M\n', M)
print('\n')
print('det(M) = ', det(M))


# Sin embargo, como habíamos determinado anteriormente, el sistema tiene múltiples soluciones.

# ## Métodos de solución directos

# ### Eliminación de Gauss
# Es un algoritmo para resolver sistemas ecuaciones lineales basado en convertir la matriz $A$ en una matriz **triangular superior**. El sistema toma la forma: 
# 
# $$\begin{bmatrix}
# a_{1,1} & a_{1,2} & a_{1,3} & a_{1,4}\\
# 0 & a_{2,2}' & a_{2,3}' & a_{2,4}'\\
# 0 & 0 & a_{3,3}' & a_{3,4}' \\
# 0 & 0 & 0 & a_{4,4}'
# \end{bmatrix}\left[\begin{array}{c} x_1 \\x_2 \\ x_3 \\x_4 \end{array}\right] =
# \left[\begin{array}{c} y_1 \\y_2' \\ y_3' \\y_4' \end{array}\right]$$

# Esta ecuación puede resolverse fácilmente, comenzando por $x_4 = y_4'/a_{4,4}'$, luego continuamos con $x_3 = \frac{y_3' - a_{3,4}x_4}{ a_{3,4}}$, y así sucesivamente hasta llegar a $x_1$. En otras palabras, utilizamos **sustitución hacia atrás**, resolviendo el sistema desde abajo hacia arriba. 

# Si $A$ es una matriz **triangular inferior**, resolveríamos el problema de arriba hacia abajo utilizando **sustitución hacia adelante.**

# La mejor forma de entender el método de eliminación Gauseana es con un ejemplo:
# 
# \begin{eqnarray*}
# 4x_1 + 3x_2 - 5x_3 &=& 2 \\
# -2x_1 - 4x_2 + 5x_3 &=& 5 \\
# 8x_1 + 8x_2  &=& -3 \\
# \end{eqnarray*}

# Paso 1: Transformamos el sistema de ecuaciones en su forma matricial $Ax=y$. 
# 
# $$
# \begin{bmatrix}
# 4 & 3 & -5\\
# -2 & -4 & 5\\
# 8 & 8 & 0\\
# \end{bmatrix}\left[\begin{array}{c} x_1 \\x_2 \\x_3 \end{array}\right] =
# \left[\begin{array}{c} 2 \\5 \\-3\end{array}\right]$$ 

# Paso 2: Determinar la matriz aumentada [A, y] 
# 
# $$
# [A, y]  = \begin{bmatrix}
# 4 & 3 & -5 & 2\\
# -2 & -4 & 5 & 5\\
# 8 & 8 & 0 & -3\\
# \end{bmatrix}$$
# <br>

# Paso 3: Determinamos la matriz triangular superior utilizando pivoteo parcial y eliminación.
# 
# - Comenzando por la primera columna. Primero, permutamos las filas de manera que el coeficiente con mayor valor absoluto quede en la primera fila:
# 
# $$
# [A, y]  = \begin{bmatrix}
# 8 & 8 & 0 & -3\\
# 4 & 3 & -5 & 2\\
# -2 & -4 & 5 & 5\\
# \end{bmatrix}$$

# - Luego, eliminamos los otros coeficientes de la primera columna, comenzando por el segundo. Multiplicamos la primera fila por $1/2$ y la restamos a la segunda fila:
# 
# $$
# [A, y]  = \begin{bmatrix}
# 8 & 8 & 0 & -3\\
# 0 & -1 & -5 & 3.5\\
# -2 & -4 & 5 & 5\\
# \end{bmatrix}$$
# 
# - Después, multiplicamos la primera fila por $- 1/4$ y la restamos a la tercera fila:
# 
# $$
# [A, y]  = \begin{bmatrix}
# 8 & 8 & 0 & -3\\
# 0 & -1 & -5 & 3.5\\
# 0 & -2 & 5 & 4.25\\
# \end{bmatrix}$$

# - Repetimos el proceso con la segunda columna. Primero, permutamos las filas:
# 
# $$
# [A, y]  = \begin{bmatrix}
# 8 & 8 & 0 & -3\\
# 0 & -2 & 5 & 4.25\\
# 0 & -1 & -5 & 3.5\\
# \end{bmatrix}$$
# 
# 
# - Luego, eliminamos el coeficiente inferior.  Multiplicamos por la segunda fila por $1/2$ y restamos a la tercera fila:
# 
# $$
# [A, y]  = \begin{bmatrix}
# 8 & 8 & 0 & -3\\
# 0 & -2 & 5 & 4.25\\
# 0 & 0 & -7.5 & 1.375\\
# \end{bmatrix}$$

# Paso 4. Realizamos sustitución hacia atras.
# 
# \begin{align*}
# x_3 &= \frac{-1.375}{7.5}=-0.183 \\
# x_2 &= \frac{4.25 - (-2)x_3}{5} = -2.583 \\
# x_1 &= \frac{-3 - 8x_2 + 0x_3}{8} = 2.208
# \end{align*}
# 
# > El método de eliminación Gaussiana es de complejidad $O(N^3)$

# ### Factorización LU
# Es posible demostrar que cualquier matriz cuadrada $A$ puede ser expresada como el producto de una matriz triangular inferor $L$, y una matriz triangular superior $U$.
# 
# $$A = LU$$

# El proceso para obtener $L$ y $U$ es conocido como *descomposición* o *factorización* LU. **Es el método de solución de ecuaciones lineales más confiable y utilizado.**

# El tipo de factorización LU no es única, ya que existen múltiples formas de representar $L$ y $U$ para un $A$ dado. Así, definimos tres tipos de factorizaciones comúnmente utilizadas:
# 
# |  Nombre   |        Condiciones        |
# |:---------:|:-------------------------:|
# | Doolittle |$L_{ii} = 1$, $i = 1, 2,... $, $n$|
# | Crout     |$U_{ii} = 1$, $i = 1, 2,... $, $n$|
# |Choleski   |$L = U^T$                  |

# Una vez ejecutada la factorización, resolvemos el sistema $Ax = y$.

# <img src="./images/LU_schematic.png" width="500" align= center>

# - Primero resolvemos el sistema $Ld = y$, por sustitución hacia adelante.
# 
# - Luego, resolvemos el sistema $Ux = d$, por sustitución hacia atrás.

# 
# > A diferencia del método de eliminación de Gauss, la factorizacion LU no depende del vector $y$. Por lo tanto, es conveniente para resolver el sistema $Ax=y$, con múltiples valores de $y$.
# 
# > Debido a que la factorización LU está basada en eliminación de Gauss, el orden de complejidad es $O(N^3)$.

# Existen diversos métodos para obtener las matrices $L$ y $U$. Uno de ellos es mediante eliminación Gaussiana.

# Como mostramos anteriormente, el método de eliminación de Gauss permite determinar una matriz triangular superior. La matriz triangular inferior, aunque no se mostró de forma explicita esta conformada por "$1$" en la diagonal, y los múltiplos utilizados para eliminar los elementos de las columnas.
# 
# En general, se puede demostrar que para una matriz $A$, se cumple la siguiente relación:
# 
# $$PA = LU$$
# 
# donde $P$ es la matriz de permutaciones.

# Por ejemplo, en el ejercicio anterior:
# \begin{eqnarray*}
# 4x_1 + 3x_2 - 5x_3 &=& 2 \\
# -2x_1 - 4x_2 + 5x_3 &=& 5 \\
# 8x_1 + 8x_2  &=& -3 \\
# \end{eqnarray*}
# 
# Tenemos:
# 
# $$
# L  = \begin{bmatrix}
# 1 & 0 & 0 \\
# -0.25 & 1 & 0 \\
# 0.5 & 0.5 & 1 \\
# \end{bmatrix};
# U  = \begin{bmatrix}
# 8 & 8 & 0 \\
# 0 & -2 & 5 \\
# 0 & 0 & -7.5 \\
# \end{bmatrix}; 
# P  = \begin{bmatrix}
# 0 & 0 & 1 \\
# 0 & 1 & 0 \\
# 1 & 0 & 0 \\
# \end{bmatrix}$$

# In[10]:


A = np.array([[ 4,  3, -5],
              [-2, -4,  5], 
              [ 8,  8,  0]])

L = np.array([[    1,   0, 0],
              [-0.25,   1, 0],
              [  0.5, 0.5, 1]])

U = np.array([[ 8,  8,   0],
              [ 0, -2,   5],
              [ 0,  0,-7.5]])

P = np.array([[0, 0, 1],
              [0, 1, 0],
              [1, 0, 0]])


# In[11]:


print('P*A =\n',np.dot(P,A))
print('\n')
print('L*U =\n',np.dot(L,U))


# ## Métodos iterativos
# Los métodos iterativos están basados en una serie repetitiva de operaciones, comenzando por un valor inicial. A diferencia de los métodos directos, el número de operaciones está condicionado por la convergencia y el valor inicial escogido.

# Las ventajas de los métodos iterativos es que tienen un orden de complejidad menor que los métodos directos, y no requieren gran capacidad de memoria (recordemos que factorización LU requiere almacenar las matrices L, U y P)

# La gran desventaja radica en la convergencia de los algoritmos. Una condición suficiente, pero no necesaria es que la matriz $A$ debe ser **diagonal dominante**, es decir, los elementos de la diagonal, $a_{i,i}$, deben satisfacer:
# 
# $$|a_{i,i}| \geq \sum_{j\neq i} |a_{i,j}|$$

# Estos métodos se utilizan, generalmente, en simulaciones con elementos finitos (FEM), o volúmenes finitos (VEM).

# ### Gauss-Seidel
# El algoritmo se puede resumir en los siguientes pasos:
# 
# Paso 1. Asumimos un valor inicial para $x_2^{(0)}, x_3^{(0)}, \cdots, x_n^{(0)}$ (con excepción de $x_1^{(0)}$).
# 
# Paso 2. Calculamos un nuevo valor para $x_1^{(1)}$ mediante:
# 
# $$
# x_1^{(1)} = \frac{1}{a_{1,1}}\Big[y_1 - \sum_{j \ne 1}^{n}{a_{1,j}x_j^{(0)}} \Big]
# $$
# 
# Paso 3. Utilizando el nuevo valor $x_1^{(1)}$ y el resto de $x^{(0)}$ (con excepción de $x_2^{(0)}$), determinamos $x_2^{(1)}$.
# 
# $$
# x_2^{(1)} = \frac{1}{a_{2,2}}\Big[y_2 - \sum_{j \ne 1,2}^{n}{a_{2,j}x_j^{(0)}}  - {a_{2,1}x_1^{(1)}}\Big]
# $$

# Paso 4. Repetimos el paso 3 hasta completar todos los elementos del vector $x$.
# 
# Paso 5. Continuamos con la iteración hasta que el valor de $x$ converge dentro de una tolerancia $\varepsilon$, definida por:
# 
#  $$\| x^{(i)} - x^{(i-1)}\| \lt \varepsilon$$

# Por ejemplo, resolvamos el siguiente sistema de ecuaciones con el métodod de Gauss-Seidel:
# 
# \begin{eqnarray*}
# 8x_1 + 3x_2 - 3x_3 &=& 14 \\
# -2x_1 - 8x_2 + 5x_3 &=& 5 \\
# 3x_1 + 5x_2 + 10x_3 & =& -8 \\
# \end{eqnarray*}
# 
# Primero, verificamos que la matriz es diagonal dominante:

# In[12]:


A = [[ 8.,  3., -3.], 
     [-2., -8.,  5.], 
      [3.,  5., 10.]]

# coeficientes de la diagonal
diagA = np.diag(np.abs(A)) 
print('diag(A) = ',diagA)

# suma de los elementos sin la diagonal
off_diagA = np.sum(np.abs(A), axis=1) - diagA
print('off_diag(A) =',off_diagA)


# In[13]:


if np.all(diagA >= off_diagA):
    print('la matriz es diagonal dominante')
else:
    print('la matriz no es diagonal dominante')


# In[14]:


def gauss_seidel(A,y,x):
    from numpy import copy
    from numpy.linalg import norm
    
    epsilon = 0.01 # tolerancia
    converged = False # verificar convergencia
    
    # guardamos el valor inicial
    x_old = copy(x)
    
    for k in range(1, 50):
        for i in range(x.size):
            x[i] = y[i]              # xi = yi
            for j in range(x.size):
                if i == j:           # saltamos i = j
                    continue
                x[i] -= A[i][j]*x[j] # xi =  yi - sum(aij*xj)
            x[i] = x[i]/A[i][i]      # xi = (yi - sum(aij*xj))/aii
    
        # comparamos el error con la tolerancia
        dx = norm(x-x_old)
        
        print("iter =",k,'; x =',x)
        if dx < epsilon:
            converged = True
            print('Converged!')
            break
        
        # guardamos el valor de x para la nueva iteracion
        x_old = copy(x)

    if not converged:
        print('No converge, incrementar número de iteraciones')
    return x


# In[15]:


import numpy as np

A = [[ 8.,  3., -3.], 
     [-2., -8.,  5.], 
      [3.,  5., 10.]]

y = np.array([14., 5., -8.])
x = np.array([0.,0.,0.]) # valores iniciales
gauss_seidel(A,y,x)


# ## Solución de sistemas de ecuaciones lineales con python
# En python la forma más facil de resolver sistemas de ecuaciones lineal es mediante la función `solve` de `numpy.linalg`. Esta función utiliza factorización LU para resolver el sistema.
# 
# Por ejemplo, tomemos el ejemplo de las personas conectadas por elásticos:
# 
# |  Persona   | Masa (kg)| Constante del resorte (N/m) | Longitud incial del elástico |
# |:----------:|:--------:|:--------:|:--------:|
# | primera    |60| 50 | 20|
# | segunda    |70| 100| 20 |
# | tercera    |80|50|20|

# Tenemos un sistema de la forma:
# 
# $$
# \begin{bmatrix}
# 150 & -100 & 0\\
# -100 & 150 & -50\\
# 0 & -50 & 50\\
# \end{bmatrix}\left[\begin{array}{c} x_1 \\x_2 \\x_3 \end{array}\right] =
# \left[\begin{array}{c} 588.6 \\686.7 \\784.8\end{array}\right]$$ 

# In[16]:


import numpy as np

A = np.array([[ 150, -100,   0], 
              [-100,  150, -50], 
              [   0,  -50,  50]])

y = np.array([588.6, 686.7, 784.8])

x = np.linalg.solve(A, y)
print(x)


# Notar que en este problema $x_1$, $x_2$ y $x_3$ representan las posiciones relativas de las personas. Así la posición final está dada por:

# In[17]:


print('Posición final de las personas: ', x + [20, 40, 60])


# Mediante la librería `scipy` podemos hacer factorización LU.

# In[18]:


from scipy.linalg import lu

P, L, U = lu(A)
print('P:\n', P)
print('L:\n', L)
print('U:\n', U)
print('LU:\n',np.dot(L, U))


# ## Referencias
# - Kong Q., Siauw T., Bayen A. M. **Chapter 14: Linear Algebra and Systems of Linear Equations** in *[Python Programming and Numerical Methods – A Guide for Engineers and Scientists](https://pythonnumericalmethods.berkeley.edu/notebooks/Index.html)*, 1st Ed., Academic Press, 2021
# 
# - Chapra S., Canale R. **Parte tres: Ecuaciones algebraicas lineales** en *Métodos Numéricos para Ingenieros*, 6ta Ed., McGraw Hill, 2011
