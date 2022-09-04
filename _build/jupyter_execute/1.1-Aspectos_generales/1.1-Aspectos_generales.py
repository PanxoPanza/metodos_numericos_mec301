#!/usr/bin/env python
# coding: utf-8

# <font size="6">MEC301 - Métodos Numéricos</font>
# # 1.1 Aspectos generales de programación y algoritmos
# <br><br><br><br>
# Profesor: Francisco Ramírez Cuevas<br>
# Fecha: 1 de Agosto 2022

# ## Complejidad de algoritmos

# ### ¿Qué es un algoritmo?
# Un algoritmo es una serie ordenada de operaciones sistemáticas que permite hacer un cálculo y hallar la solución de un tipo de problemas.

# Por ejemplo:

# In[1]:


def f(n):
    out = 0
    for i in range(n):
        for j in range(n):
            out += i*j
    return out


# La complejidad de un algoritmo es la **relación entre el tamaño del input $N$** y la **cantidad de operaciones para completarlo**.
# Una forma de determinar la complejidad del algoritmo es **contabilizar las operaciones básicas**:
# - sumas
# - restas
# - multiplicaciones
# - divisiones
# - asignación de variables
# - llamados a otras funciones

# Por ejemplo, en el siguiente algoritmo:

# In[2]:


def f(n):
    out = 0
    for i in range(n):
        for j in range(n):
            out += i*j
    return out


# El número de operaciones son:
# - sumas: $N^2$
# - restas: 0
# - multiplicaciones: $N^2$
# - divisiones: 0
# - asignación de variables: $2N^2 + N + 1$
# - llamados a otras funciones: 0

# Así, el **total de operaciónes** para completar el algoritmo es $4N^2+N+1$.

# ### Notación *Big-O*
# A medida que el tamaño de $N$ aumenta, las operaciones de mayor orden se hacen dominantes. Así, podemos decir que la complejidad del algoritmo anterior es del orden $O(N^2)$. Esta notación, denominada ***Big-O***, es comúnmente utiilzada para **determinar la complejidad del algoritmo cuando $N$ es de gran tamaño**.

# > **Nota** Un algoritmo tiene complejidad **polynomial** cuando es del tipo $O(N^c)$, donde $c$ es una constante.

# Analicemos la complejidad del siguiente algortimo:

# In[3]:


def my_divide_by_two(n):
    
    out = 0
    while n > 1:
        n /= 2
        out += 1
        
    return out


# A medida que $N$ crece podemos ver que la parte dominante de este algoritmo esta dentro de la operación ```while```.

# Si analizamos el número de iteraciones $I$ para un determinado $N$, notaremos que estos están en la relacción $N/2^I = 1$, es decir $I\approx \log N$. Así, la complejidad de este algoritmo es $O(\log N)$.

# > **Nota** Un algoritmo tiene complejidad **logaritmica** cuando es del tipo $O(\log N)$.

# ### Serie de Fibonacci y complejidad exponencial
# 
# Una operación matemática puede ser ejecutada mediante algoritmos con diferente complejidad. Por ejemplo, consideremos la serie de Fibonacci.
# 
# <img src="./imagenes/Fibonacci.jpg" width="350" align= center>

# Esta operación puede ejecutarse de dos maneras: (1) de forma iterativa, (2) de forma recursiva

# **(1) Forma iterativa.** complejidad $O(N)$

# In[4]:


def my_fib_iter(n):
    
    out = [1, 1]
    
    for i in range(2, n+1):
        out.append(out[i - 1] + out[i - 2])
        
    return out[-1]


# In[5]:


my_fib_iter(6)


# **(2) Forma recursiva.** complejidad $O\left(2^N\right)$

# In[6]:


def my_fib_rec(n):
    
    if n < 2:
        out = 1
    else:
        out = my_fib_rec(n-1) + my_fib_rec(n-2)
        
    return out


# In[7]:


my_fib_rec(5)


# > **Nota** Un algoritmo tiene complejidad **exponencial** cuando es del tipo $O(c^N)$, donde $c$ es una constante.

# ### Notación *Big-O* y tiempo de computación
# La complejidad en la notación *Big-O* nos entrega una referencia del tiempo computacional dedicado para un determinado algoritmo.

# <img src="./imagenes/08.02.01-complexity.png" width="300" align= center>

# Así, por ejemplo, si consideramos un procesador Intel i7-12700K - 5GHz *($\approx$ 5 billones de operaciones por segundo)*:
# - ```my_fib_iter(100)``` <br>tomaría $\approx$ 0.2 nanosegundos
# <br>
# - ```my_fib_recur(100)``` <br>tomaría $\approx$ 8 trillones de años

# Podemos evaluar el tiempo de ejecución con la sentencia ```%time```

# In[8]:


get_ipython().run_line_magic('time', 'a = my_fib_iter(30)')


# In[9]:


get_ipython().run_line_magic('time', 'a = my_fib_rec(30) #Nota. No probar N>30')


# > **nota** En general, se deben evitar los algoritmos de complejidad exponencial

# ## Representación binaria y errores de reondeo
# En un computador, la información es almacenada en formato binario. Un **bit** puede tener dos valores: 0 o 1.
# El computador es capaz de interpretar número utilizando códigos binarios.
# 
# Por ejemplo, el código de 8 bits $001000101$ es equivalente a:
# 
# \begin{equation*}
#  0\cdot2^7 + 0\cdot2^6 + 1\cdot2^5 + 0\cdot2^4 + 0\cdot2^3 + 1\cdot2^2 + 0\cdot2^1 + 1\cdot2^0 = 37
# \end{equation*}

# Cada variable tiene una cantidad de bits asociada.
# 
# |Tipo|Nombre|Número de bits|Rango de valores|
# |:-:|:-:|:-:|:-:|
# |bool|Boolean|1|```True``` o ```Flase```|
# |int32|Single precision integer|32|-2147483648 a 2147483647|
# |float64|Double presition float|64|$(-1)^s2^{e - 1024}\left(1 + f\right)$<br /> $s$: 1 bit; <br /> $e$: 11 bits; <br /> $f$: 52 bits|

# En python el tipo de variable se asigna dinámicamente. El número de bits depende de la versión de Python y el formato de la máquina en número de bits. Por ejemplo, para una máquina de 64 bits:

# In[10]:


type(34) #int64


# In[11]:


type(0.4e8) #float64


# Usando "Numpy", podemos controlar el número de bits asignados a una determinada variable

# In[12]:


import numpy as np
np.zeros(1, dtype='int16')


# ### Redondeo en variables tipo `float`

# En variables tipo `float`, para un determinado número de bits, existe un máximo y mínimo valor que puede ser representado. En Python, valores mayores o menores a estos son representados como `inf` o `0`, respectivamente.

# Podemos deterinar estos límites mediante la librería `sys`

# In[13]:


import sys
sys.float_info


# In[14]:


2e+308 #mayor al maximo 1.7976931348623157e+308


# In[15]:


1e-324 # menor al mínimo no normalizado 5e-324


# In[16]:


# Mínimo no normalizado
sys.float_info.min * sys.float_info.epsilon


# ### Errores de redondeo en variables tipo ```float```
# Las **variables del tipo ```int```** no son divisibles y, por lo tanto, **no sufren errores de redondeo**:

# In[17]:


5 - 2 == 3


# Sin embargo, una **variable del tipo ```float```** es divisible. Esto significa que existe una cantidad de dígitos significativos reservados para un número, lo que **puede inducir errores de redondeo**:

# In[18]:


0.1 + 0.2 + 0.3 == 0.6


# Para este tipo de operaciones es recomendable utilizar la función ```round```

# In[19]:


round(0.1 + 0.2 + 0.3) == round(0.6)


# ### Acumulacion de errores de reondeo
# Cuando un código ejecuta una secuencia de operaciones, los errores de redonde suelen amplficarse.

# In[20]:


# Si ejecutamos esta operación una vez
1 + 1/3 - 1/3


# In[21]:


def add_and_subtract(iterations):
    result = 1
    
    for i in range(iterations):
        result += 1/3

    for i in range(iterations):
        result -= 1/3
    return result


# In[22]:


add_and_subtract(100) # Si ejecutamos esta operación 100 veces


# In[23]:


add_and_subtract(1000) # Si ejecutamos esta operación 1000 veces


# In[24]:


add_and_subtract(10000) # Si ejecutamos esta operación 10000 veces


# ## Identificación de errores y debugging
# Cuando los códigos de programación son grandes, a veces es necesario utlizar herramientas de ***debugging***. Estas herramientas nos permiten revisar las distintas etapas dentro de un algoritmo.
# 
# Podemos llamar al debugger agregando mediante la librería *python debugger* ```pdb```.

# Por ejemplo, consideremos la siguiente función

# In[25]:


def square_number(x):
    sq = x**2
    sq += x
    
    return sq


# In[26]:


square_number('10')


# Agregando la sentencia ```%pdb on``` antes de llamar la función podemos analizar el codigo y detectar posibles fuentes de error.

# In[27]:


# llamamos al debugger de python
get_ipython().run_line_magic('pdb', 'on')
square_number('10')


# In[28]:


# detenemos el debugger
get_ipython().run_line_magic('pdb', 'off')


# También podemos agregar *breakpoints* en distintas líneas de código para detener el *debugger*. 

# In[29]:


import pdb
def square_number(x):
    
    pdb.set_trace() # agregamos un 1er breakpoint
    sq = x**2
    
    pdb.set_trace() # agregamos un 2do breakpoint
    
    sq += x
    
    return sq


# In[31]:


square_number(3)


# Algunos comandos útiles de ```pdb```:
# - ```help```: lista de todos los comandos del debugger
# - ```h ``` *#comando*: detalle del funcionamiento de un comando en específico
# - ```a``` o ```args```: muestra el valor del argumento de la función
# - ```p```: imprime el valor de una expresión específica. Usar ```locals()``` para mostrar valor de variables locales
# - ```pdb.trace()```: agrega un *breakpoint* (pausa en el código)
# - ```continue```: continua con el código despues de un *breakpoint*
# - ```quit```: finaliza el debugger.

# ## Referencias
# Kong Q., Siauw T., Bayen A. M. “[Python Programming and Numerical Methods – A Guide for Engineers and Scientists](https://pythonnumericalmethods.berkeley.edu/notebooks/Index.html)”, 1st Ed., Academic Press, 2021
# - Capitulo 8 (Complejidad de algoritmos)
# - Capítulo 9 (Representación binaria y errores de redondeo)
# - Calítulo 10 (Identificación de errores y debugging)
