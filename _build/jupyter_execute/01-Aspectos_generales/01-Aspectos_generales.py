#!/usr/bin/env python
# coding: utf-8

# # Aspectos generales de programación y algoritmos

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


# In[3]:


n = 10
out = 0
for i in range(n):
    for j in range(n):
        out += i*j
        
print(out)


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

# In[4]:


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
# <img src="./imagenes/Fibonacci.png" width="800" align= center>

# Esta operación puede ejecutarse de dos maneras: (1) de forma iterativa, (2) de forma recursiva

# **(1) Forma iterativa.** complejidad $O(N)$

# In[5]:


def my_fib_iter(n):
    
    out = [1, 1]
    
    for i in range(2, n+1):
        out.append(out[i - 1] + out[i - 2])
        
    return out[-1]


# In[6]:


my_fib_iter(6)


# **(2) Forma recursiva.** complejidad $O\left(2^N\right)$

# In[7]:


def my_fib_rec(n):
    
    if n < 2:
        out = 1
    else:
        out = my_fib_rec(n-1) + my_fib_rec(n-2)
        
    return out


# In[8]:


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

# In[9]:


get_ipython().run_line_magic('time', 'a = my_fib_iter(30)')


# In[10]:


get_ipython().run_line_magic('time', 'a = my_fib_rec(30) # No probar N>30')


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

# > **Nota**. En el código binario siempre existirá un único bit reservado para el signo.

# En python el tipo de variable se asigna dinámicamente dependiendo del *input* del usuario. El número de bits depende de la versión de Python y el formato de la máquina en número de bits. Por ejemplo, para una máquina de 64 bits:

# In[11]:


type(34) # int64


# In[12]:


type(0.4e8) # float64


# ### Redondeo en variables tipo `float`

# En variables tipo `float` existe un máximo y mínimo valor que puede ser representado el cual depende del número de bits.

# En python, para determinar estos límites utilizamos la librería `sys`.

# In[13]:


import sys
print(sys.float_info)


# A continuación indicamos algunos valores útiles que podemos extraer de esta información:

# |Variable|Descripción|
# |:-:|:-:|
# |`max`|máximo valor *float* **positivo**|
# |`min*epsilon`|mínimo valor *float* **positivo**|
# |`epsilon`|diferencia entre `1` y el valor superior más pequeño representable como un *float*|
# |`dig`|máximo número de dígitos decimales que pueden ser representados en un *float*|

# Un número mayor al **máximo positivo** es prepresentado en python como `inf`:

# In[14]:


print('máximo positivo:', sys.float_info.max)
print('valor menor al máximo positivo: ', 1e+308)
print('valor mayor al máximo positivo: ', 2e+308)


# Debido a que el último bit siempre será reservado para el signo, el **mínimo valor negativo** corresponde a `- sys.float_info.max`

# In[15]:


print('valor mayor al mínimo negativo: ', -1e+308)
print('valor menor al mínimo negativo: ', -2e+308)


# Igualmente, un número menor al mínimo positivo es prepresentado en python como `0`:

# In[16]:


print('mínimo positivo:', sys.float_info.min * sys.float_info.epsilon)
print('valor menor al mínimo positivo: ', 1e-324)
print('valor mayor al mínimo positivo: ', 1e-323)


# ### Errores de redondeo en variables tipo `float`
# Las **variables del tipo ```int```** no son divisibles y, por lo tanto, **no sufren errores de redondeo**:

# In[17]:


5 - 2 == 3


# Sin embargo, una **variable del tipo ```float```** es divisible. Esto significa que existe una cantidad de dígitos significativos reservados para un número, lo que **puede inducir errores de redondeo**:

# In[18]:


0.1 + 0.2 + 0.3 == 0.6


# Para este tipo de operaciones es recomendable utilizar la función ```round```

# In[19]:


round(0.1 + 0.2 + 0.3) == round(0.6)


# ### Acumulacion de errores de redondeo
# Cuando un código ejecuta una secuencia de operaciones, los errores de redondeo suelen amplificarse.

# Si ejecutamos la siguiente operación una vez, notamos que el valor es correcto

# In[20]:


1 + 1/3 - 1/3


# Analicemos el efecto de errores de rendondeo mediante la función `add_and_substract`.

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


# ## Documentación de funciones en python

# En python existen múltiples liberías, cada una con un propósito en específico. En este curso, trabajaremos con:
# - `numpy`: Esencial para operaciones matemáticas con arreglos de elementos
# - `matplotlib`: Herramientas para visualización de datos. En este curso utilizaremos particularmente el módulo `matplotlib.pyplot`.
# - `scipy`: Funciones avanzadas para ciencia e ingeniería

# A lo largo del curso iremos revisando las principales funciones de cada librería. En esta sección explicaremos como entender como acceder a la documentación oficial de cada función, y como interpretarla.

# Para revisar la documentación de una función tenemos varias alternativas. Las más recomendadas son:
# - Documentación oficial en línea
# - Función `help`
# - Caracter `?` al final de la función
# - Comando `shift + tab`

# >**Nota.** Por defecto, la documentación en línea está en base a la **última versión de la librería**. Es posible, sin embargo, encontrar la documentación para versiones anteriores.

# Para verificar la versión de la librería que se está utilizando se debe usar el comando `library.__version__`, donde `library` corresponde a la librería en cuestion.

# In[25]:


import numpy as np
print('versión de numpy: ' + np.__version__)


# In[26]:


import matplotlib as mpl
print('versión de matplotlib: ' + mpl.__version__)


# In[27]:


import scipy as sp
print('versión de scipy: ' + sp.__version__)


# ### Documentación en línea

# Como ejercicio, consideremos la documentación de la función `linspace` de la libería `numpy`. Esta función será utilizada frecuentemente en el curso, y permite generar un arreglo de números distribuidos entre dos límites establecidos.

# In[28]:


# generar un arreglo de 5 números entre 0 y 1
np.linspace(0,1,11)


# Acá podemos ver un extracto de la documentación en línea (para la información completa ver [acá](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html))

# <img src="./imagenes/linspace_online-doc.png" width="800" align= center>

# Lo primero que vemos es el encabezado que muestra **todos los parámetros que necesita la función** (En este caso: `start`, `stop`, `num`, `endpoint`, `retstep`, `dtype` y `axis`). *La función esperará que todos los valores estén asignados, de lo contrario, arrojará un error.*

# Luego, vemos parámetros con símbolo `=` (`num = 50`, por ejemplo). Esto indica que el parámetro **tiene un valor asignado por defecto**.

# **Los parámetros con valor por defecto pueden ser omitidos al llamar a una función**. En ese caso python asumirá el valor asignado por defecto:

# In[29]:


np.linspace(0,1)  # omitimos el parámetro "num"


# El resultado es un arreglo de números igualmente espaciados entre 0 y 1, con `num = 50`.

# Más abajo en la documentación, encontramos la descripción de cada parámetro. Notar que los valores con `=` están identificados como **opcional**.
# 
# <img src="./imagenes/linspace_online-doc_parameter.png" width="800" align= center>

# Algunos parámetros no están disponibles en todas las versiones de la librería. Es el caso de `axis` de la función`linspace`, el cual está disponible desde la versión 1.16.0. en adelante.
# 
# <img src="./imagenes/linspace_online-doc_new-parameter.png" width="800" align= center>

# Por último, tenemos el *output* de la función en el encabezado "**Return**"
# 
# <img src="./imagenes/linspace_online-doc_return.png" width="800" align= center>

# Acá, por ejemplo, la documentación indica que la función entregará dos *outputs*, donde el primero corresponde al arreglo y el segundo, al espaciamiento entre valores. El segundo *output* es opcional, condicionado a la variable `retstep = True` (`False` por defecto).

# In[30]:


vector, step = np.linspace(0,1,5,retstep=True)
print('Arreglo generado: ', vector)
print('Espaciamiento: ', step)


# En python también podemos llamar a la función indicando el nombre de los parámetros de entrada

# In[31]:


np.linspace(start=0, stop=1, num=5)


# Esta última forma es más conveniente para usuarios que no conocen la función, ya que permite identificar los parámetros de entrada

# De igual forma, podemos cambiar el orden de los argumentos usando este esquema:

# In[32]:


np.linspace(num=5, stop=1, start=0)


# ###  Comandos directos para acceder a la documentación

# La opción más directa es presionando `Shift + Tab` posicionando el cursor al final de la función. Este comando depende del *IDE* utilizado, pero funciona con *Jupyter Notebook*, *Visual Studio Code* y *Google colab*.
# 
# <img src="./imagenes/shift+tab.png" width="800" align= center>

# Otra alternativa es con el comando `?` o con la función `help`
# ```python
# np.linspace?
# help(np.linspace)
# ```

# ## Referencias
# Kong Q., Siauw T., Bayen A. M. “[Python Programming and Numerical Methods – A Guide for Engineers and Scientists](https://pythonnumericalmethods.berkeley.edu/notebooks/Index.html)”, 1st Ed., Academic Press, 2021
# - Chapter 8. Complexity
# - Chapter 9. Representation of numbers
# - Calítulo 10. Errors, Good Programming Practices, and Debugging
