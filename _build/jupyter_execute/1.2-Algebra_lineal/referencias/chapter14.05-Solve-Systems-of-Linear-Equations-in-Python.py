#!/usr/bin/env python
# coding: utf-8

# <!--BOOK_INFORMATION-->
# <img align="left" style="padding-right:10px;" src="images/book_cover.jpg" width="120">
# 
# *This notebook contains an excerpt from the [Python Programming and Numerical Methods - A Guide for Engineers and Scientists](https://www.elsevier.com/books/python-programming-and-numerical-methods/kong/978-0-12-819549-9), the content is also available at [Berkeley Python Numerical Methods](https://pythonnumericalmethods.berkeley.edu/notebooks/Index.html).*
# 
# *The copyright of the book belongs to Elsevier. We also have this interactive book online for a better learning experience. The code is released under the [MIT license](https://opensource.org/licenses/MIT). If you find this content useful, please consider supporting the work on [Elsevier](https://www.elsevier.com/books/python-programming-and-numerical-methods/kong/978-0-12-819549-9) or [Amazon](https://www.amazon.com/Python-Programming-Numerical-Methods-Scientists/dp/0128195495/ref=sr_1_1?dchild=1&keywords=Python+Programming+and+Numerical+Methods+-+A+Guide+for+Engineers+and+Scientists&qid=1604761352&sr=8-1)!*

# <!--NAVIGATION-->
# < [14.4 Solutions to Systems of Linear Equations](chapter14.04-Solutions-to-Systems-of-Linear-Equations.ipynb)  | [Contents](Index.ipynb) | [14.6 Matrix Inversion](chapter14.06-Matrix-Inversion.ipynb) >

# # Solve Systems of Linear Equations in Python

# Though we discussed various methods to solve the systems of linear equations, it is actually very easy to do it in Python. In this section, we will use Python to solve the systems of equations. The easiest way to get a solution is via the *solve* function in Numpy.
# 
# **TRY IT!** Use numpy.linalg.solve to solve the following equations. 
# 
# \begin{eqnarray*}
# 4x_1 + 3x_2 - 5x_3 &=& 2 \\
# -2x_1 - 4x_2 + 5x_3 &=& 5 \\
# 8x_1 + 8x_2  &=& -3 \\
# \end{eqnarray*}

# In[1]:


import numpy as np

A = np.array([[4, 3, -5], 
              [-2, -4, 5], 
              [8, 8, 0]])
y = np.array([2, 5, -3])

x = np.linalg.solve(A, y)
print(x)


# We can see we get the same results as that in the previous section when we calculated by hand. Under the hood, the solver is actually doing a LU decomposition to get the results. You can check the help of the function, it needs the input matrix to be square and of full-rank, i.e., all rows (or, equivalently, columns) must be linearly independent. 
# 
# **TRY IT!** Try to solve the above equations using the matrix inversion approach. 

# In[2]:


A_inv = np.linalg.inv(A)

x = np.dot(A_inv, y)
print(x)


# We can also get the $L$ and $U$ matrices used in the LU decomposition using the scipy package. 
# 
# **TRY IT!** Get the $L$ and $U$ for the above matrix A. 

# In[3]:


from scipy.linalg import lu

P, L, U = lu(A)
print('P:\n', P)
print('L:\n', L)
print('U:\n', U)
print('LU:\n',np.dot(L, U))


# We can see the $L$ and $U$ we get are different from the ones we got in the last section by hand. You will also see there is a **permutation matrix** $P$ that returned by the *lu* function. This permutation matrix record how do we change the order of the equations for easier calculation purposes (for example, if first element in first row is zero, it can not be the pivot equation, since you can not turn the first elements in other rows to zero. Therefore, we need to switch the order of the equations to get a new pivot equation). If you multiply $P$ with $A$, you will see that this permutation matrix reverse the order of the equations for this case. 
# 
# **TRY IT!** Multiply $P$ and $A$ and see what's the effect of the permutation matrix on $A$. 

# In[4]:


print(np.dot(P, A))


# <!--NAVIGATION-->
# < [14.4 Solutions to Systems of Linear Equations](chapter14.04-Solutions-to-Systems-of-Linear-Equations.ipynb)  | [Contents](Index.ipynb) | [14.6 Matrix Inversion](chapter14.06-Matrix-Inversion.ipynb) >
