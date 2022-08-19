#!/usr/bin/env python
# coding: utf-8

# <!--BOOK_INFORMATION-->
# <img align="left" style="padding-right:10px;" src="images/book_cover.jpg" width="120">
# 
# *This notebook contains an excerpt from the [Python Programming and Numerical Methods - A Guide for Engineers and Scientists](https://www.elsevier.com/books/python-programming-and-numerical-methods/kong/978-0-12-819549-9), the content is also available at [Berkeley Python Numerical Methods](https://pythonnumericalmethods.berkeley.edu/notebooks/Index.html).*
# 
# *The copyright of the book belongs to Elsevier. We also have this interactive book online for a better learning experience. The code is released under the [MIT license](https://opensource.org/licenses/MIT). If you find this content useful, please consider supporting the work on [Elsevier](https://www.elsevier.com/books/python-programming-and-numerical-methods/kong/978-0-12-819549-9) or [Amazon](https://www.amazon.com/Python-Programming-Numerical-Methods-Scientists/dp/0128195495/ref=sr_1_1?dchild=1&keywords=Python+Programming+and+Numerical+Methods+-+A+Guide+for+Engineers+and+Scientists&qid=1604761352&sr=8-1)!*

# <!--NAVIGATION-->
# < [14.3 Systems of Linear Equations](chapter14.03-Systems-of-Linear-Equations.ipynb)  | [Contents](Index.ipynb) | [14.5 Solve Systems of Linear Equations in Python](chapter14.05-Solve-Systems-of-Linear-Equations-in-Python.ipynb) >

# # Solutions to Systems of Linear Equations

# Consider a system of linear equations in matrix form, $Ax=y$, where $A$ is an $m \times n$ matrix. Recall that this means there are $m$ equations and $n$ unknowns in our system. A **solution** to a system of linear equations is an $x$ in ${\mathbb{R}}^n$ that satisfies the matrix form equation. Depending on the values that populate $A$ and $y$, there are three distinct solution possibilities for $x$. Either there is no solution for $x$, or there is one, unique solution for $x$, or there are an infinite number of solutions for $x$. This fact is not shown in this text.
# 
# **Case 1: There is no solution for $x$.** If ${rank}([A,y]) = {rank}(A) + 1$, then $y$ is
# linearly independent from the columns of $A$. Therefore $y$ is not in the range of $A$ and by definition, there cannot be an $x$ that satisfies the equation. Thus, comparing rank($[A,y]$) and rank($A$) provides an easy way to check if there are no solutions to a system of linear equations.
# 
# **Case 2: There is a unique solution for $x$.** If ${rank}([A,y]) = {rank}(A)$, then $y$ can be written as a linear combination of the columns of $A$ and there is at least one solution for the matrix equation. For there to be only one solution, ${rank}(A) = n$ must also be true. In other words, the number of equations must be exactly equal to the number of unknowns.To see why this property results in a unique solution, consider the following three relationships between $m$ and $n: m < n, m = n$, and $m > n$. 
# 
# * For the case where $m < n$, ${rank}(A) = n$ cannot possibly be true because this means we have a "fat" matrix with fewer equations than unknowns. Thus, we do not need to consider this subcase.
# * When $m = n$ and ${rank}(A) = n$, then $A$ is square and invertible. Since the inverse of a matrix is unique, then the matrix equation $Ax = y$ can be solved by multiplying each side of the equation, on the left, by $A^{-1}$. This results in $A^{-1}Ax = A^{-1}y\rightarrow Ix = A^{-1}y\rightarrow x = A^{-1}y$, which gives the unique solution to the equation.
# * If $m > n$, then there are more equations than unknowns. However if ${rank}(A) = n$, then it is possible to choose $n$ equations (i.e., rows of A) such that if these equations are satisfied, then the remaining $m - n$ equations will be also satisfied. In other words, they are redundant. If the $m-n$ redundant equations are removed from the system, then the resulting system has an $A$ matrix that is $n \times n$, and invertible. These facts are not proven in this text. The new system then has a unique solution, which is valid for the whole system.
# 
# **Case 3: There is an infinite number of solutions for $x$.** If ${rank}([A, y]) = {rank}(A)$, then $y$ is in the range of $A$, and there is at least one solution for the matrix equation. However, if rank($A$) $<$ $n$, then there is an infinite number of solutions. The reason for this fact is as follows: although it is not shown here, if rank($A$) $<$ $n$, then there is at least one nonzero vector, $n$, that is in the null space of $A$ (Actually there are an infinite number of null space vectors under these conditions.). If $n$ is in the nullspace of $A$, then $An = 0$ by definition. Now let $x^{{\ast}}$ be a solution to the matrix equation $Ax = y$; then necessarily, $Ax^{{\ast}} = y$. However, $Ax^{{\ast}} + An = y$ or $A(x^{{\ast}} + n) = y$. Therefore, $x^{{\ast}} + n$ is also a
# solution for $Ax = y$. In fact, since $A$ is a linear transformation, $x^{{\ast}} + \alpha n$ is a solution for any real number, $\alpha$ (you should try to show this on your own). Since there are an infinite number of acceptable values for $\alpha$, there are an infinite number of solutions for the matrix equation.
# 
# In the rest of the chapter, we will only discuss how we solve a systems of equations when it has unique solution. We will discuss some of the common methods that you often come across in your work in this section. And in the next section, we will show you how to solve it in Python. 
# 
# Let's say we have n equations with n variables, $Ax=y$, as shown in the following:
# 
# $$\begin{bmatrix}
# a_{1,1} & a_{1,2} & ... & a_{1,n}\\
# a_{2,1} & a_{2,2} & ... & a_{2,n}\\
# ... & ... & ... & ... \\
# a_{n,1} & a_{n,2} & ... & a_{n,n}
# \end{bmatrix}\left[\begin{array}{c} x_1 \\x_2 \\ ... \\x_n \end{array}\right] =
# \left[\begin{array}{c} y_1 \\y_2 \\ ... \\y_n \end{array}\right]$$

# ## Gauss Elimination Method
# 
# The **Gauss Elimination** method is a procedure to turn matrix $A$ into an **upper triangular** form to solve the system of equations. Let's use a system of 4 equations and 4 variables to illustrate the idea. The Gauss Elimination essentially turning the system of equations to:
# 
# $$\begin{bmatrix}
# a_{1,1} & a_{1,2} & a_{1,3} & a_{1,4}\\
# 0 & a_{2,2}' & a_{2,3}' & a_{2,4}'\\
# 0 & 0 & a_{3,3}' & a_{3,4}' \\
# 0 & 0 & 0 & a_{4,4}'
# \end{bmatrix}\left[\begin{array}{c} x_1 \\x_2 \\ x_3 \\x_4 \end{array}\right] =
# \left[\begin{array}{c} y_1 \\y_2' \\ y_3' \\y_4' \end{array}\right]$$
# 
# By turning the matrix form into this, we can see the equations turn into:
# 
# \begin{eqnarray*}
# \begin{array}{}
#  a_{1,1} x_1 &+& a_{1,2} x_2 & + & a_{1,3} x_{3} &+&a_{1,4} x_4 &=& y_1,\\
# & & a_{2,2}' x_{2} &+ & a_{2,3}' x_{3} &+& a_{2,4}' x_4 &=& y_{2}' \\
# && & & a_{3,3}' x_{3} &+& a_{3,4}' x_4 &=& y_{3}',\\
# && && && a_{4,4}' x_4 &=& y_{4}'.
# \end{array}
# \end{eqnarray*}
# 
# We can see by turning into this form, $x_4$ can be easily solved by dividing both sides $a_{4,4}'$, then we can back substitute it into the 3rd equation to solve $x_3$. With $x_3$ and $x_4$, we can substitute them to the 2nd equation to solve $x_2$. Finally, we can get all the solution for $x$. We solve the system of equations from bottom-up, this is called **backward substitution**. Note that, if $A$ is a lower triangular matrix, we would solve the system from top-down by **forward substitution**.
# 
# Let's work on an example to illustrate how we solve the equations using Gauss Elimination. 
# 
# **TRY IT!** Use Gauss Elimination to solve the following equations. 
# 
# \begin{eqnarray*}
# 4x_1 + 3x_2 - 5x_3 &=& 2 \\
# -2x_1 - 4x_2 + 5x_3 &=& 5 \\
# 8x_1 + 8x_2  &=& -3 \\
# \end{eqnarray*}
# 
# Step 1: Turn these equations to matrix form $Ax=y$. 
# 
# $$
# \begin{bmatrix}
# 4 & 3 & -5\\
# -2 & -4 & 5\\
# 8 & 8 & 0\\
# \end{bmatrix}\left[\begin{array}{c} x_1 \\x_2 \\x_3 \end{array}\right] =
# \left[\begin{array}{c} 2 \\5 \\-3\end{array}\right]$$
# 
# Step 2: Get the augmented matrix [A, y] 
# 
# $$
# [A, y]  = \begin{bmatrix}
# 4 & 3 & -5 & 2\\
# -2 & -4 & 5 & 5\\
# 8 & 8 & 0 & -3\\
# \end{bmatrix}$$
# 
# Step 3: Now we start to eliminate the elements in the matrix, we do this by choose a **pivot equation**, which is used to eliminate the elements in other equations. Let's choose the first equation as the pivot equation and turn the 2nd row first element to 0. To do this, we can multiply -0.5 for the 1st row (pivot equation) and subtract it from the 2nd row. The multiplier is $m_{2,1}=-0.5$. We will get
# 
# $$
# \begin{bmatrix}
# 4 & 3 & -5 & 2\\
# 0 & -2.5 & 2.5 & 6\\
# 8 & 8 & 0 & -3\\
# \end{bmatrix}$$
# 
# Step 4: Turn the 3rd row first element to 0. We can do something similar, multiply 2 to the 1st row and subtract it from the 3rd row. The multiplier is $m_{3,1}=2$. We will get
# 
# $$
# \begin{bmatrix}
# 4 & 3 & -5 & 2\\
# 0 & -2.5 & 2.5 & 6\\
# 0 & 2 & 10 & -7\\
# \end{bmatrix}$$
# 
# Step 5: Turn the 3rd row 2nd element to 0. We can multiple -4/5 for the 2nd row, and add subtract it from the 3rd row. The multiplier is $m_{3,2}=-0.8$. We will get
# 
# $$
# \begin{bmatrix}
# 4 & 3 & -5 & 2\\
# 0 & -2.5 & 2.5 & 6\\
# 0 & 0 & 12 & -2.2\\
# \end{bmatrix}$$
# 
# Step 6: Therefore, we can get $x_3=-2.2/12=-0.183$. 
# 
# Step 7: Insert $x_3$ to the 2nd equation, we get $x_2=-2.583$
# 
# Step 8: Insert $x_2$ and $x_3$ to the first equation, we have $x_1=2.208$. 
# 
# **Note!** Sometimes you will have the first element in the 1st row is 0, just switch the first row with a non-zero first element row, then you can do the same procedure as above. 
# 
# We are using "pivoting" Gauss Elimination method here, but you should know that there is also a "naive" Gauss Elimination method with the assumption that pivot values will never be zero. 

# ## Gauss-Jordan Elimination Method
# 
# Gauss-Jordan Elimination solves the systems of equations using a procedure to turn $A$ into a diagonal form, such that the matrix form of the equations becomes
# 
# $$\begin{bmatrix}
# 1 & 0 & 0 & 0\\
# 0 & 1 & 0 & 0\\
# 0 & 0 & 1 & 0 \\
# 0 & 0 & 0 & 1
# \end{bmatrix}\left[\begin{array}{c} x_1 \\x_2 \\ x_3 \\x_4 \end{array}\right] =
# \left[\begin{array}{c} y_1' \\y_2' \\ y_3' \\y_4' \end{array}\right]$$
# 
# Essentially, the equations become:
# 
# \begin{eqnarray*}
# \begin{array}{}
# x_1 &+& 0 & + & 0 &+&0 &=& y_1',\\
# 0 &+& x_2 & + & 0 &+&0 &=& y_2' \\
# 0 &+& 0 & + & x_3 &+&0 &=& y_3',\\
# 0 &+& 0 & + & 0 &+&x_4 &=& y_4'.
# \end{array}
# \end{eqnarray*}
# 
# Let's still see how we can do it by using the above example. 
# 
# **TRY IT!** Use Gauss-Jordan Elimination to solve the following equations. 
# 
# \begin{eqnarray*}
# 4x_1 + 3x_2 - 5x_3 &=& 2 \\
# -2x_1 - 4x_2 + 5x_3 &=& 5 \\
# 8x_1 + 8x_2  &=& -3 \\
# \end{eqnarray*}
# 
# Step 1: Get the augmented matrix [A, y] 
# 
# $$
# [A, y]  = \begin{bmatrix}
# 4 & 3 & -5 & 2\\
# -2 & -4 & 5 & 5\\
# 8 & 8 & 0 & -3\\
# \end{bmatrix}$$
# 
# Step 2: Get the first element in 1st row to 1, we divide 4 to the row:
# $$
# \begin{bmatrix}
# 1 & 3/4 & -5/4 & 1/2\\
# -2 & -4 & 5 & 5\\
# 8 & 8 & 0 & -3\\
# \end{bmatrix}$$
# 
# Step 3: Eliminate the first element in 2nd and 3rd rows, we multiply -2 and 8 to the 1st row and subtract it from the 2nd and 3rd rows. 
# 
# $$
# \begin{bmatrix}
# 1 & 3/4 & -5/4 & 1/2\\
# 0 & -5/2 & 5/2 & 6\\
# 0 & 2 & 10 & -7\\
# \end{bmatrix}$$
# 
# Step 4: Normalize the 2nd element in 2nd row to 1, we divide -5/2 to achieve this. 
# 
# $$
# \begin{bmatrix}
# 1 & 3/4 & -5/4 & 1/2\\
# 0 & 1 & -1 & -12/5\\
# 0 & 2 & 10 & -7\\
# \end{bmatrix}$$
# 
# Step 5: Eliminate the 2nd element the 3rd row, we multiply 2 to the 2nd row and subtract it from the 3rd row. 
# 
# $$
# \begin{bmatrix}
# 1 & 3/4 & -5/4 & 1/2\\
# 0 & 1 & -1 & -12/5\\
# 0 & 0 & 12 & -11/5\\
# \end{bmatrix}$$
# 
# Step 6: Normalize the last row by divide 8. 
# 
# $$
# \begin{bmatrix}
# 1 & 3/4 & -5/4 & 1/2\\
# 0 & 1 & -1 & -12/5\\
# 0 & 0 & 1 & -11/60\\
# \end{bmatrix}$$
# 
# Step 7: Eliminate the 3rd element in 2nd row by multiply -1 to the 3rd row and subtract it from the 2nd row. 
# 
# $$
# \begin{bmatrix}
# 1 & 3/4 & -5/4 & 1/2\\
# 0 & 1 & 0 & -155/60\\
# 0 & 0 & 1 & -11/60\\
# \end{bmatrix}$$
# 
# Step 8: Eliminate the 3rd element in 1st row by multiply -5/4 to the 3rd row and subtract it from the 1st row. 
# 
# $$
# \begin{bmatrix}
# 1 & 3/4 & 0 & 13/48\\
# 0 & 1 & 0 & -2.583\\
# 0 & 0 & 1 & -0.183\\
# \end{bmatrix}$$
# 
# Step 9: Eliminate the 2nd element in 1st row by multiply 3/4 to the 2nd row and subtract it from the 1st row. 
# 
# $$
# \begin{bmatrix}
# 1 & 0 & 0 & 2.208\\
# 0 & 1 & 0 & -2.583\\
# 0 & 0 & 1 & -0.183\\
# \end{bmatrix}$$

# ## LU Decomposition Method
# 
# We see the above two methods that involves of changing both $A$ and $y$ at the same time when trying to turn A to an upper triangular or diagonal matrix form. It involves many operations. But sometimes, we may have same set of equations but different sets of $y$ for different experiments. This is actually quite common in the real-world, that we have different experiment observations $y_a, y_b, y_c, ...$. Therefore, we have to solve $Ax=y_a$, $Ax=y_b$, ... many times, since every time the $[A, y]$ will change. This is really inefficient, is there a method that we only change the left side of $A$ but not the right hand $y$? The LU decomposition method is one of the solution that we only change the matrix $A$ instead of $y$. It has the advantages for solving the systems that have the same coefficient matrices $A$ but different constant vectors $y$.  
# 
# The LU decomposition method aims to turn $A$ into the multiply of two matrices $L$ and $U$, where $L$ is a lower triangular matrix while $U$ is an upper triangular matrix. With this decomposition, we convert the system of equations to the following form:
# 
# $$LUx=y\rightarrow
# \begin{bmatrix}
# l_{1,1} & 0 & 0 & 0\\
# l_{2,1} & l_{2,2} & 0 & 0\\
# l_{3,1} & l_{3,2} & l_{3,3} & 0 \\
# l_{4,1} & l_{4,2} & l_{4,3} & l_{4,4}
# \end{bmatrix}
# \begin{bmatrix}
# u_{1,1} & u_{1,2} & u_{1,3} & u_{1,4}\\
# 0 & u_{2,2} & u_{2,3} & u_{2,4}\\
# 0 & 0 & u_{3,3} & u_{3,4} \\
# 0 & 0 & 0 & u_{4,4}
# \end{bmatrix}\left[\begin{array}{c} x_1 \\x_2 \\ x_3 \\x_4 \end{array}\right] =
# \left[\begin{array}{c} y_1 \\y_2 \\ y_3 \\y_4 \end{array}\right]$$
# 
# If we define $Ux=M$, then the above equations become:
# 
# $$
# \begin{bmatrix}
# l_{1,1} & 0 & 0 & 0\\
# l_{2,1} & l_{2,2} & 0 & 0\\
# l_{3,1} & l_{3,2} & l_{3,3} & 0 \\
# l_{4,1} & l_{4,2} & l_{4,3} & l_{4,4}
# \end{bmatrix}M =
# \left[\begin{array}{c} y_1 \\y_2 \\ y_3 \\y_4 \end{array}\right]$$
# 
# We can easily solve the above problem by forward substitution (the opposite of the backward substitution as we saw in Gauss Elimination method). After we solve M, we can easily solve the rest of the problem using backward substitution:
# 
# $$
# \begin{bmatrix}
# u_{1,1} & u_{1,2} & u_{1,3} & u_{1,4}\\
# 0 & u_{2,2} & u_{2,3} & u_{2,4}\\
# 0 & 0 & u_{3,3} & u_{3,4} \\
# 0 & 0 & 0 & u_{4,4}
# \end{bmatrix}\left[\begin{array}{c} x_1 \\x_2 \\ x_3 \\x_4 \end{array}\right] =
# \left[\begin{array}{c} m_1 \\m_2 \\ m_3 \\m_4 \end{array}\right]$$
# 
# But how can we calculate and get the $L$ and $U$ matrices? There are different ways to get the LU decomposition, let's just look one way using the Gauss Elimination method. From the above, we know that we get an upper triangular matrix after we conduct the Gauss Elimination. But at the same time, we actually also get the lower triangular matrix, it is just we never explicitly write it out. During the Gauss Elimination procedure, the matrix $A$ actually turns into the multiplication of two matrices as shown below. With the right upper triangular form is the one we get before, but the lower triangular matrix has the diagonal are 1, and the multipliers that multiply the pivot equation to eliminate the elements during the procedure as the elements below the diagonal. 
# 
# $$A=
# \begin{bmatrix}
# 1 & 0 & 0 & 0\\
# m_{2,1} & 1 & 0 & 0\\
# m_{3,1} & m_{3,2} & 1 & 0 \\
# m_{4,1} & m_{4,2} & m_{4,3} & 1
# \end{bmatrix}
# \begin{bmatrix}
# u_{1,1} & u_{1,2} & u_{1,3} & u_{1,4}\\
# 0 & u_{2,2} & u_{2,3} & u_{2,4}\\
# 0 & 0 & u_{3,3} & u_{3,4} \\
# 0 & 0 & 0 & u_{4,4}
# \end{bmatrix}$$
# 
# We can see that, we actually can get both $L$ and $U$ at the same time when we do Gauss Elimination. Let's see the above example, where $U$ is the one we used before to solve the equations, and $L$ is composed of the multipliers (you can check the examples in the Gauss Elimination section). 
# 
# $$
# L = \begin{bmatrix}
# 1 & 0 & 0 \\
# -0.5 & 1 & 0 \\
# 2 & -0.8 & 1 \\
# \end{bmatrix}$$
# 
# $$
# U = \begin{bmatrix}
# 4 & 3 & -5 \\
# 0 & -2.5 & 2.5 \\
# 0 & 0 & 60 \\
# \end{bmatrix}$$
# 
# **TRY IT!** Verify the above $L$ and $U$ matrices are the LU decomposition of matrix $A$. We should see that $A=LU$. 

# In[1]:


import numpy as np

u = np.array([[4, 3, -5], 
              [0, -2.5, 2.5], 
              [0, 0, 12]])
l = np.array([[1, 0, 0], 
              [-0.5, 1, 0], 
              [2, -0.8, 1]])

print('LU=', np.dot(l, u))


# ## Iterative Methods - Gauss-Seidel Method
# 
# The above methods we introduced are all direct methods, in which we compute the solution with a finite number of operations. In this section, we will introduce a different class of methods, the **iterative methods**, or **indirect methods**. It starts with an initial guess of the solution and then repeatedly improve the solution until the change of the solution is below a threshold. In order to use this iterative process, we need first write the explicit form of a system of equations. If we have a system of linear equations:
# 
# $$\begin{bmatrix}
# a_{1,1} & a_{1,2} & ... & a_{1,n}\\
# a_{2,1} & a_{2,2} & ... & a_{2,n}\\
# ... & ... & ... & ... \\
# a_{m,1} & a_{m,2} & ... & a_{m,n}
# \end{bmatrix}\left[\begin{array}{c} x_1 \\x_2 \\ ... \\x_n \end{array}\right] =
# \left[\begin{array}{c} y_1 \\y_2 \\ ... \\y_m \end{array}\right]$$
# we can write its explicit form as:
# 
# $$
# x_i = \frac{1}{a_{i,i}}\Big[y_i - \sum_{j=1, j \ne i}^{j=n}{a_{i,j}x_j} \Big]
# $$
# 
# This is the basics of the iterative methods, we can assume initial values for all the $x$, and use it as $x^{(0)}$. In the first iteration, we can substitute $x^{(0)}$ into the right-hand side of the explicit equation above, and get the first iteration solution $x^{(1)}$. Thus, we can substitute $x^{(1)}$ into the equation and get substitute $x^{(2)}$. The iterations continue until the difference between $x^{(k)}$ and $x^{(k-1)}$ is smaller than some pre-defined value. 
# 
# In order to have the iterative methods work, we do need specific condition for the solution to converge. A sufficient but not necessary condition of the convergence is the coefficient matrix $a$ is a **diagonally dominant**. This means that in each row of the matrix of coefficients $a$, the absolute value of the diagonal element is greater than the sum of the absolute values of the off-diagonal elements. If the coefficient matrix satisfy the condition, the iteration will converge to the solution. The solution might still converge even when this condition is not satisfied.
# 
# ### Gauss-Seidel Method
# The **Gauss-Seidel Method** is a specific iterative method, that is always using the latest estimated value for each elements in $x$. For example, we first assume the initial values for $x_2, x_3, \cdots, x_n$ (except for $x_1$), and then we can calculate $x_1$. Using the calculated $x_1$ and the rest of the $x$ (except for $x_2$), we can calculate $x_2$. We can continue in the same manner and calculate all the elements in $x$. This will conclude the first iteration. We can see the unique part of Gauss-Seidel method is that we are always using the latest value for calculate the next value in $x$. We can then continue with the iterations until the value converges. Let us use this method to solve the same problem we just solved above. 
# 
# **EXAMPLE:** Solve the following system of linear equations using Gauss-Seidel method, use a pre-defined threshold $\epsilon = 0.01$. Do remember to check if the converge condition is satisfied or not. 
# 
# \begin{eqnarray*}
# 8x_1 + 3x_2 - 3x_3 &=& 14 \\
# -2x_1 - 8x_2 + 5x_3 &=& 5 \\
# 3x_1 + 5x_2 + 10x_3 & =& -8 \\
# \end{eqnarray*}
# 
# Let us first check if the coefficient matrix is diagonally dominant or not. 

# In[2]:


a = [[8, 3, -3], [-2, -8, 5], [3, 5, 10]]

# Find diagonal coefficients
diag = np.diag(np.abs(a)) 

# Find row sum without diagonal
off_diag = np.sum(np.abs(a), axis=1) - diag 

if np.all(diag > off_diag):
    print('matrix is diagonally dominant')
else:
    print('NOT diagonally dominant')


# Since it is guaranteed to converge, we can use Gauss-Seidel method to solve it. 

# In[3]:


from numpy.linalg import norm

x1 = 0
x2 = 0
x3 = 0
epsilon = 0.01
converged = False

x_old = np.array([x1, x2, x3])

print('Iteration results')
print(' k,    x1,    x2,    x3 ')
for k in range(1, 50):
    x1 = (14-3*x2+3*x3)/8
    x2 = (5+2*x1-5*x3)/(-8)
    x3 = (-8-3*x1-5*x2)/(10)
    x = np.array([x1, x2, x3])
    # check if it is smaller than threshold
    dx = np.sqrt(np.dot(x-x_old, x-x_old))
    
    print("%d, %.4f, %.4f, %.4f"%(k, x1, x2, x3))
    if dx < epsilon:
        converged = True
        print('Converged!')
        break
        
    # assign the latest x value to the old value
    x_old = x

if not converged:
    print('Not converge, increase the # of iterations')


# <!--NAVIGATION-->
# < [14.3 Systems of Linear Equations](chapter14.03-Systems-of-Linear-Equations.ipynb)  | [Contents](Index.ipynb) | [14.5 Solve Systems of Linear Equations in Python](chapter14.05-Solve-Systems-of-Linear-Equations-in-Python.ipynb) >
