#!/usr/bin/env python
# coding: utf-8

# <!--BOOK_INFORMATION-->
# <img align="left" style="padding-right:10px;" src="images/book_cover.jpg" width="120">
# 
# *This notebook contains an excerpt from the [Python Programming and Numerical Methods - A Guide for Engineers and Scientists](https://www.elsevier.com/books/python-programming-and-numerical-methods/kong/978-0-12-819549-9), the content is also available at [Berkeley Python Numerical Methods](https://pythonnumericalmethods.berkeley.edu/notebooks/Index.html).*
# 
# *The copyright of the book belongs to Elsevier. We also have this interactive book online for a better learning experience. The code is released under the [MIT license](https://opensource.org/licenses/MIT). If you find this content useful, please consider supporting the work on [Elsevier](https://www.elsevier.com/books/python-programming-and-numerical-methods/kong/978-0-12-819549-9) or [Amazon](https://www.amazon.com/Python-Programming-Numerical-Methods-Scientists/dp/0128195495/ref=sr_1_1?dchild=1&keywords=Python+Programming+and+Numerical+Methods+-+A+Guide+for+Engineers+and+Scientists&qid=1604761352&sr=8-1)!*

# <!--NAVIGATION-->
# < [14.2 Linear Transformations](chapter14.02-Linear-Transformations.ipynb) | [Contents](Index.ipynb) | [14.4 Solutions to Systems of Linear Equations](chapter14.04-Solutions-to-Systems-of-Linear-Equations.ipynb) >

# # Systems of Linear Equations

# A $\textbf{linear equation}$ is an equality of the form
# $$
# \sum_{i = 1}^{n} (a_i x_i) = y,
# $$
# where $a_i$ are scalars, $x_i$ are unknown variables in $\mathbb{R}$, and $y$ is a scalar.
# 
# **TRY IT!** Determine which of the following equations is linear and which is not. For the ones that are not linear, can you manipulate them so that they are?
# 
# 1. $3x_1 + 4x_2 - 3 = -5x_3$
# 2. $\frac{-x_1 + x_2}{x_3} = 2$
# 3. $x_1x_2 + x_3 = 5$
# 
# Equation 1 can be rearranged to be $3x_1 + 4x_2 + 5x_3= 3$, which
# clearly has the form of a linear equation. Equation 2 is not linear
# but can be rearranged to be $-x_1 + x_2 - 2x_3 = 0$, which is
# linear. Equation 3 is not linear.
# 
# A $\textbf{system of linear equations}$ is a set of linear equations that share the same variables. Consider the following system of linear equations:
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
# where $a_{i,j}$ and $y_i$ are real numbers. The $\textbf{matrix form}$ of a system of linear equations is $\textbf{$Ax = y$}$ where $A$ is a ${m} \times {n}$ matrix, $A(i,j) = a_{i,j}, y$ is a vector in ${\mathbb{R}}^m$, and $x$ is an unknown vector in ${\mathbb{R}}^n$. The matrix form is showing as below:
# 
# $$\begin{bmatrix}
# a_{1,1} & a_{1,2} & ... & a_{1,n}\\
# a_{2,1} & a_{2,2} & ... & a_{2,n}\\
# ... & ... & ... & ... \\
# a_{m,1} & a_{m,2} & ... & a_{m,n}
# \end{bmatrix}\left[\begin{array}{c} x_1 \\x_2 \\ ... \\x_n \end{array}\right] =
# \left[\begin{array}{c} y_1 \\y_2 \\ ... \\y_m \end{array}\right]$$
# 
# If you carry out the matrix multiplication, you will see that you arrive back at the original system of equations.
# 
# **TRY IT!** Put the following system of equations into matrix form.
# \begin{eqnarray*}
# 4x + 3y - 5z &=& 2 \\
# -2x - 4y + 5z &=& 5 \\
# 7x + 8y   &=& -3 \\
# x   + 2z &=& 1  \\
# 9 + y - 6z &=& 6 \\
# \end{eqnarray*}
# 
# $$\begin{bmatrix}
# 4 & 3 & -5\\
# -2 & -4 & 5\\
# 7 & 8 & 0\\
# 1 & 0 & 2\\
# 9 & 1 & -6
# \end{bmatrix}\left[\begin{array}{c} x \\y \\z \end{array}\right] =
# \left[\begin{array}{c} 2 \\5 \\-3 \\1 \\6 \end{array}\right]$$

# <!--NAVIGATION-->
# < [14.2 Linear Transformations](chapter14.02-Linear-Transformations.ipynb) | [Contents](Index.ipynb) | [14.4 Solutions to Systems of Linear Equations](chapter14.04-Solutions-to-Systems-of-Linear-Equations.ipynb) >
