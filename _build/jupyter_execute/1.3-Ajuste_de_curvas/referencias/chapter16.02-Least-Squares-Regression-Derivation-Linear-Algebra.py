#!/usr/bin/env python
# coding: utf-8

# <!--BOOK_INFORMATION-->
# <img align="left" style="padding-right:10px;" src="images/book_cover.jpg" width="120">
# 
# *This notebook contains an excerpt from the [Python Programming and Numerical Methods - A Guide for Engineers and Scientists](https://www.elsevier.com/books/python-programming-and-numerical-methods/kong/978-0-12-819549-9), the content is also available at [Berkeley Python Numerical Methods](https://pythonnumericalmethods.berkeley.edu/notebooks/Index.html).*
# 
# *The copyright of the book belongs to Elsevier. We also have this interactive book online for a better learning experience. The code is released under the [MIT license](https://opensource.org/licenses/MIT). If you find this content useful, please consider supporting the work on [Elsevier](https://www.elsevier.com/books/python-programming-and-numerical-methods/kong/978-0-12-819549-9) or [Amazon](https://www.amazon.com/Python-Programming-Numerical-Methods-Scientists/dp/0128195495/ref=sr_1_1?dchild=1&keywords=Python+Programming+and+Numerical+Methods+-+A+Guide+for+Engineers+and+Scientists&qid=1604761352&sr=8-1)!*

# <!--NAVIGATION-->
# < [16.1 Least Squares Regression Problem Statement](chapter16.01-Least-Squares-Regression-Problem-Statement.ipynb)  | [Contents](Index.ipynb) | [16.3 Least Squares Regression Derivation (Multivariable Calculus)](chapter16.03-Least-Squares-Regression-Derivation-Multivariable-Calculus.ipynb)   >

# # Least Squares Regression Derivation (Linear Algebra)

# First, we enumerate the estimation of the data at each data point $x_i$
# 
# \begin{eqnarray*}
# &&\hat{y}(x_1) = {\alpha}_1 f_1(x_1) + {\alpha}_2 f_2(x_1) + \cdots + {\alpha}_n f_n(x_1), \\
# &&\hat{y}(x_2) = {\alpha}_1 f_1(x_2) + {\alpha}_2 f_2(x_2) + \cdots + {\alpha}_n f_n(x_2), \\
# &&\qquad\qquad \qquad \qquad \quad \cdots\\
# &&\hat{y}(x_m) = {\alpha}_1 f_1(x_m) + {\alpha}_2 f_2(x_m) + \cdots + {\alpha}_n f_n(x_m).\end{eqnarray*}
# 
# Let $X\in {\Bbb R}^n$ be a column vector such that the $i$-th element of $X$ contains the value of the $i$-th $x$-data point, $x_i, \hat{Y}$ be a column vector with elements, $\hat{Y}_i = \hat{y}(x_i), {\beta}$ be a column vector such that ${\beta}_i = {\alpha}_i, F_i(x)$ be a function that returns a column vector of $f_i(x)$ computed on every element of $x$, and $A$ be an $m \times n$ matrix such that the $i$-th column of $A$ is $F_i(x)$. Given this notation, the previous system of equations becomes $\hat{Y} = A{\beta}$.
# 
# Now if $Y$ is a column vector such that $Y_i = y_i$, the total squared error is given by $E = \|{\hat{Y} - Y}\|_{2}^2$. You can verify this by substituting the definition of the $L_2$ norm. Since we want to make $E$ as small as possible and norms are a measure of distance, this previous expression is equivalent to saying that we want $\hat{Y}$ and $Y$ to be a "close as possible." Note that in general $Y$ will not be in the range of $A$ and therefore $E > 0$.
# 
# Consider the following simplified depiction of the range of $A$; see the following figure. Note this is $\it not$ a plot of the data points $(x_i, y_i)$.
# 
# <img src="./images/16.02.01-Illustration_of_Least_Square_Regression.png" alt="Illustration of Least Square Regression" title="Illustration of the L2 projection of Y on the range of A" width="400"/>
# 
# From observation, the vector in the range of $A, \hat{Y}$, that is closest to $Y$ is the one that can point perpendicularly to $Y$. Therefore, we want a vector $Y - \hat{Y}$ that is perpendicular to the vector $\hat{Y}$.
# 
# Recall from Linear Algebra that two vectors are perpendicular, or orthogonal, if their dot product is 0. Noting that the dot product between two vectors, $v$ and $w$, can be written as ${\text{dot}}(v,w) = v^T w$, we can state that $\hat{Y}$ and $Y - \hat{Y}$ are perpendicular if ${\text{dot}}(\hat{Y}, Y - \hat{Y}) = 0$; therefore, $\hat{Y}^T (Y - \hat{Y}) = 0$, which is equivalent to $(A{\beta})^T(Y - A{\beta}) = 0$.
# 
# Noting that for two matrices $A$ and $B, (AB)^T = B^T A^T$ and using distributive properties of vector multiplication, this is equivalent to ${\beta}^T A^T Y - {\beta}^T A^T A {\beta} = {\beta}^T(A^T Y - A^T A {\beta}) = 0$. The solution, ${\beta} = \textbf{0}$, is a trivial solution, so we use $A^T Y - A^T A {\beta} = 0$ to find a more interesting solution. Solving this equation for ${\beta}$ gives the $\textbf{least squares regression formula}$:
# 
# $$
# {\beta} = (A^T A)^{-1} A^T Y
# $$
# 
# Note that $(A^T A)^{-1}A^T$ is called the **pseudo-inverse** of $A$ and exists when $m > n$ and $A$ has linearly independent columns. Proving the invertibility of $(A^T A)$ is outside the scope of this book, but it is always invertible except for some pathological cases.

# <!--NAVIGATION-->
# < [16.1 Least Squares Regression Problem Statement](chapter16.01-Least-Squares-Regression-Problem-Statement.ipynb)  | [Contents](Index.ipynb) | [16.3 Least Squares Regression Derivation (Multivariable Calculus)](chapter16.03-Least-Squares-Regression-Derivation-Multivariable-Calculus.ipynb)   >
