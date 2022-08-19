#!/usr/bin/env python
# coding: utf-8

# <!--BOOK_INFORMATION-->
# <img align="left" style="padding-right:10px;" src="images/book_cover.jpg" width="120">
# 
# *This notebook contains an excerpt from the [Python Programming and Numerical Methods - A Guide for Engineers and Scientists](https://www.elsevier.com/books/python-programming-and-numerical-methods/kong/978-0-12-819549-9), the content is also available at [Berkeley Python Numerical Methods](https://pythonnumericalmethods.berkeley.edu/notebooks/Index.html).*
# 
# *The copyright of the book belongs to Elsevier. We also have this interactive book online for a better learning experience. The code is released under the [MIT license](https://opensource.org/licenses/MIT). If you find this content useful, please consider supporting the work on [Elsevier](https://www.elsevier.com/books/python-programming-and-numerical-methods/kong/978-0-12-819549-9) or [Amazon](https://www.amazon.com/Python-Programming-Numerical-Methods-Scientists/dp/0128195495/ref=sr_1_1?dchild=1&keywords=Python+Programming+and+Numerical+Methods+-+A+Guide+for+Engineers+and+Scientists&qid=1604761352&sr=8-1)!*

# <!--NAVIGATION-->
# < [16.2 Least Squares Regression Derivation (Linear Algebra)](chapter16.02-Least-Squares-Regression-Derivation-Linear-Algebra.ipynb)  | [Contents](Index.ipynb) | [16.4 Least Squares Regression in Python](chapter16.04-Least-Squares-Regression-in-Python.ipynb)    >

# # Least Squares Regression Derivation (Multivariable Calculus)

# Recall that the total error for $m$ data points and $n$ basis functions is:
# 
# $$
# E = \sum_{i=1}^m e_i^2 = \sum_{i=1}^m (\hat{y}(x_i) - y_i)^2 = \sum_{i=1}^m \left(\sum_{j=1}^n {\alpha}_j f_j(x_i) - y_i\right)^2.
# $$
# 
# which is an $n$-dimensional paraboloid in ${\alpha}_k$. From calculus, we know that the minimum of a paraboloid is where all the partial derivatives equal zero. So taking partial derivative of $E$ with respect to the variable ${\alpha}_k$ (remember that in this case the parameters are our variables), setting the system of equations equal to 0 and solving for the ${\alpha}_k$'s should give the correct results.
# 
# The partial derivative with respect to ${\alpha}_k$ and setting equal to 0 yields:
# $$
# \frac{\partial E}{\partial {\alpha}_k} = \sum_{i=1}^m 2\left(\sum_{j=1}^n {\alpha}_j f_j(x_i) - y_i\right)f_k(x_i) = 0.
# $$
# 
# With some rearrangement, the previous expression can be manipulated to the following:
# $$
# \sum_{i=1}^m \sum_{j=1}^n {\alpha}_j f_j(x_i)f_k(x_i) - \sum_{i=1}^m y_i f_k(x_i) = 0,
# $$
# 
# and further rearrangement taking advantage of the fact that addition commutes results in:
# $$
# \sum_{j=1}^n {\alpha}_j \sum_{i=1}^m f_j(x_i)f_k(x_i) = \sum_{i=1}^m y_i f_k(x_i).
# $$
# Now let $X$ be a column vector such that the $i$-th element of $X$ is $x_i$ and $Y$ similarly constructed, and let $F_j(X)$ be a column vector such that the $i$-th element of $F_j(X)$ is $f_j(x_i)$. Using this notation, the previous expression can be rewritten in vector notation as:
# $$
# \left[F_k^T(X)F_1(X), F_k^T(X)F_2(X), \ldots, F_k^T(X)F_j(X), \ldots, F_k^T(X)F_n(X)\right]
# \left[\begin{array}{c} {\alpha}_1 \\
# {\alpha}_2 \\
# \cdots \\
# {\alpha}_j \\
# \cdots \\
# {\alpha}_n
# \end{array}\right] = F_k^T(X)Y.
# $$
# If we repeat this equation for every $k$, we get the following system of linear equations in matrix form:
# 
# $$
# \left[\begin{array}{cc}
# F_1^T(X)F_1(X), F_1^T(X)F_2(X), \ldots, F_1^T(X)F_j(X), \ldots, F_1^T(X)F_n(X)&\\ 
# F_2^T(X)F_1(X), F_2^T(X)F_2(X), \ldots, F_2^T(X)F_j(X), \ldots, F_2^T(X)F_n(X)&\\
# & \cdots \ \cdots\\
# F_n^T(X)F_1(X), F_n^T(X)F_2(X), \ldots, F_n^T(X)F_j(X), \ldots, F_n^T(X)F_n(X)
# \end{array}\right]
# \left[\begin{array}{c} {\alpha}_1 \\
# {\alpha}_2 \\
# \cdots \\
# {\alpha}_j \\
# \cdots \\
# {\alpha}_n
# \end{array}\right] =
# \left[\begin{array}{c} F_1^T(X)Y \\
# F_2^T(X)Y \\
# \cdots \\
# F_n^T(X)Y
# \end{array}\right].
# $$
# 
# If we let $A = [F_1(X), F_2(X), \ldots, F_j(X), \ldots, F_n(X)]$ and ${\beta}$ be a column vector such that $j$-th element of ${\beta}$ is ${\alpha}_j$, then the previous system of equations becomes
# $$
# A^T A {\beta} = A^T Y,
# $$
# and solving this matrix equation for ${\beta}$ gives ${\beta} = (A^T A)^{-1} A^T Y$, which is exactly the same formula as the previous derivation.

# <!--NAVIGATION-->
# < [16.2 Least Squares Regression Derivation (Linear Algebra)](chapter16.02-Least-Squares-Regression-Derivation-Linear-Algebra.ipynb)  | [Contents](Index.ipynb) | [16.4 Least Squares Regression in Python](chapter16.04-Least-Squares-Regression-in-Python.ipynb)    >
