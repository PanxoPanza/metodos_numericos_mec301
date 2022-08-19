#!/usr/bin/env python
# coding: utf-8

# <!--BOOK_INFORMATION-->
# <img align="left" style="padding-right:10px;" src="images/book_cover.jpg" width="120">
# 
# *This notebook contains an excerpt from the [Python Programming and Numerical Methods - A Guide for Engineers and Scientists](https://www.elsevier.com/books/python-programming-and-numerical-methods/kong/978-0-12-819549-9), the content is also available at [Berkeley Python Numerical Methods](https://pythonnumericalmethods.berkeley.edu/notebooks/Index.html).*
# 
# *The copyright of the book belongs to Elsevier. We also have this interactive book online for a better learning experience. The code is released under the [MIT license](https://opensource.org/licenses/MIT). If you find this content useful, please consider supporting the work on [Elsevier](https://www.elsevier.com/books/python-programming-and-numerical-methods/kong/978-0-12-819549-9) or [Amazon](https://www.amazon.com/Python-Programming-Numerical-Methods-Scientists/dp/0128195495/ref=sr_1_1?dchild=1&keywords=Python+Programming+and+Numerical+Methods+-+A+Guide+for+Engineers+and+Scientists&qid=1604761352&sr=8-1)!*

# <!--NAVIGATION-->
# < [CHAPTER 16.  Least Squares Regression](chapter16.00-Least-Squares-Regression.ipynb) | [Contents](Index.ipynb) | [16.2 Least Squares Regression Derivation (Linear Algebra)](chapter16.02-Least-Squares-Regression-Derivation-Linear-Algebra.ipynb)  >

# # Least Squares Regression Problem Statement

# Given a set of independent data points $x_i$ and dependent data points $y_i, i = 1, \ldots, m$, we would like to find an **estimation function**, $\hat{y}(x)$, that describes the data as well as possible. Note that $\hat{y}$ can be a function of several variables, but for the sake of this discussion, we restrict the domain of $\hat{y}$ to be a single variable. In least squares regression, the estimation function must be a linear combination of **basis functions**, $f_i(x)$. That is, the estimation function must be of the form
# $$
# \hat{y}(x) = \sum_{i = 1}^n {\alpha}_i f_i(x)
# $$
# The scalars ${\alpha}_i$ are referred to as the **parameters** of the estimation function, and each basis function must be linearly independent from the others. In other words, in the proper "functional space" no basis function should be expressible as a linear combination of the other functions. Note: In general, there are significantly more data points, $m$, than basis functions, $n$ (i.e., $m >> n$).
# 
# **TRY IT!** 
# Create an estimation function for the force-displacement relationship of a linear spring. Identify the basis function(s) and model parameters. 
# 
# The relationship between the force, $F$, and the displacement, $x$, can be described by the function $F(x) = kx$ where $k$ is the spring stiffness. The only basis function is the function $f_1(x) = x$ and the model parameter to find is ${\alpha}_1 = k$.
# 
# The goal of **least squares regression** is to find the parameters of the estimation function that minimize the **total squared error**, $E$, defined by $E = \sum_{i=1}^m (\hat{y} - y_i)^2$. The **individual errors** or **residuals** are defined as $e_i = (\hat{y} - y_i)$. If $e$ is the vector containing all the individual errors, then we are also trying to minimize $E = \|{e}\|_{2}^{2}$, which is the $L_2$ norm defined in the previous chapter.
# 
# In the next two sections we derive the least squares method of finding the desired parameters. The first derivation comes from linear algebra, and the second derivation comes from multivariable calculus. Although they are different derivations, they lead to the same least squares formula. You are free to focus on the section with which you are most comfortable.

# <!--NAVIGATION-->
# < [CHAPTER 16.  Least Squares Regression](chapter16.00-Least-Squares-Regression.ipynb) | [Contents](Index.ipynb) | [16.2 Least Squares Regression Derivation (Linear Algebra)](chapter16.02-Least-Squares-Regression-Derivation-Linear-Algebra.ipynb)  >
