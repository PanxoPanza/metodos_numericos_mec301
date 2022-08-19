#!/usr/bin/env python
# coding: utf-8

# <!--BOOK_INFORMATION-->
# <img align="left" style="padding-right:10px;" src="images/book_cover.jpg" width="120">
# 
# *This notebook contains an excerpt from the [Python Programming and Numerical Methods - A Guide for Engineers and Scientists](https://www.elsevier.com/books/python-programming-and-numerical-methods/kong/978-0-12-819549-9), the content is also available at [Berkeley Python Numerical Methods](https://pythonnumericalmethods.berkeley.edu/notebooks/Index.html).*
# 
# *The copyright of the book belongs to Elsevier. We also have this interactive book online for a better learning experience. The code is released under the [MIT license](https://opensource.org/licenses/MIT). If you find this content useful, please consider supporting the work on [Elsevier](https://www.elsevier.com/books/python-programming-and-numerical-methods/kong/978-0-12-819549-9) or [Amazon](https://www.amazon.com/Python-Programming-Numerical-Methods-Scientists/dp/0128195495/ref=sr_1_1?dchild=1&keywords=Python+Programming+and+Numerical+Methods+-+A+Guide+for+Engineers+and+Scientists&qid=1604761352&sr=8-1)!*

# <!--NAVIGATION-->
# < [CHAPTER 14.  Linear Algebra and Systems of Linear Equations](chapter14.00-Linear-Algebra-and-Systems-of-Linear-Equations.ipynb) | [Contents](Index.ipynb) | [14.2 Linear Transformations](chapter14.02-Linear-Transformations.ipynb) >

# # Basics of Linear Algebra

# Before we introduce the systems of linear equations, let's first introduce some basics of linear algebra, which will be used to describe and solve the linear equations. We will just cover the very basics of it in this chapter, and you can explore more by reading a linear algebra book. 
# 
# ## Sets
# 
# We discussed the data structure - sets in chapter 2 before, here we will take a look of it from mathematics point of view using the mathematical languages. In mathematics, a **set** is a collection of objects. As we shown before, sets are usually denoted by braces {}. For example, $S = {orange, apple, banana}$ means "S is the set containing 'orange', 'apple', and 'banana'". 
# 
# The **empty set** is the set containing no objects and is typically denoted by empty braces such as $\{\}$ or by $\emptyset$. Given two sets, $A$ and $B$, the **union** of $A$ and $B$ is denoted by $A \cup B$ and equal to the set containing all the elements of $A$ and $B$. The **intersect** of $A$ and $B$ is denoted by $A \cap B$ and equal to the set containing all the elements that belong to both $A$ and $B$. In set notation, a colon is used to mean **"such that"**. The usage of these terms will become apparent shortly. The symbol **$\in$** is used to denote that an object is contained in a set. For example $a \in A$ means "$a$ is a member of $A$" or "$a$ is in $A$." A backslash, $\backslash$, in set notation means **set minus**. So if $a\in A$ then $A\backslash a$ means "$A$ minus the element, $a$."
# 
# There are several standard sets related to numbers, for example **natural numbers**, **whole numbers**, **integers**, **rational numbers**, **irrational numbers**, **real numbers**, and **complex numbers**. A description of each set and the symbol used to denote them is shown in the following table.
# 
# | Set Name | Symbol  | Description  |
# |------|------|------------|
# |   Naturals  | $\mathbb{N}$| ${\mathbb{N}} = \{1, 2, 3, 4, \cdots\}$|
# |   Wholes    | $\mathbb{W}$| ${\mathbb{W}} = \mathbb{N} \cup \{0\}$|
# |   Integers  | $\mathbb{Z}$| ${\mathbb{Z}} = \mathbb{W} \cup \{-1, -2, -3, \cdots\}$|
# |   Rationals | $\mathbb{Q}$| ${\mathbb{Q}} = \{\frac{p}{q} : p\in {\mathbb{Z}}, q\in {\mathbb{Z}} \backslash \{0\}\}$|
# | Irrationals | $\mathbb{I}$| ${\mathbb{I}}$ is the set of real numbers not expressible as a fraction of integers.|
# |   Reals     | $\mathbb{R}$| ${\mathbb{R}} = \mathbb{Q} \cup \mathbb{I}$|
# |   Complex Numbers     | $\mathbb{C}$| ${\mathbb{C}} = \{a + bi : a,b\in {\mathbb{R}}, i = \sqrt{-1}\}$|
# 
# **TRY IT!** Let $S$ be the set of all real $(x,y)$ pairs such that $x^2 + y^2 = 1$. Write $S$ using set notation.
# 
# 
# $S = \{(x,y) : x,y \in {\mathbb{R}}, x^2 + y^2 = 1\}$

# ## Vectors
# 
# The set ${\mathbb{R}}^n$ is the set of all $n$-tuples of real numbers. In set notation this is ${\mathbb{R}}^n = \{(x_1, x_2, x_3, \cdots, x_n): x_1, x_2, x_3, \cdots, x_n \in {\mathbb{R}}\}$. For example, the set ${\mathbb{R}}^3$ represents the set of real triples, $(x,y,z)$ coordinates, in three-dimensional space.
# 
# A **vector** in ${\mathbb{R}}^n$ is an $n$-tuple, or point, in ${\mathbb{R}}^n$. Vectors can be written horizontally (i.e., with the elements of the vector next to each other) in a **row vector**, or vertically (i.e., with the elements of the vector on top of each other) in a **column vector**. If the context of a vector is ambiguous, it usually means the vector is a column vector. The $i$-th element of a vector, $v$, is denoted by $v_i$. The transpose of a column vector is a row vector of the same length, and the transpose of a row vector is a column vector. In mathematics, the transpose is denoted by a superscript $T$, or $v^T$. The **zero vector** is the vector in ${\mathbb{R}}^n$ containing all zeros.
# 
# The **norm** of a vector is a measure of its length. There are many ways of defining the length of a vector depending on the metric used (i.e., the distance formula chosen). The most common is called the $L_2$ norm, which is computed according to the distance formula you are probably familiar with from grade school. The **$L_2$ norm** of a vector $v$ is denoted by $\Vert v \Vert_{2}$ and $\Vert v \Vert_{2} = \sqrt{\sum_i v_i^2}$. This is sometimes also called Euclidian length and refers to the "physical" length of a vector in one-, two-, or three-dimensional space. The $L_1$ norm, or "Manhattan Distance," is computed as $\Vert v \Vert_{1} = \sum_i |v_i|$, and is named after the grid-like road structure in New York City. In general, the **p-norm**, $L_p$, of a vector is $\Vert v \Vert_{p} = \sqrt[p]{(\sum_i v_i^p)}$. The **$L_\infty$ norm** is the $p$-norm, where $p = \infty$. The $L_\infty$ norm is written as $||v||_\infty$ and it is equal to the maximum absolute value in $v$.
# 
# **TRY IT!** Create a row vector and column vector, and show the shape of the vectors.

# In[1]:


import numpy as np
vector_row = np.array([[1, -5, 3, 2, 4]])
vector_column = np.array([[1], 
                          [2], 
                          [3], 
                          [4]])
print(vector_row.shape)
print(vector_column.shape)


# **Note!** In Python, the row vector and column vector are a little bit tricky. You can see from the above in order to get the 1 row and 4 columns or 4 rows and 1 column vectors, we have to use list of list to specify it. You can define np.array([1,2,3,4]), but you will soon notice that it doesn't contain information about row or column. 
# 
# **TRY IT!** Transpose the row vector we defined above into a column vector and calculate the $L_1$, $L_2$, and $L_\infty$ norm of it. Verify that the $L_\infty$ norm of a vector is equivalent to the maximum value of the elements in the vector.

# In[2]:


from numpy.linalg import norm
new_vector = vector_row.T
print(new_vector)
norm_1 = norm(new_vector, 1)
norm_2 = norm(new_vector, 2)
norm_inf = norm(new_vector, np.inf)
print('L_1 is: %.1f'%norm_1)
print('L_2 is: %.1f'%norm_2)
print('L_inf is: %.1f'%norm_inf)


# **Vector addition** is defined as the pairwise addition of each of the elements of the added vectors. For example, if $v$ and $w$ are vectors in ${\mathbb{R}}^n$, then $u = v + w$ is defined as $u_i = v_i + w_i$.
# 
# **Vector multiplication** can be defined in several ways depending on the context. **Scalar multiplication** of a vector is the product of a vector and a **scalar** (i.e., a number in ${\mathbb{R}}$). Scalar multiplication is defined as the product of each element of the vector by the scalar. More specifically, if $\alpha$ is a scalar and $v$ is a vector, then $u = \alpha v$ is defined as $u_i = \alpha v_i$. Note that this is exactly how Python implements scalar multiplication with a vector.
# 
# **TRY IT!** Show that $a(v + w) = av + aw$ (i.e., scalar multiplication of a vector distributes across vector addition).
# 
# By vector addition, $u = v + w$ is the vector with $u_i = v_i + w_i$. By scalar multiplication of a vector, $x = \alpha u$ is the vector with $x_i = \alpha(v_i + w_i)$. Since $\alpha, v_i$, and $w_i$ are scalars, multiplication distributes and $x_i = \alpha v_i + \alpha w_i$. Therefore, $a(v + w) = av + aw$.
# 
# The **dot product** of two vectors is the sum of the product of the respective elements in each vector and is denoted by $\cdot$, and $v \cdot w$ is read "v dot w." Therefore for $v$ and $w$ $\in {\mathbb{R}}^n, d = v\cdot w$ is defined as $d = \sum_{i = 1}^{n} v_iw_i$. The **angle between two vectors**, $\theta$, is defined by the formula:
# 
# $$
# v \cdot w = \Vert v \Vert_{2} \Vert w \Vert_{2} \cos{\theta}
# $$
# 
# The dot product is a measure of how similarly directed the two vectors are. For example, the vectors (1,1) and (2,2) are parallel. If you compute the angle between them using the dot product, you will find that $\theta = 0$. If the angle between the vectors, $\theta = \pi/2$, then the vectors are said to be perpendicular or **orthogonal**, and the dot product is 0.
# 
# **TRY IT!** Compute the angle between the vectors $v = [10, 9, 3]$ and $w = [2, 5, 12]$. 

# In[3]:


from numpy import arccos, dot

v = np.array([[10, 9, 3]])
w = np.array([[2, 5, 12]])
theta = \
    arccos(dot(v, w.T)/(norm(v)*norm(w)))
print(theta)


# Finally, the **cross product** between two vectors, $v$ and $w$, is written $v \times w$. It is defined by $v \times w = \Vert v \Vert_{2}\Vert w \Vert_{2}\sin{(\theta)} \textit{n}$, where $\theta$ is the angle between the $v$ and $w$ (which can be computed from the dot product) and **$n$** is a vector perpendicular to both $v$ and $w$ with unit length (i.e., the length is one). The geometric interpretation of the cross product is a vector perpendicular to both $v$ and $w$ with length equal to the area enclosed by the parallelogram created by the two vectors.
# 
# **TRY IT!** Given the vectors $v = [0, 2, 0]$ and $w = [3, 0, 0]$, use the Numpy function cross to compute the cross product of v and w.

# In[4]:


v = np.array([[0, 2, 0]])
w = np.array([[3, 0, 0]])
print(np.cross(v, w))


# Assuming that $S$ is a set in which addition and scalar multiplication are defined, a **linear combination** of $S$ is defined as
# $$
# \sum \alpha_i s_i,
# $$
# 
# where $\alpha_i$ is any real number and $s_i$ is the $i^{\text{th}}$ object in $S$. Sometimes the $\alpha_i$ values are called the **coefficients** of $s_i$. Linear combinations can be used to describe numerous things. For example, a grocery bill can be written $\displaystyle{\sum c_i n_i}$, where $c_i$ is the cost of item $i$ and $n_i$ is the number of item $i$ purchased. Thus, the total cost is a linear combination of the items purchased.
# 
# A set is called **linearly independent** if no object in the set can be written as a linear combination of the other objects in the set. For the purposes of this book, we will only consider the linear independence of a set of vectors. A set of vectors that is not linearly independent is **linearly dependent**.
# 
# **TRY IT!** Given the row vectors $v = [0, 3, 2]$, $w = [4, 1, 1]$, and $u = [0, -2, 0]$, write the vector $x = [-8, -1, 4]$ as a linear combination of $v$, $w$, and $u$.

# In[5]:


v = np.array([[0, 3, 2]])
w = np.array([[4, 1, 1]])
u = np.array([[0, -2, 0]])
x = 3*v-2*w+4*u
print(x)


# **TRY IT!** Determine by inspection whether the following set of vectors is linearly independent: $v = [1, 1, 0]$, $w = [1, 0, 0]$, $u = [0, 0, 1]$.
# 
# Clearly $u$ is linearly independent from $v$ and $w$ because only $u$ has a nonzero third element. The vectors $v$ and $w$ are also linearly independent because only $v$ has a nonzero second element. Therefore, $v, w$, and $u$ are linearly independent.
# 
# ## Matrices
# 
# An ${m} \times {n}$ **matrix** is a rectangular table of numbers consisting of $m$ rows and $n$ columns. The norm of a matrix can be considered as a particular kind of vector norm, if we treat the ${m} \times {n}$ elements of $M$ are the elements of an $mn$ dimensional vector, then the p-norm of this vector can be write as:
# 
# $$\Vert M \Vert_{p} = \sqrt[p]{(\sum_i^m \sum_j^n |a_{ij}|^p)}$$
# 
# You can calculate the matrix norm using the same `norm` function in `Numpy` as that for vector. 
# 
# Matrix addition and scalar multiplication for matrices work the same way as for vectors. However, **matrix multiplication** between two matrices, $P$ and $Q$, is defined when $P$ is an ${m} \times {p}$ matrix and $Q$ is a ${p} \times {n}$ matrix. The result of $M = PQ$ is a matrix $M$ that is $m \times n$. The dimension with size $p$ is called the **inner matrix dimension**, and the inner matrix dimensions must match (i.e., the number of columns in $P$ and the number of rows in $Q$ must be the same) for matrix multiplication to be defined. The dimensions $m$ and $n$ are called the **outer matrix dimensions**. Formally, if $P$ is ${m} \times {p}$ and Q is ${p} \times {n}$, then $M = PQ$ is defined as
# 
# $$
# M_{ij} = \sum_{k=1}^p P_{ik}Q_{kj}
# $$
# 
# The product of two matrices $P$ and $Q$ in Python is achieved by using the **dot** method in Numpy. The **transpose** of a matrix is a reversal of its rows with its columns. The transpose is denoted by a superscript, $T$, such as $M^T$ is the transpose of matrix $M$. In Python, the method **T** for an Numpy array is used to get the transpose. For example, if $M$ is a matrix, then $M.T$ is its transpose.
# 
# **TRY IT!** Let the Python matrices $P = [[1, 7], [2, 3], [5, 0]]$ and $Q = [[2, 6, 3, 1], [1, 2, 3, 4]]$. Compute the matrix product of $P$ and $Q$. Show that the product of $Q$ and $P$ will produce an error.

# In[6]:


P = np.array([[1, 7], [2, 3], [5, 0]])
Q = np.array([[2, 6, 3, 1], [1, 2, 3, 4]])
print(P)
print(Q)
print(np.dot(P, Q))
np.dot(Q, P)


# A **square matrix** is an ${n} \times {n}$ matrix; that is, it has the same number of rows as columns. The **determinant** is an important property of square matrices. The determinant is denoted by $det(M)$, both in mathematics and in Numpy's `linalg` package, sometimes it is also denoted as $|M|$. Some examples in the uses of a determinant will be described later. 
# 
# In the case of a $2 \times 2$ matrix, the determinant is:
# 
# $$
# |M| = \begin{bmatrix}
# a & b \\
# c & d\\
# \end{bmatrix} = ad - bc$$
# 
# Similarly, in the case of a $3 \times 3$ matrix, the determinant is:
# 
# $$
# \begin{eqnarray*}
# |M| = \begin{bmatrix}
# a & b & c \\
# d & e & f \\
# g & h & i \\
# \end{bmatrix} & = & a\begin{bmatrix}
# \Box &\Box  &\Box  \\
# \Box & e & f \\
# \Box & h & i \\
# \end{bmatrix} - b\begin{bmatrix}
# \Box &\Box  &\Box  \\
# d & \Box & f \\
# g & \Box & i \\
# \end{bmatrix}+c\begin{bmatrix}
# \Box &\Box  &\Box  \\
# d & e & \Box \\
# g & h & \Box \\
# \end{bmatrix} \\
# &&\\
# & = & a\begin{bmatrix}
# e & f \\
# h & i \\
# \end{bmatrix} - b\begin{bmatrix}
# d & f \\
# g & i \\
# \end{bmatrix}+c\begin{bmatrix}
# d & e \\
# g & h \\
# \end{bmatrix} \\ 
# &&\\
# & = & aei + bfg + cdh - ceg - bdi - afh
# \end{eqnarray*}$$
# 
# We can use similar approach to calculate the determinant for higher the dimension of the matrix, but it is much easier to calculate using Python. We will see an example below how to calculate the determinant in Python.  
# 
# 
# The **identity matrix** is a square matrix with ones on the diagonal and zeros elsewhere. The identity matrix is usually denoted by $I$, and is analagous to the real number identity, 1. That is, multiplying any matrix by $I$ (of compatible size) will produce the same matrix.
# 
# **TRY IT!** Use Python to find the determinant of the matrix $M = [[0, 2, 1, 3], [3, 2, 8, 1], [1, 0, 0, 3], [0, 3, 2, 1]]$. Use the *np.eye* function to produce a ${4} \times {4}$ identity matrix, $I$. Multiply $M$ by $I$ to show that the result is $M$.

# In[7]:


from numpy.linalg import det

M = np.array([[0,2,1,3], 
             [3,2,8,1], 
             [1,0,0,3],
             [0,3,2,1]])
print('M:\n', M)

print('Determinant: %.1f'%det(M))
I = np.eye(4)
print('I:\n', I)
print('M*I:\n', np.dot(M, I))


# The **inverse** of a square matrix $M$ is a matrix of the same size, $N$, such that $M \cdot N = I$. The inverse of a matrix is analagous to the inverse of real numbers. For example, the inverse of 3 is $\frac{1}{3}$ because $(3)(\frac{1}{3}) = 1$. A matrix is said to be **invertible** if it has an inverse. The inverse of a matrix is unique; that is, for an invertible matrix, there is only one inverse for that matrix. If $M$ is a square matrix, its inverse is denoted by $M^{-1}$ in mathematics, and it can be computed in Python using the function *inv* from Numpy's *linalg* package.
# 
# For a $2 \times 2$ matrix, the analytic solution of the matrix inverse is:
# 
# $$
# M^{-1} = \begin{bmatrix}
# a & b \\
# c & d\\
# \end{bmatrix}^{-1} = \frac{1}{|M|}\begin{bmatrix}
# d & -b \\
# -c & a\\
# \end{bmatrix}$$
# 
# The calculation of the matrix inverse for the analytic solution becomes complicated with increasing matrix dimension, there are many other methods can make things easier, such as Gaussian elimination, Newton's method, Eigendecomposition and so on. We will introduce some of these methods after we learn how to solve a system of linear equations, because the process is essentially the same. 
# 
# Recall that 0 has no inverse for multiplication in the real-numbers setting. Similarly, there are matrices that do not have inverses. These matrices are called **singular**. Matrices that do have an inverse are called **nonsingular**.
# 
# One way to determine if a matrix is singular is by computing its determinant. If the determinant is 0, then the matrix is singular; if not, the matrix is nonsingular.
# 
# **TRY IT!** The matrix $M$ (in the previous example) has a nonzero determinant. Compute the inverse of $M$. Show that the matrix $P = [[0, 1, 0], [0, 0, 0], [1, 0, 1]]$ has a determinant value of 0 and therefore has no inverse.

# In[8]:


from numpy.linalg import inv

print('Inv M:\n', inv(M))
P = np.array([[0,1,0],
              [0,0,0],
              [1,0,1]])
print('det(p):\n', det(P))


# A matrix that is close to being singular (i.e., the determinant is close to 0) is called **ill-conditioned**. Although ill-conditioned matrices have inverses, they are problematic numerically in the same way that dividing a number by a very, very small number is problematic. That is, it can result in computations that result in overflow, underflow, or numbers small enough to result in significant round-off errors (If you forget these concepts, refresh yourself with materials in chapter 9). The **condition number** is a measure of how ill-conditioned a matrix is, and it can be computed using Numpy's function *cond* from *linalg*. The higher the condition number, the closer the matrix is to being singular.
# 
# The **rank**. of an ${m} \times {n}$ matrix $A$ is the number of linearly independent columns or rows of $A$, and is denoted by rank($A$). It can be shown that the number of linearly independent rows is always equal to the number of linearly independent columns
# for any matrix. A matrix is called **full rank**. if rank $(A)=\min(m,n)$. The matrix, $A$, is also full rank if all of its columns are linearly independent. An **augmented matrix**. is a matrix, $A$, concatenated with a vector, $y$, and is written $[A,y]$. This is commonly read "$A$ augmented with $y$." You can use *np.concatenate* to concatenate the them. If $rank([A,y]) = {rank}(A) + 1$, then the vector, $y$, is "new" information. That is, it cannot be created as a linear combination of the columns in $A$. The rank is an important property of matrices because of its relationship to solutions of linear equations, which is discussed in the last section of this chapter.
# 
# **TRY IT!** Matrix $A = [[1, 1, 0], [0, 1, 0], [1, 0, 1]]$, compute the condition number and rank for this matrix. If $y = [[1], [2], [1]]$, get the augmented matrix [A, y]. 

# In[9]:


from numpy.linalg import \
             cond, matrix_rank

A = np.array([[1,1,0],
              [0,1,0],
              [1,0,1]])

print('Condition number:\n', cond(A))
print('Rank:\n', matrix_rank(A))
y = np.array([[1], [2], [1]])
A_y = np.concatenate((A, y), axis = 1)
print('Augmented matrix:\n', A_y)


# <!--NAVIGATION-->
# < [CHAPTER 14.  Linear Algebra and Systems of Linear Equations](chapter14.00-Linear-Algebra-and-Systems-of-Linear-Equations.ipynb) | [Contents](Index.ipynb) | [14.2 Linear Transformations](chapter14.02-Linear-Transformations.ipynb) >
