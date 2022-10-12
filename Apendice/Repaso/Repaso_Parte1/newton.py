import numpy as np

def divided_diff(x, y):
    '''
    función para generar los coeficientes del
    polinomio de Newton de orden "n"
    '''
    n = len(y)
    coef = np.zeros([n, n])
    coef[:,0] = y # primera columna yi
    
    for j in range(1,n):
        for i in range(n-j):  
            coef[i][j] = (coef[i+1][j-1] - coef[i][j-1]) /  \
                                  (x[i+j]-x[i])
            
    return coef[0, :] # retornamos la primera fila

def newton_poly(coef, x_data, x):
    '''
    evalúa el polinomio de newton en x con los coeficientes
    de divided_diff
    '''
    n = len(x_data) - 1 
    p = coef[n]
    for k in range(1,n+1):
        p = coef[n-k] + (x -x_data[n-k])*p
    return p