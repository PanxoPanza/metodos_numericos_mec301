from numpy import exp, log10, sqrt
from scipy.optimize import fsolve

def Iplanck(lam, T):
    """ Spectral radiance of a black-body at temperature T.

    Returns the spectral radiance, B(lam, T), in W.sr-1.m-2.um-2 of a black body
    at temperature T (in K) at a wavelength lam (in um), using Planck's law.
    
    Input
        lam: ndarray 
             wavelength (um)
        T: float
            Temperatura (K)
    Returns
        Ilam: ndarray 
            Spectral irradiance of blackbody at temperature T (W.sr-1.m-2.um-2)
    """

    C1 = 1.19104238E8                        # W*um^2/m^2
    C2 = 1.43878E4                           # um*K
    Ilam = C1/lam**5 / (exp(C2/(lam*T)) - 1) # W/m^2*um*sr
    return Ilam

def Isun(lam):
    return Iplanck(lam, 5777)

def f_colebrook(Re, eps_r):
    
    # ecuación de colebrook
    fun = lambda x: 1/sqrt(x) + 2.0*log10(eps_r/3.7 + 2.51/(Re*sqrt(x)))
    f = fsolve(fun,0.002) # raíz de la ecuacion de Colebrook
    return f