import numpy as np
from numpy.matlib import repmat
from numba import jit, njit

from derivatives import *

@njit
def Progonka(A, B, C, F):
    N = A.shape[0]
    X = np.zeros_like(A)
    
    beta = np.zeros_like(A)
    #phi = np.zeros((N, ))
    
    beta[0] = B[0] / A[0]
    X[0] = F[0] / A[0]
    
    for i in range(1, N):
        beta[i] = B[i] / (A[i] - beta[i - 1] * C[i])
        X[i] = (F[i] - C[i] * X[i - 1]) / (A[i] - beta[i - 1] * C[i])
    #X[-1] = phi[-1]
    for i in range(N-2, -1, -1):
        X[i] -=  X[i + 1] * beta[i]
    return X

def get_new_coeffs(_A, _B, _C, alpha_l, beta_l, gamma_l, alpha_r, beta_r, gamma_r, _F):
    A = _A.copy()
    B = _B.copy()
    C = _C.copy()
    F = _F.copy()

    frc = gamma_l / B[1]

    A[0] = alpha_l - C[1] * frc
    B[0] = beta_l - A[1] * frc
    F[0] = F[0] - F[1] * frc


    frc = gamma_r / C[-2]
    A[-1] = alpha_r - B[-2] * frc
    C[-1] = beta_r - A[-2] * frc
    F[-1] = F[-1] - F[-2] * frc
    return A, B, C, F

@njit
def Progonka_coefs(_A, _B, _C, alpha_l, beta_l, gamma_l, alpha_r, beta_r, gamma_r, _F):

    A = _A.copy()
    B = _B.copy()
    C = _C.copy()
    F = _F.copy()


    frc = gamma_l / B[1]
    A[0] = alpha_l - C[1] * frc
    B[0] = beta_l - A[1] * frc
    F[0] = F[0] - F[1] * frc

    frc = gamma_r / C[-2]
    A[-1] = alpha_r - B[-2] * frc
    C[-1] = beta_r - A[-2] * frc
    F[-1] = F[-1] - F[-2] * frc

    return Progonka(A, B, C, F)


@njit
def LU(a, b, c):
    N = len(a)
    l = np.zeros((N, ))
    u = np.zeros((N, ))
    v = b.copy()
    
    u[0] = a[0]
    for i in range(1, N):
        l[i] = c[i] / u[i-1]
        u[i] = a[i] - l[i] * v[i-1]
    return l, u, v


@njit
def SolveLU(l, u, v, f):
    N = len(l)
    y = np.zeros((N, ))
    x = np.zeros((N, ))
    
    y[0] = f[0]
    for i in range(1, N):
        y[i] = f[i] - l[i] * y[i - 1]
    
    x[-1] = y[-1] / u[-1]
    for i in range(N-2, -1, -1):
        x[i] = (y[i] - v[i] * x[i + 1]) / u[i]
    return x