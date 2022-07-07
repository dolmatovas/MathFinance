import numpy as np
from numpy.matlib import repmat
from numba import jit, njit

@njit
def Progonka(A, B, C, F):
    N = len(A)
    X = np.zeros((N, ))
    
    beta = np.zeros((N, ))
    phi = np.zeros((N, ))
    
    beta[0] = B[0] / A[0]
    phi[0] = F[0] / A[0]
    
    for i in range(1, N):
        beta[i] = B[i] / (A[i] - beta[i - 1] * C[i])
        phi[i] = (F[i] - C[i] * phi[i - 1]) / (A[i] - beta[i - 1] * C[i])
    X[-1] = phi[-1]
    for i in range(N-2, -1, -1):
        X[i] = phi[i] - X[i + 1] * beta[i]
    return X

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