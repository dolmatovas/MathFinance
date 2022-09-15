import numpy as np
from scipy import stats as sps


def genAntipath(Npaths):
    eps = np.random.randn(Npaths // 2)
    eps = np.r_[eps, -eps]
    if Npaths % 2:
        eps = np.r_[eps, np.random.randn(1)]
    return eps


def SabrBessel(F0: float, sig0: float, Npaths:int, tn:np.ndarray, alpha:float, beta:float, rho:float):
    '''docstring'''

    X = F0 ** (1-beta)
    y = np.log(sig0)

    for t in range(1, len(tn)):
        
        dt  = tn[t] - tn[t - 1]
        
        dW2 = genAntipath(Npaths) * np.sqrt(dt)
        dU  = genAntipath(Npaths) * np.sqrt(dt)
        
        dW1 = rho * dW2 + np.sqrt(1 - rho ** 2) * dU
     
        #update X
        b = -(X + (1-beta) * np.exp(y) * dW1)
        c = 0.5 * np.exp(2 * y) * beta * (1-beta) * dt
        D = b ** 2 - 4 * c
        D = np.maximum(D, 0.0)
        X = 0.5 * (-b + np.sqrt(D))
        X = np.maximum(X, 0.0)
        #update Y
        y += -0.5 * alpha ** 2 * dt + alpha * dW2
    
    #inverse transformation
    F = X ** (1 / (-beta + 1))
    return F


def SabrLog(F0: float, sig0: float, Npaths:int, tn:np.ndarray, alpha:float, beta:float, rho:float):
    '''docstring'''

    X = (beta - 1) * np.log(F0)
    
    y = np.log(sig0)
    
    for t in range(1, len(tn)):
        
        dt = tn[t] - tn[t - 1]
        
        dW2 = genAntipath(Npaths) * np.sqrt(dt)
        dU  = genAntipath(Npaths) * np.sqrt(dt)
        dW1 = rho * dW2 + np.sqrt(1 - rho ** 2) * dU
        
        X += 0.5 * (1-beta) * np.exp(2 * (X + y) ) * dt - (1-beta) * np.exp(X + y) * dW1 
        y += -0.5 * alpha ** 2 * dt + alpha * dW2
    F = np.exp(X / (beta - 1))
    return F


def SabrEuler(F0: float, sig0: float, Npaths:int, tn:np.ndarray, alpha:float, beta:float, rho:float):
    '''docstring'''

    F = F0 * np.ones((Npaths, ))
    
    y = np.log(sig0) * np.ones((Npaths, ))
    
    for t in range(1, len(tn)):
        
        dt = tn[t] - tn[t - 1]
        
        dW2 = genAntipath(Npaths) * np.sqrt(dt)
        dU  = genAntipath(Npaths) * np.sqrt(dt)
        dW1 = rho * dW2 + np.sqrt(1 - rho ** 2) * dU

        F += dW1 * np.exp(y) * (F ** beta)
        F = np.maximum(F, 0.0)
        
        y += -0.5 * (alpha ** 2) * dt + alpha * dW2
    return F


def SabrEulerModified(F0: float, sig0: float, Npaths:int, tn:np.ndarray, alpha:float, beta:float, rho:float):
    F = F0 * np.ones((Npaths, ))
    
    y = np.log(sig0) * np.ones((Npaths, ))
    
    for t in range(1, len(tn)):
        
        dt = tn[t] - tn[t - 1]
        
        dW2 = genAntipath(Npaths) * np.sqrt(dt)
        dU  = genAntipath(Npaths) * np.sqrt(dt)
        dW1 = rho * dW2 + np.sqrt(1 - rho ** 2) * dU

        F[F > 0] += dW1[F > 0] * np.exp(y[F > 0]) * (F[F > 0] ** beta) + 0.5 * beta * np.exp(2 * y[F > 0]) * F[F > 0] ** (2 * beta - 1) * (dW1[F > 0] ** 2 - dt)
        F = np.maximum(F, 0.0)
        
        y += -0.5 * (alpha ** 2) * dt + alpha * dW2
    return F