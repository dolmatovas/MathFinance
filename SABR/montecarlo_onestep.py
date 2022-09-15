import numpy as np
from scipy import stats as sps

from montecarlo import *

def IntegralVarSamplingMy(Npaths:int, sig0:float, alpha:float, dt:float):
    Nt = 20
    tn = np.linspace(0, dt, Nt + 1)

    sig = sig0

    A = 0
    for t in range(1, Nt + 1):
        tau = tn[t] - tn[t - 1]
        eps = genAntipath(Npaths) * np.sqrt(tau)
        sig_ = sig * np.exp( alpha * eps - 0.5 * alpha ** 2 * tau )
        A += tau * 0.5 * (sig ** 2 + sig_ ** 2)
        sig = sig_
    return sig, A


def IntegralVarSampling(Npaths:int, sig0:float, alpha:float, dt:float):
    W = genAntipath(Npaths) * np.sqrt(dt)
    sig = sig0 * np.exp( alpha * W - 0.5 * alpha ** 2 * dt)
    
    m1 = W
    m2 = (2 * W**2 - dt / 2) / 3
    m3 = (W**3 - W * dt) / 3
    m4 = (2 * W ** 4 / 3- 3 * W ** 2 * dt / 2 + 2 * dt ** 2) / 5
   
    m = sig0 ** 2 * dt * ( 1 + alpha * m1 + alpha ** 2 * m2 + alpha ** 3 * m3 + alpha ** 4 * m4 )
    v = sig0 ** 4 * alpha ** 2 * dt ** 3 / 3.0
    
    #mean of normal variable X = log A
    mu = np.log(m) - 0.5 * np.log(1 + v / m ** 2)
    #sigma of normal variable X = log A
    sigma = np.sqrt(np.log(1 + v / m ** 2))
    
    U = genAntipath(Npaths)
    A = np.exp(sigma * U + mu)
    
    return sig, A


def GenNonzero(F0, sig, sig0, v, alpha, beta, rho):
    Npaths = len(F0)
    x = 1 - beta
    lam = (F0 ** x / x + rho / alpha * (sig - sig0) ) ** 2 / v
    k = (1 - rho ** 2 * x) / (x * (1 - rho ** 2))
    Y = np.random.noncentral_chisquare(k, lam, Npaths)
    F = (x ** 2 * v * Y) ** (1 / (2 * x))
    #return F, sig
    m = k + lam
    s = np.sqrt(2 * (k + 2 * lam))



    psi = s ** 2 / m ** 2
    
    e = np.sqrt( 2 / psi - 1 + np.sqrt(2/psi) * np.sqrt(2/psi - 1) )
    
    d = m / (1 + e ** 2)
    
    Z = genAntipath(Npaths)

    #U, h = np.linspace( 0, 1, Npaths + 1, retstep=True)
    #Z = sps.norm.isf(U[:-1] + h/2)
    Y = d * (e + Z) ** 2
    F = (x ** 2 * v * Y) ** (1 / (2 * x))
    return F, sig

from scipy.special import gammainc
def DirectInversionSchemeStep(F0:float, sig0:float, Npaths:int, dt:float, alpha:float, beta:float, rho:float):
    
    sig, A = IntegralVarSampling(Npaths, sig0, alpha, dt)
    
    v = (1 - rho ** 2) * A
    
    x = 1 - beta
    
    a = 1 / (2 * x)
    lim = F0 ** (2 * x) / (2 * x ** 2 * v)

    ProbZero = 1.0 - gammainc(a, lim)
    U = np.random.rand(Npaths)
    zero = (U < ProbZero) | (F0 == 0.0)
    F = np.zeros((Npaths, ))
    F[~zero], sig[~zero] = GenNonzero(F0[~zero], sig[~zero], sig0[~zero], v[~zero], alpha, beta, rho)
    return F, sig


def SabrDirectInversionScheme(F0:float, sig0:float, Npaths:int, tn:np.ndarray, alpha:float, beta:float, rho:float):
    F = F0 * np.ones((Npaths,))
    sig = sig0 * np.ones((Npaths,))
    for t in range(1, len(tn)):
        dt = tn[t] - tn[t-1]

       # dt = tn[-1] - tn[0]
        F, sig = DirectInversionSchemeStep(F, sig, Npaths, dt, alpha, beta, rho)
    return F