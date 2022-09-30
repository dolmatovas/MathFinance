import numpy as np
import scipy.stats as sps

from numba import njit

from heston_derivatives import *
from implied_volatility import *

def getVolatilitySurface(S, Kn, Nu, Tn, r, *heston_params):
    Nk = len(Kn)
    Nt = len(Tn)
    C = np.zeros((Nk, Nt))
    IV = np.zeros((Nk, Nt))
    for t, tau in enumerate(Tn):
        C[:, t] = getOptionPriceAB(S, Kn, Nu, tau, r, *heston_params)
        IV[:, t] = getIV(C[:, t], Kn, S, r, tau)
    return C, IV


def getResudalAndGrad(C0, S0, Kn, Nu, Tn, r, weights, *heston_params):
    Nt = len(Tn)
    Nk = len(Kn)
    res  = np.zeros((0, ))
    J = np.zeros((5, 0))
    for t in range(Nt):
        w = weights[t]
        c, ders = getOptionPriceDerAB(S0, Kn, Nu, Tn[t], r, *heston_params)
        ders = np.asarray(ders)
        _res = c.reshape(-1) - C0[:, t].reshape(-1)
        res = np.r_[res, _res * w]
        J = np.c_[J, ders @ np.diag(w)]
    return res, J


def my_clip(heston_params):
    eps = 1e-6
    for i in range(len(heston_params) // 5):
        v0, theta, rho, k, sig = heston_params[i * 5 : i * 5 + 5]
        v0 = max(v0, eps)
        theta = max(theta, eps)
        rho = np.clip(rho, -1 + eps, 1 - eps)
        k = max(k, eps)
        sig = max(sig, eps)
        heston_params[i * 5 : i * 5 + 5] = v0, theta, rho, k, sig
    return heston_params
    
def MyAlgorithm(Niter, f, proj, x0):
    x = x0.copy()

    mu = 100.0
    nu1 = 2.0
    nu2 = 2.0

    fs = []
    res, J = f(x)
    F = np.linalg.norm(res)
    for i in range(Niter):
        I = np.diag(np.diag(J @ J.T))
        dx = np.linalg.solve( mu * I + J @ J.T, J @ res )
        x_ = proj(x - dx)
        res_, J_ = f(x_)
        F_ = np.linalg.norm(res_)
        if F_ < F:
            x, F, res, J = x_, F_, res_, J_
            mu /= nu1
        else:
            i -= 1
            mu *= nu2
            continue
        fs.append(F)
        eps = 1e-10
        if F < eps:
            break
    return x, fs