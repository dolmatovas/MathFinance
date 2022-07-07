import numpy as np
from numpy.matlib import repmat
from numba import jit, njit

from utils import *

#@njit
def getRhs(v, xn, yn, r, sigma, kappa, theta, rho):
    sl = slice(1, -1, 1)
    ux, uy, uxx, uyy, uxy = np.zeros_like(v), np.zeros_like(v), np.zeros_like(v), np.zeros_like(v), np.zeros_like(v)
    #shoud replace with notunirofm grid
    h1 = xn[1:-1] - xn[:-2]
    h2 = xn[2:] - xn[1:-1]

    d1 = yn[1:-1] - yn[:-2]
    d2 = yn[2:] - yn[1:-1]

    h1 = h1.reshape(-1, 1)
    h2 = h2.reshape(-1, 1)
    d1 = d1.reshape(1, -1)
    d2 = d2.reshape(1, -1)
    ux[sl, sl] = ((h1 / h2) * (v[2:, sl] - v[1:-1, sl]) + (h2 / h1) * (v[1:-1, sl] - v[:-2, sl]) ) / (h1 + h2)
    uy[sl, sl] = ((d1 / d2) * (v[sl, 2:] - v[sl, 1:-1]) + (d2 / d1) * (v[sl, 1:-1] - v[sl, :-2]) ) / (d1 + d2)

    uxx[sl, sl] = ( (v[2:, sl] - v[1:-1, sl]) / h2 - (v[1:-1, sl] - v[:-2, sl]) / h1 ) / (h1 + h2)
    uyy[sl, sl] = ( (v[sl, 2:] - v[sl, 1:-1]) / d2 - (v[sl, 1:-1] - v[sl, :-2]) / d1 ) / (d1 + d2)

    uxy[sl, sl] = ( (h1 / h2) * (uy[2:, sl] - uy[1:-1, sl]) + (h2 / h1) * (uy[1:-1, sl] - uy[:-2, sl]) ) / (h1 + h2)
    f = (r - 0.5 * yn * sigma) * ux + \
        kappa * (theta - yn * sigma) / sigma * uy + \
        0.5 * sigma * yn * (uxx + uyy) + \
        rho * sigma * yn * uxy 
    return f


def solve_euler(hestonParams : HestonParams,
                optionParams: OptionParams,
                gridParams: GridParams):
    
    r, sigma, kappa, theta, rho = hestonParams.r,hestonParams.sigma, hestonParams.kappa,hestonParams.theta,hestonParams.rho
    tn, xn, yn = GetGrid(hestonParams, optionParams, gridParams)
    tau = tn[1] - tn[0]
    hx = xn[1] - xn[0]
    hy = yn[1] - yn[0]
    
    u = np.zeros(tn.shape + xn.shape + yn.shape)
    u[0, :, :] = repmat(np.maximum(1.0 - np.exp(xn), 0.0), gridParams.Ny + 1, 1).T
    sl = slice(1, -1, 1)
    
    for t in range(gridParams.M):
        f = getRhs(u[t], xn, yn, r, sigma, kappa, theta, rho)
        u[t + 1, sl, sl] = u[t, sl, sl] + tau * f[sl, sl]

        u[t + 1, :, 0]  = u[t + 1, :, 1]
        u[t + 1, :, -1] = u[t + 1, :, -2]

        u[t + 1, 0, :]  = 1.0 - np.exp(xn[0])
        u[t + 1, -1, :] = 0.0
    return u


def solve_runge_kutta(hestonParams : HestonParams,
                optionParams: OptionParams,
                gridParams: GridParams):
    
    r, sigma, kappa, theta, rho = hestonParams.r,hestonParams.sigma, hestonParams.kappa,hestonParams.theta,hestonParams.rho
    tn, xn, yn = GetGrid(hestonParams, optionParams, gridParams)
    tau = tn[1] - tn[0]
    hx = xn[1] - xn[0]
    hy = yn[1] - yn[0]
    
    u = np.zeros(tn.shape + xn.shape + yn.shape)
    u[0, :, :] = repmat(np.maximum(1.0 - np.exp(xn), 0.0), gridParams.Ny + 1, 1).T
    sl = slice(1, -1, 1)
    
    for t in range(gridParams.M):
        k1 = getRhs(u[t],xn, yn, r, sigma, kappa, theta, rho)
        k1[:, 0] = k1[ :, 1]
        k1[:, -1] = k1[:, -2]
        k2 = getRhs(u[t] + 0.5 * tau * k1, xn, yn, r, sigma, kappa, theta, rho)
        k2[:, 0] = k2[:, 1]
        k2[:, -1] = k2[:, -2]
        k3 = getRhs(u[t] + 0.5 * tau * k2, xn, yn, r, sigma, kappa, theta, rho)
        k3[:, 0] = k3[:, 1]
        k3[:, -1] = k3[:, -2]
        k4 = getRhs(u[t] + tau * k3, xn, yn, r, sigma, kappa, theta, rho)
        k4[:, 0] = k4[:, 1]
        k4[:, -1] = k4[:, -2]
        
        u[t + 1, sl, sl] = u[t, sl, sl] + tau / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)[sl, sl]
        
        u[t + 1, :, 0] = u[t + 1, :, 1]
        u[t + 1, :, -1] = u[t + 1, :, -2]
        u[t + 1, 0, :] = 1.0 - np.exp(xn[0])
        u[t + 1, -1, :] = 0.0
    return u