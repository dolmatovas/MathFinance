from sys import path
path.append('../')


import numpy as np
import scipy.stats as sps

from numba import njit

from heston import getMesh

from typing import Tuple, Union

@njit
def getPhiAB(u:np.ndarray, tau:float, v0:float, theta:float, rho:float, k:float, sig:float) -> np.ndarray:
    '''
        return Characteristic function of log S_T
    '''
    xi = k - sig * rho * u * 1j
    d = np.sqrt( xi ** 2 + sig**2 * (1j * u + u ** 2) + 0j)
    
    s = np.sinh(d*tau/2)
    c = np.cosh(d*tau/2)
    A1 = (1j*u + u**2)*s
    A2 = d*c + xi*s
    A = A1 / A2
    
    D = np.log(d) + (k-d)*tau/2 - np.log((d+xi)/2 + (d-xi)/2*np.exp(-d * tau))
    
    pred_phi = np.exp(-k * theta * rho * tau * u * 1j / sig - A * v0 + 2 * k * theta / sig ** 2 * D)
    return pred_phi

@njit
def getPhiDerAB(u:np.ndarray, tau:float, v0:float, theta:float, 
                rho:float, k:float, sig:float)-> Tuple[np.ndarray, np.ndarray]:
    '''
        return characteristic function of log S_T phi and its derivatives with respect
        to heston parameters v0, theta, rho, k, sig
        
        output shapes:
        phi.shape == (len(tau). len(u))
        der.shape == (5, len(tau), len(u))
    '''
    xi = k - sig * rho * u * 1j
    d = np.sqrt( xi ** 2 + sig**2 * (1j * u + u ** 2) + 0j)
    
    
    c = np.cosh(d * tau / 2)
    s = np.sinh(d * tau / 2)
    
    A1 = (1j * u + u ** 2) * s
    A2 = (d * c + xi * s)
    A = A1 / A2
    
    B = d * np.exp(k * tau / 2) / A2
    
    D = np.log(d) + (k - d) * tau / 2 - np.log( (d + xi)/2 + (d-xi)/2 * np.exp(-d*tau) )
    
    phi = np.exp(-k * theta * rho * tau * u * 1j / sig - A * v0 + 2 * k * theta / sig ** 2 * D)
    
    der1 = -A
    der2 = 2 * k / sig ** 2 * D - k * rho * tau * 1j * u / sig
    
    d_rho = -1j * u * sig * xi / d

    A1_rho = -1j * u * sig * tau * xi / (2 * d) * (u ** 2 + 1j * u) * c
    A2_rho = -(2 + xi * tau) * sig * 1j * u / (2 * d) * (xi * c + d * s)
    
    B_rho = np.exp(k * tau / 2) * (d_rho - d * A2_rho / A2) / A2
    A_rho = (A1_rho - A * A2_rho) / A2
    
    D_rho = B_rho / B
       
    der3 = -k * theta * tau * 1j * u / sig - v0 * A_rho + 2 * k * theta / sig **2 * D_rho
    
    A_k = A_rho * 1j / (u * sig)
    B_k = tau / 2 * B + B_rho * 1j / (u * sig)
    D_k = B_k / B
    
    der4 = -theta * rho * tau * 1j * u / sig - v0 * A_k + 2 * theta / sig**2 * D + 2 * k * theta / sig ** 2 * D_k
    
    d_sig = (sig * (1j * u + u ** 2) + rho * 1j * u * (sig * rho * 1j * u - k)) / d
    A1_sig = (1j * u + u ** 2) * tau / 2 * c * d_sig
    A2_sig = d_sig * ( c * (1 + xi * tau / 2) + d * tau / 2 * s) - s * rho * 1j * u
    
    A_sig = (A1_sig - A * A2_sig) / A2
    
    D_sig = d_sig / d - A2_sig / A2
    
    der5 = k * theta * rho * tau * 1j * u / sig ** 2 - v0 * A_sig - 4 * k * theta / sig**3 * D\
        + 2 * k * theta / sig ** 2 * D_sig
    
    return phi, np.stack((der1, der2, der3, der4, der5))


def getPhiDerFinite(u, tau, v0, theta, rho, k, sig):
    pred_phi = getPhiAB(u, tau, v0, theta, rho, k, sig)
    
    eps = 1e-10
    denom = 1 / (2 * eps * pred_phi)
    der1 = (getPhiAB(u, tau, v0 + eps, theta, rho, k, sig) - getPhiAB(u, tau, v0 - eps, theta, rho, k, sig)) * denom 
    der2 = (getPhiAB(u, tau, v0, theta + eps, rho, k, sig) - getPhiAB(u, tau, v0, theta - eps, rho, k, sig)) * denom 
    der3 = (getPhiAB(u, tau, v0, theta, rho + eps, k, sig) - getPhiAB(u, tau, v0, theta, rho - eps, k, sig)) * denom 
    der4 = (getPhiAB(u, tau, v0, theta, rho, k + eps, sig) - getPhiAB(u, tau, v0, theta, rho, k - eps, sig)) * denom 
    der5 = (getPhiAB(u, tau, v0, theta, rho, k, sig + eps) - getPhiAB(u, tau, v0, theta, rho, k, sig - eps)) * denom 
    return pred_phi, np.stack((der1, der2, der3, der4, der5))


def getOptionPriceAB(S:Union[np.ndarray, float], K:Union[np.ndarray, float], tau:Union[np.ndarray, float], 
                    Nu:int, r:float, v0:float, theta:float, rho:float, k:float, sig:float, isCall=True) -> np.ndarray:
    '''
        return option price in heston model

        S -- current stock price. if S is np.ndarray, S should have the same length as K

        K -- strikes.

        tau -- time to expiration. If tau is np.ndarray, tau should have the same length as K

        Nu -- number of points in fourier integral

        r, v0, theta, rho, k, sig -- model parameters

        isCall -- boolean flag.
        
        output shape
        result.shape == (len(K), )
    '''    

    if not isinstance(K, np.ndarray):
        K = np.asarray([K])
    if isinstance(tau, np.ndarray):
        assert len(tau) == len(K)
    else:
        tau = np.asarray([tau])
    if isinstance(S, np.ndarray):
        assert len(S) == len(K)
    else:
        S = np.asarray([S])

    Nk = len(K)

    un, hn = getMesh(Nu)

    un = un.reshape(1, -1)
    hn = hn.reshape(1, -1)
    
    xn = np.log(S * np.exp(r * tau) / K).reshape(-1, 1)

    phi1 = getPhiAB(un, tau, v0, theta, rho, k, sig)
    phi2 = getPhiAB(un - 1j, tau, v0, theta, rho, k, sig)

    F1 = np.exp(1j * un * xn) * phi1 / (1j * un)
    F2 = np.exp(1j * un * xn) * phi2 / (1j * un)
    
    F1 = F1.real * hn
    F2 = F2.real * hn
    
    I1 = np.sum(F1, axis=-1) / np.pi
    I2 = np.sum(F2, axis=-1) / np.pi
    if isCall:
        P1 = 0.5 + I1
        P2 = 0.5 + I2
        res = S * P2 - np.exp(-r * tau) * K * P1
    else:
        P1 = 0.5 - I1
        P2 = 0.5 - I2
        res = np.exp(-r * tau) * K * P1 - S * P2
    return res



def getOptionPriceDerAB(S:Union[np.ndarray, float], K:Union[np.ndarray, float], tau:Union[np.ndarray, float], 
                     Nu:int, r:float, v0:float, theta:float, rho:float, k:float, sig:float, isCall=True) -> Tuple[np.ndarray, np.ndarray]:
    '''
        return option price and it derivatives with respect to heston params

        S -- current stock price. if S is np.ndarray, S should have the same length as K

        K -- strikes.

        tau -- time to expiration. If tau is np.ndarray, tau should have the same length as K

        Nu -- number of points in fourier integral

        r, v0, theta, rho, k, sig -- model parameters

        isCall -- boolean flag.
        
        output shape
        res.shape == (len(K), )
        resDer.shape == (5, len(K))
    '''        


    if not isinstance(K, np.ndarray):
        K = np.asarray([K])
    if isinstance(tau, np.ndarray):
        assert len(tau) == len(K)
    else:
        tau = np.asarray([tau])
    if isinstance(S, np.ndarray):
        assert len(S) == len(K)
    else:
        S = np.asarray([S])

    Nk = len(K)


    #grid for integral
    un, hn = getMesh(Nu)

    un = un.reshape(1, -1)
    hn = hn.reshape(1, -1)
    
    xn = np.log(S * np.exp(r * tau) / K).reshape(-1, 1)

    params = (tau.reshape(-1, 1), v0, theta, rho, k, sig)
    
    phi1, der1 = getPhiDerAB(un, *params)
    phi2, der2 = getPhiDerAB(un - 1j, *params)

    F1 = np.exp(1j * un * xn) * phi1 / (1j * un) * hn
    F2 = np.exp(1j * un * xn) * phi2 / (1j * un) * hn

    I1 = np.sum(F1.real, axis=-1) / np.pi
    I2 = np.sum(F2.real, axis=-1) / np.pi

    IDer1 = np.sum( (F1 * der1).real, axis=-1 ) / np.pi
    IDer2 = np.sum( (F2 * der2).real, axis=-1 ) / np.pi
    assert IDer1.shape == (5, Nk)

    if isCall:
        P1 = 0.5 + I1
        P2 = 0.5 + I2
        res = S * P2 - np.exp(-r * tau) * K * P1
    else:
        P1 = 0.5 - I1
        P2 = 0.5 - I2
        res = np.exp(-r * tau) * K * P1 - S * P2
    
    resDer = S * IDer2 - np.exp(-r * tau) * K * IDer1 
    assert resDer.shape == (5, Nk)
    return res, resDer


def getOptionPriceDerFinite(S, K, tau, Nu, r, v0, theta, rho, k, sig):
    C = getOptionPriceAB(S, K, tau, Nu, r, v0, theta, rho, k, sig)
    
    eps = 1e-10
    denom = 1 / (2 * eps)
    der1 = (getOptionPriceAB(S, K, tau, Nu, r, v0+eps, theta, rho, k, sig) - getOptionPriceAB(S, K, tau, Nu, r, v0-eps, theta, rho, k, sig)) * denom 
    der2 = (getOptionPriceAB(S, K, tau, Nu, r, v0, theta+eps, rho, k, sig) - getOptionPriceAB(S, K, tau, Nu, r, v0, theta-eps, rho, k, sig)) * denom 
    der3 = (getOptionPriceAB(S, K, tau, Nu, r, v0, theta, rho+eps, k, sig) - getOptionPriceAB(S, K, tau, Nu, r, v0, theta, rho-eps, k, sig)) * denom 
    der4 = (getOptionPriceAB(S, K, tau, Nu, r, v0, theta, rho, k+eps, sig) - getOptionPriceAB(S, K, tau, Nu, r, v0, theta, rho, k-eps, sig)) * denom 
    der5 = (getOptionPriceAB(S, K, tau, Nu, r, v0, theta, rho, k, sig+eps) - getOptionPriceAB(S, K, tau, Nu, r, v0, theta, rho, k, sig-eps)) * denom 
    return C, np.stack([der1, der2, der3, der4, der5])