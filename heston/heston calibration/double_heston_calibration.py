from sys import path
path.append('../')

from heston_derivatives import *
from implied_volatility import *

from typing import Tuple, Union

def getOptionPriceABDouble(S:Union[np.ndarray, float], K:Union[np.ndarray, float], tau:Union[np.ndarray, float], 
                    Nu:int, r:float, heston_params:np.ndarray, isCall=True) -> np.ndarray:
    '''
        return option price in Double heston model

        S -- current stock price. if S is np.ndarray, S should have the same length as K

        K -- strikes.

        tau -- time to expiration. If tau is np.ndarray, tau should have the same length as K

        Nu -- number of points in fourier integral

        r -- interest rate

        heston_params -- array of heston params. 

        isCall -- boolean flag.
        
        output shape
        res.shape == (len(K), )
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
    
    phi1 = np.ones((len(tau), Nu), complex)
    phi2 = np.ones((len(tau), Nu), complex)
    
    #размерность
    Ndim = len(heston_params) // 5
    for i in range(Ndim):
        v, theta, rho, k, sig = heston_params[5 * i : 5 * i + 5]
        params = {"v0":v, "theta":theta, "rho":rho, "k":k, "sig":sig}

        _phi1 = getPhiAB(un, tau.reshape(-1, 1), **params)
        _phi2 = getPhiAB(un - 1j, tau.reshape(-1, 1), **params)
        
        phi1 *= _phi1
        phi2 *= _phi2
        
        
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


def getOptionPriceDerABDouble(S:Union[np.ndarray, float], K:Union[np.ndarray, float], tau:Union[np.ndarray, float], 
                    Nu:int, r:float, heston_params:np.ndarray, isCall=True) -> Tuple[np.ndarray, np.ndarray]:
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
        resDer.shape == (len(heston_params), len(K))
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
    Nt = len(tau)
    Ndim = len(heston_params) // 5

    un, hn = getMesh(Nu)

    un = un.reshape(1, -1)
    hn = hn.reshape(1, -1)
    
    xn = np.log(S * np.exp(r * tau) / K).reshape(-1, 1)
    
    phi1 = np.ones((Nt, Nu), complex)
    phi2 = np.ones((Nt, Nu), complex)



    der1 = np.zeros( (Ndim * 5, Nt, Nu), complex )
    der2 = np.zeros( (Ndim * 5, Nt, Nu), complex )

    
    for i in range(Ndim):
        v, theta, rho, k, sig = heston_params[5 * i : 5 * i + 5]
        params = {"v0":v, "theta":theta, "rho":rho, "k":k, "sig":sig}

        _phi1, _der1 = getPhiDerAB(un     , tau.reshape(-1, 1), **params)
        _phi2, _der2 = getPhiDerAB(un - 1j, tau.reshape(-1, 1), **params)
        
        assert _phi1.shape == (Nt, Nu)
        assert _der1.shape == (5, Nt, Nu)

        phi1 = phi1 * _phi1
        phi2 = phi2 * _phi2
        
        der1[5 * i : 5 * i + 5, :] = _der1
        der2[5 * i : 5 * i + 5, :] = _der2

    F1 = np.exp(1j * un * xn) * phi1 / (1j * un) * hn
    F2 = np.exp(1j * un * xn) * phi2 / (1j * un) * hn

    I1 = np.sum(F1.real, axis=-1) / np.pi
    I2 = np.sum(F2.real, axis=-1) / np.pi

    IDer1 = np.sum( (F1 * der1).real, axis=-1 ) / np.pi
    IDer2 = np.sum( (F2 * der2).real, axis=-1 ) / np.pi
    assert IDer1.shape == (5 * Ndim, Nk)

    if isCall:
        P1 = 0.5 + I1
        P2 = 0.5 + I2
        res = S * P2 - np.exp(-r * tau) * K * P1
    else:
        P1 = 0.5 - I1
        P2 = 0.5 - I2
        res = np.exp(-r * tau) * K * P1 - S * P2
    
    resDer = S * IDer2 - np.exp(-r * tau) * K * IDer1 
    assert resDer.shape == (5 * Ndim, Nk)
    return res, resDer


def getVolatilitySurfaceDouble(S, Kn, Tn, Nu, r, heston_params):
    Nk = len(Kn)
    Nt = len(Tn)
    C = np.zeros((Nk, Nt))
    IV = np.zeros((Nk, Nt))
    for t, tau in enumerate(Tn):
        C[:, t] = getOptionPriceABDouble(S, Kn, tau, Nu, r, heston_params)
        IV[:, t] = getIV(C[:, t], Kn, S, r, tau)
    return C, IV


def getResudalAndGradDouble(C0, S0, Kn, Tn, Nu, r, weights, heston_params):
    Nt = len(Tn)
    Nk = len(Kn)
    res  = np.zeros((0, ))
    J = np.zeros((len(heston_params), 0))
    for t in range(Nt):
        w = weights[t]
        c, ders = getOptionPriceDerABDouble(S0, Kn, Tn[t], Nu, r, heston_params)
        _res = c.reshape(-1) - C0[:, t].reshape(-1)
        res = np.r_[res, _res * w]
        J = np.c_[J, ders @ np.diag(w)]
    return res, J