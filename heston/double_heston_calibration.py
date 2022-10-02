from heston_derivatives import *
from implied_volatility import *

def getOptionPriceABDouble(S, K, Nu, tau, r, heston_params, isCall=True):
    if not isinstance(K, np.ndarray):
        K = np.asarray([K])
    K = K.reshape(-1, 1)

    Nk = K.size

    un, hn = getMesh(Nu)

    un = un.reshape(1, -1)
    hn = hn.reshape(1, -1)
    
    xn = np.log(S * np.exp(r * tau) / K)
    
    phi  = np.ones((1, Nu), complex)
    phiT = np.ones((1, Nu), complex)
    
    for i in range(len(heston_params) // 5):
        v, theta, rho, k, sig = heston_params[5 * i : 5 * i + 5]
        params = {"v0":v, "theta":theta, "rho":rho, "k":k, "sig":sig}

        _phi = getPhiAB(un, tau, **params)
        _phiT = getPhiAB(un - 1j, tau, **params)
        
        phi  *= _phi
        phiT *= _phiT
        
        
    F1 = np.exp(1j * un * xn) * phi  / (1j * un)
    F2 = np.exp(1j * un * xn) * phiT / (1j * un)

    I1 = np.sum( (F1 * hn).real, axis=-1, keepdims=True ) / np.pi
    I2 = np.sum( (F2 * hn).real, axis=-1, keepdims=True ) / np.pi
    if isCall:
        P1 = 0.5 + I1
        P2 = 0.5 + I2
        res = S * P2 - np.exp(-r * tau) * K * P1
    else:
        P1 = 0.5 - I1
        P2 = 0.5 - I2
        res = np.exp(-r * tau) * K * P1 - S * P2
    return res.reshape(-1)


def getOptionPriceDerABDouble(S, K, Nu, tau, r, heston_params, isCall=True):
    if not isinstance(K, np.ndarray):
        K = np.asarray([K])
    if isinstance(tau, np.ndarray):
        assert len(tau) == len(K)
        tau = tau.reshape(-1, 1)
    if isinstance(S, np.ndarray):
        assert len(S) == len(K)
        S = S.reshape(-1, 1)    
    K = K.reshape(-1, 1)

    Nk = K.size

    un, hn = getMesh(Nu)

    un = un.reshape(1, -1)
    hn = hn.reshape(1, -1)
    
    xn = np.log(S * np.exp(r * tau) / K)
    
    phi  = np.ones((1, Nu), complex)
    phiT = np.ones((1, Nu), complex)
    
    ders  = []
    dersT = []
    
    for i in range(len(heston_params) // 5):
        v, theta, rho, k, sig = heston_params[5 * i : 5 * i + 5]
        params = {"v0":v, "theta":theta, "rho":rho, "k":k, "sig":sig}

        _phi,  _ders  = getPhiDerAB(un     , tau, **params)
        _phiT, _dersT = getPhiDerAB(un - 1j, tau, **params)
        
        phi  = phi * _phi
        phiT = phiT * _phiT
        
        ders  = ders + _ders
        dersT = dersT + _dersT

    F1 = np.exp(1j * un * xn) * phi  / (1j * un)
    F2 = np.exp(1j * un * xn) * phiT / (1j * un)

    I1, Ders1 = calc_int(F1, hn, ders)
    I2, Ders2 = calc_int(F2, hn, dersT)

    if isCall:
        P1 = 0.5 + I1
        P2 = 0.5 + I2
        res = S * P2 - np.exp(-r * tau) * K * P1
    else:
        P1 = 0.5 - I1
        P2 = 0.5 - I2
        res = np.exp(-r * tau) * K * P1 - S * P2
    
    res_der = []
    for der1, der2 in zip(Ders1, Ders2):
        tmp = S * der2 - np.exp(-r * tau) * K * der1
        res_der.append( tmp.reshape(-1) )
    return res.reshape(-1), res_der


def getVolatilitySurfaceDouble(S, Kn, Nu, Tn, r, heston_params):
    Nk = len(Kn)
    Nt = len(Tn)
    C = np.zeros((Nk, Nt))
    IV = np.zeros((Nk, Nt))
    for t, tau in enumerate(Tn):
        C[:, t] = getOptionPriceABDouble(S, Kn, Nu, tau, r, heston_params)
        IV[:, t] = getIV(C[:, t], Kn, S, r, tau)
    return C, IV


def getResudalAndGradDouble(C0, S0, Kn, Nu, Tn, r, weights, heston_params):
    Nt = len(Tn)
    Nk = len(Kn)
    res  = np.zeros((0, ))
    J = np.zeros((len(heston_params), 0))
    for t in range(Nt):
        w = weights[t]
        c, ders = getOptionPriceDerABDouble(S0, Kn, Nu, Tn[t], r, heston_params)
        ders = np.asarray(ders)
        _res = c.reshape(-1) - C0[:, t].reshape(-1)
        res = np.r_[res, _res * w]
        J = np.c_[J, ders @ np.diag(w)]
    return res, J