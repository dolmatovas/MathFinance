import numpy as np
from scipy import stats as sps
from scipy.optimize import root_scalar

def BS(K, F, r, tau, vol):
    d1 = (np.log(F / K) + 0.5 * vol ** 2 * tau) \
                / (vol * np.sqrt(tau))
    d2 = d1 - vol * np.sqrt(tau)
    
    D = np.exp(-r * tau)
    call_price =  D *  ( F * sps.norm.cdf(d1) - K * sps.norm.cdf(d2) )
    return call_price


def GetIv(call_prices, Ks, F, r, tau):
    
    res = []
    for call_price, K in zip(call_prices, Ks):
        def foo(vol):
            call_price_ = BS(K, F, r, tau, vol)
            return call_price_ - call_price
 
        v0 = 1e-15
        v1 = 100
        
        br = foo(v0) <= 0 and foo(v1) >= 0
        vol = np.nan
        if not br:
            if foo(v0) > 1e-10:
                vol = 0.0
            elif foo(v1) < 1e-10:
                vol = np.nan
        else:
            vol = root_scalar(foo, bracket=[v0, v1], method='bisect').root
        res.append( vol )
    return np.asarray(res)


def SabrApprox(K, F, r, tau, sig, alpha, beta, rho):
    
    #this function has problems if F = K.
    #need to think, how to fix this
   
    Fmid = np.sqrt( F * K )
    C = lambda _x : _x ** beta
    
    x = 1 - beta
    dzeta = alpha / (sig * x) * ( F ** x - K ** x )
    
    gamma1 = beta / Fmid
    gamma2 = -beta * x / Fmid**2
    D = np.log( (  np.sqrt(1 - 2 * rho * dzeta + dzeta ** 2) + dzeta - rho  ) / (1-rho) )
    eps = tau * alpha ** 2
    
    frc1 = (2 * gamma2 - gamma1 ** 2+ 1 / Fmid ** 2) / 24 * (sig / alpha * C(Fmid)) ** 2
    
    frc2 = rho * gamma1 / (4 * alpha) * sig * C(Fmid)
    
    frc3 = (2-3*rho**2)/24 
    
    vol = alpha * np.log(F / K) / D * (1 + eps * (frc1 + frc2 + frc3) )
    
    call_price =  BS(K, F, r, tau, vol)
    return call_price, vol