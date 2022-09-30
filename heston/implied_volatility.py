from scipy import stats as sps
from scipy.optimize import root_scalar
import numpy as np

def BS(K, S, r, tau, vol):
    F = S * np.exp(r * tau)
    d1 = (np.log(F / K) + 0.5 * vol ** 2 * tau) \
                / (vol * np.sqrt(tau))
    d2 = d1 - vol * np.sqrt(tau)
    
    D = np.exp(-r * tau)
    call_price =  D *  ( F * sps.norm.cdf(d1) - K * sps.norm.cdf(d2) )
    return call_price


def getIV(call_prices, Ks, S, r, tau):
    res = []
    for call_price, K in zip(call_prices, Ks):
        def foo(vol):
            call_price_ = BS(K, S, r, tau, vol)
            return call_price_ - call_price
 
        v0 = 1e-15
        v1 = 100
        br = foo(v0) <= 0 and foo(v1) >= 0
        vol = np.nan
        if br:
            vol = root_scalar(foo, bracket=[v0, v1], method='bisect').root
        res.append( vol )
    return np.asarray(res)