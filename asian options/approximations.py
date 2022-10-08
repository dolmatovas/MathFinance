import numpy as np
from scipy import stats as sps

from genPathes import *
from monte_carlo import *

def LognormalApprox(r:float, sig:float, t:float, T:float, St:float, It:float, K:np.ndarray):
    #Approximate integal int_0^T S_t dt with lognormal distribution    
    tau = T - t
    
    mu, var = ArithmeticMeanMoments(r, sig, t, T, St, It)

    m1 = mu
    m2 = var + mu ** 2
        
    nu = np.sqrt( np.log(m2 / m1 ** 2) )
    mu = np.log(m1) - nu ** 2 / 2
    
    d2 = (mu - np.log(K)) / nu
    d1 = d2 + nu
    
    I1 = np.exp(mu + nu ** 2 / 2) * sps.norm.cdf(d1)
    I2 = K * sps.norm.cdf(d2)
    
    return np.exp(-r * tau) * (I1 - I2)


def ConditionalMoments(r:float, sig:float, t, T:float, St:float, ST:float, It):
    '''
        The first and second conditional moments of IT = int_0^T S_t dt given ST.
    '''

    Phi = sps.norm.cdf
    
    x = np.log(ST / St)
    tau = T - t

    sigT = sig * np.sqrt(tau)
    
    p = 1 / np.sqrt(2 * np.pi * sigT ** 2) * np.exp( -(sigT ** 2 / 2 + x) ** 2 / (2 * sigT ** 2) )
    
    q = 1 / np.sqrt(2 * np.pi * sigT ** 2) * np.exp( -(sigT ** 2 + x) ** 2 / (2 * sigT ** 2) )
    
    
    d1 = x / (sigT) + 0.5 * sigT
    d2 = x / (sigT) - 0.5 * sigT
    d3 = x / (sigT) + 1.0 * sigT
    d4 = x / (sigT) - 1.0 * sigT
    
    a = 1 / (sig ** 2 * p) * ( Phi(d1) - Phi(d2) )
    b = 1 / (sig ** 2 * q) * ( Phi(d3) - Phi(d4) )
    
    Imean = St * a
    I2mean = (St**2) * 2 / sig ** 2 * (b - (1 + np.exp(x)) * a)
    Ivar = I2mean - Imean ** 2

    Amean = (It + Imean) / T
    Avar = Ivar / (T ** 2)
    
    return Amean, Avar  


def StratifiedLognormalApprox(r:float, sig:float, t:float, T:float, St:float, It:float, K:np.ndarray):
    Ns = 100
    tmp, h = np.linspace(0, 1, Ns + 1, retstep=True)
    
    xi     = sps.norm.ppf(tmp)
    ximid  = sps.norm.ppf(tmp[:-1] + 0.5 * h)
    
    res = 0.0
    tau = T - t
    for i in range(Ns):
        S1 = St * np.exp( (r - 0.5 * sig ** 2) * tau + sig * np.sqrt(tau) * xi[i] )
        S2 = St * np.exp( (r - 0.5 * sig ** 2) * tau + sig * np.sqrt(tau) * xi[i + 1] )
        
        d1 = -np.inf if i == 0 else (np.log(S1 / St) - (r - sig ** 2 / 2) * tau) / (sig * np.sqrt(tau))
        d2 = np.inf if i == Ns - 1 else (np.log(S2 / St) - (r - sig ** 2 / 2) * tau) / (sig * np.sqrt(tau))

        step = sps.norm.cdf(d2) - sps.norm.cdf(d1)
        

        ST = St * np.exp( (r - 0.5 * sig ** 2) * tau + sig * np.sqrt(tau) * ximid[i] )
        #moments of AT
        Amean, Avar = ConditionalMoments(r, sig, t, T, St, ST, It) 

        m1 = Amean
        m2 = Avar + Amean ** 2   
        
        #parameters of lognormal approximation
        nu = np.sqrt( np.log(m2 / m1 ** 2) )
        mu = np.log(m1) - nu ** 2 / 2
        
        d2 = (mu - np.log(K)) / nu
        d1 = d2 + nu
        
        I1 = np.exp(mu + nu ** 2 / 2) * sps.norm.cdf(d1)
        I2 = K * sps.norm.cdf(d2)
        C = np.exp(-r * tau) * (I1 - I2)
        res += C * step
    return res