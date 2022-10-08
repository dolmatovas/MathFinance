import numpy as np
from scipy import stats as sps

def GeometricMeanAnalytical(r, sig, t, T, St, It, K):
    mu = It / T + (T-t) / T * np.log(St) + (r - sig ** 2 / 2) * (T-t)**2 / (2 * T)
    nu = sig * np.sqrt( (T-t)**3 / 3 ) / T
    d2 = (mu - np.log(K)) / nu
    d1 = d2 + nu
    
    I1 = np.exp(mu + nu ** 2 / 2) * sps.norm.cdf(d1)
    I2 = K * sps.norm.cdf(d2)
    
    return np.exp(-r * (T-t)) * (I1 - I2)
