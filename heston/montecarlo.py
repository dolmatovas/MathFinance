import numpy as np
import scipy.stats as sps


def generate_path_euler(r, k, sig, theta, rho, x0, v0, tn, Npath):
    Nt = len(tn)
    shape = (Nt, Npath // 2)
    
    v = np.ones((Npath, )) * v0
    x = np.ones((Npath, )) * x0
    
    for i in range(Nt - 1):
        
        Z1 = np.random.randn(Npath // 2)
        Z2 = np.random.randn(Npath // 2)

        Z1 = np.r_[Z1, -Z1]
        Z2 = np.r_[Z2, -Z2]

        dW1 = Z1
        dW2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2
        
        tau = tn[i + 1] - tn[i]
        v += k * (theta - v) * tau + tau * 0.5 * (dW1**2 - 1) \
                                 + sig * np.sqrt(v * tau) * dW1
        v = np.maximum(v, 0)
        x += (r - v * 0.5) * tau + np.sqrt(v * tau) * dW2
    return v, x


def generate_path_chi(r, k, sig, theta, rho, x0, v0, tn, Npath):
    Nt = len(tn)
    shape = (Nt, Npath // 2)
    
    v = np.ones((Npath, )) * v0
    x = np.ones((Npath, )) * x0
    
    for i in range(Nt - 1):
        
        Z1 = np.random.randn(Npath // 2)

        Z1 = np.r_[Z1, -Z1]
        
        tau = tn[i + 1] - tn[i]
        
        c = sig**2 / (4.0 * k) * (1 - np.exp(-k * tau))
        delta = 4 * k * theta / sig**2
        kap = 4 * k * np.exp(-k * tau) / ( (sig**2) * (1 - np.exp(-k * tau)) ) * v
        
        v_prev = v
        v = c * np.random.noncentral_chisquare(delta, kap, size=Npath)
        assert all(v >= 0)
        
        x += (r - v_prev/2) * tau + \
            rho / sig * (v - v_prev - k * (theta - v_prev) * tau) + \
            np.sqrt( (1-rho**2) * tau * v_prev ) * Z1
    return v, x