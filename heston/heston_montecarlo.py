import numpy as np
import scipy.stats as sps


def generate_variance_euler(n_paths:int, tn:np.ndarray, v0:float, theta:float, rho:float, k:float, sig:float):
    Nt = len(tn)

    v = np.zeros((Nt, n_paths)) + v0

    for i in range(Nt - 1):
        tau = tn[i + 1] - tn[i]
        dW = np.random.randn(n_paths // 2)
        dW = np.r_[dW, -dW]
        
        v[i + 1, :] = v[i, :] +  k * (theta - v[i, :] ) * tau + \
                        sig * np.sqrt(v[i, :]  * tau) * dW
        v[i + 1, :] = np.maximum(v[i + 1, :], 0.0)
    return v


def generate_variance_euler_modified(n_paths:int, tn:np.ndarray, v0:float, theta:float, rho:float, k:float, sig:float):
    Nt = len(tn)

    v = np.zeros((Nt, n_paths)) + v0

    for i in range(Nt - 1):
        tau = tn[i + 1] - tn[i]
        dW = np.random.randn(n_paths // 2)
        dW = np.r_[dW, -dW]
        
        v[i + 1, :] = v[i, :] +  k * (theta - v[i, :] ) * tau + \
                        sig * np.sqrt(v[i, :]  * tau) * dW + \
                        0.25 * tau * sig * (dW ** 2 - 1)
        v[i + 1, :] = np.maximum(v[i + 1, :], 0.0)
    return v


def generate_variance_chi(n_paths:int, tn:np.ndarray, v0:float, theta:float, rho:float, k:float, sig:float):
    Nt = len(tn)

    v = np.zeros((Nt, n_paths)) + v0

    for i in range(Nt - 1):
        tau = tn[i + 1] - tn[i]
        
        c = sig ** 2 / (4.0 * k) * (1 - np.exp(-k * tau))
        delta = 4 * k * theta / sig ** 2
        kap = 4 * k * np.exp(-k * tau) / ( (sig ** 2) * (1 - np.exp(-k * tau)) ) * v[i, :]
    
        v[i + 1, :] = c * np.random.noncentral_chisquare(delta, kap, size=n_paths)
    return v


def generate_path_euler(n_paths:int, S0:float, tn:np.ndarray, r:float, heston_params:np.ndarray):
    '''
        This function simulate the stock price and variance path under heston model and
        return terminal values S_T, v_T, given that S_t = S_0, v_t = v_0 where t = tn[0], T = tn[-1]

        Args:
            n_paths(int): number of simulated pathe
            S0: initial stock price
            tn: 
            r: interest rate
            heston_params(np.ndarray): parameters of the heston model
    '''
    assert len(heston_params == 5), "TODO: add milti dimentional heston simulation"
    v0, theta, rho, k, sig = heston_params
    
    Nt = len(tn)
    
    v = v0
    x = 0.0
    
    for i in range(Nt - 1):
        
        Z1 = np.random.randn(n_paths // 2)
        Z2 = np.random.randn(n_paths // 2)

        Z1 = np.r_[Z1, -Z1]
        Z2 = np.r_[Z2, -Z2]

        dW1 = Z1
        dW2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2
        
        tau = tn[i + 1] - tn[i]
        

        
        dv = k * (theta - v) * tau + sig * np.sqrt(v * tau) * dW1
        x += (r - v * 0.5) * tau + np.sqrt(v * tau) * dW2
        
        v = np.maximum(v + dv, 0)
    return v, S0 * np.exp(x)


def generate_path_euler_modified(n_paths:int, S0:float, tn:np.ndarray, r:float, heston_params:np.ndarray):
    '''
        This function simulate the stock price and variance path under heston model and
        return terminal values S_T, v_T, given that S_t = S_0, v_t = v_0 where t = tn[0], T = tn[-1]

        Args:
            n_paths(int): number of simulated pathe
            S0: initial stock price
            tn: 
            r: interest rate
            heston_params(np.ndarray): parameters of the heston model
    '''
    assert len(heston_params == 5), "TODO: add milti dimentional heston simulation"
    v0, theta, rho, k, sig = heston_params
    
    Nt = len(tn)
    
    v = v0
    x = 0.0
    
    for i in range(Nt - 1):
        
        Z1 = np.random.randn(n_paths // 2)
        Z2 = np.random.randn(n_paths // 2)

        Z1 = np.r_[Z1, -Z1]
        Z2 = np.r_[Z2, -Z2]

        dW1 = Z1
        dW2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2
        
        tau = tn[i + 1] - tn[i]
        

        
        dv = k * (theta - v) * tau + sig * np.sqrt(v * tau) * dW1 + 0.25 * tau * sig * (dW1 ** 2 - 1)
        x += (r - v * 0.5) * tau + np.sqrt(v * tau) * dW2
        
        v = np.maximum(v + dv, 0)
    return v, S0 * np.exp(x)


def generate_path_chi(n_paths:int, S0:float, tn:np.ndarray, r:float, heston_params:np.ndarray):
    '''
        This function simulate the stock price and variance path under heston model and
        return terminal values S_T, v_T, given that S_t = S_0, v_t = v_0 where t = tn[0], T = tn[-1]

        Args:
            n_paths(int): number of simulated pathe
            S0: initial stock price
            tn: 
            r: interest rate
            heston_params(np.ndarray): parameters of the heston model
    '''
    assert len(heston_params == 5), "TODO: add milti dimentional heston simulation"
    v0, theta, rho, k, sig = heston_params    
    
    Nt = len(tn)
    v = v0
    x = 0.0
    
    for i in range(Nt - 1):
        
        Z1 = np.random.randn(n_paths // 2)
        Z1 = np.r_[Z1, -Z1]
        
        tau = tn[i + 1] - tn[i]
        
        c = sig ** 2 / (4.0 * k) * (1 - np.exp(-k * tau))
        delta = 4 * k * theta / sig ** 2
        kap = 4 * k * np.exp(-k * tau) / ( (sig**2) * (1 - np.exp(-k * tau)) ) * v
        
        v_prev = v
        v = c * np.random.noncentral_chisquare(delta, kap, size=n_paths)
        
        x += (r - v_prev / 2) * tau + \
            rho / sig * (v - v_prev - k * (theta - v_prev) * tau) + \
            np.sqrt( (1-rho**2) * tau * v_prev ) * Z1
    return v, S0 * np.exp(x)