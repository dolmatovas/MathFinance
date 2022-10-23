from pathlib import Path
from sys import path
path.append(str(Path(__file__).parent.resolve()) + '/../')


from heston_option_price import heston_option_price
from black_scholes import implied_volatility


from typing import Optional, Union, Tuple, List
import numpy as np


def gen_random_heston_params() -> np.ndarray:
    """
        This function gerenate random parameters for heston model

        Returns:
            heston_params(np.ndarray): generated sabr params
    """
    v0 = np.random.rand(1) * 0.015 + 0.01
    theta = np.random.rand(1) * 0.015 + 0.01
    
    rho = -0.9 + (1.8) * np.random.rand(1)
    k = np.random.rand(1) * 2 + 1.0
    sig = np.random.rand(1) * 0.002 + 0.01
    
    return np.asarray(v0[0], theta[0], rho[0], k[0], sig[0])


class Heston:
    """ Class for Heston model
    
    Attributes:
        r(float): interest rate
        heston_params(np.ndarray): calibrated parameters of the Heston model, heston_params = [v_0, theta, rho, k, sigma]
    """
        
    def __init__(self, heston_params:np.ndarray, interest_rate:float = 0, num_of_integration_points:int = 200):
        """
            The __init__ method just saves given parameters.
            
            Args:
                heston_params(np.ndarray): heston parameters
                interest_rate(float): risk free interest rate, default value is zero
        """
        self.r = interest_rate
        self.heston_params = heston_params
        self.num_of_integration_points = num_of_integration_points
        
        
    def __call__(self,K: np.ndarray, 
                      F: Union[float, np.ndarray], 
                      T: Union[float, np.ndarray], is_call:bool = True) -> Tuple[ np.ndarray, np.ndarray ]:
        """
            Returns option prices and implied volatility for given parameters K, F, T 
            
            Args:
                K(np.ndarray): array of strikes
                F(float | np.ndarray): underlying futures price
                T(float, np.ndarray): expiration time
            Returns:
                C(np.ndarray): option prices
                iv(np.ndarray) : implied volatility
        """
        S = F * np.exp(-self.r * T)
        C = heston_option_price(S, K, T, self.num_of_integration_points, self.r, self.heston_params)
        iv = implied_volatility(C, K, F, T, self.r) 
        P = C + np.exp(-self.r * T) * (K - F)
        X = C if is_call else P
        return X, iv

