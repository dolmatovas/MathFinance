from dataclasses import dataclass
import numpy as np

@dataclass
class HestonParams:
    r : float
    sigma : float
    kappa : float
    theta : float
    rho : float
        
        
@dataclass
class OptionParams:
    K : float
    T : float

        
@dataclass
class GridParams:
    Nx : int
    Ny : int
    M : int
    
    S0 : float
    v0 : float

    Xfact : float = 3.0


def GetGrid(hestonParams, optionParams, gridParams):
    K = optionParams.K
    F = gridParams.Xfact 
    eps = 0.02 * K
    C = K - eps
    n = 2
    a = ((F - 1) * K / C) ** (1/n)

    tn = np.linspace(-1, a, gridParams.Nx + 1)
    Sn = K + C *  (np.abs(tn) ** n) * np.sign(tn)

    #Sn = np.linspace(eps, F, gridParams.Nx + 1)
    xn = np.log(Sn / optionParams.K) 
    #xn = np.linspace(-X, X, gridParams.Nx + 1)
    
    Y = max(2.5, hestonParams.theta / hestonParams.sigma * 5)
    hy = Y / (gridParams.Ny - 1)
    yn = np.linspace(-hy / 2.0, Y + hy / 2.0, gridParams.Ny + 1)

    yn = Y * (np.linspace(0, 1, gridParams.Ny + 1) ** 3 )

    tn = np.linspace(0, optionParams.T, gridParams.M + 1)

    tn = optionParams.T * (np.linspace(0, 1.0, gridParams.M + 1) ** 1 ) 
    return tn, xn, yn 