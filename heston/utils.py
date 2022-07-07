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


def GetGrid(hestonParams, optionParams, gridParams):
    X = np.log(optionParams.K * 5)
    xn = np.linspace(-X, X, gridParams.Nx + 1)
    
    Y = max(1.0, hestonParams.theta * 5)
    hy = Y / gridParams.Ny
    yn = np.linspace(-hy / 2.0, Y + hy / 2.0, gridParams.Ny + 1)

    tn = np.linspace(0, optionParams.T, gridParams.M + 1)
    return tn, xn, yn 