from dataclasses import dataclass
import numpy as np
from typing import Callable

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

    gridX : Callable
    gridY : Callable
    gridT : Callable

    Xfact : float = 3.0