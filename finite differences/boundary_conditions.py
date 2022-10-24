from typing import Tuple, Union, Callable
import numpy as np

class Boundary:
    '''
        This class stores boundary conditions of the third type:
        alpha u' + beta u = rhs
    '''
    def __init__(self, alpha:float, beta:float, rhs:Callable[ [ np.ndarray, float ] , np.ndarray ]):
        '''
            Init method just saves coefficients and rhs

            Args:
                alpha(float) : alpha
                beta(float): beta
                rhs(Callable[ [ np.ndarray, float ] , np.ndarray ]) : right hand side. 
                The first argument is spatial variable, the second argument is time.
        '''
        self._alpha = alpha
        self._beta = beta
        self._rhs = rhs
    
    def get_rhs(self, x:np.ndarray, t:float):
        '''
            get_rhs method evaluates right hand side

            Args:
                x(np.ndarray): spatial variabple
                t(float): time

        '''
        return self._rhs(x, t)

class Dirichle(Boundary):
    def __init__(self, rhs):
        super().__init__(0.0, 1.0, rhs)


class Neuman(Boundary):
    def __init__(self, rhs):
        super().__init__(1.0, 0.0, rhs)


def get_coefs_left(boundary: Boundary, h1 : float, h2 : float) -> Tuple[float,float,float]:
    '''
        alpha u'_0 + beta u_0 = x u_0 + y u_1 + z u_2

        Args:
            boundary(Boundary): boundary
            h1(float) -- grid step
            h2(float) -- grid step
        Returns:
            x(float):
            y(float):
            z(float):
    ''' 
    
    alpha = boundary._alpha
    beta = boundary._beta
    x = alpha * (-2.0 * h1 - h2 ) / (h1 * (h1 + h2)) + beta
    y = alpha * (h1 + h2) / (h1 * h2)
    z = alpha * (-h1 / ((h1 + h2) * h2) )
    return x, y, z


def get_coefs_right(border, hm1, hm2):
    '''
        alpha u'_N + beta u_N = x u_N + y u_{N-1} + z u_{N-2}

        Args:
            boundary(Boundary): boundary
            hm1(float) -- grid step
            hm2(float) -- grid step
        Returns:
            x(float):
            y(float):
            z(float):
    ''' 
    alpha = border._alpha
    beta = border._beta
    x = alpha * (2.0 * hm1 + hm2 ) / (hm1 * (hm1 + hm2)) + beta
    y = alpha * (-hm1 - hm2) / (hm1 * hm2)
    z = alpha * hm1 / ((hm1 + hm2) * hm2)
    return x, y, z


class Boundary2D:
    def __init__(self, bxleft : Boundary, bxright : Boundary, byleft:  Boundary, byright : Boundary):
        self._bxleft = bxleft
        self._bxright = bxright
        self._byleft = byleft
        self._byright = byright
    
    def get_coefs_x(self, h1, h2, hm2, hm1):
        '''
            alpha_l u'_0 + beta_l u_0 = xl u_0 + yl u_1 + zl u_2
            alpha_r u'_N + beta_r u_N = xr u_{N} + yr u_{N - 1} + zr u_{N - 2} 
        '''
        return get_coefs_left(self._bxleft, h1, h2) + get_coefs_right(self._bxright, hm1, hm2)
    
    def get_coefs_y(self, d1, d2, dm2, dm1):
        return get_coefs_left(self._byleft, d1, d2) + get_coefs_right(self._byright, dm1, dm2)
        