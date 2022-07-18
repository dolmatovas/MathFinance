import numpy as np

from numba import jit, njit
from dataclasses import dataclass

@njit
def GetDerivatives(v, hx, hy):
    sl = slice(1, -1, 1)

    h1 = hx[:-1].reshape(-1, 1)
    h2 = hx[1:].reshape(-1, 1)
    d1 = hy[:-1].reshape(1, -1)
    d2 = hy[1:].reshape(1, -1)

    hx = hx.reshape(1, -1)
    hy = hy.reshape(-1, 1)

    
    ux, uy, uxx, uyy, uxy = np.zeros_like(v), np.zeros_like(v), \
                            np.zeros_like(v), np.zeros_like(v), \
                            np.zeros_like(v)
    uxfwd, uxbcwd, uyfwd, uybcwd = np.zeros_like(v), np.zeros_like(v), \
                                   np.zeros_like(v), np.zeros_like(v)
    
    uxfwd[sl, :]  = (v[2:, :] - v[1:-1, :]) / h2
    uxbcwd[sl, :] = (v[1:-1, :] - v[:-2, :]) / h1

    uyfwd[:, sl]  = (v[:, 2:] - v[:, 1:-1]) / d2
    uybcwd[:, sl] = (v[:, 1:-1] - v[:, :-2]) / d1

    ux[sl, :] = ( h1 * uxfwd[sl, :] + h2 * uxbcwd[sl, :] ) / (h1 + h2)
    uy[:, sl] = ( d1 * uyfwd[:, sl] + d2 * uybcwd[:, sl] ) / (d1 + d2)

    uxx[sl, :] = 2.0 * ( uxfwd[sl, :] - uxbcwd[sl, :] ) / (h1 + h2)
    uyy[:, sl] = 2.0 * ( uyfwd[:, sl] - uybcwd[:, sl] ) / (d1 + d2) 

    uxy[sl, sl] = ( (h1 / h2) * (uy[2:, sl] - uy[1:-1, sl]) + (h2 / h1) * (uy[1:-1, sl] - uy[:-2, sl]) ) / (h1 + h2)
    
    
    return ux, uy, uxx, uyy, uxy


@dataclass 
class SLAE_coefs:
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    alpha_l : float
    beta_l : float 
    gamma_l : float

    alpha_r : float
    beta_r : float
    gamma_r : float


class DerBase:
    def __init__(self):
        pass
    
    def DxFwd(self, u, h1, h2):
        ux = np.zeros_like(u)
        sl = slice(1, -1, 1)
        ux[sl, sl] = (u[2:, sl] - u[1:-1, sl]) / h2
        return ux
    
    def DxBcwd(self, u, h1, h2):
        ux = np.zeros_like(u)
        sl = slice(1, -1, 1)
        ux[sl, sl] = (u[1:-1, sl] - u[:-2, sl]) / h1
        return ux

    
    def DxCntrl(self, u, h1, h2):
        ux = np.zeros_like(u)
        sl = slice(1, -1, 1)
        ux[sl, sl] = (self.DxFwd(u, h1, h2)[sl, sl] * h1 + self.DxBcwd(u, h1, h2)[sl, sl] * h2) \
                    / ( h1 + h2 )
        return ux

    def DxFwdCoefs(self, h1, h2):
        _shape = (h1.size + 2, )
        A, B, C = np.zeros(_shape), np.zeros(_shape), np.zeros(_shape)

        A[1:-1] = -1.0 / h2
        B[1:-1] = +1.0 / h2
        return A, B, C

    def DxBcwdCoefs(self, h1, h2):
        _shape = (h1.size + 2, )
        A, B, C = np.zeros(_shape), np.zeros(_shape), np.zeros(_shape)

        A[1:-1] = 1.0 / h1
        C[1:-1] = -1.0 / h1
        return A, B, C

    def DxCntrlCoefs(self, h1, h2):
        _shape = (h1.size + 2, )
        A, B, C = np.zeros(_shape), np.zeros(_shape), np.zeros(_shape)

        A[1:-1] = (h2 - h1) / (h2 * h1)
        B[1:-1] = (h1 / h2) / (h1 + h2)
        C[1:-1] = -(h2 / h1) / (h1 + h2)
        return A, B, C


    def DyFwd(self, u, d1, d2):
        uy = np.zeros_like(u)
        sl = slice(1, -1, 1)
        uy[sl, sl] = (u[sl, 2:] - u[sl, 1:-1]) / d2
        return uy
    
    def DyBcwd(self, u, d1, d2):
        uy = np.zeros_like(u)
        sl = slice(1, -1, 1)
        uy[sl, sl] = (u[sl, 1:-1] - u[sl, :-2]) / d1
        return uy
    
    def DyCntrl(self, u, d1, d2):
        uy = np.zeros_like(u)
        sl = slice(1, -1, 1)
        uy[sl, sl] = (self.DyFwd(u, d1, d2)[sl, sl] * d1 + self.DyBcwd(u, d1, d2)[sl, sl] * d2) \
                    / ( d1 + d2 )
        return uy

    def DyFwdCoefs(self, d1, d2):
        _shape = (d1.size + 2, )
        A, B, C = np.zeros(_shape), np.zeros(_shape), np.zeros(_shape)

        A[1:-1] = -1.0 / d2
        B[1:-1] = +1.0 / d2
        return A, B, C

    def DyBcwdCoefs(self, d1, d2):
        _shape = (d1.size + 2, )
        A, B, C = np.zeros(_shape), np.zeros(_shape), np.zeros(_shape)

        A[1:-1] = +1.0 / d1
        C[1:-1] = -1.0 / d1
        return A, B, C

    def DyCntrlCoefs(self, d1, d2):
        _shape = (d1.size + 2, )
        A, B, C = np.zeros(_shape), np.zeros(_shape), np.zeros(_shape)

        A[1:-1] = (d2 - d1) / (d2 * d1)
        B[1:-1] = (d1 / d2) / (d1 + d2)
        C[1:-1] = -(d2 / d1) / (d1 + d2)
        return A, B, C


    def D2x(self, u, h1, h2):
        uxx = np.zeros_like(u)
        sl = slice(1, -1, 1)
        uxx[sl, sl] = 2.0 * (self.DxFwd(u, h1, h2)[sl, sl] - self.DxBcwd(u, h1, h2)[sl, sl]) \
                        / (h1 + h2)
        return uxx

    def D2xCoefs(self, h1, h2):
        _shape = (h1.size + 2, )
        A, B, C = np.zeros(_shape), np.zeros(_shape), np.zeros(_shape)

        A[1:-1] = -2.0 / (h2 * h1)
        B[1:-1] = 2.0 / h2 / (h1 + h2)
        C[1:-1] = 2.0 / h1 / (h1 + h2)
        return A, B, C 
    
    def D2y(self, u, d1, d2):
        uyy = np.zeros_like(u)
        sl = slice(1, -1, 1)
        uyy[sl, sl] = 2.0 * (self.DyFwd(u, d1, d2)[sl, sl] - self.DyBcwd(u, d1, d2)[sl, sl]) / \
                        (d1 + d2)
        return uyy
    
    def D2yCoefs(self, d1, d2):
        _shape = (d1.size + 2, )
        A, B, C = np.zeros(_shape), np.zeros(_shape), np.zeros(_shape)

        A[1:-1] = -2.0 / (d2 * d1)
        B[1:-1] = 2.0 / d2 / (d1 + d2)
        C[1:-1] = 2.0 / d1 / (d1 + d2)
        return A, B, C 
    
    def Dxy(self, u, h1, h2, d1, d2):
        uxy = np.zeros_like(u)
        sl = slice(1, -1, 1)
        uxfwd = np.zeros_like(u)
        uxbwd = np.zeros_like(u)
        uxctr = np.zeros_like(u)

        hx = np.append(h1, h2[-1]).reshape(-1, 1)

        uxfwd[:-1, :] = (u[1:] - u[:-1]) / hx
        uxbwd[1:, :]  = (u[1:] - u[:-1]) / hx

        uxctr[sl, :] = ( h1 * uxfwd[sl, :] + h2 * uxbwd[sl, :] ) / (h1 + h2)
        return self.DyCntrl(uxctr, d1, d2)


class DerCntrl(DerBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.Dx = self.DxCntrl
        self.Dy = self.DyCntrl

        self.DxCoefs = self.DxCntrlCoefs
        self.DyCoefs = self.DyCntrlCoefs

    
class DerFwdX(DerBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.Dx = self.DxFwd
        self.Dy = self.DyCntrl

        self.DxCoefs = self.DxFwdCoefs
        self.DyCoefs = self.DyCntrlCoefs

class DerFwdY(DerBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.Dx = self.DxCntrl
        self.Dy = self.DyFwd

        self.DxCoefs = self.DxCntrlCoefs
        self.DyCoefs = self.DyFwdCoefs

    
class DerFwdXY(DerBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.Dx = self.DxFwd
        self.Dy = self.DyFwd

        self.DxCoefs = self.DxFwdCoefs
        self.DyCoefs = self.DyFwdCoefs
