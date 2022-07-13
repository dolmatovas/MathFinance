import numpy as np

from numba import jit, njit

@njit
def GetDerivatives(v, hx, hy):
    sl = slice(1, -1, 1)

    h1 = hx[:-1].reshape(-1, 1)
    h2 = hx[1:].reshape(-1, 1)
    d1 = hy[:-1].reshape(1, -1)
    d2 = hy[1:].reshape(1, -1)

    
    ux, uy, uxx, uyy, uxy = np.zeros_like(v), np.zeros_like(v), \
                            np.zeros_like(v), np.zeros_like(v), \
                            np.zeros_like(v)
    uxfwd, uxbcwd, uyfwd, uybcwd = np.zeros_like(v), np.zeros_like(v), \
                                   np.zeros_like(v), np.zeros_like(v)
    
    uxfwd[sl, sl]  = (v[2:, sl] - v[1:-1, sl]) / h2
    uxbcwd[sl, sl] = (v[1:-1, sl] - v[:-2, sl]) / h1

    uyfwd[sl, sl]  = (v[sl, 2:] - v[sl, 1:-1]) / d2
    uybcwd[sl, sl] = (v[sl, 1:-1] - v[sl, :-2]) / d1

    ux[sl, sl] = ( h1 * uxfwd[sl, sl] + h2 * uxbcwd[sl, sl] ) / (h1 + h2)
    uy[sl, sl] = ( d1 * uyfwd[sl, sl] + d2 * uybcwd[sl, sl] ) / (d1 + d2)

    uxx[sl, sl] = ( uxfwd[sl, sl] - uxbcwd[sl, sl] ) / (h1 + h2)
    uyy[sl, sl] = ( uyfwd[sl, sl] - uybcwd[sl, sl] ) / (d1 + d2) 

    uxy[sl, sl] = ( (h1 / h2) * (uy[2:, sl] - uy[1:-1, sl]) + (h2 / h1) * (uy[1:-1, sl] - uy[:-2, sl]) ) / (h1 + h2)
    return ux, uy, uxx, uyy, uxy


class DerBase:
    def __init__(self, xn=None, yn=None, hx=None, hy=None):
        assert not xn is None or not hx is None, "error!"
        assert not yn is None or not hy is None, "error!"
        self.hx = hx
        self.hy = hy
        if hx is None:
            self.hx = xn[1:] - xn[:-1]
        if hy is None:
            self.hy = yn[1:] - yn[:-1]
        self.h1 = self.hx[:-1].reshape(-1, 1)
        self.h2 = self.hx[1:].reshape(-1, 1)

        self.d1 = self.hy[:-1].reshape(1, -1)
        self.d2 = self.hy[1:].reshape(1, -1)

        self.shape = ( self.hx.size + 1, self.hy.size + 1 )
    
    def DxFwd(self, u):
        ux = np.zeros(self.shape)
        sl = slice(1, -1, 1)
        ux[sl, sl] = (u[2:, sl] - u[1:-1, sl]) / self.h2
        return ux
    
    def DxBcwd(self, u):
        ux = np.zeros(self.shape)
        sl = slice(1, -1, 1)
        ux[sl, sl] = (u[1:-1, sl] - u[:-2, sl]) / self.h1
        return ux
    
    def DxCntrl(self, u):
        return (DxFwd(u) * self.h1 + DxBcwd(u) * self.h2) / ( self.h1 + self.h2 )

    def DyFwd(self, u):
        uy = np.zeros(self.shape)
        sl = slice(1, -1, 1)
        uy[sl, sl] = (u[sl, 2:] - u[sl, 1:-1]) / self.d2
        return uy
    
    def DyBcwd(self, u):
        uy = np.zeros(self.shape)
        sl = slice(1, -1, 1)
        uy[sl, sl] = (u[sl, 1:-1] - u[sl, :-2]) / self.d1
        return uy
    
    def DyCntrl(self, u):
        return (DyFwd(u) * self.d1 + DyBcwd(u) * self.d2) / ( self.d1 + self.d2 )

    def D2x(self, u):
        return (DxFwd(u) - DxBcwd(u)) / (self.h1 + self.h2)
    
    def D2y(self, u):
        return (DyFwd(u) - DyBcwd(u)) / (self.d1 + self.d2)
    
    def Dxy(self, u):
        return DyCntrl( DxCntrl(u) )
