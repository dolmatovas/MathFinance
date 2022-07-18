from dataclasses import dataclass
from boundary_conditions import *
from derivatives import *

class Problem:
    def __init__(self, boundary: Boundary2D, init, mux, muy, sigmax, sigmay, sigmaxy):
        self.boundary = boundary
        self.mux = mux
        self.muy = muy
        self.sigmax = sigmax
        self.sigmay = sigmay
        self.sigmaxy = sigmaxy

        self.init = init

    def getSplit(self, u, xmesh, hx, ymesh, hy, der:DerBase):
        h1 = hx[:-1].reshape(-1, 1)
        h2 = hx[1:].reshape(-1, 1)
        d1 = hy[:-1].reshape(1, -1)
        d2 = hy[1:].reshape(1, -1)

        ux = der.Dx(u, h1, h2)
        uy = der.Dy(u, d1, d2)
        uxx = der.D2x(u, h1, h2)
        uyy = der.D2y(u, d1, d2)
        uxy = der.Dxy(u, h1, h2, d1, d2)
        
        Lx = np.zeros_like(u)
        Ly = np.zeros_like(u)
        Lxy = np.zeros_like(u)

        #force coefs to have shape like xmesh
        mux = self.mux(xmesh, ymesh) + np.zeros_like(xmesh)
        muy = self.muy(xmesh, ymesh) + np.zeros_like(xmesh)
        sigmax = self.sigmax(xmesh, ymesh) + np.zeros_like(xmesh)
        sigmay = self.sigmax(xmesh, ymesh) + np.zeros_like(xmesh)
        sigmaxy = self.sigmaxy(xmesh, ymesh) + np.zeros_like(xmesh)

        Lx = ux * mux + sigmax * uxx
        Ly = uy * muy + sigmay * uyy
        Lxy = sigmaxy * uxy

        return Lx, Ly, Lxy

    def getRhs(self, u, xmesh, hx, ymesh, hy, der:DerBase):
        Lx, Ly, Lxy = self.getSplit(u, xmesh, hx, ymesh, hy, der)
        return Lx + Ly + Lxy


    def getSplitCoefsX(self, xn:np.ndarray, hx:np.ndarray, y:float, der:DerBase):
        h1 = hx[:-1]#.reshape(-1, 1)
        h2 = hx[1:]#.reshape(-1, 1)

        Ax, Bx, Cx = der.DxCoefs(h1, h2)
        Axx, Bxx, Cxx = der.D2xCoefs(h1, h2)
        mux = self.mux(xn, y)
        sigmax = self.sigmax(xn, y)

        Ax = mux * Ax + sigmax * Axx
        Bx = mux * Bx + sigmax * Bxx
        Cx = mux * Cx + sigmax * Cxx
        return Ax, Bx, Cx


    def getSplitCoefsY(self, x:float, yn:np.ndarray, hy:np.ndarray, der:DerBase):
        d1 = hy[:-1]#.reshape(1, -1)
        d2 = hy[1:]#.reshape(1, -1)

        Ay, By, Cy = der.DyCoefs(d1, d2)
        Ayy, Byy, Cyy = der.D2yCoefs(d1, d2)
        muy = self.muy(x, yn)
        sigmay = self.sigmay(x, yn)

        Ay = muy * Ay + sigmay * Ayy
        By = muy * By + sigmay * Byy
        Cy = muy * Cy + sigmay * Cyy
        return Ay, By, Cy