from dataclasses import dataclass
from boundary_conditions import Boundary2D
from derivatives import DerBase

import numpy as np

class Problem:
    def __init__(self, boundary: Boundary2D, init, mux, muy, sigmax, sigmay, sigmaxy):
        self.boundary = boundary
        self.mux = mux
        self.muy = muy
        self.sigmax = sigmax
        self.sigmay = sigmay
        self.sigmaxy = sigmaxy

        self.init = init

    def get_split(self, u, xmesh, hx, ymesh, hy, der:DerBase):
        h1 = hx[:-1].reshape(-1, 1)
        h2 = hx[1:].reshape(-1, 1)
        d1 = hy[:-1].reshape(1, -1)
        d2 = hy[1:].reshape(1, -1)

        ux = der.dx(u, h1, h2)
        uy = der.dy(u, d1, d2)
        uxx = der.d2x(u, h1, h2)
        uyy = der.d2y(u, d1, d2)
        uxy = der.dxy(u, h1, h2, d1, d2)
        
        Lx = np.zeros_like(u)
        Ly = np.zeros_like(u)
        Lxy = np.zeros_like(u)

        #force coefs to have shape like xmesh
        #mux = self.mux(xmesh, ymesh) + np.zeros_like(xmesh)
        #muy = self.muy(xmesh, ymesh) + np.zeros_like(xmesh)
        #sigmax = self.sigmax(xmesh, ymesh) + np.zeros_like(xmesh)
        #sigmay = self.sigmax(xmesh, ymesh) + np.zeros_like(xmesh)
        #sigmaxy = self.sigmaxy(xmesh, ymesh) + np.zeros_like(xmesh)

        Lx = ux * self.mux_ + self.sigmax_ * uxx
        Ly = uy * self.muy_ + self.sigmay_ * uyy
        Lxy = self.sigmaxy_ * uxy

        return Lx, Ly, Lxy

    def get_rhs(self, u, xmesh, hx, ymesh, hy, der:DerBase):
        Lx, Ly, Lxy = self.get_split(u, xmesh, hx, ymesh, hy, der)
        return Lx + Ly + Lxy


    def get_split_coefs_x(self, xmesh, hx, ymesh, hy, der:DerBase):
        h1 = hx[:-1]#.reshape(-1, 1)
        h2 = hx[1:]#.reshape(-1, 1)

        Ax, Bx, Cx = der.dx_coefs(h1, h2)
        Axx, Bxx, Cxx = der.d2x_coefs(h1, h2)
        mux = self.mux(xmesh, ymesh) + np.zeros_like(xmesh)
        sigmax = self.sigmax(xmesh, ymesh) + np.zeros_like(xmesh)

        Ax = mux * Ax.reshape(-1, 1) + sigmax * Axx.reshape(-1, 1)
        Bx = mux * Bx.reshape(-1, 1) + sigmax * Bxx.reshape(-1, 1)
        Cx = mux * Cx.reshape(-1, 1) + sigmax * Cxx.reshape(-1, 1)
        return Ax, Bx, Cx


    def get_split_coefs_y(self, xmesh, hx, ymesh, hy, der:DerBase):
        d1 = hy[:-1]#.reshape(1, -1)
        d2 = hy[1:]#.reshape(1, -1)

        Ay, By, Cy = der.dy_coefs(d1, d2)
        Ayy, Byy, Cyy = der.d2y_coefs(d1, d2)
        muy = self.muy(xmesh, ymesh) + np.zeros_like(xmesh)
        sigmay = self.sigmay(xmesh, ymesh) + np.zeros_like(xmesh)

        Ay = muy * Ay.reshape(1, -1) + sigmay * Ayy.reshape(1, -1)
        By = muy * By.reshape(1, -1) + sigmay * Byy.reshape(1, -1)
        Cy = muy * Cy.reshape(1, -1) + sigmay * Cyy.reshape(1, -1)
        return Ay, By, Cy