import numpy as np

from derivatives import *
from boundary_conditions import * 
from Problem2d import *

from SLAE_solvers import *

class BaseSolver:
    def __init__(self, problem: Problem, der: DerBase, \
            gridX, gridY, gridT):
        self.problem = problem
        self.der = der
        self.gridX = gridX
        self.gridY = gridY
        self.gridT = gridT
     
    def restore_boundary(self, u, xn, hx, yn, hy):


        xl, yl, zl, xr, yr, zr = self.problem.boundary.getCoefsX(hx[0], hx[1], hx[-2], hx[-1])
        fl = self.problem.boundary._bxleft.getRhs(yn)
        fr = self.problem.boundary._bxright.getRhs(yn)
        u[0, :] = (fl - yl * u[1, :] - zl * u[2, :]) / xl
        u[-1, :] = (fr - yr * u[-2, :] - zr * u[-3, :]) / xr

        fl = self.problem.boundary._byleft.getRhs(xn)
        fr = self.problem.boundary._byright.getRhs(xn)
        xl, yl, zl, xr, yr, zr = self.problem.boundary.getCoefsY(hy[0], hy[1], hy[-2], hy[-1])
        u[:, 0] = (fl - yl * u[:, 1] - zl * u[:, 2]) / xl
        u[:, -1] = (fr - yr * u[:, -2] - zr * u[:, -3]) / xr        
        return u

    def solve(self, Nx, Ny, Nt):
        xn, hx = self.gridX(Nx)
        yn, hy = self.gridY(Ny)
        tn = self.gridT(Nt)

        xmesh, ymesh = np.meshgrid(xn, yn, indexing='ij')

        u = np.zeros((Nt + 1, Nx + 1, Ny + 1))

        u[0, :, :] = self.problem.init(xmesh, ymesh)
        for it in range(Nt):
            tau = tn[it + 1] - tn[it]
            v = u[it, :, :]
            tmp = self.step(v, tau, xmesh, hx, ymesh, hy)
            u[it + 1, :, :] = self.restore_boundary(tmp, xn, hx, yn, hy)
        return u
    

    def step(self):
        assert False, "cannot call step from the base class"


    def preliminary_step(self):
        pass

class Euler(BaseSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def step(self, u, tau, xmesh, hx, ymesh, hy):   
        rhs =  self.problem.getRhs(u, xmesh, hx, ymesh, hy, self.der)
        return u + tau * rhs


class RungeCutta(BaseSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, u, tau, xmesh, hx, ymesh, hy):
        k1 = self.problem.getRhs(u, xmesh, hx, ymesh, hy, self.der)
        k2 = self.problem.getRhs(u + 0.5 * tau * k1, xmesh, hx, ymesh, hy, self.der)
        k3 = self.problem.getRhs(u + 0.5 * tau * k2, xmesh, hx, ymesh, hy, self.der)
        k4 = self.problem.getRhs(u + tau * k3, xmesh, hx, ymesh, hy, self.der)
        return u + tau / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


class ADI_Base(BaseSolver):
    def __init__(self, *args, **kwargs):
        if not 'th' in kwargs:
            kwargs['th'] = 0.5
        self.th = kwargs['th']
        del kwargs['th']
        super().__init__(*args, **kwargs)
    

    def preliminary_step(self, ):
        pass
    

    def slae_x(self, y0, tau, xn, hx, yn, hy, Lx, Ly):
        Ny = yn.size - 1 
        y1 = np.zeros_like(y0)

        xl, yl, zl, xr, yr, zr = self.problem.boundary.getCoefsX(hx[0], hx[1], hx[-2], hx[-1])

        Fx = y0 - self.th * tau * Lx
        Fx[0, :] = self.problem.boundary._bxleft.getRhs(yn)
        Fx[-1, :] = self.problem.boundary._bxright.getRhs(yn)
        for m in range(1, Ny): 
            Ax, Bx, Cx = self.problem.getSplitCoefsX(xn, hx, yn[m], self.der)
            Ax = 1 - self.th * tau * Ax
            Bx = -self.th * tau * Bx
            Cx = -self.th * tau * Cx
            y1[:, m] = Progonka_coefs(Ax, Bx, Cx, \
                        xl, yl, zl, xr, yr, zr, \
                        Fx[:, m])
        return y1
    

    def slae_y(self, y1, tau, xn, hx, yn, hy, Lx, Ly):
        Nx = xn.size - 1
        y2 = np.zeros_like(y1)

        xl, yl, zl, xr, yr, zr = self.problem.boundary.getCoefsY(hy[0], hy[1], hy[-2], hy[-1])    
        
        Fy = y1 - self.th * tau * Ly
        Fy[:, 0] = self.problem.boundary._byleft.getRhs(xn)
        Fy[:, -1] = self.problem.boundary._byright.getRhs(xn)
        
        for n in range(1, Nx):
            Ay, By, Cy = self.problem.getSplitCoefsY(xn[n], yn, hy, self.der)
            Ay = 1 - self.th * tau * Ay
            By = -self.th * tau * By
            Cy = -self.th * tau * Cy

            y2[n, :] = Progonka_coefs(Ay, By, Cy, \
                        xl, yl, zl, xr, yr, zr, \
                        Fy[n, :])
        return y2

class ADI_DO(ADI_Base):
    def __init__(self, *args, **kwargs):
        if not 'th' in kwargs:
            kwargs['th'] = 0.5
        super().__init__(*args, **kwargs)
    
    def step(self, u, tau, xmesh, hx, ymesh, hy):
        xn = xmesh[:, 0]
        yn = ymesh[0, :]

        #first step
        Lx, Ly, Lxy = self.problem.getSplit(u, xmesh, hx, ymesh, hy, self.der)
        
        y0 = u + tau * (  Lx + Ly + Lxy )

        y1 = self.slae_x(y0, tau, xn, hx, yn, hy, Lx, Ly)
                
        y2 = self.slae_y(y1, tau, xn, hx, yn, hy, Lx, Ly)
        return y2


class ADI_CS(ADI_Base):
    def __init__(self, *args, **kwargs):
        if not 'th' in kwargs:
            kwargs['th'] = 0.5
        super().__init__(*args, **kwargs)
    
    def step(self, u, tau, xmesh, hx, ymesh, hy):
        xn = xmesh[:, 0]
        yn = ymesh[0, :]

        #first stage
        Lx, Ly, Lxy = self.problem.getSplit(u, xmesh, hx, ymesh, hy, self.der)
        y0 = u + tau * (  Lx + Ly + Lxy )
        y1 = self.slae_x(y0, tau, xn, hx, yn, hy, Lx, Ly)
        y2 = self.slae_y(y1, tau, xn, hx, yn, hy, Lx, Ly)

        #second stage
        _, _, _Lxy = self.problem.getSplit(y2, xmesh, hx, ymesh, hy, self.der)
        y0 = y0 + 0.5 * tau * ( _Lxy - Lxy )
        y1 = self.slae_x(y0, tau, xn, hx, yn, hy, Lx, Ly)
        y2 = self.slae_y(y1, tau, xn, hx, yn, hy, Lx, Ly)
        return y2


class ADI_MCS(ADI_Base):
    def __init__(self, *args, **kwargs):
        if not 'th' in kwargs:
            kwargs['th'] = 0.5
        super().__init__(*args, **kwargs)
    
    def step(self, u, tau, xmesh, hx, ymesh, hy):
        xn = xmesh[:, 0]
        yn = ymesh[0, :]
        #first stage
        Lx, Ly, Lxy = self.problem.getSplit(u, xmesh, hx, ymesh, hy, self.der)
        y0 = u + tau * (  Lx + Ly + Lxy )
        y1 = self.slae_x(y0, tau, xn, hx, yn, hy, Lx, Ly)
        y2 = self.slae_y(y1, tau, xn, hx, yn, hy, Lx, Ly)
        #second stage
        _Lx, _Ly, _Lxy = self.problem.getSplit(y2, xmesh, hx, ymesh, hy, self.der)
        y0 = y0 + 0.5 * tau * ( _Lxy - Lxy )
        y0 = y0 + (0.5 - self.th) * tau * (_Lx + _Ly + _Lxy - Lx - Ly - Lxy)
        #third stage
        y1 = self.slae_x(y0, tau, xn, hx, yn, hy, Lx, Ly)
        y2 = self.slae_y(y1, tau, xn, hx, yn, hy, Lx, Ly)
        return y2

class ADI_HV(ADI_Base):
    def __init__(self, *args, **kwargs):
        if not 'th' in kwargs:
            kwargs['th'] = 0.5 + np.sqrt(3) / 6.0
        super().__init__(*args, **kwargs)
    
    def step(self, u, tau, xmesh, hx, ymesh, hy):
        xn = xmesh[:, 0]
        yn = ymesh[0, :]
        #first stage
        Lx, Ly, Lxy = self.problem.getSplit(u, xmesh, hx, ymesh, hy, self.der)
        y0 = u + tau * (  Lx + Ly + Lxy )
        y1 = self.slae_x(y0, tau, xn, hx, yn, hy, Lx, Ly)
        y2 = self.slae_y(y1, tau, xn, hx, yn, hy, Lx, Ly)
        #second stage
        y2 = self.restore_boundary(y2, xn, hx, yn, hy)
        _Lx, _Ly, _Lxy = self.problem.getSplit(y2, xmesh, hx, ymesh, hy, self.der)
        y0 = y0 + 0.5 * tau * (_Lx + _Ly + _Lxy - Lx - Ly - Lxy)
        #third stage
        y1 = self.slae_x(y0, tau, xn, hx, yn, hy, _Lx, _Ly)
        y2 = self.slae_y(y1, tau, xn, hx, yn, hy, _Lx, _Ly)
        return y2


