import numpy as np 
from problem_2d import Problem
from derivatives import DerBase
from slae_solvers import progonka_coefs

class BaseSolver:
    def __init__(self, problem: Problem, der: DerBase, \
            gridX, gridY, gridT):
        self.problem = problem
        self.der = der
        self.gridX = gridX
        self.gridY = gridY
        self.gridT = gridT
     
    def restore_boundary(self, u, xn, hx, yn, hy, t):


        xl, yl, zl, xr, yr, zr = self.problem.boundary.get_coefs_x(hx[0], hx[1], hx[-2], hx[-1])
        fl = self.problem.boundary._bxleft.get_rhs(yn, t)
        fr = self.problem.boundary._bxright.get_rhs(yn, t)
        u[0, :] = (fl - yl * u[1, :] - zl * u[2, :]) / xl
        u[-1, :] = (fr - yr * u[-2, :] - zr * u[-3, :]) / xr

        fl = self.problem.boundary._byleft.get_rhs(xn, t)
        fr = self.problem.boundary._byright.get_rhs(xn, t)
        xl, yl, zl, xr, yr, zr = self.problem.boundary.get_coefs_y(hy[0], hy[1], hy[-2], hy[-1])
        u[:, 0] = (fl - yl * u[:, 1] - zl * u[:, 2]) / xl
        u[:, -1] = (fr - yr * u[:, -2] - zr * u[:, -3]) / xr        
        return u

    def solve(self, Nx, Ny, Nt):
        xn, hx = self.gridX(Nx)
        yn, hy = self.gridY(Ny)
        tn = self.gridT(Nt)

        xmesh, ymesh = np.meshgrid(xn, yn, indexing='ij')

        u = np.zeros((Nt + 1, Nx + 1, Ny + 1))

        self.preliminary_step(xmesh, hx, ymesh, hy)

        zrs = np.zeros_like(xmesh)
        self.problem.mux_ = self.problem.mux(xmesh, ymesh) + zrs
        self.problem.muy_ = self.problem.muy(xmesh, ymesh) + zrs
        self.problem.sigmax_ = self.problem.sigmax(xmesh, ymesh) + zrs
        self.problem.sigmay_ = self.problem.sigmax(xmesh, ymesh) + zrs
        self.problem.sigmaxy_ = self.problem.sigmaxy(xmesh, ymesh) + zrs
        
        u[0, :, :] = self.problem.init(xmesh, ymesh)
        for it in range(Nt):
            tau = tn[it + 1] - tn[it]
            v = u[it, :, :]
            tmp = self.step(v, tn[it], tau, xmesh, hx, ymesh, hy)
            u[it + 1, :, :] = self.restore_boundary(tmp, xn, hx, yn, hy, tn[it])
        return u
    

    def step(self):
        assert False, "cannot call step from the base class"


    def preliminary_step(self, *args, **kwargs):
        pass


class Euler(BaseSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def step(self, u, t, tau, xmesh, hx, ymesh, hy):   
        rhs =  self.problem.get_rhs(u, xmesh, hx, ymesh, hy, self.der)
        return u + tau * rhs


class RungeCutta(BaseSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, u, t, tau, xmesh, hx, ymesh, hy):
        k1 = self.problem.get_rhs(u, xmesh, hx, ymesh, hy, self.der)
        k2 = self.problem.get_rhs(u + 0.5 * tau * k1, xmesh, hx, ymesh, hy, self.der)
        k3 = self.problem.get_rhs(u + 0.5 * tau * k2, xmesh, hx, ymesh, hy, self.der)
        k4 = self.problem.get_rhs(u + tau * k3, xmesh, hx, ymesh, hy, self.der)
        return u + tau / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


class ADI_Base(BaseSolver):
    def __init__(self, *args, **kwargs):
        if not 'th' in kwargs:
            kwargs['th'] = 0.5
        self.th = kwargs['th']
        del kwargs['th']
        super().__init__(*args, **kwargs)
    

    def preliminary_step(self, xmesh, hx, ymesh, hy):
        self.Ax, self.Bx, self.Cx = \
            self.problem.get_split_coefs_x(xmesh, hx, \
                                        ymesh, hy, \
                                        self.der)
        self.Ay, self.By, self.Cy = \
            self.problem.get_split_coefs_x(xmesh, hx, \
                                        ymesh, hy, \
                                        self.der)
    

    def slae_x(self, y0, t, tau, xn, hx, yn, hy, Lx, Ly):
        Ny = yn.size - 1 
        y1 = np.zeros_like(y0)

        xl, yl, zl, xr, yr, zr = self.problem.boundary.get_coefs_x(hx[0], hx[1], hx[-2], hx[-1])

        Fx = y0 - self.th * tau * Lx
        Fx[0, :] = self.problem.boundary._bxleft.get_rhs(yn, t)
        Fx[-1, :] = self.problem.boundary._bxright.get_rhs(yn, t)
        
        Ax = 1 - self.th * tau * self.Ax[:, 1:-1]
        Bx = -self.th * tau * self.Bx[:, 1:-1]
        Cx = -self.th * tau * self.Cx[:, 1:-1]
        y1[:, 1:-1]  = progonka_coefs(Ax, Bx, Cx, \
                        xl, yl, zl, xr, yr, zr, \
                        Fx[:, 1:-1])
        
        #for m in range(1, Ny): 
        #    Ax = 1 - self.th * tau * self.Ax[:, m]
        #    Bx = -self.th * tau * self.Bx[:, m]
        #    Cx = -self.th * tau * self.Cx[:, m]
        #    y1[:, m] = Progonka_coefs(Ax, Bx, Cx, \
        #                xl, yl, zl, xr, yr, zr, \
        #                Fx[:, m])
        return y1
    

    def slae_y(self, y1, t, tau, xn, hx, yn, hy, Lx, Ly):
        Nx = xn.size - 1
        y2 = np.zeros_like(y1)

        xl, yl, zl, xr, yr, zr = self.problem.boundary.get_coefs_y(hy[0], hy[1], hy[-2], hy[-1])    
        
        Fy = y1 - self.th * tau * Ly
        Fy[:, 0] = self.problem.boundary._byleft.get_rhs(xn, t)
        Fy[:, -1] = self.problem.boundary._byright.get_rhs(xn, t)
        
        Ay = 1.0 - self.th * tau * self.Ay[1:-1, :]
        By = -self.th * tau * self.By[1:-1, :]
        Cy = -self.th * tau * self.Cy[1:-1, :]

        y2[1:-1, :] = progonka_coefs(Ay.T, By.T, Cy.T, \
                        xl, yl, zl, xr, yr, zr, \
                        Fy[1:-1, :].T).T


        #for n in range(1, Nx):
        #    Ay = 1.0 - self.th * tau * self.Ay[n, :]
        #    By = -self.th * tau * self.By[n, :]
        #    Cy = -self.th * tau * self.Cy[n, :]
        #
        #    y2[n, :] = Progonka_coefs(Ay, By, Cy, \
        #                xl, yl, zl, xr, yr, zr, \
        #                Fy[n, :])
        return y2

class ADI_DO(ADI_Base):
    def __init__(self, *args, **kwargs):
        if not 'th' in kwargs:
            kwargs['th'] = 0.5
        super().__init__(*args, **kwargs)
    
    def step(self, u, t, tau, xmesh, hx, ymesh, hy):
        xn = xmesh[:, 0]
        yn = ymesh[0, :]

        #first step
        Lx, Ly, Lxy = self.problem.get_split(u, xmesh, hx, ymesh, hy, self.der)
        
        y0 = u + tau * (  Lx + Ly + Lxy )

        y0 = self.restore_boundary(y0, xn, hx, yn, hy, t + tau / 2)

        y1 = self.slae_x(y0, t + tau / 2, tau, xn, hx, yn, hy, Lx, Ly)
                
        u = self.slae_y(y1, t, tau, xn, hx, yn, hy, Lx, Ly)
        return u


class ADI_CS(ADI_Base):
    def __init__(self, *args, **kwargs):
        if not 'th' in kwargs:
            kwargs['th'] = 0.5
        super().__init__(*args, **kwargs)
    
    def step(self, u, t, tau, xmesh, hx, ymesh, hy):
        xn = xmesh[:, 0]
        yn = ymesh[0, :]

        #first stage
        Lx, Ly, Lxy = self.problem.get_split(u, xmesh, hx, ymesh, hy, self.der)
        y0 = u + tau * (  Lx + Ly + Lxy )
        y0 = self.restore_boundary(y0, xn, hx, yn, hy, t + tau / 2)
        y1 = self.slae_x(y0, t + tau/2, tau, xn, hx, yn, hy, Lx, Ly)
        y2 = self.slae_y(y1, t + tau/2, tau, xn, hx, yn, hy, Lx, Ly)

        #second stage
        _, _, _Lxy = self.problem.get_split(y2, xmesh, hx, ymesh, hy, self.der)
        y0 = y0 + 0.5 * tau * ( _Lxy - Lxy )
        y0 = self.restore_boundary(y0, xn, hx, yn, hy, t + tau / 2)
        y1 = self.slae_x(y0, t + tau/2, tau, xn, hx, yn, hy, Lx, Ly)
        u = self.slae_y(y1, t + tau, tau, xn, hx, yn, hy, Lx, Ly)
        return u


class ADI_MCS(ADI_Base):
    def __init__(self, *args, **kwargs):
        if not 'th' in kwargs:
            kwargs['th'] = 0.5
        super().__init__(*args, **kwargs)
    
    def step(self, u, t, tau, xmesh, hx, ymesh, hy):
        xn = xmesh[:, 0]
        yn = ymesh[0, :]
        #first stage
        Lx, Ly, Lxy = self.problem.get_split(u, xmesh, hx, ymesh, hy, self.der)
        y0 = u + tau * (  Lx + Ly + Lxy )
        y0 = self.restore_boundary(y0, xn, hx, yn, hy, t + tau / 2)
        y1 = self.slae_x(y0, t, tau, xn, hx, yn, hy, Lx, Ly)
        y2 = self.slae_y(y1, t, tau, xn, hx, yn, hy, Lx, Ly)
        #second stage
        _Lx, _Ly, _Lxy = self.problem.get_split(y2, xmesh, hx, ymesh, hy, self.der)
        y0 = y0 + 0.5 * tau * ( _Lxy - Lxy )
        y0 = y0 + (0.5 - self.th) * tau * (_Lx + _Ly + _Lxy - Lx - Ly - Lxy)
        y0 = self.restore_boundary(y0, xn, hx, yn, hy, t + tau / 2)
        #third stage
        y1 = self.slae_x(y0, t + tau/2, tau, xn, hx, yn, hy, Lx, Ly)
        y2 = self.slae_y(y1, t + tau, tau, xn, hx, yn, hy, Lx, Ly)
        return y2

class ADI_HV(ADI_Base):
    def __init__(self, *args, **kwargs):
        if not 'th' in kwargs:
            kwargs['th'] = 0.5 + np.sqrt(3) / 6.0
        super().__init__(*args, **kwargs)
    
    def step(self, u, t, tau, xmesh, hx, ymesh, hy):
        xn = xmesh[:, 0]
        yn = ymesh[0, :]
        #first stage
        Lx, Ly, Lxy = self.problem.get_split(u, xmesh, hx, ymesh, hy, self.der)
        y0 = u + tau * (  Lx + Ly + Lxy )
        y1 = self.slae_x(y0, t, tau, xn, hx, yn, hy, Lx, Ly)
        y2 = self.slae_y(y1, t, tau, xn, hx, yn, hy, Lx, Ly)
        #second stage
        y2 = self.restore_boundary(y2, xn, hx, yn, hy, t)
        _Lx, _Ly, _Lxy = self.problem.get_split(y2, xmesh, hx, ymesh, hy, self.der)
        y0 = y0 + 0.5 * tau * (_Lx + _Ly + _Lxy - Lx - Ly - Lxy)
        #third stage
        y1 = self.slae_x(y0, t, tau, xn, hx, yn, hy, _Lx, _Ly)
        y2 = self.slae_y(y1, t, tau, xn, hx, yn, hy, _Lx, _Ly)
        return y2


