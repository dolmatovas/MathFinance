from utils import *
from SLAE_solvers import *

from derivatives import GetDerivatives

def solve_alternating_direction_DO(hestonParams : HestonParams,
                optionParams: OptionParams,
                gridParams: GridParams):
    th = 1.0

    r, sigma, kappa, theta, rho = hestonParams.r,hestonParams.sigma, hestonParams.kappa,hestonParams.theta,hestonParams.rho
    tn, xn, yn = GetGrid(hestonParams, optionParams, gridParams)
    tau = tn[1] - tn[0]
    hx = xn[1:] - xn[:-1]
    hy = yn[1:] - yn[:-1]

    h1 = hx[:-1].reshape(-1, 1)
    h2 = hx[1:].reshape(-1, 1)
    d1 = hy[:-1].reshape(1, -1)
    d2 = hy[1:].reshape(1, -1)
    
    u = np.zeros(tn.shape + xn.shape + yn.shape)
    u[0, :, :] = repmat(np.maximum(1.0 - np.exp(xn), 0.0), gridParams.Ny + 1, 1).T
    sl = slice(1, -1, 1)
    shape = u[0].shape
    
    y0, y1, y2 = np.zeros(shape), np.zeros(shape), np.zeros(shape)

    mux = (r - 0.5 * yn * sigma)
    muy = kappa * (theta - yn * sigma) / sigma 
        
    sigmax = sigma * yn
    sigmay = sigma * yn
    sigmaxy = sigma * yn * rho
    
    Ax, Bx, Cx, Fx = np.zeros(xn.shape), np.zeros(xn.shape), np.zeros(xn.shape), np.zeros(xn.shape) 
    Ax[0] = Ax[-1] = 1.0
    Fx[0] = 1 - np.exp(xn[0])       
    
    Ay, By, Cy, Fy = np.zeros(yn.shape), np.zeros(yn.shape), np.zeros(yn.shape), np.zeros(yn.shape)
    Ay[0] = Ay[-1] = 1.0
    By[0] = Cy[-1] = -1.0    
    
    for t in range(gridParams.M):
        tau = tn[t + 1] - tn[t]
        #first step
        ux, uy, uxx, uyy, uxy = GetDerivatives(u[t, :, :], hx, hy)

        Lx = mux * ux + 0.5 * sigmax * uxx
        Ly = muy * uy + 0.5 * sigmay * uyy

        Lxy = sigmaxy * uxy 
        y0 = u[t, :, :] + tau * (  Lx + Ly + Lxy )

        #second step
        for m in range(1, gridParams.Ny):
            Ax[sl] = (1 + th * tau / (h1 * h2) * ( sigmax[m] - mux[m] * (h2 - h1) )).reshape(-1)
            Bx[sl] = (-th * tau / (h2 * (h1 + h2)) * (sigmax[m] + h1 * mux[m])).reshape(-1)
            Cx[sl] = (-th * tau / (h1 * (h1 + h2)) * (sigmax[m] - h2 * mux[m])).reshape(-1)
            
            Fx[sl] = y0[sl, m] - th * tau * Lx[sl, m]
            y1[:, m] = Progonka(Ax, Bx, Cx, Fx)
        y1[:, 0] = y1[:, 1]
        y1[:, -1] = y1[:, -1]
        #third step
        for n in range(1, gridParams.Nx):
            Ay[sl] = 1 + th * tau / (d1 * d2) * ( sigmay[sl] - muy[sl] * (d2 - d1) )
            By[sl] = -th * tau / (d2 * (d1 + d2)) * (sigmay[sl] + d1 * muy[sl])
            Cy[sl] = -th * tau / (d1 * (d1 + d2)) * (sigmay[sl] - d2 * muy[sl])

            Fy[sl] = y1[n, sl] - th * tau * Ly[n, sl]
            y2[n, :] = Progonka(Ay, By, Cy, Fy)
        y2[0, :] = 1.0 - np.exp(xn[0])
    
        u[t + 1, :, :] = y2[:, :]
        
        u[t + 1, :, 0] = u[t + 1, :, 1]
        u[t + 1, :, -1] = u[t + 1, :, -2]
        u[t + 1, 0, :] = 1.0 - np.exp(xn[0])
        u[t + 1, -1, :] = 0.0
    return u


def solve_alternating_direction_CS(hestonParams : HestonParams,
                optionParams: OptionParams,
                gridParams: GridParams):
    th = 1.0

    r, sigma, kappa, theta, rho = hestonParams.r,hestonParams.sigma, hestonParams.kappa,hestonParams.theta,hestonParams.rho
    tn, xn, yn = GetGrid(hestonParams, optionParams, gridParams)
    tau = tn[1] - tn[0]
    hx = xn[1:] - xn[:-1]
    hy = yn[1:] - yn[:-1]

    h1 = hx[:-1].reshape(-1, 1)
    h2 = hx[1:].reshape(-1, 1)
    d1 = hy[:-1].reshape(1, -1)
    d2 = hy[1:].reshape(1, -1)
    
    u = np.zeros(tn.shape + xn.shape + yn.shape)
    u0 = np.maximum(1.0 - np.exp(xn), 0.0)
    u[0, :, :] = repmat(u0, gridParams.Ny + 1, 1).T
    sl = slice(1, -1, 1)
    shape = u[0].shape
    
    y0, y1, y2 = np.zeros(shape), np.zeros(shape), np.zeros(shape)

    mux = (r - 0.5 * yn * sigma)
    muy = kappa * (theta - yn * sigma) / sigma 
        
    sigmax = sigma * yn
    sigmay = sigma * yn
    sigmaxy = sigma * yn * rho
    
    Ax, Bx, Cx, Fx = np.zeros(xn.shape), np.zeros(xn.shape), np.zeros(xn.shape), np.zeros(xn.shape) 
    Ax[0] = Ax[-1] = 1.0
    Fx[0] = 1 - np.exp(xn[0])       
    
    Ay, By, Cy, Fy = np.zeros(yn.shape), np.zeros(yn.shape), np.zeros(yn.shape), np.zeros(yn.shape)
    Ay[0] = Ay[-1] = 1.0
    By[0] = Cy[-1] = -1.0    
    
    for t in range(gridParams.M):
        tau = tn[t + 1] - tn[t]
        #first step
        ux, uy, uxx, uyy, uxy = GetDerivatives(u[t, :, :], hx, hy)

        Lx = mux * ux + 0.5 * sigmax * uxx
        Ly = muy * uy + 0.5 * sigmay * uyy

        Lxy = sigmaxy * uxy 
        y0 = u[t, :, :] + tau * (  Lx + Ly + Lxy )

        #second step
        for m in range(1, gridParams.Ny):
            Ax[sl] = (1 + th * tau / (h1 * h2) * ( sigmax[m] - mux[m] * (h2 - h1) )).reshape(-1)
            Bx[sl] = (-th * tau / (h2 * (h1 + h2)) * (sigmax[m] + h1 * mux[m])).reshape(-1)
            Cx[sl] = (-th * tau / (h1 * (h1 + h2)) * (sigmax[m] - h2 * mux[m])).reshape(-1)
            
            Fx[sl] = y0[sl, m] - th * tau * Lx[sl, m]
            y1[:, m] = Progonka(Ax, Bx, Cx, Fx)
        y1[:, 0] = y1[:, 1]
        y1[:, -1] = y1[:, -1]
        #third step
        for n in range(1, gridParams.Nx):
            Ay[sl] = 1 + th * tau / (d1 * d2) * ( sigmay[sl] - muy[sl] * (d2 - d1) )
            By[sl] = -th * tau / (d2 * (d1 + d2)) * (sigmay[sl] + d1 * muy[sl])
            Cy[sl] = -th * tau / (d1 * (d1 + d2)) * (sigmay[sl] - d2 * muy[sl])

            Fy[sl] = y1[n, sl] - th * tau * Ly[n, sl]
            y2[n, :] = Progonka(Ay, By, Cy, Fy)
        y2[0, :] = 1.0 - np.exp(xn[0])


        _, _, _, _, uxy = GetDerivatives(y2[:, :], hx, hy)
        y0 = y0 + 0.5 * tau * ( uxy * sigmaxy - Lxy )


        #second step
        for m in range(1, gridParams.Ny):
            Ax[sl] = (1 + th * tau / (h1 * h2) * ( sigmax[m] - mux[m] * (h2 - h1) )).reshape(-1)
            Bx[sl] = (-th * tau / (h2 * (h1 + h2)) * (sigmax[m] + h1 * mux[m])).reshape(-1)
            Cx[sl] = (-th * tau / (h1 * (h1 + h2)) * (sigmax[m] - h2 * mux[m])).reshape(-1)
            
            Fx[sl] = y0[sl, m] - th * tau * Lx[sl, m]
            y1[:, m] = Progonka(Ax, Bx, Cx, Fx)
        y1[:, 0] = y1[:, 1]
        y1[:, -1] = y1[:, -1]
        #third step
        for n in range(1, gridParams.Nx):
            Ay[sl] = 1 + th * tau / (d1 * d2) * ( sigmay[sl] - muy[sl] * (d2 - d1) )
            By[sl] = -th * tau / (d2 * (d1 + d2)) * (sigmay[sl] + d1 * muy[sl])
            Cy[sl] = -th * tau / (d1 * (d1 + d2)) * (sigmay[sl] - d2 * muy[sl])

            Fy[sl] = y1[n, sl] - th * tau * Ly[n, sl]
            y2[n, :] = Progonka(Ay, By, Cy, Fy)
        y2[0, :] = 1.0 - np.exp(xn[0])
    
        u[t + 1, :, :] = y2[:, :]
        
        u[t + 1, :, 0] = u[t + 1, :, 1]
        u[t + 1, :, -1] = u[t + 1, :, -2]
        u[t + 1, 0, :] = 1.0 - np.exp(xn[0])
        u[t + 1, -1, :] = 0.0
    return u


def solve_alternating_direction_MCS(hestonParams : HestonParams,
                optionParams: OptionParams,
                gridParams: GridParams):
    th = 1.0

    r, sigma, kappa, theta, rho = hestonParams.r,hestonParams.sigma, hestonParams.kappa,hestonParams.theta,hestonParams.rho
    tn, xn, yn = GetGrid(hestonParams, optionParams, gridParams)
    tau = tn[1] - tn[0]
    hx = xn[1:] - xn[:-1]
    hy = yn[1:] - yn[:-1]

    h1 = hx[:-1].reshape(-1, 1)
    h2 = hx[1:].reshape(-1, 1)
    d1 = hy[:-1].reshape(1, -1)
    d2 = hy[1:].reshape(1, -1)
    
    u = np.zeros(tn.shape + xn.shape + yn.shape)
    u[0, :, :] = repmat(np.maximum(1.0 - np.exp(xn), 0.0), gridParams.Ny + 1, 1).T
    sl = slice(1, -1, 1)
    shape = u[0].shape
    
    y0, y1, y2 = np.zeros(shape), np.zeros(shape), np.zeros(shape)

    mux = (r - 0.5 * yn * sigma)
    muy = kappa * (theta - yn * sigma) / sigma 
        
    sigmax = sigma * yn
    sigmay = sigma * yn
    sigmaxy = sigma * yn * rho
    
    Ax, Bx, Cx, Fx = np.zeros(xn.shape), np.zeros(xn.shape), np.zeros(xn.shape), np.zeros(xn.shape) 
    Ax[0] = Ax[-1] = 1.0
    Fx[0] = 1 - np.exp(xn[0])       
    
    Ay, By, Cy, Fy = np.zeros(yn.shape), np.zeros(yn.shape), np.zeros(yn.shape), np.zeros(yn.shape)
    Ay[0] = Ay[-1] = 1.0
    By[0] = Cy[-1] = -1.0    
    
    for t in range(gridParams.M):
        tau = tn[t + 1] - tn[t]
        #first step
        ux, uy, uxx, uyy, uxy = GetDerivatives(u[t, :, :], hx, hy)

        Lx = mux * ux + 0.5 * sigmax * uxx
        Ly = muy * uy + 0.5 * sigmay * uyy

        Lxy = sigmaxy * uxy 
        y0 = u[t, :, :] + tau * (  Lx + Ly + Lxy )

        #second step
        for m in range(1, gridParams.Ny):
            Ax[sl] = (1 + th * tau / (h1 * h2) * ( sigmax[m] - mux[m] * (h2 - h1) )).reshape(-1)
            Bx[sl] = (-th * tau / (h2 * (h1 + h2)) * (sigmax[m] + h1 * mux[m])).reshape(-1)
            Cx[sl] = (-th * tau / (h1 * (h1 + h2)) * (sigmax[m] - h2 * mux[m])).reshape(-1)
            
            Fx[sl] = y0[sl, m] - th * tau * Lx[sl, m]
            y1[:, m] = Progonka(Ax, Bx, Cx, Fx)
        y1[:, 0] = y1[:, 1]
        y1[:, -1] = y1[:, -1]
        #third step
        for n in range(1, gridParams.Nx):
            Ay[sl] = 1 + th * tau / (d1 * d2) * ( sigmay[sl] - muy[sl] * (d2 - d1) )
            By[sl] = -th * tau / (d2 * (d1 + d2)) * (sigmay[sl] + d1 * muy[sl])
            Cy[sl] = -th * tau / (d1 * (d1 + d2)) * (sigmay[sl] - d2 * muy[sl])

            Fy[sl] = y1[n, sl] - th * tau * Ly[n, sl]
            y2[n, :] = Progonka(Ay, By, Cy, Fy)
        y2[0, :] = 1.0 - np.exp(xn[0])


        y2x, y2y , y2xx, y2yy, y2xy = GetDerivatives(y2[:, :], hx, hy)
        y0 = y0 + th * tau * ( y2xy * sigmaxy - Lxy )

        _Lx = y2x * mux + 0.5 * sigmax * y2xx
        _Ly = y2y * muy + 0.5 * sigmay * y2yy
        _Lxy = sigmaxy * y2xy
        y0 = y0 + (0.5 - th)* tau * ( (_Lx + _Ly + _Lxy) - (Lx + Ly + Lxy) )

        y0[:, 0] = y0[:, 1]
        y0[:, -1] = y0[:, -2]
        y0[0, :] = 1.0 - np.exp(xn[0])
        y0[-1, :] = 0.0
        #second step
        for m in range(1, gridParams.Ny):
            Ax[sl] = (1 + th * tau / (h1 * h2) * ( sigmax[m] - mux[m] * (h2 - h1) )).reshape(-1)
            Bx[sl] = (-th * tau / (h2 * (h1 + h2)) * (sigmax[m] + h1 * mux[m])).reshape(-1)
            Cx[sl] = (-th * tau / (h1 * (h1 + h2)) * (sigmax[m] - h2 * mux[m])).reshape(-1)
            
            Fx[sl] = y0[sl, m] - th * tau * Lx[sl, m]
            y1[:, m] = Progonka(Ax, Bx, Cx, Fx)
        y1[:, 0] = y1[:, 1]
        y1[:, -1] = y1[:, -1]
        #third step
        for n in range(1, gridParams.Nx):
            Ay[sl] = 1 + th * tau / (d1 * d2) * ( sigmay[sl] - muy[sl] * (d2 - d1) )
            By[sl] = -th * tau / (d2 * (d1 + d2)) * (sigmay[sl] + d1 * muy[sl])
            Cy[sl] = -th * tau / (d1 * (d1 + d2)) * (sigmay[sl] - d2 * muy[sl])

            Fy[sl] = y1[n, sl] - th * tau * Ly[n, sl]
            y2[n, :] = Progonka(Ay, By, Cy, Fy)
        y2[0, :] = 1.0 - np.exp(xn[0])
    
        u[t + 1, :, :] = y2[:, :]
        
        u[t + 1, :, 0] = u[t + 1, :, 1]
        u[t + 1, :, -1] = u[t + 1, :, -2]
        u[t + 1, 0, :] = 1.0 - np.exp(xn[0])
        u[t + 1, -1, :] = 0.0
    return u