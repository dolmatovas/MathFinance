from utils import *
from SLAE_solvers import *

def solve_alternating_direction(hestonParams : HestonParams,
                optionParams: OptionParams,
                gridParams: GridParams):
    
    r, sigma, kappa, theta, rho = hestonParams.r,hestonParams.sigma, hestonParams.kappa,hestonParams.theta,hestonParams.rho
    tn, xn, yn = GetGrid(hestonParams, optionParams, gridParams)
    tau = tn[1] - tn[0]
    hx = xn[1] - xn[0]
    hy = yn[1] - yn[0]
    
    u = np.zeros(tn.shape + xn.shape + yn.shape)
    u[0, :, :] = repmat(np.maximum(1.0 - np.exp(xn), 0.0), gridParams.Ny + 1, 1).T
    sl = slice(1, -1, 1)
    shape = u[0].shape
    
    ux, uy, uyy, uxx, uxy  = np.zeros(shape), np.zeros(shape), np.zeros(shape), np.zeros(shape), np.zeros(shape)
    y0, y1, y2, y3 = np.zeros(shape), np.zeros(shape), np.zeros(shape), np.zeros(shape)

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
    Ay[sl] = 1 + 0.5 * sigma * yn[sl] * tau / hy ** 2
    By[sl] = - tau / (4.0 * hy) * ( mux[sl] + sigmax[sl] / (hy) )
    Cy[sl] = - tau / (4.0 * hy) * (-mux[sl] + sigmax[sl] / (hy) )
    
    for t in range(gridParams.M):
        #first step
        y0 = u[t, :, :]
        uxy[sl, sl] = (y0[2:, 2:] - y0[:-2, 2:] - y0[2:, :-2] + y0[:-2, :-2]) / (4 * hx * hy)
        y1 = y0 + 0.5 * tau * sigmaxy * uxy
        #second step
        uy[sl, sl] = (y1[sl, 2:] - y1[sl, :-2]) / (2 * hy)
        uyy[sl, sl] = (y1[sl, 2:] - 2 * y1[sl, 1:-1] + y1[sl, :-2]) / (hy ** 2)
        #impicit x, explicit y
        for m in range(1, gridParams.Ny):
            Ax[sl] = 1 + 0.5 * sigma * yn[m] * tau / hx ** 2
            Bx[sl] = - tau / (4.0 * hx) * ( mux[m] + sigmax[m] / (hx) )
            Cx[sl] = - tau / (4.0 * hx) * (-mux[m] + sigmax[m] / (hx) )
            Fx[sl] = y1[sl, m] + 0.5 * tau * ( muy[m] * uy[sl, m] + 0.5 * sigmay[m] * uyy[sl, m] )
            y2[:, m] = Progonka(Ax, Bx, Cx, Fx)
        y2[:, 0] = y2[:, 1]
        y2[:, -1] = y2[:, -1]
        #third step
        ux[sl, sl] = (y2[2:, sl] - y2[:-2, sl]) / (2 * hx)
        uxx[sl, sl] = (y2[2:, sl] - 2 * y2[1:-1, sl] + y1[:-2, sl]) / (hx ** 2)
        #exicit x, implicit y
        for n in range(1, gridParams.Nx):
            Fy[sl] = y2[n, sl] + 0.5 * tau * ( mux[sl] * ux[n, sl] + 0.5 * sigmax[sl] * uxx[n, sl] )
            y3[n, :] = Progonka(Ay, By, Cy, Fy)
        y3[:, 0] = 1 - np.exp(xn[0])
        #last step
        uxy[sl, sl] = (y3[2:, 2:] - y3[:-2, 2:] - y3[2:, :-2] + y3[:-2, :-2])\
            / (4 * hx * hy)
        u[t + 1, :, :] = y3 + 0.5 * tau * sigmaxy * uxy
        
        u[t + 1, :, 0] = u[t + 1, :, 1]
        u[t + 1, :, -1] = u[t + 1, :, -2]
        u[t + 1, 0, :] = 1.0 - np.exp(xn[0])
        u[t + 1, -1, :] = 0.0
    return u


def solve_alternating_direction_LU(hestonParams : HestonParams,
                optionParams: OptionParams,
                gridParams: GridParams):
    
    r, sigma, kappa, theta, rho = hestonParams.r,hestonParams.sigma, hestonParams.kappa,hestonParams.theta,hestonParams.rho
    tn, xn, yn = GetGrid(hestonParams, optionParams, gridParams)
    tau = tn[1] - tn[0]

    h1 = xn[1:-1] - xn[:-2]
    h2 = xn[2:] - xn[1:-1]

    d1 = yn[1:-1] - yn[:-2]
    d2 = yn[2:] - yn[1:-1]

    h1 = h1.reshape(-1, 1)
    h2 = h2.reshape(-1, 1)
    d1 = d1.reshape(1, -1)
    d2 = d2.reshape(1, -1)

    
    u = np.zeros(tn.shape + xn.shape + yn.shape)
    u[0, :, :] = repmat(np.maximum(1.0 - np.exp(xn), 0.0), gridParams.Ny + 1, 1).T
    sl = slice(1, -1, 1)
    shape = u[0].shape
    
    ux, uy, uyy, uxx, uxy  = np.zeros(shape), np.zeros(shape), np.zeros(shape), np.zeros(shape), np.zeros(shape)
    
    y0, y1, y2, y3 = np.zeros(shape), np.zeros(shape), np.zeros(shape), np.zeros(shape)
    
    mux = (r - 0.5 * yn * sigma)
    muy = kappa * (theta - yn * sigma) / sigma 
        
    sigmax = sigma * yn
    sigmay = sigma * yn
    sigmaxy = sigma * yn * rho
    
    Ax, Bx, Cx, Fx = np.zeros(xn.shape), np.zeros(xn.shape), np.zeros(xn.shape), np.zeros(xn.shape)    
    Ax[0] = Ax[-1] = 1.0
    Fx[0] = 1 - np.exp(xn[0])
    
    Lx, Ux, Vx = np.zeros(yn.shape + xn.shape), np.zeros(yn.shape + xn.shape), np.zeros(yn.shape + xn.shape)
    for m in range(1, gridParams.Ny):
        Ax[sl] = (1 + 0.5 * tau / (h1 * h2) * ( sigmax[m] - mux[m] * (h2 - h1) )).reshape(-1)
        Bx[sl] = (-0.5 * tau / (h2 * (h1 + h2)) * (sigmax[m] + h1 * mux[m])).reshape(-1)
        Cx[sl] = (-0.5 * tau / (h1 * (h1 + h2)) * (sigmax[m] - h2 * mux[m])).reshape(-1)

        #Ax[sl] = 1 + 0.5 * sigma * yn[m] * tau / hx ** 2
        #Bx[sl] = - tau / (4.0 * hx) * ( mux[m] + sigmax[m] / (hx) )
        #Cx[sl] = - tau / (4.0 * hx) * (-mux[m] + sigmax[m] / (hx) )
        Lx[m], Ux[m], Vx[m] = LU(Ax, Bx, Cx)
     
    Ay, By, Cy, Fy = np.zeros(yn.shape), np.zeros(yn.shape), np.zeros(yn.shape), np.zeros(yn.shape)
    Ay[0] = Ay[-1] = 1.0
    By[0] = Cy[-1] = -1.0    
    Ay[sl] = 1 + 0.5 * tau / (d1 * d2) * ( sigmay[sl] - muy[sl] * (d2 - d1) )
    By[sl] = -0.5 * tau / (d2 * (d1 + d2)) * (sigmay[sl] + d1 * muy[sl])
    Cy[sl] = -0.5 * tau / (d1 * (d1 + d2)) * (sigmay[sl] - d2 * muy[sl])
    Ly, Uy, Vy = LU(Ay, By, Cy)
    
    for t in range(gridParams.M):
        y0 = u[t, :, :]
        
        #uxy[sl, sl] = (y0[2:, 2:] - y0[:-2, 2:] - y0[2:, :-2] + y0[:-2, :-2]) /  (4 * hx * hy)

        uy[sl, sl] = ((d1 / d2) * (y0[sl, 2:] - y0[sl, 1:-1]) + (d2 / d1) * (y0[sl, 1:-1] - y0[sl, :-2]) ) / (d1 + d2)
        uxy[sl, sl] = ( (h1 / h2) * (uy[2:, sl] - uy[1:-1, sl]) + (h2 / h1) * (uy[1:-1, sl] - uy[:-2, sl]) ) / (h1 + h2)

        #first step
        y1 = y0 + 0.5 * tau * sigmaxy * uxy


        uy[sl, sl] = ((d1 / d2) * (y1[sl, 2:] - y1[sl, 1:-1]) + (d2 / d1) * (y1[sl, 1:-1] - y1[sl, :-2]) ) / (d1 + d2)
        uyy[sl, sl] = ( (y1[sl, 2:] - y1[sl, 1:-1]) / d2 - (y1[sl, 1:-1] - y1[sl, :-2]) / d1 ) / (d1 + d2)
        
        #uniform
        #uy[sl, sl] = (y1[sl, 2:] - y1[sl, :-2]) / (2 * hy)
        #uyy[sl, sl] = (y1[sl, 2:] - 2 * y1[sl, 1:-1] + y1[sl, :-2]) / (hy ** 2)
        #impicit x, explicit y
        for m in range(1, gridParams.Ny):
            Fx[sl] = y1[sl, m] + 0.5 * tau * ( muy[m] * uy[sl, m] + 0.5 * sigmay[m] * uyy[sl, m] )
            y2[:, m] = SolveLU(Lx[m], Ux[m], Vx[m], Fx)
        y2[:, 0] = y2[:, 1]
        y2[:, -1] = y2[:, -1]
        


        ux[sl, sl] = ((h1 / h2) * (y2[2:, sl] - y2[1:-1, sl]) + (h2 / h1) * (y2[1:-1, sl] - y2[:-2, sl]) ) / (h1 + h2)
        uxx[sl, sl] = ( (y2[2:, sl] - y2[1:-1, sl]) / h2 - (y2[1:-1, sl] - y2[:-2, sl]) / h1 ) / (h1 + h2)

        
        #uniform
        #ux[sl, sl] = (y2[2:, sl] - y2[:-2, sl]) / (2 * hx)
        #uxx[sl, sl] = (y2[2:, sl] - 2 * y2[1:-1, sl] + y1[:-2, sl]) / (hx ** 2)

        #exicit x, implicit y
        for n in range(1, gridParams.Nx):
            Fy[sl] = y2[n, sl] + 0.5 * tau * ( mux[sl] * ux[n, sl] + 0.5 * sigmax[sl] * uxx[n, sl] )
            y3[n, :] = SolveLU(Ly, Uy, Vy, Fy)
        y3[:, 0] = 1 - np.exp(xn[0])
        
        #last step

        #uniform
        #uxy[sl, sl] = (y3[2:, 2:] - y3[:-2, 2:] - y3[2:, :-2] + y3[:-2, :-2]) / (4 * hx * hy)
        
        ###not uniform
        uy[sl, sl] = ((d1 / d2) * (y3[sl, 2:] - y3[sl, 1:-1]) + (d2 / d1) * (y3[sl, 1:-1] - y3[sl, :-2]) ) / (d1 + d2)
        uxy[sl, sl] = ( (h1 / h2) * (uy[2:, sl] - uy[1:-1, sl]) + (h2 / h1) * (uy[1:-1, sl] - uy[:-2, sl]) ) / (h1 + h2)
        ###
        
        #first step
        u[t + 1, :, :] = y3 + 0.5 * tau * sigmaxy * uxy
        
        u[t + 1, :, 0] = u[t + 1, :, 1]
        u[t + 1, :, -1] = u[t + 1, :, -2]

        u[t + 1, 0, :] = 1.0 - np.exp(xn[0])
        u[t + 1, -1, :] = 0.0
    return u