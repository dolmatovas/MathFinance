import numpy as np

from matplotlib import pyplot as plt

from montecarlo import *
from heston import *

def MonteCarloPriceEstimation(S0, Kn, Nt, T, r, kappa, sigma, theta, rho, v0, Npath, MomentMatching=False):
    x0 = np.log(S0)
    tn = np.linspace(0, T, Nt)
    v, x = generate_path_chi(r, kappa, sigma, theta, rho, x0, v0, tn, Npath)
    S = np.exp(x)
    if MomentMatching:
        S = (S - np.mean(S)) + S0 * np.exp(r * T)
    S = S.reshape(1, -1)
    payoffs = np.maximum( S - Kn.reshape(-1, 1), 0.0 )
    C2 = np.exp(-r * T) * np.mean(payoffs, axis=1)
    return C2.squeeze()

if __name__ == '__main__':
    #model params
    r = 0.025
    sigma = 0.3
    kappa = 1.5
    theta =  0.04
    rho = -0.9

    #Option params:
    K0 = 1.0
    Kn = np.linspace(0.8, 1.2, 1000) * K0
    T = 1.0

    #initial price
    S0 = 1.0
    #initial vol
    v0 = 0.0175

    Nu = 2000
    C = getOptionPrice(S0, Kn, Nu, T, r, kappa, sigma, theta, rho, v0)
    Nt = 70
    Npath = 50000
    M1 = MonteCarloPriceEstimation(S0, Kn, Nt, T, r, kappa, sigma, theta, rho, v0, Npath)
    M2 = MonteCarloPriceEstimation(S0, Kn, Nt, T, r, kappa, sigma, theta, rho, v0, Npath, True)
    plot = False
    if plot:
        plt.plot(Kn, C, label='fourier')
        plt.plot(Kn, M1, label="WO matching")
        plt.plot(Kn, M2, label="With matching")
        plt.grid()
        plt.legend()
        plt.show()
    err1 = np.linalg.norm(C - M1) / np.linalg.norm(C) * 100
    err2 = np.linalg.norm(C - M2) / np.linalg.norm(C) * 100
    print(f"{err1:.4}%, {err2:.4}%")

    assert err1 < 1.0
    assert err2 < 1.0
