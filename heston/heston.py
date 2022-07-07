import numpy as np
import scipy.stats as sps


def BlackSholes(K, S0, T, r, theta, kappa, v0): 
    sig = np.sqrt(theta)
    d1 = (np.log(S0 / K) + (r + sig ** 2 / 2) * T) / (np.sqrt(T) * sig)
    d2 = (np.log(S0 / K) + (r - sig ** 2 / 2) * T) / (np.sqrt(T) * sig)
    Phi = sps.norm.cdf
    return S0 * Phi(d1) - np.exp(-r * T) * K * Phi(d2)


def getCD(u, tau, r, k, sig, theta, rho): 
    d = np.sqrt( (sig * rho * u * 1j - k) ** 2 + sig**2 * (1j * u + u ** 2) + 0j)
    
    g = (k - rho * sig * 1j * u - d) / (k - rho * sig * 1j * u + d)
    
    exp = np.exp(-d * tau)
    
    r1 = (k - rho * sig * 1j * u - d)
    
    D = 1.0 / (sig**2) * r1 * (1 - exp) / (1 - g * exp)
    C = r * u * tau * 1j + k * theta / (sig ** 2) * \
        ( r1 * tau - 2 * np.log( ((1 - g * exp)) / (1-g) ) )
    return C, D

def getPhi(u, tau, r, k, sig, theta, rho, x, v):
    C, D = getCD(u, tau, r, k, sig, theta, rho)
    return np.exp( C + v * D + 1j * u * x )

def getPhiTilda(u, tau, r, k, sig, theta, rho, x, v):
    return getPhi(u - 1j, tau, r, k, sig, theta, rho, x, v) / getPhi(-1j, tau, r, k, sig, theta, rho, x, v)


def getMesh(Nu):
    tn = np.linspace(0, 1, (Nu // 2) + 1)
    h = tn[1] - tn[0]
    tn = tn[:-1] + h / 2.0
    
    a = 20
    n = 1
    f = lambda t: a * (t ** n)
    df = lambda t: a * n * (t ** (n-1))
    
    g = lambda t: -np.log(1 - t)
    dg = lambda t: 1 / (1 - t)
    
    u1 = f(tn)
    h1 = h * df(tn)
    
    u2 = a + df(1.0) * g(tn)
    h2 = h * df(1.0) * dg(tn)
    
    un = np.r_[u1, u2]
    hn = np.r_[h1, h2]
    return un, hn
    
def getOptionPrice(S0, K, Nu, tau, r, k, sig, theta, rho, v):
    un, hn = getMesh(Nu)
    
    x = np.log(S0)
    
    phi      = getPhi(un, tau, r, k, sig, theta, rho, x, v)
    phitilda = getPhiTilda(un, tau, r, k, sig, theta, rho, x, v)

    
    phi  = phi.reshape(1, -1)
    phitilda = phitilda.reshape(1, -1)
    
    F1 = np.exp(-1j * un * np.log(K)) * phi / (1j * un)
    F2 = np.exp(-1j * un * np.log(K)) * phitilda / (1j * un)
    
    F1 = F1.real * hn
    F2 = F2.real * hn

    P1 = 0.5 + np.sum(F1, axis=1, keepdims=True) / np.pi
    P2 = 0.5 + np.sum(F2, axis=1, keepdims=True) / np.pi
    
    res = S0 * P2 - np.exp(-r * tau) * K * P1
    return res.reshape(-1)


def getOptionPriceFourierSeries(S0, K, Nu, tau, r, k, sig, theta, rho, v):
    
    alpha = 2.0
    
    U = 20
    un = np.linspace(0, U, Nu + 1)
    hu = un[1] - un[0]
    un = un[:-1] + hu / 2.0
    
    hn = hu * np.ones((Nu, ))
    
    x = np.log(S0)
    
    phi = getPhi(un - (alpha + 1) * 1j, tau, r, k, sig, theta, rho, x, v)
    psi = phi / (alpha ** 2 + alpha - un**2 + (2 * alpha + 1) * 1j * un)
    
    k = np.log(K)
    
    I = np.sum( np.exp(-1j * k * un) * psi , axis=-1, keepdims=True)
    C = np.exp(-r * T - alpha * k) / np.pi * hn * I
    C = C.real
    return C.reshape(-1)


def getOptionPriceFFT(S0, K, Nu, tau, r, k, sig, theta, rho, v):
    
    alpha = 1.0
    
    U = 80
    un = np.linspace(0, U, Nu + 1)
    hu = un[1] - un[0]
    un = un[:-1] + hu / 2.0
    
    logK = np.pi / hu
    kn = np.linspace(-logK, logK, Nu + 1).reshape(-1)
    hk = kn[1] - kn[0]
    kn = kn[:-1] + hk / 2.0
    
    zn = np.exp(-1j * hu * kn)
    
    x = np.log(S0)
    
    phi = getPhi(un - (alpha + 1) * 1j, tau, r, k, sig, theta, rho, x, v)
    psi = np.exp(-r * tau) * phi / (alpha ** 2 + alpha - un**2 + (2 * alpha + 1) * 1j * un)
    
    psi  = psi.reshape(-1)
    I = np.sqrt(zn) * np.fft.fft( (-1) ** (np.arange(Nu)) * psi * np.exp(-1j * np.pi  * np.arange(Nu) / Nu) )
    C = np.exp(-alpha * kn) * hu / np.pi * I
            
    C = C.reshape(-1).real
    Kn = np.exp(kn)
    i = np.where(Kn >= K[0])[0][0]
    j = np.where(Kn <= K[-1])[0][-1]
    
    
    return C[i:(j+1)], Kn[i:(j+1)]