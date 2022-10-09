import numpy as np
from scipy import stats as sps

from genPathes import *
from analytical import *


def ArithmeticMeanMoments(r, sig, t, T, St, It):
    tau = T - t

    #moments of int_t^T S_u du
    m1 = St * (np.exp(r * tau) - 1.0) / r
    m2 = 2 * St ** 2 * (
        r * np.exp( (sig ** 2 + 2 * r) * tau) - (sig**2 + 2 * r) * np.exp(r * tau) + (sig ** 2 + r)
    ) / (r * (sig ** 2 + r) * (sig**2 + 2 * r))
    var = m2 - m1 ** 2
    
    #moments of AT
    mu = (m1 + It) / T
    var = var / (T ** 2)
    return mu, var


def GeometricMeanMoments(r, sig, t, T, St, It):
    tau = T - t

    mu = It / T + tau / T * np.log(St) + (r - sig ** 2 / 2) * tau ** 2 / (2 * T)
    std = sig * np.sqrt( tau**3 / 3 ) / T
    
    return np.exp( mu + std ** 2 / 2 ), (np.exp(std ** 2) - 1) * np.exp(2 * mu + std ** 2)


def MonteCarloGeometricMean(r, sig, t, T, St, It, K, Nsim, nt, moment_matching=False):
    if not isinstance(K, np.ndarray):
        K = np.asarray([K])
    S, G = genGeometricMean(r, sig, t, T, St, It, Nsim, nt)
    
    tau = T - t

    #moments of GT
    mu, var = GeometricMeanMoments(r, sig, t, T, St, It)

    if moment_matching:  
        G = (G - np.mean(G)) / ( np.std(G) + 1e-10 ) * np.sqrt(var) + mu
    
    payoff = np.maximum(G.reshape(1, -1) - K.reshape(-1, 1), 0.0)
    C = np.exp( -r * tau ) * np.mean(payoff, axis=-1)
    return C


def MonteCarloArithmeticMean(r, sig, t, T, St, It, K, Nsim, nt, moment_matching=False):
    if not isinstance(K, np.ndarray):
        K = np.asarray([K])
    S, A = genArithmeticMean(r, sig, t, T, St, It, Nsim, nt)
    
    tau = T - t

    #moments of GT
    mu, var = ArithmeticMeanMoments(r, sig, t, T, St, It)

    if moment_matching:  
        A = (A - np.mean(A)) / ( np.std(A) + 1e-10 ) * np.sqrt(var) + mu
    
    payoff = np.maximum(A.reshape(1, -1) - K.reshape(-1, 1), 0.0)
    C = np.exp( -r * tau ) * np.mean(payoff, axis=-1)
    return C


def MonteCarloArithmeticMeanControlVariate(r, sig, t, T, St, I1, I2, K, Nsim, nt, moment_matching=False):
    ''' 
        r, sig -- model parameters
        t -- current time, T -- expiration time
        St -- current spot time
        I1 = int_0^t log S_t dt approx
        I2 = int_0^t S_t dt approx 
        Nsim -- number of simulated variables
        nt -- number of greed points in time
    '''

    if not isinstance(K, np.ndarray):
        K = np.asarray([K])
    S, G, A = genMeans(r, sig, t, T, St, I1, I2, Nsim, nt)

    tau = T - t


    #mu, var = GeometricMeanMoments(r, sig, t, T, St, I1)
    #if moment_matching:  
    #    G = (G - np.mean(G)) / ( np.std(G) + 1e-10 ) * np.sqrt(var) + mu


    mu, var = ArithmeticMeanMoments(r, sig, t, T, St, I2)
    if moment_matching:  
        A = (A - np.mean(A)) / ( np.std(A) + 1e-10 ) * np.sqrt(var) + mu

    
    payoff_A = np.exp(-r * tau) * np.maximum(A.reshape(1, -1) - K.reshape(-1, 1), 0.0)
    payoff_G = np.exp(-r * tau) * np.maximum(G.reshape(1, -1) - K.reshape(-1, 1), 0.0)
    
    C_G_analit = GeometricMeanAnalytical(r, sig, t, T, St, I1, K)
    
    beta = -np.cov(payoff_A, payoff_G)[0, 1] / np.std(payoff_G) ** 2

    X = payoff_A + beta * (payoff_G - C_G_analit.reshape(-1, 1))
    return np.mean(X, axis=-1)

