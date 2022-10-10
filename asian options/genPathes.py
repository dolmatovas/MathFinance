import numpy as np
from scipy import stats as sps

def genIncrements(nt, Nsim):
    eps = np.random.randn( nt , Nsim // 2)
    eps = np.c_[eps, -eps]

    return eps


def genBrownianMotion(tn, Nsim):
    eps = genIncrements(len(tn) - 1, Nsim)

    eps = eps * np.sqrt( np.diff(tn).reshape(-1, 1) )
    eps = np.r_[np.zeros((1, Nsim)), eps]
    W = np.cumsum(eps, axis=0)

    return W


def genGeometricMean(r:float, sig:float, t:float, T:float, St:float, It:float, Nsim:int, nt):
    ''' 
        r, sig -- model parameters
        t -- current time, T -- expiration time
        St -- current spot time
        It = int_0^t log S_t dt
        Nsim -- number of simulated variables
        nt -- number of greed points in time
    '''

    tn, ht = np.linspace(t, T, nt + 1, retstep=True)
    W = genBrownianMotion(tn, Nsim)

    X = np.log(St) + (r - sig ** 2 / 2) * (tn - t).reshape(-1, 1) + sig * W

    I = It + ht * np.sum( X[1:-1, :], axis=0 ) + 0.5 * ht * ( X[0, :] + X[-1, :] )
   
    S = np.exp(X)
    G = np.exp(I / T)

    return S, G


def genArithmeticMean(r:float, sig:float, t:float, T:float, St:float, It:float, Nsim:int, nt):
    ''' 
        r, sig -- model parameters
        t -- current time, T -- expiration time
        St -- current spot time
        It = int_0^t S_t dt
        Nsim -- number of simulated variables
        nt -- number of greed points in time
    '''

    tn, ht = np.linspace(t, T, nt + 1, retstep=True)
    W = genBrownianMotion(tn, Nsim)

    S = St * np.exp( (r - sig ** 2 / 2) * (tn - t).reshape(-1, 1) + sig * W )

    I = It + ht * np.sum( S[1:-1, :], axis=0 ) + 0.5 * ht * ( S[0, :] + S[-1, :] )
    A = I / T
    
    return S, A


def genMeans(r:float, sig:float, t:float, T:float, St:float, I1:float, I2:float, Nsim:int, nt):
    ''' 
        r, sig -- model parameters
        t -- current time, T -- expiration time
        St -- current spot time
        I1 = int_0^t log S_t dt approx
        I2 = int_0^t S_t dt approx 
        Nsim -- number of simulated variables
        nt -- number of greed points in time
    '''

    tn, ht = np.linspace(t, T, nt + 1, retstep=True)
    W = genBrownianMotion(tn, Nsim)

    X = np.log(St) + (r - sig ** 2 / 2) * (tn - t).reshape(-1, 1) + sig * W
    I = I1 + ht * np.sum( X[1:-1, :], axis=0 ) + 0.5 * ht * ( X[0, :] + X[-1, :] )
    G = np.exp(I / T)

    S = np.exp(X)
 
    I = I2 + ht * np.sum( S[1:-1, :], axis=0 ) + 0.5 * ht * ( S[0, :] + S[-1, :] )
    A = I / T   

    return S, G, A


def genConditionalArithmeticMean(r:float, sig:float, t:float, T:float, St:float, ST:float, It:float, Nsim:int, nt):
    '''
        r, sig -- model parameters
        t -- current time, T -- expiration time
        St -- current spot time
        ST -- spot price at time T
        It = int_0^t S_t dt approx 
        Nsim -- number of simulated variables
        nt -- number of greed points in time
    '''
    tau = (T - t)

    XT = np.log(ST / St)

    BT = (XT - (r - sig ** 2 / 2 ) * tau  ) / sig

    tn, ht = np.linspace(0, tau, nt + 1, retstep=True)
    W = genBrownianMotion(tn, Nsim)

    B = W + (BT - W[-1, :].reshape(1, -1)) * tn.reshape(-1, 1) / tau
    X = (r - sig ** 2 / 2) * tn.reshape(-1, 1) + sig * B

    S = St * np.exp(X)

    I = It + ht * np.sum( S[1:-1, :], axis=0 ) + 0.5 * ht * ( S[0, :] + S[-1, :] )

    return S, I / T