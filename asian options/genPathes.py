import numpy as np
from scipy import stats as sps

def genIncrements(nt, Nsim):
    eps = np.random.randn( nt - 1, Nsim // 2)
    eps = np.c_[eps, -eps]

    return eps


def genBrownianMotion(tn, Nsim):
    eps = genIncrements(len(tn), Nsim)

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

    tau = (T - t) / nt
    
    X = np.log(St)
    I = It
    
    for i in range(nt):
        
        e = np.random.randn(Nsim // 2)
        e = np.r_[e, -e]

        Xnew = X + tau * (r - sig ** 2 / 2) + sig * np.sqrt(tau) * e
        
        I += tau * (Xnew + X) / 2
        X = Xnew
        
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

    tau = (T - t) / nt
    
    S = St
    I = It
    
    for i in range(nt):

        e = np.random.randn(Nsim // 2)
        e = np.r_[e, -e]
        
        Snew = S * np.exp(tau * (r - sig ** 2 / 2) + sig * np.sqrt(tau) * e)

        I += tau * (Snew + S) / 2
        S = Snew

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

    tau = (T - t) / nt
    
    X = np.log(St)
    S = St
    
    for i in range(nt):
        
        e = np.random.randn(Nsim // 2)
        e = np.r_[e, -e]

        Xnew = X + tau * (r - sig ** 2 / 2) + sig * np.sqrt(tau) * e
        Snew = np.exp(Xnew)

        I1 += tau * (Xnew + X) / 2
        I2 += tau * (Snew + S) / 2
        X = Xnew
        S = Snew
        
    G = np.exp(I1 / T)
    A = I2 / T    

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