from scipy import stats as sps
import numpy as np


def genQuasiMonteCarlo(r, sig, t, T, St, It, Nsim, nt):
    sampler = sps.qmc.Sobol(nt)

    eps = sampler.random(Nsim).T
    eps = sps.norm.ppf(eps)
    assert eps.shape == (nt, Nsim)

    tau = T - t
    tn, ht = np.linspace(0, tau, nt + 1, retstep=True)

    eps = eps * np.sqrt( ht )
    eps = np.r_[np.zeros((1, Nsim)), eps]
    W = np.cumsum(eps, axis=0)
    assert W.shape == (nt + 1, Nsim)

    S = St * np.exp( (r - sig ** 2 / 2) * tn.reshape(-1, 1) + sig * W )

    I = It + ht * np.sum( S[1:-1, :], axis=0 ) + 0.5 * ht * ( S[0, :] + S[-1, :] )

    return S, I / T


def QuasiMonteCarloArithmeticMean(r, sig, t, T, St, It, K, Nsim, nt):
    if not isinstance(K, np.ndarray):
        K = np.asarray([K])
    S, A = genQuasiMonteCarlo(r, sig, t, T, St, It, Nsim, nt)

    payoff = np.maximum( A.reshape(1, -1) - K.reshape(-1, 1), 0.0 )
    C = np.exp(-r * (T - t)) * np.mean(payoff, axis=-1)

    return C

def QuasiMonteCarloGeometricMean(r, sig, t, T, St, It, K, Nsim, nt):
    if not isinstance(K, np.ndarray):
        K = np.asarray([K])
    S, _ = genQuasiMonteCarlo(r, sig, t, T, St, It, Nsim, nt)

    ht = (T - t) / nt
    X = np.log(S)
    I = It + ht * np.sum( X[1:-1, :], axis=0 ) + 0.5 * ht * ( X[0, :] + X[-1, :] )
    G = np.exp(I / T)

    payoff = np.maximum( G.reshape(1, -1) - K.reshape(-1, 1), 0.0 )
    C = np.exp(-r * (T - t)) * np.mean(payoff, axis=-1)
    return C