import numpy as np
from SLAE_solvers import *

def mtr_from_tri_diag(a, b, c, alpha_l, beta_l, gamma_l, alpha_r, beta_r, gamma_r):
    mtr = np.diag(a) + np.diag(c[1:], -1) + np.diag(b[:-1], 1)
    mtr[0, :3] = [alpha_l, beta_l, gamma_l]
    mtr[-1, -3:] = [gamma_r, beta_r, alpha_r]
    return mtr

def test_progonka():
    N = 1000
    
    a = 2 * np.ones((N, ))
    b = np.random.rand(N)
    c = np.random.rand(N)

    F = np.random.rand(N)

    mtr = mtr_from_tri_diag(a, b, c, a[0], b[0], 0.0, a[-1], c[-1], 0.0)
    x = np.linalg.solve(mtr, F)
    X = Progonka(a, b, c, F)
    err = np.linalg.norm(X - x) / np.linalg.norm(X)
    assert err < 1e-5


def test_progonka_coeff():
    N = 1000
    
    a = 2 * np.ones((N, ))
    b = np.random.rand(N)
    c = np.random.rand(N)

    F = np.random.rand(N)

    alpha_l, beta_l, gamma_l, alpha_r, beta_r, gamma_r = \
        (1.0, 0.01, 0.01, 1.0, 0.5, 0.01)
    
    mtr = mtr_from_tri_diag(a, b, c, alpha_l, beta_l, gamma_l, alpha_r, beta_r, gamma_r)
    x = np.linalg.solve(mtr, F)
    X = Progonka_coefs(a, b, c, alpha_l, beta_l, gamma_l, alpha_r, beta_r, gamma_r, F)
    err = np.linalg.norm(X - x) / np.linalg.norm(X)
    assert err < 1e-5

def test_n_slae(n=2):
    N = 1000
    
    a = 3 * np.ones((N, n))
    b = np.random.rand(N, n)
    c = np.random.rand(N, n)
    F = np.ones((N, n))

    x = np.zeros((N, n))
    alpha_l, beta_l, gamma_l, alpha_r, beta_r, gamma_r = \
        (1.0, 0.01, -0.01, 1.0, 0.5, 0.01)

    for i in range(n):
        x[:, i] = Progonka_coefs(
                a[:, i], 
                b[:, i], 
                c[:, i], 
                alpha_l, beta_l, gamma_l, 
                alpha_r, beta_r, gamma_r, 
                F[:, i])
    X = Progonka_coefs(a, b, c, alpha_l, beta_l, gamma_l, alpha_r, beta_r, gamma_r, F)
    err = np.linalg.norm(x - X) / np.linalg.norm(x)
    assert err < 1e-10

if __name__ == '__main__':
    test_progonka()
    test_progonka_coeff()
    test_n_slae(n=20)
