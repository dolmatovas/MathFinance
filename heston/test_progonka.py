import numpy as np
from SLAE_solvers import *

if __name__ == '__main__':
    N = 1000
    
    a = 2 * np.ones((N, ))
    b = np.random.rand(N)
    c = np.random.rand(N)

    F = np.random.rand(N)

    mtr = np.diag(a) + np.diag(c[1:], -1) + np.diag(b[:-1], 1)
    x = np.linalg.solve(mtr, F)
    X = Progonka(a, b, c, F)
    err = np.linalg.norm(X - x) / np.linalg.norm(X)
    print(err)
    assert err < 1e-5