import numpy as np


class QuasiUniformGrid:
    def __init__(self, xi, dxi):
        self._xi = xi
        self._dxi = dxi
    

    def __call__(self, N):
        tn = np.linspace(0, 1, N + 1)
        tau = 1 / N

        xn = self._xi(tn)
        hx = self._dxi(tn[:-1] + 0.5 * tau)
        return xn, hx


class UniformMesh(QuasiUniformGrid):
    def __init__(self, a, b):
        self._a = a
        self._b = b

        def xi(t):
            return a + t * (b - a)
        def dxi(t):
            return (b - a) * np.ones_like(t)
        super().__init__(xi, dxi)