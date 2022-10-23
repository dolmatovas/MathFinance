class Boundary:
    def __init__(self, alpha, beta, rhs):
        self._alpha = alpha
        self._beta = beta
        self._rhs = rhs
    
    def getRhs(self, x, t):
        return self._rhs(x, t)

class Dirichle(Boundary):
    def __init__(self, rhs):
        super().__init__(0.0, 1.0, rhs)


class Neuman(Boundary):
    def __init__(self, rhs):
        super().__init__(1.0, 0.0, rhs)


def getCoefsLeft(border, h1, h2):
    alpha = border._alpha
    beta = border._beta
    x = alpha * (-2.0 * h1 - h2 ) / (h1 * (h1 + h2)) + beta
    y = alpha * (h1 + h2) / (h1 * h2)
    z = alpha * (-h1 / ((h1 + h2) * h2) )
    return x, y, z


def getCoefsRight(border, hm1, hm2):
    alpha = border._alpha
    beta = border._beta
    x = alpha * (2.0 * hm1 + hm2 ) / (hm1 * (hm1 + hm2)) + beta
    y = alpha * (-hm1 - hm2) / (hm1 * hm2)
    z = alpha * hm1 / ((hm1 + hm2) * hm2)
    return x, y, z


class Boundary2D:
    def __init__(self, bxleft : Boundary, bxright : Boundary, byleft:  Boundary, byright : Boundary):
        self._bxleft = bxleft
        self._bxright = bxright
        self._byleft = byleft
        self._byright = byright
    
    def getCoefsX(self, h1, h2, hm2, hm1):
        '''
        alpha_l u'_0 + beta_l u_0 = xl u_0 + yl u_1 + zl u_2
        alpha_r u'_N + beta_r u_N = xr u_{N} + yr u_{N - 1} + zr u_{N - 2} 
        '''
        return getCoefsLeft(self._bxleft, h1, h2) + getCoefsRight(self._bxright, hm1, hm2)
    
    def getCoefsY(self, d1, d2, dm2, dm1):
        return getCoefsLeft(self._byleft, d1, d2) + getCoefsRight(self._byright, dm1, dm2)
        