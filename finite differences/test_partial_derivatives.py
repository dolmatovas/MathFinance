from derivatives import *
from matplotlib import pyplot as plt

def get_error(N, der, f, fx, fy, fxx, fyy, fxy, xi, dxi=None):
    tnx = np.linspace(0, 1, N + 1)
    tny = np.linspace(0, 1, 2 * N + 1)
    xn = xi(tnx).reshape(-1, 1)
    yn = xi(tny).reshape(1, -1)
    
    taux = tnx[1] - tnx[0]
    tauy = tny[1] - tny[0]
    hx = np.diff(xn)
    hy = np.diff(yn)
    if not dxi is None:
        hx = taux * dxi( tnx[:-1] + taux / 2.0 ).reshape(-1, 1)
        hy = tauy * dxi( tny[:-1] + tauy / 2.0 ).reshape(1, -1)
    h1 = hx[:-1]
    h2 = hx[1:]
    
    d1 = hy[0, :-1]
    d2 = hy[0, 1:]
    
    
    u = f(xn, yn)
    ux = der.dx(u, h1, h2)
    uy = der.dy(u, d1, d2)
    uxx = der.d2x(u, h1, h2)
    uyy = der.d2y(u, d1, d2)
    uxy = der.dxy(u, h1, h2, d1, d2)
    
    ux_, uy_, uxx_, uyy_, uxy_ = np.zeros_like(u), \
                    np.zeros_like(u), \
                    np.zeros_like(u), \
                    np.zeros_like(u), \
                    np.zeros_like(u)
    sl = slice(1, -1, 1)
    ux_[sl, sl] = fx(xn[1:-1], yn[0, 1:-1])
    uy_[sl, sl] = fy(xn[1:-1], yn[0, 1:-1])
    
    uxx_[sl, sl] = fxx(xn[1:-1], yn[0, 1:-1])
    uyy_[sl, sl] = fyy(xn[1:-1], yn[0, 1:-1])
    uxy_[sl, sl] = fxy(xn[1:-1], yn[0, 1:-1])
    
    errx = np.linalg.norm(ux - ux_) / np.linalg.norm(ux_)

    erry = np.linalg.norm(uy - uy_) / np.linalg.norm(uy_)
    
    errxx = np.linalg.norm(uxx - uxx_) / np.linalg.norm(uxx_)
    erryy = np.linalg.norm(uyy - uyy_) / np.linalg.norm(uyy_)
    errxy = np.linalg.norm(uxy - uxy_) / np.linalg.norm(uxy_)
    
    return errx, erry, errxx, erryy, errxy

if __name__ == '__main__':
    n = 2
    xi = lambda t: t ** n
    dxi = lambda t: n * (t ** (n-1))
    f = lambda x, y: np.sin(x ** 2 + y ** 2)
    fx = lambda x, y: 2 * x * np.cos(x ** 2 + y ** 2)
    fy = lambda x, y: 2 * y * np.cos(x ** 2 + y ** 2)

    fxx = lambda x, y: 2 * np.cos(x ** 2 + y ** 2) - 4 * x ** 2 * np.sin(x ** 2 + y ** 2)
    fyy = lambda x, y: 2 * np.cos(x ** 2 + y ** 2) - 4 * y ** 2 * np.sin(x ** 2 + y ** 2)
    fxy = lambda x, y: -4 * x * y * np.sin(x ** 2 + y ** 2)

    N = 4

    errx = []
    erry = []
    errxx = []
    erryy = []
    errxy = []

    Ns = []
    der = DerFwdXY()
    for i in range(18):
        _errx, _erry, _errxx, _erryy, _errxy = get_error(N, der, f, fx, fy, fxx, fyy, fxy, xi, dxi)
        
        errx.append(_errx)
        erry.append(_erry)
        errxx.append(_errxx)
        erryy.append(_erryy)
        errxy.append(_errxy)
        
        Ns.append(N)
        
        N = int(N * np.sqrt(2))
    errx = np.asarray(errx)
    erry = np.asarray(erry)
    errxx = np.asarray(errxx)
    erryy = np.asarray(erryy)
    errxy = np.asarray(errxy)

    plot = True
    if plot:
        plt.figure()
        plt.loglog(Ns, errx, '-o', label='x')
        plt.loglog(Ns, erry, '-o', label='y')
        plt.loglog(Ns, errxx, '-o', label='xx')
        plt.loglog(Ns, erryy, '-o', label='yy')
        plt.loglog(Ns, errxy, '-o', label='xy')
        plt.grid()
        plt.legend()
        plt.show()

    for err in [errx, erry, errxx, erryy, errxy]:
        peff = 2.0 * np.log(err[:-1] / err[1:]) / np.log(2.0)
        peff = peff[3:]
        print(np.mean(peff), np.std(peff))