from solver import *
from matplotlib import pyplot as plt

T = 1.0

def gridX(Nx):
    xn = np.linspace(0, np.pi, Nx + 1)
    hx = np.diff(xn)
    return xn, hx

def gridY(Ny):
    xn = np.linspace(0, np.pi, Ny + 1)
    hx = np.diff(xn)
    return xn, hx

def gridT(Nt):
    return np.linspace(0, T, Nt + 1)

def get_error(Nx, Ny, M, solver):
    xn, _ = gridX(Nx)
    yn, _ = gridY(Ny)
    res = solver.solve(Nx, Ny, M)
    
    xgrid, ygrid = np.meshgrid(xn, yn, indexing='ij')
    analit = np.exp(-2 * T) * np.cos(xgrid) * np.cos(ygrid)
    res = solver.solve(Nx, Ny, M)
    
    
    error = np.linalg.norm(res[-1] - analit) / np.linalg.norm(analit) * 100
    return error

if __name__ == "__main__":
    init = lambda x, y: np.cos(x) * np.cos(y)

    mux = lambda x, y: 0.0
    muy = lambda x, y: 0.0 

    sigmax = lambda x, y: 1.0
    sigmay = lambda x, y: 1.0
    sigmaxy = lambda x, y: 0.0

    Xleft = Neuman(lambda x: 0.0)
    Xright = Neuman(lambda x: 0.0)
    Yleft = Neuman(lambda x: 0.0)
    Yright = Neuman(lambda x: 0.0)

    boundary = Boundary2D(Xleft, Xright, Yleft, Yright)
    problem = Problem(boundary, init, mux, muy, sigmax, sigmay, sigmaxy)
    der = DerCntrl()
    solver = ADI_MCS(problem, der, gridX, gridY, gridT)

    Nx = 400
    Ny = 400
    M = 4
    err = []
    Ms = []
    for i in range(10):
        error = get_error(Nx, Ny, M, solver)
        err.append(error)
        Ms.append(M)
        M = int(M * np.sqrt(2))

    err = np.asarray(err)
    Ms = np.asarray(Ms)
    p = 2 * np.log( err[:-1] / err[1:] ) / np.log(2)
    print(*p)
    plot = True
    if plot:
        plt.loglog(Ms, err, '-ok')
        plt.grid()
        plt.show()