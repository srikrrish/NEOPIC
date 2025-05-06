from initialize import dx, NG, Q, N, L
import numpy as np
from dynamics import toPeriodic
from scipy import sparse
#import finufft

p = np.linspace(0, N - 1, N).astype(int)

def interpMatrix(XP, wp, DX=dx):
    # projection p->g
    g0, g1 = np.floor(XP[0] / DX[0]).astype(int), np.floor(XP[1] / DX[1]).astype(int)  # which grid point to project onto

    g = np.array([[g0 - 1, g0, g0 + 1],[g1 - 1, g1, g1 + 1]])  # used to determine bc
    a, b = XP[0] % DX[0], XP[1] % DX[1]
    c1, c2, c3, c4 = (DX[0]-a)**2, (DX[1]-b)**2, DX[0]**2 + 2 * DX[0] * a - 2 * a**2, DX[1]**2 + 2 * DX[1] * b - 2 * b**2
    tot = (DX[0] * DX[1]) ** 2
    A = c1 * c2 / (4*tot)
    B = c2 * c3 / (4*tot)
    C = a**2 * c2/ (4*tot)
    D = c1 * c4 / (4*tot)
    F = a**2 * c4 / (4*tot)
    G = b**2 * c1 / (4*tot)
    H = b**2 * c3 / (4*tot)
    I = a**2 * b**2 / (4*tot)
    E = 1 - A - B - C - D - F - G - H - I
    fraz = np.array([A, B, C, D, E, F, G, H, I] * wp)

    # apply bc on the projection
    g[0] = toPeriodic(g[0], int(L[0]/DX[0]), True)
    g[1] = toPeriodic(g[1], int(L[1]/DX[1]), True)
    matrices = []
    for i in range(3):
        for j in range(3):
            matrices.append(sparse.csr_matrix((fraz[3*i+j], (p, int(L[1]/DX[1]) * g[0,i] + g[1,j]))))
    return sum(matrices)


def interpolate(M, DX=dx):
    return (Q / (DX[0]*DX[1])) * M.sum(0).reshape([int(L[0]/DX[0]), int(L[1]/DX[1])])


def specInterpolate(XP, Shat, wp=1, ng=NG):
    rhoHat = np.conjugate(Q * Shat * finufft.nufft2d1(XP[0] * 2 * np.pi / L[0], XP[1] * 2 * np.pi / L[1], 0j + np.zeros(N) + wp, tuple(ng), eps=1e-12, modeord=1))
    return rhoHat
