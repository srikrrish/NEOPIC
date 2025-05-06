from initialize import Q, QM, dx, L
import numpy as np
import cupy as cp

def kinetic(vp, wp=1):
    return cp.sum(Q * wp * cp.sum(vp ** 2, axis=1) * 0.5 / QM)

def potential(rho, phi):
    #print(np.sum(rho*phi))
    return np.sum(rho * phi * dx[0] * dx[1] / 2)

def specPotential(rhoHat, phiHat):
    return np.real(np.sum(rhoHat * np.conjugate(phiHat) / (2 * L[0] * L[1])))

def energypotx(Eg):
    return np.sum(Eg[0] ** 2) * dx[0] * dx[1]
