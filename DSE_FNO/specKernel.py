import numpy as np
from scipy import special


def specKernel(NG, L, order=2):
    PL = L / NG
    Ka = np.arange(1, NG[0] // 2)
    Kb = Ka[::-1]
    K = np.append(np.append(Ka, [- NG[0] // 2]), - Kb)
    Shat0 = (L[0] * np.sin(np.pi * K * PL[0] / L[0]) / (np.pi * K * PL[0])) ** order
    Shat0 = np.append([1], Shat0).reshape(NG[0], 1)
    
    Ka = np.arange(1, NG[1] // 2)
    Kb = Ka[::-1]
    K = np.append(np.append(Ka, [- NG[1] // 2]), - Kb)
    Shat1 = (L[1] * np.sin(np.pi * K * PL[1] / L[1]) / (np.pi * K * PL[1])) ** order
    Shat1 = np.append([1], Shat1)

    return np.kron(Shat0, Shat1)

    
def circleKernel(NG, L, order=2):
    r = np.min(L / NG)
    Ja = np.arange(0, NG[0] // 2)
    Jb = Ja[:0:-1]
    J = (np.append(np.append(Ja, [- NG[0] // 2]), - Jb) * 2 * np.pi / L[0]) ** 2 * np.ones([NG[1], 1])
    Ka = np.arange(0, NG[1] // 2)
    Kb = Ka[:0:-1]
    K = (np.append(np.append(Ka, [- NG[1] // 2]), - Kb) * 2 * np.pi / L[1]) ** 2 * np.ones([NG[0], 1])
    Kabsolute = np.transpose(np.sqrt(J + np.transpose(K)))
    Kabsolute[0,0] = 1  # avoid 0 on denominator
    Shat = (2 * special.j1(r * Kabsolute) / (r * Kabsolute)) ** order
    Shat[0, 0] = 1
    return Shat