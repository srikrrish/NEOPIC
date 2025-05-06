import numpy as np
import matplotlib.pyplot as plt
from initialize import L, NG
import figures



def field(rho):
    rhoHat = np.fft.fft2(rho)
    phiHat, EHat = fieldInFourier(rhoHat, process=False)
    phi = np.real(np.fft.ifft2(phiHat))
    E = np.real(np.array([np.fft.ifft2(EHat[0]), np.fft.ifft2(EHat[1])]))
    return phi, E


def fieldInFourier(rhoHat, process=True):
    Ja = np.arange(rhoHat.shape[0] // 2)
    Jb = Ja[:0:-1]
    J = np.append(np.append(Ja, [-rhoHat.shape[0] // 2]), - Jb)
    Ka = np.arange(rhoHat.shape[1] // 2)
    Kb = Ka[:0:-1]
    K = np.append(np.append(Ka, [-rhoHat.shape[1] // 2]), - Kb)
    J = np.transpose(np.expand_dims(J, 0).repeat(rhoHat.shape[1], axis=0)) * 2 * np.pi / L[0]
    K = np.expand_dims(K, 0).repeat(rhoHat.shape[0], axis=0) * 2 * np.pi / L[1]
    absolute = J ** 2 + K ** 2
    absolute[0,0] = 1
    phiHat = rhoHat / absolute
    phiHat[0,0] = 0
    E0 = phiHat * -1j * J
    E1 = phiHat * -1J * K
    EHat = np.array([E0, E1])
    return phiHat, EHat
