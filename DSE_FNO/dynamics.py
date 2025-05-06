from initialize import Q, QM, L, DT, findsource, N
from energy import kinetic
import numpy as np
import cupy as cp
#import finufft

def accelerate(M, E, wp, Eout, it):
    Eout[it,:,0] = (M * E[0].flatten()) / Q
    Eout[it,:,1] = (M * E[1].flatten()) / Q
    #a1 = np.transpose(M * E[0].flatten()) * QM / wp
    #a2 = np.transpose(M * E[1].flatten()) * QM / wp
    a1 = np.reshape(Eout[it,:,0], (1,N)) * Q * QM / wp
    a2 = np.reshape(Eout[it,:,1], (1,N)) * Q * QM / wp
    #return np.array([a1, a2]), Eout
    return np.concatenate((a1, a2), axis=0), Eout

def accelerateML(E, wp):
    #a1 =  E[0,:] * QM / wp
    #a2 =  E[1,:] * QM / wp
    #a = cp.zeros([N, 2])
    a =  E * QM / wp
    #a[:,1] =  E[:,1] * QM / wp
    return a#np.array([a1, a2])

def acceleratePicard(Ek, En):
    a =  QM * 0.5 * (Ek + En)
    return a


def accelInFourier(xp, EgHat, Shat, wp):
    coeff1 = np.conjugate(EgHat[0] * Shat)
    a1 = np.real(finufft.nufft2d2(xp[0] * 2 * np.pi / L[0], xp[1] * 2 * np.pi / L[1], coeff1, eps=1e-12, modeord=1) * QM / (L[0] * L[1] * wp))
    coeff2 = np.conjugate(EgHat[1] * Shat)
    a2 = np.real(finufft.nufft2d2(xp[0] * 2 * np.pi / L[0], xp[1] * 2 * np.pi / L[1], coeff2, eps=1e-12, modeord=1) * QM / (L[1] * L[0] * wp))

    return np.array([a1, a2])


def push(vp, a, it):
    if it == 0:
        return vp + a * DT / 2, kinetic(vp + a * DT / 2)
    else:
        return vp + a * DT, kinetic(vp + a * DT)

def pushPicard(vp, ak):
    return vp + ak * DT, kinetic(vp + ak * DT)

def movePicard(xp, vp, ak):
    return xp + DT * (vp + 0.5 * ak * DT)

def move(xp, vp, wp, it=None):
    if wp == 1:
        return xp + vp * DT, 1
    else:
        return xp + vp * DT, wp + DT * findsource(xp + vp * DT / 2, vp, L, it + 0.5, DT)


def toPeriodic(x, L, discrete=False):
    out = (x < 0)
    x[out] = x[out] + L
    if discrete:
        out = (x > L - 1)
    else:
        out = (x >= L)
    x[out] = x[out] - L
    return x

def toPeriodicND(x, L, dim=2, discrete=False):
    for i in range(dim):
        x[:,i] = toPeriodic(x[:,i], L[i], discrete)
    return x
