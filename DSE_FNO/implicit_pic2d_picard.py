import time
t = time.time()
from initialize import *
import numpy as np
import cupy as cp
xpc = np.transpose(xp)
vpc = np.transpose(vp)
xp = cp.asarray(xpc)
vp = cp.asarray(vpc)
particle_init_time = time.time()-t
import matplotlib.pyplot as plt
import energy, interpolate, field, dynamics, figures
import specKernel
import torch
picNum = 0
t1 = time.time()
device = torch.device('cuda')
model = torch.load('_Models/3layer_dt_005_T_10_5e4/Landau_fno_dse.pt', map_location=device)
model.eval()
xp = dynamics.toPeriodicND(xp, L)
inputs = torch.as_tensor(xp[None,:,:], dtype=torch.float, device='cuda')
predictions = model(inputs)
En = cp.from_dlpack(predictions.detach())
En = En.squeeze()
for it in range(NT):
    print(it)
    ##Picard iterations
    tol = 1e-6
    err_pos = 10
    err_vel = 10
    ##Initial guess from previous time step
    xk = xp
    vk = vp
    itpicard = 0
    while (err_pos > tol) or (err_vel > tol):
        xk_wop = xk
        xk = dynamics.toPeriodicND(xk, L)
        inputs = torch.as_tensor(xk[None,:,:], dtype=torch.float, device='cuda')
        predictions = model(inputs)
        Ek = cp.from_dlpack(predictions.detach())
        Ek = Ek.squeeze()
        ak = dynamics.acceleratePicard(Ek, En)
        vkp1,kinetic = dynamics.pushPicard(vp, ak)
        xkp1 = dynamics.movePicard(xp,vp,ak)
        err_pos = cp.sqrt(cp.sum((xkp1[:,0] - xk_wop[:,0])**2) + cp.sum((xkp1[:,1] - xk_wop[:,1])**2)) / cp.sqrt(cp.sum(xkp1[:,0]**2) + cp.sum(xkp1[:,1]**2))
        err_vel = cp.sqrt(cp.sum((vkp1[:,0] - vk[:,0])**2) + cp.sum((vkp1[:,1] - vk[:,1])**2)) / cp.sqrt(cp.sum(vkp1[:,0]**2) + cp.sum(vkp1[:,1]**2))
        xk = xkp1
        vk = vkp1
        itpicard = itpicard + 1
        print('Picard iteration: ',itpicard,' error pos: ',err_pos,' error vel: ',err_vel)


    xp = xkp1
    vp = vkp1
    xp = dynamics.toPeriodicND(xp, L)
    inputs = torch.as_tensor(xp[None,:,:], dtype=torch.float, device='cuda')
    predictions = model(inputs)
    En = cp.from_dlpack(predictions.detach())
    En = En.squeeze()
    Egp = cp.sum(En[:,0]**2) * (L[0] * L[1]) / N
    Exp.append(Egp.get())
    #Ek.append(kinetic)
    #Ep.append(potential)
    #E.append(kinetic + potential)
    #momx = cp.sum(Q * vp[0,:] / QM)
    #momy = cp.sum(Q * vp[1,:] / QM)
    #momentum.append(np.sqrt(momx**2 + momy**2))

figures.landauDecayFigIppl(Exp,'weak')
#figures.conservationErrors(E,momentum)
Init_time = time.time() - t1
print('Particle initialization time:',particle_init_time)
print('Intergration time:',Init_time)
print('Total time:',time.time()-t)
