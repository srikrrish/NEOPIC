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
#pos = np.zeros([NT, N, 2]);
#Eout = np.zeros([NT, N, 2]);
#Eg = np.zeros([2,NG**2])
#xpt = np.zeros([1,N,2])
#xpt = np.zeros([1,1000000,2])
device = torch.device('cuda')
#input_x = np.load('_Data/Landau/dt_002_T_2.5/pos.npy')
#input_q = np.load('_Data/Landau/Eout.npy')
#model = torch.load('_Models/model_shuffle_dt_002/Landau_fno_dse.pt', map_location=device)
#model = torch.load('_Models/3layer_dt_005_T_10_5e4/Landau_fno_dse.pt', map_location=device)
model = torch.load('_Models/1layer_1_neuron/Landau_fno_dse.pt', map_location=device)
model.eval()
#xp=[]
#xp = input_x[0,::10,:]
#xp = np.transpose(xp)
#Ego = np.zeros([2,NG**2])
#NTE = 1200
breakpoint()
for it in range(NT):
    print(it)
    xp = dynamics.toPeriodicND(xp, L)
    #xt = input_x[it,:,:]
    #M = interpolate.interpMatrix(np.transpose(xt), wp)
    #rho = interpolate.interpolate(M)
    #print(np.abs((np.sum(rho*dx[0]*dx[1]) - (Q*N))/(Q*N)))
    #print(np.sum(rho*dx[0]*dx[1]))
    #phi, Eg = field.field(rho)
    #vp, kinetic = dynamics.accelerate(M, Eg, wp)
    #pos[it,:,:] = np.transpose(xp)
    #xpt[0,:,:] = np.transpose(xp)
    #xp = np.transpose(xp)
    #xpt[0,:,:] = input_x[it,::10,:]
    #inputs = torch.tensor(xp[None,:,:], dtype=torch.float)
    inputs = torch.as_tensor(xp[None,:,:], dtype=torch.float, device='cuda')
    #predictions = model(inputs.to(device))
    predictions = model(inputs)
    #predictions = model(input_orig[it,:,:].to(device))
    #Efieldparticle = predictions.detach().cpu().numpy()[0,:,:]
    #pred_cupy = cp.asarray(predictions)
    Efieldparticle = cp.from_dlpack(predictions.detach())
    Efieldparticle = Efieldparticle.squeeze()
    #if(it % 10 == 0):
    #    plt.plot(input_q[it,::10,0],predictions_cpu[:,0],'bo',markersize=4,label='ML prediction')
    #    plt.plot(input_q[it,::10,0], input_q[it,::10,0], label='ideal', color='seagreen')
    #    plt.ylabel('$E_x$', fontsize='14')
    #    plt.xlabel('$E_x$', fontsize='14')
    #    plt.legend()
    #    plt.grid(color='gray')
    #    plt.savefig('E_x_'+str(it)+'.png')
    #    plt.clf()
    #    plt.plot(input_q[it,::10,1],predictions_cpu[:,1],'bo',markersize=4,label='ML prediction')
    #    plt.plot(input_q[it,::10,1], input_q[it,::10,1], label='ideal', color='seagreen')
    #    plt.ylabel('$E_y$', fontsize='14')
    #    plt.xlabel('$E_y$', fontsize='14')
    #    plt.legend()
    #    plt.grid(color='gray')
    #    plt.savefig('E_y_'+str(it)+'.png')
    #    plt.clf()
    #breakpoint()
    #Efieldparticle = predictions_cpu
    #Efieldparticle = np.transpose(Efieldparticle)
    Efieldparticle[:,0] = Efieldparticle[:,0] - ((1/N) * cp.sum(Efieldparticle[:,0]))
    Efieldparticle[:,1] = Efieldparticle[:,1] - ((1/N) * cp.sum(Efieldparticle[:,1]))
    a = dynamics.accelerateML(Efieldparticle, wp)
    #Egp = np.sum(Efieldparticle[0,:]**2) * (L[0] * L[1]) / N
    Egp = cp.sum(Efieldparticle[:,0]**2) * (L[0] * L[1]) / N
    potential = cp.sum(Efieldparticle[:,0]**2 + Efieldparticle[:,1]**2) * 0.5 * (L[0] * L[1]) / N
    vp, kinetic = dynamics.push(vp, a, it)
    #xp = np.transpose(xp)
    xp, wp = dynamics.move(xp, vp, wp, it)
    Exp.append(Egp.get())
    ##########################################################################
    #Eg[0,:] = np.transpose(np.transpose(M) * np.reshape(Efieldparticle[0,:], (N,1)))
    #Eg[1,:] = np.transpose(np.transpose(M) * np.reshape(Efieldparticle[1,:], (N,1)))
    #Eg[0] = Eg[0] - (np.sum(Eg[0]) * dx[0] * dx[1] / (L[0] * L[1]))
    #Eg[1] = Eg[1] - (np.sum(Eg[1]) * dx[0] * dx[1] / (L[0] * L[1]))
    #print(np.sum(Eg[0]))
    #print(np.sum(Efieldparticle[0,:]))
    #print(np.sum(Eg[1]))
    #print(np.sum(Efieldparticle[1,:]))
    #Egp = energy.energypotx(Eg)
    #print(Egp)
    #Ego[0,:] = np.transpose(np.transpose(M) * np.reshape(input_q[it,:,0], (N,1)))
    #Ego[1,:] = np.transpose(np.transpose(M) * np.reshape(input_q[it,:,1], (N,1)))
    #Egpo = energy.energypotx(Ego)
    #print(Egpo)
    #breakpoint()
    ##potential = energy.potential(rho, phi)
    #Ek.append(kinetic)
    #Ep.append(potential)
    E.append((kinetic + potential).get())
    momx = cp.sum(Q * vp[:,0] / QM)
    momy = cp.sum(Q * vp[:,1] / QM)
    momentum.append((cp.sqrt(momx**2 + momy**2)).get())
    #momentum.append(sum(Q * vp / QM))
    ##phiMax.append(np.max(phi))
#figures.field2D(rho)

#np.save('data/pos',pos)
#np.save('data/Eout',Eout)
#dphi = np.sqrt(np.sum((phi)**2) * L[0] * L[1] / (NG[0] * NG[1]))
#dphi = np.sqrt(np.sum((phi)**2) * L[0] * L[1] / (NG * NG))
#print('error = ' + str(dphi))
#figures.landauDecayFig(Exp)
figures.landauDecayFigIppl(Exp,'weak')
#figures.twostreamIppl(Exp)
figures.conservationErrors(E,momentum)
Init_time = time.time() - t1
print('Particle initialization time:',particle_init_time)
print('Intergration time:',Init_time)
print('Total time:',time.time()-t)
#print(rho)
#figures.field2D(rho)
#plt.savefig('rho_field2.png')
#plt.show()
