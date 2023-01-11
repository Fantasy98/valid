#%%
import os
import torch
from torch.utils.data import DataLoader
import numpy as np 
import matplotlib.pyplot as plt
from utils.datas import slice_dir
from scipy import stats
y_plus = 30
var = ["u_vel","v_vel","w_vel","pr0.025"]
target  = ["pr0.025_flux"]
normalized = False
root_path = "/home/yuning/thesis/tensor"
test_data_dir = slice_dir(root_path,y_plus,var,target,"test",normalized)
test_dl = DataLoader(torch.load(os.path.join(test_data_dir,"test1.pt")),
                    batch_size=1)
#%%
# pred_path = "/home/yuning/thesis/valid/pred/y_plus_50-VARS-pr0.2_u_vel_v_vel_w_vel-TARGETS-pr0.2_flux_EPOCH=100"
pred_path = "/home/yuning/thesis/valid/pred/y_plus_30-VARS-pr0.025_u_vel_v_vel_w_vel-TARGETS-pr0.025_flux_EPOCH=100"

pred = np.load(os.path.join(pred_path,"pred.npy"))
y = np.load(os.path.join(pred_path,"y.npy"))

#%%
R,P = stats.pearsonr(pred.flatten(),y.flatten())
print(R)

#%%
# PSD
Nx = 256
Nz  = 256
Lx  = 12
Lz  = 6

dx=Lx/Nx
dz=Lz/Nz

x_range=np.linspace(1,Nx,Nx)
z_range=np.linspace(1,Nz,Nz)
x=dx*x_range
z=dz*z_range

[xx,zz]=np.meshgrid(x,z)

dkx = 2*np.pi/Lx
dkz = 2*np.pi/Lz

kx = dkx * np.append(x_range[:Nx//2], -x_range[Nx//2:0:-1])
kz = dkz * np.append(z_range[:Nz//2], -z_range[Nz//2:0:-1])

[kkx,kkz]=np.meshgrid(kx,kz)

kkx_norm= np.sqrt(kkx**2)
kkz_norm = np.sqrt(kkz**2)


Re_Tau = 395 #Direct from simulation
Re = 10400 #Direct from simulation
nu = 1/Re #Kinematic viscosity
u_tau = Re_Tau*nu

# calculating wavelength in plus units 
Lambda_x = (2*np.pi/kkx_norm)*u_tau/nu
Lambda_z = (2*np.pi/kkz_norm)*u_tau/nu


Theta_fluc_targ=y-np.mean(y)
Theta_fluc_pred=pred-np.mean(pred)

fourier_image_targ = np.fft.fftn(Theta_fluc_targ)
fourier_image_pred = np.fft.fftn(Theta_fluc_pred)


fourier_amplitudes_targ = np.mean(np.abs(fourier_image_targ)**2,axis=0)
fourier_amplitudes_pred = np.mean(np.abs(fourier_image_pred)**2,axis=0)


pct10=0.1*np.max(fourier_amplitudes_targ*kkx*kkz)
pct50=0.5*np.max(fourier_amplitudes_targ*kkx*kkz)
pct90=0.9*np.max(fourier_amplitudes_targ*kkx*kkz)
pct100=np.max(fourier_amplitudes_targ*kkx*kkz)

import matplotlib.pyplot as plt
import numpy             as np
import matplotlib        as mpl

cmap = mpl.cm.Greys(np.linspace(0,1,20))
cmap = mpl.colors.ListedColormap(cmap[5:,:-1])
fig,ax=plt.subplots(1,1,dpi=1000)
CP=plt.contourf(Lambda_x,Lambda_z,fourier_amplitudes_targ*kkx*kkz,[pct10,pct50,pct90,pct100],cmap=cmap)
CS=plt.contour(Lambda_x,Lambda_z,fourier_amplitudes_pred*kkx*kkz,[pct10,pct50,pct90,pct100],colors='orange',linestyles='solid')
plt.xscale('log')
plt.yscale('log')
ax.set_ylabel(r'$\lambda_{z}^+$')
ax.set_xlabel(r'$\lambda_{x}^+$')
ax.set_title(r'$k_x\ k_z\ \phi_{q_w}$')
plt.show()
#%%
pred_path = "/home/yuning/thesis/valid/pred/y_plus_50-VARS-pr0.2_u_vel_v_vel_w_vel-TARGETS-pr0.2_flux_EPOCH=100"

pred = np.load(os.path.join(pred_path,"pred.npy"))
y = np.load(os.path.join(pred_path,"y.npy"))

R,P = stats.pearsonr(pred.flatten(),y.flatten())
print(R)

# PSD
Nx = 256
Nz  = 256
Lx  = 12
Lz  = 6

dx=Lx/Nx
dz=Lz/Nz

x_range=np.linspace(1,Nx,Nx)
z_range=np.linspace(1,Nz,Nz)
x=dx*x_range
z=dz*z_range

[xx,zz]=np.meshgrid(x,z)

dkx = 2*np.pi/Lx
dkz = 2*np.pi/Lz

kx = dkx * np.append(x_range[:Nx//2], -x_range[Nx//2:0:-1])
kz = dkz * np.append(z_range[:Nz//2], -z_range[Nz//2:0:-1])

[kkx,kkz]=np.meshgrid(kx,kz)

kkx_norm= np.sqrt(kkx**2)
kkz_norm = np.sqrt(kkz**2)


Re_Tau = 395 #Direct from simulation
Re = 10400 #Direct from simulation
nu = 1/Re #Kinematic viscosity
u_tau = Re_Tau*nu

# calculating wavelength in plus units 
Lambda_x = (2*np.pi/kkx_norm)*u_tau/nu
Lambda_z = (2*np.pi/kkz_norm)*u_tau/nu


# ym = y-np.mean(y)
# predm = pred-np.mean(pred)
Theta_fluc_targ=y-np.mean(y)
Theta_fluc_pred=pred-np.mean(pred)

fourier_image_targ = np.fft.fftn(Theta_fluc_targ)
fourier_image_pred = np.fft.fftn(Theta_fluc_pred)


fourier_amplitudes_targ = np.mean(np.abs(fourier_image_targ)**2,axis=0)
fourier_amplitudes_pred = np.mean(np.abs(fourier_image_pred)**2,axis=0)


pct10=0.1*np.max(fourier_amplitudes_targ*kkx*kkz)
pct50=0.5*np.max(fourier_amplitudes_targ*kkx*kkz)
pct90=0.9*np.max(fourier_amplitudes_targ*kkx*kkz)
pct100=np.max(fourier_amplitudes_targ*kkx*kkz)

import matplotlib.pyplot as plt
import numpy             as np
import matplotlib        as mpl

cmap = mpl.cm.Greys(np.linspace(0,1,20))
cmap = mpl.colors.ListedColormap(cmap[5:,:-1])
fig,ax=plt.subplots(1,1,dpi=1000)
CP=plt.contourf(Lambda_x,Lambda_z,fourier_amplitudes_targ*kkx*kkz,[pct10,pct50,pct90,pct100],cmap=cmap)
CS=plt.contour(Lambda_x,Lambda_z,fourier_amplitudes_pred*kkx*kkz,[pct10,pct50,pct90,pct100],colors='orange',linestyles='solid')
plt.xscale('log')
plt.yscale('log')
ax.set_ylabel(r'$\lambda_{z}^+$')
ax.set_xlabel(r'$\lambda_{x}^+$')
ax.set_title(r'$k_x\ k_z\ \phi_{q_w}$')
plt.show()
# %%
