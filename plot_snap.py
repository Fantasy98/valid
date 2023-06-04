#%%
import torch 
import os
from torch import nn 
from utils.networks import FCN
from utils.datas import slice_dir
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import cmocean.cm as cmo
cmp = sns.color_palette('cmo.curl', as_cmap=True)
plt.set_cmap(cmp)
# from utils.plots import Plot_2D_snapshots,Plot_multi

plt.rc("font",family = "serif")
plt.rc("font",size = 22)
plt.rc("axes",labelsize = 16, linewidth = 2)
plt.rc("legend",fontsize= 12, handletextpad = 0.3)
plt.rc("xtick",labelsize = 16)
plt.rc("ytick",labelsize = 16)
#%%
var=['u_vel',"v_vel","w_vel","pr0.025"]
target=['pr0.025_flux']
normalized=False
y_plus=30
save_types= ["train","test","validation"]
root_path = "/home/yuning/thesis/tensor"
test_path = slice_dir(root_path,y_plus,var,target,"test",normalized)
print(test_path)

torch.manual_seed(100)
test_dl = DataLoader(torch.load(test_path+"/test1.pt"),shuffle=True,batch_size=1)

# %%
xt,yt= iter(test_dl).next()
x  = xt.detach().cpu().numpy().squeeze()
y  = yt.detach().cpu().numpy().squeeze()
# %%
Re_Tau = 395 #Direct from simulation
Re = 10400 #Direct from simulation
nu = 1/Re #Kinematic viscosity
u_tau = Re_Tau*nu

# xx, yy = np.mgrid[0:256:256j, 0:256:256j]
xx, yy = np.mgrid[-6:6:256j, -3:3:256j]

x_range=12
z_range=6

gridpoints_x=int(255)+1
gridponts_z=int(255)+1


x_plus_max=x_range*u_tau/nu
z_plus_max=z_range*u_tau/nu


x_plus_max=np.round(x_plus_max).astype(int)
z_plus_max=np.round(z_plus_max).astype(int)

axis_range_x=np.array([0,950,1900,2850,3980,4740])
axis_range_z=np.array([0,470,950,1420,1900,2370])


placement_x=axis_range_x*nu/u_tau
placement_x=np.round((placement_x-0)/(x_range-0)*(gridpoints_x-0)).astype(int)

placement_z=axis_range_z*nu/u_tau
placement_z=np.round((placement_z-0)/(z_range-0)*(gridponts_z-0)).astype(int)

# %%
names = [r"$u$",r"$v$",r"$w$",r"$\theta$"]
# fig, axs = plt.subplots(2,3)
for i in range(1):
    fig, ax= plt.subplots(1,1)
    cl = ax.contourf(xx,yy,x[i,:,:],levels = 50 )
    # plt.colorbar(cl)
    ax.set_xlabel(r"${x/h}$")
    ax.set_ylabel(r"${z/h}$")
    ax.set_aspect("equal")
    
    # plt.savefig("/home/yuning/thesis/valid/fig/23-5-10/"+names[i],bbox_inches="tight")
# %%

fig, ax= plt.subplots(1,1)
ax.contourf(xx,yy,y[:,:], levels=100, cmap="hot")
ax.set_aspect("equal")
ax.axis("off")
plt.savefig("/home/yuning/thesis/valid/fig/23-5-10/target",bbox_inches="tight")
# %%
p = 8 
u = x[0,:,:]
M1 = np.concatenate([ u[:,-p:],u,u[:,:p]],axis=-1)
M1 = np.concatenate([ M1[-p:,:],M1,M1[:p,:]],axis=0)
print(M1.shape)

def periodic_padding(input):
    M1 = np.concatenate([ input[:,-p:],input,input[:,:p]],axis=-1)
    M1 = np.concatenate([ M1[-p:,:],M1,M1[:p,:]],axis=0)
    return M1


x_p = periodic_padding(xx)
y_p = periodic_padding(yy)
u_p = periodic_padding(u)
# %%
xp, yp = np.mgrid[-6:6:272j, -3:3:272j]

x1 = xp[8,0]
x2 = xp[-8,0]

y1 = yp[0,8]
y2 = yp[0,-8]
xb= np.array([x1,x2,x2,x1,x1 ])
yb= np.array([y1,y1,y2,y2,y1 ])

fig, ax = plt.subplots(1,1)
ax.contourf(xp,yp,u_p,levels = 100)
ax.plot(xb,yb,"--r",lw = 2.5,zorder = 5)
ax.set_aspect('equal')
ax.set_xlabel(r"${x/h}$")
ax.set_ylabel(r"${z/h}$")
plt.tight_layout()
plt.savefig("periodic_padding.jpg",bbox_inches = "tight",dpi = 300)
# from utils.toolbox import periodic_padding

# yt_p = periodic_padding(yt,8)
# %%
