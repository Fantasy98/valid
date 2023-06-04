#%%
from utils.datas import slice_dir,JointDataset
import torch 
from torch.utils.data import DataLoader
import os
import numpy as np
from scipy.signal import correlate2d
import matplotlib.pyplot as plt 
from tqdm import tqdm
Prs = [0.025,0.2,0.71,1]
prs = ["0025","02","071","1"]

Features=["U","V","W","T"]
#%%
# for idx in tqdm(range(len(Prs))):
idx = 0
Pr = Prs[idx]
pr = prs[idx]
var=['u_vel',"v_vel","w_vel",f"pr{Pr}"]
target=[f'pr{Pr}_flux']
normalized=False
y_plus=30


save_types= ["train","test","validation"]
root_path = "/home/yuning/thesis/tensor"
test_path = slice_dir(root_path,y_plus,var,target,"test",normalized)
print(test_path)
test_ds = torch.load(test_path+"/test1.pt")
test_dl = DataLoader(test_ds,shuffle=False,batch_size=1)
batch = iter(test_dl).next()
x,y = batch
x = x.detach().cpu().numpy().squeeze()
y = y.detach().cpu().numpy().squeeze()
# Coors = []
#%%

# i = 1
# y_f = y - y.mean(0)

# yf1 = y_f[0,:,:]
# y1 = y[0,:,:]

# x_f = x - x.mean(0)
# xf1 = x_f[0,i,:,:]
# x1 = x[0,i,:,:]

# xx = 128; zz = 128

# Rx = np.arange(-25,26) ; Rz = np.arange(-25,26)

# R = np.empty(shape=(50,50))

# for rx in Rx:
#     for rz in Rz:
#         R[rx,rz] = np.mean(yf1[xx,zz]*xf1[xx+rx,zz+rz])/(np.std(x1)*np.std(y1))


# plt.imshow(R,cmap="RdBu")
#%%
xn = (x - np.mean(x,(0)))/ np.std(x,(0))
yn = (y - np.mean(y,(0)))/ np.std(y,(0))
# yn = (y - y.mean(0))/ y.std(0)
# i = 0

for i in tqdm(range(x.shape[0])):
    coor = correlate2d(xn[i,:,:],
                    yn[:,:],
                    mode="same",
                    boundary="symm"
                    )
    # coor_s = (coor - coor.min())/(coor.max()-coor.min())
    # coor_s= (coor - coor.mean())/coor.std()
    coor_s = coor
    coor_s = coor_s[128-5:128+5,128-5:128+5]
    xx , zz = np.mgrid[0:4740:256j, 0:2370:256j]
    xx_s, zz_s = xx[128-5:128+5,128-5:128+5],zz[128-5:128+5,128-5:128+5]
    fig, axs = plt.subplots(1,1)
    clb= axs.contourf(xx_s,zz_s, coor_s,cmap="RdBu_r",levels=250)
    axs.set_aspect("equal")
    plt.colorbar(clb,ax=axs)
plt.show()
# %%
