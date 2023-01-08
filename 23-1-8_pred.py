#%%
import torch 
import os
from torch import nn 
from utils.networks import FCN
from utils.datas import slice_dir
from torch.utils.data import DataLoader

var=['u_vel',"v_vel","w_vel","pr0.2"]
target=['pr0.2_flux']
normalized=False
y_plus=50
save_types= ["train","test","validation"]
root_path = "/home/yuning/thesis/tensor"
test_path = slice_dir(root_path,y_plus,var,target,"test",normalized)
print(test_path)

torch.manual_seed(0)
test_dl = DataLoader(torch.load(test_path+"/test1.pt"),shuffle=True,batch_size=1)

#%%
num_step = 500
# model = torch.load("/home/yuni/ng/thesis/valid/models/model100.pt")
model = torch.load("/home/yuning/thesis/valid/models/23-1-8{}.pt".format(num_step))
model.eval()


# %%
import numpy as np
from utils.metrics import RMS_error 

batch = iter(test_dl).next()
x,y = batch
x = x.cuda().float()
y = y.cuda().double()

with torch.no_grad():
    pred = model(x).double()
# %%
import matplotlib.pyplot as plt 
from utils.plots import Plot_2D_snapshots
plt.figure(0)
clb = Plot_2D_snapshots(pred.cpu().squeeze(),"/home/yuning/thesis/valid/fig/23-1-8/pred{}".format(num_step))
plt.colorbar(clb)

plt.figure(1)
clb = Plot_2D_snapshots(y.cpu().squeeze(),"/home/yuning/thesis/valid/fig/23-1-8/tar{}".format(num_step))
plt.colorbar(clb)

plt.figure(2)
clb = Plot_2D_snapshots(pred.cpu().squeeze()-y.cpu().squeeze(),"/home/yuning/thesis/valid/fig/23-1-8/error{}".format(num_step))
plt.colorbar(clb)

plt.show()
# %%
import numpy as np
from utils.metrics import RMS_error,Glob_error,Fluct_error

rms = RMS_error(pred.cpu().squeeze().numpy(),y.cpu().squeeze().numpy())
glbrms = Glob_error(pred.cpu().squeeze().numpy(),y.cpu().squeeze().numpy())
flbrms = Fluct_error(pred.cpu().squeeze().numpy(),y.cpu().squeeze().numpy())

print(glbrms)
print(rms)

print(flbrms)

# %%
