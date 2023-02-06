#%%
import torch
from torch.utils.data import DataLoader
from torch import nn 
from utils.datas import slice_dir 
from utils.networks import FCN_pad
from utils.toolbox import periodic_padding
import os 
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np 
device = ("cuda" if torch.cuda.is_available() else "cpu")
print(device)

HEIGHT = 256;WIDTH = 256; CHANNELS = 4; KNSIZE=3;padding = 8
batch_size = 1;

model = FCN_pad(HEIGHT,WIDTH,CHANNELS,KNSIZE,padding)
model.load_state_dict(torch.load("fcnpadding500.pt"))
#%%

var=['u_vel',"v_vel","w_vel","pr0.025"]
target=['pr0.025_flux']
normalized=False
y_plus=30
save_types= ["train","test","validation"]
root_path = "/home/yuning/thesis/tensor"
train_path = slice_dir(root_path,y_plus,var,target,"test",normalized)
print(f"Data will be loaded from {train_path}")

feature_tensor = torch.load(train_path+"/features1.pt")
target_tensor = torch.load(train_path+"/targets1.pt")

print(f" Feature data loaded, shape = {feature_tensor.tensors[0].size()}")
print(f" Target data loaded, shape = {target_tensor.tensors[0].size()}")



x_dl = DataLoader(feature_tensor,batch_size=batch_size,num_workers=2)
y_dl = DataLoader(target_tensor,batch_size=batch_size,num_workers=2)

# %%
sample = iter(x_dl).next()

model.cpu()
with torch.no_grad():
    y = model(sample[0].float())
# %%
from utils.plots import Plot_2D_snapshots
plt.figure(0)
Plot_2D_snapshots(y.squeeze(),"pred")
# %%
tar = iter(y_dl).next()
plt.figure(1)
Plot_2D_snapshots(tar[0].squeeze(),"tar")

# %%
from utils.metrics import RMS_error
error = RMS_error(y.squeeze().numpy(),tar[0].squeeze().numpy())
print(error)
# %%
plt.figure(2)
Plot_2D_snapshots(np.sqrt((y.squeeze()-tar[0].squeeze())**2/np.mean(tar[0].squeeze().numpy())),"error")
# %%
