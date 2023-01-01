#%%
import torch 
import os
from torch import nn 
from utils.networks import FCN
from utils.datas import slice_dir
from torch.utils.data import DataLoader

model = torch.load("/home/yuning/thesis/valid/models/model_epoch750.pt")
model.eval()

# %%
var=['u_vel',"v_vel","w_vel","pr0.025"]
target=['pr0.025_flux']
normalized=False
y_plus=30
save_types= ["train","test","validation"]
root_path = "/home/yuning/thesis/tensor"
train_path = slice_dir(root_path,y_plus,var,target,"train",normalized)
print(train_path)


uvel = DataLoader(torch.load(os.path.join(train_path,"u_vel.pt")),batch_size=1)
vvel = DataLoader(torch.load(os.path.join(train_path,"v_vel.pt")),batch_size=1)
wvel = DataLoader(torch.load(os.path.join(train_path,"w_vel.pt")),batch_size=1)
pr025 = DataLoader(torch.load(os.path.join(train_path,"pr0.025.pt")),batch_size=1)


target_path= os.path.join(train_path,target[0]+".pt")
target_tensor = torch.load(target_path)

target_set = DataLoader(target_tensor,batch_size=1,num_workers=2)

# %%
with torch.no_grad():
    model.cpu()
    
    u = uvel.dataset.tensors[0][200,:,:]
    v = vvel.dataset.tensors[0][200,:,:]
    w = wvel.dataset.tensors[0][200,:,:]
    pr = pr025.dataset.tensors[0][200,:,:]
    x=torch.stack([ u,
                    v,
                    w,
                    pr],dim=0).unsqueeze(0).float()            # print(x.size())
    y =target_set.dataset.tensors[0][200,:,:]
            # print(y.size())
    pred = model(x).cpu()
            
# %%
import matplotlib.pyplot as plt 
plt.figure(0)
clb = plt.imshow(pred.squeeze())
plt.colorbar(clb)

plt.figure(1)
clb = plt.imshow(y.squeeze())
plt.colorbar(clb)
plt.show()
# %%
import numpy as np 
rms = np.mean(  np.sqrt(  (pred.squeeze().numpy() - y.squeeze().numpy() / (y.squeeze().numpy()))**2  ) )
print(rms)
# %%
