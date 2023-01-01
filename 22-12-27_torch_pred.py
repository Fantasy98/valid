#%%
import torch 
import os
from torch import nn 
from utils.networks import FCN
from utils.datas import slice_dir
from torch.utils.data import DataLoader

model = torch.load("/home/yuning/thesis/valid/models/epoch508.pt")
model.eval()

# %%
var=['u_vel',"v_vel","w_vel","pr0.025"]
target=['pr0.025_flux']
normalized=False
y_plus=30
save_types= ["train","test","validation"]
root_path = "/home/yuning/thesis/tensor"
test_path = slice_dir(root_path,y_plus,var,target,"test",normalized)
print(test_path)

torch.manual_seed(1024)
test_dl = DataLoader(torch.load(test_path+"/test1.pt"),shuffle=True,batch_size=1)


# %%
import numpy as np
from utils.metrics import RMS_error 

RMS = []
model.cuda()
for batch in test_dl:
    x,y = batch
    x = x.cuda().float()
    y = y.cuda().float()
    with torch.no_grad():
        pred = model(x)
        print(pred.size())
        rms = RMS_error(pred.cpu().squeeze().numpy(),y.cpu().squeeze().numpy())
        RMS.append(rms)
RMS_np = np.array(RMS)
print(np.mean(RMS_np))
# %%
import matplotlib.pyplot as plt 
plt.figure(0)
clb = plt.imshow(pred.cpu().squeeze(),"jet")
plt.colorbar(clb)

plt.figure(1)
clb = plt.imshow(y.cpu().squeeze(),"jet")
plt.colorbar(clb)
plt.show()
# %%
import numpy as np
from utils.metrics import RMS_error 

rms = RMS_error(pred.squeeze().numpy(),y.squeeze().numpy())
print(rms)
# %%
