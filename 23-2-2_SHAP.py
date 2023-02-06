#%%
import torch 
import os
from torch import nn 
from utils.networks import FCN
from utils.datas import slice_dir
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import shap
import numpy as np
#%%
var=['u_vel',"v_vel","w_vel","pr0.025"]
target=['pr0.025_flux']
normalized=False
y_plus=30
save_types= ["train","test","validation"]
root_path = "/home/yuning/thesis/tensor"

test_path = slice_dir(root_path,y_plus,var,target,"test",normalized)
print(test_path)

torch.manual_seed(0)
test_dl = DataLoader(torch.load(test_path+"/test2.pt"),shuffle=True,batch_size=1)

#%%
model_name = "y_plus_30_fullskipCCT_2layer_16heads_1epoch_1batch"

model = torch.load("/home/yuning/thesis/valid/models/23-2-2_{}.pt".format(model_name))

model.eval()
model.cpu()
#%%

batch = iter(test_dl).next()
x,y = batch
x = x.float().view(-1,4,256,256)
pred = model(x)

# %%

def get_model_output(z):
    
    x = torch.tensor(x)
    x = x.reshape(1,4,256,256)
    
    pred = model(x)
    pred = pred.detach().numpy()
    
    rms = np.mean(np.sqrt(pred**2))
    
    return rms

# %%

x_np = x.detach().reshape(-1).unsqueeze(0).numpy()

explainer = shap.KernelExplainer(get_model_output,x_np)
shap_values = explainer.shap_values(x_np)

# %%

out =shap_values[0]
out = out.reshape(1,4,256,256)

# %%
for i in range(4):
    plt.figure(i)
    clb = plt.imshow(out[0,i,:,:],"jet")
    plt.colorbar(clb)

plt.show()
