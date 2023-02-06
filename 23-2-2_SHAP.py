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
from utils.SHAP import FullSkip_FCN_CCT_Shap,FCN_Pad_Xaiver_Shap
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
# model_name = "y_plus_30_fullskipCCT_2layer_16heads_1epoch_1batch"
model_name = "y_plus_30_FCN_Pad_Base"
#
model = torch.load("/home/yuning/thesis/valid/models/23-1-17_{}.pt".format(model_name))
torch.save(model.state_dict(),"/home/yuning/thesis/valid/models/23-2-2_{}_state_dict.pt".format(model_name))
#%%

model = FCN_Pad_Xaiver_Shap(256,256,4,3,0)
model_state = torch.load("/home/yuning/thesis/valid/models/23-2-2_{}_state_dict.pt".format(model_name))
model.load_state_dict(model_state)
#%%
model.eval()
batch = iter(test_dl).next()
x,y = batch
x = x.float().view(-1,4,256,256)
pred = model(x)
print(pred.shape)
# %%

# x_np = x.detach().unsqueeze(0)

explainer = shap.DeepExplainer(model,x)

# %%
batch = iter(test_dl).next()
x,y = batch
x = x.float().view(-1,4,256,256)
shap_values = explainer.shap_values(x)

#%%
shap_values_array = np.asarray(shap_values)
# np.save("Shap_16.npy",shap_values_array)

# %%
# shap_values_array_mean = shap_values_array.mean(0)
from utils.plots import Plot_2D_snapshots
for i in range(4):
    Plot_2D_snapshots(shap_values_array[0,i,:,:],f"FCN_SHAP_{i}")

# %%
