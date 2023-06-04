#%%
import torch 
import os
from torch import nn 
from utils.networks import FCN_Pad_Xaiver_gain
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
# model = torch.load("/home/yuning/thesis/valid/models/23-1-17_{}.pt".format(model_name))
# torch.save(model.state_dict(),"/home/yuning/thesis/valid/models/23-2-2_{}_state_dict.pt".format(model_name))
#%%

model = FCN_Pad_Xaiver_gain(256,256,4,3,8)
model_state = torch.load("/home/yuning/thesis/valid/models/23-2-2_{}_state_dict.pt".format(model_name))
model.load_state_dict(model_state)
#%%
model.eval()
batch = iter(test_dl).next()
x,y = batch
x = x.float().view(-1,4,256,256)
pred = model(x)
print(pred.shape)
#%%
masker_blur = np.zeros(x[0:1,1:2,:,:].shape,dtype=bool)
masker_blur[0,0,128,128] = True
#%%
pred_ = model(x.float()).detach().numpy()
def predict(mask):
    masker_blur = np.zeros(shape=[1,1,256,256],dtype=bool)
    masker_blur[0,0:1,mask[0,0]:mask[0,0]+1,mask[0,1]:mask[0,1]+1] = True
    return pred_[masker_blur].mean()

# %%
# x_np = x_.detach().unsqueeze(0)
# x1 = x.detach().numpy().reshape(-1,1)
ans = predict(np.array([[128,128]]))
#%%
# masker_blur = shap.maskers.Image("blur(1024,1024)", x1[0,:,:,:].shape)
mask= np.array([[128,128]])
explainer = shap.KernelExplainer(predict,mask)

# %%
# batch = iter(test_dl).next()
# x,y = batch
# x1 = x.detach().numpy().reshape(-1,1)
mask1 = np.array([[125,125]])
shap_values = explainer.shap_values(mask1)
#%%
shap_values1 = shap_values.values
#%%
shap_values_array = np.asarray(shap_values1.squeeze())
# shap_values_array = (shap_values_array-shap_values_array.mean())/shap_values_array.std()
# shap_values_array = np.abs(shap_values_array)
# for i in range(shap_values_array.shape[1]):
#     shap_values_array[:,i,:,:] = (shap_values_array[:,i,:,:] - shap_values_array[:,i,:,:].min())/(shap_values_array[:,i,:,:].max()-shap_values_array[:,i,:,:].min())
# np.save("Shap_16.npy",shap_values_array)
# %%
from utils.plots import Plot_Gradient_Map
Plot_Gradient_Map(shap_values_array.squeeze(),["u","v","w",r"$\theta$"],"/fig/23-2-9/shap_all_norm_abs")
# %%
