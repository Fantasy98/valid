#%%
import os
import torch 
from torch import nn
from torch.utils.data import DataLoader
import tensorflow as tf 
from tensorflow import keras 
from utils.datas import slice_dir
from utils.networks import FCN_Pad_Xaiver
from DataHandling.models.models import final_skip_no_sep_padding
var=['u_vel',"v_vel","w_vel","pr0.025"]
target=['pr0.025_flux']
normalized=False
y_plus=30
padding=8
activation = "elu"
#%%
model_name = "fresh-glitter-54"
all_path = "/home/yuning/thesis/models/trained/"
model_path = os.path.join(all_path,model_name)
print(model_path)
model = keras.models.load_model(model_path)
print(model.summary())

#%%
num_step = 501
torch_model = torch.load("/home/yuning/thesis/valid/models/23-1-5{}.pt".format(num_step))
print(torch_model)

# %%
root_path = "/home/yuning/thesis/tensor"
test_path = slice_dir(root_path,y_plus,var,target,"test",normalized)
print(test_path)

torch.manual_seed(0)
test_dl = DataLoader(torch.load(test_path+"/test1.pt"),shuffle=True,batch_size=1)
sample = next(iter(test_dl))
# %%
k_bn1 = model.get_layer("conv2d").weights
k = k_bn1[0].numpy().reshape(80,80)
t_bn1 = torch_model.conv1.weight
t = t_bn1.cpu().detach().numpy().reshape(80,80)
import matplotlib.pyplot as plt 
import numpy as np
plt.figure(0)
clb = plt.imshow(k,"jet")
plt.colorbar(clb)

plt.figure(1)
clb = plt.imshow(t,"jet")
plt.colorbar(clb)

plt.figure(2)
clb = plt.imshow(np.sqrt((t-k)**2),"jet")
plt.colorbar(clb)

plt.show()
# %%
