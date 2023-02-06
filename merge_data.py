#%%
import torch 
from utils.datas import JointDataset
import os
target_path = "/home/yuning/thesis/tensor"
case_path = os.path.join(target_path,"y_plus_30-VARS-pr0.025")
datatype = "train"
# var=["pr0.025","tau_wall"]
target=["pr0.025_flux"]
x_dir = case_path+"/{}_feature.pt".format(datatype)
y_dir = case_path+"/{}_{}.pt".format(target[0],datatype)

x_data = torch.load(x_dir)
y_data = torch.load(y_dir)
#%%
x_tensor = x_data.data
x_tensor = x_tensor.squeeze()
y_tensor = y_data.tensors[0]
# y_tensor=y_tensor.squeeze()
print(x_tensor.size())
print(y_tensor.size())
dataset_joint = JointDataset(x=x_tensor,y=y_tensor)
torch.save(dataset_joint,case_path+"/{}.pt".format(datatype))
# %%
