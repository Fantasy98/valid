import torch 
from transformer.FCN_CCT import FullSkip_Mul_FCN_CCT
from NNs import HeatFormer_mut
from utils.networks import FCN_Pad_Xaiver_gain, Init_Conv
from transformer.sampler import DepthToSpace, SpaceToDepth
import matplotlib.pyplot as plt
from utils.datas import slice_dir
import torch
from torch.nn import Module, Linear,Dropout,MultiheadAttention
import torch.nn.functional as F


Pr = 0.025
var=['u_vel',"v_vel","w_vel",f"pr{Pr}"]
target=[f'pr{Pr}_flux']
normalized=False
y_plus=15


save_types= ["train","test","validation"]
root_path = "/home/yuning/thesis/tensor"
test_path = slice_dir(root_path,y_plus,var,target,"test",normalized)

dl = torch.load(test_path+"/test.pt")
xd = dl.x

yd = dl.y
sampler = DepthToSpace(2)
sampler = SpaceToDepth(2)

x = xd[0:1,:,:,:]
y = yd[0:1,:,:,:]
# model =HeatFormer_mut()
model = FullSkip_Mul_FCN_CCT()
# model = FCN_Pad_Xaiver_gain(256,256,4,3,8)
# model.apply(Init_Conv)
x = model(x.float())

print(x.size())

plt.figure()
plt.imshow(x.detach().cpu().numpy().squeeze(),"jet")

plt.figure()
plt.imshow(y.detach().cpu().numpy().squeeze(),"jet")


plt.show()