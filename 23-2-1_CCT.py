from transformer.cct import CCT
from transformer.FCN_CCT import FCN_CCT, FullSkip_FCN_CCT
import torch
import matplotlib.pyplot as plt
import torch
from torch import nn 
import matplotlib.pylab as plt 
# from utils.datas import slice_dir
from torch.utils.data import DataLoader
from utils.Vision_Transformer import ViT, ViTBackbone
from utils.toolbox import periodic_padding
import matplotlib.pyplot as plt
from utils.toolbox import DepthToSpace,SpaceToDepth
from transformer.ConvBlock import ConvBlockCat
# var=['u_vel',"v_vel","w_vel","pr0.025"]
# # var=['tau_wall',"pr0.025"]
# target=['pr0.025_flux']
# normalized=False
# y_plus=30
# save_types= ["train","test","validation"]
# root_path = "/home/yuning/thesis/tensor"
# test_path = slice_dir(root_path,y_plus,var,target,"test",normalized)
# print(test_path)

# torch.manual_seed(0)
# test_dl = DataLoader(torch.load(test_path+"/test2.pt"),shuffle=True,batch_size=1)


# # import math
# batch = iter(test_dl).next()
# x,y = batch
x = torch.randn(size=(1,4,256,256))
model =nn.ModuleList([ SpaceToDepth(block_size=2**(i+1))
                                          for i in range(0,4) ])

for i in range(4):
    print(f"The {i+1} time of Downscaling")
    # upsamp = model[i](x)
    out_up = model[i](x)
    print(out_up.shape)

model =nn.ModuleList([ConvBlockCat(in_channel= int(4*256*(0.25**(i)) ) ) for i in range(4)])


x = torch.randn(size=(1,2048,16,16))
for i in range(4):
    print(f"The {i+1} time of Upsampling")
    out_up = model[i](x)
    print(out_up.shape)
    x = torch.cat([out_up,out_up],dim=1)
    print(x.shape)


    
x = torch.randn(size=(1,4,256,256))
model =FullSkip_FCN_CCT()
out = model(x.float())
print(out.shape)
# B,P2,H = out.shape

# out_up = out_re

