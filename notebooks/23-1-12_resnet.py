
#%%
# Aim at modeling the resnet and find a new way to initialize the W&B
import torch 
from torch import nn 
from utils.networks import FCN_ResNet_NoSkip, Init_Conv
model = FCN_ResNet_NoSkip(256,256,4,3,8)

# %%
# conv1_weight = model.conv1.weight
# conv1_bias = model.conv1.bias
model.apply(Init_Conv)

# %%
x = torch.zeros(1,4,256,256)
out =model(x)
print(out.size())
# %%
