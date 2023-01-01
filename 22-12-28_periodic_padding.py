# Test of periodic padding
#%%
import torch 
from torch import nn 
from torch.nn import functional as F
import numpy as np 
#%%
# Circular padding in torch
# suppose an tensor has batch = 1, channel = 4, height= width=3
# a = torch.randint(1,9,(1,4,3,3))
# print(a)
# print(a.size())
# pd = (2,2)
# result = F.pad(a,pd,mode="circular")
# print(result)
# print(result.size())
#%% 
# Periodic padding in torch self written 
# What be mentioned in  Ander's code 

# from keras.utils.conv_utils import normalize_tuple
# nt = normalize_tuple(value=1,n=3,name="padding")
# print(nt)
a = torch.randint(1,9,(1,2,3,3))
print(a)
padding = 1
M1 = torch.cat([a[:,:,:,-padding:],a,a[:,:,:,0:padding]],dim=-1)
M1 = torch.cat([M1[:,:,-padding:,:],M1,M1[:,:,0:padding,:]],dim=-2)
print(M1.size())

M2 = M1[:,:,padding:-padding,padding:-padding]
print(M2.size())
# %%
import matplotlib.pyplot as plt 
plt.figure(0)
plt.imshow(a.squeeze().permute(2,1,0)[:,:,0],cmap="rainbow")
plt.figure(1)
plt.imshow(M1.squeeze().permute(2,1,0)[:,:,0],cmap="rainbow")
plt.figure(2)
plt.imshow(M2.squeeze().permute(2,1,0)[:,:,0],cmap="rainbow")


plt.show()
# %%
def periodic_padding(input:torch.Tensor,padding:int):
    if len(input.size()) !=4:
        print("The tenor does not fit the size!")
        return 
    else:
        M1 = torch.cat([input[:,:,:,-padding:],input,input[:,:,:,0:padding]],dim=-1)
        M1 = torch.cat([M1[:,:,-padding:,:],M1,M1[:,:,0:padding,:]],dim=-2)
        return M1

a = torch.randint(1,5,(1,1,3,3))
pds = periodic_padding(a,8)

plt.figure(2)
plt.imshow(a.permute(3,2,1,0).squeeze(),"jet")

plt.figure(3)
plt.imshow(pds.permute(3,2,1,0).squeeze(),"jet")


# %%
from utils.toolbox import periodic_padding
x = torch.randint(1,9,(1,4,3,3))
xpad = periodic_padding(x,2)

# %%
import matplotlib.pyplot as plt 
plt.imshow(x.squeeze()[0,:,:])

# %%
plt.imshow(xpad.squeeze()[0,:,:])
# %%
import matplotlib.pyplot as plt 
plt.imshow(x.squeeze()[1,:,:])

# %%
plt.imshow(xpad.squeeze()[1,:,:])

# %%
import matplotlib.pyplot as plt 
plt.imshow(x.squeeze()[2,:,:])

# %%
plt.imshow(xpad.squeeze()[2,:,:])

# %%
