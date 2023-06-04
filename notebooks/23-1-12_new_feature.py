#%%
import torch 
from torch.utils.data import DataLoader
import numpy as np 
from matplotlib import pyplot as plt 
from utils.datas import slice_dir
from utils.CBAM import CBAM
from scipy import stats
from utils.toolbox import periodic_padding
from torch.nn import functional as F
device = ("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 1
var=['u_vel',"v_vel","w_vel","pr0.2"]
target=['pr0.2_flux']
normalized=False
y_plus=50
save_types= ["train","test","validation"]
root_path = "/home/yuning/thesis/tensor"
test_path = slice_dir(root_path,y_plus,var,target,"test",normalized)
print(f"Data will be loaded from {test_path}")

test_dl = DataLoader(torch.load(test_path+"/test1.pt"),batch_size=batch_size, shuffle=True)



# cbam_layer = CBAM(4,1,1)

# BN1 = torch.nn.BatchNorm2d(4,eps=1e-3,momentum=0.99)
# BN2 = torch.nn.BatchNorm2d(4,eps=1e-3,momentum=0.99)
# x1 = BN2(periodic_padding(x.float(),8))
# x = F.elu(BN2(periodic_padding(x.float(),8)))
 
# cbam = cbam_layer(x.float())
# # cbam =(BN1(cbam))
# cbam1 = cbam_layer(x1)
# # cbam1 = BN1(x1)

# x1 = x1.detach().numpy().squeeze();cbam1 = cbam1.detach().numpy().squeeze()
# x = x.detach().numpy().squeeze(); y = y.numpy().squeeze();cbam= cbam.detach().numpy().squeeze()
# print(f"sample of x has shape of {x.shape}")


count = 0
PCC = np.empty(shape=(4,len(test_dl)))
for batch in test_dl:
    x,y = batch
    x = x.squeeze()
    y = y.squeeze()
    # fig,axes = plt.subplots(nrows=4,ncols=1,figsize= (8,8))
    for i in range(x.shape[0]):
        feature = x[i,:,:]
        # clb = axes[i].imshow(feature,cmap="jet")
        s,p= stats.pearsonr(feature.flatten(),y.flatten(),alternative="greater")
        PCC[i,count]= s
        print(f"Feature has correlation {p}")
count +=1
#     plt.colorbar(clb,ax=axes[i])
#     axes[i].set_xlabel("x")
#     axes[i].set_ylabel("z")
# plt.savefig("/home/yuning/thesis/valid/fig/23-1-12/uvwp_BN_Act",bbox_inches="tight")
#%%
# ig,axes = plt.subplots(nrows=4,ncols=1,figsize= (8,8))
# for i in range(x.shape[0]):
#     feature = x1[i,:,:]
#     clb = axes[i].imshow(feature,cmap="jet")
#     r,p = stats.pearsonr(feature.flatten(),y.flatten())
#     print(f"Feature has correlation {r}")
#     # plt.colorbar(clb,ax=axes[i])
#     # axes[i].set_xlabel("x")
#     # axes[i].set_ylabel("z")
# # plt.savefig("/home/yuning/thesis/valid/fig/23-1-12/uvwp_normed",bbox_inches="tight")


# fig,axes = plt.subplots(nrows=4,ncols=1,figsize= (8,8))
# for i in range(cbam.shape[0]):
#     feature = cbam[i,:,:]

#     clb = axes[i].imshow(feature,cmap="jet")
#     # r,p = stats.pearsonr(feature.flatten(),y.flatten())
#     # print(f"Feature has correlation {r}")
#     plt.colorbar(clb,ax=axes[i])
#     axes[i].set_xlabel("x")
#     axes[i].set_ylabel("z")
# plt.savefig("/home/yuning/thesis/valid/fig/23-1-12/uvwp_CBAM",bbox_inches="tight")


# fig,axes = plt.subplots(nrows=4,ncols=1,figsize= (8,8))
# for i in range(cbam.shape[0]):
#     feature = cbam1[i,:,:]

#     clb = axes[i].imshow(feature,cmap="jet")
#     # r,p = stats.pearsonr(feature.flatten(),y.flatten())
#     # print(f"Feature has correlation {r}")
#     plt.colorbar(clb,ax=axes[i])
#     axes[i].set_xlabel("x")
#     axes[i].set_ylabel("z")
# plt.savefig("/home/yuning/thesis/valid/fig/23-1-12/uvwp_CBAM_noACT",bbox_inches="tight")

