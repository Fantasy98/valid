import torch
from torch import nn 
import matplotlib.pylab as plt 
from utils.datas import slice_dir
from torch.utils.data import DataLoader
from utils.Vision_Transformer import ViT, ViTBackbone
from utils.toolbox import periodic_padding
import matplotlib.pyplot as plt
var=['u_vel',"v_vel","w_vel","pr0.025"]
# var=['tau_wall',"pr0.025"]
target=['pr0.025_flux']
normalized=False
y_plus=30
save_types= ["train","test","validation"]
root_path = "/home/yuning/thesis/tensor"
test_path = slice_dir(root_path,y_plus,var,target,"test",normalized)
print(test_path)

torch.manual_seed(0)
test_dl = DataLoader(torch.load(test_path+"/test2.pt"),shuffle=True,batch_size=1)







batch = iter(test_dl).next()
x,y = batch
# x = periodic_padding(x,padding=8)
# print(x.shape)


H = 256
W = H 
CH = 4 
PS = 16
# x = torch.randn(size=(1,CH,H,W))

model = ViTBackbone(patch_size=PS,num_layers=1,h_dim=PS*PS,num_heads=8,n_channel= CH,max_seq_length=4096)
# model = ViT(patch_size=PS,num_layers=1,h_dim=PS*PS,num_heads=8,n_channel= CH,max_seq_length=4096)

out = model(x.float())
# out  = out.reshape(1,3,256,256)
print(out.shape)

for i in range(1):
    plt.figure(i)
    plt.imshow(out.reshape(1,256,256).squeeze().detach().numpy(),"jet")
    plt.show()
# save_dir = "/home/yuning/thesis/valid/fig/23-1-31/"

# for i in range(1):
    
#     plt.figure(i)
#     clb = plt.imshow(x[0,i,:,:],"jet")
#     plt.colorbar(clb)
#     plt.savefig(save_dir+f"padded feature{i}",bbox_inches="tight")
    
#     plt.figure(i+4)
#     clb = plt.imshow(out_sq,cmap="jet")
#     plt.colorbar(clb)
#     plt.savefig(save_dir+f"Seq padded feature{i}",bbox_inches="tight")

# plt.show()
