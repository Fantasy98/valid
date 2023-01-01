from utils.datas import slice_dir,mkdataset,TF2Torch
import torch 
from torch.utils.data import DataLoader

var=['u_vel',"v_vel","w_vel","pr0.025"]
target=['pr0.025_flux']
normalized=False
y_plus=30
save_types= ["train","test","validation"]


for save_type in save_types:
    print(f"Saving {save_type} data!")
    root_path = "/storage3/yuning/thesis/tensor/"
    TF2Torch(root_path,y_plus,var,target,save_type,normalized=normalized)
    mkdataset(root_path,y_plus,var,target,save_type,normalized=False)


train_path = slice_dir(root_path,y_plus,var,target,"train",normalized)
print(train_path)
dl = DataLoader(torch.load(train_path+"/train1.pt"),batch_size=2,shuffle=True)

sample = iter(dl).next()
x,y = sample
print(x.size())
print(y.size())