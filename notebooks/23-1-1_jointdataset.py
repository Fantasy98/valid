import torch 
from torch.utils.data import Dataset, DataLoader
from utils.datas import JointDataset,slice_dir



var=['u_vel',"v_vel","w_vel","pr0.025"]
target=['pr0.025_flux']
normalized=False
y_plus=30
save_types= ["train","test","validation"]
root_path = "/home/yuning/thesis/tensor"
train_path = slice_dir(root_path,y_plus,var,target,"train",normalized)
print(f"Data will be loaded from {train_path}")

feature_tensor = torch.load(train_path+"/features1.pt")
target_tensor = torch.load(train_path+"/targets1.pt")

print(f" Feature data loaded, shape = {feature_tensor.tensors[0].size()}")
print(f" Target data loaded, shape = {target_tensor.tensors[0].size()}")


jdata = JointDataset(feature_tensor.tensors[0].clone(),target_tensor.tensors[0].clone())
dl = DataLoader(jdata,batch_size=2,shuffle=True)

samlpe = iter(dl).next()

print(type(samlpe))
print(len(samlpe))
print(type(samlpe[0]))
x1,y1 = samlpe
print(x1.size())
print(y1.size())

torch.save(jdata,train_path+"/trainset1.pt")
print("jointdataset has been saved")