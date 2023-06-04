import torch 
from torch.utils.data import DataLoader
from utils.datas import slice_dir,JointDataset


batch_size = 1
var=['u_vel',"v_vel","w_vel","pr0.025"]
target=['pr0.025_flux']
normalized=False
y_plus=30
save_types= ["train","test","validation"]
root_path = "/home/yuning/thesis/tensor"
test_path1 = slice_dir(root_path,y_plus,var,target,"test",normalized)


var=['u_vel',"v_vel","w_vel","pr0.2"]
target=['pr0.2_flux']
y_plus=50
test_path2 = slice_dir(root_path,y_plus,var,target,"test",normalized)
print(f"Data will be loaded from \n{test_path1}\n{test_path2}")

d1= torch.load(test_path1+"/test1.pt")
d2= torch.load(test_path2+"/test1.pt")

x1 = d1.x
y1 = d1.y
x2 =d2.x
y2 =d2.y

x3 = torch.cat([x1,x2])
y3 = torch.cat([y1,y2])

print(x3.size())
print(y3.size())

d3 = JointDataset(x3,y3)
print(len(d3))
torch.save(d3,"test_all.pt")