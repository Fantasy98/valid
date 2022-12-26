import os
import torch 
from torch import nn 
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
from utils.networks import FCN
from utils.datas import parse_name,slice_dir
device = ("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = FCN(256,256,4,3)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)


var=['u_vel',"v_vel","w_vel","pr0.025"]
target=['pr0.025_flux']
normalized=False
y_plus=30
save_types= ["train","test","validation"]
root_path = "/storage3/yuning/thesis/tensor/"
train_path = slice_dir(root_path,y_plus,var,target,"test",normalized)
print(train_path)


uvel = DataLoader(torch.load(os.path.join(train_path,"u_vel.pt")),batch_size=1)
vvel = DataLoader(torch.load(os.path.join(train_path,"v_vel.pt")),batch_size=1)
wvel = DataLoader(torch.load(os.path.join(train_path,"w_vel.pt")),batch_size=1)
pr025 = DataLoader(torch.load(os.path.join(train_path,"pr0.025.pt")),batch_size=1)


target_path= os.path.join(train_path,target[0]+".pt")
target_tensor = torch.load(target_path)

target_set = DataLoader(target_tensor,batch_size=1,num_workers=2)

model.to(device)
from tqdm import tqdm
num_epochs = 2
for epoch in tqdm(range(num_epochs)):
    for u,v,w,pr,y in tqdm(zip(uvel,vvel,wvel,pr025,target_set)):
        
        x=torch.stack([u[0].to(device),
                    v[0].to(device),
                    w[0].to(device),
                    pr[0].to(device)],dim=1).float()
        print(x.size())
        y = torch.unsqueeze(y[0],dim=1).float().to(device);
        print(y.size())
        pred = model(x)
        loss =loss_fn(pred,y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(loss.item())
       
with torch.no_grad():
    pred = model(x).detach().cpu().squeeze()
print(pred.size())
plt.imshow(pred,"jet")
plt.savefig("test")