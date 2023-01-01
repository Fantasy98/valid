import torch
from torch.utils.data import DataLoader
from torch import nn 
from utils.datas import slice_dir 
from utils.networks import FCN_pad
from utils.toolbox import periodic_padding
import os 
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np 
device = ("cuda" if torch.cuda.is_available() else "cpu")
print(device)

HEIGHT = 256;WIDTH = 256; CHANNELS = 4; KNSIZE=3;padding = 8
batch_size = 2;

model = FCN_pad(HEIGHT,WIDTH,CHANNELS,KNSIZE,padding)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)


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



x_dl = DataLoader(feature_tensor,batch_size=batch_size,num_workers=2)
y_dl = DataLoader(target_tensor,batch_size=batch_size,num_workers=2)

num_epochs = 0 
EPOCH = 500
loss_hist = [0]*EPOCH


model.to(device)
for x,y in tqdm(zip(x_dl,y_dl)):
    for times in range(2):
        pred = model(x[0].float().to(device))
        loss = loss_fn(pred , y[0].float().to(device))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_hist[num_epochs] += loss.item()
        print(f"\n {loss.item()}")
    num_epochs +=1


    if num_epochs == EPOCH:
        print("Saving model at {}".format(num_epochs))
        torch.save(model.state_dict(),"fcnpadding{}.pt".format(num_epochs))
        break

plt.figure(0)
plt.semilogy(loss_hist,"r",lw=2)
plt.grid()
plt.xlabel("epochs")
plt.ylabel("Loss")
plt.savefig("loss")

