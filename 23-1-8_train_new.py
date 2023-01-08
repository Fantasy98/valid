import torch
from torch.utils.data import DataLoader
from torch import nn 
from utils.datas import slice_dir 
from utils.networks import FCN_Pad_Xaiver
from utils.toolbox import periodic_padding
import os 
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np 


# torch.backends.cudnn.deterministic = True
torch.manual_seed(100)


device = ("cuda" if torch.cuda.is_available() else "cpu")
print(device)

HEIGHT = 256;WIDTH = 256; CHANNELS = 4; KNSIZE=3;padding = 8
batch_size = 2;

model = FCN_Pad_Xaiver(HEIGHT,WIDTH,CHANNELS,KNSIZE,padding)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,eps=1e-7,betas=(0.9,0.999))

var=['u_vel',"v_vel","w_vel","pr0.2"]
target=['pr0.2_flux']
normalized=False
y_plus=50
save_types= ["train","test","validation"]
root_path = "/home/yuning/thesis/tensor"
train_path = slice_dir(root_path,y_plus,var,target,"train",normalized)
print(f"Data will be loaded from {train_path}")

train_dl = DataLoader(torch.load(train_path+"/train1.pt"),batch_size=batch_size, shuffle=True)

model.initial()
model.to(device)
model.train(True)

num_step = 500; i = 0
loss_hist = []
for batch in tqdm(train_dl):
    i += 1
    x,y  = batch
    x = x.float().to(device); y = y.double().to(device)
    for rep in range(2):
        pred  = model(x).double()
        optimizer.zero_grad()
        loss = loss_fn(pred,y)
        loss.backward()
        optimizer.step()
    loss_hist.append(loss.item())
    if i == num_step:
        break

torch.save(model,"/home/yuning/thesis/valid/models/23-1-8{}.pt".format(num_step))
print("Training finished, model has been saved!")

plt.figure(0,figsize=(12,10))
plt.semilogy(loss_hist,"r",lw=2.5)
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.grid()
plt.savefig("/home/yuning/thesis/valid/fig/23-1-8/loss{}".format(num_step))


