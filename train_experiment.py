import torch
from torch.utils.data import DataLoader
from torch import nn 
from utils.datas import slice_dir 
from utils.networks import FCN_Pad_Xaiver_gain,Init_Conv
from NNs import HeatFormer_mut, HeatFormer_passive
import os 
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np 
from transformer.cct import CCT
from transformer.FCN_CCT import FCN_CCT,Skip_FCN_CCT,FullSkip_FCN_CCT, FullSkip_Mul_FCN_CCT
# torch.backends.cudnn.deterministic = True
torch.manual_seed(100)


device = ("cuda" if torch.cuda.is_available() else "cpu")
print(device)

HEIGHT = 256;WIDTH = 256; CHANNELS = 4; KNSIZE=3;padding = 8
batch_size = 1;
# VIT_model = CCT()
# fcn_model = FCN_Pad_Xaiver_gain(HEIGHT,WIDTH,CHANNELS,KNSIZE,8)
# fcn_model.apply(Init_Conv)
model = FullSkip_Mul_FCN_CCT(num_heads=16,num_layers=1)
# model = VIT_model
model_name = "y_plus_30_pr0025_Mul_CCT_2layer_16heads_1epoch_1batch"
loss_fn = nn.MSELoss()
# loss_fn = DiceBCELoss()
optimizer = torch.optim.AdamW(model.parameters(),lr=1e-2,eps=1e-7,betas=(0.9,0.999),
                              weight_decay=2e-2)
# optimizer = torch.optim.SGD(model.parameters(),lr=1e-2,momentum=0.9,weight_decay=2e-2)
var=['u_vel',"v_vel","w_vel","pr0.025"]
# var=['tau_wall',"pr0.025"]
target=['pr0.025_flux']
normalized=False
y_plus=30
save_types= ["train","test","validation"]
root_path = "/home/yuning/thesis/tensor"
train_path = slice_dir(root_path,y_plus,var,target,"train",normalized)
print(f"Data will be loaded from {train_path}")

train_dl = DataLoader(torch.load(train_path+"/train1.pt"),batch_size=batch_size, shuffle=True)

# model.apply(Init_Conv)
model.to(device)
model.train(True)

from tqdm import tqdm
num_step = 500; 
loss_hist = []
for epoch in range(1):
    train_loss = 0
    i = 0
    print(f"Epoch {epoch}",flush=True)
    # for batch in tqdm(train_dl):
    for batch in train_dl:
                i += 1
                x,y  = batch
                
                x = x.float().to(device); y = y.double().to(device)

                pred = model(x).double()
                optimizer.zero_grad()
                loss = loss_fn(pred,y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_loss = loss.item()
    
    # loss_hist.append(train_loss/i)
                loss_hist.append(train_loss)
   
    # print(f"Loss:{loss_hist[-1]}",flush=True)
                print(f"Step:{i} Loss:{loss_hist[-1]}",flush=True)
                # if i == num_step:
                    # break

torch.save(model,"/home/yuning/thesis/valid/models/23-2-20_{}.pt".format(model_name))
print("Training finished, model has been saved!")

plt.figure(0,figsize=(12,10))
plt.semilogy(loss_hist,"r",lw=2.5)
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.grid()
plt.savefig("/home/yuning/thesis/valid/fig/23-2-20/loss{}".format(model_name))


