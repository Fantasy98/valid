import torch 
import os
from torch import nn 
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from utils.networks import FCN_pad
from utils.datas import slice_dir
from tqdm import tqdm
from datetime import datetime
torch.manual_seed(1024)
device = ("cuda:3" if torch.cuda.is_available() else "cpu")
print(device)
device_ids = [2,3]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"

################
HEIGHT=256;WIDTH = 256; CHANNELS = 4; KNSIZE=3; padding = 8
model =FCN_pad(HEIGHT,WIDTH,CHANNELS,KNSIZE,padding)
model.eval()
# if torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     model = DataParallel(model,device_ids=device_ids)
model.to(device)
optimizer= torch.optim.Adam(model.parameters(),lr= 1e-3,eps=1e-7)
# optimizer = DataParallel(optim,device_ids)

# The motivation: in keras it use sum over batch, but for pytorch the default is "mean"
loss_fn = nn.MSELoss(reduction="mean")

################
var=['u_vel',"v_vel","w_vel","pr0.025"]
target=['pr0.025_flux']
normalized=False
y_plus=30
save_types= ["train","test","validation"]
root_path = "/storage3/yuning/thesis/tensor/"

train_path = slice_dir(root_path,y_plus,var,target,"train",normalized)
valid_path = slice_dir(root_path,y_plus,var,target,"validation",normalized)

print(train_path)
print(valid_path)



train_dl = DataLoader(torch.load(train_path+"/train1.pt"),batch_size=2,shuffle=True
                                                        ,num_workers=0,pin_memory=True)
valid_dl = DataLoader(torch.load(valid_path+"/validation1.pt"),batch_size=2,shuffle=True
                                                        ,num_workers=0,pin_memory=True)

###########

model.train(True)
num_step = 53
i = 0
loss_hist = []
valid_hist =[]
EPOCH = 500
for epoch in tqdm(range(EPOCH)):
    # for batch in train_dl:
        batch = iter(train_dl).next()
        i += 1
        x,y = batch
        x = x.float().to(device); y = y.float().to(device)
        for ep in range(2):
            pred = model(x).float()
            loss = loss_fn(pred,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_hist.append(loss.item())
        with torch.no_grad():
            valid_batch = iter(valid_dl).next()
            x_val,y_val = valid_batch
            pred_val = model(x_val.float().to(device)).float()
            loss_val = loss_fn(pred_val,y_val.float().to(device))
            valid_hist.append(loss_val.item())
        # print(f"Train loss = {loss.item()}")
        # print(f"Validation loss = {loss_val.item()}")
        # if i == num_step:
        #         break

print("Train for 1st dataset ended, saving model")
torch.save(model,"models/23-1-3/epoch{}_{}.pt".format(EPOCH,datetime.now()))
print("Loading new dataloader for part 2")

del train_dl,valid_dl

train_dl = DataLoader(torch.load(train_path+"/train2.pt"),batch_size=2,shuffle=True
                                                        ,num_workers=0,pin_memory=True)
valid_dl = DataLoader(torch.load(valid_path+"/validation2.pt"),batch_size=2,shuffle=True
                                                        ,num_workers=0,pin_memory=True)

EPOCH = 500
for epoch in tqdm(range(EPOCH)):
    # for batch in train_dl:
        batch = iter(train_dl).next()
        i += 1
        x,y = batch
        x = x.float().to(device); y = y.float().to(device)
        for ep in range(2):
            pred = model(x).float()
            loss = loss_fn(pred,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_hist.append(loss.item())
        with torch.no_grad():
            valid_batch = iter(valid_dl).next()
            x_val,y_val = valid_batch
            pred_val = model(x_val.float().to(device)).float()
            loss_val = loss_fn(pred_val,y_val.float().to(device))
            valid_hist.append(loss_val.item())
        # print(f"Train loss = {loss.item()}")
        # print(f"Validation loss = {loss_val.item()}")
        # if i == num_step:
        #         break

torch.save(model,"models/23-1-3/epoch{}_{}.pt".format(EPOCH,datetime.now()))

# model.cpu()
model.train(False)


with torch.no_grad():
    # pre = model(x.detach().cpu().float())
    pre = model(x.detach())
    

import matplotlib.pyplot as plt
plt.figure(0,figsize=(10,8))
plt.semilogy(loss_hist,"r",lw=2.5,label="Train Loss")
plt.semilogy(valid_hist,"b",lw=2.5,label="Val_Loss")
plt.xlabel("epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.savefig("/storage3/yuning/thesis/fig/23-1-3/loss_{}".format(EPOCH*2))

prediction  = pre.cpu().squeeze()[1,:,:]
target = y.detach().float().cpu().squeeze()[1,:,:]
from utils.plots import Plot_2D_snapshots
Plot_2D_snapshots(prediction,"/storage3/yuning/thesis/fig/23-1-3/pred_{}".format(EPOCH*2))
Plot_2D_snapshots(target,"/storage3/yuning/thesis/fig/23-1-3/test_{}".format(EPOCH*2))

from utils.metrics import RMS_error
error = RMS_error(prediction.numpy(),target.numpy())
print(f"The loss is {error}")


torch.save(model,"models/23-1-2/epoch{}_{}.pt".format(EPOCH*2,datetime.now()))