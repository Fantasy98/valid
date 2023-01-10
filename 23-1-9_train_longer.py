import torch 
import os
from torch import nn 
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from utils.networks import FCN_Pad_Xaiver_gain
from utils.datas import slice_dir
from tqdm import tqdm
from datetime import datetime
torch.manual_seed(100024)
device = ("cuda:2" if torch.cuda.is_available() else "cpu")
print(device)
device_ids = [2,3]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"

################
HEIGHT=256;WIDTH = 256; CHANNELS = 4; KNSIZE=3; padding = 8
model =FCN_Pad_Xaiver_gain(HEIGHT,WIDTH,CHANNELS,KNSIZE,padding)
model.initial()
if torch.cuda.device_count() > 1:
    print("There are ", torch.cuda.device_count(), "GPUs!")
    model = DataParallel(model,device_ids=device_ids)
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

batch_size = 6

train_dl = DataLoader(torch.load(train_path+"/train.pt"),batch_size=batch_size,shuffle=True
                                                        ,num_workers=0,pin_memory=True)

valid_dl = DataLoader(torch.load(valid_path+"/validation.pt"),batch_size=batch_size,shuffle=True
                                                        ,num_workers=0,pin_memory=True)

###########


model.train(True)
i = 0
loss_hist = []
valid_hist =[]
EPOCH = 50
for epoch in tqdm(range(EPOCH)):
    model.train()
    train_loss = 0
    val_loss = 0
    for batch in train_dl:
        # batch = iter(train_dl).next()
        i += 1
        x,y = batch
        x = x.float().to(device); y = y.double().to(device)
    
        pred = model(x).double()
        loss = loss_fn(pred,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    loss_hist.append(train_loss/len(train_dl))
    
    with torch.no_grad():
        model.eval()
        for valid_batch in valid_dl:
            
            x_val,y_val = valid_batch
            pred_val = model(x_val.float().to(device)).float()
            loss_val = loss_fn(pred_val,y_val.float().to(device))
            val_loss += loss_val.item()
    valid_hist.append(val_loss/len(valid_dl))
        # print(f"Train loss = {loss.item()}")
        # print(f"Validation loss = {loss_val.item()}")
        # if i == num_step:
        #         break

print("Train for 1st dataset ended, saving model")
torch.save(model,"models/23-1-10/epoch{}.pt".format(EPOCH))

model.train(False)
model.eval()

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
plt.savefig("/storage3/yuning/thesis/fig/23-1-10/loss_{}".format(EPOCH))

prediction  = pre.cpu().squeeze()
target = y.detach().float().cpu().squeeze()
from utils.plots import Plot_2D_snapshots
Plot_2D_snapshots(prediction[0,:,:],"/storage3/yuning/thesis/fig/23-1-10/pred_{}".format(EPOCH))
Plot_2D_snapshots(target[0,:,:],"/storage3/yuning/thesis/fig/23-1-10/test_{}".format(EPOCH))

from utils.metrics import RMS_error
error = RMS_error(prediction[0,:,:].numpy(),target[0,:,:].numpy())
print(f"The loss is {error}")

print("Saving checkpoints")
torch.save({"model":model.state_dict(),
            "optimizer":optimizer.state_dict(),
            "train_loss" : loss_hist,
            "val_loss":valid_hist
            },
            "models/23-1-10/epoch{}_checkpoints.pt".format(EPOCH))
print("Check point saved")