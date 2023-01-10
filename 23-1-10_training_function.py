import torch 
import os
from torch import nn 
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from utils.networks import FCN_Pad_Xaiver_gain
from utils.datas import slice_dir
from utils.train_utils import fit, validation
from utils.toolbox import EarlyStopping, LRScheduler,Name_Checkpoint
from tqdm import tqdm
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np 
import time

###############

torch.manual_seed(1024)
device = ("cuda:2" if torch.cuda.is_available() else "cpu")
print(device)
device_ids = [2,3]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"

parser = argparse.ArgumentParser()
parser.add_argument("--lr-scheduler",dest="lr_scheduler",action="store_true")
parser.add_argument("--early-stopping",dest="early_stopping",action="store_true")
args = vars(parser.parse_args())
################
HEIGHT=256;WIDTH = 256; CHANNELS = 4; KNSIZE=3; padding = 8
model =FCN_Pad_Xaiver_gain(HEIGHT,WIDTH,CHANNELS,KNSIZE,padding)
model.initial()
if torch.cuda.device_count() > 1:
    print("There are ", torch.cuda.device_count(), "GPUs!")
    model = DataParallel(model,device_ids=device_ids)
model.to(device)

optimizer= torch.optim.Adam(model.parameters(),lr= 1e-3,eps=1e-7)
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

val_dl = DataLoader(torch.load(valid_path+"/validation.pt"),batch_size=batch_size,shuffle=True
                                                        ,num_workers=0,pin_memory=True)

###########

EPOCH = 100
train_loss_hist=[]
val_loss_hist=[]

if args["lr_scheduler"]:
        print("INFO: Initializing lr scheduler")
        lr_scheduler = LRScheduler(optimizer)
if args["early_stopping"]:
        print("INFO: Initalizing early stopping")
        early_stopping = EarlyStopping()

start = time.time()
for epoch in range(EPOCH):
    print(f"Epoch {epoch+1} of {EPOCH}")
    train_loss = fit(model,optimizer,loss_fn,train_dl,device)
    print(f"Training loss = {train_loss}")
    train_loss_hist.append(train_loss)

    val_loss = validation(model,loss_fn,val_dl,device)
    print(f"Validation loss = {val_loss}")
    val_loss_hist.append(val_loss)

    if args["lr_scheduler"]:
        lr_scheduler(val_loss)
    if args["early_stopping"]:
        early_stopping(val_loss)
        if early_stopping.early_stop:
            break

check_point = {"model":model,
                "optimizer":optimizer.state_dict(),
                "loss":train_loss_hist,
                "val_loss":val_loss_hist
                }


model_dir = Name_Checkpoint(y_plus,var,target,EPOCH)
print(f"The model will be saved as {model_dir}")
torch.save(check_point,model_dir)
print("Check point has been saved !")
