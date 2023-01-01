import torch 
from torch import nn 
from torch.utils.data import DataLoader
from utils.networks import FCN_pad
from utils.datas import slice_dir
from tqdm import tqdm
torch.manual_seed(1024)
device = ("cuda:2" if torch.cuda.is_available() else "cpu")
print(device)

################
HEIGHT=256;WIDTH = 256; CHANNELS = 4; KNSIZE=3; padding = 8
model =FCN_pad(HEIGHT,WIDTH,CHANNELS,KNSIZE,padding)
model.eval()

optimizer= torch.optim.Adam(model.parameters(),lr= 1e-3,eps=1e-7)
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
print(train_path)



train_dl = DataLoader(torch.load(train_path+"/train1.pt"),batch_size=2,shuffle=True
                                                        ,num_workers=0,pin_memory=True)

sample = iter(train_dl).next()
x,y = sample
print(x.size())
print(y.size())

###########
model.to(device)
model.train(True)
num_step = 508
i = 0
loss_hist = []
EPOCH =5 
for epoch in tqdm(range(EPOCH)):
    for batch in train_dl:
        i += 1
        x,y = batch
        x = x.float().to(device); y = y.float().to(device)
    
        pred = model(x).float()
        loss = loss_fn(pred,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(loss.item())
        loss_hist.append(loss.item())
    # if i == num_step:
    #     break

model.cpu()
model.train(False)

from datetime import datetime
with torch.no_grad():
    pre = model(x.detach().cpu().float())


import matplotlib.pyplot as plt
plt.figure(0)
plt.semilogy(loss_hist,"r",lw=2.5)
plt.xlabel("epochs")
plt.ylabel("Loss")
plt.grid()
plt.savefig("/storage3/yuning/thesis/fig/loss_{}".format(num_step))

prediction  = pre.squeeze()[1,:,:]
target = y.detach().float().cpu().squeeze()[1,:,:]
from utils.plots import Plot_2D_snapshots
Plot_2D_snapshots(prediction,"/storage3/yuning/thesis/fig/pred_{}".format(num_step))
Plot_2D_snapshots(target,"/storage3/yuning/thesis/fig/test_{}".format(num_step))

from utils.metrics import RMS_error
error = RMS_error(prediction.numpy(),target.numpy())
print(f"The loss is {error}")


torch.save(model,"models/epoch{}_{}.pt".format(num_step,datetime.now()))