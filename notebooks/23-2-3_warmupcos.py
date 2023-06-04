import torch 
from torch import nn 
from torch.utils.data import DataLoader
from utils.datas import slice_dir 
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from transformer.FCN_CCT import FullSkip_FCN_CCT
from tqdm import tqdm
device = ("cuda" if torch.cuda.is_available() else "cpu")
#################################
batch_size = 1
var=['u_vel',"v_vel","w_vel","pr0.025"]
target=['pr0.025_flux']
normalized=False
y_plus=30
save_types= ["train","test","validation"]
root_path = "/home/yuning/thesis/tensor"
train_path = slice_dir(root_path,y_plus,var,target,"train",normalized)
print(f"Data will be loaded from {train_path}")

train_dl = DataLoader(torch.load(train_path+"/train1.pt"),batch_size=batch_size, shuffle=True)

#####################################
model = FullSkip_FCN_CCT(num_heads=16,num_layers=2)
model_name = "y_plus_30_fullskipCCT_2layer_16heads_2epoch_1batch_warmupcos"
loss_fn = nn.MSELoss()
# Investigate consine lr warm up scheduler
optimizer = AdamW(model.parameters(),lr=1e-5,eps=1e-7,weight_decay=2e-2)
scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                          first_cycle_steps =len(train_dl),
                                          warmup_steps=len(train_dl)//2,
                                          cycle_mult = 1,
                                          max_lr = 1e-3,
                                          min_lr =5e-4,
                                            gamma = 1,
                                            )

model.to(device)
model.train(True)



LR_history=[]
loss_hist = []
for epoch in range(3):
    train_loss = 0
    i = 0
    print(f"Epoch {epoch}",flush=True)
    # for batch in tqdm(train_dl):
    for batch in train_dl:
                i += 1
                # x,y  = batch
                
                # x = x.float().to(device); y = y.double().to(device)

                # pred = model(x).double()
                # optimizer.zero_grad()
                # loss = loss_fn(pred,y)
                # loss.backward()
                # optimizer.step()
                if epoch+1 != 3:
                  scheduler.step()
                # train_loss += loss.item()
                # train_loss = loss.item()
    
    # loss_hist.append(train_loss/i)
                loss_hist.append(train_loss)
                LR_history.append(scheduler.get_lr())
    # print(f"Loss:{loss_hist[-1]}",flush=True)
                print(f"Step:{i} Loss:{loss_hist[-1]}",flush=True)
                # if i == num_step:
                    # break
print("Check the state of optimizer:")
for param_group in optimizer.param_groups:
    
      print(param_group['lr'])
# torch.save(model,"/home/yuning/thesis/valid/models/23-2-16_{}.pt".format(model_name))
# print("Training finished, model has been saved!")

# plt.figure(0,figsize=(12,10))
# plt.semilogy(loss_hist,"r",lw=2.5)
# plt.xlabel("Steps")
# plt.ylabel("Loss")
# plt.grid()
# plt.savefig("/home/yuning/thesis/valid/fig/23-2-16/loss{}".format(model_name))




plt.figure(1)
plt.semilogy(LR_history,lw=1.5)
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.grid()
plt.savefig("/home/yuning/thesis/valid/fig/23-2-16/Lr_scheduler{}".format(model_name),bbox_inches ='tight')

