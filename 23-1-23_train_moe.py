import torch 
from torch import nn 
import torch.nn.functional as F
from utils.datas import slice_dir
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.plots import Plot_2D_snapshots
from utils.metrics import Fluct_error, Glob_error,RMS_error
from torch.optim import Adam
from torch.nn import MSELoss
from scipy.stats import pearsonr
from utils.moe import Dense_Gate
from utils.networks import Init_Conv
from utils.newnets import FCN_4
import matplotlib.pyplot as plt 

device = ("cuda" if torch.cuda.is_available() else "cpu")


n_expert = 3

def get_model():
    model = torch.load("/home/yuning/thesis/valid/models/23-1-23_y_plus_30_FCN4_Partial.pt")
    model.requires_grad_(False)
    model.conv4.requires_grad_(True)
    # model.Tconv1.requires_grad_(True)
    nn.init.xavier_uniform_(model.conv4.weight)
    nn.init.zeros_(model.conv4.bias)
    return model
HEIGHT = 256;WIDTH = 256; CHANNELS = 4; KNSIZE=3;padding = 8
batch_size = 1;

# def get_model():
#     model = FCN_4(HEIGHT,WIDTH,CHANNELS,KNSIZE,padding)
#     model.apply(Init_Conv)
#     return model
experts_list = nn.ModuleList([  get_model()  for i in range(n_expert) ])
print(experts_list)

MoE = Dense_Gate(experts_list,n_expert=n_expert,device=device)

MoE.to(device)
var=['u_vel',"v_vel","w_vel","pr0.025"]
target=['pr0.025_flux']
normalized=False
y_plus=30
save_types= ["train","test","validation"]
root_path = "/home/yuning/thesis/tensor"
test_path = slice_dir(root_path,y_plus,var,target,"train",normalized)
print(test_path)

torch.manual_seed(1024)
test_dl = DataLoader(torch.load(test_path+"/train2.pt"),shuffle=True,batch_size=1)


optimizer = Adam(MoE.parameters(),lr = 1e-3,eps=1e-7)


loss_fn = MSELoss()
loss_hist = []
for batch in test_dl:
    # batch = iter(test_dl).next()
    x,y = batch
    x = x.float().to(device)
    y = y.float().to(device)
    pred_final = MoE(x)

    optimizer.zero_grad()
    loss = loss_fn(pred_final,y)
    loss.backward()
    optimizer.step()
    print(loss.item())
    loss_hist.append(loss.item())

torch.save(MoE,"/home/yuning/thesis/valid/models/23-1-24_moe_partial_init4.pt")
plt.figure(0)
plt.semilogy(loss_hist,lw = 2.5)
plt.ylabel("Loss")
plt.xlabel("Step")
plt.savefig("/home/yuning/thesis/valid/fig/23-1-24/loss_moe_partial_init4",bbox_inches="tight")

glob = Glob_error(pred_final.squeeze().detach().cpu().numpy(),y.cpu().squeeze().numpy())
rmse = RMS_error(pred_final.squeeze().detach().cpu().numpy(),y.cpu().squeeze().numpy())
fluct = Fluct_error(pred_final.squeeze().detach().cpu().numpy(),y.cpu().squeeze().numpy())
p,r = pearsonr(pred_final.squeeze().detach().cpu().numpy().flatten(),y.cpu().squeeze().numpy().flatten())
print(f"Global error {glob}")
print(f"RMS error {rmse}")
print(f"Fluct error {fluct}")
print(f"PCC {p}")

