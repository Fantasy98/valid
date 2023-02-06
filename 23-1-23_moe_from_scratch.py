
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
device = ("cuda" if torch.cuda.is_available() else "cpu")
def get_model():
    model = torch.load("/home/yuning/thesis/valid/models/23-1-23_y_plus_30_FCN4.pt")
    model.requires_grad_(False)
    model.conv1.requires_grad_(True)
    # model.Tconv1.requires_grad_(True)
    return model

def gumbel_rsample(shape):
    import torch 
    one = torch.tensor(1.0)
    zero = torch.tensor(0.0)
    gumbel = torch.distributions.gumbel.Gumbel(zero,one).rsample
    return gumbel(shape)
n_batch = 2 
n_expert = 3

experts_list = nn.ModuleList([  get_model()  for i in range(n_expert) ])
print(experts_list)

# print(experts_list[1])

from utils.CBAM import CBAM
# input = torch.rand(size=(n_batch,4,256,256))
class gate(nn.Module):
    def __init__(self) -> None:
        super(gate,self).__init__()
        self.w_gate = nn.Conv2d(in_channels=4,out_channels=n_expert,kernel_size=1)
        self.cbam = CBAM(n_expert,1,1)

        torch.nn.init.xavier_uniform_(self.w_gate.weight)
        torch.nn.init.zeros_(self.w_gate.bias)
    
    def forward(self,x):
        return self.cbam(self.w_gate(x))

w_gate = gate()
w_gate.to(device)
var=['u_vel',"v_vel","w_vel","pr0.025"]
target=['pr0.025_flux']
normalized=False
y_plus=30
save_types= ["train","test","validation"]
root_path = "/home/yuning/thesis/tensor"
test_path = slice_dir(root_path,y_plus,var,target,"test",normalized)
print(test_path)

torch.manual_seed(0)
test_dl = DataLoader(torch.load(test_path+"/test2.pt"),shuffle=True,batch_size=1)


optimizer = Adam([ {"params":experts_list.parameters()},{"params":w_gate.parameters()}    ])
# optimizer2 = Adam(experts_list.parameters())
loss_fn = MSELoss()
for batch in test_dl:
    # batch = iter(test_dl).next()
    x,y = batch
    x = x.float().to(device)
    y = y.float().to(device)
    pred_final = 0
    gate_weight = []
    pred_list = []

    ge = w_gate(x)
    print(ge.size())
    gumble_noise = gumbel_rsample(ge.size()).to(device)
    
    masks = F.gumbel_softmax( ge+gumble_noise,tau=1,dim=1)
    # print(masks.size())
    for i in range(n_expert):
        pred = experts_list[i](x)
        # pred2 = experts_list[1](x)
        pred_list.append(pred)
        out_mask = masks[:,i,:,:]
        gate_weight.append(out_mask)
        # out_mask_1 = out_mask[:,0,:,:]
        # out_mask_2 = out_mask[:,1,:,:]
        print(f"Mean prob of mask{i} {out_mask.mean()}")
        # print(f"Mean prob of mask2 {out_mask_2.mean()}")
        
    for idx,items in enumerate(gate_weight):
    #     if items.mean()>=0.5:
        pred_final += pred_list[idx]*gate_weight[idx] 
    
    # print(pred_final.size())
    # print(y.size())
    optimizer.zero_grad()
    loss = loss_fn(pred_final,y)
    loss.backward()
    optimizer.step()
    # optimizer2.step()
    print(loss.item())



Plot_2D_snapshots(out_mask.squeeze().detach().cpu().numpy(),"moe_mask{}".format(1))
glob = Glob_error(pred_final.squeeze().detach().cpu().numpy(),y.cpu().squeeze().numpy())
rmse = RMS_error(pred_final.squeeze().detach().cpu().numpy(),y.cpu().squeeze().numpy())
fluct = Fluct_error(pred_final.squeeze().detach().cpu().numpy(),y.cpu().squeeze().numpy())
p,r = pearsonr(pred_final.squeeze().detach().cpu().numpy().flatten(),y.cpu().squeeze().numpy().flatten())
print(f"Global error {glob}")
print(f"RMS error {rmse}")
print(f"Fluct error {fluct}")
print(f"PCC {p}")
Plot_2D_snapshots(pred_final.squeeze().detach().cpu().numpy(),"moe")

