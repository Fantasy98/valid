#%%
import torch 
import os
from torch import nn 
from utils.networks import FCN
from utils.datas import slice_dir
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt 
from utils.plots import Plot_2D_snapshots,Plot_multi
var=['u_vel',"v_vel","w_vel","pr0.025"]
target=['pr0.025_flux']
normalized=False
y_plus=30
save_types= ["train","test","validation"]
root_path = "/home/yuning/thesis/tensor"
test_path = slice_dir(root_path,y_plus,var,target,"test",normalized)
print(test_path)

torch.manual_seed(100)
test_dl = DataLoader(torch.load(test_path+"/test2.pt"),shuffle=True,batch_size=1)

#%%
num_step = 2
model_name = "y_plus_30_fullskipCCT_2layer_16heads_2epoch_1batch"
model = torch.load("/home/yuning/thesis/valid/models/23-2-2_{}.pt".format(model_name))
model.eval()
# %%
import numpy as np
from utils.metrics import RMS_error 

batch = iter(test_dl).next()
x,y = batch
x = x.cuda().float()
y = y.cuda().double()

with torch.no_grad():
    pred = model(x).double()
print(pred.shape)

#%%
tk1 = model.CCTEncoder.tokenizer
tk = model.CCTEncoder   
pred_trans = tk(x)
    # pred_trans = pred_trans.squeeze().detach().cpu().numpy()
pred_trans = pred_trans.transpose(2,1).reshape(4,256,256).detach().cpu().numpy()
pred_transi = np.empty(shape=(4,256,256))
import matplotlib.pyplot as plt
for i in range(4):
    plt.figure(i+10)
    pred_transi[i,:,:] = pred_trans[i,:,:].T
    # pred_transi = (pred_transi-pred_transi.min())/(pred_transi.max()-pred_transi.min())
Plot_multi(pred_transi,["u","v","w",r"$\theta$"],save_dir="/home/yuning/thesis/valid/fig/23-2-6/{}_TransformerFeature{}".format(model_name,i))




#%%
plt.figure(0)
clb = Plot_2D_snapshots(pred.cpu().squeeze(),"/home/yuning/thesis/valid/fig/23-2-6/pred{}".format(model_name))
plt.colorbar(clb)

plt.figure(1)
clb = Plot_2D_snapshots(y.cpu().squeeze(),"/home/yuning/thesis/valid/fig/23-2-6/tar{}".format(model_name))
plt.colorbar(clb)

plt.figure(2)
clb = Plot_2D_snapshots(pred.cpu().squeeze()-y.cpu().squeeze(),"/home/yuning/thesis/valid/fig/23-2-6/error{}".format(model_name))
plt.colorbar(clb)
# %%
import numpy as np
from utils.metrics import RMS_error,Glob_error,Fluct_error,PCC
from scipy import stats
RMS = [];Glob_Error = [];Fluct= [];pcc = []
for batch in tqdm(test_dl):
    x,y = batch
    x = x.cuda().float()
    y = y.cuda().double()

    with torch.no_grad():
        pred = model(x).double()
    rms = RMS_error(pred.cpu().squeeze().numpy(),y.cpu().squeeze().numpy())
    glbrms = Glob_error(pred.cpu().squeeze().numpy(),y.cpu().squeeze().numpy())
    flbrms = Fluct_error(pred.cpu().squeeze().numpy(),y.cpu().squeeze().numpy())
    Pcc = PCC(pred.cpu().squeeze().numpy(),y.cpu().squeeze().numpy())
    RMS.append(rms)
    Glob_Error.append(glbrms)
    Fluct.append(flbrms)
    pcc.append(Pcc)

rms_error = np.mean(np.array(RMS))
glob_error = np.mean(np.array(Glob_Error))
fluct_error = np.mean(np.array(Fluct))
pcc_cor = np.mean(np.array(pcc))
print(f"Glob Error = {glob_error}")
print(f"RMS Error = {rms_error}")
print(f"FLuct Error = {fluct_error}")
print(f"PCC = {pcc_cor}")
#%%