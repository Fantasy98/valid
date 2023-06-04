#%%
import numpy as np
from utils.plots import Plot_2D_2snapshots,Snap_Plot3D,Plot_multi,Plot_2D_snapshots,PSD_single
from utils.metrics import ERS,PCC, RMS_error,Glob_error,Fluct_error
import matplotlib.pyplot as plt 
import torch 
import seaborn as sns
from scipy.stats import pearsonr
import argparse
#%%
parser = argparse.ArgumentParser(description='Setting Parameters')
parser.add_argument('--y', default= 30, type=int, help='Wall normal distance')
parser.add_argument('--pr', default= "0025", type=str, help='Wall normal distance')
args = parser.parse_args()

y_plus = args.y
pr = args.pr


fig_path = "/home/yuning/thesis/valid/fig/23-3-14/"

dvit = np.load(f"pred/predictions/y{y_plus}_vit_pr{pr}.npz")
p_vit = dvit["pred"]
p_vit_f = p_vit - p_vit.mean()
rms_vit = np.sqrt(np.mean(p_vit**2))

y_vit = dvit["y"]
y_vit_f = y_vit - y_vit.mean()
rms_y = np.sqrt(np.mean(y_vit**2))

dfcn = np.load(f"pred/predictions/y{y_plus}_fcn_pr{pr}.npz")
p_fcn = dfcn["pred"]
p_fcn_f = p_fcn - p_fcn.mean()
rms_fcn = np.sqrt(np.mean(p_fcn**2))

dcnn = np.load(f"pred/predictions/y{y_plus}_cnn_pr{pr}.npz")
p_cnn = dcnn["pred"]
p_cnn_f = p_cnn - p_cnn.mean()
rms_cnn = np.sqrt(np.mean(p_cnn**2))



dcb = np.load(f"pred/predictions/y{y_plus}_cbam_pr{pr}.npz")
p_cb = dcb["pred"]
p_cb_f = p_cb - p_cb.mean()
rms_cb = np.sqrt(np.mean(p_cb**2))

print(p_cb.shape)

# %%
# sns.kdeplot(data=y_vit[0:50,:,:].flatten()/rms_y,color="red",label = "DNS")
# sns.kdeplot(data=p_vit[0:50,:,:].flatten()/rms_vit,color = "orange",label = "ViT")
# sns.kdeplot(data=p_fcn[0:50,:,:].flatten()/rms_fcn,color="blue",label = "FCN")
# sns.kdeplot(data=p_cb[0:50,:,:].flatten()/rms_cb,color = "green",label = "CBAM")
# sns.kdeplot(data=p_cnn[0:50,:,:].flatten()/rms_cnn,color="cyan",label = "Simple FCN")
# plt.legend()
# plt.savefig(fig_path+f"PDF_y30_{pr}",dpi=150,bbox_inches = "tight")
# %%
glob_vit = Glob_error(y_vit.mean(0),p_vit.mean(0))
glob_fcn = Glob_error(y_vit.mean(0),p_fcn.mean(0))
glob_cb = Glob_error(y_vit.mean(0),p_cb.mean(0))
print(glob_vit,glob_fcn,glob_cb)

# %%
PCC_vit = []
PCC_fcn = []
PCC_cbam = []
PCC_cnn= []

for i in range(y_vit.shape[0]):
    pcc_vit,_ = pearsonr(y_vit_f[i,:,:].flatten(),
                         p_vit_f[i,:,:].flatten())
    pcc_fcn,_ = pearsonr(y_vit_f[i,:,:].flatten(),
                         p_fcn_f[i,:,:].flatten())
    pcc_cbam,_ = pearsonr(y_vit_f[i,:,:].flatten(),
                         p_cb_f[i,:,:].flatten())
    
    pcc_cnn,_ = pearsonr(y_vit_f[i,:,:].flatten(),
                         p_cnn_f[i,:,:].flatten())
    PCC_vit.append(pcc_vit)
    PCC_fcn.append(pcc_fcn)
    PCC_cbam.append(pcc_cbam)
    PCC_cnn.append(pcc_cnn)
# %%
plt.figure(10)
plt.scatter(y = y_vit_f.mean(1).mean(1).flatten(),
            x = p_vit_f.mean(1).mean(1).flatten(),
            s = PCC_vit,marker="s",c="orange",lw=2.5,label="ViT")
plt.scatter(y = y_vit_f.mean(1).mean(1).flatten(),
            x = p_fcn_f.mean(1).mean(1).flatten(),
            s = PCC_fcn,marker="x",c="blue",lw=2.5,label="FCN")
plt.scatter(y = y_vit_f.mean(1).mean(1).flatten(),
            x = p_cb_f.mean(1).mean(1).flatten(),
            s = PCC_cbam,marker="v",c="green",lw=2.5,label="CBAM")
plt.scatter(y = y_vit_f.mean(1).mean(1).flatten(),
            x = p_cnn_f.mean(1).mean(1).flatten(),
            s = PCC_cnn,marker="+",c="cyan",lw=2.5,label="Simple FCN")
plt.grid()
plt.legend()
plt.xlabel("$q'_{w}$ Prediction",fontdict={"fontsize":16})
plt.ylabel("$q'_{w}$ DNS",fontdict={"fontsize":16})
plt.savefig(fig_path+f"PCC_scatter_y{y_plus}_{pr}",dpi=150,bbox_inches="tight")
# %%
PCC_vit = np.array(PCC_vit)
PCC_fcn = np.array(PCC_fcn)
PCC_cbam = np.array(PCC_cbam)
print(
        PCC_vit.mean(),
        PCC_fcn.mean(),
        PCC_cbam.mean(),
      )

# %%
