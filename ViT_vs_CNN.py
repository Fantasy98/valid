#%%
import numpy as np
from utils.plots import Plot_2D_2snapshots,Snap_Plot3D,Plot_multi,Plot_2D_snapshots,PSD_single
from utils.metrics import ERS,PCC, RMS_error,Glob_error
import matplotlib.pyplot as plt 
import torch 
from scipy.stats import pearsonr
import argparse
#%%
parser = argparse.ArgumentParser(description='Setting Parameters')
parser.add_argument('--y', default= 30, type=int, help='Wall normal distance')
parser.add_argument('--pr', default= "0025", type=str, help='Wall normal distance')
args = parser.parse_args()


#%%
y_plus = args.y
pr = args.pr

fig_path = "/home/yuning/thesis/valid/fig/23-3-14/"

dvit = np.load(f"pred/y{y_plus}_vit_16h_4l_pr{pr}.npz")
# ground truth
y_vit = dvit["y"]
y_vit_f = y_vit - y_vit.mean()

# ViT
p_vit = dvit["pred"]
p_vit_f = p_vit - p_vit.mean()

#simple cnn
dcnn = np.load(f"pred/y{y_plus}_cnn_pr{pr}.npz")
p_cnn = dcnn["pred"]
p_cnn_f = p_cnn - p_cnn.mean()

# FCN
dfcn = np.load(f"pred/y{y_plus}_fcn_pr{pr}.npz")
p_fcn = dfcn["pred"]
p_fcn_f = p_fcn - p_fcn.mean()

#  CBAM
dcb = np.load(f"pred/y{y_plus}_cbam_pr{pr}.npz")
p_cb = dcb["pred"]
p_cb_f = p_cb - p_cb.mean()

# %%

SKEW_DNS=[]
SKEW_VIT=[]
SKEW_CNN = []
SKEW_FCN=[]
SKEW_CBAM=[]


for i in range(3,10):
    print("Order="+str(i))

    skew_DNS = np.mean((  (y_vit_f)/(np.sqrt(np.mean(y_vit_f**2)))  )**i)
    skew_vit= np.mean((  (p_vit_f)/(np.sqrt(np.mean(p_vit_f**2))) )**i)
    skew_cnn= np.mean((  (p_cnn_f)/(np.sqrt(np.mean(p_cnn_f**2))) )**i)
    skew_fcn= np.mean((  (p_fcn_f)/(np.sqrt(np.mean(p_fcn_f**2))) )**i)
    skew_cb= np.mean((  (p_cb_f)/(np.sqrt(np.mean(p_cb_f**2))) )**i)
    
    SKEW_DNS.append(skew_DNS)
    SKEW_VIT.append(skew_vit)
    SKEW_CNN.append(skew_cnn)
    SKEW_FCN.append(skew_fcn)
    SKEW_CBAM.append(skew_cb)
    
#%%
plt.figure(3,dpi=400)

plt.semilogy(np.arange(3,10),SKEW_DNS,"o",color="red",lw=2.5,label="DNS")
plt.semilogy(np.arange(3,10),SKEW_VIT,"s",color="orange",lw=2.5,label="VIT")
plt.semilogy(np.arange(3,10),SKEW_CNN,"+",color="cyan",lw=2.5,label="Simple FCN")
plt.semilogy(np.arange(3,10),SKEW_FCN,"x",color="blue",lw=2.5,label="FCN")
plt.semilogy(np.arange(3,10),SKEW_CBAM,"v",color="green",lw=2.5,label="CBAM")


plt.legend()
plt.xlabel("Order k",fontdict={"fontsize":16})
plt.ylabel("$<(q_w'/q_{w,rms})^k>$",fontdict={"fontsize":16})
plt.grid()

plt.savefig(fig_path+ f"HighOrderStat_{y_plus}_{pr}",bbox_inches="tight")
#%%
np.concatenate
dns_skew =np.array(SKEW_DNS)
vit_skew =np.array(SKEW_VIT)
cnn_skew =np.array(SKEW_CNN)
fcn_skew =np.array(SKEW_FCN)
cbam_skew =np.array(SKEW_CBAM)




#%%
np.savez_compressed(f"pred/y{y_plus}_pr{pr}_skew_5.npz",
                        dns = dns_skew,
                        vit = vit_skew,
                        cnn = cnn_skew,
                        fcn = fcn_skew,
                         cbam = cbam_skew )
#%%
# dif_plot_vit = (p_vit[0:1,:,:])/y_vit[0:1,:,:]
# dif_plot_fcn = (p_fcn[0:1,:,:])/y_fcn[0:1,:,:]

# dif_plots = np.concatenate([dif_plot_vit,dif_plot_fcn])
# print(dif_plots.shape)

# Plot_multi((dif_plots),["Diff Vit","Diff FCN"],fig_path+"VIT_Vs_FCN")
# #%%
# plot_vit = p_vit[0:1,:,:]
# plot_fcn = p_fcn[0:1,:,:]
# plot_y = y_fcn[0:1,:,:]
# snap_plots = np.concatenate([plot_vit,plot_fcn,plot_y])
# print(snap_plots.shape)
# Plot_multi((snap_plots),["Vit","FCN","Reference"],fig_path+"VIT_FCN")