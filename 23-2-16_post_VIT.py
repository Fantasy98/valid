#%%
import numpy as np
from utils.plots import Plot_2D_2snapshots,Snap_Plot3D,Plot_multi,Plot_2D_snapshots,PSD_single
from utils.metrics import ERS,PCC
import matplotlib.pyplot as plt 
import torch 
from scipy.stats import pearsonr

fig_path = "/home/yuning/thesis/valid/fig/23-2-16/"
#%%
y_plus = 50
Pr = 0.2
Num_heads = 16
Num_layers = 4
model_name = f'y_plus_{y_plus}-VARS-pr{Pr}_u_vel_v_vel_w_vel-TARGETS-pr{Pr}_flux_Skip_FCN_CCT_{Num_heads}h_{Num_layers}l_EPOCH=200'
model_save_name = f"y_{y_plus}-pr02_flux_Skip_FCN_CCT_16h_4l_EPOCH=200"
d = np.load("/home/yuning/thesis/valid/pred/y_plus_50-VARS-pr0.2_u_vel_v_vel_w_vel-TARGETS-pr0.2_flux_Skip_FCN_CCT_16h_4l_EPOCH=200/y_50-pr0025_16h_4l_EPOCH=200.npz")
token  = d["tokens"]
pred = d["pred"]
pred_fluct = pred - pred.mean()
y = d["y"]
y_fluct = y - y.mean()
#%%
token_mean = token.mean(0)
print(token_mean.shape)
pred_trans = token_mean.reshape(256,256,4)
pred_transi = np.empty(shape=(4,256,256))
pred_trans_all = pred_trans.sum(0)
import matplotlib.pyplot as plt
for i in range(4):
    plt.figure(i+10)
    # pred_transi[i,:,:] = pred_trans[i,:,:]/np.abs(pred_trans_all)
    pred_transi[i,:,:] = (pred_trans[:,:,i] - pred_trans[:,:,i].min())/(pred_trans[:,:,i].max() - pred_trans[i,:,:].min())
    # pred_transi = (pred_transi-pred_transi.min())/(pred_transi.max()-pred_transi.min())
Plot_multi(pred_transi,["u","v","w",r"$\theta$"],save_dir="/home/yuning/thesis/valid/fig/23-2-16/Token_{}".format(model_save_name))


#%%
pcc= PCC(pred_fluct,y_fluct)
print(pcc)
#%%

#%%
PCC = []

for i in range(y.shape[0]):
    pcc_, _ = pearsonr(pred_fluct[i,:,:].flatten(),y_fluct[i,:,:].flatten())
    
    PCC.append(pcc_)
PCC = np.array(PCC)
y_flate = y_fluct.mean(1).mean(1).reshape(-1,1)
pred_flate = pred_fluct.mean(1).mean(1).reshape(-1,1)
print(y_flate.shape)
print(pred_flate.shape)
print(len(PCC))


plt.figure(0,dpi=400)
plt.scatter(y = y_flate,x=pred_flate,s=PCC,lw=2.5)

plt.grid()
plt.legend()
plt.xlabel("$q'_{w}$ Prediction",fontdict={"fontsize":16})
plt.ylabel("$q'_{w}$ DNS",fontdict={"fontsize":16})
plt.savefig(fig_path+ "Scatter_{}".format(model_save_name),bbox_inches="tight")



plot_np = np.concatenate([pred[10,:,:].reshape(1,256,256),y[10,:,:].reshape(1,256,256)],axis=0)
print(plot_np.shape)
Plot_2D_2snapshots(plot_np,fig_path+"compare"+model_save_name)

ers = ERS(plot_np[0,:,:],plot_np[1,:,:])
print(ers.shape)
Snap_Plot3D(ers,fig_path+"ERS"+model_save_name)

Plot_2D_snapshots(ers,fig_path+"2DERS"+model_save_name)
# %%
SKEW_DNS=[]
SKEW_Baseline=[]
for i in range(3,10):
    print("Order="+str(i))
    skew_DNS = np.mean((  (y_fluct)/(np.sqrt(np.mean(y_fluct**2)))  )**i)
    skew_base= np.mean((  (pred_fluct)/(np.sqrt(np.mean(pred_fluct**2))) )**i)
    
    # print(f"For DNS:{skew_DNS}")
    # print(f"For Attention:{skew_attention}")
    # print(f"For Base:{skew_base}")
    SKEW_DNS.append(skew_DNS)
    # SKEW_Attention.append(skew_attention)
    SKEW_Baseline.append(skew_base)
    # print(skew_attention)
    # print(skew_base)
#%%
plt.figure(3,dpi=400)
plt.semilogy(np.arange(3,10),SKEW_DNS,"ro",lw=2.5,label="DNS")
# plt.semilogy(np.arange(3,10),SKEW_Attention,"bx",lw=2.5,label="Attention")
plt.semilogy(np.arange(3,10),SKEW_Baseline,"s",color="orange",lw=2.5,label="BaseLine")
plt.legend()
plt.xlabel("Order k",fontdict={"fontsize":16})
plt.ylabel("$<(q_w'/q_{w,rms})^k>$",fontdict={"fontsize":16})
# plt.xlim((3,10))
plt.grid()
plt.savefig(fig_path+ "HighOrderStat_{}".format(model_save_name),bbox_inches="tight")

# %%

PSD_single(y,pred,fig_path+"PSD_{}".format(model_save_name))

# %%
