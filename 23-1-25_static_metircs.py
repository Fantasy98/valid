#%%
import numpy as np 
import matplotlib.pyplot as plt
from utils.metrics import ERS
import seaborn as sns
from scipy.stats import pearsonr


y_plus = 50
pr = 0.2
y30_y_cb = np.load("/home/yuning/thesis/valid/pred/y_plus_{}-VARS-pr{}_u_vel_v_vel_w_vel-TARGETS-pr{}_flux_CBAM2_EPOCH=100/y.npy".format(y_plus,pr,pr))
y30_y_base = np.load("/home/yuning/thesis/valid/pred/y_plus_{}-VARS-pr{}_u_vel_v_vel_w_vel-TARGETS-pr{}_flux_FCN_EPOCH=100/y.npy".format(y_plus,pr,pr))
y30_pred_cbam = np.load("/home/yuning/thesis/valid/pred/y_plus_{}-VARS-pr{}_u_vel_v_vel_w_vel-TARGETS-pr{}_flux_CBAM2_EPOCH=100/pred.npy".format(y_plus,pr,pr))
y30_pred_base = np.load("/home/yuning/thesis/valid/pred/y_plus_{}-VARS-pr{}_u_vel_v_vel_w_vel-TARGETS-pr{}_flux_FCN_EPOCH=100/pred.npy".format(y_plus,pr,pr))
pr = "02"
#%%
PCC_Base = []
PCC_CBAM = []
for i in range(y30_y_cb.shape[0]):
    pcc_cbam, _ =pearsonr((y30_y_cb[i,:,:].flatten()-y30_y_cb.mean()),(y30_pred_cbam[i,:,:].flatten()-y30_pred_cbam.mean()))
    pcc_base, _ =pearsonr((y30_y_base[i,:,:].flatten()-y30_y_base.mean()),(y30_pred_base[i,:,:].flatten()-y30_pred_base.mean()))
    PCC_Base.append(pcc_base)
    PCC_CBAM.append(pcc_cbam)

    
plt.figure(0,dpi=400)
plt.scatter(y = (y30_y_cb-y30_y_cb.mean()).mean(1).mean(1).flatten(),x=(y30_pred_cbam-y30_pred_cbam.mean()).mean(1).mean(1).flatten(),s=PCC_CBAM,label="Attention",lw=2.5)
plt.scatter(y = (y30_y_base-y30_y_base.mean()).mean(1).mean(1).flatten(),x=(y30_pred_base-y30_pred_base.mean()).mean(1).mean(1).flatten(),s=PCC_Base,label="FCN Baseline",lw=2.5)

plt.grid()
plt.legend()
plt.xlabel("$q'_{w}$ Prediction",fontdict={"fontsize":16})
plt.ylabel("$q'_{w}$ DNS",fontdict={"fontsize":16})
plt.savefig("y_plus_{}_pr{}_Scatter".format(y_plus,pr),bbox_inches="tight")

plt.figure(1)
plt.plot(PCC_Base,"o",label="Baseline")
plt.plot(PCC_CBAM,"x",label="Attention")
plt.grid()
plt.legend()
plt.xlabel("$q'_{w}$ Prediction",fontdict={"fontsize":16})
plt.ylabel("$q'_{w}$ DNS",fontdict={"fontsize":16})
plt.savefig("y_plus_{}_pr{}_Point".format(y_plus,pr),bbox_inches="tight")
#%%

RMS_DNS = []
RMS_base = []
RMS_cbam = []
for i in range(407):
    rms =  np.sqrt(np.mean(y30_y_cb[i,:,:]**2))
    RMS_DNS.append(rms)
    rms =  np.sqrt(np.mean(y30_pred_cbam[i,:,:]**2))
    RMS_cbam.append(rms)
    rms =  np.sqrt(np.mean(y30_pred_base[i,:,:]**2))
    RMS_base.append(rms)

mean_rms_dns = np.array(RMS_DNS).mean()
mean_rms_bsae = np.array(RMS_base).mean()
mean_rms_cbam = np.array(RMS_cbam).mean()
print(f" RMS of DNS: {mean_rms_dns}")
print(f" RMS of Baseline: {mean_rms_bsae}")
print(f" RMS of Attention: {mean_rms_cbam}")
print(mean_rms_cbam)
print(y30_y_cb.mean())
print(y30_pred_base.mean())
print(y30_pred_cbam.mean())
#%%
SKEW_DNS=[]
SKEW_Attention=[]
SKEW_Baseline=[]
for i in range(3,10):
    print("Order="+str(i))
    skew_DNS = np.mean((  (y30_y_base-y30_y_base.mean())/(np.sqrt(np.mean(y30_y_base-y30_y_base.mean())))  )**i)
    skew_attention= np.mean(( ( y30_pred_cbam-y30_pred_cbam.mean())/(np.sqrt(np.mean(y30_pred_cbam-y30_pred_cbam.mean()))) )**i)
    skew_base= np.mean((  (y30_pred_base-y30_pred_base.mean())/(np.sqrt(np.mean(y30_pred_base-y30_pred_base.mean()))))**i)
    
    # print(f"For DNS:{skew_DNS}")
    # print(f"For Attention:{skew_attention}")
    # print(f"For Base:{skew_base}")
    SKEW_DNS.append(skew_DNS)
    SKEW_Attention.append(skew_attention)
    SKEW_Baseline.append(skew_base)
    # print(skew_attention)
    # print(skew_base)
#%%
plt.figure(3,dpi=400)
plt.semilogy(np.arange(3,10),SKEW_DNS,"ro",lw=2.5,label="DNS")
plt.semilogy(np.arange(3,10),SKEW_Attention,"bx",lw=2.5,label="Attention")
plt.semilogy(np.arange(3,10),SKEW_Baseline,"s",color="orange",lw=2.5,label="BaseLine")
plt.legend()
plt.xlabel("Order k",fontdict={"fontsize":16})
plt.ylabel("$<(q_w'/q_{w,rms})^k>$",fontdict={"fontsize":16})
# plt.xlim((3,10))
plt.grid()
plt.savefig("y_plus_{}_pr{}_High_order_static".format(y_plus,pr),bbox_inches="tight")