#%%
import numpy as np 
import matplotlib.pyplot as plt
from utils.metrics import ERS
import seaborn as sns
from scipy.stats import pearsonr
items_list = ["glob","rms","fluct"]
#%%
y30_ERS_base = np.load("/home/yuning/thesis/valid/pred/y_plus_30-VARS-pr0.025_u_vel_v_vel_w_vel-TARGETS-pr0.025_flux_CBAM2_EPOCH=100/y.npy")
y50_ERS_base = np.load("/home/yuning/thesis/valid/pred/y_plus_30-VARS-pr0.025_u_vel_v_vel_w_vel-TARGETS-pr0.025_flux_FCN_EPOCH=100/y.npy")
y30_ERS_cbam = np.load("/home/yuning/thesis/valid/pred/y_plus_30-VARS-pr0.025_u_vel_v_vel_w_vel-TARGETS-pr0.025_flux_CBAM2_EPOCH=100/pred.npy")
y50_ERS_cbam = np.load("/home/yuning/thesis/valid/pred/y_plus_30-VARS-pr0.025_u_vel_v_vel_w_vel-TARGETS-pr0.025_flux_FCN_EPOCH=100/pred.npy")
# %%

sns.kdeplot(y30_ERS_base.mean(0).flatten(),label="DNS")
sns.kdeplot(y30_ERS_cbam.mean(0).flatten(),label="Attention")
sns.kdeplot(y50_ERS_cbam.mean(0).flatten(),label="FCN Baseline")
plt.legend()
plt.savefig("PDF Mean")
# sns.kdeplot(y30_ERS_base.mean(0).flatten())

# plt.show()
# # %%
# # sns.kdeplot(y50_ERS_base.mean(0).flatten())
# sns.kdeplot(y50_ERS_base.mean(0).flatten())
# sns.kdeplot(y50_ERS_base.mean(0).flatten())

# plt.show()
#%%
PCC_Base = []
PCC_CBAM = []
for i in range(y30_ERS_base.shape[0]):
    pcc_cbam, _ =pearsonr((y30_ERS_base[i,:,:].flatten()-y30_ERS_base.mean()),(y30_ERS_cbam[i,:,:].flatten()-y30_ERS_cbam.mean()))
    pcc_base, _ =pearsonr((y50_ERS_base[i,:,:].flatten()-y50_ERS_base.mean()),(y50_ERS_cbam[i,:,:].flatten()-y50_ERS_cbam.mean()))
    PCC_Base.append(pcc_base)
    PCC_CBAM.append(pcc_cbam)
plt.figure(0)
plt.scatter(y = (y30_ERS_base-y30_ERS_base.mean()).mean(1).mean(1).flatten(),x=(y30_ERS_cbam-y30_ERS_cbam.mean()).mean(1).mean(1).flatten(),s=PCC_CBAM,label="Attention",lw=2.5)
plt.scatter(y = (y50_ERS_base-y50_ERS_base.mean()).mean(1).mean(1).flatten(),x=(y50_ERS_cbam-y50_ERS_cbam.mean()).mean(1).mean(1).flatten(),s=PCC_Base,label="FCN Baseline",lw=2.5)

plt.grid()
plt.legend()
plt.xlabel("qw prediction")
plt.ylabel("qw DNS")
plt.savefig("Scatter")

plt.figure(1)
plt.plot(PCC_Base,"o",label="Baseline")
plt.plot(PCC_CBAM,"x",label="Attention")
plt.grid()
plt.legend()
plt.xlabel("qw prediction")
plt.ylabel("qw DNS")
plt.savefig("Point")
#%%

RMS_DNS = []
RMS_base = []
RMS_cbam = []
for i in range(407):
    rms =  np.sqrt(np.mean(y30_ERS_base[i,:,:]**2))
    RMS_DNS.append(rms)
    rms =  np.sqrt(np.mean(y30_ERS_cbam[i,:,:]**2))
    RMS_cbam.append(rms)
    rms =  np.sqrt(np.mean(y50_ERS_cbam[i,:,:]**2))
    RMS_base.append(rms)

mean_rms_dns = np.array(RMS_DNS).mean()
mean_rms_bsae = np.array(RMS_base).mean()
mean_rms_cbam = np.array(RMS_cbam).mean()
print(mean_rms_dns)
print(mean_rms_bsae)
print(mean_rms_cbam)
print(y30_ERS_base.mean())
print(y50_ERS_cbam.mean())
print(y30_ERS_cbam.mean())
#%%
SKEW_DNS=[]
SKEW_Attention=[]
SKEW_Baseline=[]
for i in range(3,10):
    print("Order="+str(i))
    skew_DNS = np.mean((  (y50_ERS_base-y50_ERS_base.mean())/(np.sqrt(np.mean(y50_ERS_base-y50_ERS_base.mean())))  )**i)
    skew_attention= np.mean(( ( y30_ERS_cbam-y30_ERS_cbam.mean())/(np.sqrt(np.mean(y30_ERS_cbam-y30_ERS_cbam.mean()))) )**i)
    skew_base= np.mean((  (y50_ERS_cbam-y50_ERS_cbam.mean())/(np.sqrt(np.mean(y50_ERS_cbam-y50_ERS_cbam.mean()))))**i)
    
    # print(f"For DNS:{skew_DNS}")
    # print(f"For Attention:{skew_attention}")
    # print(f"For Base:{skew_base}")
    SKEW_DNS.append(skew_DNS)
    SKEW_Attention.append(skew_attention)
    SKEW_Baseline.append(skew_base)
    # print(skew_attention)
    # print(skew_base)
#%%
plt.figure(3)
plt.semilogy(np.arange(3,10),SKEW_DNS,"ro",lw=2.5,label="DNS")
plt.semilogy(np.arange(3,10),SKEW_Attention,"bx",lw=2.5,label="Attention")
plt.semilogy(np.arange(3,10),SKEW_Baseline,"s",color="orange",lw=2.5,label="BaseLine")
plt.legend()
plt.xlabel("Order k",fontdict={"fontsize":16})
plt.ylabel("$qw^k$",fontdict={"fontsize":16})
# plt.xlim((3,10))
plt.grid()
plt.savefig("High_order_static")

# %%
from tqdm import tqdm
for i, error_item in tqdm(enumerate(items_list)):
    y30_error_base = np.load("/home/yuning/thesis/valid/pred/y_plus_30-VARS-pr0.025_u_vel_v_vel_w_vel-TARGETS-pr0.025_flux_FCN_EPOCH=100/{}.npy".format(error_item))
    y50_error_base = np.load("/home/yuning/thesis/valid/pred/y_plus_50-VARS-pr0.2_u_vel_v_vel_w_vel-TARGETS-pr0.2_flux_FCN_EPOCH=100/{}.npy".format(error_item))
    y30_error_cbam = np.load("/home/yuning/thesis/valid/pred/y_plus_30-VARS-pr0.025_u_vel_v_vel_w_vel-TARGETS-pr0.025_flux_CBAM2_EPOCH=100/{}.npy".format(error_item))
    y50_error_cbam = np.load("/home/yuning/thesis/valid/pred/y_plus_50-VARS-pr0.2_u_vel_v_vel_w_vel-TARGETS-pr0.2_flux_CBAM2_EPOCH=100/{}.npy".format(error_item))
    plt.figure(0)
    sns.kdeplot(y30_error_base,label="Baseline")
    sns.kdeplot(y30_error_cbam,label="CBAM")
    plt.legend()
    plt.savefig("{}_pdf_y30_".format(error_item))
    plt.clf()
    plt.figure(1)
    sns.kdeplot(y50_error_base,label="Baseline")
    sns.kdeplot(y50_error_cbam,label="CBAM")
    plt.legend()
    plt.savefig("{}_pdf_y50_".format(error_item))
    plt.clf()

# %%
#%%
sns.kdeplot(y30_rms_base.flatten(),label="RMSE Baseline")
sns.kdeplot(y30_rms_cbam.flatten(),label = "RMSE Attention")
plt.legend()
plt.show()
# %%
# sns.kdeplot(y50_ERS_base.mean(0).flatten())
sns.kdeplot(y50_rms_base.flatten())
sns.kdeplot(y50_rms_cbam.flatten())

plt.show()
# %%
y30_glb_base = np.load("/home/yuning/thesis/valid/pred/y_plus_30-VARS-pr0.025_u_vel_v_vel_w_vel-TARGETS-pr0.025_flux_FCN_EPOCH=100/glob.npy")
y50_glb_base = np.load("/home/yuning/thesis/valid/pred/y_plus_50-VARS-pr0.2_u_vel_v_vel_w_vel-TARGETS-pr0.2_flux_FCN_EPOCH=100/glob.npy")
y30_glb_cbam = np.load("/home/yuning/thesis/valid/pred/y_plus_30-VARS-pr0.025_u_vel_v_vel_w_vel-TARGETS-pr0.025_flux_CBAM2_EPOCH=100/glob.npy")
y50_glb_cbam = np.load("/home/yuning/thesis/valid/pred/y_plus_50-VARS-pr0.2_u_vel_v_vel_w_vel-TARGETS-pr0.2_flux_CBAM2_EPOCH=100/glob.npy")

# %%
#%%
sns.kdeplot(y30_glb_base.flatten(),"Baseline")
sns.kdeplot(y30_glb_cbam.flatten(),"CBAM")
plt.legend()
plt.savefig("Glob_pdf_y30")
# %%

sns.kdeplot(y50_glb_base.flatten())
sns.kdeplot(y50_glb_cbam.flatten())

plt.show()
# %%
y30_fluct_base = np.load("/home/yuning/thesis/valid/pred/y_plus_30-VARS-pr0.025_u_vel_v_vel_w_vel-TARGETS-pr0.025_flux_FCN_EPOCH=100/fluct.npy")
y50_fluct_base = np.load("/home/yuning/thesis/valid/pred/y_plus_50-VARS-pr0.2_u_vel_v_vel_w_vel-TARGETS-pr0.2_flux_FCN_EPOCH=100/fluct.npy")
y30_fluct_cbam = np.load("/home/yuning/thesis/valid/pred/y_plus_30-VARS-pr0.025_u_vel_v_vel_w_vel-TARGETS-pr0.025_flux_CBAM2_EPOCH=100/fluct.npy")
y50_fluct_cbam = np.load("/home/yuning/thesis/valid/pred/y_plus_50-VARS-pr0.2_u_vel_v_vel_w_vel-TARGETS-pr0.2_flux_CBAM2_EPOCH=100/fluct.npy")
#%%
sns.kdeplot(y30_fluct_base.flatten())
sns.kdeplot(y30_fluct_cbam.flatten())

plt.show()
# %%
# sns.kdeplot(y50_ERS_base.mean(0).flatten())
sns.kdeplot(y50_fluct_base.flatten())
sns.kdeplot(y50_fluct_cbam.flatten())

plt.show()
# %%
