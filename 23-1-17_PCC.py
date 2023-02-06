#%%
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

y30_pred_base = np.load("/home/yuning/thesis/valid/pred/y_plus_30-VARS-pr0.025_u_vel_v_vel_w_vel-TARGETS-pr0.025_flux_FCN_EPOCH=100/pred.npy")
y30_y_base = np.load("/home/yuning/thesis/valid/pred/y_plus_30-VARS-pr0.025_u_vel_v_vel_w_vel-TARGETS-pr0.025_flux_FCN_EPOCH=100/y.npy")
y50_pred_base = np.load("/home/yuning/thesis/valid/pred/y_plus_50-VARS-pr0.2_u_vel_v_vel_w_vel-TARGETS-pr0.2_flux_FCN_EPOCH=100/pred.npy")
y50_y_base = np.load("/home/yuning/thesis/valid/pred/y_plus_50-VARS-pr0.2_u_vel_v_vel_w_vel-TARGETS-pr0.2_flux_FCN_EPOCH=100/y.npy")
y30_pred_cbam = np.load("/home/yuning/thesis/valid/pred/y_plus_30-VARS-pr0.025_u_vel_v_vel_w_vel-TARGETS-pr0.025_flux_CBAM2_EPOCH=100/pred.npy")
y30_y_cbam = np.load("/home/yuning/thesis/valid/pred/y_plus_30-VARS-pr0.025_u_vel_v_vel_w_vel-TARGETS-pr0.025_flux_CBAM2_EPOCH=100/y.npy")
y50_pred_cbam = np.load("/home/yuning/thesis/valid/pred/y_plus_50-VARS-pr0.2_u_vel_v_vel_w_vel-TARGETS-pr0.2_flux_CBAM2_EPOCH=100/pred.npy")
y50_y_cbam = np.load("/home/yuning/thesis/valid/pred/y_plus_50-VARS-pr0.2_u_vel_v_vel_w_vel-TARGETS-pr0.2_flux_CBAM2_EPOCH=100/y.npy")
#%%
from utils.plots import Snap_Plot3D
Snap_Plot3D(y30_pred_base[0,:,:])
plt.savefig("3d_Snap",bbox_inches="tight")
#%%
from tqdm import tqdm
PCC_cbam_30 = np.empty(shape=y30_pred_cbam.shape[0])
PCC_base_30 = np.empty(shape=y30_pred_base.shape[0])
for snap in tqdm(range(y30_pred_base.shape[0])):
    PCC_cbam_30[snap],_ = stats.pearsonr(y30_pred_cbam[snap,:,:].flatten(),
                                     y30_y_cbam[snap,:,:].flatten()   )
    PCC_base_30[snap],_ = stats.pearsonr(y30_pred_base[snap,:,:].flatten(),
                                     y30_y_base[snap,:,:].flatten()   )
# %%
plt.figure(0)
sns.kdeplot(PCC_base_30,label="Baseline")
sns.kdeplot(PCC_cbam_30,label="CBAM")
plt.legend()
plt.savefig("PCC_pdf_y30_base_vs_cbam")
# %%
from tqdm import tqdm
PCC_cbam_50 = np.empty(shape=y50_pred_cbam.shape[0])
PCC_base_50 = np.empty(shape=y50_pred_base.shape[0])
for snap in tqdm(range(y50_pred_base.shape[0])):
    PCC_cbam_50[snap],_ = stats.pearsonr(y50_pred_cbam[snap,:,:].flatten(),
                                     y50_y_cbam[snap,:,:].flatten()   )
    PCC_base_50[snap],_ = stats.pearsonr(y50_pred_base[snap,:,:].flatten(),
                                     y50_y_base[snap,:,:].flatten()   )
# %%
plt.figure(1)
sns.kdeplot(PCC_base_50,label="Baseline")
sns.kdeplot(PCC_cbam_50,label="CBAM")
plt.legend()
plt.savefig("PCC_pdf_y50_base_vs_cbam")
#%%