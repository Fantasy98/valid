#%%
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.signal import correlate2d
import os
from tqdm import tqdm
#%%
y_plus = 15
pr = "1"
Pr = 1
fig_path = "/home/yuning/thesis/valid/fig/23-2-24/"
#%%
dvit = np.load(f"pred/y{y_plus}_all-pr{pr}_vit_16h_4l.npz")
y_vit = dvit["y"]
y_vit_f = y_vit - y_vit.mean()


p_vit = dvit["pred"]
p_vit_f = p_vit - p_vit.mean()

# rms_y = np.sqrt(np.mean(y_vit**2))
dfcn = np.load(f"pred/y{y_plus}_all-pr{pr}_fcn.npz")
p_fcn = dfcn["pred"]
p_fcn_f = p_fcn - p_fcn.mean()


dcb = np.load(f"pred/y{y_plus}_all-pr{pr}_cbam.npz")
p_cb = dcb["pred"]
p_cb_f = p_cb - p_cb.mean()

dcnn = np.load(f"pred/y{y_plus}_all-pr{pr}_fcn4.npz")
p_cnn = dcb["pred"]
p_cnn_f = p_cnn - p_cnn.mean()

# %%

# %%
corr_dns = correlate2d(y_vit_f[:,127:128,:].mean(0),y_vit_f[:,127:128,:].mean(0),mode="same",boundary="symm")
corr_vit = correlate2d(p_vit_f[:,127:128,:].mean(0),y_vit_f[:,127:128,:].mean(0),mode="same",boundary="symm")
corr_fcn = correlate2d(p_fcn_f[:,127:128,:].mean(0),y_vit_f[:,127:128,:].mean(0),mode="same",boundary="symm")
corr_cb = correlate2d(p_cb_f[:,127:128,:].mean(0),y_vit_f[:,127:128,:].mean(0),mode="same",boundary="symm")
corr_cnn = correlate2d(p_cnn_f[:,127:128,:].mean(0),y_vit_f[:,127:128,:].mean(0),mode="same",boundary="symm")

 
#%%
len_x = corr_dns.reshape(-1).shape[0]
x_range = np.linspace(0,4740,len_x)
plt.figure(0)
plt.plot(x_range,corr_dns.reshape(-1),"r-.",label="DNS")
plt.plot(x_range,corr_vit.reshape(-1),c="orange",label="ViT")
plt.plot(x_range,corr_fcn.reshape(-1),c="b",label="FCN")
plt.plot(x_range,corr_cb.reshape(-1),c="g",label="CBAM")
plt.plot(x_range,corr_cnn.reshape(-1),c="cyan",label="Simple FCN")

plt.xticks([0,950,1900,2850,3980,4740])
plt.xlabel(r"${x}^{+}$",fontdict={"size":16})
plt.ylabel(r"${<R_{{q'_{w}}_{DNS},{q'_{w}}_{Pred}}>}$",fontdict={"size":16})
plt.legend()
plt.title(f"Pr={Pr}",fontdict={"size":18})
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(fig_path+f"X_2Dcorr_Pr{pr}")
# plt.yscale("log")
# %%
corr_dns = correlate2d(y_vit_f[:,:,127:128].mean(0),y_vit_f[:,:,127:128].mean(0),mode="same")
corr_vit = correlate2d(p_vit_f[:,:,127:128].mean(0),y_vit_f[:,:,127:128].mean(0),mode="same")
corr_fcn = correlate2d(p_fcn_f[:,:,127:128].mean(0),y_vit_f[:,:,127:128].mean(0),mode="same")
corr_cb = correlate2d(p_cb_f[:,:,127:128].mean(0),y_vit_f[:,:,127:128].mean(0),mode="same")
corr_cnn = correlate2d(p_cnn_f[:,:,127:128].mean(0),y_vit_f[:,:,127:128].mean(0),mode="same")

#%%
len_x = corr_dns.reshape(-1).shape[0]
z_range = np.linspace(0,2370,len_x)
plt.figure(10)
plt.plot(z_range,corr_dns.reshape(-1),"r-.",label="DNS")
plt.plot(z_range,corr_vit.reshape(-1),c="orange",label="ViT")
plt.plot(z_range,corr_fcn.reshape(-1),c="b",label="FCN")
plt.plot(z_range,corr_cb.reshape(-1),c="g",label="CBAM")
plt.plot(z_range,corr_cnn.reshape(-1),c="cyan",label="Simple FCN")
plt.legend()
plt.xlabel(r"${z}^{+}$",fontdict={"size":16})
plt.title(f"Pr={Pr}",fontdict={"size":18})
plt.ylabel(r"${<R_{{q'_{w}}_{DNS},{q'_{w}}_{Pred}}>}$",fontdict={"size":16})
plt.xticks([0,470,950,1420,1900,2370])
plt.tight_layout()
plt.grid()
plt.savefig(fig_path+f"Z_2Dcorr_Pr{pr}")
# %%
