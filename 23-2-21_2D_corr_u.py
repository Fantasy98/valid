#%%
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.signal import correlate2d
import os
from tqdm import tqdm
from utils.datas import slice_dir,JointDataset
import torch
from torch.utils.data import DataLoader
#%%
y_plus = 30
pr = "071"
Pr = 0.71
fig_path = "/home/yuning/thesis/valid/fig/23-2-22/"
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

var=['u_vel',"v_vel","w_vel",f"pr{Pr}"]
target=[f'pr{Pr}_flux']
normalized=False
y_plus=30
save_types= ["train","test","validation"]
root_path = "/home/yuning/thesis/tensor"
test_path = slice_dir(root_path,y_plus,var,target,"test",normalized)
print(f"Data will be loaded from {test_path}")
    
if Pr == 0.025:

    test_data1 = torch.load(test_path+"/test1.pt")
    test_data2 = torch.load(test_path+"/test2.pt")


    test_x = torch.cat([test_data1.x,test_data2.x])
    test_y = torch.cat([test_data1.y,test_data2.y])
    test_ds = JointDataset(test_x,test_y)

        # test_ds = torch.load(test_path+"/test1.pt")

    # test_dl = DataLoader(test_ds,shuffle=False,batch_size=1)
else:
    test_ds = torch.load(test_path+"/test.pt")
#%%
x = test_ds.x.detach().cpu().numpy()
u = x[:,0,:,:]
uf = u - u.mean()
# %%
corr_dns = correlate2d(uf[:,:,127:128].mean(0),y_vit_f[:,:,127:128].mean(0),mode="same",boundary="symm")
corr_vit = correlate2d(uf[:,:,127:128].mean(0),p_vit_f[:,:,127:128].mean(0),mode="same",boundary="symm")
corr_fcn = correlate2d(uf[:,:,127:128].mean(0),p_fcn_f[:,:,127:128].mean(0),mode="same",boundary="symm")
corr_cb = correlate2d(uf[:,:,127:128].mean(0),p_cb_f[:,:,127:128].mean(0),mode="same",boundary="symm")
corr_cnn = correlate2d(uf[:,:,127:128].mean(0),p_cnn_f[:,:,127:128].mean(0),mode="same",boundary="symm")
#%%
corr_dns = correlate2d(u[:,:,127:128].mean(0),y_vit[:,:,127:128].mean(0),mode="same",boundary="fill")
corr_vit = correlate2d(u[:,:,127:128].mean(0),p_vit[:,:,127:128].mean(0),mode="same",boundary="fill")
corr_fcn = correlate2d(u[:,:,127:128].mean(0),p_fcn[:,:,127:128].mean(0),mode="same",boundary="fill")
corr_cb = correlate2d(u[:,:,127:128].mean(0),p_cb[:,:,127:128].mean(0),mode="same",boundary="fill")
corr_cnn = correlate2d(u[:,:,127:128].mean(0),p_cnn[:,:,127:128].mean(0),mode="same",boundary="fill")
#%%
um = u.mean(0)
y_vitm = y_vit.mean(0)
p_vitm = p_vit.mean(0)
p_fcnm = p_fcn.mean(0)
p_cbm = p_cb.mean(0)
p_cnnm = p_cnn.mean(0)

corr_dns = correlate2d(um[:,127:128],y_vitm[:,127:128],mode="same",boundary="fill")
corr_vit = correlate2d(um[:,127:128],p_vitm[:,127:128],mode="same",boundary="fill")
corr_fcn = correlate2d(um[:,127:128],p_fcnm[:,127:128],mode="same",boundary="fill")
corr_cb = correlate2d(um[:,127:128],p_cbm[:,127:128],mode="same",boundary="fill")
corr_cnn = correlate2d(um[:,127:128],p_cnnm[:,127:128],mode="same",boundary="fill")

# %%
len_x = corr_dns.reshape(-1).shape[0]
x_range = np.linspace(0,4740,len_x)


corr_dns = (corr_dns-corr_dns.mean())/corr_dns.std()
corr_vit = (corr_vit-corr_vit.mean())/corr_vit.std()
corr_fcn = (corr_fcn-corr_fcn.mean())/corr_fcn.std()
corr_cb = (corr_cb-corr_cb.mean())/corr_cb.std()
corr_cnn = (corr_cnn-corr_cnn.mean())/corr_cnn.std()

plt.figure(0)
plt.plot(x_range,corr_dns.reshape(-1),"r-.",label="DNS")
plt.plot(x_range,corr_vit.reshape(-1),c="orange",label="ViT")
plt.plot(x_range,corr_fcn.reshape(-1),c="b",label="FCN")
plt.plot(x_range,corr_cb.reshape(-1),c="g",label="CBAM")
plt.plot(x_range,corr_cnn.reshape(-1),c="cyan",label="Simple FCN")
# plt.yscale("log")
plt.xticks([0,950,1900,2850,3980,4740])
plt.xlabel(r"${x}^{+}$",fontdict={"size":16})
plt.ylabel(r"${<R_{{q'_{w}},{U'}}>}$",fontdict={"size":16})
plt.legend()
plt.title(f"Pr={Pr}",fontdict={"size":18})
plt.legend()
plt.grid()
plt.tight_layout()
# %%

corr_dns = correlate2d(uf[:,127:128,:].mean(0),y_vit_f[:,127:128,:].mean(0),mode="same",boundary="fill")
corr_vit = correlate2d(uf[:,127:128,:].mean(0),p_vit_f[:,127:128,:].mean(0),mode="same",boundary="fill")
corr_fcn = correlate2d(uf[:,127:128,:].mean(0),p_fcn_f[:,127:128,:].mean(0),mode="same",boundary="fill")
corr_cb = correlate2d(uf[:,127:128,:].mean(0),p_cb_f[:,127:128,:].mean(0),mode="same",boundary="fill")
corr_cnn = correlate2d(uf[:,127:128,:].mean(0),p_cnn_f[:,127:128,:].mean(0),mode="same",boundary="fill")
#%%
corr_dns = correlate2d(u[:,127:128,:].mean(0),y_vit[:,127:128,:].mean(0),mode="same",boundary="fill")
corr_vit = correlate2d(u[:,127:128,:].mean(0),p_vit[:,127:128,:].mean(0),mode="same",boundary="fill")
corr_fcn = correlate2d(u[:,127:128,:].mean(0),p_fcn[:,127:128,:].mean(0),mode="same",boundary="fill")
corr_cb = correlate2d(u[:,127:128,:].mean(0),p_cb[:,127:128,:].mean(0),mode="same",boundary="fill")
corr_cnn = correlate2d(u[:,127:128,:].mean(0),p_cnn[:,127:128,:].mean(0),mode="same",boundary="fill")

#%%


corr_dns = (corr_dns-corr_dns.mean())/corr_dns.std()
corr_vit = (corr_vit-corr_vit.mean())/corr_vit.std()
corr_fcn = (corr_fcn-corr_fcn.mean())/corr_fcn.std()
corr_cb = (corr_cb-corr_cb.mean())/corr_cb.std()
corr_cnn = (corr_cnn-corr_cnn.mean())/corr_cnn.std()

len_x = corr_dns.reshape(-1).shape[0]
z_range = np.linspace(0,2370,len_x)
plt.figure(10)
plt.plot(z_range,corr_dns.reshape(-1),"r-.",label="DNS")
plt.plot(z_range,corr_vit.reshape(-1),c="orange",label="ViT")
plt.plot(z_range,corr_fcn.reshape(-1),c="b",label="FCN")
plt.plot(z_range,corr_cb.reshape(-1),c="g",label="CBAM")
plt.plot(z_range,corr_cnn.reshape(-1),c="cyan",label="Simple FCN")
plt.legend()
# plt.yscale("log")
plt.xlabel(r"${z}^{+}$",fontdict={"size":16})
plt.title(f"Pr={Pr}",fontdict={"size":18})
plt.ylabel(r"${<R_{{q'_{w}},{U'}}>}$",fontdict={"size":16})
plt.xticks([0,470,950,1420,1900,2370])
plt.tight_layout()
plt.grid()
# %%
