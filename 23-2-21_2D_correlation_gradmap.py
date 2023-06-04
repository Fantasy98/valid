#%%
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.signal import correlate2d
import os
from tqdm import tqdm
#%%
save_fig = "fig/23-2-24/"
names = ["vit_mul","fcn","cbam","cnn"]

y_plus = 15
prs = ["0025","02","071","1"]
# prs = ["0025"]
PR = [0.025,0.2,0.71,1]
model_names = ["ViT","FCN","CBAM","Simple FCN"]
#%%

np_path = "/home/yuning/thesis/valid/results/y_plus_15/2d_correlation/"
FeatureCorr = []
for pr in prs:
    pr_path = os.path.join(np_path,f"pr{pr}.npy")
    pr_np = np.load(pr_path)
    FeatureCorr.append(pr_np)

#%%


no_var = 0
fluct = True
# fluct = False
VAR =  ["U","V","W","T"]

print(f"INFO: Began to gather Gradient Map for {VAR[no_var]}")
ALL_PR = []
for pr in tqdm(prs):
    grad_dict = {}
    for name in names:
        data_dir = f"pred/y{y_plus}_pr{pr}_{name}_gradmap.npz"
        grad_map = np.load(data_dir)
        gdmp = grad_map["gradmap"]
        
        if fluct:    # Noise cancelling 
            for i in range(4):
                gdmp[:,i,:,:] = np.sqrt(gdmp[:,i,:,:]**2) - gdmp[:,i,:,:].mean()
        
        gdmp = gdmp.mean(0)
        u = gdmp[no_var,118:138,118:138] 
        # Regularize to 0 ~ 1
        # u = (u-gdmp.min())/(gdmp.max()-gdmp.min())
        u = (u-u.min())/(u.max()-u.min())
        grad_dict[name] = u
    ALL_PR.append(grad_dict)



# %%

print("INFO: Plotting")
# index for Pr number
i = 0 
for idx in tqdm(range(len(prs))):
    i += 1
    d2u = FeatureCorr[idx][no_var,:,:]
    d2u_reg = (d2u[118:138,118:138]-d2u[118:138,118:138].min())/(d2u[118:138,118:138].max()-d2u[118:138,118:138].min())
    # d2u_reg = (d2u-d2u.min())/(d2u.max()-d2u.min())
    # d2u_reg = (d2u_reg-d2u_reg.min())/(d2u_reg.max()-d2u_reg.min())

    # d2u_reg = (d2u_reg-d2u_reg.mean())/(d2u_reg.std())
    fcn_0025 = ALL_PR[idx]["fcn"]
    vit_0025 = ALL_PR[idx]["vit_mul"]
    cb_0025 = ALL_PR[idx]["cbam"]
    cnn_0025 = ALL_PR[idx]["cnn"]



    #####
    # Z direction
    #####
    coor_fcn = correlate2d(fcn_0025[10:11,:],d2u_reg[10:11,:],mode="full",boundary="fill")
    coor_vit = correlate2d(vit_0025[10:11,:],d2u_reg[10:11,:],mode="full",boundary="fill")
    coor_cb = correlate2d(cb_0025[10:11,:],d2u_reg[10:11,:],mode="full",boundary="fill")
    coor_cnn = correlate2d(cnn_0025[10:11,:],d2u_reg[10:11,:],mode="full",boundary="fill")
    coor_self = correlate2d(d2u_reg[10:11,:],d2u_reg[10:11,:],mode="full",boundary="fill")
  
    coor_self =  (coor_self - coor_self.mean())/coor_self.std()
    coor_fcn =  (coor_fcn - coor_fcn.mean())/coor_fcn.std()
    coor_vit =  (coor_vit - coor_vit.mean())/coor_vit.std()
    coor_cb =  (coor_cb - coor_cb.mean())/coor_cb.std()
    coor_cnn =  (coor_cnn - coor_cnn.mean())/coor_cnn.std()


    plt.figure(10+i)
    axis_range_x=np.linspace(1160,1200,coor_fcn.reshape(-1).shape[0])
    plt.plot(axis_range_x,coor_fcn.reshape(-1),marker = "x",c="b",label= "FCN")
    plt.plot(axis_range_x,coor_vit.reshape(-1),marker = "s",c="orange",label= "VIT")
    plt.plot(axis_range_x,coor_cb.reshape(-1),marker = "v",c="g",label= "CBAM")
    plt.plot(axis_range_x,coor_cnn.reshape(-1),marker = "+",c="cyan",label= "Simple FCN")
    plt.plot(axis_range_x, coor_self.reshape(-1),c="red",label= "DNS")
    
    plt.xticks(np.round(axis_range_x[::5]))
    plt.xlabel(r"${z}^{+}$",fontdict={"size":16})
    plt.ylabel(r"${R_{R,G}}$",fontdict={"size":16})
    plt.legend()
    plt.title(f"Pr={PR[idx]}",fontdict={"size":18})
    plt.savefig(save_fig+f"Z_corr_vs_grad_Pr{prs[idx]}_{VAR[no_var]}",bbox_inches = "tight",dpi=200)

    #####
    # X direction
    #####
    coor_fcn = correlate2d(fcn_0025[:,10:11],d2u_reg[:,10:11],mode="full",boundary="circular")
    coor_vit = correlate2d(vit_0025[:,10:11],d2u_reg[:,10:11],mode="full",boundary="circular")
    coor_cb = correlate2d(cb_0025[:,10:11],d2u_reg[:,10:11],mode="full",boundary="circular")
    coor_cnn = correlate2d(cnn_0025[:,10:11],d2u_reg[:,10:11],mode="full",boundary="circular")
    coor_self = correlate2d(d2u_reg[:,10:11],d2u_reg[:,10:11],mode="full",boundary="circular")
    
    coor_self =  (coor_self - coor_self.mean())/coor_self.std()
    coor_fcn =  (coor_fcn - coor_fcn.mean())/coor_fcn.std()
    coor_vit =  (coor_vit - coor_vit.mean())/coor_vit.std()
    coor_cb =  (coor_cb - coor_cb.mean())/coor_cb.std()
    coor_cnn =  (coor_cnn - coor_cnn.mean())/coor_cnn.std()
    plt.figure(i+1)
    axis_range_z=np.linspace(2355,2385,39)
    plt.plot(axis_range_z,coor_fcn.reshape(-1),marker = "x",c="b",label= "FCN")
    plt.plot(axis_range_z,coor_vit.reshape(-1),marker = "s",c="orange",label= "VIT")
    plt.plot(axis_range_z,coor_cb.reshape(-1),marker = "v",c="g",label= "CBAM")
    plt.plot(axis_range_z,coor_cnn.reshape(-1),marker = "+",c="cyan",label= "Simple FCN")
    plt.xticks(np.round(axis_range_z[::5]))
    plt.xlabel(r"${x}^{+}$",fontdict={"size":16})
    plt.ylabel(r"${R_{R,G}}$",fontdict={"size":16})
    plt.plot(axis_range_z, coor_self.reshape(-1),c="red",label= "DNS")
    plt.legend()
    plt.title(f"Pr={PR[idx]}",fontdict={"size":18})
    
    plt.savefig(save_fig+f"X_corr_vs_grad_Pr{prs[idx]}_{VAR[no_var]}",bbox_inches = "tight",dpi=200)
# %%
print("INFO: Plotting")
# index for Pr number
i = 0 
for idx in tqdm(range(len(prs))):
    i += 1
    d2u = FeatureCorr[idx][no_var,:,:]
    # d2u_reg = (d2u[118:138,118:138]-d2u[118:138,118:138].min())/(d2u[118:138,118:138].max()-d2u[118:138,118:138].min())
    d2u_reg = (d2u[118:138,118:138]-d2u.min())/(d2u.max()-d2u.min())
    # d2u_reg = (d2u-d2u.min())/(d2u.max()-d2u.min())
    d2u_reg = (d2u_reg-d2u_reg.mean())

    # d2u_reg = (d2u_reg-d2u_reg.mean())/(d2u_reg.std())
    fcn_0025 = ALL_PR[idx]["fcn"]
    vit_0025 = ALL_PR[idx]["vit_16h_4l"]
    cb_0025 = ALL_PR[idx]["cbam"]
    cnn_0025 = ALL_PR[idx]["cnn"]
    # d2u_reg = d2u_reg[118:138,118:138]
    # d2u_reg = d2u[118:138,118:138]


    #####
    # Z direction
    #####
    coor_fcn = fcn_0025[10,:]
    coor_vit = vit_0025[10,:]
    coor_cb = cb_0025[10,:]
    coor_cnn = cnn_0025[10,:]
    coor_self = d2u_reg[10,:]
  
    # coor_self =  (coor_self - coor_self.mean())
    coor_fcn =  (coor_fcn - coor_fcn.mean())
    coor_vit =  (coor_vit - coor_vit.mean())
    coor_cb =  (coor_cb - coor_cb.mean())
    coor_cnn =  (coor_cnn - coor_cnn.mean())
      
    # coor_self =  (coor_self - coor_self.min())/(coor_self.max() - coor_self.min())
    # coor_fcn =  (coor_fcn - coor_fcn.min())/(coor_fcn - coor_fcn.min())
    # coor_vit =  (coor_vit - coor_vit.min())/(coor_vit - coor_vit.min())
    # coor_cb =  (coor_cb - coor_cb.min())/(coor_cb - coor_cb.min())
    # coor_cnn =  (coor_cnn - coor_cnn.min())/(coor_cnn - coor_cnn.min())

    plt.figure(20+i)
    axis_range_x=np.linspace(1160,1200,coor_fcn.reshape(-1).shape[0])
    plt.plot(axis_range_x,coor_fcn.reshape(-1),marker = "x",c="b",label= "FCN")
    plt.plot(axis_range_x,coor_vit.reshape(-1),marker = "s",c="orange",label= "VIT")
    plt.plot(axis_range_x,coor_cb.reshape(-1),marker = "v",c="g",label= "CBAM")
    plt.plot(axis_range_x,coor_cnn.reshape(-1),marker = "+",c="cyan",label= "Simple FCN")
    plt.plot(axis_range_x, coor_self.reshape(-1),c="red",label= "DNS")

   
    plt.xticks(np.round(axis_range_x[::5]))
    plt.xlabel(r"${z}^{+}$",fontdict={"size":16})
    plt.ylabel(r"${R_{R,G}}$",fontdict={"size":16})
    plt.legend()
    plt.title(f"Pr={PR[idx]}",fontdict={"size":18})
    plt.savefig(save_fig+f"Z_corr_vs_grad_Pr{prs[idx]}_{VAR[no_var]}_nocorr",bbox_inches = "tight",dpi=200)

    #####
    # X direction
    #####
    coor_fcn = fcn_0025[:,10]
    coor_vit = vit_0025[:,10]
    coor_cb = cb_0025[:,10]
    coor_cnn = cnn_0025[:,10]
    coor_self = d2u_reg[:,10]
    # coor_self =  (coor_self - coor_self.mean())/coor_self.std()
    # coor_fcn =  (coor_fcn - coor_fcn.mean())/coor_fcn.std()
    # coor_vit =  (coor_vit - coor_vit.mean())/coor_vit.std()
    # coor_cb =  (coor_cb - coor_cb.mean())/coor_cb.std()
    # coor_cnn =  (coor_cnn - coor_cnn.mean())/coor_cnn.std()
    # coor_self =  (coor_self - coor_self.mean())
    coor_fcn =  (coor_fcn - coor_fcn.mean())
    coor_vit =  (coor_vit - coor_vit.mean())
    coor_cb =  (coor_cb - coor_cb.mean())
    coor_cnn =  (coor_cnn - coor_cnn.mean())
    plt.figure(i+40)
    axis_range_z=np.linspace(2355,2385,coor_fcn.reshape(-1).shape[0])
    plt.plot(axis_range_z, coor_self.reshape(-1),c="red",label= "DNS")
    plt.plot(axis_range_z,coor_fcn.reshape(-1),marker = "x",c="b",label= "FCN")
    plt.plot(axis_range_z,coor_vit.reshape(-1),marker = "s",c="orange",label= "VIT")
    plt.plot(axis_range_z,coor_cb.reshape(-1),marker = "v",c="g",label= "CBAM")
    plt.plot(axis_range_z,coor_cnn.reshape(-1),marker = "+",c="cyan",label= "Simple FCN")
    plt.xticks(np.round(axis_range_z[::5]))
    plt.xlabel(r"${x}^{+}$",fontdict={"size":16})
    plt.ylabel(r"${R_{R,G}}$",fontdict={"size":16})
    plt.legend()
    plt.title(f"Pr={PR[idx]}",fontdict={"size":18})
    
    plt.savefig(save_fig+f"X_corr_vs_grad_Pr{prs[idx]}_{VAR[no_var]}_nocorr",bbox_inches = "tight",dpi=200)
# %%
#%%
# coor_fcn = correlate2d(fcn_0025[:,10:11],d2u_reg[:,10:11],mode="full",boundary="circular")
# coor_vit = correlate2d(vit_0025[:,10:11],d2u_reg[:,10:11],mode="full",boundary="circular")
# coor_cb = correlate2d(cb_0025[:,10:11],d2u_reg[:,10:11],mode="full",boundary="circular")
# coor_cnn = correlate2d(cnn_0025[:,10:11],d2u_reg[:,10:11],mode="full",boundary="circular")
# coor_self = correlate2d(d2u_reg[:,10:11],d2u_reg[:,10:11],mode="full",boundary="circular")
# # coor_self = d2u_reg[:,10:11]
# # coor_fcn =  (coor_fcn - coor_self.min())/(coor_self.max()-coor_self.min())
# # coor_vit =  (coor_vit - coor_self.min())/(coor_self.max()-coor_self.min())
# # coor_cb =  (coor_cb - coor_self.min())/(coor_self.max()-coor_self.min())
# # coor_cnn =  (coor_cnn - coor_self.min())/(coor_self.max()-coor_self.min())
# # coor_self =  (coor_self - coor_fcn.min())/(coor_fcn.max()-coor_fcn.min())
# # coor_self = coor_self[:,10:11]
# plt.figure(1)

# plt.plot(coor_fcn.reshape(-1),marker = "x",c="b",label= "FCN")
# plt.plot(coor_vit.reshape(-1),marker = "s",c="orange",label= "VIT")
# plt.plot(coor_cb.reshape(-1),marker = "v",c="g",label= "CBAM")
# plt.plot(coor_cnn.reshape(-1),marker = "+",c="cyan",label= "Simple FCN")
# plt.plot(coor_self.reshape(-1),c="red",label= "DNS")
# plt.legend()
# %%
#%%
# coor_fcn = correlate2d(fcn_0025,d2u_reg,mode="full",boundary="wrap")
# coor_vit = correlate2d(vit_0025,d2u_reg,mode="full",boundary="wrap")
# coor_cb = correlate2d(cb_0025,d2u_reg,mode="full",boundary="wrap")
# coor_cnn = correlate2d(cnn_0025,d2u_reg,mode="full",boundary="wrap")
# coor_self = correlate2d(d2u_reg,d2u_reg,mode="full",boundary="wrap")

# coor_fcn =  (coor_fcn - coor_fcn.min())/(coor_fcn.max()-coor_fcn.min())
# coor_vit =  (coor_vit - coor_vit.min())/(coor_vit.max()-coor_vit.min())
# coor_cb =  (coor_cb - coor_cb.min())/(coor_cb.max()-coor_cb.min())
# coor_cnn =  (coor_cnn - coor_cnn.min())/(coor_cnn.max()-coor_cnn.min())
# coor_self =  (coor_self - coor_self.min())/(coor_self.max()-coor_self.min())
# plt.figure(0)

# plt.plot(coor_fcn[10:11,:].reshape(-1),marker = "x",c="b",label= "FCN")
# plt.plot(coor_vit[10:11,:].reshape(-1),marker = "s",c="orange",label= "VIT")
# plt.plot(coor_cb[10:11,:].reshape(-1),marker = "v",c="g",label= "CBAM")
# plt.plot(coor_cnn[10:11,:].reshape(-1),marker = "+",c="cyan",label= "Simple FCN")
# plt.plot(coor_self[10:11,:].reshape(-1),c="red",label= "DNS")
# plt.legend()
# #%%
# plt.figure(0)

# plt.plot(coor_fcn[:,10:11].reshape(-1),marker = "x",c="b",label= "FCN")
# plt.plot(coor_vit[:,10:11].reshape(-1),marker = "s",c="orange",label= "VIT")
# plt.plot(coor_cb[:,10:11].reshape(-1),marker = "v",c="g",label= "CBAM")
# plt.plot(coor_cnn[:,10:11].reshape(-1),marker = "+",c="cyan",label= "Simple FCN")
# plt.plot(coor_self[:,10:11].reshape(-1),c="red",label= "DNS")
# plt.legend()