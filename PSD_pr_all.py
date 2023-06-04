#%%
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
from utils.toolbox import PSD
from tqdm import tqdm
#%%
fig_dir = "fig/23-3-14/"

y_plus = 75

DNS = []
FCN = []
CNN = []
CBAM = []
VIT = []
PR  = ["0025","02","071","1"]

for pr in tqdm(PR):

    dvit = np.load(f"pred/predictions/y{y_plus}_vit_16h_4l_pr{pr}.npz")
    p_vit = dvit["pred"]

    y_vit = dvit["y"]

    dfcn = np.load(f"pred/predictions/y{y_plus}_fcn_pr{pr}.npz")
    p_fcn = dfcn["pred"]

    dcb = np.load(f"pred/predictions/y{y_plus}_cbam_pr{pr}.npz")
    p_cb = dcb["pred"]

    dcnn = np.load(f"pred/predictions/y{y_plus}_cnn_pr{pr}.npz")
    p_cnn = dcnn["pred"]
 
    fa_y, Lambda_x,Lambda_z = PSD(y_vit)
    fa_vit,_,_ = PSD(p_vit)
    fa_fcn,_,_ = PSD(p_fcn)
    fa_cb,_,_ = PSD(p_cb)
    fa_cnn,_,_ = PSD(p_cnn)
    DNS.append(fa_y)
    VIT.append(fa_vit)
    CNN.append(fa_cnn)
    FCN.append(fa_fcn)
    CBAM.append(fa_cb)
    
    

# print(fa_y.shape, fa_vit.shape,fa_cb.shape)
print(Lambda_x.shape,Lambda_z.shape)
#%%
cm = 1/2.54  # centimeters in inches
fig, axs=plt.subplots(nrows=1,ncols=4,
                    #   figsize=([0.7*21*cm,0.7*56*cm]),
                      figsize=([0.7*56*cm,0.7*7*cm]),
                      sharey=True,sharex=True,
                      constrained_layout=False)
cmap = mpl.cm.Greys(np.linspace(0,1,20))
cmap = mpl.colors.ListedColormap(cmap[5:,:-1])



for i, dns in enumerate(DNS):
    pct10=np.max(DNS[i])*0.1
    pct50=np.max(DNS[i])*0.5
    pct90=np.max(DNS[i])*0.9
    pct100=np.max(DNS[i])*1
    # row, col = divmod(i,4)
    CP=axs[i].contourf(Lambda_x,Lambda_z,DNS[i],[pct10,pct50,pct90,pct100],cmap=cmap)
    CS=axs[i].contour(Lambda_x,Lambda_z,VIT[i],[pct10,pct50,pct90,pct100],colors='orange',linestyles='solid')
    CS=axs[i].contour(Lambda_x,Lambda_z,FCN[i],[pct10,pct50,pct90,pct100],colors='blue',linestyles='dotted')
    CS=axs[i].contour(Lambda_x,Lambda_z,CBAM[i],[pct10,pct50,pct90,pct100],colors='green',linestyles='dashdot')
    CS=axs[i].contour(Lambda_x,Lambda_z,CNN[i],[pct10,pct50,pct90,pct100],colors='cyan',linestyles='dashed',label="Simple FCN")
   

        # zero freq https://medium.com/analytics-vidhya/breaking-down-confusions-over-fast-fourier-transform-fft-1561a029b1ab
axs[i].set_xscale('log')
        #axs[i][i].set_xlim([np.min(Lambda_x[np.nonzero(Lambda_x)]),np.max(Lambda_x)])
axs[i].set_yscale('log')
        #axs[row,col].set_ylim([np.min(Lambda_z[np.nonzero(Lambda_z)]),np.max(Lambda_z)])
        #axs[row,col].minorticks_on()

fig.subplots_adjust(hspace=0.3)
for i in range(4):
    # y_plus=[15,30,50,75]
    pr=[0.025,0.2,0.71,1]
    axs[i].set_xlabel(r'$\lambda_x^+$')
    axs[0].set_ylabel(r'$\lambda_z^+$')
    # axs[0].set_title(r'$y^+=$'+str(30))
    # axs[0].legend()
    ax2 = axs[i].twinx()
    ax2.set_ylabel(r'$k_x\ k_z\ \phi_{q_w}$',fontsize=9,linespacing=2)
    ax2.get_yaxis().set_ticks([])
plt.savefig(fig_dir+f"PSD_y{y_plus}_all_legend",bbox_inches="tight")


# %%
