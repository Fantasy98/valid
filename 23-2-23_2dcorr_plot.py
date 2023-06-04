#%%
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.signal import correlate2d
import os
from tqdm import tqdm
#%%
save_fig = "fig/23-2-23/"
names = ["vit_16h_4l","fcn","cbam","cnn"]

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
# %%
def Plot_Gradient_Map(U,names,save_dir):
    from matplotlib import cbook
    from matplotlib import cm
    from matplotlib.colors import LightSource
    import matplotlib.pyplot as plt
    import numpy as np

    Re_Tau = 395 #Direct from simulation
    Re = 10400 #Direct from simulation
    nu = 1/Re #Kinematic viscosity
    u_tau = Re_Tau*nu

    # xx, yy = np.mgrid[0:256:256j, 0:256:256j]
    xx, yy = np.mgrid[0:256:20j, 0:256:20j]
    # xx, yy = np.mgrid[-10:10:20j, -10:10:20j]


    x_range=6
    z_range=6

    # gridpoints_x=int(255)+1
    gridpoints_x=int(10)+1
    # gridpoints_z=int(255)+1
    gridpoints_z=int(10)+1


    x_plus_max=x_range*u_tau/nu
    z_plus_max=z_range*u_tau/nu


    x_plus_max=np.round(x_plus_max).astype(int)
    z_plus_max=np.round(z_plus_max).astype(int)

    axis_range_x=np.array([1900,2850])
    # axis_range_x=np.array([2400,2850])
    
    axis_range_z=np.array([950,1420])
    # axis_range_z=np.array([1050,1220])
    # axis_range_x=np.array([0,950,1900,2850,3980,4740])
    # axis_range_z=np.array([0,470,950,1420,1900,2370])


    placement_x=axis_range_x*nu/u_tau
    placement_x=np.round((placement_x-0)/(x_range-0)*(gridpoints_x-0)).astype(int)
    
    placement_z=axis_range_z*nu/u_tau
    placement_z=np.round((placement_z-0)/(z_range-0)*(gridpoints_z-0)).astype(int)
    
    cms = 1/2.54

    # Set up plot
    fig,axes = plt.subplots(2,2,figsize=(20*cms,36*cms),dpi=300,
                            # sharex=True,sharey=True,
                            subplot_kw=dict(projection='3d'))
    axes = axes.flatten()
    plt.subplots_adjust(hspace=15,wspace=0.2,bottom=0.1)
    ls = LightSource(270, 45)
    for i in range(len(names)):
        # surf = axes[i].plot_surface(xx[110:-110,110:-110], yy[110:-110,110:-110], U[i,110:-110,110:-110], rstride=2, cstride=2,cmap="jet",
        #                         linewidth=1, antialiased=True, shade=True,vmax = U[:,110:-110,110:-110].max(),vmin=U[:,115:-115,115:-115].min())
        surf = axes[i].plot_surface(xx,xx.T, U[i,:,:], 
                                    # rstride=2, cstride=2,
                                    cmap="jet",
                                    
                                    linewidth=2, antialiased=False, shade=True,
                                    # vmax = U.max(),vmin=U.min()
                                    )
        
        plt.tight_layout()
        axes[i].set_xlabel(r'$x^+$',labelpad=5)
        axes[i].set_ylabel(r'$z^+$',labelpad=5)
            # axes[0].set_zlabel(r'$E_{RS}\ [\%]$',labelpad=0)
            # axes[0].zaxis._axinfo['label']['space_factor'] = 2.8
        axes[i].set_title(names[i],fontsize =20,loc="left")
        axes[i].set_xticks(placement_x)
        axes[i].set_xticklabels(axis_range_x)
        axes[i].set_yticks(placement_z)
        axes[i].set_yticklabels(axis_range_z)
        axes[i].set_box_aspect((10,10,7))
        # ax.view_init(30, 140)
        axes[i].view_init(25,-75)
        axes[i].grid(b = None)
        axes[i].axis("off")
    cbar =fig.colorbar(surf,ax=axes.flatten().tolist(),aspect =30,shrink=0.9,orientation="horizontal",location="bottom")
    # cbar.formatter.set_powerlimits((0,0))
    fig.savefig(save_dir,bbox_inches="tight") 
#%%
for idx in range(4):
    Plot_Gradient_Map(FeatureCorr[idx][:,118:138,118:138],["U","V","W",r"${\theta}$"],f"pr={prs[idx]}")
# %%
for idx in range(4):
    U = np.array([FeatureCorr[i][idx,118:138,118:138]for i in range(4)])
    Plot_Gradient_Map(U,["Pr=0.025","Pr=0.2","Pr=0.71","Pr=1"],f"pr={prs[idx]}")

# %%
