import numpy as np 
from matplotlib import cbook
from matplotlib import cm
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

save_fig = "fig/23-2-24/"
names = ["vit_mul","fcn","cbam","cnn"]

y_plus = 15
prs = ["0025","02","071","1"]
PR = [0.025,0.2,0.71,1]
model_names = ["ViT","FCN","CBAM","Simple FCN"]

no_var = 3
fluct = True
VAR =  ["U","V","W","T"]

ALL_PR = []
for pr in prs:
    grad_dict = {}
    for name in names:


        
        # T = []
    
        data_dir = f"pred/y{y_plus}_pr{pr}_{name}_gradmap.npz"
        grad_map = np.load(data_dir)
        gdmp = grad_map["gradmap"]
        
        if fluct:    # Noise cancelling 
            for i in range(4):
                gdmp[:,i,:,:] = np.sqrt(gdmp[:,i,:,:]**2) - gdmp[:,i,:,:].mean()
        
        gdmp = gdmp.mean(0)
        u = gdmp[no_var,118:138,118:138] 
        # v = gdmp[1,118:138,118:138] 
        # w = gdmp[2,118:138,118:138] 
        # t = gdmp[3,118:138,118:138] 

        # u = gdmp[0,103:153,103:153]
        # v = gdmp[1,103:153,103:153]
        # w = gdmp[2,103:153,103:153]
        # t = gdmp[3,103:153,103:153]

        # Regularize to 0 ~ 1
        u = (u-gdmp.min())/(gdmp.max()-gdmp.min())
        # v = (v-gdmp.min())/(gdmp.max()-gdmp.min())
        # w = (w-gdmp.min())/(gdmp.max()-gdmp.min())
        # t = (t-gdmp.min())/(gdmp.max()-gdmp.min())
        grad_dict[name] = u
    ALL_PR.append(grad_dict)



print(len(ALL_PR))


Re_Tau = 395 #Direct from simulation
Re = 10400 #Direct from simulation
nu = 1/Re #Kinematic viscosity
u_tau = Re_Tau*nu
xx, yy = np.mgrid[-10:10:20j, -10:10:20j]

x_range=6
z_range=6

gridpoints_x=int(10)+1
gridpoints_z=int(10)+1

x_plus_max=x_range*u_tau/nu
z_plus_max=z_range*u_tau/nu


x_plus_max=np.round(x_plus_max).astype(int)
z_plus_max=np.round(z_plus_max).astype(int)

axis_range_x=np.array([1900,2850])


axis_range_z=np.array([950,1420])

placement_x=axis_range_x*nu/u_tau
placement_x=np.round((placement_x-0)/(x_range-0)*(gridpoints_x-0)).astype(int)

placement_z=axis_range_z*nu/u_tau
placement_z=np.round((placement_z-0)/(z_range-0)*(gridpoints_z-0)).astype(int)

cms = 1/2.54

# Set up plot
fig,axes = plt.subplots(4,4,figsize=(20*cms,36*cms),dpi=300,
                            sharex=True,sharey=True,
                            subplot_kw=dict(projection='3d'))
# axes = axes.flatten()
plt.subplots_adjust(hspace=15,wspace=1)
# plt.subplots_adjust(hspace=0.5)
ls = LightSource(270, 45)
for row in range(len(prs)):
    for col in range(len(names)):
    # surf = axes[i].plot_surface(xx[110:-110,110:-110], yy[110:-110,110:-110], U[i,110:-110,110:-110], rstride=2, cstride=2,cmap="jet",
    #                         linewidth=1, antialiased=True, shade=True,vmax = U[:,110:-110,110:-110].max(),vmin=U[:,115:-115,115:-115].min())
        umin = ALL_PR[row][names[col]].min()
        umax = ALL_PR[row][names[col]].max()
        surf = axes[row,col].plot_surface(xx,xx.T, ALL_PR[row][names[col]], 
                                # rstride=2, cstride=2,
                                    cmap="jet",
                                    
                                    linewidth=2, antialiased=False, shade=True,
                                    vmax = umax,vmin=umin
                                    )
        
        axes[row,col].set_xlabel(r'$x^+$',labelpad=5)
        axes[row,col].set_ylabel(r'$z^+$',labelpad=5)
                # axes[0].set_zlabel(r'$E_{RS}\ [\%]$',labelpad=0)
                # axes[0].zaxis._axinfo['label']['space_factor'] = 2.8
        axes[row,col].set_title(model_names[col],fontsize =20,loc="right")
        axes[row,col].set_xticks(placement_x)
        axes[row,col].set_xticklabels(axis_range_x)
        axes[row,col].set_yticks(placement_z)
        axes[row,col].set_yticklabels(axis_range_z)
        axes[row,col].set_box_aspect((10,10,7))
            # ax.view_init(30, 140)
        axes[row,col].view_init(25,-75)
        axes[row,col].grid(b = None)
        axes[row,col].axis("off")
    axins = inset_axes(
                            axes[row,col],
                            width="5%",  # width: 5% of parent_bbox width
                            height="50%",  # height: 50%
                            loc="center left",
                            bbox_to_anchor=(1.05, 0., 1, 1),
                            bbox_transform=axes[row,col].transAxes,
                            borderpad=0,
                        )
    fig.colorbar(surf, cax=axins,ticks = np.linspace(umax,umin,3))
plt.tight_layout()
    
 
# position for the colorbar
    # fig.colorbar(surf,ax=axes[row,:],aspect =30,shrink=0.9,orientation="vertical",location="right")
for i in range(4):
    # y_plus=[15,30,50,75]
    # pr=[0.025,0.2,0.71,1]
    # axes[3].set_xlabel(r'$\lambda_x^+$')
    # axes[0].set_ylabel(r'$\lambda_z^+$')
    axes[i,0].set_title(r'$Pr=$'+ str(PR[i]),fontsize =20,loc="left")
    # ax2 = axes[i,-1].twinx()
    # ax2.set_ylabel(r'$k_x\ k_z\ \phi_{q_w}$' + "\n" + r'Pr = '+str(pr[i]),fontsize=9,linespacing=2)
    # ax2.get_yaxis().set_ticks([])      
# cbar =fig.colorbar(surf,ax=axes.flatten().tolist(),aspect =30,shrink=0.9,orientation="horizontal",location="bottom")
# cbar.formatter.set_powerlimits((0,0))
if fluct:
    fig.savefig(save_fig+f"y{y_plus}_{VAR[no_var]}_fluct_all",bbox_inches="tight")
else:
    fig.savefig(save_fig+f"y{y_plus}_{VAR[no_var]}_all",bbox_inches="tight")