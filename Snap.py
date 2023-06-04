import numpy as np
from utils.plots import Plot_2D_2snapshots,Snap_Plot3D,Plot_multi,Plot_2D_snapshots,PSD_single
from utils.metrics import ERS,PCC, RMS_error,Glob_error,Fluct_error
import matplotlib.pyplot as plt 
import torch 
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import argparse

parser = argparse.ArgumentParser(description='Setting Parameters')
parser.add_argument('--y', default= 30, type=int, help='Wall normal distance')
parser.add_argument('--pr', default= "0025", type=str, help='Wall normal distance')
parser.add_argument('--type', default= "snap", type=str, help='Snapshot or error')
args = parser.parse_args()


y_plus = args.y
pr = args.pr

fig_path = "/home/yuning/thesis/valid/fig/23-3-14/"

dvit = np.load(f"pred/y{y_plus}_vit_16h_4l_pr{pr}.npz")
y_vit = dvit["y"]
y_vit_f = y_vit - y_vit.mean()
ers_y = ERS(y_vit[0,:,:],y_vit[0,:,:])
ers_y = ers_y.reshape(1,256,256)


p_vit = dvit["pred"]
p_vit_f = p_vit - p_vit.mean()
# rms_vit = np.sqrt(np.mean(p_vit**2))
ers_vit = ERS(p_vit[0,:,:],y_vit[0,:,:])
ers_vit = ers_vit.reshape(1,256,256)

# rms_y = np.sqrt(np.mean(y_vit**2))
dfcn = np.load(f"pred/y{y_plus}_fcn_pr{pr}.npz")
p_fcn = dfcn["pred"]
p_fcn_f = p_fcn - p_fcn.mean()
# rms_fcn = np.sqrt(np.mean(p_fcn**2))
ers_fcn = ERS(p_fcn[0,:,:],y_vit[0,:,:])
ers_fcn = ers_fcn.reshape(1,256,256)


dcb = np.load(f"pred/y{y_plus}_cbam_pr{pr}.npz")
p_cb = dcb["pred"]
p_cb_f = p_cb - p_cb.mean()
# rms_cb = np.sqrt(np.mean(p_cb**2))
ers_cb = ERS(p_cb[0,:,:],y_vit[0,:,:])
ers_cb = ers_cb.reshape(1,256,256)

dcnn = np.load(f"pred/y{y_plus}_cnn_pr{pr}.npz")
p_cnn = dcb["pred"]
p_cnn_f = p_cnn - p_cnn.mean()
# rms_cb = np.sqrt(np.mean(p_cb**2))
ers_cnn = ERS(p_cnn[0,:,:],y_vit[0,:,:])
ers_cnn = ers_cnn.reshape(1,256,256)

if args.type == "snap":
    avg = np.concatenate([
                        y_vit[0:1,:,:],
                        p_vit[0:1,:,:],
                        p_fcn[0:1,:,:],
                        p_cb[0:1,:,:],
                        p_cnn[0:1,:,:]
                        # ers_y,
                        # ers_vit,
                        # ers_fcn,
                        # ers_cb,
                        # ers_cnn
                        
                        
                        ])
    names = ["DNS","VIT","FCN","CBAM","Simple FCN"]
    cmap = "RdBu"

else:
        avg = np.concatenate([
                        ers_y,
                        ers_vit,
                        ers_fcn,
                        ers_cb,
                        ers_cnn
                        
                        
                        ])
        names = ["DNS Error","VIT Error","FCN Error","CBAM Error","Simple FCN Error"]
        cmap="cmo.curl"

print(avg.shape)



Re_Tau = 395 #Direct from simulation
Re = 10400 #Direct from simulation
nu = 1/Re #Kinematic viscosity
u_tau = Re_Tau*nu

xx, yy = np.mgrid[0:256:256j, 0:256:256j]

x_range=12
z_range=6

gridpoints_x=int(255)+1
gridponts_z=int(255)+1


x_plus_max=x_range*u_tau/nu
z_plus_max=z_range*u_tau/nu


x_plus_max=np.round(x_plus_max).astype(int)
z_plus_max=np.round(z_plus_max).astype(int)

axis_range_x=np.array([0,950,1900,2850,3980,4740])
axis_range_z=np.array([0,470,950,1420,1900,2370])


placement_x=axis_range_x*nu/u_tau
placement_x=np.round((placement_x-0)/(x_range-0)*(gridpoints_x-0)).astype(int)

placement_z=axis_range_z*nu/u_tau
placement_z=np.round((placement_z-0)/(z_range-0)*(gridponts_z-0)).astype(int)

cm =1/2.54
fig,axes = plt.subplots(2,3,figsize=(6*(len(names))*cm,4*(len(names)+1)*cm),
                                sharex=True,sharey=True,
                                dpi=300)
min_ve = avg[0,:,:].max()
max_ve = avg[0,:,:].min()
for i in range(len(names)):
    min_now = avg[i,:,:].min()
    max_now = avg[i,:,:].max()
    if min_now <= min_ve:
        min_ve = min_now
    if max_now >=max_ve:
        max_ve=max_now
        # names = ["u","v","w",r"$\theta$"]
# axes = axes.flatten()
for i in range(len(names)):
            row,col = divmod(i,3)
            print(row,col)
            clb=axes[row,col].imshow((avg[i,:,:]),
                                 cmap=cmap,
                                #  ,
                                 aspect=0.5,
                                 vmax=max_ve,vmin=min_ve,
                                #  levels= 30
                                 )
            
            axes[row,col].set_xlabel(r'$x^+$',fontdict={"size":15})
            axes[row,col].set_ylabel(r'$z^+$',fontdict={"size":15})
            axes[row,col].set_title(names[i],fontdict={"size":16,"weight":"bold"})
            axes[row,col].set_xticks(placement_x)
            axes[row,col].set_yticks(placement_z)
            axes[row,col].set_xticklabels(axis_range_x)
            axes[row,col].set_yticklabels(axis_range_z)

axes[-1,-1].axis("off")
axins = inset_axes(
                            axes[row,col],
                            width="5%",  # width: 5% of parent_bbox width
                            height="50%",  # height: 50%
                            loc="center left",
                            bbox_to_anchor=(1.05, 0., 1, 1),
                            bbox_transform=axes[row,col].transAxes,
                            borderpad=0,
                        )
fig.colorbar(clb, cax=axins,ticks = np.linspace(max_ve,min_ve,5),label="%")
            
# cbar =fig.colorbar(clb,ax=axes.flatten(),
#                            aspect = 25,shrink=0.9,
#                            orientation="horizontal",
#                            location = "bottom"
#                            )
# cbar.formatter.set_powerlimits((0,0))
fig.tight_layout()
        # plt.colorbar()
if args.type =="snap":
    fig.savefig(fig_path+f"Snap_y{y_plus}_pr{pr}",bbox_inches="tight")
else:
    fig.savefig(fig_path+f"Snap_y{y_plus}_pr{pr}_Error",bbox_inches="tight")