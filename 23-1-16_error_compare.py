#%%
import numpy as np 
import matplotlib.pyplot as plt
from utils.metrics import ERS

pred_cbam = np.load("/home/yuning/thesis/valid/pred/CBAM2_EPOCH=100/pred.npy")
pred_base = np.load("/home/yuning/thesis/valid/pred/EPOCH=100/pred.npy")
y_cbam = np.load("/home/yuning/thesis/valid/pred/CBAM2_EPOCH=100/y.npy")
y_base = np.load("/home/yuning/thesis/valid/pred/EPOCH=100/y.npy")
#%%

cbam_pred_snap = pred_cbam[10,:,:]
base_pred_snap = pred_base[10,:,:]
cbam_y_snap = y_cbam[10,:,:]
base_y_snap = y_base[10,:,:]

cbam_pred_mean = np.mean(pred_cbam,axis=0)
base_pred_mean = np.mean(pred_base,axis=0)
cbam_y_mean = np.mean(y_cbam,axis=0)
base_y_mean = np.mean(y_base,axis=0)
# %%

cbam_RS_mean = ERS(cbam_pred_mean,cbam_y_mean)

base_RS_mean = ERS(base_pred_mean,base_y_mean)

# %%
from matplotlib import cbook
from matplotlib import cm
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt
import numpy as np

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

cms = 1/2.54
#%%
# Set up plot
fig = plt.figure(6,figsize=(15*cms,10*cms),dpi=500)
# fig,ax = plt.subplots(subplot_kw=dict(projection='3d'))
ax = plt.axes(projection="3d")
mappable = cm.ScalarMappable(cmap=cm.jet)
mappable.set_array(cbam_RS_mean)
# ax = fig.gca(projection='3d')
ls = LightSource(270, 45)
# To use a custom hillshading mode, override the built-in shading and pass
# in the rgb colors of the shaded surface calculated from "shade".
# rgb = ls.shade(cbam_RS_mean, cmap=mappable.cmap, vert_exag=0.1, blend_mode='soft')
surf = ax.plot_surface(xx, yy, cbam_RS_mean, rstride=1, cstride=1,cmap=mappable.cmap,
                       linewidth=0, antialiased=False, shade=False)
plt.colorbar(surf,pad = 0.18)
plt.tight_layout()
ax.set_xlabel(r'$x^+$',labelpad=10)
ax.set_ylabel(r'$z^+$',labelpad=5)
ax.set_zlabel(r'$E_{RS}\ [\%]$',labelpad=0)
# ax.zaxis._axinfo['label']['space_factor'] = 2.8
ax.set_xticks(placement_x)
ax.set_xticklabels(axis_range_x)
ax.set_yticks(placement_z)
ax.set_yticklabels(axis_range_z)
ax.set_box_aspect((2,1,1))
ax.set_title("RSE of ensemble average prediction by Attention Mechanism")


ax.view_init(30, 140)
# plt.savefig("RSE_CBAM",bbox_inches="tight")
# plt.colorbar(surf)
# %%
fig = plt.figure(7,figsize=(15*cms,10*cms),dpi=500)
# fig,ax = plt.subplots(subplot_kw=dict(projection='3d'))
ax = plt.axes(projection="3d")
mappable = cm.ScalarMappable(cmap=cm.jet)
mappable.set_array(base_RS_mean)
# ax = fig.gca(projection='3d')
ls = LightSource(270, 45)
# To use a custom hillshading mode, override the built-in shading and pass
# in the rgb colors of the shaded surface calculated from "shade".
# rgb = ls.shade(base_RS_mean, cmap=mappable.cmap, vert_exag=0.1, blend_mode='soft')
surf = ax.plot_surface(xx, yy, base_RS_mean, rstride=1, cstride=1,cmap=mappable.cmap,
                       linewidth=0, antialiased=False, shade=False)
plt.colorbar(surf,pad =0.18)
plt.tight_layout()
ax.set_xlabel(r'$x^+$',labelpad=10)
ax.set_ylabel(r'$z^+$',labelpad=5)
ax.set_zlabel(r'$E_{RS}\ [\%]$',labelpad=0)
# ax.zaxis._axinfo['label']['space_factor'] = 2.8
ax.set_xticks(placement_x)
ax.set_xticklabels(axis_range_x)
ax.set_yticks(placement_z)
ax.set_yticklabels(axis_range_z)
ax.set_box_aspect((2,1,1))
ax.set_title("RSE of ensemble average prediction by Baseline model")
ax.view_init(30, 140)
# 
# plt.savefig("RSE_Baseline",bbox_inches="tight")
# %%
loc = 15
cbam_pred_snap = pred_cbam[loc,:,:]
base_pred_snap = pred_base[loc,:,:]
cbam_y_snap = y_cbam[loc,:,:]
base_y_snap = y_base[loc,:,:]
cbam_RS = ERS(cbam_pred_snap,cbam_y_snap)
base_RS = ERS(base_pred_snap,base_y_snap)

# %%
fig = plt.figure(10,figsize=(15*cms,10*cms),dpi=500)
# fig,ax = plt.subplots(subplot_kw=dict(projection='3d'))
ax = plt.axes(projection="3d")
mappable = cm.ScalarMappable(cmap=cm.jet)
mappable.set_array(cbam_RS)
# ax = fig.gca(projection='3d')
ls = LightSource(270, 45)
# To use a custom hillshading mode, override the built-in shading and pass
# in the rgb colors of the shaded surface calculated from "shade".
# rgb = ls.shade(cbam_RS, cmap=mappable.cmap, vert_exag=0.1, blend_mode='soft')
surf = ax.plot_surface(xx, yy, cbam_RS, rstride=1, cstride=1,cmap=mappable.cmap,
                       linewidth=0, antialiased=False, shade=False)
plt.colorbar(surf,pad = 0.18)
plt.tight_layout()
ax.set_xlabel(r'$x^+$',labelpad=10)
ax.set_ylabel(r'$z^+$',labelpad=5)
ax.set_zlabel(r'$E_{RS}\ [\%]$',labelpad=0)
# ax.zaxis._axinfo['label']['space_factor'] = 2.8
ax.set_xticks(placement_x)
ax.set_xticklabels(axis_range_x)
ax.set_yticks(placement_z)
ax.set_yticklabels(axis_range_z)
ax.set_box_aspect((2,1,1))
ax.set_title("RSE of prediction snapshot by Attention Mechanism")


ax.view_init(30, 140)
# plt.savefig("RSE_CBAM_snap",bbox_inches="tight")
# plt.colorbar(surf)
# %%
fig = plt.figure(11,figsize=(15*cms,10*cms),dpi=500)
# fig,ax = plt.subplots(subplot_kw=dict(projection='3d'))
ax = plt.axes(projection="3d")
mappable = cm.ScalarMappable(cmap=cm.jet)
mappable.set_array(base_RS)
# ax = fig.gca(projection='3d')
ls = LightSource(270, 45)
# To use a custom hillshading mode, override the built-in shading and pass
# in the rgb colors of the shaded surface calculated from "shade".
# rgb = ls.shade(base_RS_mean, cmap=mappable.cmap, vert_exag=0.1, blend_mode='soft')
surf = ax.plot_surface(xx, yy, base_RS, rstride=1, cstride=1,cmap=mappable.cmap,
                       linewidth=0, antialiased=False, shade=False)
plt.colorbar(surf,pad =0.18)
plt.tight_layout()
ax.set_xlabel(r'$x^+$',labelpad=10)
ax.set_ylabel(r'$z^+$',labelpad=5)
ax.set_zlabel(r'$E_{RS}\ [\%]$',labelpad=0)
# ax.zaxis._axinfo['label']['space_factor'] = 2.8
ax.set_xticks(placement_x)
ax.set_xticklabels(axis_range_x)
ax.set_yticks(placement_z)
ax.set_yticklabels(axis_range_z)
ax.set_box_aspect((2,1,1))
ax.set_title("RSE of prediction snapshot by Baseline model")
ax.view_init(30, 140)
# plt.savefig("RSE_Baseline_snap",bbox_inches="tight")
# %%
import seaborn as sns
