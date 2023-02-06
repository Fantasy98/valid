#%%
import torch 
from torch.utils.data import DataLoader
import numpy as np 
from matplotlib import pyplot as plt 
from utils.datas import slice_dir
from utils.CBAM import CBAM
from utils.plots import Surface_Plot
from scipy import stats

from torch.nn import functional as F
device = ("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 1
var=['u_vel',"v_vel","w_vel","pr0.025"]
target=['pr0.025_flux']
normalized=False
y_plus=30
save_types= ["train","test","validation"]
root_path = "/home/yuning/thesis/tensor"
test_path = slice_dir(root_path,y_plus,var,target,"test",normalized)
print(f"Data will be loaded from {test_path}")

test_dl = DataLoader(torch.load(test_path+"/test1.pt"),batch_size=batch_size, shuffle=True)
model_name = "y_plus_30_InOut2_Epoch=100"
model_name = "y_plus_30_fullskipCCT_1layer_2epoch_1batch"

model = torch.load("/home/yuning/thesis/valid/models/23-2-1_{}.pt".format(model_name))
# %%

model.cuda()
Sal = []
for batch in test_dl:
    x,y = batch
    x.requires_grad= True
    pred = model(x.float().cuda())
    for i in range(6):
        for j in range(6):
            mean_pred = pred[:,:,125+i,125+j]
        # mean_pred = pred.mean()
            mean_pred.backward(retain_graph = True)
            saliency = x.grad[0]
        
            Sal.append(saliency)
    break

# %%
Saliency = np.array([i.numpy() for i in Sal])

# %%
var=['u_vel',"v_vel","w_vel","pr0025"]
for i in range(len(var)):
    u = Saliency.mean(0)[i,28:86,28:86]
    u = (u -u.mean())/(u.std())
    feature = x[0,i,28:86,28:86].squeeze().detach().numpy()
    feature = (feature-feature.mean())/(feature.std())
    to_show = 0.1*feature + u
    clb =plt.imshow(to_show,"jet")
    plt.colorbar(clb)

    # Surface_Plot((u-u.min())/(u.max()-u.min()))
    plt.savefig("/home/yuning/thesis/valid/fig/23-2-3/grad_{}_mid_{}".format(var[i],model_name),bbox_inches = "tight")
    plt.clf()
    print(f"Grad map of {var[i]} has been saved",flush=True)
    print(f"Mean Gradient of feature is {u.mean()}")
# z = u 
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

axis_range_x=np.array([1900,2850])
axis_range_z=np.array([950,1420])


placement_x=axis_range_x*nu/u_tau
placement_x=np.round((placement_x-0)/(x_range-0)*(gridpoints_x-0)).astype(int)


placement_z=axis_range_z*nu/u_tau
placement_z=np.round((placement_z-0)/(z_range-0)*(gridponts_z-0)).astype(int)

cms = 1/2.54

    # Set up plot
fig = plt.figure(6,figsize=(15*cms,10*cms),dpi=500)
    # fig,ax = plt.subplots(subplot_kw=dict(projection='3d'))
ax = plt.axes(projection="3d")
mappable = cm.ScalarMappable(cmap=cm.jet)
mappable.set_array(z)
    # ax = fig.gca(projection='3d')
ls = LightSource(270, 45)
    
surf = ax.plot_surface(xx[60:-60,60:-60], yy[60:-60,60:-60], z[60:-60,60:-60], rstride=2, cstride=2,cmap=mappable.cmap,
                        linewidth=1, antialiased=False, shade=True)
plt.colorbar(surf,pad = 0.18)
plt.tight_layout()
ax.set_xlabel(r'$x^+$',labelpad=10)
ax.set_ylabel(r'$z^+$',labelpad=5)
    # ax.set_zlabel(r'$E_{RS}\ [\%]$',labelpad=0)
    # ax.zaxis._axinfo['label']['space_factor'] = 2.8
ax.set_xticks(placement_x)
ax.set_xticklabels(axis_range_x)
ax.set_yticks(placement_z)
ax.set_yticklabels(axis_range_z)
ax.set_box_aspect((2,1,1))
# ax.view_init(30, 140)
ax.view_init(15,-70)

# %%
