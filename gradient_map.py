#%%
import torch 
from torch.utils.data import DataLoader
import numpy as np 
from matplotlib import pyplot as plt 
from utils.datas import slice_dir,JointDataset
from transformer.FCN_CCT import FullSkip_FCN_CCT,FullSkip_Mul_FCN_CCT
from utils.networks import FCN_Pad_Xaiver_gain, FCN_4
from utils.newnets import FCN_Pad_Xaiver_CBAM2
from tqdm import tqdm
from scipy import stats

from torch.nn import functional as F
device = ("cuda" if torch.cuda.is_available() else "cpu")
y_plus = 30

model_name = "cbam"
Num_heads = 16
Num_layers = 4
if  "vit" in model_name:
    # model = FullSkip_FCN_CCT(num_heads=Num_heads,num_layers=Num_layers)
    model = FullSkip_Mul_FCN_CCT(num_heads=Num_heads,num_layers=Num_layers)
    # checkpoint_path = f"/home/yuning/thesis/valid/models/y_plus_{y_plus}-VARS-prall_u_vel_v_vel_w_vel-TARGETS-prall_flux_Skip_FCN_CCT_{Num_heads}h_{Num_layers}l_EPOCH=100.pt"
    checkpoint_path = f"/home/yuning/thesis/valid/models/y_plus_{y_plus}-VARS-prall_u_vel_v_vel_w_vel-TARGETS-prall_flux_Skip_Mul_FCN_CCT_{Num_heads}h_{Num_layers}l_EPOCH=100.pt"
elif "fcn" in model_name:
    model = FCN_Pad_Xaiver_gain(256,256,4,3,8)
    checkpoint_path = f"/home/yuning/thesis/valid/models/y_plus_{y_plus}-VARS-prall_u_vel_v_vel_w_vel-TARGETS-prall_flux_FCN_baseline_EPOCH=100.pt"
elif "cbam" in model_name:
    model = FCN_Pad_Xaiver_CBAM2(256,256,4,3,8)
    checkpoint_path = f"/home/yuning/thesis/valid/models/y_plus_{y_plus}-VARS-prall_u_vel_v_vel_w_vel-TARGETS-prall_flux_FCN_CBAM_EPOCH=100.pt"

elif "cnn" in model_name:
    model = FCN_4(256,256,4,3,8)
    checkpoint_path = f"/home/yuning/thesis/valid/models/y_plus_{y_plus}-VARS-prall_u_vel_v_vel_w_vel-TARGETS-prall_flux_FCN_4_EPOCH=100.pt"


checkpoint = torch.load(checkpoint_path)


checkpoint = torch.load(checkpoint_path)

print(checkpoint.keys())
model_state = checkpoint["model"]
loss = checkpoint["loss"]
val_loss = checkpoint["val_loss"]
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in model_state.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
# load params
model.load_state_dict(new_state_dict)
model.eval()
print(model.eval())

model.to(device)

for param in model.parameters():
    param.requires_grad = True

#%%
Prs = [0.025,0.2,0.71,1]
prs  = ["0025","02","071","1"]
# for pr, Pr in zip(prs,Prs):
pr = "0025"
Pr = 0.025

batch_size = 1
var=['u_vel',"v_vel","w_vel",f"pr{Pr}"]
target=[f'pr{Pr}_flux']
normalized=False
y_plus=30
save_types= ["train","test","validation"]
root_path = "/home/yuning/thesis/tensor"
test_path = slice_dir(root_path,y_plus,var,target,"test",normalized)
print(f"Data will be loaded from {test_path}")
    
    # if Pr == 0.025:

    #     test_data1 = torch.load(test_path+"/test1.pt")
    #     test_data2 = torch.load(test_path+"/test2.pt")


    #     test_x = torch.cat([test_data1.x,test_data2.x])
    #     test_y = torch.cat([test_data1.y,test_data2.y])
    #     test_ds = JointDataset(test_x,test_y)

    #     # test_ds = torch.load(test_path+"/test1.pt")

    #     test_dl = DataLoader(test_ds,shuffle=False,batch_size=1)
    # else:
test_ds = torch.load(test_path+"/test1.pt")

test_dl = DataLoader(test_ds,shuffle=False,batch_size=1)
#%%
from torch.autograd import grad,Variable
Sal = []
# for batch in tqdm(test_dl):
batch = next(iter(test_dl))
x,y = batch

# x.requires_grad= True
# x = (x -torch.mean(x))/torch.std(x)
x = x.float().to(device)

x.requires_grad_()

output = model(x)

# Select the area to compute the saliency map
# selected_area = output[0, 0, 128:129, 128:129].sum()
selected_area = output[0, 0, 127:128, 127:128].squeeze()

# Calculate gradients using guided backpropagation
# model.zero_grad()

selected_area.backward()  # Set the gradients of the selected area to 1

print(x.grad)
print(selected_area.grad)
print(output.grad)

#%%
xx, zz = np.mgrid[0:4740:256j , 0:2370:256j]

axis_range_x=np.linspace(0,4740,256)
axis_range_z=np.linspace(0,2370,256)
# axis_range_z=np.array([0,470,950,1420,1900,2370])
axis_x=np.array([0,950,1900,2850,3980,4740])
axis_z=np.array([0,470,950,1420,1900,2370])
#%%
fig, axs = plt.subplots(4,1,figsize=(12,8),sharex=True)
for i in range(4):
    cb = axs[i].contourf(xx[128-5:128+6,128-5:128+6],
                  zz[128-5:128+6,128-5:128+6],
                  x.grad[0,i,128-5:128+6,128-5:128+6].cpu(),
                  cmap="RdBu_r",
                  levels=250,
                  )

# cb = axs.plot_surface(xx[128-5:128+5,128-5:128+5],
#                   zz[128-5:128+5,128-5:128+5],
#                     x.grad[0,0,128-5:128+5,128-5:128+5].cpu(),
#                     cstride = 1,rstride=1, 
#                     vmax = x.grad[0,0,128-5:128+5,128-5:128+5].cpu().max(),
#                     vmin = x.grad[0,0,128-5:128+5,128-5:128+5].cpu().max(),
#                     cmap="RdBu_r", 
                  
                #   levels=250,
                #   )
    bx = [ xx[128,128], xx[128,128]]

    bx2 = [ xx[128-5,128], xx[128+5,128]]

    bz = [ zz[128,128-5], zz[128,128+5]] 

    bz2=[zz[128,128], zz[128,128] ]
    
    axs[i].plot(bx,bz,"-k",lw=1)
    axs[i].plot(bx2,bz2,"-k",lw=1)
    axs[i].set_xlabel(r"$x^+$")
    axs[i].set_ylabel(r"$z^+$")
# cb = axs.contourf(xx,zz,x.grad[0,1,:,:].cpu(),cmap="RdBu_r",levels=100)
    # axs.set_xticks(axis_range_x)
    # axs.set_yticks(axis_range_z)

# axs.set_xticklabels(axis_x)
# axs.set_yticklabels(axis_z)

    axs[i].set_aspect("equal")

    plt.colorbar(cb,ax=axs[i])
plt.savefig( model_name+"_grad")
# p = p.requires_grad_(True)
#%%

#
# smap =torch.abs(selected_area.grad)

# sp = smap.cpu()
#%%
        # for i in range(6):
        #     for j in range(6):
        #         mean_pred = pred[:,:,115+i,115+j]
        # mean_pred = pred[:,:,128,128]
        # mean_pred = (pred**2).sqrt().mean()
    #     mean_pred.backward()
    #     saliency = x.grad[0]
    #     Sal.append(saliency)


    # Saliency = np.array([i.numpy() for i in Sal])
    # print(Saliency.shape)
    # np.savez_compressed(f"pred/y{y_plus}_pr{pr}_{model_name}_gradmap",
    #                     gradmap = Saliency)
    # print("INFO: Gradient map has been saved!")




#%%
    var=['u_vel',"v_vel","w_vel","pr0025"]
    U = np.empty(shape=(4,20,20))
    for i in range(len(var)):
        uu = Saliency.mean(0)
        u_sum = uu.sum()
        # u = (u -u.mean())/(u.std())
        # feature = x[0,i,28:86,28:86].squeeze().detach().numpy()
        # feature = (feature-feature.mean())/(feature.std())
        # to_show = 0.1*feature + u
        # clb =plt.imshow(to_show,"jet")
        # plt.colorbar(clb)
        # u = (u-u.min())/(u.max()-u.min())
        u = uu[i,118:138,118:138]
        # u = (u-u.mean())/(u.std())
        u = (u-uu.min())/(uu.max()-uu.min())
        U[i,:,:] = u
        # U = (U - U.min())/(U.max()-U.min())
        # Plot_2D_snapshots(u,"/home/yuning/thesis/valid/fig/23-2-6/grad_{}_rms_{}".format(var[i],model_name))
        # Surface_Plot((u-u.min())/(u.max()-u.min()))

        # plt.savefig("/home/yuning/thesis/valid/fig/23-2-6/grad_{}_3Drms_{}".format(var[i],model_name))
        # plt.clf()
        
        print(f"Grad map of {var[i]} has been saved",flush=True)
        print(f"Mean Gradient of feature is {u.mean()}")
# z = u 
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

# %%
Plot_Gradient_Map(U,["u","v","w",r"$\theta$"],f"23-2-18_{model_name}_y30_{pr}_reg")
# %%
