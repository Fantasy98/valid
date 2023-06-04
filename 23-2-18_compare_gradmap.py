import numpy as np 
save_fig = "fig/23-2-24/"
names = ["fcn","cbam","vit_mul"]

y_plus = 15
prs = ["0025","02","071","1"]

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
    # xx, yy = np.mgrid[0:256:20j, 0:256:20j]
    xx, yy = np.mgrid[-10:10:20j, -10:10:20j]
    # xx, yy = np.mgrid[-25:25:50j, -25:25:50j]


    # x_range=12
    x_range=6
    z_range=6

    # gridpoints_x=int(255)+1
    gridpoints_x=int(10)+1
    # gridpoints_x=int(50)+1
    # gridpoints_z=int(255)+1
    gridpoints_z=int(10)+1
    # gridpoints_z=int(50)+1


    x_plus_max=x_range*u_tau/nu
    z_plus_max=z_range*u_tau/nu


    x_plus_max=np.round(x_plus_max).astype(int)
    z_plus_max=np.round(z_plus_max).astype(int)

    axis_range_x=np.array([1900,2850])
    
    
    axis_range_z=np.array([950,1420])
    # axis_range_x=np.array([0,950,1900,2850,3980,4740])
    # axis_range_z=np.array([0,470,950,1420,1900,2370])


    placement_x=axis_range_x*nu/u_tau
    placement_x=np.round((placement_x-0)/(x_range-0)*(gridpoints_x-0)).astype(int)
    
    placement_z=axis_range_z*nu/u_tau
    placement_z=np.round((placement_z-0)/(z_range-0)*(gridpoints_z-0)).astype(int)
    
    cms = 1/2.54

    # Set up plot
    fig,axes = plt.subplots(4,3,figsize=(20*cms,36*cms),dpi=300,
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

for pr in prs:
    U = []
    V = []
    W = []
    T = []
    for name in names:
        data_dir = f"pred/y{y_plus}_pr{pr}_{name}_gradmap.npz"
        grad_map = np.load(data_dir)
        gdmp = grad_map["gradmap"]
        
        # Noise cancelling of gradient map
        for i in range(4):
            gdmp[:,i,:,:] = np.sqrt(gdmp[:,i,:,:]**2) - gdmp[:,i,:,:].mean()
        
        gdmp = gdmp.mean(0)
        u = gdmp[0,118:138,118:138] 
        v = gdmp[1,118:138,118:138] 
        w = gdmp[2,118:138,118:138] 
        t = gdmp[3,118:138,118:138] 

        # u = gdmp[0,103:153,103:153]
        # v = gdmp[1,103:153,103:153]
        # w = gdmp[2,103:153,103:153]
        # t = gdmp[3,103:153,103:153]
        u = (u-gdmp.min())/(gdmp.max()-gdmp.min())
        v = (v-gdmp.min())/(gdmp.max()-gdmp.min())
        w = (w-gdmp.min())/(gdmp.max()-gdmp.min())
        t = (t-gdmp.min())/(gdmp.max()-gdmp.min())
        
        

        U.append(u)
        V.append(v)
        W.append(w)
        T.append(t)
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    T = np.array(T)
    Plot_Gradient_Map(U,["FCN","CBAM","ViT"],save_fig+f"fluct_y{y_plus}_pr{pr}_U_compare")
    Plot_Gradient_Map(V,["FCN","CBAM","ViT"],save_fig+f"fluct_y{y_plus}_pr{pr}_V_compare")
    Plot_Gradient_Map(W,["FCN","CBAM","ViT"],save_fig+f"fluct_y{y_plus}_pr{pr}_W_compare")
    Plot_Gradient_Map(T,["FCN","CBAM","ViT"],save_fig+f"fluct_y{y_plus}_pr{pr}_T_compare")
    



print(U.shape,V.shape,W.shape,T.shape)


# VAR = [U,V,W,T]

#%%


