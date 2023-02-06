import matplotlib.pyplot as plt
import numpy as np 
# def Plot_2D_snapshots(avg,save_dir):
#     if avg.shape != (256,256):
        
#         print("Not valid for the function!")

#     else:
#         Re_Tau = 395 #Direct from simulation
#         Re = 10400 #Direct from simulation
#         nu = 1/Re #Kinematic viscosity
#         u_tau = Re_Tau*nu

#         xx, yy = np.mgrid[0:256:256j, 0:256:256j]


#         x_range=12
#         z_range=6

#         gridpoints_x=int(255)+1
#         gridponts_z=int(255)+1


#         x_plus_max=x_range*u_tau/nu
#         z_plus_max=z_range*u_tau/nu


#         x_plus_max=np.round(x_plus_max).astype(int)
#         z_plus_max=np.round(z_plus_max).astype(int)

#         axis_range_x=np.array([0,950,1900,2850,3980,4740])
#         axis_range_z=np.array([0,470,950,1420,1900,2370])


#         placement_x=axis_range_x*nu/u_tau
#         placement_x=np.round((placement_x-0)/(x_range-0)*(gridpoints_x-0)).astype(int)

#         placement_z=axis_range_z*nu/u_tau
#         placement_z=np.round((placement_z-0)/(z_range-0)*(gridponts_z-0)).astype(int)

#         cm =1/2.54
#         fig,ax = plt.subplots(1,figsize=(20*cm,15*cm),sharex=True,sharey=True,constrained_layout=False,dpi=500)
#         clb=plt.contourf(xx, yy, np.transpose(avg), cmap='jet')
#         plt.colorbar(clb)
#         ax.set_xlabel(r'$x^+$',fontdict={"size":15})
#         ax.set_ylabel(r'$z^+$',fontdict={"size":15})
#         ax.set_xticks(placement_x)
#         ax.set_yticks(placement_z)
#         ax.set_xticklabels(axis_range_x)
#         ax.set_yticklabels(axis_range_z)
#         fig.savefig(save_dir,bbox_inches="tight")

def Plot_2D_snapshots(avg,save_dir):
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
        fig,ax = plt.subplots(1,figsize=(15*cm,20*cm),sharex=True,sharey=True,constrained_layout=False,dpi=1000)
        clb=plt.imshow(np.transpose(avg), cmap='jet',aspect=0.5)
        cbar =fig.colorbar(clb,ax=ax,aspect = 20,shrink=0.9,orientation="horizontal",location="bottom")
        cbar.formatter.set_powerlimits((0,0))
        ax.set_xlabel(r'$x^+$',fontdict={"size":15})
        ax.set_ylabel(r'$z^+$',fontdict={"size":15})
        ax.set_xticks(placement_x)
        ax.set_yticks(placement_z)
        ax.set_xticklabels(axis_range_x)
        ax.set_yticklabels(axis_range_z)
        fig.savefig(save_dir,bbox_inches="tight")

def Plot_multi(avg,names:list,save_dir):
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
        fig,axes = plt.subplots(2,2,figsize=(25*cm,25*cm),sharex=True,sharey=True,constrained_layout=False,dpi=1000)
        min_ve = avg[0,:,:].max()
        max_ve = avg[0,:,:].min()
        for i in range(4):
            min_now = avg[i,:,:].min()
            max_now = avg[i,:,:].max()
            if min_now <= min_ve:
                  min_ve = min_now
            if max_now >=max_ve:
                  max_ve=max_now
        # names = ["u","v","w",r"$\theta$"]
        axes = axes.flatten()
        for i in range(4):
            clb=axes[i].imshow(np.transpose(avg[i,:,:]), cmap='jet',aspect=0.5,vmax=max_ve,vmin=min_ve,interpolation="bicubic")
            axes[i].set_xlabel(r'$x^+$',fontdict={"size":15})
            axes[i].set_ylabel(r'$z^+$',fontdict={"size":15})
            axes[i].set_title(names[i],fontdict={"size":16,"weight":"bold"})
            axes[i].set_xticks(placement_x)
            axes[i].set_yticks(placement_z)
            axes[i].set_xticklabels(axis_range_x)
            axes[i].set_yticklabels(axis_range_z)
        cbar =fig.colorbar(clb,ax=axes.flatten().tolist(),aspect = 20,shrink=0.9,orientation="horizontal",location="bottom")
        cbar.formatter.set_powerlimits((0,0))
        # plt.colorbar()

        fig.savefig(save_dir,bbox_inches="tight")



def Surface_Plot(z):
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
    mappable.set_array(z[100:-100,100:-100])
    # ax = fig.gca(projection='3d')
    ls = LightSource(270, 45)
    
    surf = ax.plot_surface(xx[60:-60,60:-60], yy[60:-60,60:-60], z[60:-60,60:-60], rstride=1, cstride=1,cmap=mappable.cmap,
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

def Snap_Plot3D(z):
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

    # Set up plot
    fig = plt.figure(16,figsize=(15*cms,10*cms),dpi=500)
    # fig,ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax = plt.axes(projection="3d")
    mappable = cm.ScalarMappable(cmap=cm.jet)
    mappable.set_array(z)
    # ax = fig.gca(projection='3d')
    ls = LightSource(270, 45)
    
    surf = ax.plot_surface(xx, yy, z, rstride=1, cstride=1,cmap=mappable.cmap,
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
    ax.view_init(30, 140)
    # ax.view_init(15,-70)