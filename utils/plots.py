import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns
import cmocean.cm as cmo
# cmp = sns.color_palette('YlGnBu', as_cmap=True)
# cmp = sns.color_palette('cmo.curl', as_cmap=True)
# plt.set_cmap(cmp)

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
        clb=plt.imshow(np.transpose(avg), cmap='RdBu',aspect=0.5)
        cbar =fig.colorbar(clb,ax=ax,aspect = 20,shrink=0.9,orientation="horizontal",location="bottom")
        cbar.formatter.set_powerlimits((0,0))
        ax.set_xlabel(r'$x^+$',fontdict={"size":15})
        ax.set_ylabel(r'$z^+$',fontdict={"size":15})
        ax.set_xticks(placement_x)
        ax.set_yticks(placement_z)
        ax.set_xticklabels(axis_range_x)
        ax.set_yticklabels(axis_range_z)
        fig.savefig(save_dir,bbox_inches="tight")


def Plot_2D_2snapshots(avg,save_dir):
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
        fig,ax = plt.subplots(2,1,figsize=(15*cm,20*cm),sharex=True,sharey=True,dpi=300)
        for i in range(2):
            clb=ax[i].imshow(avg[i,:,:],
                             vmin=avg.min(),vmax=avg.max(),interpolation="bicubic",level=30)
            ax[i].set_xlabel(r'$x^+$',fontdict={"size":15})
            ax[i].set_ylabel(r'$z^+$',fontdict={"size":15})
            ax[i].set_xticks(placement_x)
            ax[i].set_yticks(placement_z)
            ax[i].set_xticklabels(axis_range_x)
            ax[i].set_yticklabels(axis_range_z)
        cbar =fig.colorbar(clb,ax=ax.flatten(),aspect = 20,shrink=0.9,orientation="horizontal",location="bottom")
        cbar.formatter.set_powerlimits((0,0))
            
        fig.savefig(save_dir,bbox_inches="tight",dpi=300)


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
        fig,axes = plt.subplots(len(names),1,figsize=(6*(len(names)+6)*cm,6*(len(names)+1)*cm),
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
        axes = axes.flatten()
        for i in range(len(names)):
            clb=axes[i].imshow((avg[i,:,:]),
                                 cmap="RdBu_r",
                                 aspect=0.5,
                                 vmax=max_ve,vmin=min_ve,
                                #  levels= 30
                                 )
            
            axes[i].set_xlabel(r'$x^+$',fontdict={"size":15})
            axes[i].set_ylabel(r'$z^+$',fontdict={"size":15})
            axes[i].set_title(names[i],fontdict={"size":16,"weight":"bold"})
            axes[i].set_xticks(placement_x)
            axes[i].set_yticks(placement_z)
            axes[i].set_xticklabels(axis_range_x)
            axes[i].set_yticklabels(axis_range_z)
            
        cbar =fig.colorbar(clb,ax=axes.flatten(),
                           aspect = 25,shrink=0.9,
                           orientation="horizontal",
                           location = "bottom"
                           )
        cbar.formatter.set_powerlimits((0,0))
        fig.tight_layout()
        # plt.colorbar()

        fig.savefig(save_dir,bbox_inches="tight")
        

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
    fig,axes = plt.subplots(2,2,figsize=(25*cms,20*cms),dpi=500,subplot_kw=dict(projection='3d'))
    axes = axes.flatten()

    ls = LightSource(270, 45)
    for i in range(len(names)):
        surf = axes[i].plot_surface(xx, yy, U[i,:,:], rstride=2, cstride=2,cmap="seismic",
                                linewidth=1, antialiased=False, shade=True,vmax = U.max(),vmin=U.min())
        
        plt.tight_layout()
        axes[i].set_xlabel(r'$x^+$',labelpad=10)
        axes[i].set_ylabel(r'$z^+$',labelpad=10)
            # axes[0].set_zlabel(r'$E_{RS}\ [\%]$',labelpad=0)
            # axes[0].zaxis._axinfo['label']['space_factor'] = 2.8
        axes[i].set_title(names[i])
        axes[i].set_xticks(placement_x)
        axes[i].set_xticklabels(axis_range_x)
        axes[i].set_yticks(placement_z)
        axes[i].set_yticklabels(axis_range_z)
        axes[i].set_box_aspect((2,1,1))
        # ax.view_init(30, 140)
        axes[i].view_init(25,-75)
    cbar =fig.colorbar(surf,ax=axes.flatten().tolist(),aspect = 20,shrink=0.9,orientation="horizontal",location="bottom")
    cbar.formatter.set_powerlimits((0,0))
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

    # axis_range_x=np.array([1900,2850])
    # axis_range_z=np.array([950,1420])
    axis_range_x=np.array([0,950,1900,2850,3980,4740])
    axis_range_z=np.array([0,470,950,1420,1900,2370])


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

def Snap_Plot3D(z,save_dir):
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
    mappable = cm.ScalarMappable()
    mappable.set_array(z)
    # ax = fig.gca(projection='3d')
    ls = LightSource(270, 45)
    
    surf = ax.plot_surface(xx, yy, z, rstride=1, cstride=1,cmap=mappable.cmap,
                        linewidth=1, antialiased=False, shade=True)
    plt.colorbar(surf,pad = 0.18)
    plt.tight_layout()
    ax.set_xlabel(r'$x^+$',labelpad=10)
    ax.set_ylabel(r'$z^+$',labelpad=5)
    ax.set_zlabel(r'$E_{RS}\ [\%]$',labelpad=0)
    ax.zaxis._axinfo['label']['space_factor'] = 2.8
    ax.set_xticks(placement_x)
    ax.set_xticklabels(axis_range_x)
    ax.set_yticks(placement_z)
    ax.set_yticklabels(axis_range_z)
    ax.set_box_aspect((2,1,1))
    ax.view_init(30, 140)
    plt.savefig(save_dir,bbox_inches="tight",dpi=300)


def PSD_single(y,pred,save_dir):
    import numpy as np 
    import matplotlib.pyplot as plt
    import matplotlib        as mpl
    Nx = 256 ; Nz  = 256 ;Lx  = 12 ;Lz  = 6
    # dx=Lx/Nx ;dz=Lz/Nz
    x_range=np.linspace(1,Nx,Nx)
    z_range=np.linspace(1,Nz,Nz)
    # x=dx*x_range;z=dz*z_range;[xx,zz]=np.meshgrid(x,z)
    dkx = 2*np.pi/Lx
    dkz = 2*np.pi/Lz
    kx = dkx * np.append(x_range[:Nx//2], -x_range[Nx//2:0:-1])
    kz = dkz * np.append(z_range[:Nz//2], -z_range[Nz//2:0:-1])
    [kkx,kkz]=np.meshgrid(kx,kz)
    kkx_norm= np.sqrt(kkx**2)
    kkz_norm = np.sqrt(kkz**2)

    Re_Tau = 395 #Direct from simulation
    Re = 10400 #Direct from simulation
    nu = 1/Re #Kinematic viscosity
    u_tau = Re_Tau*nu

    # calculating wavelength in plus units 
    Lambda_x = (2*np.pi/kkx_norm)*u_tau/nu
    Lambda_z = (2*np.pi/kkz_norm)*u_tau/nu

    Theta_fluc_targ=y-np.mean(y)
    Theta_fluc_pred=pred-np.mean(pred)

    fourier_image_targ = np.fft.fftn(Theta_fluc_targ)
    fourier_image_pred = np.fft.fftn(Theta_fluc_pred)

    fourier_amplitudes_targ = np.mean(np.abs(fourier_image_targ)**2,axis=0)
    fourier_amplitudes_pred = np.mean(np.abs(fourier_image_pred)**2,axis=0)

    pct10=0.1*np.max(fourier_amplitudes_targ*kkx*kkz)
    pct50=0.5*np.max(fourier_amplitudes_targ*kkx*kkz)
    pct90=0.9*np.max(fourier_amplitudes_targ*kkx*kkz)
    pct100=np.max(fourier_amplitudes_targ*kkx*kkz)

    cmap = mpl.cm.Greys(np.linspace(0,1,20))
    cmap = mpl.colors.ListedColormap(cmap[5:,:-1])
    fig,ax=plt.subplots(1,1,dpi=1000)
    CP=plt.contourf(Lambda_x,Lambda_z,fourier_amplitudes_targ*kkx*kkz,[pct10,pct50,pct90,pct100],cmap=cmap)
    CS=plt.contour(Lambda_x,Lambda_z,fourier_amplitudes_pred*kkx*kkz,[pct10,pct50,pct90,pct100],colors='orange',linestyles='solid')
    plt.xscale('log')
    plt.yscale('log')
    ax.set_ylabel(r'$\lambda_{z}^+$')
    ax.set_xlabel(r'$\lambda_{x}^+$')
    ax.set_title(r'$k_x\ k_z\ \phi_{q_w}$')
    plt.savefig(save_dir,bbox_inches="tight",dpi = 300)
