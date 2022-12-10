import matplotlib.pyplot as plt
import numpy as np 
def Plot_2D_snapshots(avg,save_dir):
    if avg.shape != (256,256):
        
        print("Not valid for the function!")

    else:
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
        plt.figure(figsize=(20*cm,15*cm),dpi=500)
        clb=plt.contourf(xx, yy, np.transpose(avg), cmap='jet', edgecolor='none')
        plt.colorbar(clb)
        plt.xlabel(r'$x^+$',fontdict={"size":15})
        plt.ylabel(r'$z^+$',fontdict={"size":15})
        plt.xticks(placement_x)
        plt.yticks(placement_z)
        plt.savefig(save_dir,bbox_inches="tight")