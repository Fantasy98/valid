def threeD_plot(error_val,output_path):
    """3d KDE of the errors
    Args:
        error_val (numpy array): the errors
        output_path (Path): where to save
    Returns:
        None: 
    """
    import numpy as np
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    
    #change the scale to plus units
    Re_Tau = 395 #Direct from simulation
    Re = 10400 #Direct from simulation
    nu = 1/Re #Kinematic viscosity
    u_tau = Re_Tau*nu


    train_numpy=error_val.to_numpy()
    num_snapshots=int(train_numpy.shape[0]/256/256)
    reshape_t=train_numpy.reshape((num_snapshots,256,256))
    avg=np.mean(reshape_t,0)

    # Create meshgrid
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
    #fig = plt.figure(figsize=(15*cm,10*cm),dpi=200)
    fig = plt.figure(figsize=(15*cm,10*cm),dpi=500)
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(xx, yy, np.transpose(avg), cmap='viridis', edgecolor='none')
    ax.set_xlabel(r'$x^+$',labelpad=10)
    ax.set_ylabel(r'$z^+$',labelpad=5)
    ax.set_zlabel(r'$E_{RS}\ [\%]$',labelpad=5)
    ax.set_box_aspect((2,1,1))

    ax.set_xticks(placement_x)
    ax.set_xticklabels(axis_range_x)
    ax.set_xticks(placement_x)
    ax.set_xticklabels(axis_range_x)

    ax.set_yticks(placement_z)
    ax.set_yticklabels(axis_range_z)
    #fig.colorbar(surf, shrink=0.1, aspect=5,pad=0.1) # add color bar indicating the PDF
    fig.colorbar(surf,pad=0.12,shrink=0.5,aspect=10)
    ax.view_init(30, 140)
    
    fig.savefig(os.path.join(output_path,'validation_3D.pdf'),bbox_inches='tight')

    return None

def pdf_plots(error_fluc,names,output_path,target_type):
    """Makes both boxplot and pdf plot for the errors.
    Args:
        error_fluc (list): list of the train,validation,test errors in local form
        names (list): list of the names of the data. Normally train,validaiton,test
        output_path (Path): Path of where to save the figures
    """
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    import seaborn as sns
    import KDEpy
    


    sns.set_theme()
    sns.set_context("paper")
    sns.set_style("ticks")
    
    for i in range(3):
        cm =1/2.54
        fig, ax = plt.subplots(1, 1,figsize=(15*cm,10*cm),dpi=100)
        
        
        

        if target_type=="flux":
            sns.boxplot(data=error_fluc[i]['Root sq. error of local heat flux'],showfliers = False,orient='h',ax=ax)
        elif target_type=="stress":
            sns.boxplot(data=error_fluc[i]['Root sq. error of local shear stress'],showfliers = False,orient='h',ax=ax)
        
        sns.despine()
        fig.savefig(os.path.join(output_path,names[i]+'_boxplot.pdf'),bbox_inches='tight',format='pdf')
        plt.clf()
        
        fig, ax = plt.subplots(1, 1,figsize=(15*cm,10*cm),dpi=100)
        max_range_error=error_fluc[i].max().to_numpy().item()*1.1
        min_range_error=error_fluc[i].min().to_numpy().item()*0.99
        #Find coeffs to find the log equiv.

        x_grid = np.linspace(min_range_error, max_range_error, num=int(max_range_error)*2)
        y_fluct = KDEpy.FFTKDE(bw='ISJ', kernel='gaussian').fit(error_fluc[i].to_numpy(), weights=None).evaluate(x_grid)
        

        sns.lineplot(x=x_grid, y=y_fluct,ax=ax)


        ax.set_xlabel(r'Error $\left[\% \right]$')
        ax.set_ylabel('Density')
        # if target_type=="flux":
        #     y_local = KDEpy.FFTKDE(bw='ISJ', kernel='gaussian').fit(error_fluc[i]['Root sq. error of local heat flux'].to_numpy(), weights=None).evaluate(x_grid)
        #     sns.lineplot(x=x_grid, y=y_local, label='Root sq. error of local heat flux',ax=ax)
        # else:
        #     y_local = KDEpy.FFTKDE(bw='ISJ', kernel='gaussian').fit(error_fluc[i]['Root sq. error of local shear stress'].to_numpy(), weights=None).evaluate(x_grid)
        #     sns.lineplot(x=x_grid, y=y_local, label='Root sq. error of local shear stress',ax=ax)
        
        sns.despine()

        ax.fill_between(x_grid,y_fluct,alpha=0.8,color='grey')
        
        #ax.fill_between(x_grid,y_local,alpha=0.4,color='grey')
        #ax.set(xscale='log')
        ax.set_xlim(-1,100)

        fig.savefig(os.path.join(output_path,names[i]+'_PDF.png'),bbox_inches='tight')
        plt.clf()



def error(target_list,target_type,names,predctions,output_path):
    
    

    import os
    import numpy as np
    import pandas as pd
    from numba import njit

    @njit(cache=True,parallel=True)    
    def cal_func(target_list,predctions):
        
        fluc_predict=predctions-np.mean(predctions)
        fluc_target=target_list-np.mean(target_list)
        

        #Global average errors
        global_mean_err=(np.mean(predctions)-np.mean(target_list))/(np.mean(target_list))*100
       
        MSE_local_shear_stress=np.sqrt(np.mean((predctions-target_list)**2))/np.mean(target_list)*100
        #MSE_local_shear_stress=np.sqrt((np.mean((predctions-target_list)**2))/np.mean(target_list)**2)*100
        
        global_fluct_error=(np.std(fluc_predict)-np.std(fluc_target))/(np.std(fluc_target))*100
        
        MSE_local_fluc=(np.sqrt(np.mean((fluc_predict-fluc_target)**2))/np.std(fluc_target))*100
        #MSE_local_fluc=np.sqrt((np.mean((fluc_predict-fluc_target)**2))/np.std(fluc_target)**2)*100
        

        #MAE_local=np.mean(np.abs(predctions[i][:,:,:]-target_list[i][:,:,:]))/np.mean(np.abs(target_list[i][:,:,:]))*100
        #MAE_local_no_mean=(np.abs(predctions[i][:,:,:]-target_list[i][:,:,:]))/np.mean(np.abs(target_list[i][:,:,:]))*100
        #MAE_fluct_no_mean=(np.abs(fluc_predict-fluc_target))/np.mean(np.abs(fluc_target))*100
        

        

        #Local erros for PDF's and boxplots etc.
        MSE_local_no_mean=(np.abs(predctions-target_list)/np.mean(target_list))*100
        #MSE_local_no_mean=np.sqrt(((predctions-target_list)**2)/np.mean(target_list)**2)*100
        
        #MSE_local_fluc_PDF=np.sqrt(((fluc_predict-fluc_target)**2)/(np.std(fluc_target))**2)*100
        
        return MSE_local_no_mean,global_mean_err,MSE_local_shear_stress,global_fluct_error,MSE_local_fluc


    if not os.path.exists(output_path):
        os.makedirs(output_path)

 
    
    if target_type=="stress":
        error=pd.DataFrame(columns=['Global Mean Error','Root mean sq. error of local shear stress','Global fluctuations error','Root mean sq. error of local fluctuations'])
    elif target_type=="flux":
        error=pd.DataFrame(columns=['Global Mean Error','Root mean sq. error of local heat flux','Global fluctuations error','Root mean sq. error of local fluctuations'])
    
    error_fluc_list=[]
    
    


    
    for i in range(3):
        error_fluct=pd.DataFrame()
        
        MSE_local_no_mean,global_mean_err,MSE_local_shear_stress,global_fluct_error,MSE_local_fluc=cal_func(target_list[i],predctions[i])


        if target_type=="stress":
            error_fluct['Root sq. error of local shear stress']=MSE_local_no_mean.flatten()
            error=error.append({'Global Mean Error':global_mean_err,'Root mean sq. error of local shear stress':MSE_local_shear_stress,'Global fluctuations error':global_fluct_error,'Root mean sq. error of local fluctuations':MSE_local_fluc},ignore_index=True)
        elif target_type=="flux":
            error_fluct['Root sq. error of local heat flux']=MSE_local_no_mean.flatten()
            error=error.append({'Global Mean Error':global_mean_err,'Root mean sq. error of local heat flux':MSE_local_shear_stress,'Global fluctuations error':global_fluct_error,'Root mean sq. error of local fluctuations':MSE_local_fluc},ignore_index=True)
        
        #error_fluct['Root sq. error of local fluctuations']=MSE_local_fluc_PDF.flatten()
        #error_fluct['MAE local']=MAE_local_no_mean.flatten()
        #error_fluct['MAE fluct']=MAE_fluct_no_mean.flatten()

        

        error_fluct.to_parquet(os.path.join(output_path,'Error_fluct_'+names[i]+'.parquet'),engine='fastparquet',compression='GZIP')
        error_fluc_list.append(error_fluct)
        

    
    error.index=names

    error.to_csv(os.path.join(output_path,'Mean_error.csv'))

    return error_fluc_list, error



def heatmap_quarter_test(predction,target_var,output_path,target):
    from DataHandling import utility
    from DataHandling.features import slices
    from tensorflow import keras
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    import math
    sns.set_theme()
    sns.set_style("ticks")
    sns.set_context("paper")
    name='test'
    cm = 1/2.54  # centimeters in inches




    if not os.path.exists(output_path):
        os.makedirs(output_path)


    #change the scale to plus units
    Re_Tau = 395 #Direct from simulation
    Re = 10400 #Direct from simulation
    nu = 1/Re #Kinematic viscosity
    u_tau = Re_Tau*nu
    Q_avg=0.665





    if target[0]=='tau_wall':
        target_var=nu*target_var[1,:,:]/u_tau**2
        predction=nu*predction[1,:,:]/u_tau**2        

        #cut the data to 1/4
        target_var=target_var[:128,:128]
        predction=predction[:128,:128]
            
    elif target[0][-5:]=='_flux':
        fric_temp=Q_avg/u_tau
        target_var=target_var[1,:,:]/Q_avg
        predction=predction[1,:,:]/Q_avg  

        #cut the data to 1/4
        target_var=target_var[:128,:128]
        predction=predction[:128,:128]

        #Need to find the average surface heat flux Q_w
        #Friction temp = Q_w/(u_tau)
        #q^+= q/(Friction temp)



    #Find highest and lowest value to scale plot to
    max_tot=np.max([np.max(target_var),np.max(predction)])
    min_tot=np.min([np.min(target_var),np.min(predction)])



    



    #max length in plus units
    x_range=12/2
    z_range=6/2

    gridpoints_x=int(255/2)+1
    gridponts_z=int(255/2)+1


    x_plus_max=x_range*u_tau/nu
    z_plus_max=z_range*u_tau/nu


    x_plus_max=np.round(x_plus_max).astype(int)
    z_plus_max=np.round(z_plus_max).astype(int)

    axis_range_x=np.array([0,470,950,1420,1900,2370])
    axis_range_z=np.array([0,295,590,890,1185])


    placement_x=axis_range_x*nu/u_tau
    placement_x=np.round((placement_x-0)/(x_range-0)*(gridpoints_x-0)).astype(int)


    placement_z=axis_range_z*nu/u_tau
    placement_z=np.round((placement_z-0)/(z_range-0)*(gridponts_z-0)).astype(int)

    
    fig, axs=plt.subplots(2,figsize=([7*cm,10*cm]),sharex=True,sharey=True,constrained_layout=False,dpi=1000)

    #Target
    pcm=axs[0].imshow(np.transpose(target_var),cmap='viridis',vmin=min_tot,vmax=max_tot,aspect=0.5,interpolation='bicubic')
    axs[0].set_title(name.capitalize(),weight="bold")
    axs[0].set_ylabel(r'$z^+$')
    
    #prediction
    axs[1].imshow(np.transpose(predction),cmap='viridis',vmin=min_tot,vmax=max_tot,aspect=0.5,interpolation='bicubic')
    axs[1].set_xlabel(r'$x^+$')
    axs[1].set_ylabel(r'$z^+$')

    axs[1].set_xticks(placement_x)
    axs[1].set_xticklabels(axis_range_x,rotation=45)
    axs[0].set_yticks(placement_z)
    axs[0].set_yticklabels(axis_range_z)
    axs[1].set_yticks(placement_z)
    axs[1].set_yticklabels(axis_range_z)

        
    #Setting labels and stuff
    axs[0].text(-0.42, 0.20, 'Target',
            verticalalignment='bottom', horizontalalignment='right',
            transform=axs[0].transAxes,rotation=90,weight="bold")

    axs[1].text(-0.42, 0.00, 'Prediction',
            verticalalignment='bottom', horizontalalignment='right',
            transform=axs[1].transAxes,rotation=90,weight="bold")

    fig.subplots_adjust(wspace=-0.31,hspace=0.25)
    cbar=fig.colorbar(pcm,ax=axs[:],aspect=20,shrink=0.7,location="bottom",pad=0.22)
    cbar.formatter.set_powerlimits((0, 0))


    if target[0]=='tau_wall':
        cbar.ax.set_xlabel(r'$\tau_{w}^{+} $',rotation=0)
    elif target[0]=='pr0.71_flux':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.71$',rotation=0)
    elif target[0]=='pr1_flux':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=1$',rotation=0)
    elif target[0]=='pr0.2_flux':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.2$',rotation=0)
    elif target[0]=='pr0.025_flux':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.025$',rotation=0)
    elif target[0]=='pr0.71_flux_IMD_15':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.025$',rotation=0)
    elif target[0]=='pr0.025_flux_mix_pr0.71':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.025or0.71$',rotation=0)
    elif target[0]=='All_pr_flux' or target[0]=='pr0.025_flux_mix_pr0.2_pr0.71_pr1':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.025or0.2or0.71or1$',rotation=0)
    elif target[0]=='u_vel_15':
        cbar.ax.set_xlabel(r'$u_vel$',rotation=0)
    elif target[0]=='pr0.025_flux_mix_pr0.2_pr1':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.025or0.2or1$',rotation=0)
    elif target[0]=='pr0.025_flux_mix_pr0.2_pr0.71':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.025or0.2or071$',rotation=0)
    elif target[0]=='pr0.045_flux':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.045$',rotation=0)
    elif target[0]=='pr0.18_flux':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.18$',rotation=0)
    elif target[0]=='pr0.22_flux':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.22$',rotation=0)
    elif target[0]=='pr0.69_flux':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.69$',rotation=0)
    else: 
        raise Exception('target name is not defined')

    fig.savefig(os.path.join(output_path,'target_prediction_OnlyTest.pdf'),bbox_inches='tight',format='pdf')


    max_diff=np.max(target_var-predction)
    min_diff=np.min(target_var-predction)



    fig2, ax=plt.subplots(1,figsize=([7*cm,10*cm]),sharex=True,sharey=True,constrained_layout=False,dpi=1000)

    pcm=ax.imshow(np.transpose(target_var-predction),cmap="Spectral",vmin=min_diff,vmax=max_diff,aspect=0.5,interpolation='bicubic')
    ax.set_xlabel(r'$x^+$')
    ax.set_ylabel(r'$z^+$')
    ax.set_title("difference".capitalize(),weight="bold")
    ax.set_xticks(placement_x,)
    ax.set_xticklabels(axis_range_x,rotation=45)
    ax.set_xticks(placement_x)
    ax.set_xticklabels(axis_range_x,rotation=45)

    ax.set_yticks(placement_z)
    ax.set_yticklabels(axis_range_z)
    cbar=fig2.colorbar(pcm,ax=ax,aspect=20,shrink=0.9,orientation="horizontal",pad=0.23)
    cbar.formatter.set_powerlimits((0, 0))

    if target[0]=='tau_wall':
        cbar.ax.set_xlabel(r'$\tau_{w}^{+} $',rotation=0)
    elif target[0]=='pr0.71_flux':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.71$',rotation=0)
    elif target[0]=='pr1_flux':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=1$',rotation=0)
    elif target[0]=='pr0.2_flux':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.2$',rotation=0)
    elif target[0]=='pr0.025_flux':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.025$',rotation=0)
    elif target[0]=='pr0.71_flux_IMD_15':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.025$',rotation=0)
    elif target[0]=='pr0.025_flux_mix_pr0.71':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.025or0.71$',rotation=0)
    elif target[0]=='All_pr_flux' or target[0]=='pr0.025_flux_mix_pr0.2_pr0.71_pr1':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.025or0.2or0.71or1$',rotation=0)
    elif target[0]=='All_pr_flux' or target[0]=='pr0.025_flux_mix_pr0.2_pr0.71_pr1':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.025or0.2or0.71or1$',rotation=0)
    elif target[0]=='u_vel_15':
        cbar.ax.set_xlabel(r'$u_vel$',rotation=0)
    elif target[0]=='pr0.025_flux_mix_pr0.2_pr0.71':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.025or0.2or1$',rotation=0)
    elif target[0]=='pr0.045_flux':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.045$',rotation=0)
    elif target[0]=='pr0.18_flux':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.18$',rotation=0)
    elif target[0]=='pr0.22_flux':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.22$',rotation=0)
    elif target[0]=='pr0.69_flux':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.69$',rotation=0)

    else: 
        raise Exception('target name is not defined')

    fig2.savefig(os.path.join(output_path,'difference_OnlyTest.pdf'),bbox_inches='tight',format='pdf')






def heatmap_quarter(predctions,target_list,output_path,target):
    from DataHandling import utility
    from DataHandling.features import slices
    from tensorflow import keras
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme()
    sns.set_style("ticks")
    sns.set_context("paper")
    names=['training',"validation",'test']
    cm = 1/2.54  # centimeters in inches




    if not os.path.exists(output_path):
        os.makedirs(output_path)


    #change the scale to plus units
    Re_Tau = 395 #Direct from simulation
    Re = 10400 #Direct from simulation
    nu = 1/Re #Kinematic viscosity
    u_tau = Re_Tau*nu
    Q_avg=0.665





    if 'tau_wall' in target[0]:
        for i in range(len(target_list)):
            target_list[i]=nu*target_list[i][1,:,:]/u_tau**2
            #target_list[i]=(target_list[i]-np.mean(target_list[i]))/np.std(target_list[i])
            predctions[i]=nu*predctions[i][1,:,:]/u_tau**2    
            #predctions[i]=(predctions[i]-np.mean(predctions[i]))/np.std(predctions[i])    

            #cut the data to 1/4
            target_list[i]=target_list[i][:128,:128]
            predctions[i]=predctions[i][:128,:128]
            
    elif 'flux' in target[0]:
        fric_temp=Q_avg/u_tau
        for i in range(len(target_list)):
            target_list[i]=target_list[i][1,:,:]/Q_avg
            predctions[i]=predctions[i][1,:,:]/Q_avg  

            #cut the data to 1/4
            target_list[i]=target_list[i][:128,:128]
            predctions[i]=predctions[i][:128,:128]
    elif len(target)==4:
        target=['u_vel_15']
        for i in range(len(target_list)):
            target_list[i]=target_list[i][1,:,:,0]
            predctions[i]=predctions[i][1,:,:,0]

            #cut the data to 1/4
            target_list[i]=target_list[i][:128,:128]
            predctions[i]=predctions[i][:128,:128]


    #Find highest and lowest value to scale plot to
    max_tot=0
    min_tot=1000
    for i in range(3):
        max_inter=np.max([np.max(target_list[i]),np.max(predctions[i])])
        min_inter=np.min([np.min(target_list[i]),np.min(predctions[i])])
        
        
        if max_inter>max_tot:
            max_tot=max_inter
        if min_inter<min_tot:
            min_tot=min_inter


    fig, axs=plt.subplots(2,3,figsize=([21*cm,10*cm]),sharex=True,sharey=True,constrained_layout=False,dpi=1000)



    #max length in plus units
    x_range=12/2
    z_range=6/2

    gridpoints_x=int(255/2)+1
    gridponts_z=int(255/2)+1


    x_plus_max=x_range*u_tau/nu
    z_plus_max=z_range*u_tau/nu


    x_plus_max=np.round(x_plus_max).astype(int)
    z_plus_max=np.round(z_plus_max).astype(int)

    axis_range_x=np.array([0,470,950,1420,1900,2370])
    axis_range_z=np.array([0,295,590,890,1185])


    placement_x=axis_range_x*nu/u_tau
    placement_x=np.round((placement_x-0)/(x_range-0)*(gridpoints_x-0)).astype(int)


    placement_z=axis_range_z*nu/u_tau
    placement_z=np.round((placement_z-0)/(z_range-0)*(gridponts_z-0)).astype(int)

    for i in range(3):  

        #Target
        pcm=axs[0,i].imshow(np.transpose(target_list[i]),cmap='viridis',vmin=min_tot,vmax=max_tot,aspect=0.5)
        axs[0,i].set_title(names[i].capitalize(),weight="bold")
        axs[0,0].set_ylabel(r'$z^+$')
        
        #prediction
        axs[1,i].imshow(np.transpose(predctions[i]),cmap='viridis',vmin=min_tot,vmax=max_tot,aspect=0.5)
        axs[1,i].set_xlabel(r'$x^+$')
        axs[1,0].set_ylabel(r'$z^+$')

        axs[1,0].set_xticks(placement_x)
        axs[1,0].set_xticklabels(axis_range_x,rotation=45)
        axs[1,1].set_xticks(placement_x)
        axs[1,1].set_xticklabels(axis_range_x,rotation=45)
        axs[1,2].set_xticks(placement_x)
        axs[1,2].set_xticklabels(axis_range_x,rotation=45)
        axs[0,0].set_yticks(placement_z)
        axs[0,0].set_yticklabels(axis_range_z)
        axs[1,0].set_yticks(placement_z)
        axs[1,0].set_yticklabels(axis_range_z)

        
    #Setting labels and stuff
    axs[0,0].text(-0.45, 0.20, 'Target',
            verticalalignment='bottom', horizontalalignment='right',
            transform=axs[0,0].transAxes,rotation=90,weight="bold")

    axs[1,0].text(-0.45, 0.00, 'Prediction',
            verticalalignment='bottom', horizontalalignment='right',
            transform=axs[1,0].transAxes,rotation=90,weight="bold")

    fig.subplots_adjust(wspace=-0.31,hspace=0.25)
    cbar=fig.colorbar(pcm,ax=axs[:,:],aspect=30,shrink=0.55,location="bottom",pad=0.24)
    cbar.formatter.set_powerlimits((0, 0))


    if target[0]=='tau_wall':
        cbar.ax.set_xlabel(r'$\tau_{w}^{+} $',rotation=0)
    elif target[0]=='pr0.71_flux':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.71$',rotation=0)
    elif target[0]=='pr1_flux':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=1$',rotation=0)
    elif target[0]=='pr0.2_flux':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.2$',rotation=0)
    elif target[0]=='pr0.025_flux':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.025$',rotation=0)
    elif target[0]=='pr0.71_flux_IMD_15':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.025$',rotation=0)
    elif target[0]=='pr0.025_flux_mix_pr0.71':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.025or0.71$',rotation=0)
    elif target[0]=='All_pr_flux' or target[0]=='pr0.025_flux_mix_pr0.2_pr0.71_pr1':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.025or0.2or0.71or1$',rotation=0)
    elif target[0]=='u_vel_15':
        cbar.ax.set_xlabel(r'$u_vel$',rotation=0)
    elif target[0]=='pr0.025_flux_mix_pr0.2_pr1':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.025or0.2or1$',rotation=0)
    elif target[0]=='pr0.025_flux_mix_pr0.2_pr0.71':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.025or0.2or071$',rotation=0)
    elif target[0]=='pr0.045_flux':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.045$',rotation=0)
    elif target[0]=='pr0.18_flux':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.18$',rotation=0)
    elif target[0]=='pr0.22_flux':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.22$',rotation=0)
    elif target[0]=='pr0.69_flux':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.69$',rotation=0)
    else: 
        raise Exception('target name is not defined')

    fig.savefig(os.path.join(output_path,'target_prediction.pdf'),bbox_inches='tight',format='pdf')


    max_diff=np.max([np.max(target_list[0]-predctions[0]),np.max(target_list[1]-predctions[1]),np.max(target_list[2]-predctions[2])])
    min_diff=np.min([np.min(target_list[0]-predctions[0]),np.min(target_list[1]-predctions[1]),np.min(target_list[2]-predctions[2])])

    fig2, axs=plt.subplots(1,3,figsize=([21*cm,10*cm]),sharex=True,sharey=True,constrained_layout=False,dpi=1000)
    for i in range(3):
        pcm=axs[i].imshow(np.transpose(target_list[i]-predctions[i]),cmap="Spectral",vmin=min_diff,vmax=max_diff,aspect=0.5)
        axs[i].set_xlabel(r'$x^+$')
        axs[0].set_ylabel(r'$z^+$')
        axs[i].set_title(names[i].capitalize(),weight="bold")
        axs[0].set_xticks(placement_x,)
        axs[0].set_xticklabels(axis_range_x,rotation=45)
        axs[1].set_xticks(placement_x)
        axs[1].set_xticklabels(axis_range_x,rotation=45)
        axs[2].set_xticks(placement_x)
        axs[2].set_xticklabels(axis_range_x,rotation=45)

    axs[0].set_yticks(placement_z)
    axs[0].set_yticklabels(axis_range_z)
    fig2.subplots_adjust(wspace=0.13,hspace=0.05)
    cbar=fig.colorbar(pcm,ax=axs[:],aspect=30,shrink=0.55,location="bottom",pad=0.20)
    cbar.formatter.set_powerlimits((0, 0))

    if target[0]=='tau_wall':
        cbar.ax.set_xlabel(r'$\tau_{w}^{+} $',rotation=0)
    elif target[0]=='pr0.71_flux':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.71$',rotation=0)
    elif target[0]=='pr1_flux':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=1$',rotation=0)
    elif target[0]=='pr0.2_flux':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.2$',rotation=0)
    elif target[0]=='pr0.025_flux':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.025$',rotation=0)
    elif target[0]=='pr0.71_flux_IMD_15':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.025$',rotation=0)
    elif target[0]=='pr0.025_flux_mix_pr0.71':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.025or0.71$',rotation=0)
    elif target[0]=='All_pr_flux' or target[0]=='pr0.025_flux_mix_pr0.2_pr0.71_pr1':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.025or0.2or0.71or1$',rotation=0)
    elif target[0]=='All_pr_flux' or target[0]=='pr0.025_flux_mix_pr0.2_pr0.71_pr1':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.025or0.2or0.71or1$',rotation=0)
    elif target[0]=='u_vel_15':
        cbar.ax.set_xlabel(r'$u_vel$',rotation=0)
    elif target[0]=='pr0.025_flux_mix_pr0.2_pr0.71':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.025or0.2or1$',rotation=0)
    elif target[0]=='pr0.045_flux':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.045$',rotation=0)
    elif target[0]=='pr0.18_flux':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.18$',rotation=0)
    elif target[0]=='pr0.22_flux':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.22$',rotation=0)
    elif target[0]=='pr0.69_flux':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.69$',rotation=0)

    else: 
        raise Exception('target name is not defined')


    fig2.savefig(os.path.join(output_path,'difference.pdf'),bbox_inches='tight',format='pdf')

    return None

def heatmaps(target_list,names,predctions,output_path,model_path,target):
    """makes heatmaps of the Train validation and test data for target and prediction. Also plots the difference. Save to the output folder
    Args:
        target_list (list): list of arrays of the target
        names (list): list of names for the target_list
        predctions (list): list of array of the prediction
        output_path (Path): Path to the output folder
        model_path (Path): Path to the saved model
    Raises:
        Exception: if the target has no defined plot name
        Exception: Same as above
    Returns:
        None: 
    """
    from DataHandling import utility
    from DataHandling.features import slices
    from tensorflow import keras
    import numpy as np
    import shutil
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = 1/2.54  # centimeters in inches


 

    if not os.path.exists(output_path):
        os.makedirs(output_path)


    #change the scale to plus units
    Re_Tau = 395 #Direct from simulation
    Re = 10400 #Direct from simulation
    nu = 1/Re #Kinematic viscosity
    u_tau = Re_Tau*nu
    Q_avg=0.665
    
    
    if 'tau_wall' in target[0]:
        for i in range(len(target_list)):
            target_list[i]=target_list[i][1,:,:]/u_tau
            predctions[i]=predctions[i][1,:,:]/u_tau        

            #cut the data to 1/4
            target_list[i]=target_list[i][128,128]
            predctions[i]=predctions[i][128,128]
            

    
    elif 'flux' in target[0]:
        fric_temp=Q_avg/u_tau
        for i in range(len(target_list)):
            target_list[i]=target_list[i][1,:,:]/Q_avg
            predctions[i]=predctions[i][1,:,:]/Q_avg  

            #cut the data to 1/4
            target_list[i]=target_list[i][128,128]
            predctions[i]=predctions[i][128,128]

        #Need to find the average surface heat flux Q_w
        #Friction temp = Q_w/(u_tau)
        #q^+= q/(Friction temp)


    #Find highest and lowest value to scale plot to
    max_tot=0
    min_tot=1000
    for i in range(3):
        max_inter=np.max([np.max(target_list[i]),np.max(predctions[i])])
        min_inter=np.min([np.min(target_list[i]),np.min(predctions[i])])
        
        
        if max_inter>max_tot:
            max_tot=max_inter
        if min_inter<min_tot:
            min_tot=min_inter


    fig, axs=plt.subplots(2,3,figsize=([21*cm,10*cm]),sharex=True,sharey=True,constrained_layout=False,dpi=150)


    #TODO lavet det her om s� akserne passer p� den 1/4 cut jeg har lavet
    
    #max length in plus units
    x_plus_max=12*u_tau/nu
    z_plus_max=6*u_tau/nu


    #To display the correct axis on the plot
    


    axis_range_x=np.linspace(0,255,7)
    x_axis_range=(axis_range_x-0)/(255-0)*(12-0)+0
    x_axis_range=np.round(x_axis_range/u_tau).astype(int)
    
    axis_range_z=np.linspace(0,255,4)
    z_axis_range=(axis_range_z-0)/(255-0)*(6-0)+0
    z_axis_range=np.flip(z_axis_range)
    z_axis_range=np.round(z_axis_range*u_tau/nu).astype(int)
    for i in range(3):  

        #Target
        pcm=axs[0,i].imshow(np.transpose(target_list[i]),cmap='viridis',vmin=min_tot,vmax=max_tot,aspect=0.5)
        axs[0,i].set_title(names[i].capitalize(),weight="bold")
        axs[0,0].set_ylabel(r'$z^+$')
        
        #prediction
        axs[1,i].imshow(np.transpose(predctions[i]),cmap='viridis',vmin=min_tot,vmax=max_tot,aspect=0.5)
        axs[1,i].set_xlabel(r'$x^+$')
        axs[1,0].set_ylabel(r'$z^+$')

        axs[1,i].set_xticks(axis_range_x)
        axs[1,i].set_xticklabels(x_axis_range)
        axs[0,0].set_yticks(axis_range_z)
        axs[0,0].set_yticklabels(z_axis_range)
        axs[1,0].set_yticks(axis_range_z)
        axs[1,0].set_yticklabels(z_axis_range)

        
    #Setting labels and stuff
    axs[0,0].text(-0.23, 0.30, 'Target',
            verticalalignment='bottom', horizontalalignment='right',
            transform=axs[0,0].transAxes,rotation=90,weight="bold")

    axs[1,0].text(-0.23, 0.20, 'Prediction',
            verticalalignment='bottom', horizontalalignment='right',
            transform=axs[1,0].transAxes,rotation=90,weight="bold")

    fig.subplots_adjust(wspace=0.09,hspace=0.15)
    cbar=fig.colorbar(pcm,ax=axs[:,:],aspect=20,shrink=0.5,location="bottom")
    cbar.formatter.set_powerlimits((0, 0))


    if target[0]=='tau_wall':
        cbar.ax.set_xlabel(r'$\tau_{w}^{+} $',rotation=0)
    elif target[0]=='pr0.71_flux':
        cbar.ax.set_xlabel(r'$q_w^+,\quad Pr=0.71$',rotation=0)
    else: 
        raise Exception('target name is not defined')

    fig.savefig(os.path.join(output_path,'target_prediction.pdf'),bbox_inches='tight',format='pdf')


    max_diff=np.max([np.max(target_list[0]-predctions[0]),np.max(target_list[1]-predctions[1]),np.max(target_list[2]-predctions[2])])
    min_diff=np.min([np.min(target_list[0]-predctions[0]),np.min(target_list[1]-predctions[1]),np.min(target_list[2]-predctions[2])])

    fig2, axs=plt.subplots(1,3,figsize=([21*cm,10*cm]),sharex=True,sharey=True,constrained_layout=False,dpi=150)
    for i in range(3):
        pcm=axs[i].imshow(target_list[i]-predctions[i],cmap="Spectral",vmin=min_diff,vmax=max_diff,aspect=0.5)
        axs[i].set_xlabel(r'$x^+$')
        axs[0].set_ylabel(r'$z^+$')
        axs[i].set_title(names[i].capitalize(),weight="bold")
        axs[i].set_xticks(axis_range_x)
        axs[i].set_xticklabels(x_axis_range)

    axs[0].set_yticks(axis_range_z)
    axs[0].set_yticklabels(z_axis_range)
    fig2.subplots_adjust(wspace=0.09,hspace=0.05)
    cbar=fig.colorbar(pcm,ax=axs[:],aspect=20,shrink=0.5,location="bottom")
    cbar.formatter.set_powerlimits((0, 0))

    if target[0]=='tau_wall':
        cbar.ax.set_xlabel(r'$\tau_{w}^{+} $',rotation=0)
    elif target[0]=='pr0.71_flux':
        cbar.ax.set_xlabel(r'$q_w^+,\quad Pr=0.71$',rotation=0)
    else: 
        raise Exception('target name is not defined')



    fig2.savefig(os.path.join(output_path,'difference.pdf'),bbox_inches='tight',format='pdf')
    
    return None





def stat_plots(mean_dataset_loc,batches,pr_list):
    from DataHandling.features.stats import get_valdata
    import matplotlib.pyplot as plt
    import xarray as xr
    import numpy as np
    
    mean = xr.open_mfdataset(mean_dataset_loc, parallel=True)

    mean = mean.persist()
    mean = mean.groupby_bins("time", batches).mean()

    #Validation data
    val_u = get_valdata('u')

    linerRegion = np.linspace(0, 9)
    logRegion = 1 / 0.4 * np.log(np.linspace(20, 180)) + 5
    figScale = 2
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12 * figScale, 6 * figScale))
    ax1.plot('y+', 'u_plusmean', 'ok', data=val_u, label='DNS validation data')
    colorList = ['*b', '*c', '*y', '*g', '.b', '.c', '.y', '.g', 'vb', 'vc', 'vy', 'vg', '<b', '<c', '<y', '<g'] * 30
    # Plotting the batches in mean for U
    for i in range(len(mean.time_bins)):
        ax1.plot(mean.y_plus, mean.u_plusmean.isel(time_bins=i), colorList[i], label='DNS batch ' + str(i))

    ax1.plot(linerRegion, linerRegion, 'r', label='Linear Region')
    ax1.plot(np.linspace(20, 180), logRegion, 'm', linewidth=5, label='Log Region')

    ax1.set_title('Normalized mean values', fontsize=26)
    ax1.set_xscale('log')
    ax1.set_xlabel('$y^{+}$', fontsize=20)
    ax1.set_ylabel(r'$\overline{u^{+}}$', fontsize=20)
    ax1.set_xlim(1, 300)
    ax1.set_ylim(0, 20)
    ax1.minorticks_on()
    ax1.grid(True, which="both", linestyle='--')
    ax1.legend(prop={"size": 17})

    # Now for <u_rms>

    ax2.plot('y+', 'u_plusRMS', 'ok', data=val_u, label='DNS validation data')
    ax2.set_title('Normalized RMS of fluctuations', fontsize=26)
    for i in range(len(mean.time_bins)):
        ax2.plot(mean.y_plus, mean.u_plusRMS.isel(time_bins=i), colorList[i], label='DNS batch ' + str(i))

    ax2.set_xscale('log')
    ax2.set_xlabel('$y^{+}$', fontsize=20)
    ax2.set_ylabel(r"$\overline{u^{+}_{RMS}}$", fontsize=20)
    ax2.set_xlim(1, 300)
    ax2.minorticks_on()
    ax2.grid(True, which="both", linestyle='--')
    ax2.legend(prop={"size": 17})
    plt.tight_layout()
    plt.savefig("/home/au567859/DataHandling/reports/figures_test/u_val.pdf", bbox_inches='tight')

    a = get_valdata('pr1')
    b = get_valdata('pr71')
    c = get_valdata('pr0025')

    val_pr = [a, b, c]
    val_pr = val_pr[0].join(val_pr[1:])

    linerRegion = np.linspace(0, 9)
    logRegion = 1 / 0.43 * np.log(np.linspace(20, 180)) + 3
    figScale = 2

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12 * figScale, 6 * figScale))
    #ax1.plot('pr1_y+', 'pr1_plusmean', 'ok', data=val_pr, label='DNS validation data Pr=1')
    ax1.plot('pr0.71_y+', 'pr0.71_plusmean', 'or', data=val_pr, label='DNS validation data Pr=0.71')
    ax1.plot('pr0.025_y+', 'pr0.025_plusmean', 'om', data=val_pr, label='DNS validation data Pr=0.025')
    colorList = ['*b', '*c', '*y', '*g', '.b', '.c', '.y', '.g', 'vb', 'vc', 'vy', 'vg', '<b', '<c', '<y', '<g'] * 30

    # Plotting the batches in mean for the different Pr
    #pr_list = ['pr1', 'pr0.71', 'pr0.2', 'pr0.025']
    j = 0
    for i in range(len(mean.time_bins)):
        for Pr in pr_list:
            ax1.plot(mean.y_plus, mean[Pr + '_plusmean'].isel(time_bins=i), colorList[j * 4 + i],
                     label='DNS batch' + str(i) + 'Pr=' + Pr[2:])
            j = j + 1

    dualColor = ['k', 'r']
    j = 0
    for Pr in pr_list:
        ax1.plot(linerRegion, linerRegion * float(Pr[2:]), dualColor[i % 2],
                 label='Linear Region y+*Pr ' + 'Pr=' + Pr[2:])
        j = j + 1
    ax1.set_title('Normalized mean values', fontsize=26)
    ax1.set_xscale('log')
    ax1.set_xlabel('$y^+$', fontsize=20)
    ax1.set_ylabel(r'$\overline{\theta^{+}}$', fontsize=20)
    ax1.set_xlim(1, 300)
    ax1.set_ylim(0, 20)
    ax1.grid(True, which="both", linestyle='--')
    ax1.legend(loc='best', prop={"size": 15})

    # Now for <Pr_rms>

    #ax2.plot('pr1_y+', 'pr1_plusRMS', 'ok', data=val_pr, label='DNS validation data Pr=1')
    ax2.plot('pr0.71_y+', 'pr0.71_plusRMS', 'or', data=val_pr, label='DNS validation data Pr=0.71')
    ax2.plot('pr0.025_y+', 'pr0.025_plusRMS', 'om', data=val_pr, label='DNS validation data Pr=0.025')
    ax2.set_title('Normalized RMS of fluctuations', fontsize=26)
    j = 0
    for i in range(len(mean.time_bins)):
        for Pr in pr_list:
            ax2.plot(mean.y_plus, mean[Pr + '_plusRMS'].isel(time_bins=i), colorList[j * 4 + i],
                     label='DNS batch' + str(i) + 'Pr=' + Pr[2:])
            j = j + 1

    ax2.set_xscale('log')
    ax2.set_xlabel('$y^+$', fontsize=20)
    ax2.set_ylabel(r'$\overline{\theta ^{+}_{RMS}}$', fontsize=20)
    ax2.set_xlim(1, 300)
    ax2.set_ylim(0, 3)
    ax2.grid(True, which="both", linestyle='--')
    ax2.legend(loc='best', prop={"size": 15})

    plt.tight_layout()
    plt.savefig("/home/au567859/DataHandling/reports/figures_test/Pr_val.pdf", bbox_inches='tight')