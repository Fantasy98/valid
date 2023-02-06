# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 12:05:38 2022

@author: andre
"""

import os
import h5py
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import requests
from skimage.segmentation import slic
from skimage import measure
import matplotlib.pyplot as plt
import numpy as np
import shap
import PIL.Image as image
from clusters_functions import get_cluster_3D6P, get_volume_cluster_box, get_volume_cluster_event
#from matplotlib import cm

#%%
 
os.environ["CUDA_VISIBLE_DEVICES"]='2'
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
 
physical_devices = tf.config.list_physical_devices('GPU')
available_gpus   = len(physical_devices)

print('Using TensorFlow version: ', tf.__version__, ', GPU:',available_gpus)
print('Using Keras version: ', tf.keras.__version__)

if physical_devices:
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu,True)
    except RuntimeError as ee:
        print(ee)
        
#%% Read data function  
def get_data(file_read,start,end,ylen,zlen,xlen,umax,umin,vmax,vmin,wmax,wmin,dumax,dumin,delta,Umean,vol):
    """
    Function for reading the data of the h5 files of the velocity
    file_read : directory of the database
    start     : starting point for reading
    end       : ending point for reading
    ylen      : number of pixels in y
    zlen      : number of pixels in z
    xlen      : number of pixels in x
    ---------------------------------------------------------------------------
    data_u    : 3d u velocity
    data_v    : 3d v velocity
    data_w    : 3d w velocity
    """  
    datalist_u = []
    datalist_v = []
    datalist_w = []
    datalist_du = []
    for ii in range(start, end):
        file_ii = file_read+'.'+str(ii)+'.h5.uvw'
        print('Reading: --- ' + file_ii)
        file = h5py.File(file_ii,'r+')
        uu_read = file['u'][:ylen:delta,:zlen:delta,:xlen:delta]
        vv_read = file['v'][:ylen:delta,:zlen:delta,:xlen:delta]
        ww_read = file['w'][:ylen:delta,:zlen:delta,:xlen:delta]
        du_read = uu_read-Umean
        uu_read = np.multiply(uu_read,vol)
        vv_read = np.multiply(vv_read,vol)
        ww_read = np.multiply(ww_read,vol)
        du_read = np.multiply(du_read,vol)
#        for yy_vol in np.arange(len(vol[:,0,0])):
#            for zz_vol in np.arange(len(vol[0,:,0])):
#                for xx_vol in np.arange(len(vol[0,0,:])):
#                    uu_read[yy_vol,zz_vol,xx_vol] *= vol[yy_vol,zz_vol,xx_vol]
#                    vv_read[yy_vol,zz_vol,xx_vol] *= vol[yy_vol,zz_vol,xx_vol]
#                    ww_read[yy_vol,zz_vol,xx_vol] *= vol[yy_vol,zz_vol,xx_vol]
        datalist_u.append(uu_read)
        datalist_v.append(vv_read)
        datalist_w.append(ww_read)
        datalist_du.append(du_read)
        file.close()
    print('out')
    uu     = np.array(datalist_u)
    vv     = np.array(datalist_v)
    ww     = np.array(datalist_w)
    du     = np.array(datalist_du)
#    umax   = np.max(np.max(np.max(uu)))
#    vmax   = np.max(np.max(np.max(vv)))
#    wmax   = np.max(np.max(np.max(ww)))
#    umin   = np.min(np.min(np.min(uu)))
#    vmin   = np.min(np.min(np.min(vv)))
#    wmin   = np.min(np.min(np.min(ww)))
    data_u = (uu-umin)/(umax-umin)
    data_v = (vv-vmin)/(vmax-vmin)
    data_w = (ww-wmin)/(wmax-wmin)
    data_du = (du-dumin)/(dumax-dumin)
    return data_u,data_v,data_w,data_du


    
#%% Read data function  
def get_data_post(file_read,start,end,ylen,zlen,xlen,delta,Umeanreal):
    """
    Function for reading the data of the h5 files of the velocity
    file_read : directory of the database
    start     : starting point for reading
    end       : ending point for reading
    ylen      : number of pixels in y
    zlen      : number of pixels in z
    xlen      : number of pixels in x
    ---------------------------------------------------------------------------
    data_u    : 3d u velocity
    data_v    : 3d v velocity
    data_w    : 3d w velocity
    """
    datalist_u = []
    datalist_v = []
    datalist_w = []
    for ii in range(start, end):
        file_ii = file_read+'.'+str(ii)+'.h5.uvw'
        print('Reading: --- ' + file_ii)
        fileh5 = h5py.File(file_ii,'r+')
        uu_read = fileh5['u'][:ylen:delta,:zlen:delta,:xlen:delta]
        vv_read = fileh5['v'][:ylen:delta,:zlen:delta,:xlen:delta]
        ww_read = fileh5['w'][:ylen:delta,:zlen:delta,:xlen:delta]
        datalist_u.append(uu_read)
        datalist_v.append(vv_read)
        datalist_w.append(ww_read)
    print('out')
    uu     = np.array(datalist_u)
    vv     = np.array(datalist_v)
    ww     = np.array(datalist_w)
    du     = uu-Umeanreal
#    umax   = np.max(np.max(np.max(uu)))
#    vmax   = np.max(np.max(np.max(vv)))
#    wmax   = np.max(np.max(np.max(ww)))
#    umin   = np.min(np.min(np.min(uu)))
#    vmin   = np.min(np.min(np.min(vv)))
#    wmin   = np.min(np.min(np.min(ww)))
    return uu,vv,ww,du
 
#%% Read data function  
def get_data_Q(file_read,start,end,ylen,zlen,xlen,umax,umin,vmax,vmin,wmax,wmin,delta):
    """
    Function for reading the data of the h5 files of the velocity
    file_read : directory of the database
    start     : starting point for reading
    end       : ending point for reading
    ylen      : number of pixels in y
    zlen      : number of pixels in z
    xlen      : number of pixels in x
    ---------------------------------------------------------------------------
    data_u    : 3d u velocity
    data_v    : 3d v velocity
    data_w    : 3d w velocity
    """
    datalist_Q = []
    for ii in range(start, end):
        file_ii = file_read+'.'+str(ii)+'.h5.Q'
        print('Reading: --- ' + file_ii)
        fileh5 = h5py.File(file_ii,'r+')
        datalist_Q.append(fileh5['Q'][:ylen:delta,:zlen:delta,:xlen:delta])
    data_Q     = np.array(datalist_Q)
    data_y     = np.array(fileh5['ycoord'][:ylen:delta])
    data_z     = np.array(fileh5['zcoord'][:zlen:delta])
    data_x     = np.array(fileh5['xcoord'][:xlen:delta])
    vol = np.zeros((len(data_y),len(data_z),len(data_x)))
    for yy_vol in np.arange(len(data_y)):
        for zz_vol in np.arange(len(data_z)):
            for xx_vol in np.arange(len(data_x)):
                if yy_vol == 0:
                    delta_yy = (data_y[yy_vol+1]-data_y[yy_vol])/2
                elif yy_vol == len(data_y)-1:
                    delta_yy = (data_y[yy_vol]-data_y[yy_vol-1])/2
                else:
                    delta_yy = (data_y[yy_vol+1]-data_y[yy_vol])/2+(data_y[yy_vol]-data_y[yy_vol-1])/2
                if zz_vol == 0:
                    delta_zz = (data_z[zz_vol+1]-data_z[zz_vol])/2
                elif zz_vol == len(data_z)-1:
                    delta_zz = (data_z[zz_vol]-data_z[zz_vol-1])/2
                else:
                    delta_zz = (data_z[zz_vol+1]-data_z[zz_vol])/2+(data_z[zz_vol]-data_z[zz_vol-1])/2
                if xx_vol == 0:
                    delta_xx = (data_x[xx_vol+1]-data_x[xx_vol])/2
                elif xx_vol == len(data_x)-1:
                    delta_xx = (data_x[xx_vol]-data_x[xx_vol-1])/2
                else:
                    delta_xx = (data_x[xx_vol+1]-data_x[xx_vol])/2+(data_x[xx_vol]-data_x[xx_vol-1])/2
                vol[yy_vol,zz_vol,xx_vol] = delta_yy*delta_zz*delta_xx
    return data_Q,data_y,data_z,data_x,vol
#%%
def def_model(model,inputs,outputs):
    mse = np.zeros((inputs.shape[0],1))
    ii = 0
    for inputs_set in inputs:
#        print('Starting prediction '+str(ii))
        pred = model.predict(inputs_set.reshape(1,inputs.shape[1],inputs.shape[2],inputs.shape[3],inputs.shape[4]))
        len_y = inputs.shape[1]
        len_z = inputs.shape[2]
        len_x = inputs.shape[3]
        mse[ii]  = np.mean(np.sqrt((outputs.reshape(-1,len_y,len_z,len_x,3)-pred)**2))
        ii += 1
    return mse
#%% Read max min
ff = open("norm_u_v2_2022_09_16.txt", "r")
uline = np.array(ff.readline().replace('[','').replace(']','').split(','),dtype='float')
vline = np.array(ff.readline().replace('[','').replace(']','').split(','),dtype='float')
wline = np.array(ff.readline().replace('[','').replace(']','').split(','),dtype='float')
duline = np.array(ff.readline().replace('[','').replace(']','').split(','),dtype='float')
Umeanline = np.array(ff.readline().replace('[','').replace(']','').split(','),dtype='float')
urealline = np.array(ff.readline().replace('[','').replace(']','').split(','),dtype='float')
vrealline = np.array(ff.readline().replace('[','').replace(']','').split(','),dtype='float')
wrealline = np.array(ff.readline().replace('[','').replace(']','').split(','),dtype='float')
durealline = np.array(ff.readline().replace('[','').replace(']','').split(','),dtype='float')
Umeanrealline = np.array(ff.readline().replace('[','').replace(']','').split(','),dtype='float')
umin = uline[0]
umax = uline[1]
vmin = vline[0]
vmax = vline[1]
wmin = wline[0]
wmax = wline[1] 
dumin = duline[0]
dumax = duline[1]
Umean = Umeanline[0]
uminreal = urealline[0]
umaxreal = urealline[1]
vminreal = vrealline[0]
vmaxreal = vrealline[1]
wminreal = wrealline[0]
wmaxreal = wrealline[1] 
duminreal = durealline[0]
dumaxreal = durealline[1]
Umeanreal = Umeanrealline[0]
#%%
model = tf.keras.models.load_model('CNN_model_v2_2022_09_16.h5')
#%%
delta = 3
len_y = 201 # 100  
len_z = 96 #192 # 48  
len_x = 192 # 96  
len_y0 = int((len_y+delta-1)/delta)
len_z0 = int((len_z+delta-1)/delta)
len_x0 = int((len_x+delta-1)/delta)
folder = 'ddbb_uvw/' #  '../bbdd/P125_21Phys/' # 
folderQ =  'ddbb_Q/' # '../../../storage0/andres/DNS_ddbb/P125_21Phys/Q_complete_corrected/' #  '../bbdd/P125_21Phys/Q_complete_corrected/' # 
file   = 'P125_21pir2'
data_step = {'x':[],'y':[],'z':[],'N_struc':[],'SHAP':[],'dx':[],'dz':[],'ymin':[],'ymax':[],'volume':[],'event':[]}
for im_num in range(5010,6891,10): #6891
    
    
    img_Q,ypos,zpos,xpos,vol= get_data_Q(folderQ+file,im_num,im_num+1,len_y,len_z,len_x,umax,umin,vmax,vmin,wmax,wmin,delta)
    hx = xpos[1]-xpos[0]
    hz = zpos[1]-zpos[0]
    mx = len(xpos)
    mz = len(zpos)
    inp_col = 0    
        
    print(im_num)
    img_u,img_v,img_w,img_du = get_data(folder+file,im_num,im_num+1,len_y,len_z,len_x,umax,umin,vmax,vmin,wmax,wmin,dumax,dumin,delta,Umeanreal,vol)
    img_orig = np.stack((img_du,img_v,img_w),axis=4)[0]
    del img_u,img_v,img_w,img_du
    
    ii_t = 0
    vortex = get_cluster_3D6P(img_Q[0,:,:,:])
    vortex_info,cm = get_volume_cluster_box(vortex,hx,hz,ypos,mx,mz,inp_col)
    segments_slic1 = np.zeros((len_y0,len_z0,len_x0),dtype='int')
    nmax = int(len(vortex.nodes))
    for iivor in np.arange(nmax):
        nodes_ii = vortex.nodes[iivor]
        for jjcoor in np.transpose(nodes_ii):
            segments_slic1[int(jjcoor[0]),int(jjcoor[1]),int(jjcoor[2])] = iivor+1
    nmax = np.max(np.max(np.max(segments_slic1)))
    segments_slic2 =  slic(img_Q[0,:,:,:].reshape(len_y0,len_z0,len_x0,1).astype('double'), n_segments=1, compactness=30, sigma=3)
    segments_slic2b = (segments_slic2+1+nmax)*(1-img_Q[0,:,:,:])
    del img_Q
    segments_slic =  segments_slic1+segments_slic2b-1 #segments_slic2 #
    del segments_slic2,segments_slic2b
    nmax2 = len(np.unique(segments_slic))
    
    img_u_out,img_v_out,img_w_out,img_du_out = get_data(folder+file,im_num+1,im_num+2,len_y,len_z,len_x,umax,umin,vmax,vmin,wmax,wmin,dumax,dumin,delta,Umeanreal,vol)
    img_out = np.stack((img_du_out,img_v_out,img_w_out),axis=4)[0]
    del img_u_out,img_v_out,img_w_out,img_du_out
    img_u,img_v,img_w,img_du = get_data_post(folder+file,im_num,im_num+1,len_y,len_z,len_x,delta,Umeanreal)
    del img_u
    img_orig_real = np.stack((img_du,img_v,img_w),axis=4)[0] # img_du
    event,event_poin = get_volume_cluster_event(vortex,img_du[0,:,:,:],img_v[0,:,:,:],vol)
    del img_v,img_w,img_du
    del vortex

    #%%
    # segment the image so we don't have to explain every pixel
    # define a function that depends on a binary mask representing if an image region is hidden
    def mask_image(zs, segmentation, image, background=None):
        if background is None:
            background = image.mean((0,1))
        out = np.zeros((zs.shape[0], image.shape[0], image.shape[1], image.shape[2], image.shape[3]))
        for i in range(zs.shape[0]):
            out[i,:,:,:,:] = image
            for j in range(zs.shape[1]):
                if zs[i,j] == 0:
                    out[i][segmentation == j,:] = background
        return out
    def f(z):
        model_input     = mask_image(z, segments_slic, img_orig, 0)
        return def_model(model,model_input,img_out)
    #%%
    # use Kernel SHAP to explain the network's predictions
    explainer = shap.KernelExplainer(f, np.zeros((1,nmax2)))
    shap_values = explainer.shap_values(np.ones((1,nmax2)), nsamples="auto") # runs VGG16 1000 times
    #%%
    # get the top predictions from the model
    pred_prim = model.predict(img_orig.copy().reshape(-1,len_y0,len_z0,len_x0,3))
    preds  = np.mean(np.mean(np.mean(np.mean(np.sqrt((img_orig.copy()[:,:,:,:].reshape(-1,len_y0,len_z0,len_x0,3)-pred_prim)**2),axis=4),axis=3),axis=2),axis=1)
    top_preds = np.argsort(-preds)
    data_step['x'].append(cm.xx)
    data_step['y'].append(cm.yy)
    data_step['z'].append(cm.zz)
    data_step['N_struc'].append(nmax)
    data_step['SHAP'].append(shap_values[0][0,:])
    data_step['dx'].append(vortex_info[:,0])
    data_step['dz'].append(vortex_info[:,1])
    data_step['ymin'].append(vortex_info[:,2])
    data_step['ymax'].append(vortex_info[:,3])
    data_step['volume'].append(vortex_info[:,4])
    data_step['event'].append(event)
    # make a color map
    from matplotlib.colors import LinearSegmentedColormap
    colors = []
    for l in np.linspace(1,0,100):
        colors.append((245/255,39/255,87/255,l)) # 
    for l in np.linspace(0,1,100):
        colors.append((24/255,196/255,93/255,l)) #K24/255,196/255,93/255
#    cm = LinearSegmentedColormap.from_list("shap", colors)
    #%%
    iimaxy_neg = int(len(ypos)/2)
    if len(ypos)%2==1:
        iimaxy_pos = int(len(ypos)/2+1)
    else:
        iimaxy_pos = int(len(ypos)/2)
    plt.close(1)
    def fill_segmentation(values, segmentation):
        out = np.zeros(segmentation.shape)
        for i in range(len(values)):
            out[segmentation == i] = values[i]
        return out



    inds = top_preds  
    xscat = []
    yscat = []
    zscat = []
    Qscat = []
    yposb = ypos[:iimaxy_neg]
    zz,yy,xx=np.meshgrid(zpos,ypos,xpos)
    zz1,yy1,xx1=np.meshgrid(zpos,yposb,xpos)
    yposc = ypos[iimaxy_pos:]
    zz2,yy2,xx2=np.meshgrid(zpos,yposc,xpos)
    
                
                
    for ii_y in np.arange(len(segments_slic1[:,0,0])):
        for ii_z in np.arange(len(segments_slic1[0,:,0])):
           for ii_x in np.arange(len(segments_slic1[0,0,:])):
               if segments_slic1[ii_y,ii_z,ii_x] > 0:
                   xscat.append(xx[ii_y,ii_z,ii_x])
                   yscat.append(yy[ii_y,ii_z,ii_x])
                   zscat.append(zz[ii_y,ii_z,ii_x])
                   Qscat.append(shap_values[0][0,int(segments_slic[ii_y,ii_z,ii_x])])

#    fig = plt.figure(1)
#    ax = fig.add_subplot(111,projection='3d')
#    imag = ax.scatter(xscat,zscat,yscat,c=Qscat,vmin=-0.009,vmax=0.001,s=2,cmap='viridis')
#    fig.colorbar(imag,ax=ax,label='Shap Values', orientation="horizontal")
#    longx = np.max(xpos)-np.min(xpos)
#    longy = np.max(ypos)-np.min(ypos)
#    longz = np.max(zpos)-np.min(zpos)
#    meanx = (np.max(xpos)+np.min(xpos))/2
#    meany = (np.max(ypos)+np.min(ypos))/2
#    meanz = (np.max(zpos)+np.min(zpos))/2
#    long = np.max([longx,longy,longz])
#    ax.set_xlim(meanx-long/2,meanx+long/2)
#    ax.set_ylim(meany-long/2,meany+long/2)
#    ax.set_zlim(meanz-long/2,meanz+long/2)
#    ax.set_xlabel('x/h')
#    ax.set_ylabel('z/h')
#    ax.set_zlabel('y/h')

#    plt.savefig('explain/explain'+str(im_num)+'.png')
    

    iizz = int(len_z0/2)
    # plot our explanations
    
    m = fill_segmentation(shap_values[inds[0]][0], segments_slic)
    evm = fill_segmentation(event, segments_slic)
    xplot = xx1[:,iizz,:]
    yplot = yy1[:,iizz,:]
    duplot = img_orig_real[:iimaxy_neg,iizz,:,0]
    dvplot = img_orig_real[:iimaxy_neg,iizz,:,1]
    dwplot = img_orig_real[:iimaxy_neg,iizz,:,2]
    shapplot = m[:iimaxy_neg,iizz,:]
    eventplot = evm[iimaxy_pos:,iizz,:]
    contplot = np.heaviside(segments_slic1[:iimaxy_neg,iizz,:],0) 
    fileh5 = h5py.File('explain/explain_uvw'+str(im_num)+'.h5','w')
    fileh5.create_dataset('xplot', data=xplot)
    fileh5.create_dataset('yplot', data=yplot)
    fileh5.create_dataset('duplot', data=duplot)
    fileh5.create_dataset('dvplot', data=dvplot)
    fileh5.create_dataset('dwplot', data=dwplot)
    fileh5.create_dataset('contplot', data=contplot)
    fileh5.create_dataset('shapplot', data=shapplot)
    fileh5.create_dataset('eventplot', data=eventplot)
    
    type_quadrant = np.zeros((len(xplot[:,0]),len(xplot[0,:])))
    for iiaux in np.arange(len(xplot[:,0])):
        for jjaux in np.arange(len(xplot[0,:])):
            if duplot[iiaux,jjaux] > 0 and dvplot[iiaux,jjaux] > 0:
                type_quadrant[iiaux,jjaux] = 1 
            elif duplot[iiaux,jjaux] < 0 and dvplot[iiaux,jjaux] > 0:
                type_quadrant[iiaux,jjaux] = 2
            elif duplot[iiaux,jjaux] < 0 and dvplot[iiaux,jjaux] < 0:
                type_quadrant[iiaux,jjaux] = 3  
            elif duplot[iiaux,jjaux] > 0 and dvplot[iiaux,jjaux] < 0:
                type_quadrant[iiaux,jjaux] = 4 
            else:
                type_quadrant[iiaux,jjaux] = 0
    fileh5.create_dataset('type',data=type_quadrant)
    fileh5.close()  
    
    

    
    #%%
        # plot our explanations

    m = fill_segmentation(shap_values[inds[0]][0], segments_slic)
    evm = fill_segmentation(event, segments_slic)
    xplot = xx2[:,iizz,:]
    yplot = -yy2[:,iizz,:]
    duplot = img_orig_real[iimaxy_pos:,iizz,:,0]
    dvplot = -img_orig_real[iimaxy_pos:,iizz,:,1]
    dwplot = img_orig_real[iimaxy_pos:,iizz,:,2]
    contplot = np.heaviside(segments_slic1[iimaxy_pos:,iizz,:],0) 
    shapplot = m[iimaxy_pos:,iizz,:]
    eventplot = evm[iimaxy_pos:,iizz,:]
    fileh5 = h5py.File('explain/explain_uvwb'+str(im_num)+'.h5','w')
    fileh5.create_dataset('xplot', data=xplot)
    fileh5.create_dataset('yplot', data=yplot)
    fileh5.create_dataset('duplot', data=duplot)
    fileh5.create_dataset('dvplot', data=dvplot)
    fileh5.create_dataset('dwplot', data=dwplot)
    fileh5.create_dataset('contplot', data=contplot)
    fileh5.create_dataset('shapplot', data=shapplot)
    fileh5.create_dataset('eventplot', data=eventplot)
    
    type_quadrant = np.zeros((len(xplot[:,0]),len(xplot[0,:])))
    for iiaux in np.arange(len(xplot[:,0])):
        for jjaux in np.arange(len(xplot[0,:])):
            if duplot[iiaux,jjaux] > 0 and dvplot[iiaux,jjaux] > 0:
                type_quadrant[iiaux,jjaux] = 1 
            elif duplot[iiaux,jjaux] < 0 and dvplot[iiaux,jjaux] > 0:
                type_quadrant[iiaux,jjaux] = 2
            elif duplot[iiaux,jjaux] < 0 and dvplot[iiaux,jjaux] < 0:
                type_quadrant[iiaux,jjaux] = 3  
            elif duplot[iiaux,jjaux] > 0 and dvplot[iiaux,jjaux] < 0:
                type_quadrant[iiaux,jjaux] = 4 
            else:
                type_quadrant[iiaux,jjaux] = 0
    fileh5.create_dataset('type',data=type_quadrant)
    fileh5.close()
#%%
    import csv

    with open('data_x.csv', 'w') as ff:      
        # using csv.writer method from CSV package
        write = csv.writer(ff)      
        write.writerows(data_step['x'])
    with open('data_y.csv', 'w') as ff:      
        # using csv.writer method from CSV package
        write = csv.writer(ff)      
        write.writerows(data_step['y'])
    with open('data_z.csv', 'w') as ff:      
        # using csv.writer method from CSV package
        write = csv.writer(ff)      
        write.writerows(data_step['z'])
    with open('data_N_struc.csv', 'w') as ff:      
        # using csv.writer method from CSV package
        write = csv.writer(ff)      
        write.writerow(data_step['N_struc'])
    with open('data_SHAP.csv', 'w') as ff:      
        # using csv.writer method from CSV package
        write = csv.writer(ff)      
        write.writerows(data_step['SHAP'])
    with open('data_dx.csv', 'w') as ff:      
        # using csv.writer method from CSV package
        write = csv.writer(ff)      
        write.writerows(data_step['dx'])
    with open('data_dz.csv', 'w') as ff:      
        # using csv.writer method from CSV package
        write = csv.writer(ff)      
        write.writerows(data_step['dz'])
    with open('data_ymin.csv', 'w') as ff:      
        # using csv.writer method from CSV package
        write = csv.writer(ff)      
        write.writerows(data_step['ymin'])
    with open('data_ymax.csv', 'w') as ff:      
        # using csv.writer method from CSV package
        write = csv.writer(ff)      
        write.writerows(data_step['ymax'])
    with open('data_volume.csv', 'w') as ff:      
        # using csv.writer method from CSV package
        write = csv.writer(ff)      
        write.writerows(data_step['volume'])
    with open('event.csv', 'w') as ff:      
        # using csv.writer method from CSV package
        write = csv.writer(ff)      
        write.writerows(data_step['event'])