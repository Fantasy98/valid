#%%
import torch 
import os
from torch import nn 
from utils.networks import FCN
from utils.datas import slice_dir
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import shap
import numpy as np
from scipy import signal
from utils.plots import Plot_2D_snapshots,Plot_multi
#%%
np_path = "/home/yuning/thesis/valid/results/y_plus_30/2d_correlation/"
pr0025 = os.path.join(np_path,"pr0025.npy")
pr0025_np = np.load(pr0025)
pr02 = os.path.join(np_path,"pr02.npy")
pr02_np = np.load(pr02)
pr071 = os.path.join(np_path,"pr071.npy")
pr071_np = np.load(pr071)
pr1 = os.path.join(np_path,"pr1.npy")
pr1_np = np.load(pr1)
# %%
names=["u","v","w",r"$\theta$"]
prs = ["0.025","0.2","0.71","1"]
for i in range(4):
    corr_feature = np.empty(shape=(4,256,256))
    
    corr_feature[0,:,:] = pr0025_np[i,:,:]
    corr_feature[1,:,:] = pr02_np[i,:,:]
    corr_feature[2,:,:] = pr071_np[i,:,:]
    corr_feature[3,:,:] = pr1_np[i,:,:]
    name_list = [  names[i]+"@"+"pr="+pr for pr in prs  ]
    Plot_multi(corr_feature,name_list,"y_plus30_allpr_feature{}_normlized".format(names[i]))
# %%
