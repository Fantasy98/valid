#%%
# Test to use model predict a single snap shot 
# The model is far more to be trained
import os
from tensorflow import keras
from keras import layers
import tensorflow as tf
import wandb
import numpy as np
import matplotlib.pyplot as plt
from DataHandling.features.slices import read_tfrecords,feature_description,slice_loc
from wandb.keras import WandbCallback
from DataHandling.features import slices
from DataHandling import utility
from DataHandling.models import predict
os.environ['WANDB_DISABLE_CODE']='True'
# physical_devices = tf.config.list_physical_devices('GPU')
# try:
#   tf.config.experimental.set_memory_growth(physical_devices[-1], True)
# except:
#   # Invalid device or cannot modify virtual devices once initialized.
#   pass
#%%
model_name = "fresh-glitter-54"
all_path = "/home/yuning/thesis/models/trained/"
model_path = os.path.join(all_path,model_name)
print(model_path)
overwrite = False
var=['u_vel',"v_vel","w_vel","pr0.025"]
target=['pr0.025_flux']
normalized=False
y_plus=30
# %%
from keras import models
model = models.load_model(model_path)
print(model.summary())
# %%
file_path = slice_loc(y_plus,var,target,normalized=False)
path_test = os.path.join(file_path,"validation")
print(path_test)
dataset = tf.data.TFRecordDataset(
                                    filenames=path_test,
                                    compression_type="GZIP",
                                    num_parallel_reads=tf.data.experimental.AUTOTUNE
                                    )

feature_dict = feature_description(file_path)
print(feature_dict)

#%%
num = 0
for snap in dataset:
    # parsed = tf.io.parse_single_example(snap,feature_dict)
    num+=1
   


# %%
pred_pr = model.predict(inputs)
#%%

pred = np.squeeze(pred_pr)
avg = np.sqrt( (target_array-pred)**2)
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
plt.figure(figsize=(15*cm,10*cm),dpi=500)
clb =plt.contourf(xx, yy, avg, cmap='jet')
plt.colorbar(clb)
plt.xlabel(r'$x^+$',fontdict={"size":15})
plt.ylabel(r'$z^+$',fontdict={"size":15})
plt.xticks(placement_x)
plt.yticks(placement_z)
    
#%%
# fig_path = "/home/yuning/thesis/valid/fig"
# fig_path_save = os.path.join(fig_path,model_name)
# if os.path.exists(fig_path_save) is False:
#   os.mkdir(fig_path_save)

# fig =plt.figure(0)
# clc = plt.imshow(np.squeeze(pred_pr),cmap = "jet")
# plt.title("Prediction")
# plt.colorbar(clc)

# fig.savefig(os.path.join(fig_path_save,"Prediction"))
# fig =plt.figure(1)
# clc = plt.imshow(target_array,cmap = "jet")
# plt.colorbar(clc)
# plt.title("Origin")
# fig.savefig(os.path.join(fig_path_save,"Target"))

# %%
pred_array = np.array(PRED)
preds = np.squeeze(pred_array)
targets = np.array(TARGET)
print(preds.shape == targets.shape)
#%%
def RMS_error(pred,true):

  error = 100*np.sqrt( np.mean( (pred-true)**2 ) )/np.mean(true)
  return error
def Glob_error(pred,true):
  return 100*(np.mean(pred)-np.mean(true))/np.mean(true)
# Error = keras.losses.mean_squared_error(target_array,np.squeeze(pred_pr))


rms = RMS_error(np.mean(preds,0),np.mean(targets,0))
globs = Glob_error(np.mean(preds,0),np.mean(targets,0))
print("RMSE = {}%".format(rms))
print("Glob_Error = {}%".format(globs))
#%%
fig =plt.figure(3)

clc = plt.imshow(np.sqrt((target_array-np.squeeze(pred_pr))**2),cmap = "jet")
plt.colorbar(clc)
plt.title("Error")
fig.savefig(os.path.join(fig_path_save,"RMSE"))

# %%
