#%%
import os
from tensorflow import keras
from keras import layers
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback
from DataHandling.features import slices
from DataHandling import utility
from DataHandling.models import predict
os.environ['WANDB_DISABLE_CODE']='True'
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[-1], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass
#%%
model_name = "sage-universe-33"
overwrite = False
var=['u_vel',"v_vel","w_vel","pr0.025"]
target=['pr0.025_flux']
normalized=False
y_plus=30
# %%
from keras import models
model = models.load_model("/home/yuning/thesis/models/trained/sage-universe-33")
#%%
predict.predict(model_name,overwrite,model,y_plus,var,target,normalized,test=False)
    
# %%
