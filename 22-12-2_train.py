#%%
import numpy as np
import os
from tensorflow import keras
from keras import layers
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback
from DataHandling.features import slices
from DataHandling import utility
from DataHandling.models import models
import xarray as xr
os.environ['WANDB_DISABLE_CODE']='True'
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[-1], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

#%% Specific target and variables
var=['u_vel',"v_vel","w_vel","pr0.025"]
target=['pr0.025_flux']
normalized=False
y_plus=30

#Parameters 
dropout=False
skip=4
model_type="baseline"
repeat=1
shuffle=100
batch_size=2
activation='elu'
optimizer="adam"
loss='mean_squared_error'
patience=50
#%%
data=slices.load_from_scratch(y_plus,var,target,normalized,repeat=repeat,shuffle_size=shuffle,batch_s=batch_size)
train=data[0]
validation=data[1]

from keras.backend import clear_session
clear_session()


# %%
padding=8
model=models.final_skip_no_sep_padding(var,activation,padding)
model.summary()

# %%
wandb.init(project="Thesis",notes="8 layers of periodic padding")



config=wandb.config
config.y_plus=y_plus
config.repeat=repeat
config.shuffle=shuffle
config.batch_size=batch_size
config.activation=activation
config.optimizer=optimizer
config.loss=loss
config.patience=patience
config.variables=var
config.target=target[0]
config.dropout=dropout
config.normalized=normalized
config.skip=skip
config.model=model_type
config.padding=padding



model.compile(loss=loss, optimizer=optimizer)
# %%
logdir, backupdir= utility.get_run_dir(wandb.run.name)
print(logdir)
#%%

backup_cb=tf.keras.callbacks.ModelCheckpoint(os.path.join(backupdir,'weights.{epoch:02d}'),save_best_only=False)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=patience,
restore_best_weights=True)
#%%
from tqdm import tqdm
Loss = []
import time
backup_cb=tf.keras.callbacks.ModelCheckpoint(os.path.join(backupdir,'weights.{epoch:02d}'),save_best_only=False)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=patience,restore_best_weights=True)
Time = time.time()
for i in tqdm(range(500)):
  t = time.time()
  print("On Step {}".format(i))
  train_sub = train.take(1).cache()
  val_sub = validation.take(1).cache()
  history= model.fit(x=train_sub,
                      epochs=2,
                      validation_data=val_sub,
                      # callbacks= [backup_cb, early_stopping_cb]
                      )
  loss = history.history["loss"]
  Loss.append(loss)
  print("Train loss ={}".format(loss))
  interval = time.time() - t 
  print("Time used {}".format(interval))
End_time = time.time() - Time
print("The time totally used {}".format(End_time))
#%%
pth = os.path.join("/home/yuning/thesis/models/trained",wandb.run.name)
model.save(pth)
Loss_array = np.array(Loss)
np.savetxt(os.path.join(pth,"loss.out"),Loss_array)
print("Loss history saved !")
# %%
