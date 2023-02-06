#%%
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
#%%
# Specific target and variables
var=['u_vel',"v_vel","w_vel","pr0.025"]
target=['pr0.025_flux']
normalized=False
y_plus=30

#Parameters 
dropout=False
skip=4
model_type="baseline"
repeat=3
shuffle=100
batch_size=10
activation='elu'
optimizer="adam"
loss='mean_squared_error'
patience=50

data=slices.load_from_scratch(y_plus,var,target,normalized,repeat=repeat,shuffle_size=shuffle,batch_s=batch_size)
train=data[0]
validation=data[1]
# %%
#Recompile data into Dataset by 
for example in validation:
    print(example[0].keys())
    
    print(example[0]["u_vel"].numpy().shape)
    u_vel = example[0]["u_vel"].numpy()[0]
    v_vel = example[0]["v_vel"].numpy()[0]
    w_vel = example[0]["w_vel"].numpy()[0]
    pr = example[0]["pr0.025"].numpy()[0]
    break
# %%
import matplotlib.pyplot as plt
clc=plt.imshow(u_vel)
plt.colorbar(clc)

#%%
plt.imshow(v_vel)
#%%
plt.imshow(w_vel)
#%%
plt.imshow(pr)

# %%
## Illustrate how the input data is concatenated
import numpy as np
vel = np.concatenate([np.expand_dims(u_vel,-1),np.expand_dims(v_vel,-1),np.expand_dims(w_vel,-1)],axis=-1)
plt.clf()
plt.imshow(vel)
input= layers.Input(shape = (256,256,1))
conv = layers.Concatenate()([input,input,input])
model = keras.Model(input,conv)
print(model.summary())
# %%
## How Periodic Padding work?
from keras.layers import InputSpec 
from tensorflow.python.keras.utils import conv_utils
class PeriodicPadding2D(keras.layers.Layer):
        def __init__(self, padding=1, **kwargs):
            super(PeriodicPadding2D, self).__init__(**kwargs)
            self.padding = conv_utils.normalize_tuple(padding, 1, 'padding')
            self.input_spec = InputSpec(ndim=3)

        def wrap_pad(self, input, size):
            M1 = tf.concat([input[:,:, -size:], input, input[:,:, 0:size]], 2)
            M1 = tf.concat([M1[:,-size:, :], M1, M1[:,0:size, :]], 1)
            return M1

        def compute_output_shape(self, input_shape):
            shape = list(input_shape)
            assert len(shape) == 3  
            if shape[1] is not None:
                length = shape[1] + 2*self.padding[0]
            else:
                length = None
            return tuple([shape[0], length, length])

        def call(self, inputs): 
            return self.wrap_pad(inputs, self.padding[0])

        def get_config(self):
            config = {'padding': self.padding}
            base_config = super(PeriodicPadding2D, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))
padding_layers = 8
input_list=[]
reshape_list=[]
input_features = ["a","b","c"]
for features in input_features:
        input=keras.layers.Input(shape=(256,256),name=features)
        input_list.append(input)
        pad=PeriodicPadding2D(padding=padding_layers)(input)
        reshape=keras.layers.Reshape((256+2*padding_layers,256+2*padding_layers,1))(pad)
        reshape_list.append(reshape)
conc=keras.layers.Concatenate()(reshape_list)

from keras.backend import clear_session
clear_session()

model_padding = keras.Model(inputs=input_list,outputs=conc)
print(model_padding.summary())
# %%

vel_padding = model_padding([np.expand_dims(u_vel,0),np.expand_dims(v_vel,0),np.expand_dims(w_vel,0)])
val = vel_padding.numpy()
for i in range(3):
    clb = plt.imshow(val[0,:,:,i])
    plt.colorbar(clb)
# %%
## Normalizing data
u_normal = (u_vel - np.mean(u_vel))/np.std(u_vel)
clc = plt.imshow(u_normal)
plt.colorbar(clc)
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
# logdir, backupdir= utility.get_run_dir(wandb.run.name)



# backup_cb=tf.keras.callbacks.ModelCheckpoint(os.path.join(backupdir,'weights.{epoch:02d}'),save_best_only=False)
# early_stopping_cb = keras.callbacks.EarlyStopping(patience=patience,
# restore_best_weights=True)
model.fit(x=train,epochs=10,validation_data=validation)

# %%
