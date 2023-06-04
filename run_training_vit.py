from cmath import sqrt
from ctypes import resize
from gc import callbacks
import os
from traceback import print_tb
import re
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
import sys 
import einops
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K

os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'

physical_devices = tf.config.list_physical_devices('GPU')
availale_GPUs = len(physical_devices) 

print('Using TensorFlow version: ', tf.__version__, ', GPU:', availale_GPUs)
print('Using Keras version: ', tf.keras.__version__)


if physical_devices:

  try:

    for gpu in physical_devices:

      tf.config.experimental.set_memory_growth(gpu, True)

  except RuntimeError as e:

    print(e)

def main(epochs):

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    
    #Load training and valdiation dataset 
    dataset_train, dataset_valid, nx, nz = load_tfrecords(tfr_path, yp_flow, validation_split, GLOBAL_BATCH_SIZE, n_prefetch, shuffle_buffer)

    with strategy.scope():
        
        ##### Generate vision transformer ##### 
        vit_regressor = create_vit_regressor()
        print(vit_regressor.summary())

        if pretrained == True:
            vit_regressor = tf.keras.models.load_model(f'{root_folder}models/checkpoints_{model_name}/')
            print('Succesfully loaded transformer network')
            tLoss_old = np.load(root_folder+f'Losses.npz')['training_loss']
            vLoss_old = np.load(root_folder+f'Losses.npz')['validation_loss']
            initial_epoch = len(tLoss_old) 

        else:
            initial_epoch = 0

        ##### Define checkpoints #####
        checkpoints = ModelCheckpoint(f'{root_folder}models/checkpoints_{model_name}', save_best_only=True,verbose=1)

        ##### Define learning rate schedule #####
        lr = WarmupCosineDecay(total_steps=total_steps, 
                             warmup_steps=warmup_steps,
                             hold=int(warmup_steps/4), 
                             start_lr=start_lr, 
                             target_lr=target_lr)

        ##### Define callbacks #####
        callbacks = [checkpoints,lr]


        ##### Define optimizer #####
        j = 1
        exclude_list = []
        for layer in vit_regressor.layers:

            if layer.name == f'Linear_{j}':
                bias = layer.get_weights()[1] 
                tf_var = tf.Variable(bias)
                exclude_list.append(tf_var)
                
                j += 1
        
        optimizer = tfa.optimizers.AdamW(learning_rate=start_lr, weight_decay=[weight_decay,exclude_from_weight_decay])
        # optimizer.exclude_from_weight_decay(var_list = exclude_list,var_names=None)

        ##### Compile model #####
        vit_regressor.compile(
            optimizer=optimizer, 
                # Loss function to minimize
            loss = tf.keras.losses.MeanSquaredError(),
                # List of metrics to monitor
            metrics = [tf.keras.metrics.MeanSquaredError(name='mean_squared_error', dtype=None)]
            )

    epochs = initial_epoch + epochs 

    ##### Train model #####
    history = vit_regressor.fit(
        dataset_train,
        epochs=epochs,
        callbacks = callbacks,
        validation_data = dataset_valid,
        verbose = 2, initial_epoch = initial_epoch)

    ##### Saving history #####
    tLoss = history.history['loss']
    vLoss = history.history['val_loss']

    if pretrained == True:
        tLoss = np.concatenate((tLoss_old,tLoss))
        vLoss = np.concatenate((vLoss_old,vLoss))

    epoch_vec = np.arange(1,epochs+1,1)

    np.savez(root_folder+f'Losses', training_loss=tLoss, validation_loss=vLoss) 

    ##### Plots losses #####
    plt.figure(1)
    plt.semilogy(epoch_vec,tLoss,color='blue',label='Training',ls = None)
    plt.semilogy(epoch_vec,vLoss,color='red',label='Validation',ls = None)
    plt.title('Transformer loss')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss')
    plt.legend(loc = "upper right")
    plt.tight_layout()
    plt.savefig(root_folder+f'Losses.png')

    ##### Plot Learning rate #####
    plt.figure(2)
    plt.plot(lr.lrs)
    plt.xlabel('Steps')
    plt.ylabel('Learning rate')
    plt.tight_layout()
    plt.savefig('Learning_rate.png')   

class WarmupCosineDecay(keras.callbacks.Callback):
    
    def __init__(self, total_steps=0, warmup_steps=0, start_lr=0.0, target_lr=1e-3, hold=0):

        super(WarmupCosineDecay, self).__init__()
        self.start_lr = start_lr
        self.hold = hold
        self.total_steps = total_steps
        self.global_step = 0
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps
        self.lrs = []

    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = self.model.optimizer.lr.numpy()
        self.lrs.append(lr)

    def on_batch_begin(self, batch, logs=None):
        lr = lr_warmup_cosine_decay(global_step=self.global_step,
                                    total_steps=self.total_steps,
                                    warmup_steps=self.warmup_steps,
                                    start_lr=self.start_lr,
                                    target_lr=self.target_lr,
                                    hold=self.hold)
        K.set_value(self.model.optimizer.lr, lr)

def get_lr_metric(optimizer):
    lrr = optimizer.lr
    return lrr

def lr_warmup_cosine_decay(global_step,
                           warmup_steps,
                           hold = 0,
                           total_steps=0,
                           start_lr=0.0,
                           target_lr=1e-3):
    # Cosine decay
    learning_rate = 0.5 * target_lr * (1 + np.cos(np.pi * (global_step - warmup_steps - hold) / float(total_steps - warmup_steps - hold)))

    # Target LR * progress of warmup (=1 at the final warmup step)
    warmup_lr = target_lr * (global_step / warmup_steps)

    # Choose between `warmup_lr`, `target_lr` and `learning_rate` based on whether `global_step < warmup_steps` and we're still holding.
    # i.e. warm up if we're still warming up and use cosine decayed lr otherwise
    if hold > 0:
        learning_rate = np.where(global_step > warmup_steps + hold,
                                 learning_rate, target_lr)
    
    learning_rate = np.where(global_step < warmup_steps, warmup_lr, learning_rate)
    
    return learning_rate

class Patches(layers.Layer):
    
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size
        # self.num_patches = num_patches

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])

        return patches

class PatchEncoder(layers.Layer):

    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim, name='encoding')

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

def create_vit_regressor():

    inputs = layers.Input(shape=input_shape)

    patches = Patches(patch_size)(inputs)

    patches  = tf.convert_to_tensor(patches)

    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    j = 1

    for ii in range(transformer_layers):
        
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=hidden_dim, dropout=drop_rate)(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(j, x3, hidden_units=transformer_units, dropout_rate=drop_rate)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])
        j = ii + 2

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    proj_1 = layers.Dense(units=channels*patch_size**2,name=f'Linear_{j+1}')
    representation = proj_1(representation)

    representation = tf.keras.layers.Reshape([dim, dim, 3*projection_dim])(representation)
    
    ## Using einops
    representation = einops.rearrange(representation, "a h w (p q c) ->  a (h p) (w q) c", p=patch_size, q=patch_size)
    
    ## Using tf.reshape 
    # x = tf.reshape(representation,[-1,dim, dim, patch_size, patch_size, channels]) 
    # x = tf.transpose(x,(0,2,3,1,4,5))
    # x = tf.reshape(x,[-1,dim*patch_size, dim*patch_size, channels])
    model = keras.Model(inputs=inputs, outputs=representation)

    return model

def mlp(j, x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation='relu',name=f'Linear_{j}')(x)
        x = layers.Dropout(dropout_rate)(x)
        j = j+1
    return x

def load_tfrecords(tfr_path, yp_flow, validation_split, batch_size, n_prefetch, shuffle_buffer):
    
    tfr_files = sorted([os.path.join(tfr_path, f) for f in os.listdir(tfr_path) if os.path.isfile(os.path.join(tfr_path,f))])
    regex = re.compile(f'.tfrecords')
    tfr_files = sorted([string for string in tfr_files if re.search(regex, string)])
    regex = re.compile(f"yp{yp_flow}")
    tfr_files = sorted([string for string in tfr_files if re.search(regex, string)])
    regex = re.compile(f"interv")
    tfr_files = sorted([string for string in tfr_files if re.search(regex, string)])
    n_samples_per_tfr = np.array([int(s.split('.')[-2][-3:]) for s in tfr_files])
    n_samples_per_tfr = n_samples_per_tfr[np.argsort(-n_samples_per_tfr)]
    cumulative_samples_per_tfr = np.cumsum(np.array(n_samples_per_tfr))
    tot_samples_per_ds = sum(n_samples_per_tfr)
    n_tfr_loaded_per_ds = int(tfr_files[0].split('_')[-2][-3:])//factor
    n_tfr_loaded_per_ds = int(n_tfr_loaded_per_ds)
    tfr_files = [string for string in tfr_files if int(string.split('_')[-2][:3])<= n_tfr_loaded_per_ds]
    n_samp_train = int(sum(n_samples_per_tfr) * (1 - validation_split))
    n_samp_valid = sum(n_samples_per_tfr) - n_samp_train

    (n_files_train, samples_train_left) = np.divmod(n_samp_train, n_samples_per_tfr[0])

    if samples_train_left > 0:
        n_files_train += 1

    tfr_files_train = [string for string in tfr_files if int(string.split('_')[-2][:3])<= n_files_train]
    n_tfr_left = np.sum(np.where(cumulative_samples_per_tfr < samples_train_left, 1, 0)) + 1

    if sum([int(s.split('.')[-2][-3:])for s in tfr_files_train]) != n_samp_train:
        shared_tfr = tfr_files_train[-1]
        tfr_files_valid = [shared_tfr]

    else:
        shared_tfr = ''
        tfr_files_valid = list()

    tfr_files_valid.extend([string for string in tfr_files if string not in tfr_files_train])
    tfr_files_valid = sorted(tfr_files_valid)

    if temporal == True:
        (nx, nz, ny) = [int(val) for val in tfr_files[0].split('_')[-9].split('x')]
    elif temporal == False:
        (nx, nz, ny) = [int(val) for val in tfr_files[0].split('_')[-8].split('x')]
    
    shared_tfr_out = tf.constant(shared_tfr)
    n_tfr_per_ds = tf.constant(n_tfr_loaded_per_ds)
    n_samples_loaded_per_tfr = list()

    if n_tfr_loaded_per_ds>1:
        n_samples_loaded_per_tfr.extend(n_samples_per_tfr[:n_tfr_loaded_per_ds-1])
        n_samples_loaded_per_tfr.append(tot_samples_per_ds - cumulative_samples_per_tfr[n_tfr_loaded_per_ds-2])

    else:
        n_samples_loaded_per_tfr.append(tot_samples_per_ds)
    
    n_samples_loaded_per_tfr = np.array(n_samples_loaded_per_tfr)
    
    if shuffle_samples == True:
        tfr_files_train_ds = tf.data.Dataset.list_files(tfr_files_train, seed=666)
        tfr_files_val_ds = tf.data.Dataset.list_files(tfr_files_valid, seed=686)

    elif shuffle_samples == False:
        tfr_files_train_ds = tf.data.Dataset.list_files(tfr_files_train,shuffle=False)
        tfr_files_val_ds = tf.data.Dataset.list_files(tfr_files_valid,shuffle=False)

    if n_tfr_left-1>0:
        samples_train_shared = samples_train_left - cumulative_samples_per_tfr[n_tfr_left-2]
        n_samples_tfr_shared = n_samples_loaded_per_tfr[n_tfr_left-1]

    else:
        samples_train_shared = samples_train_left
        n_samples_tfr_shared = n_samples_loaded_per_tfr[0]

    tfr_files_train_ds = tfr_files_train_ds.interleave(lambda x : tf.cond(x == shared_tfr_out, lambda: tf.data.TFRecordDataset(x).take(samples_train_shared), lambda: tf.data.TFRecordDataset(x).take(tf.gather(n_samples_loaded_per_tfr,tf.strings.to_number(tf.strings.split(tf.strings.split(x, sep='_')[-2],sep='-')[0],tf.int32)-1))), cycle_length=16, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    tfr_files_val_ds = tfr_files_val_ds.interleave(lambda x : tf.cond(x == shared_tfr_out, lambda: tf.data.TFRecordDataset(x).skip(samples_train_shared).take(n_samples_tfr_shared -samples_train_shared), lambda: tf.data.TFRecordDataset(x).take(tf.gather(n_samples_loaded_per_tfr, tf.strings.to_number(tf.strings.split(tf.strings.split(x, sep='_')[-2],sep='-')[0],tf.int32)-1))),cycle_length=16,num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset_train = tfr_files_train_ds.map(lambda x: tf_parser(x,tfr_path,yp_flow), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset_valid = tfr_files_val_ds.map(lambda x: tf_parser(x,tfr_path,yp_flow), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if shuffle_samples == True:    
        dataset_train = dataset_train.shuffle(shuffle_buffer)
        dataset_valid = dataset_valid.shuffle(shuffle_buffer)

    dataset_train = dataset_train.batch(batch_size=batch_size)
    dataset_train = dataset_train.prefetch(n_prefetch)
    dataset_valid = dataset_valid.batch(batch_size=batch_size)
    dataset_valid = dataset_valid.prefetch(n_prefetch) 

    return dataset_train, dataset_valid, nx, nz

@tf.function
def tf_parser(rec,tfr_path,yp_flow):

    if temporal == True:
        features = {'i_samp': tf.io.FixedLenFeature([], tf.int64),'n_s': tf.io.FixedLenFeature([], tf.int64),'n_x': tf.io.FixedLenFeature([], tf.int64),'n_y': tf.io.FixedLenFeature([], tf.int64),'n_z': tf.io.FixedLenFeature([], tf.int64),'wall_raw1': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),'wall_raw2': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),'wall_raw3': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),'flow_raw1': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),'flow_raw2': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),'flow_raw3': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),}
    
    elif temporal == False:
        features = {'i_samp': tf.io.FixedLenFeature([], tf.int64),'n_x': tf.io.FixedLenFeature([], tf.int64),'n_y': tf.io.FixedLenFeature([], tf.int64),'n_z': tf.io.FixedLenFeature([], tf.int64),'wall_raw1': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),'wall_raw2': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),'wall_raw3': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),'flow_raw1': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),'flow_raw2': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),'flow_raw3': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),}
    

    parsed_rec = tf.io.parse_single_example(rec, features)

    nx = tf.cast(parsed_rec['n_x'], tf.int32)
    nz = tf.cast(parsed_rec['n_z'], tf.int32)
    
    if temporal == True:
        ns = tf.cast(parsed_rec['n_s'], tf.int32)

    # Scaling data at wall

    print('The inputs are normalized to have a unit Gaussian distribution')
    
    with np.load(tfr_path+f'scaling_yp{yp_flow:03d}.npz') as data:

        avgs_in = tf.constant(data['X_mean'].astype(np.float32))
        avgs_out = tf.constant(data['Y_mean'].astype(np.float32))
        std_in = tf.constant(data['X_std'].astype(np.float32))
        std_out = tf.constant(data['Y_std'].astype(np.float32))
   

    # Low-resolution processing --------------------------------------------------------
    
    if temporal == True:
        inputs = tf.reshape((parsed_rec['wall_raw1']-avgs_in[0])/std_in[0], (ns, nz, nx, 1))[:, ::us, ::us, :]

        for i_comp in range(1, channels):

            inputs = tf.concat((inputs, tf.reshape((parsed_rec[f'wall_raw{i_comp+1}']-avgs_in[i_comp])/std_in[i_comp], (ns, nz, nx, 1))[:, ::us, ::us, :]), -1)
        
        outputs = tf.reshape((parsed_rec['flow_raw1']-avgs_out[0])/std_out[0], (nz, nx, 1))

        for i_comp in range(1, channels):

            outputs = tf.concat((outputs, tf.reshape((parsed_rec[f'flow_raw{i_comp+1}']-avgs_out[i_comp])/std_out[i_comp], (nz, nx, 1))), -1)

    elif temporal == False:
        
        inputs = tf.reshape((parsed_rec['wall_raw1']-avgs_in[0])/std_in[0], (nz, nx, 1))

        for i_comp in range(1, channels):

            inputs = tf.concat((inputs, tf.reshape((parsed_rec[f'wall_raw{i_comp+1}']-avgs_in[i_comp])/std_in[i_comp],(nz, nx, 1))),-1)
    
        outputs = tf.reshape((parsed_rec['flow_raw1']-avgs_out[0])/std_out[0], (nz, nx, 1))
        for i_comp in range(1, channels):

            outputs = tf.concat((outputs, tf.reshape((parsed_rec[f'flow_raw{i_comp+1}']-avgs_out[i_comp])/std_out[i_comp], (nz, nx, 1))), -1)



    return inputs, outputs


if __name__ == '__main__':

    #Physical parameters 
    Ret = 180
    yp_wall = 0 
    yp_flow = 15
    interv = 3
    channels = 3

    #Data parameters 
    input_shape = (192, 192, channels)
    n_samples = (4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200)
    factor = 1
    batch_size = 16
    n_prefetch = 16
    GLOBAL_BATCH_SIZE = batch_size*availale_GPUs 
    availale_GPUs = len(physical_devices) 
    physical_devices = tf.config.list_physical_devices('GPU')
    shuffle_buffer = 12000
    shuffle_samples = True
    temporal = False

    #Paths
    tfr_path = "/proj/deepmech/users/adrian/tf_records/wall/wall_u15/"
    root_folder = "/proj/deepmech/users/adrian/simulations/transformers/wall_u15/network_arc/sim_test/"

    # Image parameters 
    image_size = 192 #We'll resize input images to this size
    num_patches = 24*24
    patch_size = int(image_size // (num_patches)**(0.5)) 
    
    if int(image_size/patch_size) != image_size/patch_size:
        print('ERROR IMAGE SIZE DIVIDED BY PATCH SIZE NOT EVEN NUMBER')
        exit()

    dim = int(np.real(sqrt(num_patches)))
    projection_dim = patch_size**2

    #Network parameters
    epochs = 80

    pretrained = False
    model_name = "architecture-01"
    validation_split = 0.2
    weight_decay = 0.02
    drop_rate = 0.1

    hidden_dim = 4*patch_size**2*channels
    num_heads = 12
    transformer_units = [2*projection_dim,projection_dim] #Size of the transformer layers
    transformer_layers = 1
    
    #Learning rate parameters 
    start_lr = 5e-4
    target_lr = 5e-4
    warmup_epoch = 5
    total_steps = epochs*batch_size
    warmup_steps = int(warmup_epoch*batch_size)

    print('####################################')
    print('Number of patches:', num_patches)
    print('Number of patches figure:', (num_patches)**(0.5),'x',(num_patches)**(0.5))
    print('Size of patches:',patch_size,'x',patch_size)
    print('Number of heads: ',num_heads)
    print('Number of transformer layers: ', transformer_layers)
    print('Batch size: ',batch_size)
    print('Weight decay: ',weight_decay)
    print('####################################')

    
main(epochs)


