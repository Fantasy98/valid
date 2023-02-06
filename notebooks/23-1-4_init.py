import torch 
from torch import nn 
from tensorflow import keras
from keras import layers
import numpy as np 

torch.manual_seed(0)

conv= nn.Conv2d(in_channels=3, out_channels=1,kernel_size=3)


print("Generating model in keras")
input = layers.Input(shape=(32,32,1))
conv_keras = layers.Conv2D(3,3,activation="elu",name="conv1")(input)
model = keras.Model(input,conv_keras)
print(model.summary())


print("conv layer in pytorch before init")
print(conv.weight)
print(conv.bias)

nn.init.xavier_uniform_(conv.weight)
nn.init.zeros_(conv.bias)
print("conv layer in pytorch after init not considering gain of elu")
print(conv.weight)
print(conv.bias)

gain = nn.init.calculate_gain("leaky_relu")
print(gain)
nn.init.xavier_uniform_(conv.weight,gain = gain)
nn.init.zeros_(conv.bias)
print("conv layer in pytorch after init considering gain of elu")
print(conv.weight)
print(conv.bias)


print("conv layer in keras")
keras_weight = model.get_layer("conv1").get_weights()[0]
print(keras_weight)
print(type(keras_weight))


# bn = nn.BatchNorm2d(num_features=1)
# print(bn.bias)
# print(bn.weight)
