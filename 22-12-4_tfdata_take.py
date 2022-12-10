import tensorflow as tf
import numpy as np 

npdata = np.arange(100)
data = tf.data.Dataset.from_tensor_slices(npdata)
data = data.shuffle(100).batch(2)

data1 = data.cache()
sample = data1.take(5).repeat(1).cache()
for elem in sample:
        print(elem)


data1 = data1.shuffle(100)
for i in range(5):
    sample = data1.take(1).cache()
    for elem in sample:
        print(elem)