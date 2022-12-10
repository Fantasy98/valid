# package for predicting the results
from DataHandling.features.slices import read_tfrecords, slice_loc,feature_description
import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
def predict(model,y_plus,var,target,normalized=False):
    
    
    
    file_path = slice_loc(y_plus,var,target,normalized=False)
    path_test = os.path.join(file_path,"test")
    print(path_test)
    dataset = tf.data.TFRecordDataset(
                                        filenames=path_test,
                                        compression_type="GZIP",
                                        num_parallel_reads=tf.data.experimental.AUTOTUNE
                                        )

    feature_dict = feature_description(file_path)
    
    TARGET = []
    INPUTS = []
    for snap in dataset:
        
        (dict_for_dataset,target_array) = read_tfrecords(snap,feature_dict,target)
        inputs = [ np.expand_dims(item,0) for item in dict_for_dataset.values() ]
        names = dict_for_dataset.keys()
       
        pr = inputs[0]
        inputs.pop(0)
        inputs.append(pr)
        
        INPUTS.append(inputs)
        TARGET.append(target_array)

    print("Totally {} test snapshots".format(len(TARGET)))

    PRED = []
    for input in tqdm (INPUTS):
            pred_pr = model.predict(input,verbose= 0)
            PRED.append(pred_pr)

    pred_array = np.array(PRED)
    preds = np.squeeze(pred_array)
    targets = np.array(TARGET)
    if preds.shape == targets.shape:
        print("targets and prediction shape are matched!")

    return (preds,targets)