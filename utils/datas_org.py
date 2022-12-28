import torch 
from torch.utils.data import TensorDataset
from tqdm import tqdm
import os
import tensorflow as tf
from DataHandling.features.slices import read_tfrecords,feature_description,slice_loc


def parse_name(y_plus:int, var:list,target:list,save_type:str,normalized=False):
    # return the filepath name:
    
    var_sort=sorted(var)
    var_string="_".join(var_sort)
    target_sort=sorted(target)
    target_string="_".join(target_sort)
    if normalized:
            name = 'y_plus_'+str(y_plus)+"-VARS-"+var_string+"-TARGETS-"+target_string+"-normalized"
    else:
            name = 'y_plus_'+str(y_plus)+"-VARS-"+var_string+"-TARGETS-"+target_string
    return name
    

def slice_dir(root_path,y_plus:int, var:list,target:list,save_type:str,normalized=False):
    
    if os.path.exists(root_path) is False:
        os.mkdir(root_path) 
        print(f"The path {root_path} is new, now it is made!")
    name = parse_name(y_plus, var,target,save_type,normalized)

    branch = os.path.join(root_path,name)
    if os.path.exists(branch) is False:
        os.mkdir(branch) 
        print(f"The path {branch} is new, now it is made!")
    
    final_path = os.path.join(branch,save_type)
 
    if os.path.exists(final_path) is False:
        os.mkdir(final_path)
        print(f"The path {final_path} is new, now it is made!")
    return final_path



def TF2Torch(root_path,y_plus,var,target,save_type,normalized=False):
    # Function for transfer tensorflow dataset into Pytorch data format".pt"
    

    file_path = slice_loc(y_plus,var,target,normalized=False)
    path_test = os.path.join(file_path,save_type)
    feature_dict = feature_description(file_path)

    dataset = tf.data.TFRecordDataset(
                                    filenames=path_test,
                                    compression_type="GZIP",
                                    num_parallel_reads=tf.data.experimental.AUTOTUNE
                                    )


    numpy_dict = {}
    names = list(feature_dict.keys())
    names.remove(target[0])
    for name in names:
        numpy_dict[name] = []
    numpy_dict[target[0]] = []

    for snap in tqdm(dataset):
        (dict_for_dataset,target_array) = read_tfrecords(snap,feature_dict,target)
        for name in names:
            value = dict_for_dataset[name].numpy()
            
            numpy_dict[name].append(value)
            
        target_array = target_array.numpy()
        numpy_dict[target[0]].append(target_array)
    
    case_path = slice_dir(root_path,y_plus, var,target,save_type,normalized)
    
    if os.path.exists(case_path) is False:
        os.mkdir(case_path)
        print(f"Made case path {case_path}")
    
    for name in numpy_dict.keys():
        tensor_list = [ torch.tensor(i) for i in  numpy_dict[name] ]
        tensors = TensorDataset(torch.stack(tensor_list))
        print(tensors)
    
        torch.save(tensors,case_path+"/{}.pt".format(name))
        print(f"Tensor {name} has been saved!")
        
    