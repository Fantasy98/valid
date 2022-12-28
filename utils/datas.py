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
    case_path = slice_dir(root_path,y_plus, var,target,save_type,normalized)
    
    
    feature_dict = feature_description(file_path)

    dataset = tf.data.TFRecordDataset(
                                    filenames=path_test,
                                    compression_type="GZIP",
                                    num_parallel_reads=tf.data.experimental.AUTOTUNE
                                    )
    num_snap=0
    for i in dataset:
        num_snap +=1 
    print(f"There are {num_snap} in dataset")

    names = list(feature_dict.keys())
    for tar in target:
        names.remove(tar)
    names.sort()
    features = []
    y = []
    indx = 0; t = 0
    for snap in tqdm(dataset.cache()):
        indx +=1
        (dict_for_dataset,target_array) = read_tfrecords(snap,feature_dict,target)
        snap_list = []  
        tar_list = []
        for name in names:
            snap_list.append(torch.from_numpy(dict_for_dataset[name].numpy()))
        snap_tensor = torch.stack(snap_list,dim=0)
        
        target_array = target_array.numpy()
        for tar in target:
            tar_list.append(torch.from_numpy(target_array))

        tar_tensor = torch.stack(tar_list,dim=0)
        features.append(snap_tensor)
        y.append(tar_tensor)
        if indx % (num_snap//2) == 0:
            t +=1
            features_tensor = TensorDataset(torch.stack(features,dim=0))
            targets_tensor = TensorDataset(torch.stack(y,dim=0))
            features.clear()
            y.clear()
            torch.save(features_tensor,case_path+"/{}{}.pt".format("features",t))
            print(f"The {t} part of feature has been saved, shape = {features_tensor.tensors[0].size()}")
            torch.save(targets_tensor,case_path+"/{}{}.pt".format("targets",t))
            print(f"The {t} part of target has been saved, shape = {targets_tensor.tensors[0].size()}")
            
    # features_tensor = TensorDataset(torch.stack(features,dim=0))
    # tragets_tensor = TensorDataset(torch.stack(y,dim=0))

    # print(f"feature dataset has shape of {features_tensor.tensors[0].size()}")
    # print(f"targets dataset has shape of {tragets_tensor.tensors[0].size()}")
    
    # torch.save(features_tensor,case_path+"/{}.pt".format("features"))
    # torch.save(tragets_tensor,case_path+"/{}.pt".format("targets"))

    