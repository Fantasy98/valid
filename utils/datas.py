import torch 
from torch.utils.data import TensorDataset
from tqdm import tqdm
import os
import tensorflow as tf
from DataHandling.features.slices import read_tfrecords,feature_description,slice_loc

from torch.utils.data import Dataset
class JointDataset(Dataset):
    def __init__(self,x,y) -> None:
        self.x = x 
        self.y = y 
    def __getitem__(self, index):
        return self.x[index,:,:,:],self.y[index,:,:,:]
    def __len__(self):
        return len(self.x)


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



def TF2Torch(root_path,y_plus,var,target,save_type,normalized):
    # Function for transfer tensorflow dataset into Pytorch data format".pt"
    

    file_path = slice_loc(y_plus,var,target,normalized=normalized)
    path_test = os.path.join(file_path,save_type)
    case_path = slice_dir(root_path,y_plus, var,target,save_type,normalized)
    
    
    feature_dict = feature_description(file_path)
    print(feature_dict)
    dataset = tf.data.TFRecordDataset(
                                    filenames=path_test,
                                    compression_type="GZIP",
                                    num_parallel_reads=tf.data.experimental.AUTOTUNE
                                    ).cache()
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
    
    
        # if indx % (num_snap//2) == 0:
        #     t +=1
    features_tensor = TensorDataset(torch.stack(features,dim=0))
    targets_tensor = TensorDataset(torch.stack(y,dim=0))
    features.clear()
    y.clear()
    torch.save(features_tensor,case_path+"/{}.pt".format("features"))
    print(f"The {t} part of feature has been saved, shape = {features_tensor.tensors[0].size()}")
    torch.save(targets_tensor,case_path+"/{}.pt".format("targets"))
    print(f"The {t} part of target has been saved, shape = {targets_tensor.tensors[0].size()}")


def mkdataset(root_path,y_plus,var,target,save_type,normalized=False):
    import os
    file_path = slice_loc(y_plus,var,target,normalized=False)
    path_test = os.path.join(file_path,save_type)
    case_path = slice_dir(root_path,y_plus, var,target,save_type,normalized)
    
    list_dir = os.listdir(case_path)
    file_name = save_type+".pt"
    if file_name in list_dir:
        print("Dataset has already exist!")
        return

    # for i in range(2):
    feature_tensor = torch.load(case_path+f"/features.pt")
    target_tensor = torch.load(case_path+f"/targets.pt")

    print(f" Feature data loaded, shape = {feature_tensor.tensors[0].size()}")
    print(f" Target data loaded, shape = {target_tensor.tensors[0].size()}")


    jdata = JointDataset(feature_tensor.tensors[0].clone(),target_tensor.tensors[0].clone())
    torch.save(jdata,case_path+"/{}.pt".format(save_type))
    print("jointdataset has been saved")
    
    print("All jointdatasets have been created, now all the tensor will be removed")
    list_dir = os.listdir(case_path)
    for item in list_dir:
        if save_type  not in item:
            rm_path = os.path.join(case_path,item)
            print(f"Removing {rm_path}")
            os.remove(rm_path)
    
    list_dir = os.listdir(case_path)
    print(f"Now left in dir is {list_dir}")
            