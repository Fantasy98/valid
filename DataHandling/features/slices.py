

from re import S

# from scipy.fftpack import ss_diff

 
def feature_description(save_loc):
    """Loads the json file descriping the file format for parsing the tfRecords. For now only array_serial has been implemented!
       Furthermore the last entry is allways the target.

    Args:
        save_loc (string): The file path to the folder where the data is saved

    Returns:
        dict: dict used to read tfRecords
    """
    import os
    import tensorflow as tf
    import json
    feature_format={}

    with open(os.path.join(save_loc,"format.json"),'r') as openfile:
        format_json=json.load(openfile)
    


    for key in list(format_json.keys()):
        if format_json[key] =="array_serial":
            feature_format[key]= tf.io.FixedLenFeature([], tf.string, default_value="")
        else:
            print("other features than array has not yet been implemented!")
    return feature_format



def read_tfrecords(serial_data,format,target):
    """Reads the tfRecords and converts them to a tuple where first entry is the features and the second is the targets

    Args:
        serial_data (TFrecordDataset): The output of the function tf.data.TFRecordDataset
        format (dict): dict used to parse the TFrecord example format

    Returns:
        tuple: tuple of (features,labels)
    """
    import tensorflow as tf
      
    features=tf.io.parse_single_example(serial_data, format)

    dict_for_dataset={}

    #Loops through the features and saves them into a dict
    for key, value in features.items():
        if value.dtype == tf.string:
            dict_for_dataset[key]=tf.io.parse_tensor(value,tf.float64)
        else:
            print("only arrays have been implemented")
       
    if len(target)==1:
        target_array=dict_for_dataset[target[0]]
    #elif len(target)==4:
    #    target_array=tf.stack([dict_for_dataset[target[0]],dict_for_dataset[target[1]],dict_for_dataset[target[2]],dict_for_dataset[target[3]]],axis=2)
        
    else:
        target_array_list=[]
        for i in range(len(target)):
            target_array_i=dict_for_dataset[target[i]]
            target_array_list.append(target_array_i)
        target_array=tf.stack(target_array_list,axis=2)


    #Removes the target from the dict
    for i in target:
        dict_for_dataset.pop(i)

     
    return (dict_for_dataset,target_array)






def load_from_scratch(y_plus,var,target,normalized,repeat=10,shuffle_size=100,batch_s=10):
    """Copyes TFrecord to scratch and loads the data from there

    Args:
        y_plus (int): the y_plus value to load data from
        var (list): list of features
        target (list): list of targets. only 1 for now
        repeat (int, optional): number of repeats of the dataset for each epoch. Defaults to 10.
        shuffle_size (int, optional): the size of the shuffle buffer. Defaults to 100.
        batch_s (int, optional): the number of snapshots in each buffer. Defaults to 10.

    Returns:
        [type]: [description]
    """
      
    import tensorflow as tf
    import os
    import shutil
    import xarray as xr
    save_loc=slice_loc(y_plus,var,target,normalized)

    if not os.path.exists(save_loc):
        raise Exception("data does not exist. Make som new")


    #copying the data to scratch
    # scratch=os.path.join('/scratch/', os.environ['SLURM_JOB_ID'])
    pwd = os.getcwd()
    scratch=os.path.join(pwd, 'scratch/')
    # shutil.copytree(save_loc,scratch)
    #print("copying data to scratch")

    features_dict=feature_description(save_loc)


    
    splits=['train','validation','test']

    data=[]
    for name in splits:
        data_loc=os.path.join(save_loc,name)
        # shutil.copy2(os.path.join(save_loc,name),data_loc)
        dataset = tf.data.TFRecordDataset([data_loc],compression_type='GZIP',buffer_size=100,num_parallel_reads=tf.data.experimental.AUTOTUNE)
        dataset=dataset.map(lambda x: read_tfrecords(x,features_dict,target),num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset=dataset.shuffle(buffer_size=shuffle_size)
        dataset=dataset.repeat(repeat)
        dataset=dataset.batch(batch_size=batch_s)
        dataset=dataset.prefetch(3) 
        data.append(dataset)
    return data



def load_validation(y_plus,var,target,normalized,test=False):
    """A load function where the validation, test and train are loaded as one giant batch

    Args:
        y_plus (int): At what y_plus value
        var (list): the variables used as input
        target (list): list of target. So far can only hold one entry
        normalized (Bool): If the data is normalized or not

    Raises:
        Exception: If the data does not exist a exception is raised

    Returns:
        list: list containing the data in the following order [train,validation,test]
    """

      
    import tensorflow as tf
    import os
    import shutil
    from DataHandling.features.slices import slice_loc,feature_description,read_tfrecords
    
    save_loc=slice_loc(y_plus,var,target,normalized,test)

    if not os.path.exists(save_loc):
        raise Exception("data does not exist. Make som new")


    #copying the data to scratch
    pwd = os.getcwd()
    # scratch=os.path.join('/scratch/', os.environ['SLURM_JOB_ID'])
    scratch=os.path.join(pwd,'scratch')
    #shutil.copytree(save_loc,scratch)
    #print("copying data to scratch")

    features_dict=feature_description(save_loc)


    data_unorder=[]

    #validation
    data_loc=os.path.join(scratch,'validation')
    shutil.copy2(os.path.join(save_loc,'validation'),data_loc)
    dataset = tf.data.TFRecordDataset([data_loc],compression_type='GZIP',buffer_size=100,num_parallel_reads=tf.data.experimental.AUTOTUNE)
    dataset=dataset.map(lambda x: read_tfrecords(x,features_dict,target),num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset=dataset.take(-1)
    dataset=dataset.cache()
    len_data_val=len(list(dataset))
    dataset=dataset.batch(len_data_val)
    dataset=dataset.get_single_element()
    data_unorder.append(dataset)

    #train
    data_loc=os.path.join(scratch,'train')
    shutil.copy2(os.path.join(save_loc,'train'),data_loc)
    dataset = tf.data.TFRecordDataset([data_loc],compression_type='GZIP',buffer_size=100,num_parallel_reads=tf.data.experimental.AUTOTUNE)
    dataset=dataset.map(lambda x: read_tfrecords(x,features_dict,target),num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset=dataset.batch(len_data_val)
    #dataset=dataset.batch(10)
    dataset=dataset.take(1)
    for i in dataset:
        dataset=i
    data_unorder.append(dataset)    
    
    
    
    #test
    data_loc=os.path.join(scratch,'test')
    shutil.copy2(os.path.join(save_loc,'test'),data_loc)
    dataset = tf.data.TFRecordDataset([data_loc],compression_type='GZIP',buffer_size=100,num_parallel_reads=tf.data.experimental.AUTOTUNE)
    dataset=dataset.map(lambda x: read_tfrecords(x,features_dict,target),num_parallel_calls=tf.data.experimental.AUTOTUNE)
    len_data_test=len(list(dataset))
    dataset=dataset.batch(len_data_test)
    dataset=dataset.get_single_element()
    data_unorder.append(dataset)

    data=[data_unorder[1],data_unorder[0],data_unorder[2]]

    return data
    



def save_tf(y_plus,var,target,data,normalized=False,var_second=None,y_plus_second=None,test_split=0.1,validation_split=0.2,test=False):
    """Takes a xarray dataset extracts the variables in var and saves them as a tfrecord

    Args:
        y_plus (int): at which y_plus to take a slice
        var (list): list of inputs to save. NOT with target
        target (list): list of target. Only 1 target for now
        data (xarray): dataset of type xarray
        normalized(bool): if the data is normalized or not

    Returns:
        None:
    """

    import os
    import xarray as xr
    import numpy as np
    import dask
    import tensorflow as tf
    from DataHandling import utility
    import shutil
    import json

    def custom_optimize(dsk, keys):
        # Parallelism the task by reducing inter-task communication
        dsk = dask.optimization.inline(dsk, inline_constants=True)
        return dask.array.optimization.optimize(dsk, keys)



    def numpy_to_feature(numpy_array):
        """Takes an numpy array and returns a tf feature

        Args:
            numpy_array (ndarray): numpy array to convert to tf feature

        Returns:
            Feature: Feature object to use in an tf example
        """
        feature=tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tf.convert_to_tensor(numpy_array)).numpy()]))
        return feature



    def serialize(slice_array,var):
        """Constructs an serialzied tf.Example package

        Args:
            slice_array (xarray): A xaray
            var (list): a list of the variables that are to be serialized

        Returns:
            protostring: protostring of tf.train.Example
        """

        feature_dict={}
        for name in var:
            feature=slice_array[name].values
            if type(feature) is np.ndarray:
                feature_dict[name] = numpy_to_feature(feature)
            else:
                raise Exception("other inputs that xarray/ numpy has not yet been defined")
        
        proto=tf.train.Example(features=tf.train.Features(feature=feature_dict))
        return proto.SerializeToString()


    def split_test_train_val(slice_array,test_split,validation_split):
        """Splits the data into train,test,val

        Args:
            slice_array (xarray): The sliced data to be split
            test_split (float, optional): the test split. Defaults to 0.1.
            validation_split (float, optional): the validation split. Defaults to 0.2.

        Returns:
            tuple: returns the selected indices for the train, validation,test split
        """
        num_snapshots=len(slice_array['time'])
        train=np.arange(0,num_snapshots)
        validation=np.random.choice(train,size=int(num_snapshots*validation_split),replace=False) # replace secures that the same value can't be used twice
        train=np.setdiff1d(train,validation)
        test=np.random.choice(train,size=int(num_snapshots*test_split),replace=False)
        train=np.setdiff1d(train,test)
        np.random.shuffle(train)

        return train, validation, test



    def save_load_dict(var,save_loc):
        """Saves an json file with the file format. Makes it possible to read the data back again

        Args:
            var (list): list of variables to include
        """
        load_dict={}

        for name in var:
            load_dict[name] = "array_serial"

        with open(os.path.join(save_loc,'format.json'), 'w') as outfile:
            json.dump(load_dict,outfile)


    client, cluster =utility.slurm_q64(1,time='0-01:30:00',ram='50GB')

    


    #select y_plus value and remove unessary components. Normalize if needed

    slice_array=data
    Re = 10400 #Direct from simulation
    nu = 1/Re #Kinematic viscosity


    if target[0]=='tau_wall':
        target_slice1=slice_array['u_vel'].differentiate('y').sel(y=utility.y_plus_to_y(0),method="nearest")
        target_slice1=nu*target_slice1
        # To get values positive:
        #posi=np.abs(np.floor((target_slice1.min().values) * 1000)/1000.0)
        #target_slice1=target_slice1+posi
        
        
        #target_slice2=slice_array['u_vel'].differentiate('y').sel(y=slice_array['y'].max(),method="nearest")
        #target_slice2=nu*target_slice2
        
        #if normalized==True:
            #target_slice1=(target_slice1-target_slice1.mean(dim=('time','x','z')))/(target_slice1.std(dim=('time','x','z')))
    
    #Checking if the target contains _flux
    elif target[0][-5:] =='_flux':
        if 'All' in target[0]:
            target_slice_pr0025=nu/(0.025)*(slice_array['pr0.025'].differentiate('y').sel(y=utility.y_plus_to_y(0),method="nearest"))
            target_slice_pr02=nu/(0.2)*(slice_array['pr0.2'].differentiate('y').sel(y=utility.y_plus_to_y(0),method="nearest"))
            target_slice_pr071=nu/(0.71)*(slice_array['pr0.71'].differentiate('y').sel(y=utility.y_plus_to_y(0),method="nearest"))
            target_slice_pr1=nu/(1)*(slice_array['pr1'].differentiate('y').sel(y=utility.y_plus_to_y(0),method="nearest"))

            num_snapshots=len(slice_array['time'])
            q, mod = divmod(num_snapshots,4)

            target_slice1=xr.concat((target_slice_pr0025[0:q],target_slice_pr02[q:2*q],target_slice_pr071[2*q:3*q],target_slice_pr1[3*q:4*q+mod]),dim="time")

        else:
            target_slice1=slice_array[target[0][:-5]].differentiate('y').sel(y=utility.y_plus_to_y(0),method="nearest")
            pr_number=float(target[0][2:-5])
            target_slice1=nu/(pr_number)*target_slice1
        
            #target_slice2=slice_array[target[0][:-5]].differentiate('y').sel(y=slice_array['y'].max(),method="nearest")
            #target_slice2=nu/(pr_number)*target_slice2
            
            if normalized==True:
                target_slice1=(target_slice1-target_slice1.mean(dim=('time','x','z')))/(target_slice1.std(dim=('time','x','z')))
    

    elif 'mix' in target[0]:
        split = target[0].split('_')
        del split[1:3]
        num_snapshots=len(slice_array['time'])
        # divmod(7,2) == (3,1)
        q, mod = divmod(num_snapshots,len(split))

        target_slice_list=[]
        var_slice_list=[]
        pr=[]

        for i in range(len(split)):
            mod_i= 0 if i<(len(split)-1) else mod
            pr=np.append(pr,split[i].replace('pr',''))
            
            target_slice_pr=nu/(float(pr[i]))*(slice_array['pr'+pr[i]][(i*q):((i+1)*q+mod_i)].differentiate('y').sel(y=utility.y_plus_to_y(0),method="nearest"))
            target_slice_list.append(target_slice_pr)

            var_slice_pr = slice_array['pr'+pr[i]][(i*q):((i+1)*q+mod_i)]
            var_slice_list.append(var_slice_pr)

        target_slice1=xr.concat(target_slice_list,dim="time")
        slice_array[var[3]]=xr.concat(var_slice_list,dim="time")


    #Checking for InterMeDiate target
    elif 'IMD' in target[0]:
        y_plus_IMD = float(target[0][-2:])
        target_slice_list= [None] * len(target)
        for i in range(len(target)):
            target_slice_list[i]=slice_array[target[i][:-6]].sel(y=utility.y_plus_to_y(y_plus_IMD),method="nearest")


    #other_wall_y_plus=utility.y_to_y_plus(slice_array['y'].max())-y_plus
    
    if normalized==True:
        slice_array=(slice_array-slice_array.mean(dim=('time','x','z')))/(slice_array.std(dim=('time','x','z')))

    

    wall_1=slice_array.sel(y=utility.y_plus_to_y(y_plus),method="nearest")

    # New input if two y_plus_values
    if var_second !=None and y_plus_second!=None:
        wall_2 = slice_array.sel(y=utility.y_plus_to_y(y_plus_second),method="nearest")
        wall_1[var_second]=wall_2[var]
        for feature in var_second:
            var.append(feature)
    


    if len(target)>1:
        for i in range(len(target)):
            wall_1[target[i]]=target_slice_list[i]
    else:
        wall_1[target[0]]=target_slice1

    save_loc=slice_loc(y_plus,var,target,normalized,test)
    
    #append the target
    if len(target)>1:
        for i in range(len(target)):
            var.append(target[i])
    else:
        var.append(target[0])

    wall_1=wall_1[var]  # Remember target is appended to var further up



    #wall_2=slice_array.sel(y=utility.y_plus_to_y(other_wall_y_plus),method="nearest")
    #wall_2[target[0]]=target_slice2
    #wall_2=wall_2[var]
    
 
    #wall_1,wall_2=dask.compute(*[wall_1,wall_2])

    #shuffle the data, split into 3 parts and save
    train_1, validation_1, test_1 = split_test_train_val(wall_1,test_split,validation_split)

    wall_1=wall_1.compute()
    

    #train_2, validation_2, test_2 = split_test_train_val(wall_2)

    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
    else:
        print('deleting old version')
        shutil.rmtree(save_loc)           
        os.makedirs(save_loc)


    options = tf.io.TFRecordOptions(compression_type="GZIP")
    with tf.io.TFRecordWriter(os.path.join(save_loc,'train'),options) as writer:
        print('train',flush=True)
        for i in train_1:
            write_d=serialize(wall_1.isel(time=i),var)
            writer.write(write_d)
        # for i in train_2:
        #         write_d=serialize(wall_2.isel(time=i),var)
        #         writer.write(write_d)
        writer.close()


    with tf.io.TFRecordWriter(os.path.join(save_loc,'test'),options) as writer:
        print('test',flush=True)
        for i in test_1:
            write_d=serialize(wall_1.isel(time=i),var)
            writer.write(write_d)
        # for i in test_2:
        #         write_d=serialize(wall_2.isel(time=i),var)
        #         writer.write(write_d)
        writer.close()

    with tf.io.TFRecordWriter(os.path.join(save_loc,'validation'),options) as writer:
        print('validation',flush=True)
        for i in validation_1:
            write_d=serialize(wall_1.isel(time=i),var)
            writer.write(write_d)
        # for i in validation_2:
        #         write_d=serialize(wall_2.isel(time=i),var)
        #         writer.write(write_d)    
        writer.close()


    save_load_dict(var,save_loc)
    client.close()
    del var   # Remove target again, as it can change the original var used next
    del wall_1
    return None


def slice_loc(y_plus,var,target,normalized,test=False):
    """where to save the slices

    Args:
        y_plus (int): y_plus value of slice
        var (list): list of variables
        target (list): list of targets
        normalized (bool): if the data is normalized or not
    
    Returns:
        str: string of file save location
    """
    import os

    var_sort=sorted(var)
    var_string="_".join(var_sort)
    target_sort=sorted(target)
    target_string="|".join(target_sort)
    if test==False:
        if normalized==True:
            slice_loc=os.path.join("/home/yuning/thesis/data",'y_plus_'+str(y_plus)+"-VARS-"+var_string+"-TARGETS-"+target_string+"-normalized")
        else:
            slice_loc=os.path.join("/home/yuning/thesis/data",'y_plus_'+str(y_plus)+"-VARS-"+var_string+"-TARGETS-"+target_string)
    else:
        if normalized==True:
            slice_loc=os.path.join("/home/yuning/thesis/data_test",'y_plus_'+str(y_plus)+"-VARS-"+var_string+"-TARGETS-"+target_string+"-normalized")
        else:
            slice_loc=os.path.join("/home/yuning/thesis/data_test",'y_plus_'+str(y_plus)+"-VARS-"+var_string+"-TARGETS-"+target_string)
   

    return slice_loc