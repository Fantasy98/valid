def load_dataset(file_path,data_path,target):
    """Load dataset from scratch"""
# Input: file_path: where you save the TFRecord dataset
#         target: str the traget to predict 
# Output: A net TFRecord dataset
    from DataHandling.features.slices import feature_description,read_tfrecords
    import tensorflow as tf 
    dataset = tf.data.TFRecordDataset(
                                    filenames=data_path,
                                    compression_type="GZIP",
                                    num_parallel_reads=tf.data.experimental.AUTOTUNE
                                    )
    
    feature_dict = feature_description(file_path)

    dataset = dataset.map(lambda x: read_tfrecords(x,feature_dict,target),
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return dataset

def decompse_feature(file_path,data_path,target):
    """Decompose features from dataset"""

    # Input : file_path
    #            target
    #             feature
    # output: A list of feature at all snapshots
    from DataHandling.features.slices import feature_description,read_tfrecords
    import tensorflow as tf 
    import numpy as np
    dataset = load_dataset(file_path,data_path,target)
    feature_dict = feature_description(file_path)
    for ele in dataset.as_numpy_iterator():


        feature_names = ele[0].keys()
        break
    data_dict = {}
    # Create a list for each features    
    for name in feature_names: 
            data_dict[name] = []


    # iterating in dataset to retreive all the features snapshots
    for element in dataset.as_numpy_iterator():
        for name in feature_names: 
            # tuple(dict) thus ele[0] we get dict and accroding to its key we git value
            data_dict[name].append(element[0][name])



    # Convert them into numpy array
    for name in feature_names: 
            data_dict[name] = np.array(data_dict[name])

            print("The shape of {} snapshots = {}\n".format(name,data_dict[name].shape))
    return data_dict

def slice_single_features(X_data,Batch_Size,Shuffle,Repeat,Prefetch):
    import tensorflow as tf
    U_TF = tf.data.Dataset.from_tensor_slices(X_data,)
    U_TF = U_TF.batch(batch_size=Batch_Size)
    U_TF = U_TF.shuffle(buffer_size=Shuffle)
    U_TF = U_TF.repeat(Repeat)
    U_TF = U_TF.prefetch(Prefetch)
    return U_TF


def Train_Test_Split(Feature_Data,Ratio,Batch_Size,Shuffle,Repeat,Prefetch):
    import numpy as np 
    N_snap = Feature_Data.shape[0]
    N_Train = int(np.ceil(N_snap*Ratio))
    Train_data = Feature_Data[:N_Train,:,:]
    Test_data = Feature_Data[N_Train:,:,:]
    TF_Train = slice_single_features(Train_data,Batch_Size,Shuffle,Repeat,Prefetch)
    TF_Test = slice_single_features(Test_data,Batch_Size,Shuffle,Repeat,Prefetch)
    return TF_Train,TF_Test


def Save_TFdata(data_dict,Ratio,Batch_Size,Shuffle,Repeat,Prefetch,save_dir):
    import tensorflow as tf
    import os
    feature_names = data_dict.keys()
    for name in feature_names:
        TF_Train,TF_Test = Train_Test_Split(data_dict[name],
                                            Ratio,Batch_Size,Shuffle,Repeat,Prefetch
                                            )
        path_save = os.path.join(save_dir,name)
        if os.path.exists(path_save):
            TF_Train.save(os.path.join(path_save,"train"))
            TF_Test.save(os.path.join(path_save,"test"))
        else:
            os.mkdir(path_save)
            TF_Train.save(os.path.join(path_save,"train"))
            TF_Test.save(os.path.join(path_save,"test"))
        print("{} Has been saved!".format(name))