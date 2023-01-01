from utils.datas import TF2Torch
var=['u_vel',"v_vel","w_vel"]
target=['tau_wall']
normalized=False
y_plus=75
save_types= ["train","test","validation"]

for save_type in save_types:
<<<<<<< HEAD
    print(f"Saving {save_type} data!")
    root_path = "/storage3/yuning/thesis/tensor/"
    TF2Torch(root_path,y_plus,var,target,save_type,normalized=normalized)
=======
    print(f"Dealing with {save_type}")
    root_path = "/home/yuning/thesis/tensor/"
    TF2Torch(root_path,y_plus,var,target,save_type,normalized=False)
>>>>>>> 758c5c460a9b0180130b9196fe6049386240052c
