from utils.datas import TF2Torch
var=['u_vel',"v_vel","w_vel"]
target=['tau_wall']
normalized=False
y_plus=75
save_types= ["train","test","validation"]
for save_type in save_types:
    print(f"Saving {save_type} data!")
    root_path = "/storage3/yuning/thesis/tensor/"
    TF2Torch(root_path,y_plus,var,target,save_type,normalized=normalized)