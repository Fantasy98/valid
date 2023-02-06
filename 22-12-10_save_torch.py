from utils.datas import TF2Torch,mkdataset
var=['u_vel',"v_vel","w_vel","pr1"]
target=['pr1_flux']
normalized=False
y_plus=30
# save_types= ["test","validation"]
save_types= ["test"]
# save_types= ["train"]

for save_type in save_types:
    print(f"Dealing with {save_type}")
    root_path = "/home/yuning/thesis/tensor/"
    TF2Torch(root_path,y_plus,var,target,save_type,normalized=False)
    mkdataset(root_path,y_plus,var,target,save_type,normalized=False)